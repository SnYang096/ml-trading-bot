from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.time_series_model.rule.router_3action import Rule3ActionConfig

from src.time_series_model.portfolio.mean_system import compute_mean_system_health

from src.time_series_model.diagnostics.ood_config import (
    compute_size_cap_multiplier,
    load_ood_config,
)

from .bc_dataset import (
    BCStateSchema,
    Router3Action,
    Router3ActionInferConfig,
    infer_router3_action,
)
from .bc_trainer_3action import BC3TrainConfig, train_bc_router3_policy
from .sim_env_3action import SimEnvConfig, simulate_3action_episode
from .walk_forward import WalkForwardSplitConfig, time_ordered_split_by_symbol


@dataclass(frozen=True)
class CounterfactualEvalConfig:
    """
    Counterfactual A/B evaluation using precomputed per-step mode returns:
      - ret_mean: next-step return if running MEAN execution
      - ret_trend: next-step return if running TREND execution

    This produces:
      Rule mode equity vs BC-pred mode equity (same returns, same costs/constraints).

    Note: Mode is inferred from archetype (TC/TE → TREND, FR/ET → MEAN, others → NO_TRADE).
    """

    archetype_col: str = "gate_archetype"  # Primary: use gate_archetype if available
    regime_col: str = "regime"  # Fallback: use regime if archetype not available
    mode_col: str = (
        "mode"  # Legacy fallback: use mode if archetype/regime not available
    )
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"

    # BC observation
    state_keys: Sequence[str] = (
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "drawdown",
    )

    split_cfg: WalkForwardSplitConfig = WalkForwardSplitConfig()
    bc_cfg: BC3TrainConfig = BC3TrainConfig(
        epochs=5, batch_size=256, hidden=128, depth=2, dropout=0.1
    )

    # Simulation config (costs, slippage, risk)
    sim_cfg: SimEnvConfig = SimEnvConfig(entry_delay=1)

    # Reporting / scoring config (recorder-friendly)
    # score = Sharpe_mean - lambda*Sharpe_std - mu*DD_mean
    score_lambda: float = 1.0
    score_mu: float = 0.5

    # Router-aligned diagnostics (unified report language across tuning/e2e)
    router_cfg: Optional[Rule3ActionConfig] = None
    preds_in_log1p: bool = True
    rolling_enabled: bool = True
    rolling_window: int = 300
    rolling_min_periods: int = 60
    rolling_tail_points: int = 120

    # Optional: Survival Head integration (research + live share semantics)
    survival_prob_col: str = "survival_prob"
    ood_score_col: str = "ood_score"
    ood_config_yaml: Optional[str] = None


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Simple AUC without sklearn:
    Uses rank-based U-statistic. Returns 0.5 when undefined.
    """
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.5
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average precision (area under PR curve) without sklearn.
    Returns 0 when undefined.
    """
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.0
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    if n_pos == int(y_true.size):
        return 1.0
    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(precision[y_sorted == 1]) / float(n_pos))
    return float(max(0.0, min(1.0, ap)))


def _maybe_expm1(x: np.ndarray, *, preds_in_log1p: bool) -> np.ndarray:
    if not preds_in_log1p:
        return x
    return np.expm1(np.clip(x, 0.0, 15.0))


def _router_derived_from_heads(
    df: pd.DataFrame, *, cfg: Rule3ActionConfig, preds_in_log1p: bool
) -> pd.DataFrame:
    """
    Derive Router-aligned columns from logs' head_* columns.
    """
    out = pd.DataFrame(index=df.index)
    mfe_p = pd.to_numeric(df.get("head_mfe_atr"), errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df.get("head_mae_atr"), errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df.get("head_t_to_mfe"), errors="coerce").to_numpy(
        dtype=float
    )
    dir_s = pd.to_numeric(df.get("head_dir_score"), errors="coerce").to_numpy(
        dtype=float
    )

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)

    # head_dir_score in [-1,1] (2*p-1) -> p in [0,1] -> confidence in [0,1]
    p = np.clip((dir_s + 1.0) / 2.0, 0.0, 1.0)
    dir_conf = np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)
    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )
    out["router_mfe_atr"] = mfe
    out["router_mae_atr"] = mae
    out["router_t_to_mfe"] = ttm
    out["router_eff"] = eff
    out["router_dir_conf"] = dir_conf
    return out


def _rank_metrics(y_bin: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_bin) & np.isfinite(score)
    yb = y_bin[mask].astype(int)
    sc = score[mask]
    if yb.size == 0:
        return {"auc": 0.5, "ap": 0.0, "pos_rate": 0.0}
    return {
        "auc": _binary_auc(yb, sc),
        "ap": _average_precision(yb, sc),
        "pos_rate": float(np.mean(yb)),
    }


def _rolling_trade_preview(
    df_trade: pd.DataFrame,
    *,
    symbol_col: str,
    ts_col: str,
    window: int,
    minp: int,
    tail_n: int,
) -> pd.DataFrame:
    if df_trade is None or len(df_trade) == 0:
        return pd.DataFrame()
    work = df_trade.copy()
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = (
        work.dropna(subset=[ts_col])
        .sort_values([symbol_col, ts_col])
        .reset_index(drop=True)
    )

    rows = []
    for sym, g in work.groupby(symbol_col, sort=False):
        g = g.sort_values(ts_col).reset_index(drop=True)
        rg = pd.DataFrame(
            {
                "timestamp": g[ts_col],
                "router_mfe_atr": pd.to_numeric(g["router_mfe_atr"], errors="coerce"),
                "router_eff": pd.to_numeric(g["router_eff"], errors="coerce"),
                "router_dir_conf": pd.to_numeric(g["router_dir_conf"], errors="coerce"),
                "ret_used": pd.to_numeric(g["ret_used"], errors="coerce"),
            }
        )
        roll = rg.rolling(window=window, min_periods=minp)
        out = pd.DataFrame(
            {
                "timestamp": rg["timestamp"],
                "symbol": str(sym),
                "roll_mfe_mean": roll["router_mfe_atr"].mean(),
                "roll_eff_mean": roll["router_eff"].mean(),
                "roll_dir_conf_mean": roll["router_dir_conf"].mean(),
                "roll_ret_mean": roll["ret_used"].mean(),
                "roll_winrate": roll["ret_used"].apply(
                    lambda x: (
                        float(np.mean(np.asarray(x) > 0.0)) if len(x) else float("nan")
                    ),
                    raw=False,
                ),
            }
        )
        rows.append(out)
    df_roll = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if df_roll.empty:
        return df_roll
    agg = (
        df_roll.groupby("timestamp", sort=True)
        .mean(numeric_only=True)
        .reset_index()
        .tail(tail_n)
    )
    return agg


def _infer_mode_from_archetype(archetype: Any, regime: Any = None) -> str:
    """
    Infer mode (TREND/MEAN/NO_TRADE) from archetype or regime.

    Rules:
    - TC/TE → TREND
    - FR/ET → MEAN
    - Others or missing → NO_TRADE
    - If archetype not available, use regime (TC_REGIME/TE_REGIME → TREND, MEAN_REGIME → MEAN)
    """
    if archetype is not None:
        arch_str = str(archetype).upper()
        if "TC" in arch_str or "TE" in arch_str:
            return "TREND"
        elif "FR" in arch_str or "ET" in arch_str:
            return "MEAN"

    # Fallback to regime
    if regime is not None:
        reg_str = str(regime).upper()
        if "TC_REGIME" in reg_str or "TE_REGIME" in reg_str or reg_str == "TREND":
            return "TREND"
        elif "MEAN_REGIME" in reg_str or reg_str == "MEAN":
            return "MEAN"

    return "NO_TRADE"


def _mode_to_action(mode: Any) -> int:
    m = "" if mode is None else str(mode).upper()
    if m in {"NO_TRADE", "NOTRADE", "OFF", "OBSERVE", "PAUSE"}:
        return int(Router3Action.NO_TRADE)
    if m in {"MEAN", "MEAN_REVERT", "MEANREVERT"}:
        return int(Router3Action.MEAN)
    if m in {"TREND", "TREND_FOLLOW", "TRENDFOLLOW"}:
        return int(Router3Action.TREND)
    # fallback: treat unknown as NO_TRADE for safety
    return int(Router3Action.NO_TRADE)


def _apply_mean_only_actions(actions: np.ndarray) -> np.ndarray:
    """
    Mean-only mode for 3-action router:
    - keep MEAN as-is
    - convert TREND -> NO_TRADE (disable trend engine entirely)
    - keep NO_TRADE as-is
    """
    if actions is None:
        return np.zeros((0,), dtype=np.int64)
    a = np.asarray(actions, dtype=np.int64).copy()
    a[a == int(Router3Action.TREND)] = int(Router3Action.NO_TRADE)
    return a


def _predict_modes(
    df: pd.DataFrame,
    *,
    model,
    state_schema: BCStateSchema,
    state_keys: Sequence[str],
    device: Optional[str],
) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros((0,), dtype=np.int64)
    X = np.stack(
        [
            state_schema.encode_state({k: r.get(k, 0.0) for k in state_keys})
            for r in df.to_dict(orient="records")
        ],
        axis=0,
    ).astype(np.float32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(dev))
        y = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    return y


def _max_drawdown(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        peak = max(peak, float(x))
        dd = (peak - float(x)) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def _total_return(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    return float(equity[-1] / equity[0] - 1.0) if equity[0] != 0 else 0.0


def _entropy(actions: np.ndarray) -> float:
    if actions is None or len(actions) == 0:
        return 0.0
    counts = {0: 0, 1: 0, 2: 0}
    for a in actions.tolist():
        counts[int(a)] = counts.get(int(a), 0) + 1
    total = float(sum(counts.values()))
    ent = 0.0
    for c in counts.values():
        p = float(c) / total if total > 0 else 0.0
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)


def _switch_rate(actions: np.ndarray) -> float:
    if actions is None or len(actions) < 2:
        return 0.0
    sw = int(np.sum(actions[1:] != actions[:-1]))
    return float(sw) / float(len(actions) - 1)


def _infer_steps_per_year(ts: pd.Series) -> Optional[float]:
    """
    Infer annualization factor from timestamps (median step size).
    Returns steps_per_year, or None if cannot infer.
    """
    if ts is None or len(ts) < 3:
        return None
    try:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
        dt = dt.dropna()
        if len(dt) < 3:
            return None
        dt = dt.sort_values().drop_duplicates()
        diffs = dt.diff().dropna()
        if len(diffs) == 0:
            return None
        step_sec = float(diffs.median().total_seconds())
        if not np.isfinite(step_sec) or step_sec <= 0:
            return None
        sec_per_year = 365.25 * 24.0 * 3600.0
        steps = sec_per_year / step_sec
        if not np.isfinite(steps) or steps <= 0:
            return None
        return float(max(1.0, min(1e6, steps)))
    except Exception:
        return None


def _ann_return_from_equity(
    equity: np.ndarray, *, steps_per_year: Optional[float]
) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    eq0 = float(equity[0])
    eq1 = float(equity[-1])
    if eq0 <= 0 or eq1 <= 0:
        return 0.0
    if steps_per_year is None:
        return float(eq1 / eq0 - 1.0)
    n = float(len(equity) - 1)
    if n <= 0:
        return 0.0
    # annualize log growth for numerical stability
    g = math.log(eq1 / eq0)
    return float(math.exp(g * float(steps_per_year) / n) - 1.0)


def _ann_vol_from_pnl(pnl: np.ndarray, *, steps_per_year: Optional[float]) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    s = float(np.std(pnl, ddof=1))
    if not np.isfinite(s):
        return 0.0
    if steps_per_year is None:
        return s
    return float(s * math.sqrt(float(steps_per_year)))


def _sharpe_from_pnl(pnl: np.ndarray, *, steps_per_year: Optional[float]) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    if steps_per_year is None:
        return float(mu / sd)
    return float(mu / sd * math.sqrt(float(steps_per_year)))


def _sortino_from_pnl(pnl: np.ndarray, *, steps_per_year: Optional[float]) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    downside = pnl[pnl < 0.0]
    if len(downside) < 2:
        return 0.0
    dd = float(np.std(downside, ddof=1))
    if not np.isfinite(mu) or not np.isfinite(dd) or dd <= 1e-12:
        return 0.0
    if steps_per_year is None:
        return float(mu / dd)
    return float(mu / dd * math.sqrt(float(steps_per_year)))


def train_and_counterfactual_eval_bc3(
    df_logs: pd.DataFrame,
    *,
    cfg: CounterfactualEvalConfig = CounterfactualEvalConfig(),
    out_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Train BC(3-action) on train split and run counterfactual A/B on test split:
      A: rule mode (df_logs[mode_col])
      B: BC predicted mode

    Requires df_logs to contain:
      - mode_col (rule mode label)
      - ret_mean_col / ret_trend_col from cfg.sim_cfg
      - state_keys columns

    Returns: (meta, metrics, per_symbol_df)
    """
    # Check for at least one of archetype/regime/mode columns
    has_archetype = cfg.archetype_col in df_logs.columns
    has_regime = cfg.regime_col in df_logs.columns
    has_mode = cfg.mode_col in df_logs.columns

    if not (has_archetype or has_regime or has_mode):
        raise ValueError(
            f"Missing required columns. Need at least one of: "
            f"'{cfg.archetype_col}' (preferred), '{cfg.regime_col}' (fallback), "
            f"or '{cfg.mode_col}' (legacy fallback) in logs."
        )
    if cfg.sim_cfg.ret_mean_col not in df_logs.columns:
        raise ValueError(f"Missing return column '{cfg.sim_cfg.ret_mean_col}' in logs.")
    if cfg.sim_cfg.ret_trend_col not in df_logs.columns:
        raise ValueError(
            f"Missing return column '{cfg.sim_cfg.ret_trend_col}' in logs."
        )
    for k in cfg.state_keys:
        if k not in df_logs.columns:
            raise ValueError(f"Missing state key column '{k}' in logs.")

    train_df, test_df = time_ordered_split_by_symbol(df_logs, cfg=cfg.split_cfg)
    state_schema = BCStateSchema(keys=list(cfg.state_keys))
    infer_cfg = Router3ActionInferConfig(
        mean_routers=[], trend_routers=[]
    )  # mode is inferred from archetype

    # Infer mode from archetype/regime/mode for training transitions
    transitions_train = []
    for r in train_df.to_dict(orient="records"):
        state = {k: r.get(k, 0.0) for k in cfg.state_keys}

        # Try to infer mode from archetype/regime/mode in order of preference
        archetype = r.get(cfg.archetype_col)
        regime = r.get(cfg.regime_col)
        mode = r.get(cfg.mode_col)

        if archetype is not None or regime is not None:
            inferred_mode = _infer_mode_from_archetype(archetype, regime)
        elif mode is not None:
            inferred_mode = str(mode).upper()
        else:
            inferred_mode = "NO_TRADE"

        transitions_train.append(
            {
                "state": state,
                "action": {"mode": inferred_mode},
            }
        )

    model, train_meta = train_bc_router3_policy(
        transitions=transitions_train,
        state_schema=state_schema,
        infer_cfg=infer_cfg,
        cfg=cfg.bc_cfg,
    )

    # test actions: infer mode from archetype/regime/mode for rule actions
    rule_modes = []
    for _, row in test_df.iterrows():
        archetype = row.get(cfg.archetype_col)
        regime = row.get(cfg.regime_col)
        mode = row.get(cfg.mode_col)

        if archetype is not None or regime is not None:
            inferred_mode = _infer_mode_from_archetype(archetype, regime)
        elif mode is not None:
            inferred_mode = str(mode).upper()
        else:
            inferred_mode = "NO_TRADE"

        rule_modes.append(inferred_mode)

    rule_actions = np.asarray([_mode_to_action(m) for m in rule_modes], dtype=np.int64)
    pred_actions = _predict_modes(
        test_df,
        model=model,
        state_schema=state_schema,
        state_keys=cfg.state_keys,
        device=cfg.bc_cfg.device,
    )

    # Router-aligned diagnostics (Rule side) on test split for unified reporting.
    router_diag: Dict[str, Any] = {}
    router_diag_per_symbol: pd.DataFrame = pd.DataFrame()
    router_roll_tail: pd.DataFrame = pd.DataFrame()
    try:
        router_cfg = cfg.router_cfg or Rule3ActionConfig()

        test_work = test_df.reset_index(drop=True).copy()
        rd = _router_derived_from_heads(
            test_work, cfg=router_cfg, preds_in_log1p=bool(cfg.preds_in_log1p)
        )
        test_work = pd.concat([test_work, rd.reset_index(drop=True)], axis=1)

        # realized return given the action actually executed by rule mode
        rm = (
            pd.to_numeric(test_work[cfg.sim_cfg.ret_mean_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        rt = (
            pd.to_numeric(test_work[cfg.sim_cfg.ret_trend_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        rule_a = np.asarray(
            [_mode_to_action(m) for m in test_work[cfg.mode_col].tolist()],
            dtype=np.int64,
        )
        ret_used = np.where(rule_a == int(Router3Action.MEAN), rm, 0.0)
        ret_used = np.where(rule_a == int(Router3Action.TREND), rt, ret_used)
        test_work["rule_action"] = rule_a.astype(int)
        test_work["ret_used"] = ret_used.astype(float)

        trade = (rule_a == int(Router3Action.MEAN)) | (
            rule_a == int(Router3Action.TREND)
        )

        # binary target: profitable step in executed mode
        y_win = (ret_used > 0.0).astype(int)
        s_mfe = pd.to_numeric(test_work["router_mfe_atr"], errors="coerce").to_numpy(
            dtype=float
        )
        s_eff = pd.to_numeric(test_work["router_eff"], errors="coerce").to_numpy(
            dtype=float
        )
        s_dc = pd.to_numeric(test_work["router_dir_conf"], errors="coerce").to_numpy(
            dtype=float
        )

        rank_metrics: Dict[str, Any] = {}
        if bool(np.any(trade)):
            met_mfe = _rank_metrics(y_win[trade], s_mfe[trade])
            met_eff = _rank_metrics(y_win[trade], s_eff[trade])
            met_dc = _rank_metrics(y_win[trade], s_dc[trade])
            rank_metrics = {
                "mfe": met_mfe,
                "eff": met_eff,
                "dir_conf": met_dc,
            }

        # threshold subsets (precision-ish)
        def _subset(mask: np.ndarray) -> Dict[str, float]:
            if mask.size == 0 or not mask.any():
                return {"n": 0.0, "win_rate": 0.0, "avg_ret": 0.0}
            return {
                "n": float(np.sum(mask)),
                "win_rate": float(np.mean(y_win[mask].astype(float))),
                "avg_ret": float(np.mean(ret_used[mask].astype(float))),
            }

        router_diag = {
            "thresholds": {
                "mfe_min": float(router_cfg.mfe_min),
                "eff_min": float(router_cfg.eff_min),
                "dir_conf_trend_min": float(router_cfg.dir_conf_trend_min),
            },
            "rank_metrics_trade_slice": rank_metrics,
            "trade": _subset(trade),
            "mfe_ge_mfe_min": _subset(trade & (s_mfe >= float(router_cfg.mfe_min))),
            "eff_ge_eff_min": _subset(trade & (s_eff >= float(router_cfg.eff_min))),
            "dir_conf_ge_dir_conf_trend_min": _subset(
                trade & (s_dc >= float(router_cfg.dir_conf_trend_min))
            ),
        }

        # per-symbol diag on trade slice
        per_rows = []
        if cfg.symbol_col in test_work.columns:
            for sym, g in test_work.groupby(cfg.symbol_col, sort=False):
                ga = (
                    pd.to_numeric(g["rule_action"], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=int)
                )
                gt = (ga == int(Router3Action.MEAN)) | (ga == int(Router3Action.TREND))
                if int(np.sum(gt)) < 10:
                    continue
                gr = (
                    pd.to_numeric(g["ret_used"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                gw = (gr > 0.0).astype(int)
                sm = pd.to_numeric(g["router_mfe_atr"], errors="coerce").to_numpy(
                    dtype=float
                )
                se = pd.to_numeric(g["router_eff"], errors="coerce").to_numpy(
                    dtype=float
                )
                sd = pd.to_numeric(g["router_dir_conf"], errors="coerce").to_numpy(
                    dtype=float
                )
                per_rows.append(
                    {
                        "symbol": str(sym),
                        "trade_rate": float(np.mean(gt.astype(float))),
                        "win_rate": float(np.mean(gw[gt].astype(float))),
                        "avg_ret": float(np.mean(gr[gt])),
                        "mfe_auc": float(_rank_metrics(gw[gt], sm[gt])["auc"]),
                        "eff_auc": float(_rank_metrics(gw[gt], se[gt])["auc"]),
                        "dir_conf_auc": float(_rank_metrics(gw[gt], sd[gt])["auc"]),
                    }
                )
        router_diag_per_symbol = (
            pd.DataFrame(per_rows).sort_values("symbol").reset_index(drop=True)
            if per_rows
            else pd.DataFrame()
        )

        # rolling tail on trade slice
        if (
            bool(cfg.rolling_enabled)
            and cfg.timestamp_col in test_work.columns
            and bool(np.any(trade))
        ):
            router_roll_tail = _rolling_trade_preview(
                test_work.loc[trade].copy(),
                symbol_col=cfg.symbol_col,
                ts_col=cfg.timestamp_col,
                window=int(cfg.rolling_window),
                minp=int(cfg.rolling_min_periods),
                tail_n=int(cfg.rolling_tail_points),
            )

        # (we merge scalars into metrics later, after metrics dict is constructed)
    except Exception:
        router_diag = {}
        router_diag_per_symbol = pd.DataFrame()
        router_roll_tail = pd.DataFrame()

    # per-symbol simulation

    rows = []
    for sym, g in test_df.groupby(cfg.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        ra = rule_actions[test_df[cfg.symbol_col].values == sym]
        pa = pred_actions[test_df[cfg.symbol_col].values == sym]
        ma = _apply_mean_only_actions(ra)

        mean_mult_arr = None
        trend_mult_arr = None
        surv_mean_mult_arr = None
        surv_trend_mult_arr = None

        # Optional: Survival baseline multipliers (size_cap = survival^a * (1-ood)^b).
        try:
            if (
                cfg.ood_config_yaml
                and cfg.survival_prob_col in g.columns
                and cfg.ood_score_col in g.columns
            ):
                ood_cfg = load_ood_config(str(cfg.ood_config_yaml))
                ood_arr = (
                    pd.to_numeric(g[cfg.ood_score_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                surv_arr = (
                    pd.to_numeric(g[cfg.survival_prob_col], errors="coerce")
                    .fillna(1.0)
                    .to_numpy(dtype=float)
                )
                cap = [
                    compute_size_cap_multiplier(
                        cfg=ood_cfg, ood_score=float(o), survival_prob=float(s)
                    )
                    for o, s in zip(ood_arr.tolist(), surv_arr.tolist())
                ]
                surv_mean_mult_arr = cap
                surv_trend_mult_arr = cap
        except Exception:
            surv_mean_mult_arr = None
            surv_trend_mult_arr = None

        # Baselines:
        # - rule: the original rule router execution (full exposure per mode)
        # - rule_pcm: rule router execution with PCM multipliers (partial cash / budgets)
        out_rule = simulate_3action_episode(g, actions=ra.tolist(), cfg=cfg.sim_cfg)
        out_rule_pcm = (
            simulate_3action_episode(
                g,
                actions=ra.tolist(),
                mean_multiplier=mean_mult_arr,
                trend_multiplier=trend_mult_arr,
                cfg=cfg.sim_cfg,
            )
            if (mean_mult_arr is not None or trend_mult_arr is not None)
            else out_rule
        )
        out_rule_survival = (
            simulate_3action_episode(
                g,
                actions=ra.tolist(),
                mean_multiplier=surv_mean_mult_arr,
                trend_multiplier=surv_trend_mult_arr,
                cfg=cfg.sim_cfg,
            )
            if (surv_mean_mult_arr is not None or surv_trend_mult_arr is not None)
            else out_rule
        )
        out_pred = simulate_3action_episode(g, actions=pa.tolist(), cfg=cfg.sim_cfg)
        out_mean_only = simulate_3action_episode(
            g, actions=ma.tolist(), cfg=cfg.sim_cfg
        )

        eq_r = out_rule["equity"].to_numpy(dtype=float)
        eq_r_pcm = out_rule_pcm["equity"].to_numpy(dtype=float)
        eq_r_surv = out_rule_survival["equity"].to_numpy(dtype=float)
        eq_p = out_pred["equity"].to_numpy(dtype=float)
        eq_m = out_mean_only["equity"].to_numpy(dtype=float)
        pnl_r = (
            out_rule["pnl"].to_numpy(dtype=float) if "pnl" in out_rule.columns else None
        )
        pnl_r_pcm = (
            out_rule_pcm["pnl"].to_numpy(dtype=float)
            if "pnl" in out_rule_pcm.columns
            else None
        )
        pnl_r_surv = (
            out_rule_survival["pnl"].to_numpy(dtype=float)
            if "pnl" in out_rule_survival.columns
            else None
        )
        pnl_p = (
            out_pred["pnl"].to_numpy(dtype=float) if "pnl" in out_pred.columns else None
        )
        pnl_m = (
            out_mean_only["pnl"].to_numpy(dtype=float)
            if "pnl" in out_mean_only.columns
            else None
        )

        steps_per_year = (
            _infer_steps_per_year(g[cfg.timestamp_col])
            if cfg.timestamp_col in g.columns
            else None
        )
        rows.append(
            {
                "symbol": str(sym),
                "n": int(len(g)),
                "steps_per_year": float(steps_per_year) if steps_per_year else 0.0,
                "rule_total_return": _total_return(eq_r),
                "rule_pcm_total_return": _total_return(eq_r_pcm),
                "rule_survival_total_return": _total_return(eq_r_surv),
                "pred_total_return": _total_return(eq_p),
                "mean_only_total_return": _total_return(eq_m),
                "rule_max_dd": _max_drawdown(eq_r),
                "rule_pcm_max_dd": _max_drawdown(eq_r_pcm),
                "rule_survival_max_dd": _max_drawdown(eq_r_surv),
                "pred_max_dd": _max_drawdown(eq_p),
                "mean_only_max_dd": _max_drawdown(eq_m),
                "rule_ann_return": _ann_return_from_equity(
                    eq_r, steps_per_year=steps_per_year
                ),
                "rule_pcm_ann_return": _ann_return_from_equity(
                    eq_r_pcm, steps_per_year=steps_per_year
                ),
                "rule_survival_ann_return": _ann_return_from_equity(
                    eq_r_surv, steps_per_year=steps_per_year
                ),
                "pred_ann_return": _ann_return_from_equity(
                    eq_p, steps_per_year=steps_per_year
                ),
                "mean_only_ann_return": _ann_return_from_equity(
                    eq_m, steps_per_year=steps_per_year
                ),
                "rule_ann_vol": (
                    _ann_vol_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "rule_pcm_ann_vol": (
                    _ann_vol_from_pnl(pnl_r_pcm, steps_per_year=steps_per_year)
                    if pnl_r_pcm is not None
                    else 0.0
                ),
                "rule_survival_ann_vol": (
                    _ann_vol_from_pnl(pnl_r_surv, steps_per_year=steps_per_year)
                    if pnl_r_surv is not None
                    else 0.0
                ),
                "pred_ann_vol": (
                    _ann_vol_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "mean_only_ann_vol": (
                    _ann_vol_from_pnl(pnl_m, steps_per_year=steps_per_year)
                    if pnl_m is not None
                    else 0.0
                ),
                "rule_sharpe": (
                    _sharpe_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "rule_pcm_sharpe": (
                    _sharpe_from_pnl(pnl_r_pcm, steps_per_year=steps_per_year)
                    if pnl_r_pcm is not None
                    else 0.0
                ),
                "rule_survival_sharpe": (
                    _sharpe_from_pnl(pnl_r_surv, steps_per_year=steps_per_year)
                    if pnl_r_surv is not None
                    else 0.0
                ),
                "pred_sharpe": (
                    _sharpe_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "mean_only_sharpe": (
                    _sharpe_from_pnl(pnl_m, steps_per_year=steps_per_year)
                    if pnl_m is not None
                    else 0.0
                ),
                "rule_sortino": (
                    _sortino_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "rule_pcm_sortino": (
                    _sortino_from_pnl(pnl_r_pcm, steps_per_year=steps_per_year)
                    if pnl_r_pcm is not None
                    else 0.0
                ),
                "rule_survival_sortino": (
                    _sortino_from_pnl(pnl_r_surv, steps_per_year=steps_per_year)
                    if pnl_r_surv is not None
                    else 0.0
                ),
                "pred_sortino": (
                    _sortino_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "mean_only_sortino": (
                    _sortino_from_pnl(pnl_m, steps_per_year=steps_per_year)
                    if pnl_m is not None
                    else 0.0
                ),
                "rule_turnover_mean": (
                    float(out_rule["turnover"].mean()) if len(out_rule) else 0.0
                ),
                "rule_pcm_turnover_mean": (
                    float(out_rule_pcm["turnover"].mean()) if len(out_rule_pcm) else 0.0
                ),
                "rule_survival_turnover_mean": (
                    float(out_rule_survival["turnover"].mean())
                    if len(out_rule_survival)
                    else 0.0
                ),
                "pred_turnover_mean": (
                    float(out_pred["turnover"].mean()) if len(out_pred) else 0.0
                ),
                "mean_only_turnover_mean": (
                    float(out_mean_only["turnover"].mean())
                    if len(out_mean_only)
                    else 0.0
                ),
                "rule_switch_rate": _switch_rate(ra),
                "pred_switch_rate": _switch_rate(pa),
                "mean_only_switch_rate": _switch_rate(ma),
                "rule_mode_entropy": _entropy(ra),
                "pred_mode_entropy": _entropy(pa),
                "mean_only_mode_entropy": _entropy(ma),
                "rule_final_equity": (
                    float(eq_r[-1]) if len(eq_r) else float(cfg.sim_cfg.initial_equity)
                ),
                "rule_pcm_final_equity": (
                    float(eq_r_pcm[-1])
                    if len(eq_r_pcm)
                    else float(cfg.sim_cfg.initial_equity)
                ),
                "rule_survival_final_equity": (
                    float(eq_r_surv[-1])
                    if len(eq_r_surv)
                    else float(cfg.sim_cfg.initial_equity)
                ),
                "pred_final_equity": (
                    float(eq_p[-1]) if len(eq_p) else float(cfg.sim_cfg.initial_equity)
                ),
                "mean_only_final_equity": (
                    float(eq_m[-1]) if len(eq_m) else float(cfg.sim_cfg.initial_equity)
                ),
            }
        )

    per_symbol = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    # Aggregate (simple average across symbols)
    metrics: Dict[str, Any] = {
        "test_symbols": float(per_symbol.shape[0]),
        "test_steps": float(per_symbol["n"].sum()) if len(per_symbol) else 0.0,
        "rule_avg_total_return": (
            float(per_symbol["rule_total_return"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_pcm_avg_total_return": (
            float(per_symbol["rule_pcm_total_return"].mean())
            if len(per_symbol)
            else 0.0
        ),
        "rule_survival_avg_total_return": (
            float(per_symbol["rule_survival_total_return"].mean())
            if len(per_symbol)
            else 0.0
        ),
        "pred_avg_total_return": (
            float(per_symbol["pred_total_return"].mean()) if len(per_symbol) else 0.0
        ),
        "mean_only_avg_total_return": (
            float(per_symbol["mean_only_total_return"].mean())
            if len(per_symbol)
            else 0.0
        ),
        "rule_avg_max_dd": (
            float(per_symbol["rule_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_pcm_avg_max_dd": (
            float(per_symbol["rule_pcm_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_survival_avg_max_dd": (
            float(per_symbol["rule_survival_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_max_dd": (
            float(per_symbol["pred_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "mean_only_avg_max_dd": (
            float(per_symbol["mean_only_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_avg_switch_rate": (
            float(per_symbol["rule_switch_rate"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_switch_rate": (
            float(per_symbol["pred_switch_rate"].mean()) if len(per_symbol) else 0.0
        ),
        "mean_only_avg_switch_rate": (
            float(per_symbol["mean_only_switch_rate"].mean())
            if len(per_symbol)
            else 0.0
        ),
        "rule_avg_mode_entropy": (
            float(per_symbol["rule_mode_entropy"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_mode_entropy": (
            float(per_symbol["pred_mode_entropy"].mean()) if len(per_symbol) else 0.0
        ),
        "mean_only_avg_mode_entropy": (
            float(per_symbol["mean_only_mode_entropy"].mean())
            if len(per_symbol)
            else 0.0
        ),
    }

    # Risk metrics (mean/std across symbols)
    if len(per_symbol):
        for side in ["rule", "rule_pcm", "rule_survival", "pred", "mean_only"]:
            for k in ["sharpe", "sortino", "ann_return", "ann_vol"]:
                col = f"{side}_{k}"
                if col in per_symbol.columns:
                    metrics[f"{side}_{k}_mean"] = float(per_symbol[col].mean())
                    metrics[f"{side}_{k}_std"] = float(
                        per_symbol[col].std(ddof=1) if len(per_symbol) > 1 else 0.0
                    )

    # Recorder-friendly score field (fixed formula; compare across runs)
    # score = Sharpe_mean - lambda*Sharpe_std - mu*DD_mean
    lam = float(cfg.score_lambda)
    mu = float(cfg.score_mu)
    metrics["score_formula"] = "Sharpe_mean - lambda*Sharpe_std - mu*DD_mean"
    rule_sh_m = float(metrics.get("rule_sharpe_mean", 0.0))
    rule_sh_s = float(metrics.get("rule_sharpe_std", 0.0))
    rule_dd_m = float(metrics.get("rule_avg_max_dd", 0.0))
    pred_sh_m = float(metrics.get("pred_sharpe_mean", 0.0))
    pred_sh_s = float(metrics.get("pred_sharpe_std", 0.0))
    pred_dd_m = float(metrics.get("pred_avg_max_dd", 0.0))
    metrics["score_lambda"] = lam
    metrics["score_mu"] = mu
    metrics["rule_score"] = float(rule_sh_m - lam * rule_sh_s - mu * rule_dd_m)
    metrics["pred_score"] = float(pred_sh_m - lam * pred_sh_s - mu * pred_dd_m)

    # Mean-only survivability (gateable, stable)
    try:
        ms = compute_mean_system_health(metrics).as_metrics()
        metrics.update(ms)
    except Exception:
        pass

    meta = {
        "cfg": {
            "mode_col": cfg.mode_col,
            "state_keys": list(cfg.state_keys),
            "split_cfg": asdict(cfg.split_cfg),
            "bc_cfg": asdict(cfg.bc_cfg),
            "sim_cfg": asdict(cfg.sim_cfg),
            "score": {
                "formula": "Sharpe_mean - lambda*Sharpe_std - mu*DD_mean",
                "lambda": float(cfg.score_lambda),
                "mu": float(cfg.score_mu),
            },
        },
        "train_meta": train_meta,
    }

    # Merge router-aligned metrics/meta (best-effort)
    try:
        if isinstance(router_diag, dict) and router_diag:
            meta["router_diag"] = router_diag
            # also store thresholds explicitly for traceability
            meta["router_thresholds"] = router_diag.get("thresholds", {})
        if isinstance(router_diag_per_symbol, pd.DataFrame) and len(
            router_diag_per_symbol
        ):
            meta["router_diag_per_symbol"] = router_diag_per_symbol.to_dict(
                orient="records"
            )
        if isinstance(router_roll_tail, pd.DataFrame) and len(router_roll_tail):
            meta["router_diag_rolling_trade_tail"] = router_roll_tail.to_dict(
                orient="records"
            )

        # add scalar metrics for quick scanning
        if "router_diag" in meta and isinstance(meta["router_diag"], dict):
            tr = (
                meta["router_diag"].get("trade", {})
                if isinstance(meta["router_diag"], dict)
                else {}
            )
            if isinstance(tr, dict):
                metrics["router_diag__trade_n"] = float(tr.get("n", 0.0))
                metrics["router_diag__trade_win_rate"] = float(tr.get("win_rate", 0.0))
                metrics["router_diag__trade_avg_ret"] = float(tr.get("avg_ret", 0.0))
            metrics["router_diag__trade_rate"] = float(tr.get("n", 0.0)) / float(
                max(1.0, float(metrics.get("test_steps", 0.0)))
            )

            rm = meta["router_diag"].get("rank_metrics_trade_slice", {})
            if isinstance(rm, dict):
                for name in ["mfe", "eff", "dir_conf"]:
                    d = rm.get(name, {})
                    if isinstance(d, dict):
                        metrics[f"router_diag__{name}_auc"] = float(d.get("auc", 0.5))
                        metrics[f"router_diag__{name}_ap"] = float(d.get("ap", 0.0))
    except Exception:
        pass

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (p / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        per_symbol.to_csv(p / "per_symbol.csv", index=False)
        if isinstance(router_diag_per_symbol, pd.DataFrame) and len(
            router_diag_per_symbol
        ):
            router_diag_per_symbol.to_csv(p / "router_diag_per_symbol.csv", index=False)
        if isinstance(router_roll_tail, pd.DataFrame) and len(router_roll_tail):
            router_roll_tail.to_csv(
                p / "router_diag_rolling_trade_tail.csv", index=False
            )

        def _fmt(v: Any, digits: int = 4) -> str:
            try:
                if v is None:
                    return "N/A"
                x = float(v)
                if not np.isfinite(x):
                    return "N/A"
                return f"{x:.{digits}f}"
            except Exception:
                return str(v)

        def _pill(text: str, kind: str) -> str:
            cls = f"pill pill-{kind}"
            return f"<span class='{cls}'>{text}</span>"

        # Build an executive summary (conclusion-oriented)
        rule_sh = float(metrics.get("rule_sharpe_mean", 0.0))
        rule_pcm_sh = float(metrics.get("rule_pcm_sharpe_mean", 0.0))
        pred_sh = float(metrics.get("pred_sharpe_mean", 0.0))
        mean_only_sh = float(metrics.get("mean_only_sharpe_mean", 0.0))
        rule_dd = float(metrics.get("rule_avg_max_dd", 0.0))
        rule_pcm_dd = float(metrics.get("rule_pcm_avg_max_dd", 0.0))
        pred_dd = float(metrics.get("pred_avg_max_dd", 0.0))
        mean_only_dd = float(metrics.get("mean_only_avg_max_dd", 0.0))
        rule_tr = float(metrics.get("rule_avg_total_return", 0.0))
        rule_pcm_tr = float(metrics.get("rule_pcm_avg_total_return", 0.0))
        rule_surv_sh = float(metrics.get("rule_survival_sharpe_mean", 0.0))
        rule_surv_dd = float(metrics.get("rule_survival_avg_max_dd", 0.0))
        rule_surv_tr = float(metrics.get("rule_survival_avg_total_return", 0.0))
        pred_tr = float(metrics.get("pred_avg_total_return", 0.0))
        mean_only_tr = float(metrics.get("mean_only_avg_total_return", 0.0))
        trade_rate = float(metrics.get("router_diag__trade_rate", 0.0))

        def _judge() -> Tuple[str, str]:
            # heuristic: this is NOT a deployment gate, just a quick reading aid
            if rule_sh >= 1.0 and rule_dd <= 0.12:
                return ("Rule looks promising under this execution assumption", "good")
            if rule_sh > 0.0:
                return (
                    "Rule is mildly positive; needs robustness check (cost/delay/slippage)",
                    "warn",
                )
            return ("Rule is not positive under this execution assumption", "bad")

        headline, kind = _judge()

        # Key metrics table (remove noisy info by default)
        key_rows = [
            ("Rule Sharpe (mean)", _fmt(rule_sh, 3)),
            ("Rule MaxDD (mean)", _fmt(rule_dd, 3)),
            ("Rule TotalReturn (mean)", _fmt(rule_tr, 3)),
            ("Rule+PCM Sharpe (mean)", _fmt(rule_pcm_sh, 3)),
            ("Rule+PCM MaxDD (mean)", _fmt(rule_pcm_dd, 3)),
            ("Rule+PCM TotalReturn (mean)", _fmt(rule_pcm_tr, 3)),
            ("Rule+Survival Sharpe (mean)", _fmt(rule_surv_sh, 3)),
            ("Rule+Survival MaxDD (mean)", _fmt(rule_surv_dd, 3)),
            ("Rule+Survival TotalReturn (mean)", _fmt(rule_surv_tr, 3)),
            ("Mean-only Sharpe (mean)", _fmt(mean_only_sh, 3)),
            ("Mean-only MaxDD (mean)", _fmt(mean_only_dd, 3)),
            ("Mean-only TotalReturn (mean)", _fmt(mean_only_tr, 3)),
            ("Pred Sharpe (mean)", _fmt(pred_sh, 3)),
            ("Pred MaxDD (mean)", _fmt(pred_dd, 3)),
            ("Pred TotalReturn (mean)", _fmt(pred_tr, 3)),
            ("Trade rate (Rule, test split)", _fmt(trade_rate, 3)),
            ("Test symbols", _fmt(metrics.get("test_symbols"), 0)),
            ("Test steps", _fmt(metrics.get("test_steps"), 0)),
        ]

        key_tbl = "".join(
            [f"<tr><td>{k}</td><td><code>{v}</code></td></tr>" for k, v in key_rows]
        )

        # Router-aligned table (pretty; no raw json dump)
        router_diag = meta.get("router_diag") if isinstance(meta, dict) else None
        router_tbl = ""
        if isinstance(router_diag, dict) and router_diag:
            th = router_diag.get("thresholds") or {}
            rm = router_diag.get("rank_metrics_trade_slice") or {}
            router_tbl += "<div class='card' style='margin-top:16px;'>"
            router_tbl += (
                "<h2>Router-aligned diagnostics (Rule actions, test split)</h2>"
            )
            router_tbl += "<p class='muted'>目的：用 Router 一致的派生量（mfe/eff/dir_conf）在 <b>trade slice</b> 上做“阈值可用性/排序可用性”诊断，解释为什么“回归指标一般但 Router 仍可能有效”。</p>"

            # thresholds
            router_tbl += "<h3>Thresholds</h3>"
            router_tbl += "<p class='muted' style='margin-top:0;'>Key points：这是本次 Router 的关键旋钮（用于复盘与对齐口径）。如果你发现 e2e 与 tuner 不一致，第一步先核对这里是否一致。</p>"
            router_tbl += (
                "<table><thead><tr><th>name</th><th>value</th></tr></thead><tbody>"
            )
            for k in ["mfe_min", "eff_min", "dir_conf_trend_min"]:
                if k in th:
                    router_tbl += f"<tr><td>{k}</td><td><code>{_fmt(th.get(k), 6)}</code></td></tr>"
            router_tbl += "</tbody></table>"
            try:
                tr = router_diag.get("trade") or {}
                tr_n = float(tr.get("n", 0.0)) if isinstance(tr, dict) else 0.0
                tr_rate = float(metrics.get("router_diag__trade_rate", 0.0))
                if tr_rate < 0.12:
                    msg = f"结论：trade slice 覆盖率偏低（trade_rate={_fmt(tr_rate,3)}，n={_fmt(tr_n,0)}），容易样本不足；优先检查阈值是否过严或 data coverage 是否异常。"
                elif tr_rate > 0.65:
                    msg = f"结论：trade slice 覆盖率偏高（trade_rate={_fmt(tr_rate,3)}，n={_fmt(tr_n,0)}），可能过度交易；优先检查成本/延迟敏感性。"
                else:
                    msg = f"结论：trade slice 覆盖率在可用区间（trade_rate={_fmt(tr_rate,3)}，n={_fmt(tr_n,0)}），可以继续看阈值过滤是否带来 avg_ret 改善。"
                router_tbl += f"<p style='margin:8px 0 0;'>{msg}</p>"
            except Exception:
                pass

            # subsets
            router_tbl += "<h3>Threshold subsets (trade slice)</h3>"
            router_tbl += "<p class='muted' style='margin-top:0;'>Key points：看 <code>n</code>（样本量是否足够）+ <code>win_rate</code>/<code>avg_ret</code>（阈值过滤后是否更“干净”）。如果过滤后 avg_ret 没提升，说明阈值可能只是调参与噪声。</p>"
            router_tbl += "<table><thead><tr><th>subset</th><th>n</th><th>win_rate</th><th>avg_ret</th></tr></thead><tbody>"
            for name in [
                "trade",
                "mfe_ge_mfe_min",
                "eff_ge_eff_min",
                "dir_conf_ge_dir_conf_trend_min",
            ]:
                d = router_diag.get(name) or {}
                if isinstance(d, dict):
                    router_tbl += (
                        "<tr>"
                        f"<td><code>{name}</code></td>"
                        f"<td><code>{_fmt(d.get('n'), 0)}</code></td>"
                        f"<td><code>{_fmt(d.get('win_rate'), 3)}</code></td>"
                        f"<td><code>{_fmt(d.get('avg_ret'), 6)}</code></td>"
                        "</tr>"
                    )
            router_tbl += "</tbody></table>"
            try:
                base = router_diag.get("trade") or {}
                base_avg = (
                    float(base.get("avg_ret", 0.0)) if isinstance(base, dict) else 0.0
                )
                base_n = float(base.get("n", 0.0)) if isinstance(base, dict) else 0.0
                improvements = []
                for name in [
                    "mfe_ge_mfe_min",
                    "eff_ge_eff_min",
                    "dir_conf_ge_dir_conf_trend_min",
                ]:
                    d = router_diag.get(name) or {}
                    if not isinstance(d, dict):
                        continue
                    n = float(d.get("n", 0.0))
                    avg = float(d.get("avg_ret", 0.0))
                    if n >= max(30.0, 0.1 * max(1.0, base_n)):
                        improvements.append((name, avg - base_avg, n, avg))
                if improvements:
                    best = sorted(improvements, key=lambda x: x[1], reverse=True)[0]
                    delta = best[1]
                    sign = "提升" if delta > 0 else "下降"
                    router_tbl += (
                        "<p style='margin:8px 0 0;'>"
                        f"结论：相对 trade 基线 avg_ret={_fmt(base_avg,6)}，"
                        f"过滤子集里“最有提升”的是 <b><code>{best[0]}</code></b>（n={_fmt(best[2],0)}，avg_ret={_fmt(best[3],6)}，Δ={_fmt(delta,6)}，{sign}）。"
                        "</p>"
                    )
                else:
                    router_tbl += (
                        "<p style='margin:8px 0 0;'>"
                        "结论：过滤子集样本量不足或提升不明显（无法稳定判断阈值是否真的带来 avg_ret 改善）；建议扩大窗口或降低阈值严苛度再看。"
                        "</p>"
                    )
            except Exception:
                pass

            # rank metrics
            if isinstance(rm, dict) and rm:
                router_tbl += "<h3>Rank metrics (trade slice)</h3>"
                router_tbl += "<p class='muted' style='margin-top:0;'>Key points：AUC 基线是 <b>0.5</b>。明显 >0.5 说明排序有用；AP 要结合正样本比例看（越高越好）。如果 AUC≈0.5，则 head 在 trade slice 里对“好/坏一步收益”几乎没区分。</p>"
                router_tbl += "<table><thead><tr><th>signal</th><th>AUC</th><th>AP</th></tr></thead><tbody>"
                for sig in ["mfe", "eff", "dir_conf"]:
                    d = rm.get(sig) or {}
                    if isinstance(d, dict):
                        router_tbl += (
                            "<tr>"
                            f"<td><code>{sig}</code></td>"
                            f"<td><code>{_fmt(d.get('auc'), 3)}</code></td>"
                            f"<td><code>{_fmt(d.get('ap'), 3)}</code></td>"
                            "</tr>"
                        )
                router_tbl += "</tbody></table>"
                try:
                    aucs = []
                    for sig in ["mfe", "eff", "dir_conf"]:
                        d = rm.get(sig) or {}
                        if isinstance(d, dict):
                            aucs.append((sig, float(d.get("auc", 0.5))))
                    if aucs:
                        best = sorted(aucs, key=lambda x: x[1], reverse=True)[0]
                        auc = best[1]
                        if auc >= 0.52:
                            msg = f"结论：排序信号整体<b>略强于随机</b>，最佳是 <b><code>{best[0]}</code></b>（AUC={_fmt(auc,3)}）。"
                        elif auc <= 0.48:
                            msg = f"结论：排序信号整体<b>弱于随机</b>（可能反向/噪声），最佳也只有 AUC={_fmt(auc,3)}。"
                        else:
                            msg = f"结论：排序信号整体<b>接近随机</b>（AUC≈0.5），当前 head 对 trade slice 的一步好坏区分不强。"
                        router_tbl += f"<p style='margin:8px 0 0;'>{msg}</p>"
                except Exception:
                    pass
            router_tbl += "</div>"

        # Per-symbol main table (sorted by rule_sharpe desc)
        per_symbol_view = per_symbol.copy()
        if "rule_sharpe" in per_symbol_view.columns:
            per_symbol_view = per_symbol_view.sort_values(
                "rule_sharpe", ascending=False
            ).reset_index(drop=True)

        # Per-symbol conclusion (one-liner)
        per_symbol_conclusion = ""
        try:
            if len(per_symbol_view) > 0 and "rule_sharpe" in per_symbol_view.columns:
                best_sym = str(per_symbol_view.iloc[0]["symbol"])
                best_sh = float(per_symbol_view.iloc[0]["rule_sharpe"])
                worst_sym = str(per_symbol_view.iloc[-1]["symbol"])
                worst_sh = float(per_symbol_view.iloc[-1]["rule_sharpe"])
                n_pos = int(
                    np.sum(
                        pd.to_numeric(per_symbol_view["rule_sharpe"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy()
                        > 0.0
                    )
                )
                per_symbol_conclusion = (
                    f"结论：<b>{best_sym}</b> 贡献最大（rule_sharpe={_fmt(best_sh,3)}），"
                    f"<b>{worst_sym}</b> 拖累最大（rule_sharpe={_fmt(worst_sh,3)}）；"
                    f"正 Sharpe 的币数：<b>{n_pos}/{int(len(per_symbol_view))}</b>。"
                )
        except Exception:
            per_symbol_conclusion = ""

        # Executive summary conclusion (one-liner)
        summary_conclusion = ""
        try:
            summary_conclusion = (
                "结论：在当前执行假设下，Rule 的 <b>Sharpe</b> 为 <b>"
                + _fmt(rule_sh, 3)
                + "</b>，<b>MaxDD</b> 为 <b>"
                + _fmt(rule_dd, 3)
                + "</b>，参与率（trade rate）为 <b>"
                + _fmt(trade_rate, 3)
                + "</b>；下一步应优先做 <b>cost/slippage/entry_delay</b> 稳健性复验。"
            )
        except Exception:
            summary_conclusion = ""

        # Router per-symbol table (optional)
        router_sym_html = ""
        try:
            p_sym = p / "router_diag_per_symbol.csv"
            if p_sym.exists():
                df_sym = pd.read_csv(p_sym)
                router_sym_html = (
                    "<div class='card' style='margin-top:16px;'>"
                    "<h3>Router diag per symbol (trade slice)</h3>"
                    "<p class='muted'>Key points：用来快速定位“某个币拖累/某个币很强”。优先看 <code>trade_rate</code>（是否被频繁交易）与 <code>avg_ret</code>/<code>win_rate</code>（是否负贡献），再结合 AUC（是否 head 在该币上有排序力）。</p>"
                    f"{df_sym.to_html(index=False, escape=True)}"
                    "</div>"
                )
        except Exception:
            router_sym_html = ""

        # Rolling drift tail (optional)
        roll_html = ""
        try:
            p_roll = p / "router_diag_rolling_trade_tail.csv"
            if p_roll.exists():
                df_roll = pd.read_csv(p_roll)
                if len(df_roll):
                    roll_html = (
                        "<div class='card' style='margin-top:16px;'>"
                        "<h3>Rolling drift preview (trade slice, tail)</h3>"
                        "<p class='muted'>Key points：滚动窗口内的 mfe/eff/dir_conf 均值与 ret/胜率预览，用于检查近期是否漂移。若 <code>roll_ret_mean</code> 由正转负、或信号均值结构突变，说明可能进入不适配 regime（需要降级/停机/换阈值）。</p>"
                        f"{df_roll.to_html(index=False, escape=True)}"
                        "</div>"
                    )
        except Exception:
            roll_html = ""

        # Raw metrics table (collapsed)
        raw_rows = []
        for k in sorted(metrics.keys()):
            v = metrics.get(k)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                s = f"{float(v):.6g}"
            else:
                s = str(v)
            raw_rows.append(f"<tr><td><code>{k}</code></td><td>{s}</td></tr>")
        raw_tbl = (
            "<details style='margin-top:16px;'>"
            "<summary style='cursor:pointer;font-weight:600;'>Raw metrics (debug)</summary>"
            "<div class='card' style='margin-top:10px;'>"
            "<p class='muted' style='margin-top:0;'>Key points：这里是完整指标字典，主要用于排查异常（比如 trade_rate=0、Pred collapsed、std 极大导致 score 异常等）。日常复盘只看上面的 summary/Router 诊断即可。</p>"
            "<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>"
            + "".join(raw_rows)
            + "</tbody></table></div></details>"
        )

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Counterfactual Eval - Rule vs BC(3-action)</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#0f172a;background:#fafafa}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff;border-radius:12px;overflow:hidden}}
th,td{{border-bottom:1px solid #eef2f7;text-align:left;padding:8px 10px;vertical-align:top}}
th{{background:#f8fafc;font-weight:600}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{border:1px solid #e5e7eb;border-radius:16px;padding:16px 18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04)}}
.muted{{color:#64748b}}
.kpi{{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}}
.pill{{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid #e2e8f0;background:#f8fafc}}
.pill-good{{border-color:#86efac;background:#f0fdf4;color:#166534}}
.pill-warn{{border-color:#fde68a;background:#fffbeb;color:#92400e}}
.pill-bad{{border-color:#fecaca;background:#fef2f2;color:#991b1b}}
.h1{{font-size:22px;margin:0}}
.sub{{margin:6px 0 0}}
</style></head>
<body>
<div class="card">
  <div class="h1">Counterfactual Eval (3-action) — Rule vs BC Router</div>
  <p class="sub muted">同一份 logs（同一份 <code>{cfg.sim_cfg.ret_mean_col}</code>/<code>{cfg.sim_cfg.ret_trend_col}</code>），对比 Rule mode 与 BC 预测 mode 的收益表现。下面优先展示“结论导向”的统计。</p>
  <div class="kpi">
    {_pill(headline, kind)}
    {_pill(f"Rule Sharpe={_fmt(rule_sh,3)}", "good" if rule_sh>0 else "bad")}
    {_pill(f"Rule MaxDD={_fmt(rule_dd,3)}", "good" if rule_dd<0.12 else "warn")}
    {_pill(f"Trade rate={_fmt(trade_rate,3)}", "warn" if trade_rate<0.15 else "good")}
  </div>
</div>

<div class="card" style="margin-top:16px;">
  <h2 style="margin:0 0 10px;">Executive summary (key stats)</h2>
  <p class="muted" style="margin-top:0;">Key points：这是“最少信息量”的结论面板——先看 Rule 的 Sharpe/DD/Trade rate（是否值得进一步做稳健性复验），再对比 Pred（BC Router）是否退化。</p>
  <table><thead><tr><th>item</th><th>value</th></tr></thead><tbody>{key_tbl}</tbody></table>
  <p class="muted" style="margin-top:10px;">
    <b>解读建议</b>：先看 Rule 的 Sharpe/DD/Trade rate；再看 Router-aligned 的 AUC/AP（是否 > 0.5）；最后看 per-symbol 是否被单一币驱动。
  </p>
  <p style="margin:8px 0 0;">{summary_conclusion}</p>
</div>

<div class="card" style="margin-top:16px;">
  <h2 style="margin:0 0 10px;">Per symbol (PnL)</h2>
  <p class="muted" style="margin:0 0 10px;">Key points：按 <code>rule_sharpe</code> 从高到低排序（用于找贡献者/拖累者）。如果组合 Sharpe 很好但只有 1 个币很强，说明存在集中风险；反之多币同时为正更稳健。</p>
  {per_symbol_view.to_html(index=False, escape=True)}
  <p style="margin:8px 0 0;">{per_symbol_conclusion}</p>
</div>

{router_tbl}
{router_sym_html}
{roll_html}
{raw_tbl}
</body></html>
"""
        (p / "report.html").write_text(html, encoding="utf-8")

    return meta, metrics, per_symbol
