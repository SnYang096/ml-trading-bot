from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

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
    """

    mode_col: str = "mode"
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
    if cfg.mode_col not in df_logs.columns:
        raise ValueError(f"Missing mode column '{cfg.mode_col}' in logs.")
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
    )  # mode is first-class

    transitions_train = [
        {
            "state": {k: r.get(k, 0.0) for k in cfg.state_keys},
            "action": {"mode": r.get(cfg.mode_col)},
        }
        for r in train_df.to_dict(orient="records")
    ]

    model, train_meta = train_bc_router3_policy(
        transitions=transitions_train,
        state_schema=state_schema,
        infer_cfg=infer_cfg,
        cfg=cfg.bc_cfg,
    )

    # test actions
    rule_actions = np.asarray(
        [_mode_to_action(m) for m in test_df[cfg.mode_col].tolist()], dtype=np.int64
    )
    pred_actions = _predict_modes(
        test_df,
        model=model,
        state_schema=state_schema,
        state_keys=cfg.state_keys,
        device=cfg.bc_cfg.device,
    )

    # per-symbol simulation
    rows = []
    for sym, g in test_df.groupby(cfg.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        ra = rule_actions[test_df[cfg.symbol_col].values == sym]
        pa = pred_actions[test_df[cfg.symbol_col].values == sym]

        out_rule = simulate_3action_episode(g, actions=ra.tolist(), cfg=cfg.sim_cfg)
        out_pred = simulate_3action_episode(g, actions=pa.tolist(), cfg=cfg.sim_cfg)

        eq_r = out_rule["equity"].to_numpy(dtype=float)
        eq_p = out_pred["equity"].to_numpy(dtype=float)
        pnl_r = (
            out_rule["pnl"].to_numpy(dtype=float) if "pnl" in out_rule.columns else None
        )
        pnl_p = (
            out_pred["pnl"].to_numpy(dtype=float) if "pnl" in out_pred.columns else None
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
                "pred_total_return": _total_return(eq_p),
                "rule_max_dd": _max_drawdown(eq_r),
                "pred_max_dd": _max_drawdown(eq_p),
                "rule_ann_return": _ann_return_from_equity(
                    eq_r, steps_per_year=steps_per_year
                ),
                "pred_ann_return": _ann_return_from_equity(
                    eq_p, steps_per_year=steps_per_year
                ),
                "rule_ann_vol": (
                    _ann_vol_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "pred_ann_vol": (
                    _ann_vol_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "rule_sharpe": (
                    _sharpe_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "pred_sharpe": (
                    _sharpe_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "rule_sortino": (
                    _sortino_from_pnl(pnl_r, steps_per_year=steps_per_year)
                    if pnl_r is not None
                    else 0.0
                ),
                "pred_sortino": (
                    _sortino_from_pnl(pnl_p, steps_per_year=steps_per_year)
                    if pnl_p is not None
                    else 0.0
                ),
                "rule_turnover_mean": (
                    float(out_rule["turnover"].mean()) if len(out_rule) else 0.0
                ),
                "pred_turnover_mean": (
                    float(out_pred["turnover"].mean()) if len(out_pred) else 0.0
                ),
                "rule_switch_rate": _switch_rate(ra),
                "pred_switch_rate": _switch_rate(pa),
                "rule_mode_entropy": _entropy(ra),
                "pred_mode_entropy": _entropy(pa),
                "rule_final_equity": (
                    float(eq_r[-1]) if len(eq_r) else float(cfg.sim_cfg.initial_equity)
                ),
                "pred_final_equity": (
                    float(eq_p[-1]) if len(eq_p) else float(cfg.sim_cfg.initial_equity)
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
        "pred_avg_total_return": (
            float(per_symbol["pred_total_return"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_avg_max_dd": (
            float(per_symbol["rule_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_max_dd": (
            float(per_symbol["pred_max_dd"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_avg_switch_rate": (
            float(per_symbol["rule_switch_rate"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_switch_rate": (
            float(per_symbol["pred_switch_rate"].mean()) if len(per_symbol) else 0.0
        ),
        "rule_avg_mode_entropy": (
            float(per_symbol["rule_mode_entropy"].mean()) if len(per_symbol) else 0.0
        ),
        "pred_avg_mode_entropy": (
            float(per_symbol["pred_mode_entropy"].mean()) if len(per_symbol) else 0.0
        ),
    }

    # Risk metrics (mean/std across symbols)
    if len(per_symbol):
        for side in ["rule", "pred"]:
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

        # Minimal HTML report (dependency-free)
        html_rows = []
        for k in sorted(metrics.keys()):
            v = metrics.get(k)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                s = f"{float(v):.6g}"
            else:
                s = str(v)
            html_rows.append(f"<tr><td><code>{k}</code></td><td>{s}</td></tr>")
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Counterfactual Eval - Rule vs BC(3-action)</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#111}}
code{{background:#f4f4f5;padding:2px 6px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border-bottom:1px solid #eee;text-align:left;padding:6px 8px;vertical-align:top}}
th{{background:#fafafa}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;background:#fff}}
.muted{{color:#6b7280}}
</style></head>
<body>
<h1>Counterfactual Eval - Rule vs BC(3-action)</h1>
<p class="muted">A/B compare on the <em>same</em> precomputed mode returns and the <em>same</em> simulator costs/constraints.</p>

<div class="card" style="margin:16px 0;">
  <h2 style="margin-top:0;">How to read this report</h2>
  <ul style="margin:0; padding-left:18px;">
    <li><b>Rule</b> metrics: equity built by using the logged <code>mode</code> column (NO_TRADE/MEAN/TREND).</li>
    <li><b>Pred</b> metrics: equity built by a small BC Router trained on the first <code>train_ratio</code> portion, then used to predict mode on the test portion.</li>
    <li>Returns are counterfactual step returns from logs: <code>{cfg.sim_cfg.ret_mean_col}</code> and <code>{cfg.sim_cfg.ret_trend_col}</code>. When mode=NO_TRADE, return is 0.</li>
    <li>If <b>Pred Sharpe = 0</b> and <b>Pred mode entropy = 0</b>, it usually means the BC Router collapsed (often all NO_TRADE). This is a Router-learning issue, not necessarily an NN-head issue.</li>
    <li><b>Per symbol</b> table helps spot whether performance is driven by a single asset (e.g., ETH good, BTC flat).</li>
  </ul>
</div>
<div class="grid">
  <div class="card">
    <h2>Aggregate metrics</h2>
    <table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table>
  </div>
  <div class="card">
    <h2>Per symbol</h2>
    {per_symbol.to_html(index=False, escape=True)}
  </div>
</div>
</body></html>
"""
        (p / "report.html").write_text(html, encoding="utf-8")

    return meta, metrics, per_symbol
