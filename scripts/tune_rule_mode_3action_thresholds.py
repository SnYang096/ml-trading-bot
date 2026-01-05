#!/usr/bin/env python3
"""
Fast threshold tuning for `mlbot rule mode-3action` against counterfactual returns.

Why this script exists:
  - Recomputing rr_execution (ret_mean/ret_trend) per trial is slow.
  - `rl counterfactual-eval-3action` trains a BC model each run; for tuning rule thresholds,
    we only need rule-mode PnL metrics.

This script:
  1) Loads nnmultihead preds (preds_*.parquet) containing pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe.
  2) Loads pre-built logs_3action.parquet that already contains ret_mean/ret_trend (rr_execution).
  3) For each threshold set, recomputes mode_action, simulates equity, and reports rule_sharpe_mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.sim_env_3action import (
    SimEnvConfig,
    simulate_3action_episode,
)
from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_dir: Path) -> List[Path]:
    if preds_dir.is_dir():
        files = sorted(preds_dir.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_dir.glob("*.parquet"))
        return files
    return [preds_dir]


def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out["timestamp"] = out.index
        return out
    raise ValueError("Expected preds to have a timestamp column or DatetimeIndex.")


def _infer_steps_per_year(ts: pd.Series) -> float:
    dt = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    dt = dt.sort_values().drop_duplicates()
    if len(dt) < 3:
        return 1.0
    diffs = dt.diff().dropna()
    if len(diffs) == 0:
        return 1.0
    step_sec = float(diffs.median().total_seconds())
    if not np.isfinite(step_sec) or step_sec <= 0:
        return 1.0
    sec_per_year = 365.25 * 24.0 * 3600.0
    return float(max(1.0, min(1e6, sec_per_year / step_sec)))


def _sharpe(pnl: np.ndarray, steps_per_year: float) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(mu / sd * np.sqrt(float(steps_per_year)))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = float(equity[0])
    mdd = 0.0
    for x in equity:
        peak = max(peak, float(x))
        dd = (peak - float(x)) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def _load_preds_by_symbol(preds_path: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for f in _collect_pred_files(preds_path):
        df = _ensure_timestamp_col(_read_parquet_or_csv(f))
        sym = (
            str(df["symbol"].iloc[0])
            if "symbol" in df.columns
            else f.stem.replace("preds_", "")
        )
        df = df.copy()
        df["symbol"] = sym
        # Ensure timestamp dtype is consistent (tz-naive like logs)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        ).dt.tz_localize(None)
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        out[sym] = df
    return out


def _load_logs_by_symbol(logs_path: Path) -> Dict[str, pd.DataFrame]:
    df = _read_parquet_or_csv(logs_path)
    if "symbol" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("logs must contain columns: symbol,timestamp")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(
        None
    )
    df = df.dropna(subset=["timestamp"])
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=False):
        out[str(sym)] = g.sort_values("timestamp").reset_index(drop=True)
    return out


def _eval_thresholds(
    *,
    preds_by_sym: Dict[str, pd.DataFrame],
    logs_by_sym: Dict[str, pd.DataFrame],
    cfg: Rule3ActionConfig,
    preds_in_log1p: bool,
    sim_cfg: SimEnvConfig,
) -> Dict[str, float]:
    sharpe_by_sym: List[float] = []
    dd_by_sym: List[float] = []
    trade_rate_by_sym: List[float] = []

    for sym, logs in logs_by_sym.items():
        preds = preds_by_sym.get(sym)
        if preds is None or preds.empty:
            continue

        mode_df = compute_mode_3action(preds, cfg=cfg, preds_in_log1p=preds_in_log1p)
        # Align to logs timestamps
        s = mode_df["mode_action"].astype(int)
        s.index = pd.to_datetime(preds["timestamp"], errors="coerce").dt.tz_localize(
            None
        )
        actions = (
            s.reindex(
                pd.to_datetime(logs["timestamp"], errors="coerce").dt.tz_localize(None)
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        sim = simulate_3action_episode(logs, actions=actions.tolist(), cfg=sim_cfg)
        pnl = (
            sim["pnl"].to_numpy(dtype=float)
            if "pnl" in sim.columns
            else np.zeros(len(sim))
        )
        equity = (
            sim["equity"].to_numpy(dtype=float)
            if "equity" in sim.columns
            else np.ones(len(sim))
        )

        steps_per_year = _infer_steps_per_year(logs["timestamp"])
        sharpe_by_sym.append(_sharpe(pnl, steps_per_year))
        dd_by_sym.append(_max_drawdown(equity))
        trade_rate_by_sym.append(float(np.mean(actions != 0)))

    if not sharpe_by_sym:
        return {
            "rule_sharpe_mean": 0.0,
            "rule_sharpe_std": 0.0,
            "rule_dd_mean": 0.0,
            "trade_rate_mean": 0.0,
        }

    return {
        "rule_sharpe_mean": float(np.mean(sharpe_by_sym)),
        "rule_sharpe_std": (
            float(np.std(sharpe_by_sym, ddof=1)) if len(sharpe_by_sym) > 1 else 0.0
        ),
        "rule_dd_mean": float(np.mean(dd_by_sym)),
        "trade_rate_mean": float(np.mean(trade_rate_by_sym)),
        "n_symbols": float(len(sharpe_by_sym)),
    }


def _load_preds_in_log1p_from_model(model_path: Path) -> bool:
    payload = torch.load(str(model_path), map_location="cpu")
    meta = payload.get("meta") or {}
    ds_cfg = meta.get("dataset_cfg") or {}
    return bool(ds_cfg.get("log1p_targets", True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds file/dir (preds_*.parquet)")
    ap.add_argument(
        "--logs",
        required=True,
        help="logs_3action.parquet (must contain ret_mean/ret_trend)",
    )
    ap.add_argument("--model", required=True, help="model.pt (to infer preds_in_log1p)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entry-delay", type=int, default=0)
    ap.add_argument("--cost-per-turnover", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = Path(args.preds)
    logs_path = Path(args.logs)
    model_path = Path(args.model)

    preds_in_log1p = _load_preds_in_log1p_from_model(model_path)
    preds_by_sym = _load_preds_by_symbol(preds_path)
    logs_by_sym = _load_logs_by_symbol(logs_path)

    # Intersect symbols to avoid missing data
    keep = sorted(set(preds_by_sym.keys()) & set(logs_by_sym.keys()))
    preds_by_sym = {k: preds_by_sym[k] for k in keep}
    logs_by_sym = {k: logs_by_sym[k] for k in keep}

    sim_cfg = SimEnvConfig(
        entry_delay=int(args.entry_delay),
        cost_per_turnover=float(args.cost_per_turnover),
        slippage_bps=float(args.slippage_bps),
        initial_equity=1.0,
    )

    rng = np.random.default_rng(int(args.seed))

    # Seed candidates: include defaults + a few relaxed presets
    base = Rule3ActionConfig()
    candidates: List[Rule3ActionConfig] = [
        base,
        Rule3ActionConfig(**{**asdict(base), "mfe_min": 0.2, "eff_min": 1.02}),
        Rule3ActionConfig(
            **{**asdict(base), "mfe_min": 0.15, "eff_min": 1.01, "eff_mean_min": 1.05}
        ),
        Rule3ActionConfig(
            **{**asdict(base), "mfe_min": 0.25, "eff_min": 1.03, "ttm_mean_max": 20.0}
        ),
    ]

    def sample_cfg() -> Rule3ActionConfig:
        # Sample within plausible ranges; enforce a few monotonic constraints.
        mfe_min = float(rng.uniform(0.05, 0.6))
        eff_min = float(rng.uniform(1.0, 1.25))
        eff_mean_min = float(rng.uniform(max(eff_min, 1.02), 1.6))
        ttm_mean_max = float(rng.uniform(4.0, 30.0))
        dir_conf_trend_min = float(rng.uniform(0.05, 0.6))
        mfe_trend_min = float(rng.uniform(max(mfe_min, 0.2), 1.6))
        ttm_trend_min = float(rng.uniform(2.0, 24.0))
        return Rule3ActionConfig(
            **{
                **asdict(base),
                "mfe_min": mfe_min,
                "eff_min": eff_min,
                "eff_mean_min": eff_mean_min,
                "ttm_mean_max": ttm_mean_max,
                "dir_conf_trend_min": dir_conf_trend_min,
                "mfe_trend_min": mfe_trend_min,
                "ttm_trend_min": ttm_trend_min,
            }
        )

    for _ in range(int(args.n_trials)):
        candidates.append(sample_cfg())

    rows = []
    best = None
    for i, cfg in enumerate(candidates):
        metrics = _eval_thresholds(
            preds_by_sym=preds_by_sym,
            logs_by_sym=logs_by_sym,
            cfg=cfg,
            preds_in_log1p=preds_in_log1p,
            sim_cfg=sim_cfg,
        )
        row = {
            "trial": i,
            **{
                f"cfg_{k}": v
                for k, v in asdict(cfg).items()
                if k.endswith(("min", "max"))
            },
            **metrics,
        }
        rows.append(row)
        if best is None or float(metrics["rule_sharpe_mean"]) > float(
            best["rule_sharpe_mean"]
        ):
            best = {**metrics, "cfg": asdict(cfg), "trial": i}

    df = pd.DataFrame(rows)
    df = df.sort_values(["rule_sharpe_mean", "trade_rate_mean"], ascending=False)

    df.to_csv(out_dir / "tuning_trials.csv", index=False)
    if best is None:
        best = {"rule_sharpe_mean": 0.0, "cfg": asdict(base), "trial": 0}
    (out_dir / "best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("✅ Wrote:", out_dir / "tuning_trials.csv")
    print("✅ Best:", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
