#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.exec_control import (  # noqa: E402
    ExecControlConfig,
    control_check_from_logs,
    write_exec_control_artifacts,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig  # noqa: E402


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _inject_nan(df: pd.DataFrame, *, col: str, ratio: float, seed: int) -> pd.DataFrame:
    if col not in df.columns or ratio <= 0:
        return df
    rng = np.random.default_rng(int(seed))
    out = df.copy()
    m = rng.random(len(out)) < float(ratio)
    out.loc[m, col] = np.nan
    return out


def _scale_returns(df: pd.DataFrame, *, factor: float, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * float(factor)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chaos test: perturb logs and run exec control check."
    )
    ap.add_argument("--logs", required=True, help="Logs .csv/.parquet")
    ap.add_argument(
        "--out", required=True, help="Output directory root (baseline/chaos_*)."
    )
    ap.add_argument("--seed", type=int, default=0)

    # Chaos knobs
    ap.add_argument(
        "--nan-ratio",
        type=float,
        default=0.0,
        help="Inject NaNs into ret_mean/ret_trend.",
    )
    ap.add_argument(
        "--return-scale", type=float, default=1.0, help="Multiply ret_* by this factor."
    )
    ap.add_argument("--slippage-bps", type=float, default=0.0, help="Sim slippage bps.")
    ap.add_argument(
        "--cost-per-turnover", type=float, default=0.0002, help="Sim cost per turnover."
    )
    ap.add_argument("--entry-delay", type=int, default=1)

    # Invariants thresholds (same as control-check defaults)
    ap.add_argument("--max-dd", type=float, default=0.35)
    ap.add_argument("--max-turnover-mean", type=float, default=0.35)
    ap.add_argument("--max-turnover-p95", type=float, default=1.0)
    ap.add_argument("--max-cost-mean", type=float, default=0.002)
    ap.add_argument("--max-cost-p95", type=float, default=0.01)
    ap.add_argument("--max-nan-ratio", type=float, default=0.001)
    ap.add_argument("--max-abs-return", type=float, default=0.5)
    args = ap.parse_args()

    df0 = _read_any(args.logs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # baseline check
    sim_cfg = SimEnvConfig(
        entry_delay=int(args.entry_delay),
        cost_per_turnover=float(args.cost_per_turnover),
        slippage_bps=float(args.slippage_bps),
    )
    cfg = ExecControlConfig(
        sim_cfg=sim_cfg,
        max_dd=float(args.max_dd),
        max_turnover_mean=float(args.max_turnover_mean),
        max_turnover_p95=float(args.max_turnover_p95),
        max_cost_mean=float(args.max_cost_mean),
        max_cost_p95=float(args.max_cost_p95),
        max_nan_ratio=float(args.max_nan_ratio),
        max_abs_return=float(args.max_abs_return),
    )

    m0, ps0 = control_check_from_logs(df0, cfg=cfg)
    write_exec_control_artifacts(
        out_dir=str(out_root / "baseline"), metrics=m0, per_symbol=ps0
    )

    # chaos variant
    df = _scale_returns(
        df0, factor=float(args.return_scale), cols=["ret_mean", "ret_trend"]
    )
    df = _inject_nan(
        df, col="ret_mean", ratio=float(args.nan_ratio), seed=int(args.seed)
    )
    df = _inject_nan(
        df, col="ret_trend", ratio=float(args.nan_ratio), seed=int(args.seed) + 1
    )
    m1, ps1 = control_check_from_logs(df, cfg=cfg)
    write_exec_control_artifacts(
        out_dir=str(out_root / "chaos"), metrics=m1, per_symbol=ps1
    )

    # one-line summary
    print("✅ chaos test saved to:", str(out_root))
    print(
        "baseline kill_switch:",
        m0.get("kill_switch"),
        "chaos kill_switch:",
        m1.get("kill_switch"),
    )


if __name__ == "__main__":
    main()
