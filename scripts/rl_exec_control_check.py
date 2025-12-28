#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Execution control check on logs (invariants + kill-switch)."
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Logs .csv/.parquet (symbol,timestamp,mode,ret_mean,ret_trend,...)",
    )
    ap.add_argument("--out", required=True, help="Output directory for report/metrics.")

    # Sim env knobs (common)
    ap.add_argument("--entry-delay", type=int, default=1)
    ap.add_argument("--cost-per-turnover", type=float, default=0.0002)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--max-drawdown-stop", type=float, default=None)
    ap.add_argument("--cooldown-steps", type=int, default=0)

    # Invariants thresholds
    ap.add_argument("--max-dd", type=float, default=0.35)
    ap.add_argument("--max-turnover-mean", type=float, default=0.35)
    ap.add_argument("--max-turnover-p95", type=float, default=1.0)
    ap.add_argument("--max-cost-mean", type=float, default=0.002)
    ap.add_argument("--max-cost-p95", type=float, default=0.01)
    ap.add_argument("--max-nan-ratio", type=float, default=0.001)
    ap.add_argument("--max-abs-return", type=float, default=0.5)
    args = ap.parse_args()

    df = _read_any(args.logs)
    sim_cfg = SimEnvConfig(
        entry_delay=int(args.entry_delay),
        cost_per_turnover=float(args.cost_per_turnover),
        slippage_bps=float(args.slippage_bps),
        max_drawdown_stop=args.max_drawdown_stop,
        cooldown_steps=int(args.cooldown_steps),
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
    metrics, per_symbol = control_check_from_logs(df, cfg=cfg)
    write_exec_control_artifacts(
        out_dir=str(args.out), metrics=metrics, per_symbol=per_symbol
    )
    print("✅ exec control check saved to:", args.out)
    print("kill_switch:", metrics.get("kill_switch"))


if __name__ == "__main__":
    main()
