"""Compare dual_add_trend flip handling: reseed on trend_flip vs wait for next regime.

Runs ``scripts/diagnose_dual_add_trend.py`` for each variant and writes
``ablation_summary.csv`` under --out-root.

Example::

    python scripts/experiment_dual_add_flip_reseed.py \\
      --out-root results/dual_add_flip_reseed_2022_2026 \\
      -- \\
      --config config/strategies/trend_scalp/research/calibrate_roll.default.yaml \\
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \\
      --start 2022-01-01 --end 2026-03-31 \\
      --timeframe 2h --no-initial-hedge \\
      --take-profit-mode basket --risk-stop-mode regime_only --fee-bps 8 \\
      --no-maps
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSE = PROJECT_ROOT / "scripts" / "diagnose_dual_add_trend.py"


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return list(argv[:i]), list(argv[i + 1 :])
    return list(argv), []


def _portfolio_dd_from_segments(segments: pd.DataFrame) -> float:
    if segments.empty or "end" not in segments.columns:
        return 0.0
    df = segments[["end", "pnl_per_capital"]].copy()
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["end"]).sort_values("end")
    if df.empty:
        return 0.0
    cum = df["pnl_per_capital"].cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def _run_variant(
    *,
    variant: str,
    reseed_on_flip: bool,
    flip_action: str,
    out_dir: Path,
    forward: List[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(DIAGNOSE),
        *forward,
        "--flip-action",
        flip_action,
        "--out-dir",
        str(out_dir),
        "--no-maps",
    ]
    if reseed_on_flip:
        cmd.append("--reseed-on-flip")
    else:
        cmd.append("--no-reseed-on-flip")

    print(f"\n=== {variant} ===\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    summary = pd.read_csv(out_dir / "summary.csv")
    segments = pd.read_csv(out_dir / "dual_add_segments.csv")
    row = summary.iloc[0].to_dict()
    row["variant"] = variant
    row["flip_action"] = flip_action
    row["reseed_on_flip"] = reseed_on_flip
    row["portfolio_cum_dd"] = _portfolio_dd_from_segments(segments)
    row["median_segment_max_dd"] = float(segments["max_drawdown"].median())
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        default="results/dual_add_flip_reseed",
        help="Parent directory for per-variant outputs",
    )
    args, forward = _split_argv(sys.argv[1:])
    ns = ap.parse_args(args)

    out_root = Path(ns.out_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    variants = [
        (
            "reseed_on_flip_close_offside",
            True,
            "close_offside_all",
            "flip_close_offside_then_reseed (production default)",
        ),
        (
            "flat_until_next_regime",
            False,
            "close_offside_all",
            "flip_close_offside_no_reseed_wait_regime",
        ),
        (
            "keep_offside_legacy",
            True,
            "keep",
            "flip_keep_offside_hedge_legacy",
        ),
    ]

    rows = []
    for name, reseed, flip, _desc in variants:
        rows.append(
            _run_variant(
                variant=name,
                reseed_on_flip=reseed,
                flip_action=flip,
                out_dir=out_root / name,
                forward=forward,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("sum_pnl_per_capital", ascending=False)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    (out_root / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
