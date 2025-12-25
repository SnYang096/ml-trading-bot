#!/usr/bin/env python3
"""
Seed-sweep runner for `mlbot diagnose model-comparison`.

Goal:
- Quantify variance across random seeds (not within-seed nondeterminism).
- Produce summary (mean/std/min/max) of key metrics per strategy.

Example:
  python3 scripts/run_model_comparison_seed_sweep.py \
    --strategies sr_reversal_long_mvp,sr_reversal_rr_reg_long_mvp \
    --symbol BTCUSDT --timeframe 240T \
    --start-date 2023-01-01 --end-date 2025-10-31 \
    --test-size 0.3 \
    --seeds 1,2,3,4,5 \
    --output-dir results/model_comparison/seed_sweep_mvp
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--strategies", required=True, help="Comma-separated strategy config dirs"
    )
    p.add_argument(
        "--symbol", required=False, help="Single symbol (legacy). Prefer --symbols."
    )
    p.add_argument(
        "--symbols", default=None, help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)"
    )
    p.add_argument("--timeframe", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument(
        "--seeds", default="42", help="Comma-separated seeds (e.g. 1,2,3,4,5)"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Base output dir (each seed gets its own subdir)",
    )
    p.add_argument("--no-docker", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    strategies = args.strategies.strip()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    symbols: list[str]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip()]
    else:
        raise SystemExit("Must provide --symbols or --symbol")

    rows = []
    for symbol in symbols:
        for seed in seeds:
            out_dir = base_out / symbol / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "mlbot",
                "diagnose",
                "model-comparison",
                "--strategy-config",
                strategies,
                "--symbol",
                symbol,
                "--timeframe",
                args.timeframe,
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--test-size",
                str(args.test_size),
                "--seed",
                str(seed),
                "--output-dir",
                str(out_dir),
                "--no-docker",
            ]

            print(f"\n=== Running symbol={symbol} seed={seed} → {out_dir} ===")
            subprocess.run(cmd, check=True)

            csv_path = out_dir / args.timeframe / "strategy_pipeline_metrics.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            df["seed"] = seed
            df["symbol"] = symbol
            rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_out = base_out / "seed_sweep_all_rows.csv"
    all_df.to_csv(all_out, index=False)

    metric_cols = [
        c
        for c in ["train", "CV", "corr", "return%", "Sharpe", "DD%", "trades"]
        if c in all_df.columns
    ]
    gb = all_df.groupby(["symbol", "strategy", "task"], dropna=False)
    summary = gb[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()

    # flatten columns
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary_out = base_out / "seed_sweep_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("\n=== Seed sweep summary (mean/std/min/max) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 200):
        print(summary.to_string(index=False))
    # Winner tables (by Sharpe_mean) per symbol for quick reading.
    try:
        sharpe_col = "Sharpe_mean"
        if sharpe_col in summary.columns:
            winners = (
                summary.sort_values(["symbol", sharpe_col], ascending=[True, False])
                .groupby("symbol", as_index=False)
                .head(1)
            )
        else:
            winners = pd.DataFrame()
    except Exception:
        winners = pd.DataFrame()

    # Simple HTML report
    html_out = base_out / "seed_sweep_report.html"
    try:
        parts = []
        parts.append(
            "<html><head><meta charset='utf-8'><title>Seed Sweep Report</title>"
        )
        parts.append(
            "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} th{background:#f5f5f5}</style>"
        )
        parts.append("</head><body>")
        parts.append("<h2>Seed Sweep Report</h2>")
        parts.append(f"<p><b>Symbols</b>: {', '.join(symbols)}<br/>")
        parts.append(f"<b>Seeds</b>: {', '.join(map(str,seeds))}<br/>")
        parts.append(
            f"<b>Timeframe</b>: {args.timeframe} &nbsp; <b>Window</b>: {args.start_date} → {args.end_date} &nbsp; <b>Test-size</b>: {args.test_size}</p>"
        )
        if not winners.empty:
            parts.append("<h3>Best per symbol (by Sharpe_mean)</h3>")
            parts.append(winners.to_html(index=False, escape=False))
        parts.append("<h3>Full summary (mean/std/min/max)</h3>")
        parts.append(summary.to_html(index=False, escape=False))
        parts.append("<h3>All rows</h3>")
        parts.append(all_df.to_html(index=False, escape=False))
        parts.append("</body></html>")
        html_out.write_text("\n".join(parts), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Failed to write HTML report: {e}")

    print(f"\nSaved:\n- {all_out}\n- {summary_out}\n- {html_out}\n")


if __name__ == "__main__":
    # Keep stdout order stable.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
