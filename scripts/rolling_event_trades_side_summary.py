#!/usr/bin/env python3
"""
Summarise per-month event backtest trades under a rolling_sim run directory.

For run_id=20260424_191639 style trees:
  results/.../tpc/.../_rolling_sim/<run_id>/fast_month_<YYYY-MM>/tpc/event_trades_<strategy>.csv

Output: table with rows per month, n_trades, n_long, n_short (from ``side`` column), sum pnl_r.

Investigation when a month has **zero trades** and "no short" in recent windows:
- Not necessarily a bug: TPC is often long-heavy; a few shorts may still appear in early months
  (see long/short counts in this table).
- For 0 rows in 2026 etc., open the sibling ``fast_month_.../tpc/pipeline.log`` and look for
  event_backtest lines such as ``reject_gate_deny``, ``reject_no_direction``,
  ``reject_prefilter_deny`` and per-symbol ``N trades`` — the pipeline can reject all
  intrabar events before any fill, so you get neither long nor short.

Usage:
  python scripts/rolling_event_trades_side_summary.py --run-root \\
    results/tpc/turbo-rolling-sim/_rolling_sim/20260424_191639 --strategy tpc
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Rolling sim root, e.g. results/.../tpc/.../.../_rolling_sim/<run_id>/",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="tpc",
        help="Strategy short name (event_trades_{strategy}.csv)",
    )
    args = p.parse_args()
    run_root: Path = args.run_root
    st = str(args.strategy).strip() or "tpc"
    if not run_root.is_dir():
        print(f"not a directory: {run_root}", file=sys.stderr)
        return 1

    month_re = re.compile(r"^fast_month_(\d{4}-\d{2})$")
    months: list[tuple[str, Path]] = []
    for d in sorted(run_root.iterdir(), key=lambda x: x.name):
        if not d.is_dir():
            continue
        m = month_re.match(d.name)
        if not m:
            continue
        months.append((m.group(1), d))

    try:
        import pandas as pd
    except ImportError:
        print("need pandas", file=sys.stderr)
        return 1

    print(
        f"run_root={run_root}\n"
        f"{'month':<9} {'rows':>5} {'long':>5} {'short':>5} "
        f"{'pnl_r sum':>12}  (side from CSV column: LONG/SHORT)"
    )
    print("-" * 64)
    tot_r = 0.0
    tot_n = 0
    long_all = 0
    short_all = 0
    for mtag, mdir in months:
        path = mdir / st / f"event_trades_{st}.csv"
        if not path.is_file():
            print(f"{mtag:<9} {'(missing)':>5} {'-':>5} {'-':>5} {'-':>12}")
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            print(f"{mtag:<9}     0     0     0        0.00  (empty file)")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"{mtag:<9} parse error: {exc}")
            continue
        if df.empty:
            print(f"{mtag:<9}     0     0     0        0.00")
            continue
        n = len(df)
        s = (
            df["side"].astype(str).str.upper()
            if "side" in df.columns
            else pd.Series([""] * n)
        )
        n_long = int(s.str.contains("LONG", na=False).sum())
        n_short = int(s.str.contains("SHORT", na=False).sum())
        pnl = 0.0
        if "pnl_r" in df.columns:
            pnl = float(pd.to_numeric(df["pnl_r"], errors="coerce").fillna(0.0).sum())
        tot_r += pnl
        tot_n += n
        long_all += n_long
        short_all += n_short
        print(f"{mtag:<9} {n:5d} {n_long:5d} {n_short:5d} {pnl:12.2f}")
    print("-" * 64)
    print(
        f"{'ALL':<9} {tot_n:5d} {long_all:5d} {short_all:5d} {tot_r:12.4f}  "
        f"(stitched_total_r in stitched_summary.json should match when PCM fallback uses these CSVs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
