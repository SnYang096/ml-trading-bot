#!/usr/bin/env python3
"""
SRB 实验报告：汇总 stitched_summary + 多个月 event_trades CSV 的 exit_reason 分布。

用法:
  python scripts/srb_experiment_report.py --run-dir results/srb/.../_rolling_sim/<id>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="SRB rolling experiment summary")
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="rolling_sim 单次运行根目录（含 stitched_summary.json）",
    )
    args = ap.parse_args()
    root: Path = args.run_dir
    ss = root / "stitched_summary.json"
    if not ss.exists():
        raise SystemExit(f"missing {ss}")

    with open(ss, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("=== stitched_summary ===")
    for k in sorted(summary.keys()):
        if "stitch" in k.lower() or k in ("total_trades", "symbols", "months"):
            print(f"  {k}: {summary[k]}")

    csvs = sorted(root.glob("fast_month_*/srb/event_trades_srb.csv"))
    if not csvs:
        print("\n(no event_trades_srb.csv under fast_month_*/srb/)")
        return

    parts = [pd.read_csv(p) for p in csvs]
    df = pd.concat(parts, ignore_index=True)
    print(f"\n=== trades merged n={len(df)} ===")
    if "pnl_r" in df.columns:
        s = df["pnl_r"].astype(float)
        print(f"  total_r: {s.sum():.4f}")
        print(f"  mean_r: {s.mean():.4f}")
        print(f"  win_rate: {(s > 0).mean():.4f}")
    if "exit_reason" in df.columns:
        c = Counter(df["exit_reason"].astype(str))
        print("  exit_reason:")
        for k, v in c.most_common():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
