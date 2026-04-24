"""Summarize leverage capacity analysis across symbols.

Reads per-symbol feature_lift_*.csv outputs from analyze_leverage_capacity.py,
aggregates across symbols and produces a readable summary ranking features
by lift for the target bucket (default: ">=100x").

Also prints bucket share table grouped by (horizon, period, side).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_lifts(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(root.glob("*_120T/feature_lift_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_lift(lifts: pd.DataFrame) -> pd.DataFrame:
    # weight by sample count n so symbols with more data dominate less per-bucket
    g = (
        lifts.groupby(["horizon", "period", "side", "feature", "q"], dropna=False)
        .agg(
            n_total=("n", "sum"),
            hit_total=("hit", "sum"),
            feat_med_mean=("feat_med", "mean"),
            feat_lo=("feat_lo", "min"),
            feat_hi=("feat_hi", "max"),
        )
        .reset_index()
    )
    base = (
        lifts.groupby(["horizon", "period", "side"])
        .apply(
            lambda d: pd.Series(
                {
                    "base_rate": (
                        (d["hit"] * 1.0).sum() / d["n"].sum()
                        if d["n"].sum()
                        else np.nan
                    ),
                }
            )
        )
        .reset_index()
    )
    g = g.merge(base, on=["horizon", "period", "side"], how="left")
    g["rate"] = g["hit_total"] / g["n_total"]
    g["lift"] = g["rate"] / g["base_rate"]
    return g


def top_feature_quantiles(
    agg: pd.DataFrame,
    horizon: int,
    period: str,
    side: str,
    top_k: int = 15,
    min_n: int = 2000,
) -> pd.DataFrame:
    d = agg[
        (agg["horizon"] == horizon)
        & (agg["period"] == period)
        & (agg["side"] == side)
        & (agg["n_total"] >= min_n)
    ].copy()
    d = d.sort_values("lift", ascending=False)
    return d.head(top_k)[
        [
            "feature",
            "q",
            "n_total",
            "hit_total",
            "rate",
            "base_rate",
            "lift",
            "feat_med_mean",
            "feat_lo",
            "feat_hi",
        ]
    ]


def format_buckets(root: Path) -> pd.DataFrame:
    g = pd.read_csv(root / "global_bucket_counts_agg.csv")
    return g


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="reports/leverage_capacity_v1")
    p.add_argument("--horizons", default="12,48,120")
    p.add_argument(
        "--periods",
        default="all,bull_2020_2021,bull_2023_2024",
    )
    p.add_argument("--sides", default="long,short")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    root = Path(args.root)
    lifts = load_lifts(root)
    if lifts.empty:
        print("No lift csvs found")
        return
    agg = aggregate_lift(lifts)

    lines: List[str] = []
    lines.append(f"# Leverage capacity summary — {root}\n")

    # 1) bucket share table
    bk = format_buckets(root)
    lines.append("## 1. Bucket shares (cross-symbol aggregate)\n")
    for (h, per, side), gg in bk.groupby(["horizon", "period", "side"]):
        lines.append(f"### H={h} period={per} side={side}\n")
        lines.append(
            gg[
                ["bucket", "count", "share", "mae_p50", "mae_p90", "mfe_p50", "mfe_p90"]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")

    # 2) top feature quantiles by lift
    lines.append("\n## 2. Top feature quantiles predicting >=100x bucket\n")
    horizons = [int(x) for x in args.horizons.split(",")]
    periods = [s.strip() for s in args.periods.split(",")]
    sides = [s.strip() for s in args.sides.split(",")]
    for h in horizons:
        for per in periods:
            for side in sides:
                top = top_feature_quantiles(agg, h, per, side, top_k=args.top_k)
                if top.empty:
                    continue
                lines.append(f"### H={h} period={per} side={side}\n")
                lines.append(top.to_markdown(index=False, floatfmt=".4f"))
                lines.append("")

    out = "\n".join(lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
