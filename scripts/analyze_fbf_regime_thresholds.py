#!/usr/bin/env python3
"""Offline calibration for FBF prefilter/regime gates from features_labeled.parquet.

Replays production prefilter rules (defaults match config/strategies/fbf/archetypes/prefilter.yaml)
and prints quantiles + pass rates for trend/vol features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = [
    "fer_sr_failed_breakout_score",
    "sr_strength_max",
    "fer_ols_width_norm",
    "fer_ols_pos",
    "dist_to_nearest_sr",
    "rsi",
    "trend_r2_20",
    "bb_width_normalized_pct",
    "atr_percentile",
    "fer_sr_failed_breakout_direction_signed",
    "forward_rr",
]


def _prefilter_mask(
    df: pd.DataFrame, score_min: float, sr_min: float, ols_w_min: float
) -> pd.Series:
    return (
        (df["fer_sr_failed_breakout_score"] >= score_min)
        & (df["sr_strength_max"] >= sr_min)
        & (df["fer_ols_width_norm"] >= ols_w_min)
    )


def _qtable(s: pd.Series, qs: Iterable[float]) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series({f"p{int(q*100)}": np.nan for q in qs})
    return s.quantile(list(qs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "parquet",
        nargs="?",
        default="results/train_final_20260411_061030_rr_extreme/fbf/features_labeled.parquet",
        help="Path to features_labeled.parquet (fbf feature set)",
    )
    ap.add_argument("--score-min", type=float, default=0.30)
    ap.add_argument("--sr-min", type=float, default=0.45)
    ap.add_argument("--ols-width-min", type=float, default=0.18)
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.is_file():
        raise SystemExit(f"missing parquet: {path}")

    schema_names = set(pq.read_schema(path).names)
    read_cols = [c for c in COLS if c in schema_names]
    df = pd.read_parquet(path, columns=read_cols)

    n = len(df)
    m = _prefilter_mask(df, args.score_min, args.sr_min, args.ols_width_min)
    sub = df.loc[m]
    print(
        f"rows={n} prefilter_pass={m.sum()} ({100*m.mean():.2f}%) rules score>={args.score_min} sr>={args.sr_min} ols_w>={args.ols_width_min}"
    )

    qcols = [
        "trend_r2_20",
        "bb_width_normalized_pct",
        "atr_percentile",
        "fer_ols_pos",
        "dist_to_nearest_sr",
        "rsi",
    ]
    print("\n== Quantiles on prefilter-pass rows ==")
    for c in qcols:
        if c not in sub.columns:
            continue
        t = _qtable(sub[c], [0.1, 0.2, 0.25, 0.5, 0.7, 0.75, 0.85, 0.9])
        print(c, t.to_dict())

    # Regime gate pass rates (constants chosen from typical FBF: non-extreme trend, mid vol band)
    tr = sub["trend_r2_20"].dropna()
    bb = sub["bb_width_normalized_pct"].dropna()
    apct = sub["atr_percentile"].dropna()
    q70_tr = tr.quantile(0.70) if len(tr) else 0.55
    q20_bb, q85_bb = bb.quantile(0.20), bb.quantile(0.85) if len(bb) else (0.2, 0.85)
    q20_atr, q85_atr = apct.quantile(0.20), (
        apct.quantile(0.85) if len(apct) else (0.2, 0.85)
    )

    print("\n== Empirical quantiles for gate constants ==")
    print(f"trend_r2_20 p70 (suggested cap): {q70_tr:.4f}")
    print(f"bb_width_normalized_pct p20/p85 band: {q20_bb:.4f} .. {q85_bb:.4f}")
    print(f"atr_percentile p20/p85 band: {q20_atr:.4f} .. {q85_atr:.4f}")

    g1 = sub["trend_r2_20"] <= q70_tr
    print("\n== Pass rates within prefilter-pass (fraction of prefilter-pass rows) ==")
    print(f"trend_r2_20 <= p70 ({q70_tr:.4f}): {g1.mean():.4f}")

    bb_nonnull = sub["bb_width_normalized_pct"].notna()
    atr_nonnull = sub["atr_percentile"].notna()
    print(f"bb_width_normalized_pct non-null rate: {bb_nonnull.mean():.4f}")
    print(f"atr_percentile non-null rate: {atr_nonnull.mean():.4f}")
    if bb_nonnull.sum() > 30:
        g2 = (sub["bb_width_normalized_pct"] >= q20_bb) & (
            sub["bb_width_normalized_pct"] <= q85_bb
        )
        print(f"bb in [p20,p85] (among all prefilter rows): {g2.mean():.4f}")
        print(f"bb in [p20,p85] | non-null bb only: {g2.loc[bb_nonnull].mean():.4f}")
    else:
        print("bb band: skipped (too few non-null rows for stable calibration)")
    if atr_nonnull.sum() > 30:
        g3 = (sub["atr_percentile"] >= q20_atr) & (sub["atr_percentile"] <= q85_atr)
        print(f"atr_pct in [p20,p85] (among all prefilter rows): {g3.mean():.4f}")
        print(f"atr_pct in [p20,p85] | non-null only: {g3.loc[atr_nonnull].mean():.4f}")
    else:
        print("atr band: skipped (too few non-null rows for stable calibration)")
    if bb_nonnull.sum() > 30 and atr_nonnull.sum() > 30:
        g2 = (sub["bb_width_normalized_pct"] >= q20_bb) & (
            sub["bb_width_normalized_pct"] <= q85_bb
        )
        g3 = (sub["atr_percentile"] >= q20_atr) & (sub["atr_percentile"] <= q85_atr)
        combo = g1 & g2 & g3
        print(f"all three (note: NaN in bb/atr fails AND): {combo.mean():.4f}")

    if (
        "fer_sr_failed_breakout_direction_signed" in sub.columns
        and "forward_rr" in sub.columns
    ):
        for name, cond in [
            ("long_hint", sub["fer_sr_failed_breakout_direction_signed"] >= 1),
            ("short_hint", sub["fer_sr_failed_breakout_direction_signed"] <= -1),
        ]:
            chunk = sub.loc[cond]
            if len(chunk) < 50:
                continue
            print(f"\n-- {name} n={len(chunk)} --")
            print(
                "trend_r2_20 median",
                chunk["trend_r2_20"].median(),
                "bb_pct median",
                chunk["bb_width_normalized_pct"].median(),
            )


if __name__ == "__main__":
    main()
