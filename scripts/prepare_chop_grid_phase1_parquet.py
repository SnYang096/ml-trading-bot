#!/usr/bin/env python3
"""Prepare chop_grid Phase 1 parquet: merge feature_store + compute forward_rr.

Reads feature_store/features_chop_grid_120T_*/<SYM>/120T/*.parquet,
computes forward_rr = (mfe - mae) / atr over 50-bar horizon, and writes
a single merged parquet for IC/label scan.

Usage:
    python scripts/prepare_chop_grid_phase1_parquet.py \
        --feature-store-layer features_chop_grid_120T_c5a8c96e46 \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
        --horizon 50 \
        --output results/rd_loop/chop_grid_prefilter_fix_phase1/features_with_fwd_rr.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_STORE_ROOT = PROJECT_ROOT / "feature_store"


def compute_forward_rr(
    df: pd.DataFrame,
    horizon: int = 50,
    direction: str = "long",
    clip_range: tuple = (-5, 5),
) -> pd.DataFrame:
    """Compute forward_rr = (mfe - mae) / atr for a single symbol DataFrame."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    n = len(df)
    EPS = 1e-10

    forward_rr = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price
            mae = entry_price - future_low
        else:  # bidirectional: use max(rr_long, rr_short) — captures best direction
            mfe_long = future_high - entry_price
            mae_long = entry_price - future_low
            mfe_short = entry_price - future_low
            mae_short = future_high - entry_price
            rr_long = (mfe_long - mae_long) / max(current_atr, EPS)
            rr_short = (mfe_short - mae_short) / max(current_atr, EPS)
            forward_rr[i] = max(rr_long, rr_short)
            continue

        forward_rr[i] = (mfe - mae) / max(current_atr, EPS)

    df = df.copy()
    df["forward_rr"] = forward_rr
    df["forward_rr_clipped"] = np.clip(forward_rr, clip_range[0], clip_range[1])
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-store-layer",
        default="features_chop_grid_120T_c5a8c96e46",
        help="Feature store layer directory name",
    )
    p.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        help="Comma-separated symbols",
    )
    p.add_argument("--timeframe", default="120T")
    p.add_argument("--horizon", type=int, default=50, help="Forward lookback bars")
    p.add_argument("--direction", default="bidir", choices=["long", "short", "bidir"])
    p.add_argument(
        "--output",
        default="results/rd_loop/chop_grid_prefilter_fix_phase1/features_with_fwd_rr.parquet",
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    layer_dir = FEATURE_STORE_ROOT / args.feature_store_layer

    if not layer_dir.exists():
        print(f"ERROR: feature store layer not found: {layer_dir}")
        sys.exit(1)

    frames = []
    for sym in symbols:
        sym_dir = layer_dir / sym / args.timeframe
        if not sym_dir.exists():
            print(f"WARNING: {sym} not found in {layer_dir.name}, skipping")
            continue

        pq_files = sorted(sym_dir.glob("*.parquet"))
        if not pq_files:
            print(f"WARNING: no parquet files for {sym}, skipping")
            continue

        df = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=False)
        df = df.sort_index()

        if args.start_date:
            df = df[df.index >= args.start_date]
        if args.end_date:
            df = df[df.index <= args.end_date]

        # Compute forward_rr
        direction = args.direction if args.direction != "bidir" else "bidir"
        df = compute_forward_rr(df, horizon=args.horizon, direction=direction)

        if "_symbol" not in df.columns:
            df["_symbol"] = sym

        frames.append(df)
        print(
            f"  ✅ {sym}: {len(df)} bars, forward_rr non-null: {df['forward_rr'].notna().sum()}"
        )

    if not frames:
        print("ERROR: no data loaded")
        sys.exit(1)

    merged = pd.concat(frames)
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(f"\n✅ Wrote {out_path} ({len(merged)} rows, {len(merged.columns)} cols)")

    # Summary
    fwd = merged["forward_rr"]
    print(
        f"   forward_rr: mean={fwd.mean():.4f} std={fwd.std():.4f} "
        f"pos_rate={(fwd > 0).mean():.4f} non-null={fwd.notna().sum()}"
    )


if __name__ == "__main__":
    main()
