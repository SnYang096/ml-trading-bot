#!/usr/bin/env python3
"""
Incremental feature generation script for SR Reversal strategy.

Goal:
- Run the SAME feature pipeline as training, but materialize features to a
  compact monthly Parquet feature store, so research / backtest can reuse
  them without recomputing.

Usage:
    python scripts/run_feature_store_sr_reversal.py \\
        --symbol BTCUSDT \\
        --timeframe 15T \\
        --strategy-config config/strategies/sr_reversal \\
        --data-path data/parquet_data \\
        --output-dir feature_store/sr_reversal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental feature generator for SR Reversal strategy"
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Symbol, e.g. BTCUSDT"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Kline timeframe (must match training)",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="config/strategies/sr_reversal",
        help="Path to strategy config directory (with features.yaml)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/parquet_data",
        help="Root path of OHLCV parquet data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="feature_store/sr_reversal",
        help="Directory to store monthly feature Parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy_cfg_path = Path(args.strategy_config)
    cfg_loader = StrategyConfigLoader(strategy_cfg_path)
    strategy_config = cfg_loader.load()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading raw data for {args.symbol}, timeframe={args.timeframe}")
    df_raw = load_raw_data(
        data_path=args.data_path, symbol=args.symbol, timeframe=args.timeframe
    )
    if df_raw.empty:
        raise ValueError("Loaded empty raw DataFrame; check data-path/symbol/timeframe")

    # Ensure datetime index
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        for col in ("datetime", "timestamp", "date"):
            if col in df_raw.columns:
                df_raw.index = pd.to_datetime(df_raw[col])
                break
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            raise ValueError("Cannot find datetime-like index/column in raw data")

    df_raw = df_raw.sort_index()

    feature_loader = StrategyFeatureLoader()

    # Incremental by month: only compute features for months not yet materialized.
    monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
    requested = strategy_config.features.requested_features

    print(
        f"▶️ Incremental feature generation for strategy={strategy_config.name}, "
        f"requested_features={len(requested)}"
    )

    for period, df_month in monthly_groups:
        if df_month.empty:
            continue
        month_str = period.strftime("%Y-%m")
        out_file = out_root / f"{args.symbol}_{args.timeframe}_{month_str}.parquet"

        if out_file.exists():
            print(f"   ✅ Skip {month_str}: {out_file} already exists")
            continue

        print(
            f"\n   🚀 Computing features for {month_str}: rows={len(df_month)}, "
            f"symbol={args.symbol}"
        )
        # NOTE: fit=True for each month is fine here because the underlying
        # feature_loader/ParallelFeatureComputer already uses monthly cache.
        df_feats = feature_loader.load_features_from_requested(
            df_month,
            requested_features=requested,
            fit=True,
        )
        # Keep all computed columns; downstream scripts can select subsets.
        print(
            f"   ✅ Features for {month_str} computed: rows={len(df_feats)}, "
            f"cols={len(df_feats.columns)}"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_feats.to_parquet(out_file)
        print(f"   💾 Saved features to {out_file}")

    print("\n🎉 Incremental feature generation completed.")


if __name__ == "__main__":
    main()
