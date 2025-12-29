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
        --strategy-config config/strategies/sr_reversal_long \\
        --data-path data/parquet_data \\
        --output-dir feature_store/sr_reversal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly without installing the project package.
# (So `import src.*` works when executed from the repo root.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_handler import DataHandler
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
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
        default="config/strategies/sr_reversal_long",
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
    parser.add_argument(
        "--layer",
        type=str,
        default="heavy_v1",
        help="FeatureStore layer name (e.g. base_v1, heavy_v1)",
    )
    parser.add_argument(
        "--warmup-months",
        type=int,
        default=0,
        help="Warmup calendar months to prepend when computing each month, then trim before writing. "
        "Helps stateful/ticks/rolling features at month boundaries.",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=0,
        help="Optional warmup by bars (fallback). If warmup-months > 0, bars is ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy_cfg_path = Path(args.strategy_config)
    cfg_loader = StrategyConfigLoader(strategy_cfg_path)
    strategy_config = cfg_loader.load()

    store = FeatureStore(Path(args.output_dir))

    # Initialize DataHandler for unified data loading
    data_handler = DataHandler(data_path=args.data_path)

    print(f"📂 Loading raw data for {args.symbol}, timeframe={args.timeframe}")
    df_raw = data_handler.load_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if df_raw.empty:
        raise ValueError("Loaded empty raw DataFrame; check data-path/symbol/timeframe")

    feature_loader = StrategyFeatureLoader()

    df_raw = df_raw.sort_index()
    # Incremental by month: only compute features for months not yet materialized.
    monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
    requested = strategy_config.features.requested_features

    print(
        f"▶️ Incremental feature generation for strategy={strategy_config.name}, "
        f"requested_features={len(requested)}"
    )

    spec = FeatureStoreSpec(
        layer=args.layer, symbol=args.symbol, timeframe=args.timeframe
    )

    base_cols = ["open", "high", "low", "close", "volume", "_symbol"]

    for period, df_month in monthly_groups:
        if df_month.empty:
            continue
        month_str = period.strftime("%Y-%m")
        if store.has_month(spec, month_str):
            print(f"   ✅ Skip {month_str}: already in store")
            continue

        print(
            f"\n   🚀 Computing features for {month_str}: rows={len(df_month)}, "
            f"symbol={args.symbol}"
        )
        month_start = df_month.index.min()
        month_end = df_month.index.max()
        warmup_months = max(0, int(args.warmup_months))
        warmup_bars = max(0, int(args.warmup_bars))
        if warmup_months > 0:
            start_ts = pd.Timestamp(month_start) - pd.DateOffset(months=warmup_months)
            df_window = df_raw.loc[
                (df_raw.index >= start_ts) & (df_raw.index <= month_end)
            ]
        elif warmup_bars > 0:
            pos_end = df_raw.index.searchsorted(month_start, side="left")
            pos_start = max(0, pos_end - warmup_bars)
            df_window = df_raw.iloc[pos_start:].loc[:month_end]
        else:
            df_window = df_raw.loc[
                (df_raw.index >= month_start) & (df_raw.index <= month_end)
            ]

        df_feats_window = feature_loader.load_features_from_requested(
            df_window, requested_features=requested, fit=True
        )
        df_feats = df_feats_window.loc[
            (df_feats_window.index >= month_start)
            & (df_feats_window.index <= month_end)
        ]
        # Keep computed columns; store can also trim columns if needed.
        print(
            f"   ✅ Features for {month_str} computed: rows={len(df_feats)}, "
            f"cols={len(df_feats.columns)}"
        )
        store.write_month(
            spec,
            month_str,
            df_feats,
            base_columns=base_cols,
            feature_columns=None,
            overwrite=False,
            metadata={
                "requested_features": requested,
                "warmup_months": warmup_months,
                "warmup_bars": warmup_bars,
            },
        )
        print(
            f"   💾 Saved features to store: {spec.layer}/{spec.symbol}/{spec.timeframe}/{month_str}"
        )

    print("\n🎉 Incremental feature generation completed.")


if __name__ == "__main__":
    main()
