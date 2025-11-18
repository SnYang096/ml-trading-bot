#!/usr/bin/env python3
"""
Generate a cross-sectional feature panel from raw OHLCV data.

Example:
    python scripts/cross_sectional/generate_panel.py \
        --symbols BTCUSDT ETHUSDT SOLUSDT \
        --timeframe 15T \
        --horizon 12 \
        --start-date 2024-11-01 \
        --end-date 2025-04-30 \
        --feature-type baseline \
        --output results/feature_exports/15T_baseline_12b.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_sectional import (
    PanelGenerationConfig,
    generate_cross_sectional_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multi-asset feature panels for cross-sectional modelling."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols (e.g., BTCUSDT ETHUSDT SOLUSDT).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Resample frequency for OHLCV data (default: 15T).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward return horizon in bars (default: 12).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional root data directory passed to MarketDataLoader.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--feature-type",
        choices=["baseline", "comprehensive"],
        default="baseline",
        help="Feature engineering recipe (default: baseline).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet path for the generated panel.",
    )
    parser.add_argument(
        "--no-dropna",
        action="store_true",
        help="Keep rows with NaNs instead of dropping them after generation.",
    )
    parser.add_argument(
        "--no-orderflow",
        action="store_true",
        help="Disable order-flow augmentation when reading raw agg-trade files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PanelGenerationConfig(
        symbols=args.symbols,
        timeframe=args.timeframe,
        horizon=args.horizon,
        data_path=args.data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=args.feature_type,
        dropna=not args.no_dropna,
        save_path=args.output,
        include_order_flow=not args.no_orderflow,
    )
    panel, target_col = generate_cross_sectional_panel(config)
    print(f"✅ Panel shape: {panel.shape}, target column: {target_col}")


if __name__ == "__main__":
    main()
