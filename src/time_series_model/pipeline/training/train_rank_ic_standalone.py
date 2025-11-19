#!/usr/bin/env python3
"""
Standalone Rank IC regression training script.

This script implements a complete Rank IC-optimized training pipeline:
- Time Series Cross-Validation (TSCV)
- Out-of-Sample (OOS) testing
- Volatility-normalized targets
- Historical quantile labels
- Confidence-based signal generation

Usage:
    python -m time_series_model.pipeline.training.train_rank_ic_standalone \
        --data-path /data/parquet_data \
        --symbol ETHUSDT \
        --train-start 2024-01-01 \
        --train-end 2024-12-31 \
        --horizon 5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_tools.data_loader import MarketDataLoader
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from time_series_model.pipeline.dimensionality.utils import load_top_factors_list
from time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
    generate_ensemble_signals,
    evaluate_model_performance,
)
from time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic
from time_series_model.pipeline.training.data_leakage_detector import (
    detect_data_leakage,
)


def load_data(
    data_path: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "15T",
    feature_type: str = "comprehensive",
    top_factors: Optional[str] = None,
) -> pd.DataFrame:
    """Load and prepare market data with features."""
    print(f"📊 Loading data for {symbol}...")

    # Support multiple symbols
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]

    loader = MarketDataLoader(data_path)
    all_dfs = []

    for sym in symbol_list:
        symbol_loader = MarketDataLoader(data_path)
        df_single = symbol_loader.load_data(
            symbol=sym, start_date=start_date, end_date=end_date
        )

        if df_single is not None and not df_single.empty:
            # Resample
            if isinstance(df_single.index, pd.DatetimeIndex):
                # Define aggregation rules for different column types
                agg_dict = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }

                # Add order flow columns if they exist (use last value for resampling)
                order_flow_cols = [
                    "cvd",
                    "taker_buy_ratio",
                    "cvd_short",
                    "cvd_medium",
                    "cvd_long",
                    "cvd_change_1",
                    "cvd_change_5",
                    "cvd_change_20",
                    "cvd_normalized",
                    "buy_qty",
                    "sell_qty",
                    "delta",
                ]
                for col in order_flow_cols:
                    if col in df_single.columns:
                        agg_dict[col] = "last"  # Use last value for order flow metrics

                # Add other numeric columns (use last value as default)
                for col in df_single.columns:
                    if col not in agg_dict and pd.api.types.is_numeric_dtype(
                        df_single[col]
                    ):
                        agg_dict[col] = "last"

                df_single = df_single.resample(timeframe).agg(agg_dict).dropna()

            if df_single is not None and not df_single.empty:
                df_single["_symbol"] = sym
                all_dfs.append(df_single)

    if not all_dfs:
        raise ValueError(f"No data found for symbol(s): {symbol}")

    df = pd.concat(all_dfs, axis=0).sort_index()
    print(f"   ✅ Loaded {len(df)} samples from {len(symbol_list)} asset(s)")

    # Load top factors if specified
    required_features = None
    if top_factors:
        print(f"📋 Loading top factors from {top_factors}...")
        try:
            top_factors_list = load_top_factors_list(top_factors)
            required_features = set(top_factors_list)
            print(
                f"   ✅ Loaded {len(required_features)} features from top_factors.json"
            )
            print(f"   📊 Will only generate these features (others will be skipped)")
        except Exception as e:
            print(f"   ⚠️  Failed to load top factors: {e}")
            print(f"   ⚠️  Will generate all features for {feature_type}")

    # Feature engineering
    print(f"🔧 Engineering features ({feature_type})...")
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
    df_features = engineer.engineer_all_features(
        df, fit=True, required_features=required_features
    )

    # Keep close price for label preparation
    if "close" not in df_features.columns and "close" in df.columns:
        df_features["close"] = df["close"]

    # Filter out label columns and raw prices
    exclude_exact = {
        "timestamp",
        "open",
        "high",
        "low",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
    }
    exclude_prefixes = ("signal_", "binary_signal_", "future_return_")

    # First, get all potential feature columns
    all_potential_features = [
        col
        for col in df_features.columns
        if (col not in exclude_exact)
        and (not any(col.startswith(pfx) for pfx in exclude_prefixes))
        and col != "_symbol"  # Keep _symbol but don't include in features
    ]

    # If required_features is specified, only keep those features
    if required_features is not None:
        feature_cols = [
            col for col in all_potential_features if col in required_features
        ]
        print(
            f"   ✅ Generated {len(all_potential_features)} features, filtered to {len(feature_cols)} features from top_factors.json"
        )
        if len(feature_cols) < len(required_features):
            missing = required_features - set(feature_cols)
            print(
                f"   ⚠️  Warning: {len(missing)} features from top_factors.json were not generated:"
            )
            for feat in list(missing)[:10]:  # Show first 10 missing
                print(f"      - {feat}")
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")
    else:
        feature_cols = all_potential_features
        print(f"   ✅ Generated {len(feature_cols)} features")

    # Keep symbol column and close for multi-asset support and label prep
    keep_cols = [*feature_cols, "close"]
    if "_symbol" in df_features.columns:
        keep_cols.append("_symbol")

    df_features = df_features[keep_cols].copy()

    return df_features, feature_cols


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and OOS test sets."""
    # Sort by date if index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    n_total = len(df)
    split_idx = int(n_total * (1 - test_size))

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"   📊 Train: {len(df_train)} samples ({len(df_train)/n_total:.1%})")
    print(f"   📊 Test:  {len(df_test)} samples ({len(df_test)/n_total:.1%})")

    return df_train, df_test


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Rank IC regression training with TSCV and OOS testing"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/parquet_data",
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol(s), comma-separated for multi-asset (e.g., ETHUSDT or ETHUSDT,BTCUSDT)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Training start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Training end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon (number of periods)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Data timeframe (e.g., 15T, 1H)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="Feature engineering type (comprehensive, baseline, technical, default, enhanced, or comma-separated combination like 'baseline,technical')",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of time series CV folds",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="OOS test set size (fraction)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rank_ic_training",
        help="Output directory for results",
    )
    parser.add_argument(
        "--filter-high-confidence",
        action="store_true",
        help="Filter to high-confidence samples only for training",
    )
    parser.add_argument(
        "--min-trend-strength",
        type=float,
        default=1.0,
        help="Minimum trend strength for sample filtering",
    )
    parser.add_argument(
        "--smooth-target",
        action="store_true",
        help="Smooth target variable to reduce noise",
    )
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Run data leakage detection tests",
    )
    parser.add_argument(
        "--top-factors",
        type=str,
        default=None,
        help="Path to top_factors.json file to load specific features (e.g., from feature-eval)",
    )
    parser.add_argument(
        "--leakage-random-walk",
        action="store_true",
        default=True,
        help="Run random walk leakage test (default: True if --check-leakage)",
    )
    parser.add_argument(
        "--leakage-correlation",
        action="store_true",
        default=True,
        help="Run feature-future correlation test (default: True if --check-leakage)",
    )

    args = parser.parse_args()

    # Enable leakage checks if --check-leakage is set
    if args.check_leakage:
        args.leakage_random_walk = True
        args.leakage_correlation = True

    print("=" * 60)
    print("🚀 Rank IC Regression Training (Standalone)")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Horizon: {args.horizon}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Feature Type: {args.feature_type}")
    print(f"TSCV Folds: {args.n_splits}")
    print(f"OOS Test Size: {args.test_size:.1%}")
    print("=" * 60)

    # Load data
    df_features, feature_cols = load_data(
        args.data_path,
        args.symbol,
        args.train_start,
        args.train_end,
        args.timeframe,
        args.feature_type,
        args.top_factors,
    )

    # Prepare Rank IC labels
    print("\n📝 Preparing Rank IC labels...")
    asset_col = "_symbol" if "_symbol" in df_features.columns else None
    date_col = None
    if isinstance(df_features.index, pd.DatetimeIndex):
        date_col = "date"
        df_features["date"] = df_features.index

    df_with_labels = prepare_rank_ic_labels(
        df_features,
        price_col="close",
        asset_col=asset_col,
        date_col=date_col,
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
    )

    print(
        f"   ✅ Labels prepared: {df_with_labels['volatility_normalized_target'].notna().sum()} valid samples"
    )

    # Data leakage detection
    leakage_results_storage = None
    if args.check_leakage or args.leakage_random_walk or args.leakage_correlation:
        print("\n🔍 Running data leakage detection...")
        leakage_results_storage = detect_data_leakage(
            df=df_with_labels,
            feature_cols=feature_cols,
            future_return_col="future_return",
            run_random_walk_test=args.leakage_random_walk,
            run_correlation_test=args.leakage_correlation,
            random_walk_params={
                "n_samples": 2000,  # Use more samples for better statistical power
                "n_features": min(100, len(feature_cols)),  # More features to test
                "hold_period": args.horizon,
                "n_splits": 5,  # Use more splits for better statistical stability
                "threshold": 0.03,  # Stricter threshold for leakage detection
            },
            correlation_params={
                "correlation_threshold": 0.1,
                "min_samples": 100,
            },
        )

    # Split train/test
    print("\n✂️  Splitting data...")
    df_train, df_test = split_train_test(df_with_labels, test_size=args.test_size)

    # Train with TSCV
    print("\n🌲 Training Rank IC model with Time Series Cross-Validation...")
    models, avg_rank_ic_cv, cv_results = train_rank_ic_model(
        df_train,
        feature_cols=feature_cols,
        target_col="volatility_normalized_target",
        date_col=date_col,
        n_splits=args.n_splits,
        use_gpu=False,  # Set to True if GPU available
        filter_high_confidence=args.filter_high_confidence,
        min_trend_strength=args.min_trend_strength,
        smooth_target=args.smooth_target,
        hold_period=args.horizon,  # Pass horizon for adaptive anti-overfitting parameters
    )

    print(f"\n✅ TSCV Training Complete")
    print(f"   Average Rank IC: {avg_rank_ic_cv:.4f}")
    print(f"   CV Results:\n{cv_results}")

    # Generate signals on test set
    print("\n📊 Generating signals on OOS test set...")
    df_test_signals = generate_ensemble_signals(
        df_test,
        models=models,
        feature_cols=feature_cols,
        confidence_threshold=0.85,
        asset_col=asset_col,
    )

    # Evaluate on test set
    print("\n📈 Evaluating on OOS test set...")
    test_eval = evaluate_model_performance(
        df_test_signals,
        signals=df_test_signals["signal"],
        confidence_threshold=0.85,
    )

    # Compute Rank IC on test set
    if "future_return" in df_test_signals.columns and "pred" in df_test_signals.columns:
        valid_mask = (
            df_test_signals["future_return"].notna() & df_test_signals["pred"].notna()
        )
        if valid_mask.sum() > 10:
            test_rank_ic = compute_rank_ic(
                df_test_signals.loc[valid_mask, "pred"],
                df_test_signals.loc[valid_mask, "future_return"],
            )
            print(f"\n📊 OOS Test Rank IC: {test_rank_ic:.4f}")
        else:
            test_rank_ic = None
            print(f"\n⚠️  Insufficient valid samples for OOS Rank IC")
    else:
        test_rank_ic = None

    # Save results
    print("\n💾 Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"rank_ic_results_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "config": {
            "symbol": args.symbol,
            "horizon": args.horizon,
            "timeframe": args.timeframe,
            "feature_type": args.feature_type,
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "filter_high_confidence": args.filter_high_confidence,
            "min_trend_strength": args.min_trend_strength,
            "smooth_target": args.smooth_target,
        },
        "data_info": {
            "total_samples": len(df_with_labels),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "n_features": len(feature_cols),
        },
        "cv_results": {
            "avg_rank_ic": float(avg_rank_ic_cv),
            "fold_results": cv_results.to_dict("records"),
        },
        "oos_results": {
            "rank_ic": float(test_rank_ic) if test_rank_ic is not None else None,
            "evaluation": test_eval,
        },
    }

    # Add leakage detection results if available
    if leakage_results_storage is not None:
        results["leakage_detection"] = leakage_results_storage

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"   ✅ Results saved to {results_file}")

    # Save models (optional)
    import joblib

    models_file = output_dir / f"rank_ic_models_{timestamp}.pkl"
    joblib.dump(models, models_file)
    print(f"   ✅ Models saved to {models_file}")

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"TSCV Average Rank IC: {avg_rank_ic_cv:.4f}")
    if test_rank_ic is not None:
        print(f"OOS Test Rank IC: {test_rank_ic:.4f}")
    print(f"Results: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
