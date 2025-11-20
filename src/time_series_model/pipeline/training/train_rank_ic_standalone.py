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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from data_tools.data_loader import MarketDataLoader
from data_tools.alpha_factors.alpha101_timeseries_adapted import (
    compute_adapted_alpha101_factors,
)
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


MINIMAL_SAFE_FEATURES: Set[str] = {
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_20",
    "sma_ratio_5_20",
    "sma_ratio_10_20",
}


def _safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean().shift(1)


def _safe_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean().shift(1)


def build_minimal_safe_features(
    df: pd.DataFrame, selected_features: Set[str]
) -> Tuple[pd.DataFrame, Set[str]]:
    """Build strictly causal features for the minimal safe set."""

    df_features = pd.DataFrame(index=df.index)

    if "sma_5" in selected_features:
        df_features["sma_5"] = _safe_sma(df["close"], 5)
    if "sma_10" in selected_features:
        df_features["sma_10"] = _safe_sma(df["close"], 10)
    if "sma_20" in selected_features:
        df_features["sma_20"] = _safe_sma(df["close"], 20)
    if "ema_20" in selected_features:
        df_features["ema_20"] = _safe_ema(df["close"], 20)

    if "sma_ratio_5_20" in selected_features:
        sma5 = df_features.get("sma_5", _safe_sma(df["close"], 5))
        sma20 = df_features.get("sma_20", _safe_sma(df["close"], 20))
        df_features["sma_ratio_5_20"] = sma5 / sma20.replace(0, np.nan)

    if "sma_ratio_10_20" in selected_features:
        sma10 = df_features.get("sma_10", _safe_sma(df["close"], 10))
        sma20 = df_features.get("sma_20", _safe_sma(df["close"], 20))
        df_features["sma_ratio_10_20"] = sma10 / sma20.replace(0, np.nan)

    alpha_needed = {f for f in selected_features if f.startswith("alpha101_")}
    if alpha_needed:
        alpha_source = df[["open", "high", "low", "close", "volume"]]
        alpha_df = compute_adapted_alpha101_factors(
            alpha_source,
            use_ts_rank=True,
            alpha001_window=5,
            alpha022_corr_window=10,
            alpha022_delta_window=5,
            alpha022_vol_window=20,
            alpha043_vol_rank_window=20,
            alpha043_mom_rank_window=8,
            alpha043_adv_window=20,
            alpha043_mom_period=7,
        ).reindex(df.index)
        alpha_columns = [c for c in alpha_needed if c in alpha_df.columns]
        df_features = df_features.join(alpha_df[alpha_columns], how="left")

    available = set(df_features.columns) & selected_features
    df_features = df_features.loc[:, sorted(available)]
    return df_features, available


def load_data(
    data_path: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "15T",
    feature_type: str = "comprehensive",
    top_factors: Optional[str] = None,
    engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, List[str], Optional[ComprehensiveFeatureEngineer]]:
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
    selected_features = None
    if top_factors:
        print(f"📋 Loading top factors from {top_factors}...")
        try:
            top_factors_list = load_top_factors_list(top_factors)
            selected_features = set(top_factors_list)
            print(
                f"   ✅ Loaded {len(selected_features)} features from top_factors.json"
            )
            print(f"   📊 Will only generate these features (others will be skipped)")
        except Exception as e:
            print(f"   ⚠️  Failed to load top factors: {e}")
            print(f"   ⚠️  Will generate all features for {feature_type}")

    use_minimal_pipeline = bool(
        selected_features and selected_features.issubset(MINIMAL_SAFE_FEATURES)
    )

    if use_minimal_pipeline:
        print("🧪 Using minimal safe feature builder (alpha101 + simple MAs only)...")
        df_features, available = build_minimal_safe_features(df, selected_features)
        if len(available) < len(selected_features):
            missing = sorted(selected_features - available)
            print(f"   ⚠️ Missing {len(missing)} requested features:")
            for name in missing[:10]:
                print(f"      - {name}")
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")
        # For minimal pipeline, engineer is None (no state to save)
        return_engineer = None
    else:
        if engineer is None:
            print(f"🔧 Engineering features ({feature_type})...")
            engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
            df_features = engineer.engineer_all_features(
                df, fit=fit, required_features=None
            )
        else:
            print(
                f"🔧 Transforming features using pre-fitted engineer ({feature_type})..."
            )
            df_features = engineer.engineer_all_features(
                df, fit=False, required_features=None
            )
        return_engineer = engineer

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

    # If selected_features is specified, only keep those features
    if selected_features is not None:
        feature_cols = [
            col for col in all_potential_features if col in selected_features
        ]
        print(
            f"   ✅ Generated {len(all_potential_features)} features, filtered to {len(feature_cols)} features from top_factors.json"
        )
        if len(feature_cols) < len(selected_features):
            missing = selected_features - set(feature_cols)
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

    return df_features, feature_cols, return_engineer


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
        "--tscv-gap",
        type=int,
        default=0,
        help="Gap (in samples) between training and validation folds to reduce leakage",
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
    parser.add_argument(
        "--signal-method",
        type=str,
        default="quantile",
        choices=["quantile", "sign", "hybrid", "optimized"],
        help="Signal generation method: 'quantile' (current), 'sign' (use prediction sign), 'hybrid' (combine sign and quantile), 'optimized' (optimize threshold)",
    )
    parser.add_argument(
        "--calibrate-predictions",
        action="store_true",
        default=False,
        help="Calibrate predictions to match true return distribution",
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

    # Load raw data first (without feature engineering to prevent leakage)
    print("📊 Loading raw data...")
    loader = MarketDataLoader(args.data_path)
    symbol_list = [s.strip() for s in args.symbol.split(",") if s.strip()]
    all_dfs = []

    for sym in symbol_list:
        df_single = loader.load_data(
            symbol=sym, start_date=args.train_start, end_date=args.train_end
        )
        if df_single is not None and not df_single.empty:
            if isinstance(df_single.index, pd.DatetimeIndex):
                agg_dict = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
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
                        agg_dict[col] = "last"
                for col in df_single.columns:
                    if col not in agg_dict and pd.api.types.is_numeric_dtype(
                        df_single[col]
                    ):
                        agg_dict[col] = "last"
                df_single = df_single.resample(args.timeframe).agg(agg_dict).dropna()
            if df_single is not None and not df_single.empty:
                df_single["_symbol"] = sym
                all_dfs.append(df_single)

    if not all_dfs:
        raise ValueError(f"No data found for symbol(s): {args.symbol}")

    df_raw = pd.concat(all_dfs, axis=0).sort_index()
    print(f"   ✅ Loaded {len(df_raw)} raw samples from {len(symbol_list)} asset(s)")

    # Split raw data FIRST (before feature engineering to prevent leakage)
    print("\n✂️  Splitting raw data (before feature engineering to prevent leakage)...")
    df_raw_train, df_raw_test = split_train_test(df_raw, test_size=args.test_size)
    print(
        f"   📊 Train: {len(df_raw_train)} samples ({len(df_raw_train)/len(df_raw)*100:.1f}%)"
    )
    print(
        f"   📊 Test:  {len(df_raw_test)} samples ({len(df_raw_test)/len(df_raw)*100:.1f}%)"
    )

    # Load top factors if specified
    selected_features = None
    if args.top_factors:
        print(f"📋 Loading top factors from {args.top_factors}...")
        try:
            top_factors_list = load_top_factors_list(args.top_factors)
            selected_features = set(top_factors_list)
            print(
                f"   ✅ Loaded {len(selected_features)} features from top_factors.json"
            )
        except Exception as e:
            print(f"   ⚠️  Failed to load top factors: {e}")

    # Feature engineering: fit on train, transform on test
    print(
        "\n🔧 Engineering features (fit on train, transform on test to prevent leakage)..."
    )

    # Fit on training set
    engineer = ComprehensiveFeatureEngineer(feature_types=args.feature_type)
    df_train_features = engineer.engineer_all_features(
        df_raw_train, fit=True, required_features=selected_features
    )
    print(
        f"   ✅ Fitted features on training set: {len(df_train_features.columns)} columns"
    )
    print(
        f"   📊 Training set columns after feature engineering: {list(df_train_features.columns)[:40]}"
    )

    # Check which selected features were actually generated
    if selected_features:
        generated_set = set(df_train_features.columns)
        matched = selected_features & generated_set
        missing = selected_features - generated_set
        print(
            f"   📊 Feature matching: {len(matched)}/{len(selected_features)} selected features found"
        )
        if len(missing) > 0:
            print(
                f"   ⚠️  Missing {len(missing)} features (first 10): {list(missing)[:10]}"
            )

    # Transform on test set (using fitted engineer from train)
    print("   🔄 Transforming test set features using fitted engineer...")
    df_test_features = engineer.engineer_all_features(
        df_raw_test, fit=False, required_features=selected_features
    )
    print(
        f"   ✅ Transformed features on test set: {len(df_test_features.columns)} columns"
    )

    # Get feature columns (same logic as before)
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
    all_potential_features = [
        col
        for col in df_train_features.columns
        if (col not in exclude_exact)
        and (not any(col.startswith(pfx) for pfx in exclude_prefixes))
        and col != "_symbol"
    ]

    print(
        f"   📊 Found {len(all_potential_features)} potential features in training set"
    )
    print(f"   📊 Training set columns: {list(df_train_features.columns)[:30]}")

    if selected_features is not None:
        print(f"   📊 Selected features from top_factors: {len(selected_features)}")
        print(f"   📊 First 10 selected: {list(selected_features)[:10]}")

        # Match selected features with actual generated features
        feature_cols = [
            col for col in all_potential_features if col in selected_features
        ]

        if len(feature_cols) < len(selected_features):
            missing = selected_features - set(feature_cols)
            print(
                f"   ⚠️  Warning: {len(missing)} selected features not found in generated features:"
            )
            for feat in list(missing)[:20]:
                print(f"      - {feat}")
            if len(missing) > 20:
                print(f"      ... and {len(missing) - 20} more")

            # Show what was actually generated
            print(f"   📊 Actually generated features (first 30):")
            for i, feat in enumerate(all_potential_features[:30]):
                print(f"      {i+1}. {feat}")
    else:
        feature_cols = all_potential_features

    print(f"   ✅ Final feature columns: {len(feature_cols)}")
    if len(feature_cols) > 0:
        print(f"   📊 First 10 final features: {feature_cols[:10]}")

    # Merge train and test
    df_features = pd.concat([df_train_features, df_test_features]).sort_index()

    # Verify feature columns exist after merge
    available_feature_cols = [col for col in feature_cols if col in df_features.columns]
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(f"   ⚠️  Warning: {len(missing)} feature columns missing after merge:")
        for col in list(missing)[:10]:
            print(f"      - {col}")
        feature_cols = available_feature_cols

    if len(feature_cols) == 0:
        raise ValueError(
            f"No valid feature columns after merge! "
            f"Train columns: {len(df_train_features.columns)}, "
            f"Test columns: {len(df_test_features.columns)}, "
            f"Available: {list(df_features.columns)[:30]}..."
        )

    print(f"   ✅ Feature engineering complete: {len(feature_cols)} features")

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

    # Verify feature columns still exist after label preparation
    available_feature_cols_after = [
        col for col in feature_cols if col in df_with_labels.columns
    ]
    if len(available_feature_cols_after) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols_after)
        print(
            f"   ⚠️  Warning: {len(missing)} feature columns missing after label preparation:"
        )
        for col in list(missing)[:10]:
            print(f"      - {col}")
        feature_cols = available_feature_cols_after

    if len(feature_cols) == 0:
        raise ValueError(
            f"No valid feature columns after label preparation! "
            f"Available columns: {list(df_with_labels.columns)[:30]}..."
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
                # Ensure at least 150 samples per feature for tree models
                # If using 20 features, need at least 3000 samples
                # If using 100 features, need at least 15000 samples
                "n_features": min(100, len(feature_cols)),  # More features to test
                "n_samples": max(
                    4000, min(100, len(feature_cols)) * 150
                ),  # At least 150 samples/feature
                "hold_period": args.horizon,
                "n_splits": 5,  # Use more splits for better statistical stability
                "threshold": 0.03,  # Stricter threshold for leakage detection
            },
            correlation_params={
                "correlation_threshold": 0.1,
                "min_samples": 100,
            },
        )

    # Split train/test (now on labeled data)
    print("\n✂️  Splitting labeled data...")
    df_train, df_test = split_train_test(df_with_labels, test_size=args.test_size)

    # Check feature columns in train/test sets
    train_feature_cols = [col for col in feature_cols if col in df_train.columns]
    test_feature_cols = [col for col in feature_cols if col in df_test.columns]
    missing_in_train = set(feature_cols) - set(train_feature_cols)
    missing_in_test = set(feature_cols) - set(test_feature_cols)

    print(f"   📊 Feature columns check after split:")
    print(f"      Requested: {len(feature_cols)} features")
    print(f"      In train set: {len(train_feature_cols)} features")
    print(f"      In test set: {len(test_feature_cols)} features")
    if missing_in_train:
        print(f"      ⚠️  Missing in train: {len(missing_in_train)} features")
        for col in list(missing_in_train)[:10]:
            print(f"         - {col}")
    if missing_in_test:
        print(f"      ⚠️  Missing in test: {len(missing_in_test)} features")
        for col in list(missing_in_test)[:10]:
            print(f"         - {col}")

    # Train with TSCV
    print("\n🌲 Training Rank IC model with Time Series Cross-Validation...")
    models, avg_rank_ic_cv, cv_results, trained_feature_cols = train_rank_ic_model(
        df_train,
        feature_cols=feature_cols,
        target_col="volatility_normalized_target",
        date_col=date_col,
        n_splits=args.n_splits,
        tscv_gap=args.tscv_gap,
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
        feature_cols=trained_feature_cols,
        confidence_threshold=0.85,
        asset_col=asset_col,
        signal_method=args.signal_method,
        calibrate_predictions=args.calibrate_predictions,
    )

    # Evaluate on test set
    print("\n📈 Evaluating on OOS test set...")
    test_eval = evaluate_model_performance(
        df_test_signals,
        signals=df_test_signals["signal"],
        confidence_threshold=0.85,
        hold_period=args.horizon,  # FIXED: Prevent overlapping trades
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
            "n_features": len(trained_feature_cols),
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
