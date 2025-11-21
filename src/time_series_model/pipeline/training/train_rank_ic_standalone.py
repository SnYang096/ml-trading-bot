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
from data_tools.data_utils import load_raw_data
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from time_series_model.pipeline.dimensionality.utils import load_top_factors_list
from time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
    generate_ensemble_signals,
    evaluate_model_performance,
)
from time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic


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


def load_and_split_data(
    data_path: str,
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    timeframe: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load raw data and split into train/test sets."""
    print("📊 Loading raw data...")
    df_raw = load_raw_data(
        data_path=data_path,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
    )
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
    print(f"   ✅ Loaded {len(df_raw)} raw samples from {len(symbol_list)} asset(s)")

    print("\n✂️  Splitting raw data (before feature engineering to prevent leakage)...")
    df_raw_train, df_raw_test = split_train_test(df_raw, test_size=test_size)

    return df_raw_train, df_raw_test, symbol_list


def load_top_factors_if_specified(
    top_factors_path: Optional[str],
) -> Optional[set[str]]:
    """Load top factors from JSON file if specified."""
    if not top_factors_path:
        return None

    print(f"📋 Loading top factors from {top_factors_path}...")
    try:
        top_factors_list = load_top_factors_list(top_factors_path)
        selected_features = set(top_factors_list)
        print(f"   ✅ Loaded {len(selected_features)} features from top_factors.json")
        return selected_features
    except Exception as e:
        print(f"   ⚠️  Failed to load top factors: {e}")
        return None


def engineer_features(
    df_raw_train: pd.DataFrame,
    df_raw_test: pd.DataFrame,
    feature_type: str,
    selected_features: Optional[set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, ComprehensiveFeatureEngineer, List[str]]:
    """Engineer features: fit on train, transform on test."""
    print(
        "\n🔧 Engineering features (fit on train, transform on test to prevent leakage)..."
    )

    # Fit on training set
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
    df_train_features = engineer.engineer_all_features(
        df_raw_train, fit=True, required_features=selected_features
    )
    print(
        f"   ✅ Fitted features on training set: {len(df_train_features.columns)} columns"
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

    # Transform on test set
    print("   🔄 Transforming test set features using fitted engineer...")
    df_test_features = engineer.engineer_all_features(
        df_raw_test, fit=False, required_features=selected_features
    )
    print(
        f"   ✅ Transformed features on test set: {len(df_test_features.columns)} columns"
    )

    # Extract feature columns
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

    # Filter by selected features if specified
    if selected_features is not None:
        feature_cols = [
            col for col in all_potential_features if col in selected_features
        ]
        if len(feature_cols) < len(selected_features):
            missing = selected_features - set(feature_cols)
            print(f"   ⚠️  Warning: {len(missing)} selected features not found")
            for feat in list(missing)[:20]:
                print(f"      - {feat}")
    else:
        feature_cols = all_potential_features

    # Merge train and test
    df_features = pd.concat([df_train_features, df_test_features]).sort_index()

    # Verify feature columns exist
    available_feature_cols = [col for col in feature_cols if col in df_features.columns]
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(f"   ⚠️  Warning: {len(missing)} feature columns missing after merge")
        feature_cols = available_feature_cols

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns after feature engineering!")

    print(f"   ✅ Feature engineering complete: {len(feature_cols)} features")

    return df_features, engineer, feature_cols


def prepare_labels(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
) -> tuple[pd.DataFrame, List[str], Optional[str], Optional[str]]:
    """Prepare Rank IC labels and verify feature columns."""
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
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
    )

    # Verify feature columns still exist
    available_feature_cols = [
        col for col in feature_cols if col in df_with_labels.columns
    ]
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(
            f"   ⚠️  Warning: {len(missing)} feature columns missing after label preparation"
        )
        feature_cols = available_feature_cols

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns after label preparation!")

    print(
        f"   ✅ Labels prepared: {df_with_labels['volatility_normalized_target'].notna().sum()} valid samples"
    )

    return df_with_labels, feature_cols, asset_col, date_col


def train_model(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    date_col: Optional[str],
    args,
) -> tuple[List, float, pd.DataFrame, List[str]]:
    """Train Rank IC model with Time Series Cross-Validation."""
    print("\n🌲 Training Rank IC model with Time Series Cross-Validation...")

    models, avg_rank_ic_cv, cv_results, trained_feature_cols = train_rank_ic_model(
        df_train,
        feature_cols=feature_cols,
        target_col="volatility_normalized_target",
        date_col=date_col,
        n_splits=args.n_splits,
        tscv_gap=args.tscv_gap,
        use_gpu=False,
        filter_high_confidence=args.filter_high_confidence,
        min_trend_strength=args.min_trend_strength,
        smooth_target=args.smooth_target,
        hold_period=args.horizon,
    )

    print(f"\n✅ TSCV Training Complete")
    print(f"   Average Rank IC: {avg_rank_ic_cv:.4f}")

    return models, avg_rank_ic_cv, cv_results, trained_feature_cols


def evaluate_and_save_results(
    df_test: pd.DataFrame,
    models: List,
    trained_feature_cols: List[str],
    asset_col: Optional[str],
    args,
    avg_rank_ic_cv: float,
    cv_results: pd.DataFrame,
    df_with_labels: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test_final: pd.DataFrame,
) -> None:
    """Generate signals, evaluate model, and save results."""
    # Generate signals
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
        hold_period=args.horizon,
    )

    # Compute Rank IC on test set
    test_rank_ic = None
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
            print(f"\n⚠️  Insufficient valid samples for OOS Rank IC")

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
            "test_samples": len(df_test_final),
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

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    # Save models
    import joblib

    models_file = output_dir / f"rank_ic_models_{timestamp}.pkl"
    joblib.dump(models, models_file)
    print(f"   ✅ Models saved to {models_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"TSCV Average Rank IC: {avg_rank_ic_cv:.4f}")
    if test_rank_ic is not None:
        print(f"OOS Test Rank IC: {test_rank_ic:.4f}")
    print(f"Results: {results_file}")
    print("=" * 60)


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
        "--top-factors",
        type=str,
        default=None,
        help="Path to top_factors.json file to load specific features (e.g., from feature-eval)",
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

    # Print configuration
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

    # Step 1: Load and split raw data
    df_raw_train, df_raw_test, symbol_list = load_and_split_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.train_start,
        end_date=args.train_end,
        timeframe=args.timeframe,
        test_size=args.test_size,
    )

    # Step 2: Load top factors if specified
    selected_features = load_top_factors_if_specified(args.top_factors)

    # Step 3: Engineer features
    df_features, engineer, feature_cols = engineer_features(
        df_raw_train=df_raw_train,
        df_raw_test=df_raw_test,
        feature_type=args.feature_type,
        selected_features=selected_features,
    )

    # Step 4: Prepare labels
    df_with_labels, feature_cols, asset_col, date_col = prepare_labels(
        df_features=df_features,
        feature_cols=feature_cols,
        horizon=args.horizon,
    )

    # Step 5: Split labeled data
    print("\n✂️  Splitting labeled data...")
    df_train, df_test = split_train_test(df_with_labels, test_size=args.test_size)

    # Verify feature columns in train/test sets
    train_feature_cols = [col for col in feature_cols if col in df_train.columns]
    test_feature_cols = [col for col in feature_cols if col in df_test.columns]
    if len(train_feature_cols) != len(feature_cols) or len(test_feature_cols) != len(
        feature_cols
    ):
        print(f"   ⚠️  Feature column mismatch after split")
        print(
            f"      Requested: {len(feature_cols)}, Train: {len(train_feature_cols)}, Test: {len(test_feature_cols)}"
        )
        feature_cols = [
            col
            for col in feature_cols
            if col in train_feature_cols and col in test_feature_cols
        ]

    # Step 6: Train model
    models, avg_rank_ic_cv, cv_results, trained_feature_cols = train_model(
        df_train=df_train,
        feature_cols=feature_cols,
        date_col=date_col,
        args=args,
    )

    # Step 7: Evaluate and save results
    evaluate_and_save_results(
        df_test=df_test,
        models=models,
        trained_feature_cols=trained_feature_cols,
        asset_col=asset_col,
        args=args,
        avg_rank_ic_cv=avg_rank_ic_cv,
        cv_results=cv_results,
        df_with_labels=df_with_labels,
        df_train=df_train,
        df_test_final=df_test,
    )


if __name__ == "__main__":
    main()
