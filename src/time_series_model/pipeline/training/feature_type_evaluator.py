"""
Feature Type Evaluator

This script evaluates different feature types for:
1. Data leakage detection
2. Rank IC (Information Coefficient) calculation
3. Feature usefulness assessment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from data_tools.data_loader import MarketDataLoader
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
)
from time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic
from time_series_model.pipeline.training.data_leakage_detector import (
    check_feature_future_correlation,
    test_random_walk_leakage,
)


def evaluate_feature_type(
    df: pd.DataFrame,
    feature_type: str,
    hold_period: int = 5,
    n_splits: int = 3,
    test_leakage: bool = True,
    leakage_threshold: float = 0.03,
) -> Dict:
    """
    Evaluate a specific feature type for IC and data leakage.

    Args:
        df: DataFrame with price data
        feature_type: Feature type to evaluate (e.g., 'baseline', 'enhanced', etc.)
        hold_period: Holding period for labels
        n_splits: Number of CV folds
        test_leakage: Whether to run leakage tests

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating Feature Type: {feature_type}")
    print(f"{'='*60}")

    results = {
        "feature_type": feature_type,
        "status": "unknown",
    }

    try:
        # Feature engineering
        print(f"Engineering {feature_type} features...")
        engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
        df_features = engineer.engineer_all_features(df.copy(), fit=True)

        # Get feature columns (exclude price/volume/label columns)
        exclude_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "datetime",
            "_symbol",
            "signal",
            "binary_signal",
            "future_return",
        }
        exclude_prefixes = ("signal_", "binary_signal_", "future_return_")

        feature_cols = [
            col
            for col in df_features.columns
            if col not in exclude_cols
            and not any(col.startswith(pfx) for pfx in exclude_prefixes)
            and pd.api.types.is_numeric_dtype(df_features[col])
            and df_features[col].notna().sum() > 10
        ]

        if len(feature_cols) == 0:
            results["status"] = "no_features"
            results["message"] = "No features generated"
            return results

        print(f"   ✅ Generated {len(feature_cols)} features")
        results["n_features"] = len(feature_cols)

        # Keep close price for label preparation
        if "close" not in df_features.columns and "close" in df.columns:
            df_features["close"] = df["close"]

        # Prepare labels
        print("Preparing labels...")
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
            hold_period=hold_period,
            lookback_window=60,
            ensure_volatility=True,
        )

        valid_samples = df_with_labels["volatility_normalized_target"].notna().sum()
        print(f"   ✅ {valid_samples} valid samples")

        if valid_samples < 100:
            results["status"] = "insufficient_data"
            results["valid_samples"] = valid_samples
            return results

        results["valid_samples"] = valid_samples

        # Check sample size and feature count
        n_samples = len(df_with_labels)
        n_features = len(feature_cols)
        samples_per_feature = n_samples / max(n_features, 1)

        print(f"Training model and calculating Rank IC...")
        print(
            f"   Samples: {n_samples}, Features: {n_features}, Ratio: {samples_per_feature:.1f} samples/feature"
        )

        # Warn if sample size is too small relative to features
        if samples_per_feature < 10:
            print(
                f"   ⚠️  Warning: Low samples/feature ratio ({samples_per_feature:.1f})"
            )
            print(f"      High risk of overfitting! Consider:")
            print(f"      - Using fewer features")
            print(f"      - Using stronger regularization")
            print(f"      - Collecting more data")

        # Pass hold_period to train_rank_ic_model for adaptive parameters
        models, avg_rank_ic, cv_results = train_rank_ic_model(
            df_with_labels,
            feature_cols=feature_cols,
            target_col="volatility_normalized_target",
            date_col=date_col,
            n_splits=n_splits,
            use_gpu=False,
            filter_high_confidence=False,
            min_trend_strength=0.0,
            smooth_target=False,
            weight_col=None,  # Don't use weights for evaluation
            hold_period=hold_period,  # Pass hold_period for adaptive parameters
        )

        results["avg_rank_ic"] = float(avg_rank_ic)
        results["rank_ic_std"] = (
            float(cv_results["rank_ic"].std()) if cv_results is not None else 0.0
        )
        results["cv_results"] = (
            cv_results.to_dict("records") if cv_results is not None else None
        )

        print(
            f"   ✅ Average Rank IC: {avg_rank_ic:.4f} ± {results['rank_ic_std']:.4f}"
        )

        # Data leakage tests
        if test_leakage:
            print("Running data leakage tests...")

            # Random walk test
            # For long horizons (e.g., 24 forwards = 4 days), use a more lenient threshold
            # Longer horizons may have slightly higher Rank IC on random data due to statistical noise
            adaptive_threshold = leakage_threshold
            if hold_period >= 20:  # Long horizon (e.g., 4 days for 240T)
                adaptive_threshold = max(
                    leakage_threshold, 0.04
                )  # More lenient for long horizons
                print(
                    f"   ℹ️  Long horizon detected (hold_period={hold_period}), using threshold={adaptive_threshold:.3f}"
                )

            leakage_test = test_random_walk_leakage(
                feature_cols=feature_cols,
                n_samples=2000,
                n_features=min(100, len(feature_cols)),
                hold_period=hold_period,
                n_splits=3,
                threshold=adaptive_threshold,
            )
            results["random_walk_test"] = leakage_test

            # Correlation test
            corr_test = check_feature_future_correlation(
                df=df_with_labels,
                feature_cols=feature_cols,
                future_return_col="future_return",
                correlation_threshold=0.1,
                min_samples=100,
            )
            results["correlation_test"] = corr_test

            # Overall leakage assessment
            has_leakage = leakage_test.get("has_leakage", False) or corr_test.get(
                "has_leakage", False
            )
            results["has_leakage"] = has_leakage

            if has_leakage:
                print(f"   ⚠️  Data leakage detected!")
            else:
                print(f"   ✅ No data leakage detected")

        results["status"] = "completed"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"   ❌ Error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate different feature types for IC and data leakage"
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
        default="BTCUSDT",
        help="Trading symbol",
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
        "--timeframe",
        type=str,
        default="240T",
        help="Data timeframe",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        default="baseline,default,enhanced,hurst,wavelet,hilbert,spectral,order_flow,alpha101",
        help="Comma-separated list of feature types to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--test-leakage",
        action="store_true",
        default=True,
        help="Run data leakage tests",
    )
    parser.add_argument(
        "--leakage-threshold",
        type=float,
        default=0.03,
        help="Threshold for data leakage detection (default: 0.03, use 0.04-0.05 for long horizons like 24 forwards)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 Feature Type Evaluator")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Horizon: {args.horizon}")
    print(f"Feature Types: {args.feature_types}")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    loader = MarketDataLoader(args.data_path)
    df = loader.load_data(
        symbol=args.symbol,
        start_date=args.train_start,
        end_date=args.train_end,
    )

    if df is None or df.empty:
        print("❌ No data loaded")
        return

    # Resample if needed
    if isinstance(df.index, pd.DatetimeIndex):
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Add order flow columns if they exist
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
            if col in df.columns:
                agg_dict[col] = "last"

        # Add other numeric columns
        for col in df.columns:
            if col not in agg_dict and pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "last"

        df = df.resample(args.timeframe).agg(agg_dict).dropna()

    print(f"   ✅ Loaded {len(df)} samples")

    # Evaluate each feature type
    feature_types = [ft.strip() for ft in args.feature_types.split(",")]
    all_results = {}

    for feat_type in feature_types:
        result = evaluate_feature_type(
            df.copy(),
            feature_type=feat_type,
            hold_period=args.horizon,
            n_splits=3,
            test_leakage=args.test_leakage,
            leakage_threshold=args.leakage_threshold,
        )
        all_results[feat_type] = result

    # Summary
    print("\n" + "=" * 60)
    print("📊 Feature Type Evaluation Summary")
    print("=" * 60)

    summary_data = []
    for feat_type, result in all_results.items():
        if result.get("status") == "completed":
            avg_ic = result.get("avg_rank_ic", 0.0)
            ic_std = result.get("rank_ic_std", 0.0)
            n_features = result.get("n_features", 0)
            has_leakage = result.get("has_leakage", False)

            summary_data.append(
                {
                    "feature_type": feat_type,
                    "n_features": n_features,
                    "avg_rank_ic": avg_ic,
                    "rank_ic_std": ic_std,
                    "has_leakage": has_leakage,
                }
            )

            leakage_status = "⚠️  LEAKAGE" if has_leakage else "✅ CLEAN"
            print(
                f"{feat_type:20s} | IC: {avg_ic:7.4f} ± {ic_std:.4f} | Features: {n_features:4d} | {leakage_status}"
            )

    # Sort by Rank IC
    summary_data.sort(key=lambda x: x["avg_rank_ic"], reverse=True)

    print("\n📈 Ranked by Rank IC (best to worst):")
    for i, data in enumerate(summary_data, 1):
        leakage_marker = " ⚠️" if data["has_leakage"] else ""
        print(
            f"{i:2d}. {data['feature_type']:20s} | IC: {data['avg_rank_ic']:7.4f} ± {data['rank_ic_std']:.4f} | {data['n_features']:4d} features{leakage_marker}"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "feature_type_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
