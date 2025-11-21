"""
Feature Type Evaluator

This script evaluates different feature types for:
1. Rank IC (Information Coefficient) calculation
2. Feature usefulness assessment
3. Top factors selection and export

Note: Data leakage detection has been moved to verify-feature-correlation script.
This script focuses solely on finding the best features by IC.
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


def evaluate_feature_type(
    df: pd.DataFrame,
    feature_type: str,
    hold_period: int = 5,
    n_splits: int = 3,
    train_only: bool = False,
    test_size: float = 0.15,
) -> Dict:
    """
    Evaluate a specific feature type for Rank IC.

    Args:
        df: DataFrame with price data
        feature_type: Feature type to evaluate (e.g., 'baseline', 'enhanced', etc.)
        hold_period: Holding period for labels
        n_splits: Number of CV folds
        train_only: Whether to use only training data

    Returns:
        Dictionary with evaluation results (IC metrics, feature list)
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

        # Split data if train_only mode (to avoid feature selection bias)
        if train_only:
            print(f"✂️  Splitting data (train_only mode, test_size={test_size})...")
            # Sort by date if index is DatetimeIndex
            if isinstance(df_features.index, pd.DatetimeIndex):
                df_features = df_features.sort_index()

            n_total = len(df_features)
            split_idx = int(n_total * (1 - test_size))
            df_features = df_features.iloc[:split_idx].copy()  # Only use training set
            print(
                f"   📊 Using {len(df_features)} samples ({len(df_features)/n_total:.1%}) for feature selection"
            )
            print(
                f"   📊 Excluding {n_total - len(df_features)} samples ({1 - len(df_features)/n_total:.1%}) from feature selection"
            )

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
        models, avg_rank_ic, cv_results, _used_feature_cols = train_rank_ic_model(
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

        # Calculate individual feature ICs for feature selection
        print("\n📊 Calculating individual feature ICs...")
        feature_ics = []
        future_return = df_with_labels["future_return"]

        for col in feature_cols:
            try:
                feature_series = df_with_labels[col]
                # Filter NaN values
                valid_mask = feature_series.notna() & future_return.notna()
                if valid_mask.sum() < 10:
                    continue

                feature_valid = feature_series[valid_mask]
                return_valid = future_return[valid_mask]

                # Compute Spearman correlation (Rank IC)
                from scipy.stats import spearmanr

                ic, p_value = spearmanr(
                    feature_valid.values, return_valid.values, nan_policy="omit"
                )

                if not np.isnan(ic):
                    # Handle p-value: if it's very small, clamp to minimum displayable value
                    p_val = float(p_value) if not np.isnan(p_value) else 1.0
                    # Clamp very small p-values to avoid display issues
                    if p_val < 1e-10:
                        p_val = 1e-10  # Minimum displayable p-value

                    feature_ics.append(
                        {
                            "feature": col,
                            "ic": float(ic),
                            "abs_ic": abs(float(ic)),
                            "p_value": p_val,
                        }
                    )
            except Exception as e:
                # Skip features that cause errors
                continue

        # Sort by absolute IC (descending)
        feature_ics.sort(key=lambda x: x["abs_ic"], reverse=True)
        results["feature_ics"] = feature_ics

        # Print top features
        print(f"   ✅ Calculated IC for {len(feature_ics)} features")
        print(f"\n   📈 Top 20 features by |Rank IC|:")
        for i, feat_ic in enumerate(feature_ics[:20], 1):
            ic_sign = "+" if feat_ic["ic"] >= 0 else "-"
            # Format p-value: use scientific notation for very small values
            p_val = feat_ic["p_value"]
            if p_val < 0.0001:
                p_str = f"{p_val:.2e}"
            else:
                p_str = f"{p_val:.4f}"
            print(
                f"      {i:2d}. {feat_ic['feature']:40s} | IC: {ic_sign}{feat_ic['abs_ic']:.4f} (p={p_str})"
            )

        if len(feature_ics) > 20:
            print(f"      ... and {len(feature_ics) - 20} more features")

        # Note: Data leakage detection has been moved to verify-feature-correlation script
        # This script focuses solely on IC evaluation and feature selection

        results["status"] = "completed"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"   ❌ Error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate different feature types for Rank IC and select top factors"
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
        "--top-factors-count",
        type=int,
        default=None,
        help="Number of top factors to include in top_factors.json (default: None, use IC threshold instead)",
    )
    parser.add_argument(
        "--top-factors-ic-threshold",
        type=float,
        default=0.02,
        help="Minimum |IC| threshold for including factors in top_factors.json (default: 0.02)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        default=False,
        help="Only use training set for feature selection (to avoid selection bias). Will split data using --test-size before feature evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set size for train-only mode (default: 0.15, should match ts-r-rank-ic-train's test_size)",
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

    # If no dates specified, try to load more data
    if args.train_start is None and args.train_end is None:
        print("   ℹ️  No date range specified, loading all available data...")
        # Try to load from a reasonable start date (e.g., 2 years ago)
        from datetime import datetime, timedelta

        # First, try to load all available data
        df = loader.load_data(
            symbol=args.symbol,
            start_date=None,
            end_date=None,
        )

        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            # If we have data, try to load more by going back further
            min_date = df.index.min()
            max_date = df.index.max()
            print(f"   ℹ️  Available data range: {min_date.date()} to {max_date.date()}")
            print(f"   ℹ️  Current samples: {len(df)}")

            # Try to load from 2 years ago if possible
            two_years_ago = max_date - timedelta(days=730)
            if two_years_ago > min_date:
                print(
                    f"   ℹ️  Attempting to load more data from {two_years_ago.date()}..."
                )
                df_extended = loader.load_data(
                    symbol=args.symbol,
                    start_date=two_years_ago.strftime("%Y-%m-%d"),
                    end_date=None,
                )
                if df_extended is not None and not df_extended.empty:
                    print(f"   ✅ Extended data loaded: {len(df_extended)} samples")
                    df = df_extended
    else:
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
            train_only=args.train_only,
            test_size=args.test_size,
        )
        all_results[feat_type] = result

    # Summary
    print("\n" + "=" * 60)
    print("📊 Feature Type Evaluation Summary")
    print("=" * 60)

    summary_data = []
    all_feature_ics = []  # Collect all features across all types

    for feat_type, result in all_results.items():
        if result.get("status") == "completed":
            avg_ic = result.get("avg_rank_ic", 0.0)
            ic_std = result.get("rank_ic_std", 0.0)
            n_features = result.get("n_features", 0)
            feature_ics = result.get("feature_ics", [])

            summary_data.append(
                {
                    "feature_type": feat_type,
                    "n_features": n_features,
                    "avg_rank_ic": avg_ic,
                    "rank_ic_std": ic_std,
                    "feature_ics": feature_ics,
                }
            )

            # Collect features with their type prefix
            for feat_ic in feature_ics:
                all_feature_ics.append(
                    {
                        "feature": f"{feat_type}::{feat_ic['feature']}",
                        "feature_type": feat_type,
                        "feature_name": feat_ic["feature"],
                        "ic": feat_ic["ic"],
                        "abs_ic": feat_ic["abs_ic"],
                        "p_value": feat_ic["p_value"],
                    }
                )

            print(
                f"{feat_type:20s} | IC: {avg_ic:7.4f} ± {ic_std:.4f} | Features: {n_features:4d}"
            )

    # Sort by Rank IC
    summary_data.sort(key=lambda x: x["avg_rank_ic"], reverse=True)

    print("\n📈 Ranked by Rank IC (best to worst):")
    for i, data in enumerate(summary_data, 1):
        print(
            f"{i:2d}. {data['feature_type']:20s} | IC: {data['avg_rank_ic']:7.4f} ± {data['rank_ic_std']:.4f} | {data['n_features']:4d} features"
        )

    # Print all features sorted by IC
    if all_feature_ics:
        all_feature_ics.sort(key=lambda x: x["abs_ic"], reverse=True)
        print("\n" + "=" * 60)
        print("📊 All Features Ranked by |Rank IC| (for feature selection)")
        print("=" * 60)
        print(f"Total features: {len(all_feature_ics)}")

        # Print all features (not just top 50)
        print(f"\nAll {len(all_feature_ics)} features:")
        for i, feat_ic in enumerate(all_feature_ics, 1):
            ic_sign = "+" if feat_ic["ic"] >= 0 else "-"
            # Format p-value: use scientific notation for very small values
            p_val = feat_ic["p_value"]
            if p_val < 0.0001:
                p_str = f"{p_val:.2e}"
            else:
                p_str = f"{p_val:.4f}"
            print(
                f"{i:4d}. {feat_ic['feature']:50s} | IC: {ic_sign}{feat_ic['abs_ic']:.4f} (p={p_str})"
            )

        # Save feature IC ranking to file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_ic_file = output_dir / "feature_ic_ranking.csv"
        feature_ic_df = pd.DataFrame(all_feature_ics)
        feature_ic_df = feature_ic_df[
            ["feature", "feature_type", "feature_name", "ic", "abs_ic", "p_value"]
        ]
        feature_ic_df.to_csv(feature_ic_file, index=False)
        print(f"\n💾 Feature IC ranking saved to {feature_ic_file}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "feature_type_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to {results_file}")

    # Generate top_factors.json for ts-r-rank-ic-train compatibility
    if all_feature_ics:
        print("\n📝 Generating top_factors.json for ts-r-rank-ic-train...")
        top_factors_file = output_dir / "top_factors.json"
        try:
            # Filter features: exclude label columns and _symbol
            label_prefixes = ("signal_", "binary_signal_", "future_return_")
            label_exact = {"signal", "binary_signal", "future_return", "_symbol"}

            # Select top features based on IC
            # First, collect features with deduplication (keep the one with highest IC)
            feature_dict = {}  # feature_name -> best_feat_ic
            for feat_ic in all_feature_ics:
                feat_name = feat_ic["feature_name"]
                # Skip label columns
                if feat_name in label_exact or any(
                    feat_name.startswith(prefix) for prefix in label_prefixes
                ):
                    continue

                # Keep the feature with highest abs_ic if duplicate
                if feat_name not in feature_dict:
                    feature_dict[feat_name] = feat_ic
                elif feat_ic["abs_ic"] > feature_dict[feat_name]["abs_ic"]:
                    feature_dict[feat_name] = feat_ic

            # Convert to sorted list
            deduplicated_features = sorted(
                feature_dict.values(), key=lambda x: x["abs_ic"], reverse=True
            )

            # Select based on criteria
            if args.top_factors_count is not None:
                # Use top N features
                selected_features = [
                    feat_ic["feature_name"]
                    for feat_ic in deduplicated_features[: args.top_factors_count]
                ]
            else:
                # Use IC threshold
                selected_features = [
                    feat_ic["feature_name"]
                    for feat_ic in deduplicated_features
                    if feat_ic["abs_ic"] >= args.top_factors_ic_threshold
                ]

            # Calculate statistics
            if selected_features:
                avg_ic = np.mean(
                    [
                        feat_ic["abs_ic"]
                        for feat_ic in all_feature_ics
                        if feat_ic["feature_name"] in selected_features
                    ]
                )
                max_ic = max(
                    [
                        feat_ic["abs_ic"]
                        for feat_ic in all_feature_ics
                        if feat_ic["feature_name"] in selected_features
                    ]
                )
            else:
                avg_ic = 0.0
                max_ic = 0.0

            top_factors_data = {
                "top_factors": [{"name": factor} for factor in selected_features],
                "count": len(selected_features),
                "source": "feature_evaluation",
                "stage": "Feature IC ranking",
                "effective": True,
                "selection_criteria": {
                    "method": "top_n" if args.top_factors_count else "ic_threshold",
                    "top_n": args.top_factors_count,
                    "ic_threshold": args.top_factors_ic_threshold,
                },
                "performance": {
                    "avg_abs_ic": float(avg_ic),
                    "max_abs_ic": float(max_ic),
                    "total_features_evaluated": len(all_feature_ics),
                },
            }

            with open(top_factors_file, "w", encoding="utf-8") as f:
                json.dump(top_factors_data, f, indent=2, ensure_ascii=False)

            print(
                f"   ✅ Generated top_factors.json with {len(selected_features)} features"
            )
            print(f"   📄 File location: {top_factors_file}")
            print(f"   📊 Avg |IC|: {avg_ic:.4f}, Max |IC|: {max_ic:.4f}")

            # Print all selected features
            print(
                f"\n   📋 Selected {len(selected_features)} features for top_factors.json:"
            )
            for i, feat_name in enumerate(selected_features, 1):
                # Find the IC value for this feature
                feat_ic_info = next(
                    (
                        f
                        for f in deduplicated_features
                        if f["feature_name"] == feat_name
                    ),
                    None,
                )
                if feat_ic_info:
                    ic_sign = "+" if feat_ic_info["ic"] >= 0 else "-"
                    p_val = feat_ic_info["p_value"]
                    if p_val < 0.0001:
                        p_str = f"{p_val:.2e}"
                    else:
                        p_str = f"{p_val:.4f}"
                    print(
                        f"      {i:3d}. {feat_name:50s} | IC: {ic_sign}{feat_ic_info['abs_ic']:.4f} (p={p_str})"
                    )
                else:
                    print(f"      {i:3d}. {feat_name}")
        except Exception as e:
            print(f"   ⚠️  Failed to generate top_factors.json: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
