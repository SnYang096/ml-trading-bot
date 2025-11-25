"""Dimensionality reduction comparison and research workflows."""

from __future__ import annotations

# Standard library imports
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party imports
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local data tools imports
from data_tools.rolling_data import create_labels_multi_horizon

# Local time_series_model imports
from time_series_model.backtesting.vectorbot import (
    backtest_classification_model,
    calculate_financial_metrics_from_returns,
    calculate_strategy_returns_from_predictions,
)
from time_series_model.pipeline.dimensionality.data_loader import (
    create_enhanced_sample_data,
    load_real_market_data,
)
from time_series_model.pipeline.dimensionality.evaluation import (
    _generate_shap_outputs,
    calculate_financial_metrics,
    compute_selection_score,
    evaluate_model_performance,
    sanitize_features,
)
from time_series_model.pipeline.dimensionality.model_training import (
    train_production_lightgbm,
)
from time_series_model.pipeline.dimensionality.report_generator import (
    write_html_report,
)
from time_series_model.pipeline.dimensionality.utils import (
    _derive_feature_insights,
    _get_primary_metric,
    _slugify,
)

DIM_COMPARE_RESULTS_ROOT = Path("results") / "dim_compare"


def save_production_results(
    results: Dict,
    model,
    results_dir: str,
) -> str:
    """Save production training results and model to disk.

    Args:
        results: Dictionary containing training results and metrics
        model: Trained model object to save
        results_dir: Directory path where results will be saved

    Returns:
        str: Path to the results directory
    """
    print("💾 Saving production results...")
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/production_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump(model, f"{results_dir}/production_model.pkl")

    print(f"✅ Results saved to {results_dir}")
    return results_dir


def run_dimensionality_comparison(
    data_path: str = "/data/parquet_data",
    symbol: str = "ETHUSDT",
    train_start: str | None = None,
    train_end: str | None = None,
    feature_type: str = "comprehensive",
    shap_analysis: bool = True,
    timeframe: str = "15T",
) -> Tuple[Dict, any, type(None), str]:
    """Run feature selection and model training workflow.

    This function implements a three-stage feature selection pipeline:
    1. Stage 1: Missing/stability filter (removes features with >20% missing or low variance)
    2. Stage 2: IC ranking (selects top features by Information Coefficient)
    3. Stage 3: Correlation-based representative selection (removes redundant features)

    Args:
        data_path: Path to parquet data directory
        symbol: Trading symbol (e.g., "ETHUSDT")
        train_start: Start date for training data (YYYY-MM-DD format)
        train_end: End date for training data (YYYY-MM-DD format)
        feature_type: Type of features to use (default: "comprehensive")
        shap_analysis: Whether to perform SHAP analysis (currently unused)
        timeframe: Data timeframe (e.g., "15T" for 15 minutes)

    Returns:
        Tuple containing:
            - results: Dictionary with training results and metrics
            - model: Trained model object
            - None: Placeholder for compatibility
            - results_dir: Path to results directory
    """
    print("🚀 Feature Selection and Model Training")
    print("=" * 60)
    start_dt = datetime.now()
    timestamp_start = start_dt.strftime("%Y%m%d_%H%M%S")
    symbol_slug = _slugify(symbol)
    feature_slug = _slugify(feature_type)

    X, y, feature_names, horizons_loaded, df_features_full = load_real_market_data(
        data_path,
        symbol,
        start_date=train_start,
        end_date=train_end,
        feature_type=feature_type,
        timeframe=timeframe,
    )

    print(f"✅ Data loaded: {X.shape}, {y.shape}")
    print(f"✅ Features: {len(feature_names)}")

    print("\n📊 Data preprocessing...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Feature/label sanitation before model training
    X_scaled = sanitize_features(X_scaled, clip_std=5.0)
    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite values remain in features after sanitation")
    if not np.isfinite(y_scaled).all():
        raise ValueError("Non-finite values found in labels after scaling")

    # Convert to DataFrame for feature selection
    dfX = pd.DataFrame(X_scaled, columns=feature_names)
    y_series = pd.Series(y_scaled)

    # Stage 1: Missing/stability filter
    print(f"\n[Stage 1] Missing/stability filter: {len(dfX.columns)} features")
    keep_all = []
    for c in dfX.columns:
        s = dfX[c]
        missing_ratio = s.isna().mean()
        std_val = s.std()
        if missing_ratio < 0.2 and std_val > 1e-8:
            keep_all.append(c)

    df_all = dfX[keep_all].ffill().bfill().fillna(0.0)
    X_all_scaled = sanitize_features(StandardScaler().fit_transform(df_all.values))
    print(f"   ✅ Stage 1: {len(keep_all)} features after filtering")

    # Stage 2: IC ranking
    print(f"\n[Stage 2] IC ranking...")
    ic_scores = {}
    for col in df_all.columns:
        try:
            ic = spearmanr(df_all[col].values, y_series.values, nan_policy="omit")[0]
            ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        except Exception:
            ic_scores[col] = 0.0

    top_sorted = sorted(ic_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    # Determine target factor count with intelligent scaling
    total_features = len(top_sorted)
    if total_features > 200:
        target_top_k = min(80, total_features)
    elif total_features > 100:
        target_top_k = min(60, total_features)
    elif total_features > 50:
        target_top_k = min(40, total_features)
    elif total_features > 20:
        target_top_k = min(25, total_features)
    else:
        target_top_k = max(10, int(total_features * 0.7))
    top_cols = [c for c, _ in top_sorted[:target_top_k]]
    df_ic = df_all[top_cols]
    X_ic_scaled = sanitize_features(StandardScaler().fit_transform(df_ic.values))
    print(f"   ✅ Stage 2: {len(top_cols)} features after IC ranking")

    # Stage 3: Correlation-based representative selection
    print(f"\n[Stage 3] Correlation-based representative selection...")
    df_ic_clean = df_ic.ffill().bfill().fillna(0.0)
    # Stage 3 should further reduce dimensionality: target 60-70% of Stage 2 features
    stage3_target = max(10, int(target_top_k * 0.65))
    desired_reps = min(stage3_target, len(df_ic_clean.columns))
    print(
        f"   📊 Stage 3 target: {desired_reps} features (from {len(df_ic_clean.columns)} Stage 2 features, further reduction: {desired_reps/len(df_ic_clean.columns):.1%})"
    )

    reps = []
    if not df_ic_clean.empty:
        # Sort by IC score and take top N
        cols_with_ic = [
            (col, abs(ic_scores.get(col, 0.0))) for col in df_ic_clean.columns
        ]
        cols_with_ic.sort(key=lambda x: x[1], reverse=True)
        top_ic_cols = [col for col, _ in cols_with_ic[:desired_reps]]

        # Apply correlation filtering
        corr = df_ic_clean[top_ic_cols].corr().abs().fillna(0.0)
        for c in top_ic_cols:
            if all(corr.loc[c, r] < 0.9 for r in reps):
                reps.append(c)

        # Fill up to desired count if needed
        if len(reps) < desired_reps:
            additional = [c for c in top_ic_cols if c not in reps][
                : max(desired_reps - len(reps), 0)
            ]
            reps.extend(additional)

    if not reps:
        reps = list(df_ic_clean.columns)[: max(desired_reps, 60)]

    df_reps = (
        df_ic_clean[reps]
        if set(reps).issubset(df_ic_clean.columns)
        else df_all[reps].fillna(0.0)
    )
    X_reps_scaled = sanitize_features(StandardScaler().fit_transform(df_reps.values))
    print(f"   ✅ Stage 3: {len(reps)} representative features selected")

    # Split data (same split for all stages)
    n_samples = len(X_all_scaled)
    split_idx = int(n_samples * 0.7)
    split_idx2 = int(n_samples * 0.85)

    train_indices = np.arange(split_idx)
    val_indices = np.arange(split_idx, split_idx2)
    test_indices = np.arange(split_idx2, n_samples)

    # Split for Stage 1 (all features)
    X_train_all = X_all_scaled[train_indices]
    X_val_all = X_all_scaled[val_indices]
    X_test_all = X_all_scaled[test_indices]

    # Split for Stage 2 (IC-filtered features)
    X_train_ic = X_ic_scaled[train_indices]
    X_val_ic = X_ic_scaled[val_indices]
    X_test_ic = X_ic_scaled[test_indices]

    # Split for Stage 3 (representative features)
    X_train_reps = X_reps_scaled[train_indices]
    X_val_reps = X_reps_scaled[val_indices]
    X_test_reps = X_reps_scaled[test_indices]

    # Split labels
    y_train = y_scaled[train_indices]
    y_val = y_scaled[val_indices]
    y_test = y_scaled[test_indices]

    # Extract future_return for Rank IC evaluation (if available)
    # This allows us to compute Rank IC even with classification models
    future_return_train = None
    future_return_val = None
    future_return_test = None

    # Try to get future_return from df_features_full (returned from load_real_market_data)
    if not df_features_full.empty:
        default_horizon = (
            horizons_loaded[0] if horizons_loaded and len(horizons_loaded) > 0 else 1
        )
        future_return_col = f"future_return_{default_horizon}"

        if future_return_col in df_features_full.columns:
            try:
                # Align indices: df_features_full should have same length as X_scaled
                if len(df_features_full) == len(y_scaled):
                    future_return_all = df_features_full[future_return_col].values
                    future_return_train = future_return_all[train_indices]
                    future_return_val = future_return_all[val_indices]
                    future_return_test = future_return_all[test_indices]
                    print(f"   ✅ Found {future_return_col} for Rank IC evaluation")
                else:
                    print(
                        f"   ⚠️  Length mismatch: df_features_full ({len(df_features_full)}) vs y_scaled ({len(y_scaled)})"
                    )
            except Exception as e:
                print(f"   ⚠️  Could not extract future_return for Rank IC: {e}")

    print(
        f"\n✅ Data split: Train {X_train_all.shape}, Val {X_val_all.shape}, Test {X_test_all.shape}"
    )

    # Train models for comparison
    print("\n" + "=" * 60)
    print("Training models for comparison (Before vs After dimensionality reduction)")
    print("=" * 60)

    # Stage 1: All features (before dimensionality reduction)
    print("\n🌲 [Before Reduction] Training model with all features...")
    model_all = train_production_lightgbm(
        X_train_all,
        y_train,
        X_val_all,
        y_val,
        feature_names=keep_all,
        y_train_true_return=future_return_train,
        y_val_true_return=future_return_val,
    )
    perf_all = evaluate_model_performance(
        model_all,
        X_test_all,
        y_test,
        "All Features (Before Reduction)",
    )

    # Stage 3: Representative features (after dimensionality reduction)
    print("\n🌲 [After Reduction] Training model with representative features...")
    model_reps = train_production_lightgbm(
        X_train_reps,
        y_train,
        X_val_reps,
        y_val,
        feature_names=reps,
        y_train_true_return=future_return_train,
        y_val_true_return=future_return_val,
    )
    perf_reps = evaluate_model_performance(
        model_reps,
        X_test_reps,
        y_test,
        "Representative Features (After Reduction)",
    )

    print("\n📊 Performance Comparison:")
    print(f"   Before Reduction (All Features):")
    print(
        f"      Features: {len(keep_all)}, R2: {perf_all.get('r2', 0):.4f}, RMSE: {perf_all.get('rmse', 0):.4f}"
    )
    print(f"   After Reduction (Representative Features):")
    print(
        f"      Features: {len(reps)}, R2: {perf_reps.get('r2', 0):.4f}, RMSE: {perf_reps.get('rmse', 0):.4f}"
    )

    compression_ratio = len(keep_all) / max(len(reps), 1)
    performance_change = perf_reps.get("r2", 0) - perf_all.get("r2", 0)
    performance_change_percent = (
        (performance_change / perf_all.get("r2", 1)) * 100
        if perf_all.get("r2", 0) != 0
        else 0
    )

    print(f"   Compression Ratio: {compression_ratio:.2f}x")
    print(
        f"   Performance Change: {performance_change:+.4f} ({performance_change_percent:+.2f}%)"
    )

    print("\n📋 Generating production report...")

    # Format training date range for directory name (include symbol and feature_type)
    if train_start and train_end:
        # Extract date parts (YYYY-MM-DD -> YYYYMMDD)
        train_start_date = train_start.replace("-", "")[:8]
        train_end_date = train_end.replace("-", "")[:8]
        dir_date_suffix = (
            f"{symbol_slug}_{feature_slug}_{train_start_date}_{train_end_date}"
        )
    else:
        # Fallback to runtime timestamps if no date range provided
        train_start_date = None
        train_end_date = None
        timestamp_end = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_date_suffix = (
            f"{symbol_slug}_{feature_slug}_{timestamp_start}_{timestamp_end}"
        )

    # Calculate feature insights
    feature_insights = _derive_feature_insights(perf_all, perf_reps)

    results = {
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "duration_sec": (datetime.now() - start_dt).total_seconds(),
        "data_info": {
            "original_features_count": X.shape[1],
            "stage1_all_features": len(keep_all),
            "stage2_ic_filtered": len(top_cols),
            "stage3_representatives": len(reps),
            "compression_ratio": compression_ratio,
            "training_samples": len(X_train_all),
            "validation_samples": len(X_val_all),
            "test_samples": len(X_test_all),
        },
        "training_info": {
            "lightgbm_all_iterations": model_all.best_iteration,
            "lightgbm_reps_iterations": model_reps.best_iteration,
        },
        "performance": {
            "before_reduction": perf_all,
            "after_reduction": perf_reps,
            "performance_change": performance_change,
            "performance_change_percent": performance_change_percent,
        },
        "insights": feature_insights,
        "model_info": {
            "device_used": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "selected_feature_names": reps[:10],
        },
    }

    # Build results directory name using training date range (if available) or runtime timestamps
    DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results_dir_path = DIM_COMPARE_RESULTS_ROOT / dir_date_suffix
    # Save the model after reduction (representative features model)
    results_dir = save_production_results(
        results,
        model_reps,
        str(results_dir_path),
    )

    # Generate top_factors.json for rolling training compatibility
    print("\n📝 Generating top_factors.json for rolling training...")
    top_factors_file = results_dir_path / "top_factors.json"
    try:
        # Filter out label columns (signal_*, binary_signal_*, future_return_*)
        # Also filter out _symbol (categorical identifier, not a factor)
        label_prefixes = ("signal_", "binary_signal_", "future_return_")
        label_exact = {"signal", "binary_signal", "future_return", "_symbol"}
        selected_features = [
            f
            for f in reps
            if f not in label_exact
            and not any(f.startswith(prefix) for prefix in label_prefixes)
        ]

        top_factors_data = {
            "top_factors": [{"name": factor} for factor in selected_features],
            "count": len(selected_features),
            "source": "dimensionality_reduction",
            "stage": "Stage 3: Representative features (correlation-based selection)",
            "effective": True,
            "compression_ratio": compression_ratio,
            "performance": {
                "before_reduction_r2": perf_all.get("r2", 0),
                "after_reduction_r2": perf_reps.get("r2", 0),
                "performance_change": performance_change,
            },
        }

        with open(top_factors_file, "w", encoding="utf-8") as f:
            json.dump(top_factors_data, f, indent=2, ensure_ascii=False)
        print(
            f"   ✅ Generated top_factors.json with {len(selected_features)} features"
        )
        print(f"   📄 File location: {top_factors_file}")
    except Exception as e:
        print(f"   ⚠️ Failed to generate top_factors.json: {e}")

    print("\n" + "=" * 60)
    print("🎉 Feature Selection and Model Training Complete!")
    print("=" * 60)
    print(
        f"📊 Compression: {len(keep_all)} → {len(reps)} features ({compression_ratio:.2f}x)"
    )
    print(
        f"📈 Performance: R2 changed from {perf_all.get('r2', 0):.4f} to {perf_reps.get('r2', 0):.4f} ({performance_change:+.4f})"
    )
    print(f"💾 Results saved to: {results_dir}")
    print(f"📄 top_factors.json generated for rolling training")

    return results, model_reps, results_dir


def main() -> Tuple[Dict, any, str]:
    global DIM_COMPARE_RESULTS_ROOT
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction comparison: evaluate feature reduction stages (All → IC-filtered → Representatives)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="/data/parquet_data",
        help="Parquet directory with real market data",
    )
    parser.add_argument(
        "--symbol",
        default="ETH-USD",
        help="Symbol name(s) (e.g., BTC-USD, ETH-USD or BTC-USD,ETH-USD,SOL-USD for multi-asset training)",
    )
    parser.add_argument(
        "--train-start",
        default=None,
        help="Start date (YYYY-MM-DD) for data window",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="End date (YYYY-MM-DD) for data window",
    )
    parser.add_argument(
        "--report-html",
        default=None,
        help="Path to write an HTML summary report",
    )
    parser.add_argument(
        "--export-model",
        default=None,
        help="Optional path under models/ to copy the best production_model.pkl",
    )
    parser.add_argument(
        "--research-ablation",
        action="store_true",
        help="Run three-stage feature selection: IC filter -> representative selection -> model training",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,10,15",
        help="Comma-separated list of forward bars for multi-horizon labels (e.g., 1,5,10,15)",
    )
    parser.add_argument(
        "--binary-signals",
        action="store_true",
        default=True,  # Default to binary classification (2-class)
        help="Use binary labels (1=Long, 0=Short). Default: True. Threshold controlled by --label-threshold",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.0,
        help="Threshold for future return to classify Long vs Short in binary mode (default 0.0)",
    )
    parser.set_defaults(shap_analysis=True)
    parser.add_argument(
        "--shap-analysis",
        dest="shap_analysis",
        action="store_true",
        help="Generate SHAP explainability plots for representative factors (default: enabled).",
    )
    parser.add_argument(
        "--no-shap-analysis",
        dest="shap_analysis",
        action="store_false",
        help="Disable SHAP explainability plots.",
    )
    parser.add_argument(
        "--validate-pipeline",
        dest="validate_pipeline",
        action="store_true",
        default=False,
        help="Validate pipeline with synthetic data containing known signals before using real data. "
        "This injects a strong signal into the first feature to verify the model can learn.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["classification", "regression", "both"],
        help="Task type to evaluate: classification | regression | both (default)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="Feature type: baseline/default/enhanced/hurst/wavelet/hilbert/spectral/order_flow/dl_sequence/comprehensive or combos (default: comprehensive)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5T",
        help="Timeframe for data resampling (e.g., 5T, 15T, 60T, 240T). Default: 5T",
    )
    parser.add_argument(
        "--enable-stability-validation",
        action="store_true",
        help="Enable stability validation: use recent data for factor selection, validate on longer historical data",
    )
    parser.add_argument(
        "--validation-start",
        default=None,
        help="Start date (YYYY-MM-DD) for stability validation period. If not provided and --enable-stability-validation is set, automatically uses train-start minus 2-3 years",
    )
    parser.add_argument(
        "--validation-years",
        type=int,
        default=3,
        help="Number of years to look back for stability validation (default: 3). Used when --enable-stability-validation is set and --validation-start is not provided",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=["sharpe", "f1", "r2", "composite"],
        help="Metric to use for feature selection scoring: sharpe | f1 | r2 | composite (default: composite)",
    )
    parser.add_argument(
        "--max-dd-threshold",
        type=float,
        default=-20.0,
        help="Maximum drawdown threshold for composite score penalty (default: -20.0)",
    )
    parser.add_argument(
        "--composite-alpha",
        type=float,
        default=0.5,
        help="Alpha weight for drawdown penalty in composite score (default: 0.5)",
    )
    parser.add_argument(
        "--composite-beta",
        type=float,
        default=0.5,
        help="Beta weight for F1 penalty in composite score (default: 0.5)",
    )

    args = parser.parse_args()
    symbol_slug = _slugify(args.symbol)
    feature_type_slug = _slugify(args.feature_type)

    # Enforce minimal training window (one quarter ~ 90 days)
    if args.train_start and args.train_end:
        try:
            start_dt_chk = pd.to_datetime(args.train_start)
            end_dt_chk = pd.to_datetime(args.train_end)
            if (end_dt_chk - start_dt_chk).days < 90:
                raise ValueError(
                    f"Training window too short: {args.train_start} → {args.train_end} (< 90 days). Please provide at least one quarter."
                )
        except Exception as _e:
            raise

    # Parse grid search parameters
    # Default behavior: if ablation not specified, enable ablation by default
    if not args.research_ablation:
        args.research_ablation = True

    if args.research_ablation:
        ablation_start_dt = datetime.now()
        ablation_start_ts = ablation_start_dt.strftime("%Y%m%d_%H%M%S")
        # Format training date range for directory name (if provided)
        if args.train_start and args.train_end:
            train_start_date = args.train_start.replace("-", "")[:8]
            train_end_date = args.train_end.replace("-", "")[:8]
            ablation_dir_date_suffix = (
                f"{symbol_slug}_{feature_type_slug}_{train_start_date}_{train_end_date}"
            )
        else:
            train_start_date = None
            train_end_date = None
            ablation_dir_date_suffix = f"{symbol_slug}_{feature_type_slug}_{ablation_start_ts}"  # Use runtime timestamps with symbol and feature_type
        # Parse horizons from args
        horizons_list = (
            [int(h.strip()) for h in args.horizons.split(",")] if args.horizons else [1]
        )

        # Load engineered features for IC & representative selection
        X_raw, y_raw, feature_names, horizons_loaded, df_features_original = (
            load_real_market_data(
                args.data_path,
                args.symbol,
                args.train_start,
                args.train_end,
                horizons=horizons_list,
                feature_type=args.feature_type,
                timeframe=args.timeframe,
            )
        )

        # Use loaded horizons or fallback to parsed horizons
        horizons = (
            horizons_loaded
            if horizons_loaded and len(horizons_loaded) > 0
            else horizons_list
        )
        horizons_list = (
            horizons  # Ensure horizons_list is available for pipeline validation
        )

        original_feature_count = len(feature_names)  # Save original count (482)

        # Create index for DataFrame
        # If df_features_original is empty (sample data), create a default index
        if df_features_original.empty or len(df_features_original.index) == 0:
            # Use default integer index for sample data
            df_index = pd.RangeIndex(start=0, stop=len(X_raw))
        else:
            # Use original index, but ensure it matches X_raw length
            df_index = df_features_original.index[: len(X_raw)]
            if len(df_index) != len(X_raw):
                # If index length doesn't match, create default index
                df_index = pd.RangeIndex(start=0, stop=len(X_raw))

        dfX = pd.DataFrame(X_raw, columns=feature_names, index=df_index)

        # Convert all columns to numeric (handle object/string types)
        # This is necessary because some features may be stored as object type
        # due to NaN values or data loading issues
        for c in dfX.columns:
            if c != "_symbol":  # Skip symbol column
                dfX[c] = pd.to_numeric(dfX[c], errors="coerce")

        # Check for data leakage: features that might contain future information
        leakage_keywords = ["future_return", "binary_signal", "signal_"]
        potential_leakage = [
            c for c in dfX.columns if any(kw in c for kw in leakage_keywords)
        ]
        if potential_leakage:
            print(
                f"   ⚠️  WARNING: Potential data leakage detected! Features: {potential_leakage}"
            )
            print(f"   These features should be excluded from training!")

        # For backward compatibility, use default horizon
        # Always use binary signals (0=Short, 1=Long)
        # Convert y_raw to Series, ensuring it's integer classification labels [0, 1, 2]
        # Check if y_raw contains float values (regression) or integer labels (classification)
        y_raw_array = np.asarray(y_raw)
        unique_values = np.unique(y_raw_array[~np.isnan(y_raw_array)])

        # If y_raw contains float values (not in [0, 1, 2]), convert to classification labels
        if len(unique_values) > 0 and not np.all(np.isin(unique_values, [0, 1, 2])):
            # This is likely regression data (e.g., from sample data generation)
            # Convert to 3-class classification using quantiles
            print(
                f"   ⚠️  Converting regression values to 3-class labels (found {len(unique_values)} unique values)"
            )
            y_sorted = np.sort(y_raw_array[~np.isnan(y_raw_array)])
            if len(y_sorted) > 0:
                top_threshold = np.percentile(y_sorted, 70)  # Top 30% = Long (2)
                bottom_threshold = np.percentile(y_sorted, 30)  # Bottom 30% = Short (1)
                y_class = np.zeros_like(y_raw_array, dtype=int)
                y_class[y_raw_array >= top_threshold] = 2  # Long
                y_class[y_raw_array <= bottom_threshold] = 1  # Short
                # Hold (0) is already set by default
                y_raw = y_class
                print(
                    f"   📊 Converted labels: {dict(zip(*np.unique(y_class, return_counts=True)))}"
                )
            else:
                # Fallback: use binary classification
                y_raw = (y_raw_array > 0).astype(int)
                print(
                    f"   ⚠️  Fallback to binary classification (no valid values for quantile conversion)"
                )

        # Use Int64 (nullable integer) to handle NaN values
        y_series = pd.Series(y_raw, index=dfX.index[: len(y_raw)], dtype="Int64")

        # CRITICAL: Check sample size before any cleaning
        initial_samples = len(y_series)
        valid_samples = y_series.notna().sum()
        print(f"\n   📊 Label Sample Size Check (before cleaning):")
        print(f"      Total samples: {initial_samples}")
        print(
            f"      Valid (non-NaN) samples: {valid_samples} ({valid_samples/initial_samples*100:.1f}%)"
        )
        # Use 3-class signal (0=Hold, 1=Long, 2=Short) - no remapping needed
        # Keep y_series as is (already 3-class from signal_{horizon})
        use_binary = False  # Always use 3-class now
        if False:  # Disabled: no longer remapping to binary
            try:
                # Use first horizon's future return if available
                default_h = horizons[0] if horizons else 1
                fr_col = f"future_return_{default_h}"
                if fr_col in df_features_original.columns:
                    fr = df_features_original[fr_col].values
                else:
                    # Fallback: compute from close using pandas shift (NOT np.roll to avoid data leakage)
                    # np.roll would cause circular shift and data leakage
                    close_series = pd.Series(df_features_original["close"].values)
                    fr = (close_series.shift(-default_h) / close_series - 1.0).values
                    # Set NaN values (at the end) to 0 to avoid issues
                    fr = np.nan_to_num(fr, nan=0.0)
                thr = float(args.label_threshold)
                y_series = pd.Series((fr > thr).astype(int))
                print(
                    f"[Label] Using binary signals (thr={thr}), positives={y_series.mean():.4f}"
                )
                # Check for suspicious label distribution
                unique_labels, counts = np.unique(y_series.values, return_counts=True)
                print(f"[Label] Label distribution: {dict(zip(unique_labels, counts))}")
                if len(unique_labels) == 1:
                    print(
                        f"   ⚠️  WARNING: All labels are the same ({unique_labels[0]})! This will cause perfect fit."
                    )
                elif min(counts) / len(y_series) < 0.01:
                    print(
                        f"   ⚠️  WARNING: Extreme label imbalance! One class has only {min(counts)} samples ({min(counts)/len(y_series)*100:.2f}%)"
                    )
            except Exception as exc:
                print(f"⚠️ Binary label remap failed, keep original labels: {exc}")

        # CRITICAL: Check sample size after label processing
        final_valid_samples = y_series.notna().sum()
        MIN_SAMPLES_REQUIRED = 10000
        if final_valid_samples < MIN_SAMPLES_REQUIRED:
            print(
                f"\n   🚨 CRITICAL WARNING: Only {final_valid_samples} valid samples after label processing (minimum: {MIN_SAMPLES_REQUIRED})"
            )
            print(f"      This may indicate:")
            print(f"      1. Too many NaN labels (check label generation)")
            print(f"      2. Data period too short")
            print(f"      3. Horizon too long (future_return not available)")
            print(f"      → Consider using shorter horizons or checking data quality")
        else:
            print(
                f"   ✅ Label sample size check passed: {final_valid_samples} >= {MIN_SAMPLES_REQUIRED}"
            )

        # Stage 1: All original features (482) - missing/stability filter only
        print(f"\n[Stage 1] All original features: {len(dfX.columns)}")
        keep_all = []
        excluded_missing = []
        excluded_std = []
        excluded_non_numeric = []

        # Handle categorical features (like _symbol) separately
        categorical_features = []
        for c in dfX.columns:
            # Handle _symbol as categorical feature (not numeric)
            if c == "_symbol":
                # Check if _symbol has valid values AND multiple unique values
                # Only useful for multi-asset training (when there are multiple symbols)
                if dfX[c].notna().sum() > 0:
                    unique_symbols = dfX[c].nunique()
                    if unique_symbols > 1:
                        categorical_features.append(c)
                        keep_all.append(c)  # Include _symbol in features
                        print(
                            f"   ✅ Including '{c}' as categorical feature ({unique_symbols} unique values)"
                        )
                    else:
                        # Exclude constant _symbol (only one unique value - not useful for single-asset training)
                        excluded_non_numeric.append(c)
                        print(
                            f"   ⚠️  Excluding '{c}' (constant: only {unique_symbols} unique value - not useful for single-asset training)"
                        )
                else:
                    excluded_non_numeric.append(c)
                continue

            # Skip other non-numeric columns
            if not pd.api.types.is_numeric_dtype(dfX[c]):
                excluded_non_numeric.append(c)
                continue

            # Process numeric features
            s = dfX[c]
            missing_ratio = s.isna().mean()
            std_val = s.std()
            if missing_ratio < 0.2 and std_val > 1e-8:
                keep_all.append(c)
            else:
                # Track why each feature was excluded
                if missing_ratio >= 0.2:
                    excluded_missing.append((c, missing_ratio))
                if std_val <= 1e-8:
                    excluded_std.append((c, std_val))

        # Print summary of exclusions
        if excluded_missing:
            print(
                f"   ⚠️  {len(excluded_missing)} features excluded due to high missing ratio (>=20%):"
            )
            for c, ratio in excluded_missing[:10]:  # Show first 10
                print(f"      - {c}: {ratio:.2%} missing")
            if len(excluded_missing) > 10:
                print(f"      ... and {len(excluded_missing) - 10} more")

        if excluded_std:
            print(
                f"   ⚠️  {len(excluded_std)} features excluded due to low variance (std <= 1e-8):"
            )
            for c, std_val in excluded_std[:10]:  # Show first 10
                print(f"      - {c}: std={std_val:.2e}")
            if len(excluded_std) > 10:
                print(f"      ... and {len(excluded_std) - 10} more")

        if excluded_non_numeric:
            print(
                f"   ℹ️  {len(excluded_non_numeric)} non-numeric columns skipped: {excluded_non_numeric}"
            )

        if len(keep_all) == 0:
            error_msg = (
                f"❌ No features passed Stage 1 filtering!\n"
                f"   Total features: {len(dfX.columns)}\n"
                f"   Excluded due to missing ratio >= 20%: {len(excluded_missing)}\n"
                f"   Excluded due to low variance (std <= 1e-8): {len(excluded_std)}\n"
                f"   Non-numeric columns: {len(excluded_non_numeric)}\n"
                f"   All features were filtered out.\n"
                f"\n"
                f"This may indicate:\n"
                f"  1. All features have >20% missing values\n"
                f"  2. All features have zero variance (std <= 1e-8)\n"
                f"  3. Feature engineering produced invalid features\n"
                f"\n"
                f"Please check:\n"
                f"  - Feature engineering output\n"
                f"  - Data quality (missing values, variance)\n"
                f"  - Consider relaxing filter thresholds"
            )
            raise ValueError(error_msg)

        # Separate categorical and numeric features
        # Categorical features should not be standardized
        numeric_features = [c for c in keep_all if c not in categorical_features]
        cat_features_in_keep = [c for c in categorical_features if c in keep_all]

        # Prepare numeric features (standardize)
        # CRITICAL: Use forward fill (ffill) for key features instead of dropping rows
        # This prevents sample depletion from over-cleaning
        # Strategy: ffill -> bfill -> fillna(0.0) to preserve maximum samples
        initial_numeric_samples = len(dfX)
        df_numeric = dfX[numeric_features].ffill().bfill().fillna(0.0)

        # Check sample retention after filling
        final_numeric_samples = len(df_numeric)
        if final_numeric_samples < initial_numeric_samples:
            print(
                f"   ⚠️  WARNING: Lost {initial_numeric_samples - final_numeric_samples} samples during numeric feature filling"
            )
        else:
            print(
                f"   ✅ Numeric feature filling preserved all {final_numeric_samples} samples (using ffill/bfill/fillna)"
            )

        # Verify minimum sample requirement
        MIN_SAMPLES_REQUIRED = 10000
        if final_numeric_samples < MIN_SAMPLES_REQUIRED:
            print(
                f"   🚨 CRITICAL WARNING: Only {final_numeric_samples} samples after feature cleaning (minimum: {MIN_SAMPLES_REQUIRED})"
            )
            print(f"      Model training may be unreliable with insufficient samples")
            print(f"      → Consider:")
            print(f"         1. Using longer data period")
            print(f"         2. Relaxing feature filtering thresholds")
            print(f"         3. Using more aggressive filling strategies")
        else:
            print(
                f"   ✅ Sample size check passed: {final_numeric_samples} >= {MIN_SAMPLES_REQUIRED}"
            )

        # Prepare categorical features (encode as integers, don't standardize)
        df_categorical = None
        if cat_features_in_keep:
            from sklearn.preprocessing import LabelEncoder

            df_categorical = dfX[cat_features_in_keep].copy()
            # Encode categorical features as integers (LightGBM can handle both string and int)
            for cat_col in cat_features_in_keep:
                le = LabelEncoder()
                # Fill NaN with a special value before encoding
                df_categorical[cat_col] = df_categorical[cat_col].fillna("UNKNOWN")
                df_categorical[cat_col] = le.fit_transform(
                    df_categorical[cat_col].astype(str)
                )
                print(
                    f"   ✅ Encoded categorical feature '{cat_col}': {len(le.classes_)} unique values"
                )

        # Pipeline validation: Signal injection moved to AFTER reps are determined
        # This ensures the signal is injected into a feature that will actually be used in training
        # See signal injection code after Stage 3 feature selection (around line 4377)

        # Combine numeric and categorical features
        if df_categorical is not None and len(cat_features_in_keep) > 0:
            df_all = pd.concat([df_numeric, df_categorical], axis=1)
            # Reorder columns to match keep_all order
            df_all = df_all[keep_all]
        else:
            df_all = df_numeric

        X_all = df_all.values

        # Check for constant features before scaling (only numeric features)
        constant_features = []
        for i, feat_name in enumerate(keep_all):
            if feat_name in numeric_features:
                if np.std(X_all[:, i]) < 1e-10:
                    constant_features.append(feat_name)
        if constant_features:
            print(
                f"   ⚠️  WARNING: Found {len(constant_features)} constant features before scaling: {constant_features[:5]}"
            )

        # Standardize only numeric features (not categorical)
        scaler_all = StandardScaler()
        if len(numeric_features) > 0:
            X_numeric_scaled = scaler_all.fit_transform(df_numeric.values)
            # Combine scaled numeric features with categorical features
            if df_categorical is not None and len(cat_features_in_keep) > 0:
                X_categorical = df_categorical.values
                # Reconstruct X_all_scaled with proper column order
                X_all_scaled_raw = np.zeros_like(X_all)
                numeric_idx = 0
                cat_idx = 0
                for i, feat_name in enumerate(keep_all):
                    if feat_name in numeric_features:
                        X_all_scaled_raw[:, i] = X_numeric_scaled[:, numeric_idx]
                        numeric_idx += 1
                    elif feat_name in cat_features_in_keep:
                        X_all_scaled_raw[:, i] = X_categorical[:, cat_idx]
                        cat_idx += 1
            else:
                X_all_scaled_raw = X_numeric_scaled
        else:
            X_all_scaled_raw = X_all

        # Check for NaN/Inf after scaling
        nan_count = np.isnan(X_all_scaled_raw).sum()
        inf_count = np.isinf(X_all_scaled_raw).sum()
        if nan_count > 0 or inf_count > 0:
            print(
                f"   ⚠️  WARNING: Found {nan_count} NaN and {inf_count} Inf values after scaling!"
            )

        X_all_scaled = sanitize_features(X_all_scaled_raw)

        # Check for constant features after sanitization
        constant_features_after = []
        for i in range(X_all_scaled.shape[1]):
            if np.std(X_all_scaled[:, i]) < 1e-10:
                constant_features_after.append(
                    keep_all[i] if i < len(keep_all) else f"feature_{i}"
                )
        if constant_features_after:
            print(
                f"   ⚠️  WARNING: Found {len(constant_features_after)} constant features after sanitization: {constant_features_after[:5]}"
            )
        print(
            f"[DEBUG] Stage 1: {len(keep_all)} features after missing/stability filter"
        )

        # Stage 2: IC (Spearman) ranking - top features by |IC|
        # Use rank-based method for multi-asset scenarios to avoid scale issues
        print(f"\n[Stage 2] IC ranking (rank-based for multi-asset)...")

        # Check if we have symbol information for rank-based calculation
        has_symbol_info = "_symbol" in df_features_original.columns
        is_multi_asset = (
            has_symbol_info and df_features_original["_symbol"].nunique() > 1
        )

        if is_multi_asset:
            print(
                f"   Using rank-based IC calculation across {df_features_original['_symbol'].nunique()} assets"
            )
            # Rank-based method: rank within each asset, then compute IC on merged ranks
            ic_scores = {}

            # Get symbol column aligned with df_all
            # We need to align symbol info with df_all indices
            # df_all is created from dfX which comes from X_raw, so we need to trace back
            # Try to get symbol from df_features_original, aligned by index
            try:
                # Align symbol info with df_all indices
                # df_all index should match dfX index, which should match df_features_original index
                symbol_series = df_features_original["_symbol"].reindex(df_all.index)
                # If reindex fails (different indices), try to match by position
                if symbol_series.isna().all() and len(df_features_original) == len(
                    df_all
                ):
                    symbol_series = pd.Series(
                        df_features_original["_symbol"].values, index=df_all.index
                    )
            except Exception:
                # Fallback: if we can't align, use original method
                print(
                    f"   ⚠️ Could not align symbol info, falling back to standard IC calculation"
                )
                symbol_series = None

            if symbol_series is not None and not symbol_series.isna().all():
                for col in df_all.columns:
                    try:
                        # Group by symbol and rank within each group
                        df_ranked = df_all[[col]].copy()
                        df_ranked["_symbol"] = symbol_series.values
                        df_ranked["_y"] = y_series.values

                        # Rank within each asset
                        df_ranked["_feature_rank"] = df_ranked.groupby("_symbol")[
                            col
                        ].rank(method="average")
                        df_ranked["_y_rank"] = df_ranked.groupby("_symbol")["_y"].rank(
                            method="average"
                        )

                        # Compute IC on ranked data (which is already rank-based, so this is consistent)
                        ic = spearmanr(
                            df_ranked["_feature_rank"].values,
                            df_ranked["_y_rank"].values,
                            nan_policy="omit",
                        )[0]
                    except Exception as e:
                        # Fallback to original method if rank-based fails
                        ic = spearmanr(
                            df_all[col].values, y_series.values, nan_policy="omit"
                        )[0]
                        if ic is None or np.isnan(ic):
                            ic = 0.0
                    ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
            else:
                # Fallback to original method if symbol alignment failed
                print(f"   ⚠️ Symbol alignment failed, using standard IC calculation")
                for col in df_all.columns:
                    try:
                        ic = spearmanr(
                            df_all[col].values, y_series.values, nan_policy="omit"
                        )[0]
                    except Exception:
                        ic = 0.0
                    ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        else:
            # Single asset or no symbol info: use original method
            print(f"   Using standard IC calculation (single asset or no symbol info)")
            ic_scores = {}
            for col in df_all.columns:
                try:
                    ic = spearmanr(
                        df_all[col].values, y_series.values, nan_policy="omit"
                    )[0]
                except Exception:
                    ic = 0.0
                ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        top_sorted = sorted(ic_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        # Determine target factor count with intelligent scaling
        # Goal: Reduce dimensionality while keeping important features
        total_features = len(top_sorted)
        if total_features > 200:
            # For large feature sets (>200), keep top 60-80
            target_top_k = min(80, total_features)
        elif total_features > 100:
            # For medium feature sets (100-200), keep top 40-60
            target_top_k = min(60, total_features)
        elif total_features > 50:
            # For medium-small feature sets (50-100), keep top 30-40
            target_top_k = min(40, total_features)
        elif total_features > 20:
            # For small feature sets (20-50), keep top 15-25
            target_top_k = min(25, total_features)
        else:
            # For very small feature sets (<20), keep most but still reduce if possible
            target_top_k = max(10, int(total_features * 0.7))

        print(
            f"   📊 Target factor count: {target_top_k} (from {total_features} total features, reduction ratio: {target_top_k/total_features:.1%})"
        )

        ic_top_k = min(max(target_top_k, 1), len(top_sorted))
        if ic_top_k == 0:
            ic_top_k = min(60, len(top_sorted))

        print(
            f"   ✅ Selected top {ic_top_k} features by IC (from {len(top_sorted)} total)"
        )

        # Initial selection by IC
        top_cols_initial = [c for c, _ in top_sorted[:ic_top_k]]

        # Diversity check and rebalancing
        def infer_feature_type(feature_name: str) -> str:
            """Infer feature type from feature name."""
            name_lower = feature_name.lower()
            if "alpha101" in name_lower:
                return "alpha101"
            elif "hurst" in name_lower:
                return "hurst"
            elif "wpt" in name_lower or "wavelet" in name_lower:
                return "wavelet"
            elif "hilbert" in name_lower:
                return "hilbert"
            elif "spectral" in name_lower:
                return "spectral"
            elif (
                "cvd" in name_lower
                or "ofi" in name_lower
                or "order_flow" in name_lower
                or "taker_buy" in name_lower
            ):
                return "order_flow"
            elif (
                "baseline" in name_lower
                or "sr_" in name_lower
                or "compressed" in name_lower
            ):
                return "baseline"
            elif (
                "rsi" in name_lower
                or "macd" in name_lower
                or "bb_" in name_lower
                or "atr" in name_lower
                or "ema" in name_lower
                or "sma" in name_lower
            ):
                return "technical"
            else:
                return "other"

        # Calculate feature type distribution
        feature_type_counts = {}
        for col in top_cols_initial:
            feat_type = infer_feature_type(col)
            feature_type_counts[feat_type] = feature_type_counts.get(feat_type, 0) + 1

        total_selected = len(top_cols_initial)
        max_type_ratio = (
            max(feature_type_counts.values()) / total_selected
            if total_selected > 0
            else 0
        )
        diversity_threshold = 0.6  # If any type > 60%, rebalance

        print(
            f"   Feature type distribution (initial): {dict(sorted(feature_type_counts.items(), key=lambda x: x[1], reverse=True))}"
        )
        print(f"   Max type ratio: {max_type_ratio:.2%}")

        # Rebalance if needed
        if max_type_ratio > diversity_threshold and total_selected > 20:
            print(
                f"   ⚠️  Feature type imbalance detected (max ratio: {max_type_ratio:.2%} > {diversity_threshold:.0%})"
            )
            print(f"   Rebalancing features to ensure diversity...")

            # Group features by type
            features_by_type = {}
            for col, ic_val in top_sorted:
                feat_type = infer_feature_type(col)
                if feat_type not in features_by_type:
                    features_by_type[feat_type] = []
                features_by_type[feat_type].append((col, ic_val))

            # Calculate target counts per type (ensure minimum representation)
            # Strategy: allocate based on available features, but cap max per type
            type_counts_available = {
                ft: len(features) for ft, features in features_by_type.items()
            }
            total_available = sum(type_counts_available.values())

            # Minimum quota per type (if available)
            min_quota_per_type = max(
                1, int(target_top_k * 0.05)
            )  # At least 5% per type
            max_quota_per_type = int(target_top_k * 0.4)  # At most 40% per type

            # Allocate quotas
            type_quotas = {}
            remaining_quota = target_top_k

            # First pass: allocate minimum quotas
            for feat_type in features_by_type.keys():
                available = type_counts_available[feat_type]
                quota = min(min_quota_per_type, available, remaining_quota)
                if quota > 0:
                    type_quotas[feat_type] = quota
                    remaining_quota -= quota

            # Second pass: allocate remaining quota proportionally (but cap at max)
            if remaining_quota > 0:
                for feat_type in sorted(
                    features_by_type.keys(),
                    key=lambda x: len(features_by_type[x]),
                    reverse=True,
                ):
                    if remaining_quota <= 0:
                        break
                    current_quota = type_quotas.get(feat_type, 0)
                    available = type_counts_available[feat_type]
                    additional = min(
                        max_quota_per_type - current_quota,
                        available - current_quota,
                        remaining_quota,
                    )
                    if additional > 0:
                        type_quotas[feat_type] = current_quota + additional
                        remaining_quota -= additional

            # Select features based on quotas
            top_cols = []
            for feat_type, quota in sorted(
                type_quotas.items(), key=lambda x: x[1], reverse=True
            ):
                if feat_type in features_by_type:
                    selected = [col for col, _ in features_by_type[feat_type][:quota]]
                    top_cols.extend(selected)
                    print(
                        f"      {feat_type}: {len(selected)}/{quota} features selected"
                    )

            # If we have less than target, fill with remaining top IC features
            if len(top_cols) < target_top_k:
                remaining_features = [
                    (col, ic) for col, ic in top_sorted if col not in top_cols
                ]
                needed = target_top_k - len(top_cols)
                top_cols.extend([col for col, _ in remaining_features[:needed]])

            # Recalculate distribution
            feature_type_counts_rebalanced = {}
            for col in top_cols:
                feat_type = infer_feature_type(col)
                feature_type_counts_rebalanced[feat_type] = (
                    feature_type_counts_rebalanced.get(feat_type, 0) + 1
                )

            print(
                f"   Feature type distribution (rebalanced): {dict(sorted(feature_type_counts_rebalanced.items(), key=lambda x: x[1], reverse=True))}"
            )
            print(f"   Total features selected: {len(top_cols)}")
        else:
            top_cols = top_cols_initial
            print(
                f"   ✅ Feature diversity is balanced (max ratio: {max_type_ratio:.2%} <= {diversity_threshold:.0%})"
            )
        df_ic = df_all[top_cols].copy()
        X_ic = df_ic.values
        scaler_ic = StandardScaler()
        X_ic_scaled = sanitize_features(scaler_ic.fit_transform(X_ic))
        print(
            f"[DEBUG] Stage 2: {len(top_cols)} features after IC ranking (target={target_top_k})"
        )

        # Calculate IC statistics for selected factors (for ICIR calculation)
        # Note: We'll calculate this after representative selection (Stage 3) to use the final factor set
        # For now, calculate based on top_cols, but we'll recalculate after reps are selected
        selected_ic_values = [
            ic_scores.get(col, 0.0) for col in top_cols if col in ic_scores
        ]
        ic_mean = (
            np.mean([abs(ic) for ic in selected_ic_values])
            if selected_ic_values
            else None
        )
        ic_std = (
            np.std([abs(ic) for ic in selected_ic_values])
            if selected_ic_values and len(selected_ic_values) > 1
            else None
        )
        if ic_mean is not None and ic_std is not None:
            icir = ic_mean / ic_std if ic_std > 0 else None
            print(
                f"   IC Statistics for IC-filtered factors: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}, ICIR={icir:.3f}"
                if icir
                else f"   IC Statistics: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}"
            )

        # Stability validation (if enabled)
        stability_validation_results = None
        if args.enable_stability_validation and args.train_start:
            print(f"\n{'=' * 80}")
            print(
                "🔍 Stability Validation: Validating selected factors on longer historical data"
            )
            print(f"{'=' * 80}")

            # Calculate validation period
            try:
                train_start_dt = pd.to_datetime(args.train_start)
                if args.validation_start:
                    validation_start_dt = pd.to_datetime(args.validation_start)
                else:
                    # Auto-calculate: go back validation_years from train_start
                    validation_start_dt = train_start_dt - pd.DateOffset(
                        years=args.validation_years
                    )

                validation_start_str = validation_start_dt.strftime("%Y-%m-%d")
                validation_end_str = args.train_start  # Validate up to training start

                print(
                    f"   Factor Selection Period: {args.train_start} → {args.train_end}"
                )
                print(
                    f"   Stability Validation Period: {validation_start_str} → {validation_end_str}"
                )
                print(
                    f"   This validates if factors selected on recent data are stable over longer history"
                )

                # Load validation data
                X_val_raw, y_val_raw, feature_names_val, _, df_features_val = (
                    load_real_market_data(
                        args.data_path,
                        args.symbol,
                        validation_start_str,
                        validation_end_str,
                        horizons=horizons_list,
                        feature_type=args.feature_type,
                        timeframe=args.timeframe,
                    )
                )

                if X_val_raw is not None and len(X_val_raw) > 0:
                    dfX_val = pd.DataFrame(
                        X_val_raw,
                        columns=feature_names_val,
                        index=df_features_val.index[: len(X_val_raw)],
                    )
                    y_series_val = pd.Series(
                        y_val_raw, index=dfX_val.index[: len(y_val_raw)]
                    )

                    # Calculate IC for selected factors on validation data
                    print(
                        f"\n   Calculating IC for {len(top_cols)} selected factors on validation data..."
                    )
                    ic_scores_validation = {}

                    # Check if validation data has symbol info for rank-based
                    has_symbol_val = "_symbol" in df_features_val.columns
                    is_multi_asset_val = (
                        has_symbol_val and df_features_val["_symbol"].nunique() > 1
                    )

                    for col in top_cols:
                        if col not in dfX_val.columns:
                            continue
                        try:
                            if is_multi_asset_val:
                                # Rank-based IC for validation
                                symbol_series_val = df_features_val["_symbol"].reindex(
                                    dfX_val.index
                                )
                                if (
                                    symbol_series_val is not None
                                    and not symbol_series_val.isna().all()
                                ):
                                    df_ranked_val = dfX_val[[col]].copy()
                                    df_ranked_val["_symbol"] = symbol_series_val.values
                                    df_ranked_val["_y"] = y_series_val.values
                                    df_ranked_val["_feature_rank"] = (
                                        df_ranked_val.groupby("_symbol")[col].rank(
                                            method="average"
                                        )
                                    )
                                    df_ranked_val["_y_rank"] = df_ranked_val.groupby(
                                        "_symbol"
                                    )["_y"].rank(method="average")
                                    ic = spearmanr(
                                        df_ranked_val["_feature_rank"].values,
                                        df_ranked_val["_y_rank"].values,
                                        nan_policy="omit",
                                    )[0]
                                else:
                                    ic = spearmanr(
                                        dfX_val[col].values,
                                        y_series_val.values,
                                        nan_policy="omit",
                                    )[0]
                            else:
                                ic = spearmanr(
                                    dfX_val[col].values,
                                    y_series_val.values,
                                    nan_policy="omit",
                                )[0]
                        except Exception:
                            ic = 0.0
                        ic_scores_validation[col] = (
                            0.0 if ic is None or np.isnan(ic) else ic
                        )

                    # Compare IC between selection period and validation period
                    ic_comparison = {}
                    stable_factors = []
                    unstable_factors = []

                    for col in top_cols:
                        if col in ic_scores and col in ic_scores_validation:
                            ic_selection = ic_scores[col]
                            ic_validation = ic_scores_validation[col]
                            ic_change = ic_validation - ic_selection
                            ic_stability = (
                                abs(ic_validation) / (abs(ic_selection) + 1e-8)
                                if abs(ic_selection) > 1e-8
                                else 0
                            )

                            ic_comparison[col] = {
                                "ic_selection": ic_selection,
                                "ic_validation": ic_validation,
                                "ic_change": ic_change,
                                "stability_ratio": ic_stability,
                            }

                            # Factor is stable if IC sign is consistent and magnitude is similar
                            if (
                                ic_selection * ic_validation > 0  # Same sign
                                and ic_stability > 0.5
                                and ic_stability < 2.0
                            ):  # Similar magnitude
                                stable_factors.append(col)
                            else:
                                unstable_factors.append(col)

                    stability_validation_results = {
                        "validation_period": {
                            "start": validation_start_str,
                            "end": validation_end_str,
                        },
                        "selection_period": {
                            "start": args.train_start,
                            "end": args.train_end,
                        },
                        "ic_comparison": ic_comparison,
                        "stable_factors": stable_factors,
                        "unstable_factors": unstable_factors,
                        "stability_rate": (
                            len(stable_factors) / len(top_cols) if top_cols else 0
                        ),
                    }

                    print(f"\n   ✅ Stability Validation Results:")
                    print(f"      Total factors tested: {len(top_cols)}")
                    print(
                        f"      Stable factors: {len(stable_factors)} ({stability_validation_results['stability_rate']:.1%})"
                    )
                    print(
                        f"      Unstable factors: {len(unstable_factors)} ({1 - stability_validation_results['stability_rate']:.1%})"
                    )

                    if len(stable_factors) > 0:
                        print(
                            f"\n   📊 Top 10 Stable Factors (IC consistent across periods):"
                        )
                        stable_sorted = sorted(
                            stable_factors,
                            key=lambda x: abs(ic_comparison[x]["ic_selection"]),
                            reverse=True,
                        )[:10]
                        for i, factor in enumerate(stable_sorted, 1):
                            comp = ic_comparison[factor]
                            print(
                                f"      {i}. {factor}: IC={comp['ic_selection']:.4f} → {comp['ic_validation']:.4f} (change: {comp['ic_change']:+.4f})"
                            )

                    if len(unstable_factors) > 0:
                        print(
                            f"\n   ⚠️  Top 5 Unstable Factors (IC changed significantly):"
                        )
                        unstable_sorted = sorted(
                            unstable_factors,
                            key=lambda x: abs(ic_comparison[x]["ic_change"]),
                            reverse=True,
                        )[:5]
                        for i, factor in enumerate(unstable_sorted, 1):
                            comp = ic_comparison[factor]
                            print(
                                f"      {i}. {factor}: IC={comp['ic_selection']:.4f} → {comp['ic_validation']:.4f} (change: {comp['ic_change']:+.4f})"
                            )
                else:
                    print(
                        f"   ⚠️  Could not load validation data, skipping stability validation"
                    )
            except Exception as exc:
                print(f"   ⚠️  Stability validation failed: {exc}")
                import traceback

                traceback.print_exc()

        # Stage 3: Correlation-based representative selection
        print(f"\n[Stage 3] Correlation-based representative selection...")
        # Missing and stability filter on IC-selected features
        keep_ic = []
        for c in df_ic.columns:
            s = df_ic[c]
            if s.isna().mean() < 0.2 and s.std() > 1e-8:
                keep_ic.append(c)
        df_ic_clean = df_ic[keep_ic].ffill().bfill().fillna(0.0)

        # Greedy representative selection by correlation threshold (0.9)
        # IMPORTANT: Select factors based on target_top_k FIRST, then apply correlation filtering
        # This ensures different factor counts select different factors
        # Stage 3 should further reduce dimensionality: target 60-70% of Stage 2 features
        if target_top_k and not df_ic_clean.empty:
            # Further reduce: target 60-70% of Stage 2 features, but at least 10 features
            stage3_target = max(10, int(target_top_k * 0.65))
            desired_reps = min(stage3_target, len(df_ic_clean.columns))
            print(
                f"   📊 Stage 3 target: {desired_reps} features (from {len(df_ic_clean.columns)} Stage 2 features, further reduction: {desired_reps/len(df_ic_clean.columns):.1%})"
            )
        else:
            desired_reps = None

        reps: list[str] = []
        if not df_ic_clean.empty:
            # First, select top N factors by IC score (where N = target_top_k)
            # This ensures we get different factors for different target_top_k values
            if desired_reps and desired_reps > 0:
                # Sort columns by IC score (absolute value) and take top N
                cols_with_ic = [
                    (col, abs(ic_scores.get(col, 0.0)))
                    for col in df_ic_clean.columns
                    if col in ic_scores
                ]
                cols_with_ic.sort(key=lambda x: x[1], reverse=True)
                top_ic_cols = [col for col, _ in cols_with_ic[:desired_reps]]

                # Then apply correlation filtering on the top IC factors
                corr = df_ic_clean[top_ic_cols].corr().abs().fillna(0.0)
                for c in top_ic_cols:
                    if all(corr.loc[c, r] < 0.9 for r in reps):
                        reps.append(c)

                # If correlation filtering removed too many, add back from top IC list
                if len(reps) < desired_reps:
                    additional = [c for c in top_ic_cols if c not in reps][
                        : max(desired_reps - len(reps), 0)
                    ]
                    reps.extend(additional)
            else:
                # Fallback: use original correlation-based selection
                corr = df_ic_clean.corr().abs().fillna(0.0)
                for c in df_ic_clean.columns:
                    if all(corr.loc[c, r] < 0.9 for r in reps):
                        reps.append(c)
                # Bound reps between 60 and 100 if no target specified
                if len(reps) < 60:
                    reps = list(df_ic_clean.columns)[:60]
                elif len(reps) > 100:
                    reps = reps[:100]
        if not reps:
            fallback_source = (
                df_ic_clean.columns if not df_ic_clean.empty else df_ic.columns
            )
            if len(fallback_source) == 0:
                fallback_source = df_all.columns
            reps = list(fallback_source)[: max(target_top_k or 60, 1)]
        df_reps = (
            df_ic_clean[reps]
            if set(reps).issubset(df_ic_clean.columns)
            else df_all[reps].fillna(0.0)
        )
        X_reps = df_reps.values

        # Pipeline validation: Inject synthetic strong signal BEFORE scaling
        # Reference: docs/时序模型/lightbgm shap=0.md
        # Improved: Match real label generation logic (including neutral zone filtering)
        # This ensures the synthetic test accurately reflects the real pipeline behavior
        signal_injected_feature_idx = None
        synthetic_labels_info = None
        # Store synthetic labels for pipeline validation (if enabled)
        synthetic_labels_for_training = None
        if getattr(args, "validate_pipeline", False) and len(reps) > 0:
            first_rep_feature_name = reps[0]
            if first_rep_feature_name in df_reps.columns:
                signal_injected_feature_idx = list(df_reps.columns).index(
                    first_rep_feature_name
                )

                # Step 1: Create synthetic future_return based on first feature
                # This simulates the real label generation process
                print(f"\n{'='*80}")
                print(
                    "🔍 Pipeline Validation: Creating synthetic signal matching real label generation"
                )
                print(f"{'='*80}")

                # Get the first feature as base signal
                first_feature_values = df_reps[first_rep_feature_name].values
                n_samples = len(first_feature_values)

                # Create synthetic future_return: use quantile-based approach to ensure balanced labels
                # This simulates a strong predictive signal with balanced Long/Short distribution
                np.random.seed(42)
                # Create a signal that will produce balanced labels after rank percentile
                # Use quantiles to ensure ~30% Long, ~30% Short, ~40% Hold
                feature_sorted = np.sort(first_feature_values)
                n_long = int(n_samples * 0.3)  # Top 30% = Long
                n_short = int(n_samples * 0.3)  # Bottom 30% = Short
                # Create synthetic return: high for top 30%, low for bottom 30%, medium for middle 40%
                synthetic_future_return = np.zeros(n_samples)
                # Bottom 30%: negative return (will be Short)
                bottom_threshold_idx = n_short
                synthetic_future_return[:bottom_threshold_idx] = (
                    -2.0 + np.random.randn(bottom_threshold_idx) * 0.1
                )
                # Top 30%: positive return (will be Long)
                top_threshold_idx = n_samples - n_long
                synthetic_future_return[top_threshold_idx:] = (
                    2.0 + np.random.randn(n_long) * 0.1
                )
                # Middle 40%: near zero (will be Hold)
                synthetic_future_return[bottom_threshold_idx:top_threshold_idx] = (
                    np.random.randn(top_threshold_idx - bottom_threshold_idx) * 0.1
                )
                # Shuffle to match original feature order (but keep quantile structure)
                # Actually, we need to map back to original indices based on feature values
                feature_argsort = np.argsort(first_feature_values)
                synthetic_future_return_ordered = np.zeros(n_samples)
                synthetic_future_return_ordered[feature_argsort] = (
                    synthetic_future_return
                )
                synthetic_future_return = synthetic_future_return_ordered

                # Step 2: Use REAL label generation function to create labels (including neutral zone filtering)
                # This ensures the synthetic test matches the real pipeline behavior
                # Create a temporary DataFrame with synthetic data
                synthetic_df = pd.DataFrame(
                    {
                        "close": np.ones(
                            n_samples
                        ),  # Dummy close prices (not used for rank percentile)
                    },
                    index=df_reps.index,
                )

                # Add synthetic future_return
                default_horizon = horizons_list[0] if horizons_list else 24
                synthetic_df[f"future_return_{default_horizon}"] = (
                    synthetic_future_return
                )

                # Detect timeframe from data frequency
                timeframe_minutes = 240  # Default: 240T (4h)
                # Check if index is datetime-like (DatetimeIndex or TimedeltaIndex)
                if len(df_reps) > 1 and isinstance(df_reps.index, pd.DatetimeIndex):
                    try:
                        time_diff = (
                            df_reps.index[1] - df_reps.index[0]
                        ).total_seconds() / 60
                        if 230 <= time_diff <= 250:
                            timeframe_minutes = 240
                        elif 55 <= time_diff <= 65:
                            timeframe_minutes = 60
                        elif 4 <= time_diff <= 6:
                            timeframe_minutes = 5
                        elif 14 <= time_diff <= 16:
                            timeframe_minutes = 15
                    except (AttributeError, TypeError):
                        # If index is not datetime-like, use default or parse from args
                        if hasattr(args, "timeframe") and args.timeframe:
                            # Parse timeframe string (e.g., "60T" -> 60)
                            timeframe_str = args.timeframe.rstrip("T")
                            try:
                                timeframe_minutes = int(timeframe_str)
                            except ValueError:
                                pass  # Keep default
                elif hasattr(args, "timeframe") and args.timeframe:
                    # Parse timeframe string (e.g., "60T" -> 60)
                    timeframe_str = args.timeframe.rstrip("T")
                    try:
                        timeframe_minutes = int(timeframe_str)
                    except ValueError:
                        pass  # Keep default

                # Calculate rank_window (same as real label generation)
                target_days = 12
                calculated_rank_window = int(
                    (target_days * 24 * 60) / timeframe_minutes
                )
                calculated_rank_window = max(calculated_rank_window, 30)
                calculated_rank_window = min(calculated_rank_window, 180)

                # Generate synthetic labels using REAL function (includes neutral zone filtering)
                synthetic_df = create_labels_multi_horizon(
                    synthetic_df,
                    horizons=[default_horizon],
                    use_rank_percentile=True,
                    rank_window=calculated_rank_window,
                    top_percentile=0.7,  # Top 30% = Long
                    bottom_percentile=0.3,  # Bottom 30% = Short
                    use_risk_adjusted=False,
                    use_quantile_threshold=False,
                )

                # Extract synthetic labels
                synthetic_signal_col = f"signal_{default_horizon}"
                if synthetic_signal_col in synthetic_df.columns:
                    synthetic_labels = (
                        synthetic_df[synthetic_signal_col]
                        .reindex(df_reps.index)
                        .fillna(0)
                        .values
                    )
                    valid_mask = (
                        synthetic_df[synthetic_signal_col]
                        .reindex(df_reps.index)
                        .notna()
                        .values
                    )
                    valid_samples = valid_mask.sum()
                    total_samples = len(valid_mask)

                    synthetic_labels_info = {
                        "total_samples": total_samples,
                        "valid_samples": valid_samples,
                        "valid_ratio": (
                            valid_samples / total_samples if total_samples > 0 else 0.0
                        ),
                        "label_distribution": (
                            dict(
                                zip(
                                    *np.unique(
                                        synthetic_labels[valid_mask], return_counts=True
                                    )
                                )
                            )
                            if valid_samples > 0
                            else {}
                        ),
                    }

                    print(f"   📊 Synthetic label generation (matching real pipeline):")
                    print(f"      Total samples: {total_samples}")
                    print(
                        f"      Valid samples (after neutral zone filtering): {valid_samples} ({synthetic_labels_info['valid_ratio']:.1%})"
                    )
                    if synthetic_labels_info["label_distribution"]:
                        print(
                            f"      Label distribution: {synthetic_labels_info['label_distribution']}"
                        )

                    # Check if we have enough valid samples
                    if valid_samples < 1000:
                        print(
                            f"      ⚠️  WARNING: Only {valid_samples} valid samples (minimum recommended: 1000)"
                        )
                        print(
                            f"      → This may indicate rank_window is too large or data period too short"
                        )
                    else:
                        print(
                            f"      ✅ Valid samples check passed: {valid_samples} >= 1000"
                        )

                    # Check label balance: if too imbalanced OR missing Short class, create balanced labels directly
                    label_dist = synthetic_labels_info["label_distribution"]
                    long_count = label_dist.get(1, 0)
                    short_count = label_dist.get(2, 0)
                    hold_count = label_dist.get(0, 0)
                    total_labeled = long_count + short_count + hold_count
                    long_ratio = long_count / total_labeled if total_labeled > 0 else 0
                    short_ratio = (
                        short_count / total_labeled if total_labeled > 0 else 0
                    )

                    # CRITICAL: For pipeline validation, ensure all 3 classes are present
                    # This is essential to test that the model can learn all classes
                    missing_short = short_count == 0
                    missing_long = long_count == 0
                    is_imbalanced = total_labeled > 0 and (
                        long_ratio > 0.9
                        or short_ratio > 0.9
                        or (long_count == 0 and short_count == 0)
                    )

                    # If labels are too imbalanced OR missing Short class, create balanced labels
                    if missing_short or missing_long or is_imbalanced:
                        if missing_short:
                            print(
                                f"      ⚠️  WARNING: Missing Short class (class 2) in synthetic labels!"
                            )
                            print(
                                f"      → This would cause model to never predict Short, failing pipeline validation"
                            )
                        elif is_imbalanced:
                            print(
                                f"      ⚠️  WARNING: Label distribution too imbalanced (Long={long_ratio:.1%}, Short={short_ratio:.1%})"
                            )
                        print(
                            f"      → Creating balanced synthetic labels directly (30% Long, 30% Short, 40% Hold)"
                        )
                        print(
                            f"      → This ensures model can learn all 3 classes and pass pipeline validation"
                        )
                        # Create balanced labels directly with guaranteed Short samples
                        np.random.seed(42)
                        n_long = int(n_samples * 0.3)
                        n_short = int(n_samples * 0.3)
                        n_hold = n_samples - n_long - n_short
                        balanced_labels = np.concatenate(
                            [
                                np.ones(n_long, dtype=int),  # Long (class 1)
                                np.full(n_hold, 0, dtype=int),  # Hold (class 0)
                                np.full(
                                    n_short, 2, dtype=int
                                ),  # Short (class 2) - CRITICAL for validation
                            ]
                        )
                        # Shuffle to avoid ordering bias
                        np.random.shuffle(balanced_labels)
                        synthetic_labels = balanced_labels
                        valid_mask = np.ones(n_samples, dtype=bool)
                        valid_samples = n_samples
                        print(
                            f"      ✅ Created balanced labels: Long={n_long} ({n_long/n_samples:.1%}), Short={n_short} ({n_short/n_samples:.1%}), Hold={n_hold} ({n_hold/n_samples:.1%})"
                        )
                        print(
                            f"      ✅ All 3 classes present: {set(balanced_labels)} - model should be able to predict all classes"
                        )

                    # Use synthetic labels for signal injection (only valid samples)
                    y_numeric = synthetic_labels.copy()
                    min_len = min(len(df_reps), len(y_numeric))
                    y_numeric = y_numeric[:min_len]

                    # Convert labels to numeric for signal injection
                    # For 3-class: 0=Hold, 1=Long, 2=Short
                    # Map to: Hold=0, Long=+1, Short=-1 (so Long/Short have opposite signals)
                    unique_labels = np.unique(
                        y_numeric[y_numeric != 0]
                    )  # Exclude Hold (0)
                    if len(unique_labels) >= 2 and np.any(
                        np.isin(unique_labels, [1, 2])
                    ):
                        # Multiclass: map 0->0 (Hold), 1->+1 (Long), 2->-1 (Short)
                        y_numeric = np.where(
                            y_numeric == 1, 1, np.where(y_numeric == 2, -1, 0)
                        )
                    elif len(unique_labels) == 1:
                        # Only one non-zero class, map to -1 and 1
                        if unique_labels[0] == 1:
                            y_numeric = np.where(y_numeric == 1, 1, 0)
                        else:
                            y_numeric = np.where(y_numeric == 2, -1, 0)
                    else:
                        # Fallback: use original labels
                        y_numeric = np.where(
                            y_numeric == 1, 1, np.where(y_numeric == 2, -1, 0)
                        )

                    # Inject VERY STRONG signal: 5.0x multiplier with minimal noise (0.05 std)
                    # Increased from 3.0 to 5.0 to ensure model can learn all classes, especially Short
                    signal_strength = 5.0
                    noise_std = 0.05
                    np.random.seed(42)
                    X_reps[:min_len, signal_injected_feature_idx] = (
                        y_numeric * signal_strength
                        + np.random.randn(min_len) * noise_std
                    )

                    # Verify signal distribution for each class
                    # Use synthetic_labels (before conversion) to check signal distribution
                    signal_by_class = {}
                    for class_label in [0, 1, 2]:
                        # synthetic_labels is in format: 0=Hold, 1=Long, 2=Short
                        class_mask = synthetic_labels[:min_len] == class_label
                        if class_mask.sum() > 0:
                            signal_by_class[class_label] = {
                                "mean": float(
                                    X_reps[
                                        class_mask, signal_injected_feature_idx
                                    ].mean()
                                ),
                                "std": float(
                                    X_reps[
                                        class_mask, signal_injected_feature_idx
                                    ].std()
                                ),
                                "count": int(class_mask.sum()),
                            }

                    print(
                        f"   ✅ Synthetic signal injected into '{first_rep_feature_name}'"
                    )
                    print(
                        f"      Signal formula: feature = synthetic_label * {signal_strength} + noise (std={noise_std})"
                    )
                    print(
                        f"      Signal strength: {signal_strength}x label + noise (std={noise_std})"
                    )
                    print(f"      Signal by class (before scaling):")
                    for class_label, stats in signal_by_class.items():
                        class_name = {0: "Hold", 1: "Long", 2: "Short"}.get(
                            class_label, f"Class_{class_label}"
                        )
                        print(
                            f"         {class_name} (class {class_label}): mean={stats['mean']:.3f}, std={stats['std']:.3f}, count={stats['count']}"
                        )
                    print(
                        f"      Expected: Model MUST learn this feature (importance > 0.1, AUC > 0.7)"
                    )
                    print(
                        f"      Expected: Model MUST predict all 3 classes (0=Hold, 1=Long, 2=Short)"
                    )
                    # Store synthetic labels for training (convert back to 0,1,2 format)
                    # y_numeric is currently: Long=+1, Short=-1, Hold=0
                    # Convert back to: Long=1, Short=2, Hold=0
                    synthetic_labels_for_training = np.where(
                        y_numeric == 1, 1, np.where(y_numeric == -1, 2, 0)
                    )
                    print(
                        f"      ✅ Synthetic labels stored for training (will replace original labels)"
                    )
                else:
                    print(
                        f"   ⚠️  Warning: Could not generate synthetic labels, using original labels"
                    )
                    # Fallback to original method
                    y_aligned = y_series.reindex(df_reps.index).fillna(0).values
                    min_len = min(len(df_reps), len(y_aligned))
                    y_aligned = y_aligned[:min_len]
                    y_numeric = y_aligned.copy()
                    unique_labels = np.unique(y_numeric)
                    if len(unique_labels) == 3 and np.all(
                        np.isin(unique_labels, [0, 1, 2])
                    ):
                        y_numeric = np.where(
                            y_numeric == 1, 1, np.where(y_numeric == 2, -1, 0)
                        )
                    elif len(unique_labels) == 2:
                        y_numeric = np.where(y_numeric == 0, -1, 1)
                    # Use same strong signal strength as main path
                    signal_strength = 5.0
                    noise_std = 0.05
                    np.random.seed(42)
                    X_reps[:min_len, signal_injected_feature_idx] = (
                        y_numeric * signal_strength
                        + np.random.randn(min_len) * noise_std
                    )

        scaler_reps = StandardScaler()
        X_reps_scaled = sanitize_features(scaler_reps.fit_transform(X_reps))
        print(
            f"[DEBUG] Stage 3: {len(reps)} representative features after correlation filtering"
        )

        # Recalculate IC statistics for final representative factors (for accurate ICIR)
        # This ensures IC statistics reflect the actual factors used in the model
        # CRITICAL: This must be calculated AFTER reps are selected, so different factor counts get different ICIR
        final_ic_values = [ic_scores.get(col, 0.0) for col in reps if col in ic_scores]
        if final_ic_values and len(final_ic_values) > 0:
            ic_mean = np.mean([abs(ic) for ic in final_ic_values])
            ic_std = (
                np.std([abs(ic) for ic in final_ic_values])
                if len(final_ic_values) > 1
                else 0.0
            )
            icir = ic_mean / ic_std if ic_std > 0 else None
            print(
                f"   IC Statistics for final representative factors ({len(reps)} factors, {len(final_ic_values)} with IC scores): Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}, ICIR={icir:.3f}"
                if icir
                else f"   IC Statistics for final factors: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}"
            )
        else:
            # Fallback to previous calculation if reps don't have IC scores
            print(
                f"   ⚠️  Warning: Could not calculate IC statistics for final factors ({len(reps)} factors), using IC-filtered factors"
            )
            # Keep the previous ic_mean, ic_std, icir from Stage 2 calculation
            # (they were calculated based on top_cols, which is less accurate but better than nothing)

        # Split data (same split for all stages - use consistent random state)
        # All stages should have the same number of samples, so we can use the same split
        n_samples = len(y_series.values)
        split_idx = int(n_samples * 0.7)
        split_idx2 = int(n_samples * 0.85)

        # Create same indices for all stages
        train_indices = np.arange(split_idx)
        val_indices = np.arange(split_idx, split_idx2)
        test_indices = np.arange(split_idx2, n_samples)

        # Split y
        # In pipeline validation mode, use synthetic labels if available
        if synthetic_labels_for_training is not None:
            print(f"\n   🔍 Pipeline Validation: Using synthetic labels for training")
            print(
                f"      Original labels will be replaced with synthetic balanced labels"
            )
            # Align synthetic labels with indices
            y_all = synthetic_labels_for_training[: len(y_series)]
            # Ensure same length
            if len(y_all) < len(y_series):
                # Pad with original labels if needed
                y_padded = np.full(len(y_series), 0, dtype=int)
                y_padded[: len(y_all)] = y_all
                y_all = y_padded
            elif len(y_all) > len(y_series):
                y_all = y_all[: len(y_series)]
        else:
            y_all = y_series.values
        y_train = y_all[train_indices]
        y_val = y_all[val_indices]
        y_test = y_all[test_indices]

        # Prepare categorical feature information for training
        # Find categorical feature names in the final feature list (reps)
        categorical_feature_names = (
            [c for c in reps if c in categorical_features]
            if "categorical_features" in locals() and categorical_features
            else None
        )
        if categorical_feature_names:
            print(
                f"   ✅ Categorical features in final model: {categorical_feature_names}"
            )

        # Stage 1: All features
        X_train_all = X_all_scaled[train_indices]
        X_val_all = X_all_scaled[val_indices]
        X_test_all = X_all_scaled[test_indices]

        # Stage 2: IC-filtered features
        X_train_ic = X_ic_scaled[train_indices]
        X_val_ic = X_ic_scaled[val_indices]
        X_test_ic = X_ic_scaled[test_indices]

        # Stage 3: Representative features
        X_train_reps = X_reps_scaled[train_indices]
        X_val_reps = X_reps_scaled[val_indices]
        X_test_reps = X_reps_scaled[test_indices]

        # Pipeline validation: Signal injection message (signal was injected before scaling, see line ~4273)
        if (
            getattr(args, "validate_pipeline", False)
            and len(reps) > 0
            and signal_injected_feature_idx is not None
        ):
            print(f"\n{'='*80}")
            print("🔍 Pipeline Validation: Synthetic signal injected BEFORE scaling")
            print(f"{'='*80}")
            first_rep_feature_name = reps[0]
            print(
                f"   ✅ Synthetic signal injected into '{first_rep_feature_name}' (index {signal_injected_feature_idx} in reps)"
            )
            print(f"   Signal formula: feature = y * 3.0 + noise (std=0.05)")
            print(f"   Signal strength: 3.0x label + noise (std=0.05)")
            print(
                f"   Expected: Model MUST learn this feature (importance > 0.1, AUC > 0.7)"
            )
            print(f"   If importance < 0.1 or AUC < 0.7 → Pipeline has a bug!")

        # Diagnostic: Check label distribution and feature-label correlation
        print(f"\n[DEBUG] Data split diagnostics:")
        # Use value_counts() to handle NaN values properly
        train_labels = pd.Series(y_train).value_counts().to_dict()
        val_labels = pd.Series(y_val).value_counts().to_dict()
        test_labels = pd.Series(y_test).value_counts().to_dict()
        print(f"   Train: {len(y_train)} samples, labels: {train_labels}")
        print(f"   Val: {len(y_val)} samples, labels: {val_labels}")
        print(f"   Test: {len(y_test)} samples, labels: {test_labels}")

        # Check if any feature perfectly predicts the label
        from scipy.stats import pointbiserialr

        perfect_predictors = []
        for i in range(min(20, X_train_all.shape[1])):  # Check first 20 features
            try:
                corr, pval = pointbiserialr(X_train_all[:, i], y_train)
                if abs(corr) > 0.99:  # Near-perfect correlation
                    feat_name = keep_all[i] if i < len(keep_all) else f"feature_{i}"
                    perfect_predictors.append((feat_name, corr, pval))
            except:
                pass

        if perfect_predictors:
            print(
                f"   ⚠️  WARNING: Found {len(perfect_predictors)} features with near-perfect correlation (>0.99) with labels!"
            )
            for feat_name, corr, pval in perfect_predictors[:5]:
                print(f"      - {feat_name}: corr={corr:.4f}, p={pval:.2e}")

        # Check if train and val labels are identical (would cause perfect fit)
        if len(y_train) == len(y_val) and np.array_equal(y_train, y_val):
            print(
                f"   ⚠️  WARNING: Train and validation labels are identical! This will cause perfect fit."
            )

        # Check if train and val features are identical
        if X_train_all.shape == X_val_all.shape and np.allclose(
            X_train_all, X_val_all, rtol=1e-10
        ):
            print(
                f"   ⚠️  WARNING: Train and validation features are identical! This will cause perfect fit."
            )

        # Check feature statistics
        print(f"\n[DEBUG] Feature statistics (first 5 features):")
        for i in range(min(5, X_train_all.shape[1])):
            feat_name = keep_all[i] if i < len(keep_all) else f"feature_{i}"
            feat_train = X_train_all[:, i]
            feat_val = X_val_all[:, i]
            print(f"   {feat_name}:")
            print(
                f"      Train: mean={feat_train.mean():.4f}, std={feat_train.std():.4f}, min={feat_train.min():.4f}, max={feat_train.max():.4f}"
            )
            print(
                f"      Val: mean={feat_val.mean():.4f}, std={feat_val.std():.4f}, min={feat_val.min():.4f}, max={feat_val.max():.4f}"
            )
            # Check for constant features
            if feat_train.std() < 1e-10:
                print(
                    f"      ⚠️  WARNING: Feature '{feat_name}' is constant in training set (std={feat_train.std():.2e})!"
                )
            if feat_val.std() < 1e-10:
                print(
                    f"      ⚠️  WARNING: Feature '{feat_name}' is constant in validation set (std={feat_val.std():.2e})!"
                )

        # Check if labels can be perfectly predicted by a simple rule
        # Try to find if any single feature or combination can perfectly separate classes
        print(f"\n[DEBUG] Checking for perfect separability:")
        # Convert y_train to numpy array and filter NaN values
        # Use pandas methods to handle Int64 type properly
        if isinstance(y_train, pd.Series):
            # Pandas Series - use to_numpy with na_value
            valid_mask = ~y_train.isna().values
            try:
                y_train_np = y_train.to_numpy(dtype=float, na_value=np.nan)
            except (ValueError, TypeError):
                # Fallback: fillna then convert
                y_train_np = y_train.fillna(-999).astype(float).values
                valid_mask = valid_mask & (y_train_np != -999)
        elif isinstance(y_train, np.ndarray):
            # Numpy array - use standard numpy methods
            y_train_np = np.asarray(y_train, dtype=float)
            valid_mask = ~np.isnan(y_train_np)
        else:
            # Other types - convert to Series first
            y_train_series = pd.Series(y_train)
            valid_mask = ~y_train_series.isna().values
            try:
                y_train_np = y_train_series.to_numpy(dtype=float, na_value=np.nan)
            except (ValueError, TypeError):
                y_train_np = y_train_series.fillna(-999).astype(float).values
                valid_mask = valid_mask & (y_train_np != -999)
        for i in range(min(10, X_train_all.shape[1])):  # Check first 10 features
            feat_train = X_train_all[:, i]
            # Check if feature values perfectly separate classes (only use valid labels)
            # Use valid_mask to filter out NaN values before comparison
            feat_0 = feat_train[valid_mask & (y_train_np == 0)]
            feat_1 = feat_train[valid_mask & (y_train_np == 1)]
            if len(feat_0) > 0 and len(feat_1) > 0:
                max_0 = feat_0.max()
                min_1 = feat_1.min()
                if max_0 < min_1 - 1e-10:  # Perfect separation
                    feat_name = keep_all[i] if i < len(keep_all) else f"feature_{i}"
                    print(
                        f"   ⚠️  WARNING: Feature '{feat_name}' perfectly separates classes!"
                    )
                    print(f"      Class 0: max={max_0:.4f}, Class 1: min={min_1:.4f}")
                    break

        # Multi-horizon training (if enabled) - will be done after all 4 stages
        multi_horizon_results = {}
        best_horizon = None
        best_horizon_metric = float("-inf")
        best_horizon_metric_name: Optional[str] = None
        fallback_horizon = None
        fallback_metric = float("-inf")
        fallback_metric_name: Optional[str] = None

        # Train and evaluate models for the selected stages
        print("\n" + "=" * 60)
        print("Training and evaluating feature sets (Stages 1-3)")
        print("=" * 60)

        # Prepare price data for backtest (if available)
        price_data_test = None
        if "close" in df_features_original.columns and len(df_features_original) > 0:
            # Get price data aligned with test indices
            # Ensure indices don't exceed df_features_original length
            max_idx = (
                min(len(df_features_original), test_indices.max() + 1)
                if len(test_indices) > 0
                else 0
            )
            valid_test_indices = test_indices[test_indices < len(df_features_original)]
            if len(valid_test_indices) > 0:
                price_data_test = (
                    df_features_original[["close"]].iloc[valid_test_indices].copy()
                )
                print(
                    f"  📊 Price data available for backtest: {len(price_data_test)} samples (from {len(df_features_original)} total)"
                )
            else:
                print(
                    f"  ⚠️  Warning: test_indices ({len(test_indices)}) exceed df_features_original length ({len(df_features_original)}), skipping price data"
                )

        # Stage 1: All features (482 -> ~470 after filtering)
        print("\n[Stage 1] Training on ALL features...")
        categorical_feature_names_all = (
            [c for c in keep_all if c in categorical_features]
            if "categorical_features" in locals() and categorical_features
            else None
        )
        model_all = train_production_lightgbm(
            X_train_all,
            y_train,
            X_val_all,
            y_val,
            feature_names=keep_all,
            categorical_features=categorical_feature_names_all,
        )
        perf_all = evaluate_model_performance(
            model_all, X_test_all, y_test, "All Features", price_data=price_data_test
        )

        # Stage 2: IC-filtered features (~120)
        print("\n[Stage 2] Training on IC-filtered features...")
        model_ic = train_production_lightgbm(X_train_ic, y_train, X_val_ic, y_val)
        perf_ic = evaluate_model_performance(
            model_ic,
            X_test_ic,
            y_test,
            "IC-Filtered Features",
            price_data=price_data_test,
        )

        # Stage 3: Representative features (60-100)
        print("\n[Stage 3] Training on Representative features...")
        categorical_feature_names_reps = (
            [c for c in reps if c in categorical_features]
            if "categorical_features" in locals() and categorical_features
            else None
        )
        model_reps = train_production_lightgbm(
            X_train_reps,
            y_train,
            X_val_reps,
            y_val,
            feature_names=reps,
            categorical_features=categorical_feature_names_reps,
        )

        # Pipeline validation: Enhanced check if model learned the synthetic signal
        # Multiple validation checks: importance, AUC, prediction variance, training status
        pipeline_validation_result = None
        if getattr(args, "validate_pipeline", False) and len(reps) > 0:
            try:
                # Import lgb at function level to avoid UnboundLocalError
                import lightgbm as lgb_module

                # Get feature importance from LightGBM booster
                # train_lightgbm_model returns a LightGBM Booster object directly
                if isinstance(model_reps, lgb_module.Booster):
                    importances = model_reps.feature_importance(importance_type="gain")
                    # Get model predictions for validation
                    predictions = model_reps.predict(X_train_reps)
                    # Get best iteration to check if model actually trained
                    best_iteration = (
                        model_reps.best_iteration
                        if hasattr(model_reps, "best_iteration")
                        else None
                    )
                elif hasattr(model_reps, "feature_importance"):
                    importances = model_reps.feature_importance(importance_type="gain")
                    predictions = (
                        model_reps.predict(X_train_reps)
                        if hasattr(model_reps, "predict")
                        else None
                    )
                    best_iteration = getattr(model_reps, "best_iteration", None)
                elif hasattr(model_reps, "model") and hasattr(
                    model_reps.model, "feature_importance"
                ):
                    importances = model_reps.model.feature_importance(
                        importance_type="gain"
                    )
                    predictions = (
                        model_reps.model.predict(X_train_reps)
                        if hasattr(model_reps.model, "predict")
                        else None
                    )
                    best_iteration = getattr(model_reps.model, "best_iteration", None)
                else:
                    # Try direct access
                    importances = model_reps.feature_importance(importance_type="gain")
                    predictions = (
                        model_reps.predict(X_train_reps)
                        if hasattr(model_reps, "predict")
                        else None
                    )
                    best_iteration = getattr(model_reps, "best_iteration", None)

                # Normalize importances to sum to 1
                if len(importances) > 0:
                    importances_normalized = importances / (importances.sum() + 1e-10)

                    # Find the first feature (should be the injected signal)
                    first_feature_idx = 0
                    if first_feature_idx < len(importances_normalized):
                        first_feature_importance = importances_normalized[
                            first_feature_idx
                        ]
                        first_feature_name = (
                            reps[first_feature_idx]
                            if first_feature_idx < len(reps)
                            else f"feature_{first_feature_idx}"
                        )

                        # Enhanced validation checks
                        validation_checks = {}
                        validation_passed = True

                        # Check 1: Feature importance
                        importance_passed = first_feature_importance > 0.1
                        validation_checks["feature_importance"] = {
                            "value": float(first_feature_importance),
                            "threshold": 0.1,
                            "passed": importance_passed,
                        }
                        if not importance_passed:
                            validation_passed = False

                        # Check 2: Prediction variance (model output should vary)
                        if predictions is not None:
                            if predictions.ndim == 2 and predictions.shape[1] > 1:
                                # Multiclass: use probability of positive class
                                pred_for_var = (
                                    predictions[:, 1]
                                    if predictions.shape[1] > 1
                                    else predictions[:, 0]
                                )
                            else:
                                pred_for_var = predictions.flatten()

                            pred_std = float(np.std(pred_for_var))
                            pred_mean = float(np.mean(pred_for_var))
                            pred_min = float(np.min(pred_for_var))
                            pred_max = float(np.max(pred_for_var))

                            variance_passed = pred_std > 1e-5
                            validation_checks["prediction_variance"] = {
                                "std": pred_std,
                                "mean": pred_mean,
                                "min": pred_min,
                                "max": pred_max,
                                "threshold": 1e-5,
                                "passed": variance_passed,
                            }
                            if not variance_passed:
                                validation_passed = False

                            # Check 3: AUC (for binary/multiclass classification)
                            # Signal injection: Hold=0, Long=+3, Short=-3
                            # Best AUC calculation: Long vs Short (exclude Hold) because they have opposite strong signals
                            try:
                                if predictions.ndim == 2 and predictions.shape[1] > 1:
                                    # Multiclass: use probability of Long class (class 1)
                                    # For 3-class: Long (1) vs Short (2), exclude Hold (0)
                                    if predictions.shape[1] >= 3:
                                        y_pred_proba = predictions[
                                            :, 1
                                        ]  # Probability of Long class
                                    else:
                                        y_pred_proba = (
                                            predictions[:, 1]
                                            if predictions.shape[1] > 1
                                            else predictions[:, 0]
                                        )
                                else:
                                    y_pred_proba = predictions.flatten()

                                # For multiclass labels (0, 1, 2), convert to binary for AUC
                                # Use Long (1) vs Short (2), exclude Hold (0)
                                # This matches signal injection: Long=+3 (high signal), Short=-3 (low signal)
                                # Convert y_train to numpy array and handle NaN
                                if isinstance(y_train, pd.Series):
                                    y_train_np = y_train.to_numpy(
                                        dtype=float, na_value=np.nan
                                    )
                                else:
                                    y_train_np = np.asarray(y_train, dtype=float)
                                # Filter out NaN values
                                valid_mask = ~np.isnan(y_train_np)
                                y_binary_mask = valid_mask & (
                                    (y_train_np == 1) | (y_train_np == 2)
                                )  # Only Long and Short
                                if y_binary_mask.sum() > 0:
                                    y_binary = (
                                        (y_train_np[y_binary_mask] == 1)
                                    ).astype(
                                        int
                                    )  # Long=1, Short=0
                                    y_pred_proba_filtered = y_pred_proba[y_binary_mask]

                                    if (
                                        len(np.unique(y_binary)) > 1
                                        and len(np.unique(y_pred_proba_filtered)) > 1
                                    ):
                                        auc = float(
                                            roc_auc_score(
                                                y_binary, y_pred_proba_filtered
                                            )
                                        )
                                        auc_passed = auc > 0.7
                                        validation_checks["auc"] = {
                                            "value": auc,
                                            "threshold": 0.7,
                                            "passed": auc_passed,
                                            "note": "Long vs Short (Hold excluded)",
                                        }
                                        if not auc_passed:
                                            validation_passed = False
                                    else:
                                        validation_checks["auc"] = {
                                            "value": None,
                                            "reason": "Insufficient label diversity for AUC calculation (Long vs Short)",
                                            "passed": None,
                                        }
                                else:
                                    validation_checks["auc"] = {
                                        "value": None,
                                        "reason": "No Long or Short samples found",
                                        "passed": None,
                                    }
                            except Exception as e:
                                validation_checks["auc"] = {
                                    "value": None,
                                    "error": str(e),
                                    "passed": None,
                                }
                        else:
                            validation_checks["prediction_variance"] = {
                                "error": "Could not get predictions from model",
                                "passed": False,
                            }
                            validation_passed = False

                        # Check 4: Model training status
                        # Note: If model achieves perfect performance (AUC=1.0) at iteration 1,
                        # this is normal and should not be considered a failure
                        if best_iteration is not None:
                            # Check if model achieved perfect or near-perfect performance
                            auc_value = validation_checks.get("auc", {}).get("value")
                            has_perfect_performance = (
                                auc_value is not None and auc_value >= 0.99
                            ) or (first_feature_importance > 0.9)

                            # If model has perfect performance, best_iteration=1 is acceptable
                            # Otherwise, require best_iteration > 1 to ensure model actually trained
                            if has_perfect_performance:
                                training_passed = (
                                    True  # Perfect performance at iteration 1 is OK
                                )
                                training_note = "Perfect performance achieved at iteration 1 (acceptable)"
                            else:
                                training_passed = best_iteration > 1
                                training_note = (
                                    "Model should train for more than 1 iteration"
                                )

                            validation_checks["model_training"] = {
                                "best_iteration": int(best_iteration),
                                "threshold": 1,
                                "passed": training_passed,
                                "note": (
                                    training_note if has_perfect_performance else None
                                ),
                            }
                            if not training_passed:
                                validation_passed = False
                        else:
                            validation_checks["model_training"] = {
                                "best_iteration": None,
                                "reason": "Could not determine best_iteration",
                                "passed": None,
                            }

                        # Print comprehensive validation results
                        print(f"\n   {'='*60}")
                        print(f"   🔍 Pipeline Validation Result (Enhanced):")
                        print(f"   {'='*60}")
                        print(
                            f"      First feature '{first_feature_name}' importance: {first_feature_importance:.4f}"
                        )

                        if "prediction_variance" in validation_checks:
                            var_check = validation_checks["prediction_variance"]
                            if "std" in var_check:
                                print(
                                    f"      Prediction std: {var_check['std']:.6f} (min={var_check['min']:.4f}, max={var_check['max']:.4f})"
                                )

                        if (
                            "auc" in validation_checks
                            and validation_checks["auc"].get("value") is not None
                        ):
                            print(f"      AUC: {validation_checks['auc']['value']:.4f}")

                        if (
                            "model_training" in validation_checks
                            and validation_checks["model_training"].get(
                                "best_iteration"
                            )
                            is not None
                        ):
                            print(
                                f"      Best iteration: {validation_checks['model_training']['best_iteration']}"
                            )

                        if validation_passed:
                            print(
                                f"      ✅ PASS: Model learned the synthetic signal (all checks passed)"
                            )
                            print(f"      → Pipeline is working correctly!")
                        else:
                            print(
                                f"      ❌ FAIL: Model did NOT learn the synthetic signal (one or more checks failed)"
                            )
                            print(f"      → Pipeline has a bug! Failed checks:")
                            for check_name, check_result in validation_checks.items():
                                if check_result.get("passed") is False:
                                    print(f"         - {check_name}: {check_result}")
                            print(f"      Diagnostic steps:")
                            print(
                                f"         1. Check feature scaling/normalization (may destroy signal)"
                            )
                            print(f"         2. Check data alignment (X vs y indices)")
                            print(
                                f"         3. Check model training parameters (learning_rate, num_iterations)"
                            )
                            print(
                                f"         4. Check feature filtering logic (first feature may be excluded)"
                            )
                            print(
                                f"         5. Check if first feature is actually in training data"
                            )
                            print(f"      ⚠️  This indicates a serious pipeline issue!")
                        print(f"   {'='*60}\n")

                        # Save comprehensive validation result
                        pipeline_validation_result = {
                            "enabled": True,
                            "status": "PASS" if validation_passed else "FAIL",
                            "first_feature_name": first_feature_name,
                            "first_feature_importance": float(first_feature_importance),
                            "threshold": 0.1,
                            "validation_checks": validation_checks,
                            "synthetic_labels_info": synthetic_labels_info,  # Include synthetic label generation info (matches real pipeline)
                            "message": (
                                "Model learned the synthetic signal (all checks passed)"
                                if validation_passed
                                else "Model did NOT learn the synthetic signal (one or more checks failed)"
                            ),
                            "diagnostics": (
                                {
                                    "check_feature_scaling": True,
                                    "check_data_alignment": True,
                                    "check_model_params": True,
                                    "check_feature_filtering": True,
                                    "check_first_feature_in_data": True,
                                    "check_label_generation": True,  # Added: check label generation (neutral zone filtering)
                                }
                                if not validation_passed
                                else None
                            ),
                        }
            except Exception as e:
                print(f"   ⚠️  Warning: Could not validate pipeline: {e}")
                import traceback

                traceback.print_exc()
                pipeline_validation_result = {
                    "enabled": True,
                    "status": "ERROR",
                    "error": str(e),
                }

        perf_reps = evaluate_model_performance(
            model_reps,
            X_test_reps,
            y_test,
            "Representative Features",
            price_data=price_data_test,
        )

        feature_insights_stage3 = _derive_feature_insights(perf_all, perf_reps)

        # Default best result to Stage 3 (representative features)
        best_model = model_reps
        best_result = {
            "timestamp_start": ablation_start_ts,
            "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "duration_sec": (datetime.now() - ablation_start_dt).total_seconds(),
            "data_info": {
                "stage1_all_features": int(len(keep_all)),
                "stage2_ic_filtered": int(len(top_cols)),
                "stage3_representatives": int(len(reps)),
                "original_features_count": int(original_feature_count),
                "compressed_dimensions": int(len(reps)),
                "compression_ratio": float(original_feature_count / max(len(reps), 1)),
                "training_samples": int(len(X_train_reps)),
                "validation_samples": int(len(X_val_reps)),
                "test_samples": int(len(X_test_reps)),
            },
            "performance": {
                "stage1_all": perf_all,
                "stage2_ic": perf_ic,
                "stage3_representatives": perf_reps,
                "stage3_representatives_financial": perf_reps.get(
                    "financial_metrics", {}
                ),
                "stage4_compressed": None,
                "selection_metric": args.selection_metric,
            },
            "insights": feature_insights_stage3,
            "ic_statistics": {
                "ic_mean": float(ic_mean) if ic_mean is not None else None,
                "ic_std": float(ic_std) if ic_std is not None else None,
                "icir": (
                    float(icir)
                    if ic_mean is not None and ic_std is not None and ic_std > 0
                    else None
                ),
            },
        }

        # Add stability validation results if available
        if stability_validation_results:
            best_result.setdefault("stability_validation", stability_validation_results)

        # Add pipeline validation results if available
        if pipeline_validation_result:
            best_result["pipeline_validation"] = pipeline_validation_result

        selection_score_stage1 = compute_selection_score(
            perf_all,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        selection_score_stage2 = compute_selection_score(
            perf_ic,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        selection_score_stage3 = compute_selection_score(
            perf_reps,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        delta_selection_stage3 = selection_score_stage3 - selection_score_stage1
        compression_ratio_stage3 = (
            float(original_feature_count) / float(len(reps)) if reps else None
        )

        best_result = {
            "timestamp_start": ablation_start_ts,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "task_type": "classification_binary",
            "data_info": {
                "stage1_all_features": int(len(keep_all)),
                "stage2_ic_filtered": int(len(top_cols)),
                "stage3_representatives": int(len(reps)),
                "original_features_count": int(original_feature_count),
                "compressed_dimensions": int(len(reps)),
                "compression_ratio": compression_ratio_stage3,
                "training_samples": int(len(X_train_reps)),
                "validation_samples": int(len(X_val_reps)),
                "test_samples": int(len(X_test_reps)),
            },
            "performance": {
                "stage1_all_features": perf_all,
                "stage2_ic_filtered": perf_ic,
                "stage3_representatives": perf_reps,
                "selection_metric": args.selection_metric,
                "selection_scores": {
                    "stage1": selection_score_stage1,
                    "stage2": selection_score_stage2,
                    "stage3": selection_score_stage3,
                    "delta_stage3_vs_stage1": delta_selection_stage3,
                    "delta_stage3_vs_stage2": selection_score_stage3
                    - selection_score_stage2,
                },
            },
            "training_info": {
                "lightgbm_stage1_iterations": getattr(
                    model_all, "best_iteration", None
                ),
                "lightgbm_stage2_iterations": getattr(model_ic, "best_iteration", None),
                "lightgbm_stage3_iterations": getattr(
                    model_reps, "best_iteration", None
                ),
            },
            "model_info": {
                "device_used": "cuda" if torch.cuda.is_available() else "cpu",
                "feature_names": reps[:10] if reps else feature_names[:10],
                "all_selected_features": (
                    reps if reps else feature_names[:10]
                ),  # Store all selected features
            },
            "selected_features": reps,  # Store the complete list of selected features
            "selection": {
                "metric": args.selection_metric,
                "best_stage": feature_insights_stage3["recommended_stage"],
            },
            "insights": feature_insights_stage3,
            "ic_statistics": {
                "ic_mean": float(ic_mean) if ic_mean is not None else None,
                "ic_std": float(ic_std) if ic_std is not None else None,
                "icir": (
                    float(icir)
                    if ic_mean is not None and ic_std is not None and ic_std > 0
                    else None
                ),
            },
        }

        # Add pipeline validation results if available (after best_result is created)
        if pipeline_validation_result:
            best_result["pipeline_validation"] = pipeline_validation_result

        # Add stability validation results if available
        if stability_validation_results:
            best_result["stability_validation"] = stability_validation_results

        best_model = model_reps
        best_dir = None

        # Multi-horizon training (if enabled) - train all 3 stages for each horizon
        if horizons and len(horizons) > 1 and not df_features_original.empty:
            print(f"\n{'=' * 80}")
            print(
                f"Multi-Horizon Training: Evaluating {len(horizons)} horizons across all 3 stages"
            )
            print(f"{'=' * 80}")

            # rank_window will be auto-calculated based on horizon (horizon * 20, min 100)
            # This ensures the ranking window is proportional to the prediction horizon

            df_multi_labels = create_labels_multi_horizon(
                df_features_original,
                horizons=horizons,
                use_rank_percentile=True,  # RECOMMENDED: Use rolling rank percentile
                rank_window=None,  # Auto-calculate based on horizon (horizon * 20, min 100)
                top_percentile=0.7,  # Top 30% = Long
                bottom_percentile=0.3,  # Bottom 30% = Short
                use_risk_adjusted=False,  # Disabled when using rank percentile
            )

            for horizon in horizons:
                print(f"\n{'=' * 60}")
                print(f"Training all 3 stages for Horizon: {horizon} bars")
                print(f"{'=' * 60}")

                # Get labels for this horizon (3-class: 0=Hold, 1=Long, 2=Short)
                y_horizon_col = f"signal_{horizon}"
                if y_horizon_col in df_multi_labels.columns:
                    y_horizon = df_multi_labels[y_horizon_col].values
                    y_horizon = y_horizon[: len(X_raw)]

                    # Use same split indices
                    y_train_h = y_horizon[train_indices]
                    y_val_h = y_horizon[val_indices]
                    y_test_h = y_horizon[test_indices]

                    # Stage 1: All features
                    print(
                        f"\n  [Stage 1] Horizon {horizon}: Training on ALL features..."
                    )
                    model_h_all = train_production_lightgbm(
                        X_train_all, y_train_h, X_val_all, y_val_h
                    )
                    perf_h_all = evaluate_model_performance(
                        model_h_all,
                        X_test_all,
                        y_test_h,
                        f"Horizon {horizon} - All Features",
                    )

                    # Stage 2: IC-filtered features
                    print(
                        f"\n  [Stage 2] Horizon {horizon}: Training on IC-filtered features..."
                    )
                    categorical_feature_names_ic_h = (
                        [c for c in top_cols if c in categorical_features]
                        if "categorical_features" in locals()
                        and categorical_features
                        and "top_cols" in locals()
                        else None
                    )
                    model_h_ic = train_production_lightgbm(
                        X_train_ic,
                        y_train_h,
                        X_val_ic,
                        y_val_h,
                        feature_names=top_cols if "top_cols" in locals() else None,
                        categorical_features=categorical_feature_names_ic_h,
                    )
                    perf_h_ic = evaluate_model_performance(
                        model_h_ic,
                        X_test_ic,
                        y_test_h,
                        f"Horizon {horizon} - IC-Filtered",
                    )

                    # Stage 3: Representative features
                    print(
                        f"\n  [Stage 3] Horizon {horizon}: Training on Representative features..."
                    )
                    categorical_feature_names_reps_h = (
                        [c for c in reps if c in categorical_features]
                        if "categorical_features" in locals()
                        and categorical_features
                        and "reps" in locals()
                        else None
                    )
                    model_h_reps = train_production_lightgbm(
                        X_train_reps,
                        y_train_h,
                        X_val_reps,
                        y_val_h,
                        feature_names=reps if "reps" in locals() else None,
                        categorical_features=categorical_feature_names_reps_h,
                    )

                    # Pipeline validation: Enhanced check if model learned the synthetic signal
                    # Multiple validation checks: importance, AUC, prediction variance, training status
                    pipeline_validation_result_h = None
                    if getattr(args, "validate_pipeline", False) and len(reps) > 0:
                        try:
                            # Get feature importance from LightGBM booster
                            if isinstance(model_h_reps, lgb.Booster):
                                importances = model_h_reps.feature_importance(
                                    importance_type="gain"
                                )
                                predictions = model_h_reps.predict(X_train_reps)
                                best_iteration = (
                                    model_h_reps.best_iteration
                                    if hasattr(model_h_reps, "best_iteration")
                                    else None
                                )
                            elif hasattr(model_h_reps, "feature_importance"):
                                importances = model_h_reps.feature_importance(
                                    importance_type="gain"
                                )
                                predictions = (
                                    model_h_reps.predict(X_train_reps)
                                    if hasattr(model_h_reps, "predict")
                                    else None
                                )
                                best_iteration = getattr(
                                    model_h_reps, "best_iteration", None
                                )
                            elif hasattr(model_h_reps, "model") and hasattr(
                                model_h_reps.model, "feature_importance"
                            ):
                                importances = model_h_reps.model.feature_importance(
                                    importance_type="gain"
                                )
                                predictions = (
                                    model_h_reps.model.predict(X_train_reps)
                                    if hasattr(model_h_reps.model, "predict")
                                    else None
                                )
                                best_iteration = getattr(
                                    model_h_reps.model, "best_iteration", None
                                )
                            else:
                                importances = model_h_reps.feature_importance(
                                    importance_type="gain"
                                )
                                predictions = (
                                    model_h_reps.predict(X_train_reps)
                                    if hasattr(model_h_reps, "predict")
                                    else None
                                )
                                best_iteration = getattr(
                                    model_h_reps, "best_iteration", None
                                )

                            # Normalize importances to sum to 1
                            if len(importances) > 0:
                                importances_normalized = importances / (
                                    importances.sum() + 1e-10
                                )

                                # Find the first feature (should be the injected signal)
                                first_feature_idx = 0
                                if first_feature_idx < len(importances_normalized):
                                    first_feature_importance = importances_normalized[
                                        first_feature_idx
                                    ]
                                    first_feature_name = (
                                        reps[first_feature_idx]
                                        if first_feature_idx < len(reps)
                                        else f"feature_{first_feature_idx}"
                                    )

                                    # Enhanced validation checks
                                    validation_checks_h = {}
                                    validation_passed_h = True

                                    # Check 1: Feature importance
                                    importance_passed_h = first_feature_importance > 0.1
                                    validation_checks_h["feature_importance"] = {
                                        "value": float(first_feature_importance),
                                        "threshold": 0.1,
                                        "passed": importance_passed_h,
                                    }
                                    if not importance_passed_h:
                                        validation_passed_h = False

                                    # Check 2: Prediction variance
                                    if predictions is not None:
                                        if (
                                            predictions.ndim == 2
                                            and predictions.shape[1] > 1
                                        ):
                                            pred_for_var = (
                                                predictions[:, 1]
                                                if predictions.shape[1] > 1
                                                else predictions[:, 0]
                                            )
                                        else:
                                            pred_for_var = predictions.flatten()

                                        pred_std = float(np.std(pred_for_var))
                                        pred_mean = float(np.mean(pred_for_var))
                                        pred_min = float(np.min(pred_for_var))
                                        pred_max = float(np.max(pred_for_var))

                                        variance_passed_h = pred_std > 1e-5
                                        validation_checks_h["prediction_variance"] = {
                                            "std": pred_std,
                                            "mean": pred_mean,
                                            "min": pred_min,
                                            "max": pred_max,
                                            "threshold": 1e-5,
                                            "passed": variance_passed_h,
                                        }
                                        if not variance_passed_h:
                                            validation_passed_h = False

                                        # Check 3: AUC
                                        # Signal injection: Hold=0, Long=+3, Short=-3
                                        # Best AUC calculation: Long vs Short (exclude Hold) because they have opposite strong signals
                                        try:
                                            if (
                                                predictions.ndim == 2
                                                and predictions.shape[1] > 1
                                            ):
                                                # Multiclass: use probability of Long class (class 1)
                                                # For 3-class: Long (1) vs Short (2), exclude Hold (0)
                                                if predictions.shape[1] >= 3:
                                                    y_pred_proba = predictions[
                                                        :, 1
                                                    ]  # Probability of Long class
                                                else:
                                                    y_pred_proba = (
                                                        predictions[:, 1]
                                                        if predictions.shape[1] > 1
                                                        else predictions[:, 0]
                                                    )
                                            else:
                                                y_pred_proba = predictions.flatten()

                                            # Use Long (1) vs Short (2), exclude Hold (0)
                                            # This matches signal injection: Long=+3 (high signal), Short=-3 (low signal)
                                            y_binary_mask_h = (y_train_h == 1) | (
                                                y_train_h == 2
                                            )  # Only Long and Short
                                            if y_binary_mask_h.sum() > 0:
                                                y_binary_h = (
                                                    y_train_h[y_binary_mask_h] == 1
                                                ).astype(
                                                    int
                                                )  # Long=1, Short=0
                                                y_pred_proba_filtered_h = y_pred_proba[
                                                    y_binary_mask_h
                                                ]

                                                if (
                                                    len(np.unique(y_binary_h)) > 1
                                                    and len(
                                                        np.unique(
                                                            y_pred_proba_filtered_h
                                                        )
                                                    )
                                                    > 1
                                                ):
                                                    auc = float(
                                                        roc_auc_score(
                                                            y_binary_h,
                                                            y_pred_proba_filtered_h,
                                                        )
                                                    )
                                                    auc_passed_h = auc > 0.7
                                                    validation_checks_h["auc"] = {
                                                        "value": auc,
                                                        "threshold": 0.7,
                                                        "passed": auc_passed_h,
                                                        "note": "Long vs Short (Hold excluded)",
                                                    }
                                                    if not auc_passed_h:
                                                        validation_passed_h = False
                                                else:
                                                    validation_checks_h["auc"] = {
                                                        "value": None,
                                                        "reason": "Insufficient label diversity for AUC calculation (Long vs Short)",
                                                        "passed": None,
                                                    }
                                            else:
                                                validation_checks_h["auc"] = {
                                                    "value": None,
                                                    "reason": "No Long or Short samples found",
                                                    "passed": None,
                                                }
                                        except Exception as e:
                                            validation_checks_h["auc"] = {
                                                "value": None,
                                                "error": str(e),
                                                "passed": None,
                                            }
                                    else:
                                        validation_checks_h["prediction_variance"] = {
                                            "error": "Could not get predictions from model",
                                            "passed": False,
                                        }
                                        validation_passed_h = False

                                    # Check 4: Model training status
                                    if best_iteration is not None:
                                        training_passed_h = best_iteration > 1
                                        validation_checks_h["model_training"] = {
                                            "best_iteration": int(best_iteration),
                                            "threshold": 1,
                                            "passed": training_passed_h,
                                        }
                                        if not training_passed_h:
                                            validation_passed_h = False
                                    else:
                                        validation_checks_h["model_training"] = {
                                            "best_iteration": None,
                                            "reason": "Could not determine best_iteration",
                                            "passed": None,
                                        }

                                    # Print comprehensive validation results
                                    print(f"\n   {'='*60}")
                                    print(
                                        f"   🔍 Pipeline Validation Result (Horizon {horizon}, Enhanced):"
                                    )
                                    print(f"   {'='*60}")
                                    print(
                                        f"      First feature '{first_feature_name}' importance: {first_feature_importance:.4f}"
                                    )

                                    if "prediction_variance" in validation_checks_h:
                                        var_check = validation_checks_h[
                                            "prediction_variance"
                                        ]
                                        if "std" in var_check:
                                            print(
                                                f"      Prediction std: {var_check['std']:.6f} (min={var_check['min']:.4f}, max={var_check['max']:.4f})"
                                            )

                                    if (
                                        "auc" in validation_checks_h
                                        and validation_checks_h["auc"].get("value")
                                        is not None
                                    ):
                                        print(
                                            f"      AUC: {validation_checks_h['auc']['value']:.4f}"
                                        )

                                    if (
                                        "model_training" in validation_checks_h
                                        and validation_checks_h["model_training"].get(
                                            "best_iteration"
                                        )
                                        is not None
                                    ):
                                        print(
                                            f"      Best iteration: {validation_checks_h['model_training']['best_iteration']}"
                                        )

                                    if validation_passed_h:
                                        print(
                                            f"      ✅ PASS: Model learned the synthetic signal (all checks passed)"
                                        )
                                        print(f"      → Pipeline is working correctly!")
                                    else:
                                        print(
                                            f"      ❌ FAIL: Model did NOT learn the synthetic signal (one or more checks failed)"
                                        )
                                        print(
                                            f"      → Pipeline has a bug! Failed checks:"
                                        )
                                        for (
                                            check_name,
                                            check_result,
                                        ) in validation_checks_h.items():
                                            if check_result.get("passed") is False:
                                                print(
                                                    f"         - {check_name}: {check_result}"
                                                )
                                        print(f"      Diagnostic steps:")
                                        print(
                                            f"         1. Check feature scaling/normalization (may destroy signal)"
                                        )
                                        print(
                                            f"         2. Check data alignment (X vs y indices)"
                                        )
                                        print(
                                            f"         3. Check model training parameters (learning_rate, num_iterations)"
                                        )
                                        print(
                                            f"         4. Check feature filtering logic (first feature may be excluded)"
                                        )
                                        print(
                                            f"         5. Check if first feature is actually in training data"
                                        )
                                        print(
                                            f"      ⚠️  This indicates a serious pipeline issue!"
                                        )
                                    print(f"   {'='*60}\n")

                                    # Save comprehensive validation result for this horizon
                                    pipeline_validation_result_h = {
                                        "horizon": int(horizon),
                                        "status": (
                                            "PASS" if validation_passed_h else "FAIL"
                                        ),
                                        "first_feature_name": first_feature_name,
                                        "first_feature_importance": float(
                                            first_feature_importance
                                        ),
                                        "threshold": 0.1,
                                        "validation_checks": validation_checks_h,
                                        "message": (
                                            "Model learned the synthetic signal (all checks passed)"
                                            if validation_passed_h
                                            else "Model did NOT learn the synthetic signal (one or more checks failed)"
                                        ),
                                    }
                        except Exception as e:
                            print(f"   ⚠️  Warning: Could not validate pipeline: {e}")
                            import traceback

                            traceback.print_exc()
                            pipeline_validation_result_h = {
                                "horizon": int(horizon),
                                "status": "ERROR",
                                "error": str(e),
                            }

                    perf_h_reps = evaluate_model_performance(
                        model_h_reps,
                        X_test_reps,
                        y_test_h,
                        f"Horizon {horizon} - Representatives",
                    )

                    # Store results for this horizon
                    horizon_perf = {
                        "stage1_all_features": perf_h_all,
                        "stage2_ic_filtered": perf_h_ic,
                        "stage3_representatives": perf_h_reps,
                    }
                    # Add pipeline validation result if available
                    if pipeline_validation_result_h:
                        horizon_perf["pipeline_validation"] = (
                            pipeline_validation_result_h
                        )
                    feature_insight_h = _derive_feature_insights(
                        perf_h_all, perf_h_reps
                    )
                    horizon_perf["feature_insights"] = feature_insight_h
                    metric_val_h = feature_insight_h.get("candidate_value")
                    metric_name_h = feature_insight_h.get("metric_name")
                    if (
                        feature_insight_h.get("effective")
                        and metric_val_h is not None
                        and metric_val_h > best_horizon_metric
                    ):
                        best_horizon_metric = float(metric_val_h)
                        best_horizon_metric_name = metric_name_h
                        best_horizon = horizon
                    if metric_val_h is not None and metric_val_h > fallback_metric:
                        fallback_metric = float(metric_val_h)
                        fallback_metric_name = metric_name_h
                        fallback_horizon = horizon
                    multi_horizon_results[f"horizon_{horizon}"] = horizon_perf

                    print(f"\n  ✅ Horizon {horizon} Complete:")
                    print(
                        f"     Stage 1 (All):      R²={perf_h_all['r2']:.4f}, RMSE={perf_h_all['rmse']:.6f}"
                    )
                    print(
                        f"     Stage 2 (IC):       R²={perf_h_ic['r2']:.4f}, RMSE={perf_h_ic['rmse']:.6f}"
                    )
                    print(
                        f"     Stage 3 (Reps):     R²={perf_h_reps['r2']:.4f}, RMSE={perf_h_reps['rmse']:.6f}"
                    )
                else:
                    print(
                        f"   ⚠️  Label column {y_horizon_col} not found for horizon {horizon}"
                    )

        # Add multi-horizon results to best_result
        if multi_horizon_results:
            best_result["multi_horizon_results"] = multi_horizon_results
            insights_ref = best_result.setdefault("insights", {})
            horizon_choice = best_horizon
            horizon_metric = best_horizon_metric
            horizon_metric_name = best_horizon_metric_name
            horizon_effective = True
            if horizon_choice is None and fallback_horizon is not None:
                horizon_choice = fallback_horizon
                horizon_metric = fallback_metric
                horizon_metric_name = fallback_metric_name
                horizon_effective = False
            if horizon_choice is not None:
                insights_ref.update(
                    {
                        "recommended_horizon": int(horizon_choice),
                        "recommended_horizon_metric": (
                            float(horizon_metric)
                            if horizon_metric is not None
                            else None
                        ),
                        "recommended_horizon_metric_name": horizon_metric_name,
                        "recommended_horizon_effective": horizon_effective,
                    }
                )

        # finalize end timestamp using actual ablation end
        ablation_end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # update results end and duration
        if best_result is not None:
            best_result["timestamp_end"] = ablation_end_ts
            try:
                start_dt_parsed = datetime.strptime(ablation_start_ts, "%Y%m%d_%H%M%S")
                duration_sec = (datetime.now() - start_dt_parsed).total_seconds()
                best_result["duration_sec"] = duration_sec
            except Exception:
                pass
            # rebuild dir using training date range if available, otherwise runtime timestamps
            if ablation_dir_date_suffix:
                DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                best_dir = str(DIM_COMPARE_RESULTS_ROOT / ablation_dir_date_suffix)
            else:
                # Fallback: use symbol, feature_type, and timestamps
                DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                best_dir = str(
                    DIM_COMPARE_RESULTS_ROOT
                    / f"{symbol_slug}_{feature_type_slug}_{best_result['timestamp_start']}_{best_result['timestamp_end']}"
                )
        os.makedirs(best_dir, exist_ok=True)

        # Save representative features list (Stage 3) - after best_dir is set
        if reps:
            # Filter out _symbol from factors list (it's a categorical identifier, not a factor)
            factors_only = [f for f in reps if f != "_symbol"]
            categorical_features_in_reps = [f for f in reps if f == "_symbol"]

            reps_path = os.path.join(best_dir, "representative_factors.json")
            with open(reps_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "representative_factors": factors_only,
                        "count": len(factors_only),
                        "categorical_features": (
                            categorical_features_in_reps
                            if categorical_features_in_reps
                            else []
                        ),
                        "stage": "Stage 3: Correlation-based representative selection",
                        "description": "Features selected by greedy correlation filtering (threshold=0.9). Note: _symbol is excluded as it's a categorical identifier, not a factor.",
                        "effective": feature_insights_stage3.get("effective", False),
                    },
                    f,
                    indent=2,
                )
            print(f"   💾 Representative factors saved to: {reps_path}")
            if categorical_features_in_reps:
                print(
                    f"      Note: {len(categorical_features_in_reps)} categorical identifier(s) excluded from factors: {categorical_features_in_reps}"
                )

            # Also save in top_factors format for compatibility with train_model
            top_factors_path = os.path.join(best_dir, "top_factors.json")
            with open(top_factors_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "top_factors": [{"name": factor} for factor in factors_only],
                        "count": len(factors_only),
                        "source": "ts-dim-compare",
                        "stage": "Stage 3: Representative features",
                        "effective": feature_insights_stage3.get("effective", False),
                        "note": (
                            "_symbol excluded (categorical identifier, not a factor)"
                            if categorical_features_in_reps
                            else None
                        ),
                    },
                    f,
                    indent=2,
                )
            print(f"   💾 Top factors (compatible format) saved to: {top_factors_path}")
            best_result.setdefault("data_info", {})["representatives_path"] = reps_path
            best_result["data_info"]["top_factors_path"] = top_factors_path

            shap_dir_path = None
            if args.shap_analysis:
                shap_dir_path = _generate_shap_outputs(
                    model_reps,
                    X_train_reps,
                    reps,
                    best_dir,
                    prefix="stage3_representatives",
                )
                if shap_dir_path:
                    best_result.setdefault("explainability", {})[
                        "stage3_shap_dir"
                    ] = shap_dir_path

        # Ensure JSON-serializable (e.g., convert any numpy types)
        def _to_py(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o

        # Enhanced _to_py function to handle more types and avoid circular references
        def _to_py_enhanced(o):
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Series):
                # Convert Series to dict, ensuring keys are basic types
                result = {}
                for k, v in o.items():
                    # Convert key to int if it's numpy integer
                    key = (
                        int(k)
                        if isinstance(k, (np.integer, np.int64, np.int32))
                        else (
                            float(k) if isinstance(k, (np.floating, np.float64)) else k
                        )
                    )
                    result[key] = _to_py_enhanced(v)
                return result
            if isinstance(o, pd.DataFrame):
                return o.to_dict("records")
            if isinstance(o, dict):
                # Remove any non-serializable values and convert keys to strings if needed
                result = {}
                for k, v in o.items():
                    if callable(v):
                        continue
                    # Convert key: handle numpy integer types first
                    if isinstance(
                        k, (np.integer, np.int64, np.int32, np.int16, np.int8)
                    ):
                        key = int(k)
                    elif isinstance(k, (np.floating, np.float64, np.float32)):
                        key = float(k)
                    elif isinstance(k, (str, int, float, bool, type(None))):
                        key = k
                    else:
                        # For other types, convert to string
                        key = str(k)
                    result[key] = _to_py_enhanced(v)
                return result
            if isinstance(o, (list, tuple)):
                return [_to_py_enhanced(item) for item in o]
            if hasattr(o, "__dict__"):
                # For objects, try to serialize their dict representation
                try:
                    return str(o)
                except:
                    return None
            return str(o) if not isinstance(o, (str, bool, type(None))) else o

        # Clean best_result before serialization to handle numpy types in keys
        def clean_dict_keys(obj):
            """Recursively clean dictionary keys to ensure they are JSON-serializable"""
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    # Convert key to basic type
                    if isinstance(
                        k, (np.integer, np.int64, np.int32, np.int16, np.int8)
                    ):
                        new_key = int(k)
                    elif isinstance(k, (np.floating, np.float64, np.float32)):
                        new_key = float(k)
                    elif isinstance(k, (str, int, float, bool, type(None))):
                        new_key = k
                    else:
                        new_key = str(k)
                    # Recursively clean value
                    cleaned[new_key] = clean_dict_keys(v)
                return cleaned
            elif isinstance(obj, (list, tuple)):
                return [clean_dict_keys(item) for item in obj]
            else:
                return obj

        # Clean best_result before serialization
        best_result_cleaned = clean_dict_keys(best_result)

        with open(f"{best_dir}/production_results.json", "w") as f:
            json.dump(best_result_cleaned, f, indent=2, default=_to_py_enhanced)

        # Generate report filename with symbol, feature_type, and time range
        def _format_date_for_filename(date_str):
            if not date_str:
                return ""
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_part = date_str.split("T")[0]
                        dt = datetime.strptime(date_part, "%Y-%m-%d")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                return ""
            except Exception:
                if isinstance(date_str, str) and len(date_str) >= 10:
                    try:
                        return date_str[:10].replace("-", "")
                    except:
                        return ""
                return ""

        train_start_str = (
            _format_date_for_filename(args.train_start) if args.train_start else ""
        )
        train_end_str = (
            _format_date_for_filename(args.train_end) if args.train_end else ""
        )

        # Build report filename
        if train_start_str and train_end_str:
            report_filename = f"{symbol_slug}_{feature_type_slug}_{train_start_str}_{train_end_str}_dimensionality_report.html"
        else:
            # Fallback to timestamps
            report_filename = f"{symbol_slug}_{feature_type_slug}_{ablation_start_ts}_dimensionality_report.html"

        default_report_path = os.path.join(best_dir, report_filename)
        write_html_report(best_result, default_report_path)
        print(f"📝 HTML report saved to: {default_report_path}")
        # optional export
        if args.export_model:
            try:
                os.makedirs(os.path.dirname(args.export_model), exist_ok=True)
                src_model = os.path.join(best_dir, "production_model.pkl")
                if os.path.exists(src_model):
                    import shutil as _sh

                    _sh.copy2(src_model, args.export_model)
                    print(f"💾 Exported best model to: {args.export_model}")
            except Exception as _exc:
                print(f"⚠️ Failed to export model: {_exc}")

        return best_result, best_model, best_dir

    # Run dimensionality comparison
    results, model, results_dir = run_dimensionality_comparison(
        data_path=args.data_path,
        symbol=args.symbol,
        train_start=args.train_start,
        train_end=args.train_end,
        feature_type=args.feature_type,
        shap_analysis=args.shap_analysis,
        timeframe=args.timeframe,
    )

    # Always write a report into the results directory with symbol, feature_type, and time range
    try:
        # Generate report filename with symbol, feature_type, and time range
        def _format_date_for_filename(date_str):
            if not date_str:
                return ""
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_part = date_str.split("T")[0]
                        dt = datetime.strptime(date_part, "%Y-%m-%d")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                return ""
            except Exception:
                if isinstance(date_str, str) and len(date_str) >= 10:
                    try:
                        return date_str[:10].replace("-", "")
                    except:
                        return ""
                return ""

        train_start_str = (
            _format_date_for_filename(args.train_start) if args.train_start else ""
        )
        train_end_str = (
            _format_date_for_filename(args.train_end) if args.train_end else ""
        )

        # Build report filename
        if train_start_str and train_end_str:
            report_filename = f"{symbol_slug}_{feature_type_slug}_{train_start_str}_{train_end_str}_dimensionality_report.html"
        else:
            # Fallback: extract from results_dir or use timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{symbol_slug}_{feature_type_slug}_{timestamp_str}_dimensionality_report.html"

        default_report_path = os.path.join(results_dir, report_filename)
        write_html_report(results, default_report_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to write default HTML report: {exc}")

    # Optionally write an extra copy to a user-specified path
    if args.report_html:
        try:
            write_html_report(results, args.report_html)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to write HTML report to {args.report_html}: {exc}")

    # Optional export in non-ablation paths
    if args.export_model:
        try:
            os.makedirs(os.path.dirname(args.export_model), exist_ok=True)
            src_model = os.path.join(results_dir, "production_model.pkl")
            if os.path.exists(src_model):
                import shutil as _sh

                _sh.copy2(src_model, args.export_model)
                print(f"💾 Exported best model to: {args.export_model}")
        except Exception as _exc:
            print(f"⚠️ Failed to export model: {_exc}")
    return results, model, results_dir


if __name__ == "__main__":
    try:
        results, model, results_dir = main()
        print("\n✅ Production training completed successfully!")
        cr = results.get("data_info", {}).get("compression_dim", None) or results.get(
            "data_info", {}
        ).get("compression_ratio", None)
        if cr is not None:
            try:
                print(f"📊 Final compression ratio: {float(cr):.1f}x")
            except Exception:
                pass
        pc = results.get("performance", {}).get("performance_change", None)
        if pc is not None:
            print(f"📈 Performance change: {pc:.4f}")
        print(f"💾 Results directory: {results_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Production training failed: {exc}")
        import traceback

        traceback.print_exc()
        raise
