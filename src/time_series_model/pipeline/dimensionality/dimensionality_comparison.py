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


# Core utilities (define locally to avoid import issues)
def sanitize_features(X: np.ndarray, clip_std: float = 5.0) -> np.ndarray:
    """Sanitize features by clipping extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_clipped = np.clip((X - mean) / std, -clip_std, clip_std)
    return X_clipped * std + mean


def _slugify(text: str) -> str:
    """Convert text to a slug (lowercase, alphanumeric + hyphens)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text


# Config-driven imports
import sys
from typing import List, Any
from importlib import import_module

# Add project root to path for config-driven imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_tools.data_utils import load_raw_data
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
    from src.strategy_config import StrategyConfigLoader
    from scripts.train_strategy_pipeline import (
        apply_filters,
        apply_post_label_filters,
        determine_feature_columns,
        import_callable,
        run_feature_pipeline,
    )

    CONFIG_DRIVEN_AVAILABLE = True
except ImportError:
    CONFIG_DRIVEN_AVAILABLE = False

DIM_COMPARE_RESULTS_ROOT = Path("results") / "dim_compare"


def train_lightgbm_model_simple(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> Any:
    """Train a LightGBM model for config-driven dim_compare."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    return model


def evaluate_model_simple(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model performance for config-driven dim_compare."""
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def run_dim_compare(
    config_dir: Path,
    symbol: str,
    data_path: str,
    timeframe: str,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run dimensionality comparison for a strategy config (config-driven mode).

    This function performs three-stage feature selection:
    1. Stage 1: Missing/stability filter
    2. Stage 2: IC ranking
    3. Stage 3: Correlation-based representative selection

    Outputs top_factors.json for use in rolling training.

    Args:
        config_dir: Path to strategy config directory
        symbol: Trading symbol (e.g., "BTCUSDT")
        data_path: Path to data directory
        timeframe: Data timeframe (e.g., "15T")
        train_start: Training start date (YYYY-MM-DD, optional)
        train_end: Training end date (YYYY-MM-DD, optional)

    Returns:
        Tuple of (results dictionary, top_factors.json path)
    """
    if not CONFIG_DRIVEN_AVAILABLE:
        raise ImportError(
            "Config-driven mode requires src.data_tools.data_utils, "
            "src.features.loader.strategy_feature_loader, and "
            "src.strategy_config modules"
        )

    print("🚀 Config-Driven Dimensionality Comparison")
    print("=" * 60)

    # Load strategy config
    loader = StrategyConfigLoader(config_dir)
    strategy_config = loader.load()
    print(f"📂 Strategy: {strategy_config.name}")

    # Load data
    print("\n1. Loading data...")
    df_raw = load_raw_data(
        data_path=data_path,
        symbol=symbol,
        timeframe=timeframe,
        start_date=train_start,
        end_date=train_end,
    )

    if df_raw.empty:
        raise ValueError(f"No data loaded for {symbol}")

    print(f"   ✅ Data loaded: {len(df_raw):,} bars")

    # Engineer features
    print("\n2. Engineering features...")
    feature_loader = StrategyFeatureLoader()
    df_features = run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )

    feature_cols = determine_feature_columns(df_features, strategy_config.features)
    print(f"   ✅ Features: {len(feature_cols)}")

    # Generate labels
    print("\n3. Generating labels...")
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )

    df_features[strategy_config.labels.target_column] = label_func(
        df_features.copy(), **strategy_config.labels.generator.params
    )

    # Apply filters
    df_filtered = apply_filters(df_features, strategy_config.labels.filters)
    df_filtered = apply_post_label_filters(
        df_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )

    print(f"   ✅ Valid samples: {len(df_filtered):,}")

    # Prepare features and target
    X = df_filtered[feature_cols].ffill().bfill().fillna(0.0)
    y = df_filtered[strategy_config.labels.target_column].values

    # Remove samples with NaN labels
    valid_mask = ~np.isnan(y) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"   ✅ Final samples: {len(X):,}")

    # Preprocessing
    print("\n4. Preprocessing...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X.values)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_scaled = sanitize_features(X_scaled, clip_std=5.0)
    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite values remain in features after sanitation")
    if not np.isfinite(y_scaled).all():
        raise ValueError("Non-finite values found in labels after scaling")

    dfX = pd.DataFrame(X_scaled, columns=feature_cols)
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
    stage3_target = max(10, int(target_top_k * 0.65))
    desired_reps = min(stage3_target, len(df_ic_clean.columns))

    reps = []
    if not df_ic_clean.empty:
        cols_with_ic = [
            (col, abs(ic_scores.get(col, 0.0))) for col in df_ic_clean.columns
        ]
        cols_with_ic.sort(key=lambda x: x[1], reverse=True)
        top_ic_cols = [col for col, _ in cols_with_ic[:desired_reps]]

        corr = df_ic_clean[top_ic_cols].corr().abs().fillna(0.0)
        for c in top_ic_cols:
            if all(corr.loc[c, r] < 0.9 for r in reps):
                reps.append(c)

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

    # Split data
    n_samples = len(X_all_scaled)
    split_idx = int(n_samples * 0.7)
    split_idx2 = int(n_samples * 0.85)

    train_indices = np.arange(split_idx)
    val_indices = np.arange(split_idx, split_idx2)
    test_indices = np.arange(split_idx2, n_samples)

    X_train_all = X_all_scaled[train_indices]
    X_val_all = X_all_scaled[val_indices]
    X_test_all = X_all_scaled[test_indices]

    X_train_reps = X_reps_scaled[train_indices]
    X_val_reps = X_reps_scaled[val_indices]
    X_test_reps = X_reps_scaled[test_indices]

    y_train = y_scaled[train_indices]
    y_val = y_scaled[val_indices]
    y_test = y_scaled[test_indices]

    print(
        f"\n✅ Data split: Train {X_train_all.shape}, Val {X_val_all.shape}, Test {X_test_all.shape}"
    )

    # Train models
    print("\n5. Training models...")
    print("   [Before Reduction] Training with all features...")
    model_all = train_lightgbm_model_simple(
        X_train_all, y_train, X_val_all, y_val, keep_all
    )
    perf_all = evaluate_model_simple(model_all, X_test_all, y_test)

    print("   [After Reduction] Training with representative features...")
    model_reps = train_lightgbm_model_simple(
        X_train_reps, y_train, X_val_reps, y_val, reps
    )
    perf_reps = evaluate_model_simple(model_reps, X_test_reps, y_test)

    compression_ratio = len(keep_all) / max(len(reps), 1)
    performance_change = perf_reps.get("r2", 0) - perf_all.get("r2", 0)

    print("\n📊 Performance Comparison:")
    print(
        f"   Before Reduction: R2={perf_all.get('r2', 0):.4f}, RMSE={perf_all.get('rmse', 0):.4f}"
    )
    print(
        f"   After Reduction: R2={perf_reps.get('r2', 0):.4f}, RMSE={perf_reps.get('rmse', 0):.4f}"
    )
    print(f"   Compression Ratio: {compression_ratio:.2f}x")
    print(f"   Performance Change: {performance_change:+.4f}")

    # Save results
    results_dir = (
        Path("results")
        / "dim_compare"
        / f"{strategy_config.name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate top_factors.json
    print("\n6. Generating top_factors.json...")
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

    top_factors_file = results_dir / "top_factors.json"
    with open(top_factors_file, "w", encoding="utf-8") as f:
        json.dump(top_factors_data, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Generated top_factors.json with {len(selected_features)} features")
    print(f"   📄 File location: {top_factors_file}")

    results = {
        "strategy": strategy_config.name,
        "symbol": symbol,
        "data_info": {
            "original_features_count": len(feature_cols),
            "stage1_all_features": len(keep_all),
            "stage2_ic_filtered": len(top_cols),
            "stage3_representatives": len(reps),
            "compression_ratio": compression_ratio,
            "training_samples": len(X_train_all),
            "validation_samples": len(X_val_all),
            "test_samples": len(X_test_all),
        },
        "performance": {
            "before_reduction": perf_all,
            "after_reduction": perf_reps,
            "performance_change": performance_change,
        },
        "top_factors_path": str(top_factors_file),
    }

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}")
    return results, str(top_factors_file)


def main() -> None:
    """Main entry point for config-driven dimensionality comparison.

    This script performs three-stage feature selection:
    1. Stage 1: Missing/stability filter
    2. Stage 2: IC ranking
    3. Stage 3: Correlation-based representative selection

    Outputs top_factors.json for use in rolling training.
    """
    parser = argparse.ArgumentParser(
        description="Config-driven dimensionality comparison and feature selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config directory (e.g., config/strategies/sr_reversal)",
    )
    parser.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Data timeframe (e.g., 15T)",
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
    args = parser.parse_args()

    config_dir = Path(args.config)
    if not (config_dir / "features.yaml").exists():
        print(f"❌ Config directory not found or invalid: {config_dir}")
        sys.exit(1)

    try:
        results, top_factors_path = run_dim_compare(
            config_dir=config_dir,
            symbol=args.symbol,
            data_path=args.data_path,
            timeframe=args.timeframe,
            train_start=args.train_start,
            train_end=args.train_end,
        )
        print(f"\n✅ Dimensionality comparison complete!")
        print(f"   Top factors saved to: {top_factors_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
