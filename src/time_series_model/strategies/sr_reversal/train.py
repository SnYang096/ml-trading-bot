"""
SR 反转策略训练脚本

模型：XGBoost Classifier（二元分类）
标签：二元标签（≥2R 成功率）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools.data_utils import load_raw_data
from src.features.time_series.comprehensive_features import ComprehensiveFeatureEngineer
from src.time_series_model.strategies.sr_reversal.features import (
    build_sr_reversal_features,
    select_sr_reversal_features,
)
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label,
)
from src.time_series_model.strategies.models.strategy_trainer import (
    train_strategy_model,
)
from src.time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic


def main():
    parser = argparse.ArgumentParser(description="Train SR Reversal Strategy Model")
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--timeframe", type=str, default="15T")
    parser.add_argument("--feature-type", type=str, default="baseline,default")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", type=str, default="results/strategies/sr_reversal"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔄 SR Reversal Strategy Training")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    # Split train/test
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()

    print(f"   ✅ Train: {len(df_train_raw)} samples, Test: {len(df_test_raw)} samples")

    # Engineer features
    print("\n🔧 Engineering features...")
    engineer = ComprehensiveFeatureEngineer(feature_types=args.feature_type)
    df_train_features = engineer.engineer_all_features(df_train_raw, fit=True)
    df_test_features = engineer.engineer_all_features(df_test_raw, fit=False)

    # Build SR reversal specific features
    print("\n🔧 Building SR reversal features...")
    df_train_features = build_sr_reversal_features(df_train_features)
    df_test_features = build_sr_reversal_features(df_test_features)

    feature_cols = engineer.get_feature_columns()
    reversal_features = select_sr_reversal_features(df_train_features, feature_cols)
    print(f"   ✅ Selected {len(reversal_features)} reversal features")

    # Generate signal if not exists
    if "signal" not in df_train_features.columns:
        df_train_features["signal"] = 0
        df_test_features["signal"] = 0

    # Compute labels
    print("\n📝 Computing SR reversal labels...")
    df_train_features["label"] = compute_sr_reversal_label(
        df_train_features,
        signal_col="signal",
        hold_bars=args.horizon,
        rr_ratio=2.0,
    )
    df_test_features["label"] = compute_sr_reversal_label(
        df_test_features,
        signal_col="signal",
        hold_bars=args.horizon,
        rr_ratio=2.0,
    )

    # Filter valid samples
    df_train_valid = df_train_features[
        df_train_features["label"].notna()
        & df_train_features[reversal_features].notna().all(axis=1)
    ].copy()
    df_test_valid = df_test_features[
        df_test_features["label"].notna()
        & df_test_features[reversal_features].notna().all(axis=1)
    ].copy()

    print(f"   ✅ Train: {len(df_train_valid)} valid samples")
    print(f"   ✅ Test: {len(df_test_valid)} valid samples")
    print(f"   ✅ Success rate: {df_train_valid['label'].mean():.2%}")

    if len(df_train_valid) < 100:
        print("   ⚠️  Warning: Too few training samples")
        return

    # Train model
    print("\n🚀 Training XGBoost Classifier...")
    models, avg_metric, cv_results, used_features = train_strategy_model(
        df_train_valid,
        feature_cols=reversal_features,
        target_col="label",
        model_type="xgboost",
        task_type="binary",  # Binary classification
        n_splits=5,
        tscv_gap=24,
    )

    print(f"   ✅ Average CV Metric: {avg_metric:.4f}")

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    import xgboost as xgb

    X_test = df_test_valid[used_features].values
    y_test = df_test_valid["label"].values

    # Ensemble prediction
    pred_test = np.zeros(len(X_test))
    for model in models:
        dtest = xgb.DMatrix(X_test)
        pred_test += model.predict(dtest) / len(models)

    # Compute test metric
    test_metric = np.corrcoef(pred_test, y_test)[0, 1]
    print(f"   ✅ Test Metric: {test_metric:.4f}")

    # Save results
    results = {
        "strategy": "sr_reversal",
        "model_type": "xgboost",
        "task_type": "binary",
        "avg_cv_metric": float(avg_metric),
        "test_metric": float(test_metric),
        "n_features": len(used_features),
        "n_train_samples": len(df_train_valid),
        "n_test_samples": len(df_test_valid),
        "success_rate": float(df_train_valid["label"].mean()),
        "features": used_features,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
