"""
压缩区突破策略训练脚本

模型：CatBoost Classifier（三元分类）
标签：三元标签（-1, 0, +1）
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
from src.time_series_model.strategies.strategies.compression_breakout.features import (
    build_compression_breakout_features,
    select_compression_breakout_features,
)
from src.time_series_model.strategies.labels.compression_breakout_label import (
    compute_compression_breakout_label,
)
from src.time_series_model.strategies.models.strategy_trainer import (
    train_strategy_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train Compression Breakout Strategy Model"
    )
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--timeframe", type=str, default="15T")
    parser.add_argument("--feature-type", type=str, default="baseline,default")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", type=str, default="results/strategies/compression_breakout"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("💥 Compression Breakout Strategy Training")
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

    # Build compression breakout specific features
    print("\n🔧 Building compression breakout features...")
    df_train_features = build_compression_breakout_features(df_train_features)
    df_test_features = build_compression_breakout_features(df_test_features)

    feature_cols = engineer.get_feature_columns()
    compression_features = select_compression_breakout_features(
        df_train_features, feature_cols
    )
    print(f"   ✅ Selected {len(compression_features)} compression features")

    # Compute labels
    print("\n📝 Computing compression breakout labels...")
    df_train_features["label"] = compute_compression_breakout_label(
        df_train_features,
        lookback_window=10,
        confirmation_bars=3,
    )
    df_test_features["label"] = compute_compression_breakout_label(
        df_test_features,
        lookback_window=10,
        confirmation_bars=3,
    )

    # Filter valid samples (exclude 0 if needed, or keep all)
    df_train_valid = df_train_features[
        df_train_features["label"].notna()
        & df_train_features[compression_features].notna().all(axis=1)
    ].copy()
    df_test_valid = df_test_features[
        df_test_features["label"].notna()
        & df_test_features[compression_features].notna().all(axis=1)
    ].copy()

    # Map labels: -1 -> 0, 0 -> 1, +1 -> 2 (for multiclass)
    df_train_valid["label_class"] = df_train_valid["label"].map(
        {-1.0: 0, 0.0: 1, 1.0: 2}
    )
    df_test_valid["label_class"] = df_test_features["label"].map(
        {-1.0: 0, 0.0: 1, 1.0: 2}
    )

    print(f"   ✅ Train: {len(df_train_valid)} valid samples")
    print(f"   ✅ Test: {len(df_test_valid)} valid samples")
    label_dist = df_train_valid["label"].value_counts()
    print(f"   ✅ Label distribution: {dict(label_dist)}")

    if len(df_train_valid) < 100:
        print("   ⚠️  Warning: Too few training samples")
        return

    # Train model
    print("\n🚀 Training CatBoost Classifier...")
    models, avg_metric, cv_results, used_features = train_strategy_model(
        df_train_valid,
        feature_cols=compression_features,
        target_col="label_class",
        model_type="catboost",
        task_type="multiclass",  # Multiclass classification
        n_splits=5,
        tscv_gap=24,
    )

    print(f"   ✅ Average CV Metric: {avg_metric:.4f}")

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    import catboost as cb

    X_test = df_test_valid[compression_features].values
    y_test = df_test_valid["label_class"].values

    # Ensemble prediction
    pred_test_proba = np.zeros((len(X_test), 3))
    for model in models:
        pred_test_proba += model.predict_proba(X_test) / len(models)

    pred_test_class = np.argmax(pred_test_proba, axis=1)
    test_accuracy = (pred_test_class == y_test).mean()
    print(f"   ✅ Test Accuracy: {test_accuracy:.4f}")

    # Save results
    results = {
        "strategy": "compression_breakout",
        "model_type": "catboost",
        "task_type": "multiclass",
        "avg_cv_metric": float(avg_metric),
        "test_accuracy": float(test_accuracy),
        "n_features": len(used_features),
        "n_train_samples": len(df_train_valid),
        "n_test_samples": len(df_test_valid),
        "label_distribution": df_train_valid["label"].value_counts().to_dict(),
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
