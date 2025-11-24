"""
独立训练压缩区突破模型。

核心思想：
- 专注压缩区突破逻辑，使用压缩和突破相关特征
- 只使用 compression_reaction == "compression_breakout" 的样本
- 预测是否实现 ≥2R 的压缩突破盈利
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools.data_utils import load_raw_data
from src.features.time_series.comprehensive_features import ComprehensiveFeatureEngineer
from src.time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
    generate_ensemble_signals,
    evaluate_model_performance,
)
from src.time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic
from src.time_series_model.pipeline.training.label_utils import (
    classify_compression_breakout,
)


def select_compression_breakout_features(
    df: pd.DataFrame, all_features: List[str]
) -> List[str]:
    """
    为压缩区突破模型选择特征。

    压缩区突破机会的特征侧重：
    - 压缩指标（Bollinger Band 宽度、ATR 相对值）
    - 突破动量（速度、强度）
    - 成交量放大（突破时成交量）
    - 价格位置（在压缩区内的位置）
    - 突破确认（是否站稳）
    """
    compression_keywords = [
        # 压缩相关
        "compression",
        "compression_score",
        "compression_confidence",
        "bb_width",
        "bb_position",
        "volatility",
        # 突破相关
        "breakout",
        "breakout_speed",
        "breakout_momentum",
        "breakout_strength",
        "follow_through",
        "momentum_persistence",
        # 成交量
        "volume",
        "vol_ratio",
        "volume_spike",
        "volume_confirmation",
        "order_flow",
        "cvd",
        "taker_buy_ratio",
        # 价格行为
        "price_action",
        "candlestick",
        # 趋势
        "trend",
        "trend_strength",
        # Default 特征（压缩区检测）
        "bbands",
        "bollinger",
        "keltner",
        "donchian",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in compression_keywords):
            selected.append(feat)
        # 保留通用特征
        elif any(keyword in feat_lower for keyword in ["atr", "volatility", "vol"]):
            selected.append(feat)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Train Compression Breakout Model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/parquet_data",
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
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
        help="Feature engineering type (baseline, default, enhanced, comprehensive, or comma-separated)",
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
        default="results/compression_breakout_model",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("💥 Compression Breakout Model Training")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Horizon: {args.horizon}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Feature Type: {args.feature_type}")
    print("=" * 60)

    # Load and prepare data
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

    feature_cols = engineer.get_feature_columns()
    print(f"   ✅ Generated {len(feature_cols)} features")

    # Generate initial signal if not exists
    if "signal" not in df_train_features.columns:
        if "breakout_status" in df_train_features.columns:
            df_train_features["signal"] = (
                df_train_features["breakout_status"].fillna(0).astype(int)
            )
            df_test_features["signal"] = (
                df_test_features["breakout_status"].fillna(0).astype(int)
            )
            print("   ✅ Generated signal from breakout_status")
        else:
            print("   ⚠️  Warning: No signal column found, creating dummy signal")
            df_train_features["signal"] = 0
            df_test_features["signal"] = 0

    # Classify compression breakout opportunities
    print(f"\n   📝 Classifying compression breakout opportunities...")
    df_train_features["compression_reaction"] = classify_compression_breakout(
        df_train_features,
        price_col="close",
        atr_col="atr",
    )

    compression_breakout_mask = (
        df_train_features["compression_reaction"] == "compression_breakout"
    )
    df_train_compression = df_train_features[compression_breakout_mask].copy()

    print(
        f"   ✅ Compression breakout samples: {len(df_train_compression)} (from {len(df_train_features)} total)"
    )

    if len(df_train_compression) < 100:
        print(
            f"   ⚠️  Warning: Too few compression breakout samples ({len(df_train_compression)})"
        )
        return

    # Select compression breakout-specific features
    compression_features = select_compression_breakout_features(
        df_train_compression, feature_cols
    )
    print(
        f"\n   📊 Selected {len(compression_features)} compression breakout features (from {len(feature_cols)} total)"
    )

    # Prepare labels with R/R
    print(f"\n   📝 Preparing labels for compression breakout model...")
    df_train_labeled = prepare_rank_ic_labels(
        df_train_compression.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_train_compression.columns else None,
        date_col="date" if "date" in df_train_compression.columns else None,
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=args.horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=False,  # Compression breakout doesn't need reaction split
    )

    # Prepare test set
    df_test_features["compression_reaction"] = classify_compression_breakout(
        df_test_features,
        price_col="close",
        atr_col="atr",
    )
    df_test_compression = df_test_features[
        df_test_features["compression_reaction"] == "compression_breakout"
    ].copy()

    df_test_labeled = prepare_rank_ic_labels(
        df_test_compression.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_test_compression.columns else None,
        date_col="date" if "date" in df_test_compression.columns else None,
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=args.horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=False,
    )

    # Train model
    print(f"\n   🚀 Training compression breakout model...")
    models, avg_rank_ic, cv_results, used_features = train_rank_ic_model(
        df_train_labeled,
        feature_cols=compression_features,
        target_col="volatility_normalized_target",
        n_splits=5,
        tscv_gap=24,
        hold_period=args.horizon,
    )

    print(f"   ✅ Compression breakout model trained: Avg Rank IC = {avg_rank_ic:.4f}")

    # Evaluate on test set
    print(f"\n   📊 Evaluating on test set...")
    df_test_signals = generate_ensemble_signals(
        df_test_labeled,
        models,
        used_features,
        confidence_threshold=0.85,
        signal_method="hybrid",
    )

    # Compute test Rank IC
    test_rank_ic = None
    if "pred" in df_test_signals.columns and "rr_achieved" in df_test_signals.columns:
        valid_mask = (
            df_test_signals["pred"].notna() & df_test_signals["rr_achieved"].notna()
        )
        if valid_mask.sum() > 10:
            test_rank_ic = compute_rank_ic(
                df_test_signals.loc[valid_mask, "pred"],
                df_test_signals.loc[valid_mask, "rr_achieved"],
            )
            print(f"   ✅ Test Rank IC: {test_rank_ic:.4f}")

    # Evaluate performance
    signals = df_test_signals.get("signal", pd.Series(0, index=df_test_signals.index))
    test_eval = evaluate_model_performance(
        df_test_signals,
        signals,
        true_return_col="rr_achieved",
        hold_period=args.horizon,
    )

    # Save results
    results = {
        "model_type": "compression_breakout",
        "avg_rank_ic_cv": float(avg_rank_ic),
        "test_rank_ic": float(test_rank_ic) if test_rank_ic is not None else None,
        "n_features": len(used_features),
        "n_train_samples": len(df_train_labeled),
        "n_test_samples": len(df_test_labeled),
        "features": used_features,
        "evaluation": test_eval,
    }

    results_file = output_dir / "compression_breakout_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Compression Breakout Model:")
    print(f"  CV Rank IC: {avg_rank_ic:.4f}")
    print(
        f"  Test Rank IC: {test_rank_ic:.4f}"
        if test_rank_ic is not None
        else "  Test Rank IC: N/A"
    )
    print(f"  Features: {len(used_features)}")
    print(f"  Train Samples: {len(df_train_labeled)}")
    print(f"  Test Samples: {len(df_test_labeled)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
