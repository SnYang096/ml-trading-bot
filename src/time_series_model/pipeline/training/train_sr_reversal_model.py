"""
独立训练 SR 反转型模型。

核心思想：
- 专注 SR 反转逻辑，使用反转相关特征
- 只使用 sr_reaction == "reversal" 的样本
- 预测是否实现 ≥2R 的反向盈利
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


def select_reversal_features(df: pd.DataFrame, all_features: List[str]) -> List[str]:
    """
    为 SR 反转型模型选择特征。

    反转型机会的特征侧重：
    - SR 区域强度（SQS、距离、密度）
    - 反转信号（价格回测、试探、反向运行）
    - 成交量确认（萎缩、异常）
    - 多时间框架结构对齐
    - 压缩区域质量
    """
    reversal_keywords = [
        # SR 区域相关
        "sqs",
        "dist_to_nearest_sr",
        "sr_density",
        "sr_confluence",
        "nearest_sr",
        "direction_to_nearest_sr",
        # 反转信号相关
        "reversal",
        "reversed",
        "fake_breakout",
        "breakout_status",
        "price_reversed_before_sr",
        # SR 边界强度
        "sr_strength",
        "boundary_strength",
        "support_sqs",
        "resistance_sqs",
        # 成交量确认
        "volume_ratio",
        "vol_ratio",
        "volume_confirmation",
        # 压缩和结构
        "compression",
        "compression_score",
        "compression_confidence",
        # 多时间框架
        "trend_context",
        "trend_4h",
        "volatility_regime",
        # 边界质量
        "breakout_quality",
        "breakout_confirmation",
        "role_flip",
        # 价格行为
        "price_action",
        "candlestick",
        # Default 特征（技术指标，可能对反转有用）
        "rsi",
        "macd",
        "stochastic",
        "williams",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        # 检查是否包含关键词
        if any(keyword in feat_lower for keyword in reversal_keywords):
            selected.append(feat)
        # 排除突破相关特征
        elif "breakout_speed" in feat_lower or "momentum_decay" in feat_lower:
            continue  # 这些更适合突破型
        else:
            # 保留通用特征（如 ATR、波动率等）
            if any(
                keyword in feat_lower
                for keyword in ["atr", "volatility", "vol", "trend_strength"]
            ):
                selected.append(feat)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Train SR Reversal Model")
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
        default="results/sr_reversal_model",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔄 SR Reversal Model Training")
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

    # Select reversal-specific features
    reversal_features = select_reversal_features(df_train_features, feature_cols)
    print(
        f"\n   📊 Selected {len(reversal_features)} reversal features (from {len(feature_cols)} total)"
    )

    # Prepare labels with R/R and reaction type split
    print(f"\n   📝 Preparing labels for reversal model...")
    df_train_labeled = prepare_rank_ic_labels(
        df_train_features.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_train_features.columns else None,
        date_col="date" if "date" in df_train_features.columns else None,
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=args.horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=True,  # Enable reaction type split
    )

    # Filter to only reversal samples
    if "sr_reaction" not in df_train_labeled.columns:
        print("   ⚠️  Warning: sr_reaction column not found, using all samples")
        df_train_reversal = df_train_labeled.copy()
    else:
        reversal_mask = df_train_labeled["sr_reaction"] == "reversal"
        df_train_reversal = df_train_labeled[reversal_mask].copy()

    print(
        f"   ✅ Reversal samples: {len(df_train_reversal)} (from {len(df_train_labeled)} total)"
    )

    if len(df_train_reversal) < 100:
        print(f"   ⚠️  Warning: Too few reversal samples ({len(df_train_reversal)})")
        return

    # Prepare test set
    df_test_labeled = prepare_rank_ic_labels(
        df_test_features.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_test_features.columns else None,
        date_col="date" if "date" in df_test_features.columns else None,
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=args.horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=True,
    )
    if "sr_reaction" not in df_test_labeled.columns:
        df_test_reversal = df_test_labeled.copy()
    else:
        df_test_reversal = df_test_labeled[
            df_test_labeled["sr_reaction"] == "reversal"
        ].copy()

    # Train model
    print(f"\n   🚀 Training reversal model...")
    models, avg_rank_ic, cv_results, used_features = train_rank_ic_model(
        df_train_reversal,
        feature_cols=reversal_features,
        target_col="volatility_normalized_target",
        n_splits=5,
        tscv_gap=24,
        hold_period=args.horizon,
    )

    print(f"   ✅ Reversal model trained: Avg Rank IC = {avg_rank_ic:.4f}")

    # Evaluate on test set
    print(f"\n   📊 Evaluating on test set...")
    df_test_signals = generate_ensemble_signals(
        df_test_reversal,
        models,
        used_features,
        confidence_threshold=0.85,
        signal_method="hybrid",
    )

    # Compute test Rank IC
    test_rank_ic = None
    if (
        "pred" in df_test_signals.columns
        and "rr_reversal_achieved" in df_test_signals.columns
    ):
        valid_mask = (
            df_test_signals["pred"].notna()
            & df_test_signals["rr_reversal_achieved"].notna()
        )
        if valid_mask.sum() > 10:
            test_rank_ic = compute_rank_ic(
                df_test_signals.loc[valid_mask, "pred"],
                df_test_signals.loc[valid_mask, "rr_reversal_achieved"],
            )
            print(f"   ✅ Test Rank IC: {test_rank_ic:.4f}")

    # Evaluate performance
    signals = df_test_signals.get("signal", pd.Series(0, index=df_test_signals.index))
    test_eval = evaluate_model_performance(
        df_test_signals,
        signals,
        true_return_col="rr_reversal_achieved",
        hold_period=args.horizon,
    )

    # Save results
    results = {
        "model_type": "reversal",
        "avg_rank_ic_cv": float(avg_rank_ic),
        "test_rank_ic": float(test_rank_ic) if test_rank_ic is not None else None,
        "n_features": len(used_features),
        "n_train_samples": len(df_train_reversal),
        "n_test_samples": len(df_test_reversal),
        "features": used_features,
        "evaluation": test_eval,
    }

    results_file = output_dir / "reversal_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Reversal Model:")
    print(f"  CV Rank IC: {avg_rank_ic:.4f}")
    print(
        f"  Test Rank IC: {test_rank_ic:.4f}"
        if test_rank_ic is not None
        else "  Test Rank IC: N/A"
    )
    print(f"  Features: {len(used_features)}")
    print(f"  Train Samples: {len(df_train_reversal)}")
    print(f"  Test Samples: {len(df_test_reversal)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
