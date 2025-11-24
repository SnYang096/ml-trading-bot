"""
独立训练 SR 反转型和突破型两个模型系统。

核心思想：
- 反转型模型：专注 SR 反转逻辑，使用反转相关特征
- 突破型模型：专注趋势延续，使用突破相关特征
- 两个模型独立训练、独立评估，避免逻辑混淆
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


def select_breakout_features(df: pd.DataFrame, all_features: List[str]) -> List[str]:
    """
    为 SR 突破型模型选择特征。

    突破型机会的特征侧重：
    - 突破动量（速度、强度、持续性）
    - 成交量放大
    - 趋势延续信号
    - 流动性池位置
    - 波动率扩张
    """
    breakout_keywords = [
        # 突破动量
        "breakout_speed",
        "momentum",
        "momentum_decay",
        "follow_through",
        "breakout_momentum",
        "breakout_strength",
        # 成交量
        "volume_ratio",
        "vol_ratio",
        "volume_spike",
        "volume_confirmation",
        "order_flow",
        "cvd",
        "taker_buy_ratio",
        # 趋势
        "trend",
        "trend_strength",
        "trend_context",
        "trend_4h",
        "momentum_persistence",
        # 流动性
        "liquidity",
        "liquidity_pool",
        "order_block",
        # 波动率
        "volatility",
        "volatility_regime",
        "atr",
        "volatility_expansion",
        # 价格行为
        "price_action",
        "breakout_status",
        "breakout_confirmation",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        # 检查是否包含关键词
        if any(keyword in feat_lower for keyword in breakout_keywords):
            selected.append(feat)
        # 排除反转相关特征
        elif "reversal" in feat_lower or "reversed" in feat_lower:
            continue  # 这些更适合反转型
        else:
            # 保留通用特征
            if any(keyword in feat_lower for keyword in ["atr", "volatility", "vol"]):
                selected.append(feat)

    return selected


def train_reversal_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    output_dir: Path,
    model_name: str = "reversal",
) -> Tuple[List, float, pd.DataFrame, Dict]:
    """训练 SR 反转型模型"""
    print(f"\n{'='*60}")
    print(f"🔄 Training SR Reversal Model")
    print(f"{'='*60}")

    # Select reversal-specific features
    reversal_features = select_reversal_features(df_train, feature_cols)
    print(
        f"   📊 Selected {len(reversal_features)} reversal features (from {len(feature_cols)} total)"
    )

    # Prepare labels with R/R and reaction type split
    print(f"\n   📝 Preparing labels for reversal model...")
    df_train_labeled = prepare_rank_ic_labels(
        df_train.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_train.columns else None,
        date_col="date" if "date" in df_train.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
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
        return [], 0.0, pd.DataFrame(), {}

    # Prepare test set
    df_test_labeled = prepare_rank_ic_labels(
        df_test.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_test.columns else None,
        date_col="date" if "date" in df_test.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
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
        hold_period=horizon,
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
        hold_period=horizon,
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

    results_file = output_dir / f"{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    return models, avg_rank_ic, df_test_signals, results


def train_breakout_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    output_dir: Path,
    model_name: str = "breakout",
) -> Tuple[List, float, pd.DataFrame, Dict]:
    """训练 SR 突破型模型"""
    print(f"\n{'='*60}")
    print(f"📈 Training SR Breakout Model")
    print(f"{'='*60}")

    # Select breakout-specific features
    breakout_features = select_breakout_features(df_train, feature_cols)
    print(
        f"   📊 Selected {len(breakout_features)} breakout features (from {len(feature_cols)} total)"
    )

    # Prepare labels with R/R and reaction type split
    print(f"\n   📝 Preparing labels for breakout model...")
    df_train_labeled = prepare_rank_ic_labels(
        df_train.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_train.columns else None,
        date_col="date" if "date" in df_train.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=True,  # Enable reaction type split
    )

    # Filter to only breakout samples
    if "sr_reaction" not in df_train_labeled.columns:
        print("   ⚠️  Warning: sr_reaction column not found, using all samples")
        df_train_breakout = df_train_labeled.copy()
    else:
        breakout_mask = df_train_labeled["sr_reaction"] == "breakout"
        df_train_breakout = df_train_labeled[breakout_mask].copy()

    print(
        f"   ✅ Breakout samples: {len(df_train_breakout)} (from {len(df_train_labeled)} total)"
    )

    if len(df_train_breakout) < 100:
        print(f"   ⚠️  Warning: Too few breakout samples ({len(df_train_breakout)})")
        return [], 0.0, pd.DataFrame(), {}

    # Prepare test set
    df_test_labeled = prepare_rank_ic_labels(
        df_test.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_test.columns else None,
        date_col="date" if "date" in df_test.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=True,
    )
    if "sr_reaction" not in df_test_labeled.columns:
        df_test_breakout = df_test_labeled.copy()
    else:
        df_test_breakout = df_test_labeled[
            df_test_labeled["sr_reaction"] == "breakout"
        ].copy()

    # Train model
    print(f"\n   🚀 Training breakout model...")
    models, avg_rank_ic, cv_results, used_features = train_rank_ic_model(
        df_train_breakout,
        feature_cols=breakout_features,
        target_col="volatility_normalized_target",
        n_splits=5,
        tscv_gap=24,
        hold_period=horizon,
    )

    print(f"   ✅ Breakout model trained: Avg Rank IC = {avg_rank_ic:.4f}")

    # Evaluate on test set
    print(f"\n   📊 Evaluating on test set...")
    df_test_signals = generate_ensemble_signals(
        df_test_breakout,
        models,
        used_features,
        confidence_threshold=0.85,
        signal_method="hybrid",
    )

    # Compute test Rank IC
    test_rank_ic = None
    if (
        "pred" in df_test_signals.columns
        and "rr_breakout_achieved" in df_test_signals.columns
    ):
        valid_mask = (
            df_test_signals["pred"].notna()
            & df_test_signals["rr_breakout_achieved"].notna()
        )
        if valid_mask.sum() > 10:
            test_rank_ic = compute_rank_ic(
                df_test_signals.loc[valid_mask, "pred"],
                df_test_signals.loc[valid_mask, "rr_breakout_achieved"],
            )
            print(f"   ✅ Test Rank IC: {test_rank_ic:.4f}")

    # Evaluate performance
    signals = df_test_signals.get("signal", pd.Series(0, index=df_test_signals.index))
    test_eval = evaluate_model_performance(
        df_test_signals,
        signals,
        true_return_col="rr_breakout_achieved",
        hold_period=horizon,
    )

    # Save results
    results = {
        "model_type": "breakout",
        "avg_rank_ic_cv": float(avg_rank_ic),
        "test_rank_ic": float(test_rank_ic) if test_rank_ic is not None else None,
        "n_features": len(used_features),
        "n_train_samples": len(df_train_breakout),
        "n_test_samples": len(df_test_breakout),
        "features": used_features,
        "evaluation": test_eval,
    }

    results_file = output_dir / f"{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    return models, avg_rank_ic, df_test_signals, results


def train_compression_breakout_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    output_dir: Path,
    model_name: str = "compression_breakout",
) -> Tuple[List, float, pd.DataFrame, Dict]:
    """训练压缩区突破模型"""
    print(f"\n{'='*60}")
    print(f"💥 Training Compression Breakout Model")
    print(f"{'='*60}")

    # Select compression breakout-specific features
    compression_features = select_compression_breakout_features(df_train, feature_cols)
    print(
        f"   📊 Selected {len(compression_features)} compression breakout features (from {len(feature_cols)} total)"
    )

    # Classify compression breakout opportunities
    print(f"\n   📝 Classifying compression breakout opportunities...")
    from src.time_series_model.pipeline.training.label_utils import (
        classify_compression_breakout,
    )

    df_train["compression_reaction"] = classify_compression_breakout(
        df_train,
        price_col="close",
        atr_col="atr",
    )

    compression_breakout_mask = (
        df_train["compression_reaction"] == "compression_breakout"
    )
    df_train_compression = df_train[compression_breakout_mask].copy()

    print(
        f"   ✅ Compression breakout samples: {len(df_train_compression)} (from {len(df_train)} total)"
    )

    if len(df_train_compression) < 100:
        print(
            f"   ⚠️  Warning: Too few compression breakout samples ({len(df_train_compression)})"
        )
        return [], 0.0, pd.DataFrame(), {}

    # Prepare labels with R/R
    print(f"\n   📝 Preparing labels for compression breakout model...")
    df_train_labeled = prepare_rank_ic_labels(
        df_train_compression.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_train_compression.columns else None,
        date_col="date" if "date" in df_train_compression.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
        signal_col="signal",
        use_continuous_rr_label=False,
        split_by_reaction_type=False,  # Compression breakout doesn't need reaction split
    )

    # Prepare test set
    df_test["compression_reaction"] = classify_compression_breakout(
        df_test,
        price_col="close",
        atr_col="atr",
    )
    df_test_compression = df_test[
        df_test["compression_reaction"] == "compression_breakout"
    ].copy()

    df_test_labeled = prepare_rank_ic_labels(
        df_test_compression.copy(),
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_test_compression.columns else None,
        date_col="date" if "date" in df_test_compression.columns else None,
        hold_period=horizon,
        lookback_window=60,
        ensure_volatility=True,
        use_risk_reward_label=True,
        rr_ratio_threshold=2.0,
        max_holding_bars=horizon,
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
        hold_period=horizon,
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
        hold_period=horizon,
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

    results_file = output_dir / f"{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ Results saved to {results_file}")

    return models, avg_rank_ic, df_test_signals, results


def main():
    parser = argparse.ArgumentParser(
        description="Train separate SR Reversal and Breakout models"
    )
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
        help="Feature engineering type",
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
        default="results/sr_reaction_models",
        help="Output directory for results",
    )
    parser.add_argument(
        "--train-reversal-only",
        action="store_true",
        help="Only train reversal model",
    )
    parser.add_argument(
        "--train-breakout-only",
        action="store_true",
        help="Only train breakout model",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 SR Reaction Models Training")
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

    # Train models
    results_summary = {}

    if not args.train_breakout_only:
        # Train reversal model
        reversal_models, reversal_ic, reversal_test, reversal_results = (
            train_reversal_model(
                df_train_features,
                df_test_features,
                feature_cols,
                args.horizon,
                output_dir,
                model_name="reversal",
            )
        )
        results_summary["reversal"] = reversal_results

    if not args.train_reversal_only:
        # Train breakout model
        breakout_models, breakout_ic, breakout_test, breakout_results = (
            train_breakout_model(
                df_train_features,
                df_test_features,
                feature_cols,
                args.horizon,
                output_dir,
                model_name="breakout",
            )
        )
        results_summary["breakout"] = breakout_results

    # Print summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    if "reversal" in results_summary:
        rev = results_summary["reversal"]
        print(f"Reversal Model:")
        print(f"  CV Rank IC: {rev['avg_rank_ic_cv']:.4f}")
        print(f"  Test Rank IC: {rev.get('test_rank_ic', 'N/A')}")
        print(f"  Features: {rev['n_features']}")
        print(f"  Train Samples: {rev['n_train_samples']}")
        print(f"  Test Samples: {rev['n_test_samples']}")
    if "breakout" in results_summary:
        brk = results_summary["breakout"]
        print(f"Breakout Model:")
        print(f"  CV Rank IC: {brk['avg_rank_ic_cv']:.4f}")
        print(f"  Test Rank IC: {brk.get('test_rank_ic', 'N/A')}")
        print(f"  Features: {brk['n_features']}")
        print(f"  Train Samples: {brk['n_train_samples']}")
        print(f"  Test Samples: {brk['n_test_samples']}")
    if "compression_breakout" in results_summary:
        comp = results_summary["compression_breakout"]
        print(f"Compression Breakout Model:")
        print(f"  CV Rank IC: {comp['avg_rank_ic_cv']:.4f}")
        print(f"  Test Rank IC: {comp.get('test_rank_ic', 'N/A')}")
        print(f"  Features: {comp['n_features']}")
        print(f"  Train Samples: {comp['n_train_samples']}")
        print(f"  Test Samples: {comp['n_test_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
