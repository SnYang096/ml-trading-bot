#!/usr/bin/env python3
"""
检查两个策略的标签数量差异和SR过滤情况

Usage:
    python scripts/check_label_difference.py --symbol BTCUSDT --timeframe 240T
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data_tools.data_handler import DataHandler
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_full_scan,
    compute_sr_reversal_label_with_weights,
)


def main():
    parser = argparse.ArgumentParser(description="检查标签数量差异")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易符号")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("检查两个策略的标签数量差异")
    print("=" * 80)
    print()

    # 加载数据
    print("📊 加载数据...")
    data_handler = DataHandler(args.data_path)
    df = data_handler.load_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=None,
        end_date=None,
    )
    print(f"   数据行数: {len(df)}")
    print()

    # 加载特征
    print("🔧 加载特征...")
    feature_loader = StrategyFeatureLoader()

    # 加载无权重版本的配置
    config_dir_no_weight = Path("config/strategies/sr_reversal_long")
    config_no_weight = StrategyConfigLoader(config_dir_no_weight)
    df_no_weight = feature_loader.load_features(df.copy(), config_no_weight.features)
    print(f"   特征列数: {len(df_no_weight.columns)}")
    print()

    # 加载带权重版本的配置
    config_dir_weighted = Path("config/strategies/sr_reversal_long_weighted")
    config_weighted = StrategyConfigLoader(config_dir_weighted)
    df_weighted = feature_loader.load_features(df.copy(), config_weighted.features)
    print(f"   特征列数: {len(df_weighted.columns)}")
    print()

    # 生成无权重版本的标签
    print("1️⃣  生成无权重版本标签 (compute_sr_reversal_label_full_scan)...")
    label_params_no_weight = config_no_weight.labels.generator.params
    labels_no_weight = compute_sr_reversal_label_full_scan(
        df_no_weight, **label_params_no_weight
    )
    valid_labels_no_weight = labels_no_weight.notna().sum()
    pos_labels_no_weight = (labels_no_weight == 1.0).sum()
    neg_labels_no_weight = (labels_no_weight == 0.0).sum()
    print(f"   ✅ 有效标签数: {valid_labels_no_weight:,}")
    print(
        f"   ✅ 正样本: {pos_labels_no_weight:,} ({pos_labels_no_weight/valid_labels_no_weight*100:.2f}%)"
    )
    print(
        f"   ✅ 负样本: {neg_labels_no_weight:,} ({neg_labels_no_weight/valid_labels_no_weight*100:.2f}%)"
    )
    print()

    # 检查SR相关列
    print("🔍 检查SR相关特征列...")
    sr_cols = [
        col
        for col in df_weighted.columns
        if "sr" in col.lower() or "is_near" in col.lower() or "dist_to" in col.lower()
    ]
    print(f"   找到 {len(sr_cols)} 个SR相关列:")
    for col in sr_cols[:10]:  # 只显示前10个
        print(f"      - {col}")
    if len(sr_cols) > 10:
        print(f"      ... 还有 {len(sr_cols) - 10} 个")
    print()

    # 检查 is_near_sr 列
    if "is_near_sr" in df_weighted.columns:
        is_near_sr_count = df_weighted["is_near_sr"].fillna(False).astype(bool).sum()
        print(f"   ✅ is_near_sr 列存在")
        print(
            f"   ✅ 在SR附近的样本数: {is_near_sr_count:,} ({is_near_sr_count/len(df_weighted)*100:.2f}%)"
        )
    else:
        print(f"   ⚠️  is_near_sr 列不存在，将使用 dist_to_sr_col 或全量扫描")
    print()

    # 生成带权重版本的标签
    print("2️⃣  生成带权重版本标签 (compute_sr_reversal_label_with_weights)...")
    label_params_weighted = config_weighted.labels.generator.params
    labels_weighted = compute_sr_reversal_label_with_weights(
        df_weighted, **label_params_weighted
    )
    valid_labels_weighted = labels_weighted.notna().sum()
    pos_labels_weighted = (labels_weighted == 1.0).sum()
    neg_labels_weighted = (labels_weighted == 0.0).sum()
    print(f"   ✅ 有效标签数: {valid_labels_weighted:,}")
    print(
        f"   ✅ 正样本: {pos_labels_weighted:,} ({pos_labels_weighted/valid_labels_weighted*100:.2f}%)"
    )
    print(
        f"   ✅ 负样本: {neg_labels_weighted:,} ({neg_labels_weighted/valid_labels_weighted*100:.2f}%)"
    )
    print()

    # 对比分析
    print("=" * 80)
    print("📊 对比分析")
    print("=" * 80)
    print()

    label_diff = valid_labels_no_weight - valid_labels_weighted
    label_diff_pct = (
        (label_diff / valid_labels_no_weight * 100) if valid_labels_no_weight > 0 else 0
    )

    print(f"标签数量差异:")
    print(f"   无权重版本: {valid_labels_no_weight:,}")
    print(f"   带权重版本: {valid_labels_weighted:,}")
    print(f"   差异: {label_diff:,} ({label_diff_pct:+.2f}%)")
    print()

    if label_diff > 0:
        print(f"   ✅ 带权重版本减少了 {label_diff:,} 个标签 ({label_diff_pct:.2f}%)")
        print(f"   ✅ 这些标签都是不在SR附近的样本")
    elif label_diff < 0:
        print(f"   ⚠️  带权重版本标签更多（不应该发生）")
    else:
        print(f"   ➡️  两个版本标签数量相同")
        print(f"   💡 可能原因:")
        print(f"      1. is_near_sr 列不存在，未应用SR过滤")
        print(f"      2. 所有样本都在SR附近（1.5*ATR内）")
        print(f"      3. dist_to_sr_col 列不存在，回退到全量扫描")
    print()

    # 检查是否真的应用了SR过滤
    if "is_near_sr" in df_weighted.columns:
        is_near_sr_mask = df_weighted["is_near_sr"].fillna(False).astype(bool)
        labels_in_sr = labels_weighted[is_near_sr_mask].notna().sum()
        labels_out_sr = labels_weighted[~is_near_sr_mask].notna().sum()
        print(f"SR过滤验证:")
        print(f"   在SR附近的标签: {labels_in_sr:,}")
        print(f"   不在SR附近的标签: {labels_out_sr:,}")
        if labels_out_sr > 0:
            print(
                f"   ⚠️  警告：有 {labels_out_sr:,} 个标签不在SR附近，说明SR过滤未生效"
            )
        else:
            print(f"   ✅ 所有标签都在SR附近，SR过滤生效")
    print()

    # 检查权重
    if "sample_weight" in df_weighted.columns:
        weights = df_weighted["sample_weight"]
        weighted_samples = (weights != 1.0).sum()
        print(f"样本权重统计:")
        print(
            f"   加权样本数: {weighted_samples:,} ({weighted_samples/valid_labels_weighted*100:.2f}%)"
        )
        print(f"   权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
        print(f"   权重均值: {weights.mean():.2f}")
    print()


if __name__ == "__main__":
    main()
