#!/usr/bin/env python3
"""
检查 dist_to_nearest_sr 列是否存在，并分析距离分布

Usage:
    python scripts/check_dist_to_sr_column.py --symbol BTCUSDT --timeframe 240T
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
    parser = argparse.ArgumentParser(description="检查 dist_to_nearest_sr 列和距离分布")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易符号")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("检查 dist_to_nearest_sr 列和距离分布")
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
    print(f"   原始数据行数: {len(df):,}")
    print()

    # 加载特征
    print("🔧 加载特征...")
    feature_loader = StrategyFeatureLoader()

    config_dir = Path("config/strategies/sr_reversal_long_weighted")
    config_loader = StrategyConfigLoader(config_dir)
    config = config_loader.load()

    df_features = feature_loader.load_features_from_requested(
        df.copy(),
        config.features.requested_features,
        fit=False,
    )
    print(f"   特征数据行数: {len(df_features):,}")
    print(f"   特征列数: {len(df_features.columns)}")
    print()

    # 检查 dist_to_nearest_sr 列
    print("=" * 80)
    print("1️⃣  检查 dist_to_nearest_sr 列")
    print("=" * 80)
    print()

    if "dist_to_nearest_sr" in df_features.columns:
        print("✅ dist_to_nearest_sr 列存在")
        print()

        # 分析列的基本信息
        dist_col = df_features["dist_to_nearest_sr"]
        print("列的基本信息:")
        print(
            f"   非空值数量: {dist_col.notna().sum():,} ({dist_col.notna().sum()/len(dist_col)*100:.1f}%)"
        )
        print(
            f"   空值数量: {dist_col.isna().sum():,} ({dist_col.isna().sum()/len(dist_col)*100:.1f}%)"
        )
        print()

        # 分析值的分布
        valid_dist = dist_col.dropna()
        if len(valid_dist) > 0:
            print("值的分布（相对百分比）:")
            print(f"   最小值: {valid_dist.min():.6f}")
            print(f"   25%分位: {valid_dist.quantile(0.25):.6f}")
            print(f"   中位数: {valid_dist.median():.6f}")
            print(f"   75%分位: {valid_dist.quantile(0.75):.6f}")
            print(f"   最大值: {valid_dist.max():.6f}")
            print(f"   均值: {valid_dist.mean():.6f}")
            print(f"   标准差: {valid_dist.std():.6f}")
            print()

            # 检查是否全为0
            if (valid_dist == 0).all():
                print("⚠️  警告: 所有值都是 0（可能未正确计算）")
            elif (valid_dist == 0).sum() > len(valid_dist) * 0.5:
                zero_count = (valid_dist == 0).sum()
                print(
                    f"⚠️  警告: {zero_count:,} 个值为 0 ({zero_count/len(valid_dist)*100:.1f}%)"
                )
            print()
    else:
        print("❌ dist_to_nearest_sr 列不存在")
        print()
        print("可能原因:")
        print("   1. sr_strength_max_close_f 特征未正确生成该列")
        print("   2. 特征依赖关系未正确解析")
        print()
        print("检查相关列:")
        sr_cols = [
            col
            for col in df_features.columns
            if "sr" in col.lower() or "dist" in col.lower()
        ]
        if sr_cols:
            print(f"   找到 {len(sr_cols)} 个SR相关列:")
            for col in sr_cols[:15]:
                print(f"     - {col}")
            if len(sr_cols) > 15:
                print(f"     ... 还有 {len(sr_cols) - 15} 个")
        else:
            print("   ⚠️  未找到任何SR相关列")
        print()
        return

    # 分析距离分布（转换为ATR倍数）
    print("=" * 80)
    print("2️⃣  分析距离分布（转换为ATR倍数）")
    print("=" * 80)
    print()

    # 确保有必要的列
    required_cols = ["dist_to_nearest_sr", "atr", "close"]
    missing_cols = [col for col in required_cols if col not in df_features.columns]
    if missing_cols:
        print(f"❌ 缺少必要的列: {missing_cols}")
        return

    # 计算归一化距离
    dist_pct = df_features["dist_to_nearest_sr"].abs()
    price = df_features["close"]
    atr = df_features["atr"].fillna(df_features["atr"].median())

    # 将百分比转换为绝对价格距离，再归一化到ATR
    abs_distance = dist_pct * price
    dist_normalized = abs_distance / (atr + 1e-8)

    # 统计在SR附近的样本
    dist_atr_mult = 1.5
    near_sr_mask = dist_normalized <= dist_atr_mult
    near_sr_count = near_sr_mask.sum()
    far_sr_count = (~near_sr_mask).sum()

    print(f"距离阈值: {dist_atr_mult} ATR")
    print()
    print(f"在SR附近（≤{dist_atr_mult} ATR）:")
    print(f"   样本数: {near_sr_count:,} ({near_sr_count/len(df_features)*100:.1f}%)")
    print()
    print(f"不在SR附近（>{dist_atr_mult} ATR）:")
    print(f"   样本数: {far_sr_count:,} ({far_sr_count/len(df_features)*100:.1f}%)")
    print()

    if far_sr_count == 0:
        print("⚠️  所有样本都在SR附近！")
        print("   这意味着SR过滤不会减少标签数量")
        print("   建议: 减小 dist_atr_mult（如从 1.5 改为 1.0 或 0.8）")
        print()

    # 距离分布统计
    print("距离分布统计（ATR倍数）:")
    print(f"   最小值: {dist_normalized.min():.2f}")
    print(f"   25%分位: {dist_normalized.quantile(0.25):.2f}")
    print(f"   中位数: {dist_normalized.median():.2f}")
    print(f"   75%分位: {dist_normalized.quantile(0.75):.2f}")
    print(f"   最大值: {dist_normalized.max():.2f}")
    print()

    # 不同阈值下的覆盖率
    print("不同阈值下的覆盖率:")
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        coverage = (dist_normalized <= threshold).sum() / len(df_features) * 100
        print(f"   ≤{threshold} ATR: {coverage:.1f}%")
    print()

    # 测试标签生成
    print("=" * 80)
    print("3️⃣  测试标签生成和SR过滤")
    print("=" * 80)
    print()

    # 生成无权重版本的标签
    print("生成无权重版本标签（全量扫描）...")
    labels_no_weight = compute_sr_reversal_label_full_scan(
        df_features,
        max_holding_bars=20,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
    )
    valid_labels_no_weight = labels_no_weight.notna().sum()
    print(f"   有效标签数: {valid_labels_no_weight:,}")
    print()

    # 生成带权重版本的标签（使用 dist_to_sr_col）
    print("生成带权重版本标签（使用 dist_to_sr_col 过滤）...")
    try:
        labels_weighted = compute_sr_reversal_label_with_weights(
            df_features,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=dist_atr_mult,
            compute_weights=False,
        )
        valid_labels_weighted = labels_weighted.notna().sum()
        print(f"   有效标签数: {valid_labels_weighted:,}")
        print()

        # 对比
        label_diff = valid_labels_no_weight - valid_labels_weighted
        label_diff_pct = (
            (label_diff / valid_labels_no_weight * 100)
            if valid_labels_no_weight > 0
            else 0
        )

        print("标签数量对比:")
        print(f"   无权重版本: {valid_labels_no_weight:,}")
        print(f"   带权重版本: {valid_labels_weighted:,}")
        print(f"   差异: {label_diff:,} ({label_diff_pct:.2f}%)")
        print()

        if label_diff > 0:
            print(f"   ✅ SR过滤生效，减少了 {label_diff:,} 个标签")
        elif label_diff == 0:
            print(f"   ⚠️  标签数量未减少")
            if far_sr_count == 0:
                print(f"   原因: 所有样本都在SR附近（≤{dist_atr_mult} ATR）")
            else:
                print(f"   原因: 需要进一步检查")

        # 验证过滤的正确性
        print()
        print("验证过滤正确性:")
        labels_in_sr = labels_weighted[near_sr_mask].notna().sum()
        labels_out_sr = labels_weighted[~near_sr_mask].notna().sum()
        print(f"   在SR附近的标签: {labels_in_sr:,}")
        print(f"   不在SR附近的标签: {labels_out_sr:,}")

        if labels_out_sr == 0:
            print("   ✅ 过滤正确：不在SR附近的样本标签为 NaN")
        else:
            print(f"   ⚠️  警告：有 {labels_out_sr:,} 个不在SR附近的标签")
            print("   可能原因: 标签生成逻辑在过滤前已经生成了标签")

    except Exception as e:
        print(f"   ❌ 标签生成失败: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 80)
    print("✅ 检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
