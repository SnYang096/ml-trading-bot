#!/usr/bin/env python3
"""
重新检查两个策略的标签数量差异（修复后）

验证修复后的代码是否能正确减少标签数量（只在SR附近生成标签）
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.time_series.utils_interaction_features import compute_is_near_sr
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_full_scan,
    compute_sr_reversal_label_with_weights,
)


def create_test_data():
    """创建测试数据"""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1H")

    # 基础价格数据
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(200) * 0.5)

    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "open": prices * 0.995,
            "volume": np.random.uniform(1000, 5000, 200),
            "atr": np.full(200, 2.0),  # 固定 ATR = 2.0
            # 创建混合的 dist_to_nearest_sr：一部分在SR附近，一部分不在
            "dist_to_nearest_sr": np.concatenate(
                [
                    np.random.uniform(-0.02, 0.02, 100),  # 在SR附近（±2%，约1 ATR内）
                    np.random.uniform(
                        -0.10, -0.05, 50
                    ),  # 不在SR附近（-10% 到 -5%，约2.5-5 ATR）
                    np.random.uniform(
                        0.05, 0.10, 50
                    ),  # 不在SR附近（5% 到 10%，约2.5-5 ATR）
                ]
            ),
        },
        index=dates,
    )

    return df


def main():
    print("=" * 80)
    print("重新检查标签生成差异（修复后）")
    print("=" * 80)
    print()

    # 创建测试数据
    print("📊 创建测试数据...")
    df = create_test_data()
    print(f"   数据行数: {len(df)}")
    print()

    # 分析 dist_to_nearest_sr 分布
    print("🔍 分析 dist_to_nearest_sr 分布...")
    dist_pct = df["dist_to_nearest_sr"].abs()
    price = df["close"]
    atr = df["atr"]

    # 计算归一化距离
    abs_distance = dist_pct * price
    dist_normalized = abs_distance / atr

    # 统计在SR附近的样本（1.5 ATR内）
    near_sr_mask = dist_normalized <= 1.5
    near_sr_count = near_sr_mask.sum()
    far_sr_count = (~near_sr_mask).sum()

    print(
        f"   在SR附近（≤1.5 ATR）: {near_sr_count} ({near_sr_count/len(df)*100:.1f}%)"
    )
    print(
        f"   不在SR附近（>1.5 ATR）: {far_sr_count} ({far_sr_count/len(df)*100:.1f}%)"
    )
    print()

    # 生成无权重版本的标签（全量扫描）
    print("1️⃣  生成无权重版本标签 (compute_sr_reversal_label_full_scan)...")
    labels_no_weight = compute_sr_reversal_label_full_scan(
        df,
        max_holding_bars=20,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
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

    # 生成带权重版本的标签（使用 dist_to_sr_col 过滤）
    print(
        "2️⃣  生成带权重版本标签 (compute_sr_reversal_label_with_weights, 使用 dist_to_sr_col)..."
    )
    labels_weighted_dist = compute_sr_reversal_label_with_weights(
        df,
        max_holding_bars=20,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
        dist_to_sr_col="dist_to_nearest_sr",
        dist_atr_mult=1.5,
        compute_weights=False,
    )
    valid_labels_weighted_dist = labels_weighted_dist.notna().sum()
    pos_labels_weighted_dist = (labels_weighted_dist == 1.0).sum()
    neg_labels_weighted_dist = (labels_weighted_dist == 0.0).sum()
    print(f"   ✅ 有效标签数: {valid_labels_weighted_dist:,}")
    print(
        f"   ✅ 正样本: {pos_labels_weighted_dist:,} ({pos_labels_weighted_dist/valid_labels_weighted_dist*100:.2f}%)"
    )
    print(
        f"   ✅ 负样本: {neg_labels_weighted_dist:,} ({neg_labels_weighted_dist/valid_labels_weighted_dist*100:.2f}%)"
    )
    print()

    # 计算 is_near_sr 特征
    print("3️⃣  计算 is_near_sr 特征...")
    is_near_sr = compute_is_near_sr(
        df,
        dist_col="dist_to_nearest_sr",
        atr_col="atr",
        price_col="close",
        dist_atr_mult=1.5,
    )
    df["is_near_sr"] = is_near_sr
    is_near_sr_count = is_near_sr.sum()
    print(
        f"   ✅ is_near_sr=True 的样本数: {is_near_sr_count:,} ({is_near_sr_count/len(df)*100:.1f}%)"
    )
    print()

    # 生成带权重版本的标签（使用 sr_mask_col 过滤）
    print(
        "4️⃣  生成带权重版本标签 (compute_sr_reversal_label_with_weights, 使用 sr_mask_col)..."
    )
    labels_weighted_mask = compute_sr_reversal_label_with_weights(
        df,
        max_holding_bars=20,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
        sr_mask_col="is_near_sr",
        compute_weights=False,
    )
    valid_labels_weighted_mask = labels_weighted_mask.notna().sum()
    pos_labels_weighted_mask = (labels_weighted_mask == 1.0).sum()
    neg_labels_weighted_mask = (labels_weighted_mask == 0.0).sum()
    print(f"   ✅ 有效标签数: {valid_labels_weighted_mask:,}")
    print(
        f"   ✅ 正样本: {pos_labels_weighted_mask:,} ({pos_labels_weighted_mask/valid_labels_weighted_mask*100:.2f}%)"
    )
    print(
        f"   ✅ 负样本: {neg_labels_weighted_mask:,} ({neg_labels_weighted_mask/valid_labels_weighted_mask*100:.2f}%)"
    )
    print()

    # 对比分析
    print("=" * 80)
    print("📊 对比分析")
    print("=" * 80)
    print()

    print("标签数量对比:")
    print(f"   无权重版本（全量扫描）: {valid_labels_no_weight:,}")
    print(f"   带权重版本（dist_to_sr_col）: {valid_labels_weighted_dist:,}")
    print(f"   带权重版本（sr_mask_col）: {valid_labels_weighted_mask:,}")
    print()

    diff_dist = valid_labels_no_weight - valid_labels_weighted_dist
    diff_mask = valid_labels_no_weight - valid_labels_weighted_mask
    diff_pct_dist = (
        (diff_dist / valid_labels_no_weight * 100) if valid_labels_no_weight > 0 else 0
    )
    diff_pct_mask = (
        (diff_mask / valid_labels_no_weight * 100) if valid_labels_no_weight > 0 else 0
    )

    print("差异分析:")
    print(f"   使用 dist_to_sr_col 过滤:")
    print(f"     减少标签数: {diff_dist:,} ({diff_pct_dist:.2f}%)")
    if diff_dist > 0:
        print(f"     ✅ 成功减少了 {diff_dist:,} 个标签")
    else:
        print(f"     ⚠️  标签数量未减少（可能所有样本都在SR附近）")

    print(f"   使用 sr_mask_col 过滤:")
    print(f"     减少标签数: {diff_mask:,} ({diff_pct_mask:.2f}%)")
    if diff_mask > 0:
        print(f"     ✅ 成功减少了 {diff_mask:,} 个标签")
    else:
        print(f"     ⚠️  标签数量未减少（可能所有样本都在SR附近）")
    print()

    # 验证过滤的正确性
    print("验证过滤正确性:")

    # 检查 dist_to_sr_col 过滤
    labels_in_sr_dist = labels_weighted_dist[near_sr_mask].notna().sum()
    labels_out_sr_dist = labels_weighted_dist[~near_sr_mask].notna().sum()
    print(f"   使用 dist_to_sr_col:")
    print(f"     在SR附近的标签: {labels_in_sr_dist:,}")
    print(f"     不在SR附近的标签: {labels_out_sr_dist:,}")
    if labels_out_sr_dist == 0:
        print(f"     ✅ 过滤正确：不在SR附近的样本标签为 NaN")
    else:
        print(
            f"     ⚠️  警告：有 {labels_out_sr_dist:,} 个不在SR附近的标签（可能由于标签生成逻辑）"
        )

    # 检查 sr_mask_col 过滤
    labels_in_sr_mask = labels_weighted_mask[is_near_sr].notna().sum()
    labels_out_sr_mask = labels_weighted_mask[~is_near_sr].notna().sum()
    print(f"   使用 sr_mask_col:")
    print(f"     is_near_sr=True 的标签: {labels_in_sr_mask:,}")
    print(f"     is_near_sr=False 的标签: {labels_out_sr_mask:,}")
    if labels_out_sr_mask == 0:
        print(f"     ✅ 过滤正确：is_near_sr=False 的样本标签为 NaN")
    else:
        print(f"     ⚠️  警告：有 {labels_out_sr_mask:,} 个 is_near_sr=False 的标签")
    print()

    # 验证两种过滤方法的一致性
    print("验证两种过滤方法的一致性:")
    if valid_labels_weighted_dist == valid_labels_weighted_mask:
        print(f"   ✅ 两种过滤方法结果一致（标签数相同）")
    else:
        print(f"   ⚠️  两种过滤方法结果不一致")
        print(
            f"     差异: {abs(valid_labels_weighted_dist - valid_labels_weighted_mask):,} 个标签"
        )
    print()

    print("=" * 80)
    print("✅ 检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
