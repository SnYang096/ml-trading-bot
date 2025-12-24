#!/usr/bin/env python3
"""
分析回归（带权重）版本性能下降的原因

检查：
1. 标签分布变化
2. 预测分布变化
3. 样本质量变化
4. SR过滤的影响
"""

from __future__ import annotations

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
    compute_sr_reversal_rr_continuous_label,
    compute_sr_reversal_rr_continuous_label_with_weights,
)


def main():
    print("=" * 80)
    print("分析回归（带权重）版本性能下降的原因")
    print("=" * 80)
    print()

    # 加载数据
    print("📊 加载数据...")
    data_handler = DataHandler("data/parquet_data")
    df = data_handler.load_ohlcv(
        symbol="BTCUSDT",
        timeframe="240T",
        start_date=None,
        end_date=None,
    )
    print(f"   原始数据行数: {len(df):,}")
    print()

    # 加载特征
    print("🔧 加载特征...")
    feature_loader = StrategyFeatureLoader()

    config_dir = Path("config/strategies/sr_reversal_rr_reg_long_weighted")
    config_loader = StrategyConfigLoader(config_dir)
    config = config_loader.load()

    df_features = feature_loader.load_features_from_requested(
        df.copy(),
        config.features.requested_features,
        fit=False,
    )
    print(f"   特征数据行数: {len(df_features):,}")
    print()

    # 生成标签
    print("=" * 80)
    print("1️⃣  生成标签对比")
    print("=" * 80)
    print()

    # 无权重版本
    print("生成无权重版本标签...")
    labels_no_weight = compute_sr_reversal_rr_continuous_label(
        df_features,
        max_holding_bars=50,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
    )
    valid_no_weight = labels_no_weight.notna().sum()
    print(f"   有效标签数: {valid_no_weight:,}")

    # 带权重版本
    print("生成带权重版本标签（SR过滤）...")
    labels_weighted = compute_sr_reversal_rr_continuous_label_with_weights(
        df_features,
        max_holding_bars=50,
        take_profit_r=2.0,
        stop_loss_r=1.0,
        combine_mode="long_only",
        dist_to_sr_col="dist_to_nearest_sr",
        dist_atr_mult=1.5,
        compute_weights=False,
    )
    valid_weighted = labels_weighted.notna().sum()
    print(f"   有效标签数: {valid_weighted:,}")
    print()

    # 标签分布分析
    print("=" * 80)
    print("2️⃣  标签分布分析")
    print("=" * 80)
    print()

    # 无权重版本
    valid_labels_no_weight = labels_no_weight.dropna()
    print("无权重版本标签分布:")
    print(f"   有效标签数: {len(valid_labels_no_weight):,}")
    print(f"   最小值: {valid_labels_no_weight.min():.4f}")
    print(f"   25%分位: {valid_labels_no_weight.quantile(0.25):.4f}")
    print(f"   中位数: {valid_labels_no_weight.median():.4f}")
    print(f"   75%分位: {valid_labels_no_weight.quantile(0.75):.4f}")
    print(f"   最大值: {valid_labels_no_weight.max():.4f}")
    print(f"   均值: {valid_labels_no_weight.mean():.4f}")
    print(f"   标准差: {valid_labels_no_weight.std():.4f}")

    # 统计盈利/亏损样本
    profit_no_weight = (valid_labels_no_weight >= 1.0).sum()
    loss_no_weight = (valid_labels_no_weight < 1.0).sum()
    print(
        f"   盈利样本（RR >= 1.0）: {profit_no_weight:,} ({profit_no_weight/len(valid_labels_no_weight)*100:.1f}%)"
    )
    print(
        f"   亏损样本（RR < 1.0）: {loss_no_weight:,} ({loss_no_weight/len(valid_labels_no_weight)*100:.1f}%)"
    )
    print()

    # 带权重版本
    valid_labels_weighted = labels_weighted.dropna()
    print("带权重版本标签分布:")
    print(f"   有效标签数: {len(valid_labels_weighted):,}")
    print(f"   最小值: {valid_labels_weighted.min():.4f}")
    print(f"   25%分位: {valid_labels_weighted.quantile(0.25):.4f}")
    print(f"   中位数: {valid_labels_weighted.median():.4f}")
    print(f"   75%分位: {valid_labels_weighted.quantile(0.75):.4f}")
    print(f"   最大值: {valid_labels_weighted.max():.4f}")
    print(f"   均值: {valid_labels_weighted.mean():.4f}")
    print(f"   标准差: {valid_labels_weighted.std():.4f}")

    # 统计盈利/亏损样本
    profit_weighted = (valid_labels_weighted >= 1.0).sum()
    loss_weighted = (valid_labels_weighted < 1.0).sum()
    print(
        f"   盈利样本（RR >= 1.0）: {profit_weighted:,} ({profit_weighted/len(valid_labels_weighted)*100:.1f}%)"
    )
    print(
        f"   亏损样本（RR < 1.0）: {loss_weighted:,} ({loss_weighted/len(valid_labels_weighted)*100:.1f}%)"
    )
    print()

    # 对比分析
    print("=" * 80)
    print("3️⃣  对比分析")
    print("=" * 80)
    print()

    # 标签数量变化
    reduction_pct = (valid_no_weight - valid_weighted) / valid_no_weight * 100
    print(
        f"标签数量变化: {valid_no_weight:,} → {valid_weighted:,} (减少 {reduction_pct:.1f}%)"
    )
    print()

    # 盈利样本比例变化
    profit_rate_no_weight = profit_no_weight / len(valid_labels_no_weight) * 100
    profit_rate_weighted = profit_weighted / len(valid_labels_weighted) * 100
    profit_rate_change = profit_rate_weighted - profit_rate_no_weight

    print(f"盈利样本比例变化:")
    print(f"   无权重版本: {profit_rate_no_weight:.1f}%")
    print(f"   带权重版本: {profit_rate_weighted:.1f}%")
    print(f"   变化: {profit_rate_change:+.1f}%")
    print()

    # 标签均值变化
    mean_change = valid_labels_weighted.mean() - valid_labels_no_weight.mean()
    print(f"标签均值变化:")
    print(f"   无权重版本: {valid_labels_no_weight.mean():.4f}")
    print(f"   带权重版本: {valid_labels_weighted.mean():.4f}")
    print(f"   变化: {mean_change:+.4f}")
    print()

    # 标签标准差变化
    std_change = valid_labels_weighted.std() - valid_labels_no_weight.std()
    print(f"标签标准差变化:")
    print(f"   无权重版本: {valid_labels_no_weight.std():.4f}")
    print(f"   带权重版本: {valid_labels_weighted.std():.4f}")
    print(f"   变化: {std_change:+.4f}")
    print()

    # 结论
    print("=" * 80)
    print("4️⃣  结论")
    print("=" * 80)
    print()

    if profit_rate_change < 0:
        print(f"⚠️  盈利样本比例下降 {abs(profit_rate_change):.1f}%")
        print("   说明：SR过滤可能过滤掉了一些盈利样本")
        print("   建议：检查SR过滤逻辑，或调整 dist_atr_mult")
    else:
        print(f"✅ 盈利样本比例提升 {profit_rate_change:.1f}%")
        print("   说明：SR过滤保留了更多盈利样本")

    if mean_change < 0:
        print(f"⚠️  标签均值下降 {abs(mean_change):.4f}")
        print("   说明：过滤后的样本平均R/R值更低")
        print("   可能原因：过滤掉了更多高R/R的样本")
    else:
        print(f"✅ 标签均值提升 {mean_change:.4f}")
        print("   说明：过滤后的样本平均R/R值更高")

    if std_change < 0:
        print(f"⚠️  标签标准差下降 {abs(std_change):.4f}")
        print("   说明：过滤后的样本R/R值分布更集中")
        print("   可能影响：模型学习难度增加")
    else:
        print(f"✅ 标签标准差提升 {std_change:.4f}")
        print("   说明：过滤后的样本R/R值分布更分散")
        print("   可能影响：模型学习难度降低")

    print()
    print("=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
