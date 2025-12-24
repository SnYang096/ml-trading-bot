#!/usr/bin/env python3
"""
对比修复后的两个策略性能

Usage:
    python scripts/compare_fixed_strategies.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main():
    results_dir = Path("results/strategies_comparison_fixed")

    print("=" * 80)
    print("📊 模型性能对比（修复后）")
    print("=" * 80)
    print()

    # 读取结果文件
    results_file = results_dir / "sr_reversal" / "results.json"

    if not results_file.exists():
        print("❌ 结果文件不存在")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    print("⚠️  注意：两个策略的结果都保存在同一个文件中（因为策略名称相同）")
    print()
    print("从训练日志分析:")
    print()

    # 从日志中提取信息
    print("1️⃣  无权重版本 (sr_reversal_long)")
    print("-" * 80)
    print("   标签统计: Train=10823, Test=1857")
    print("   平均CV指标: 0.0237")
    print("   测试集相关性: 0.0634")
    print()

    print("2️⃣  带权重版本 (sr_reversal_long_weighted)")
    print("-" * 80)
    print("   标签统计: Train=10823, Test=1857")
    print("   平均CV指标: 0.0235")
    print("   测试集相关性: 0.0631")
    print("   样本权重: 49.15%样本被加权，权重范围[0.06, 2.73]")
    print()

    print("=" * 80)
    print("📊 对比分析")
    print("=" * 80)
    print()

    print("标签数量:")
    print("   两个版本标签数量相同: 10823训练，1857测试")
    print("   ⚠️  说明SR过滤可能未生效")
    print()

    print("可能原因:")
    print("   1. dist_to_nearest_sr 列在标签生成时不存在")
    print("   2. 所有样本都在SR附近（1.5 ATR内）")
    print("   3. 需要先计算 sr_strength_max_close_f 特征才能生成 dist_to_nearest_sr")
    print()

    print("模型性能:")
    cv_diff = 0.0235 - 0.0237
    print(f"   CV指标差异: {cv_diff:+.4f}")
    if cv_diff > 0:
        print("   ✅ 带权重版本CV指标更高")
    elif cv_diff < 0:
        print("   ⚠️  带权重版本CV指标略低")
    else:
        print("   ➡️  两个版本CV指标相同")

    corr_diff = 0.0631 - 0.0634
    print(f"   相关性差异: {corr_diff:+.4f}")
    print()

    print("=" * 80)
    print("💡 建议")
    print("=" * 80)
    print("1. 检查 dist_to_nearest_sr 列是否在标签生成时存在")
    print("2. 如果不存在，需要确保 sr_strength_max_close_f 特征已计算")
    print("3. 或者使用 is_near_sr 特征（需要先在 features.yaml 中添加）")
    print()


if __name__ == "__main__":
    main()
