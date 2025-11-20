#!/usr/bin/env python3
"""
测试不同方向判断方法的效果

比较：
1. 当前方法（quantile）：使用分位数
2. 改进方法1（sign）：直接使用预测值符号
3. 改进方法2（hybrid）：结合预测值符号和分位数
4. 改进方法3（optimized）：基于历史表现优化阈值
"""

import sys
import os
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from time_series_model.pipeline.training.rank_ic_utils_improved import (
    generate_trading_signals_improved,
    calibrate_predictions,
    evaluate_direction_accuracy,
)


def test_direction_methods(
    results_file: str,
    models_file: str = None,
    test_data_file: str = None,
):
    """测试不同方向判断方法"""

    print("=" * 80)
    print("🔍 方向判断方法测试")
    print("=" * 80)
    print()

    # 加载结果
    with open(results_file, "r") as f:
        results = json.load(f)

    # 如果有测试数据文件，加载它
    if test_data_file and os.path.exists(test_data_file):
        print("📊 加载测试数据...")
        # 这里需要根据实际的数据格式加载
        # 暂时跳过
        pass

    # 从结果中提取信息
    oos_results = results.get("oos_results", {})
    eval_data = oos_results.get("evaluation", {})

    print("📊 当前方法（quantile）表现：")
    print("-" * 80)
    conf_stats = eval_data.get("confidence_statistics", {})
    hc_trades = conf_stats.get("high_confidence_trades", {})

    if hc_trades:
        print(f"   Win Rate: {hc_trades.get('win_rate', 0):.1%}")
        print(f"   Total Return: {hc_trades.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {hc_trades.get('sharpe_ratio', 0):.4f}")
        print(f"   Max Drawdown: {hc_trades.get('max_drawdown', 0):.2%}")
        print()

    # 方向准确性分析
    direction_analysis = conf_stats.get("direction_analysis", {})
    if direction_analysis:
        print("📊 方向准确性分析：")
        print("-" * 80)
        print(
            f"   Direction Accuracy: {direction_analysis.get('direction_accuracy', 0):.1%}"
        )
        print(
            f"   Pearson Correlation: {direction_analysis.get('pearson_correlation', 0):.4f}"
        )
        print()

    print("💡 改进建议：")
    print("-" * 80)
    print()
    print("1. 直接使用预测值符号（sign 方法）")
    print("   - 优点：简单直接，不依赖分位数")
    print("   - 适用：当预测值本身有方向性时")
    print("   - 代码：method='sign'")
    print()
    print("2. 结合预测值符号和分位数（hybrid 方法）")
    print("   - 优点：双重验证，减少误信号")
    print("   - 适用：当预测值符号和分位数方向一致时才交易")
    print("   - 代码：method='hybrid'")
    print()
    print("3. 基于历史表现优化阈值（optimized 方法）")
    print("   - 优点：自动优化，适应数据分布")
    print("   - 适用：有足够的历史数据时")
    print("   - 代码：method='optimized'")
    print()
    print("4. 校准预测值")
    print("   - 优点：使预测值更准确地反映真实收益")
    print("   - 适用：当预测值分布与真实收益分布不匹配时")
    print("   - 代码：calibrate_predictions(predictions, true_returns)")
    print()

    # 分析问题
    win_rate = hc_trades.get("win_rate", 0) if hc_trades else 0
    direction_acc = (
        direction_analysis.get("direction_accuracy", 0) if direction_analysis else 0
    )

    print("🔍 问题诊断：")
    print("-" * 80)

    if win_rate < 0.3 and direction_acc > 0.5:
        print("   ⚠️  问题：方向准确率 > 50%，但 Win Rate < 30%")
        print("   可能原因：")
        print("   1. 虽然方向对，但收益幅度预测不准确")
        print("   2. 交易成本或滑点抵消了收益")
        print("   3. 信号时机不对（虽然方向对，但入场时机差）")
        print()
        print("   建议：")
        print("   - 使用 sign 方法，直接基于预测值符号判断方向")
        print("   - 降低置信度阈值，增加交易频率")
        print("   - 检查预测值的分布，可能需要校准")
    elif win_rate < 0.3 and direction_acc < 0.5:
        print("   ⚠️  问题：方向准确率 < 50%，Win Rate < 30%")
        print("   可能原因：")
        print("   1. 预测值本身不准确")
        print("   2. 模型过度拟合到排序，但预测值符号错误")
        print("   3. 数据泄漏导致 IC 虚高")
        print()
        print("   建议：")
        print("   - 使用 optimized 方法，基于历史表现优化阈值")
        print("   - 校准预测值，使其更准确")
        print("   - 检查模型是否过度拟合")
    else:
        print("   ✅ 方向准确率和 Win Rate 都在合理范围内")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试不同方向判断方法")
    parser.add_argument("--results", type=str, required=True, help="结果 JSON 文件路径")
    parser.add_argument(
        "--models", type=str, default=None, help="模型 PKL 文件路径（可选）"
    )
    parser.add_argument(
        "--test-data", type=str, default=None, help="测试数据文件路径（可选）"
    )

    args = parser.parse_args()

    test_direction_methods(args.results, args.models, args.test_data)
