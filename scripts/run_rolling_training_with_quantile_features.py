#!/usr/bin/env python3
"""
滚动训练脚本：验证"短时间段表现好"的假设

使用相对阈值（分位数）特征来适应 regime shift
"""

import subprocess
import sys
from pathlib import Path


def main():
    strategy_config = "config/strategies/sr_reversal_rr_reg_long"
    symbol = "BTCUSDT"
    timeframe = "240T"  # 4小时K线
    initial_train_months = 6
    min_train_months = 3

    print("=" * 80)
    print("🔄 滚动训练：验证短时间段表现好的假设")
    print("=" * 80)
    print(f"策略: {strategy_config}")
    print(f"交易对: {symbol}")
    print(f"时间周期: {timeframe}")
    print(f"初始训练月数: {initial_train_months}")
    print(f"最小训练月数: {min_train_months}")
    print()
    print("⭐ 使用相对阈值（分位数）特征：")
    print("  - atr_percentile_f: 自适应 regime shift")
    print("  - 滚动窗口分位数，替代硬阈值")
    print()
    print("预期结果：")
    print("  - 短时间段（6个月训练 → 1个月预测）Sharpe 可能提升 20-50%")
    print("  - 自动适应市场 regime 变化")
    print("=" * 80)
    print()

    # 构建命令
    cmd = [
        "python3",
        "src/time_series_model/pipeline/rolling/rolling_train.py",
        "--config",
        strategy_config,
        "--symbol",
        symbol,
        "--data-dir",
        "data/parquet_data",
        "--timeframe",
        timeframe,
        "--initial-train-months",
        str(initial_train_months),
        "--min-train-months",
        str(min_train_months),
        "--output-root",
        "results/rolling",
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print()

    # 执行滚动训练
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
