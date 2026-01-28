#!/usr/bin/env python3
"""
运行所有策略的短时间（滚动训练）和长时间（固定训练）测试

短时间测试：6个月训练 → 1个月预测（滚动训练）
长时间测试：3年训练 → 3年预测（固定训练，使用 train_strategy_pipeline.py）
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import time

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]

SYMBOL = "BTCUSDT"
TIMEFRAME = "240T"  # 4小时K线


def run_rolling_training(strategy: str):
    """运行滚动训练（短时间测试）"""
    print(f"\n{'='*80}")
    print(f"🔄 滚动训练（短时间测试）: {strategy}")
    print(f"{'='*80}")

    output_dir = f"results/rolling_short/{strategy}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "src/time_series_model/pipeline/rolling/rolling_train.py",
        "--config",
        f"config/strategies/{strategy}",
        "--symbol",
        SYMBOL,
        "--data-dir",
        "data/parquet_data",
        "--timeframe",
        TIMEFRAME,
        "--initial-train-months",
        "6",
        "--min-train-months",
        "3",
        "--output-root",
        output_dir,
    ]

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {strategy} 滚动训练完成")
        print(f"   结果目录: {output_dir}")
    else:
        print(f"❌ {strategy} 滚动训练失败:")
        print(result.stderr[:1000])

    return result.returncode, output_dir


def run_fixed_training(strategy: str):
    """运行固定训练（长时间测试）"""
    print(f"\n{'='*80}")
    print(f"📊 固定训练（长时间测试）: {strategy}")
    print(f"{'='*80}")

    output_dir = f"results/fixed_long/{strategy}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 使用 train_strategy_pipeline.py 进行固定训练
    # 训练期：2023-01-01 到 2025-12-31（3年）
    cmd = [
        "python3",
        "scripts/train_strategy_pipeline.py",
        "--config",
        f"config/strategies/{strategy}",
        "--symbol",
        SYMBOL,
        "--data-path",
        "data/parquet_data",
        "--timeframe",
        TIMEFRAME,
        "--start-date",
        "2023-01-01",
        "--end-date",
        "2025-12-31",
        "--test-size",
        "0.15",  # 15% 作为测试集
        "--output-root",
        output_dir,
        "--seed",
        "42",
    ]

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {strategy} 固定训练完成")
        print(f"   结果目录: {output_dir}")
    else:
        print(f"❌ {strategy} 固定训练失败:")
        print(result.stderr[:1000])

    return result.returncode, output_dir


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "strategies": {},
        "summary": {
            "rolling_short": {"success": 0, "failed": 0},
            "fixed_long": {"success": 0, "failed": 0},
        },
    }

    print("=" * 80)
    print("🚀 启动所有策略测试")
    print("=" * 80)
    print(f"策略列表: {', '.join(STRATEGIES)}")
    print(f"交易对: {SYMBOL}")
    print(f"时间周期: {TIMEFRAME}")
    print()
    print("测试类型:")
    print("  1. 短时间测试（滚动训练）: 6个月训练 → 1个月预测")
    print("     - 输出目录: results/rolling_short/<strategy>/")
    print("  2. 长时间测试（固定训练）: 3年训练（2023-2025）→ 15%测试集")
    print("     - 输出目录: results/fixed_long/<strategy>/")
    print("=" * 80)

    # 创建输出目录
    Path("results/rolling_short").mkdir(parents=True, exist_ok=True)
    Path("results/fixed_long").mkdir(parents=True, exist_ok=True)

    # 运行所有测试
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n{'#'*80}")
        print(f"策略 {i}/{len(STRATEGIES)}: {strategy}")
        print(f"{'#'*80}")

        strategy_results = {
            "rolling_short": None,
            "fixed_long": None,
        }

        # 短时间测试（滚动训练）
        print(f"\n[1/2] 短时间测试（滚动训练）")
        rolling_code, rolling_dir = run_rolling_training(strategy)
        strategy_results["rolling_short"] = {
            "status": "success" if rolling_code == 0 else "failed",
            "return_code": rolling_code,
            "output_dir": rolling_dir,
        }
        if rolling_code == 0:
            results["summary"]["rolling_short"]["success"] += 1
        else:
            results["summary"]["rolling_short"]["failed"] += 1

        # 等待一下，避免资源竞争
        time.sleep(2)

        # 长时间测试（固定训练）
        print(f"\n[2/2] 长时间测试（固定训练）")
        fixed_code, fixed_dir = run_fixed_training(strategy)
        strategy_results["fixed_long"] = {
            "status": "success" if fixed_code == 0 else "failed",
            "return_code": fixed_code,
            "output_dir": fixed_dir,
        }
        if fixed_code == 0:
            results["summary"]["fixed_long"]["success"] += 1
        else:
            results["summary"]["fixed_long"]["failed"] += 1

        results["strategies"][strategy] = strategy_results

        # 等待一下，避免资源竞争
        if i < len(STRATEGIES):
            time.sleep(5)

    # 保存结果摘要
    results_file = Path(f"results/test_summary_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("📊 测试摘要")
    print(f"{'='*80}")
    for strategy, strategy_results in results["strategies"].items():
        rolling_status = strategy_results["rolling_short"]["status"]
        fixed_status = strategy_results["fixed_long"]["status"]
        print(f"\n{strategy}:")
        print(f"  滚动训练（短时间）: {rolling_status}")
        if rolling_status == "success":
            print(f"    → {strategy_results['rolling_short']['output_dir']}")
        print(f"  固定训练（长时间）: {fixed_status}")
        if fixed_status == "success":
            print(f"    → {strategy_results['fixed_long']['output_dir']}")

    print(f"\n总体统计:")
    print(
        f"  滚动训练: {results['summary']['rolling_short']['success']}/{len(STRATEGIES)} 成功"
    )
    print(
        f"  固定训练: {results['summary']['fixed_long']['success']}/{len(STRATEGIES)} 成功"
    )

    print(f"\n结果摘要已保存到: {results_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
