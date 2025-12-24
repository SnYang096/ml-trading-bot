#!/usr/bin/env python3
"""
分析带权重和不带权重策略的对比结果

Usage:
    python scripts/analyze_weight_comparison.py --results-dir results/strategies_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_results(results_dir: Path) -> Dict[str, Any]:
    """加载所有结果文件"""
    results = {}

    # 查找所有 results.json 文件
    for results_file in results_dir.rglob("results.json"):
        strategy_name = results_file.parent.name
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                results[strategy_name] = data
        except Exception as e:
            print(f"⚠️  无法读取 {results_file}: {e}")

    return results


def compare_strategies(results: Dict[str, Any]) -> None:
    """对比策略结果"""
    print("=" * 80)
    print("📊 二分类反转策略对比：带权重 vs 无权重")
    print("=" * 80)
    print()

    # 提取关键指标
    baseline = results.get("sr_reversal", {})
    weighted = results.get("sr_reversal_long_weighted", {})

    if not baseline and not weighted:
        print("❌ 未找到结果文件")
        return

    print("1️⃣  无权重版本 (sr_reversal_long)")
    print("-" * 80)
    if baseline:
        print(
            f"   ✅ 平均CV指标: {baseline.get('avg_cv_metric', 'N/A'):.4f}"
            if isinstance(baseline.get("avg_cv_metric"), (int, float))
            else f"   ⚠️  平均CV指标: {baseline.get('avg_cv_metric', 'N/A')}"
        )
        print(
            f"   ✅ 测试集相关性: {baseline.get('test_correlation', 'N/A'):.4f}"
            if isinstance(baseline.get("test_correlation"), (int, float))
            else f"   ⚠️  测试集相关性: {baseline.get('test_correlation', 'N/A')}"
        )
        if "backtest" in baseline:
            bt = baseline["backtest"]
            print(f"   📈 回测Sharpe: {bt.get('sharpe_ratio', 'N/A')}")
            print(f"   📈 回测胜率: {bt.get('win_rate', 'N/A')}")
    else:
        print("   ❌ 结果未找到")

    print()
    print("2️⃣  带权重版本 (sr_reversal_long_weighted)")
    print("-" * 80)
    if weighted:
        print(
            f"   ✅ 平均CV指标: {weighted.get('avg_cv_metric', 'N/A'):.4f}"
            if isinstance(weighted.get("avg_cv_metric"), (int, float))
            else f"   ⚠️  平均CV指标: {weighted.get('avg_cv_metric', 'N/A')}"
        )
        print(
            f"   ✅ 测试集相关性: {weighted.get('test_correlation', 'N/A'):.4f}"
            if isinstance(weighted.get("test_correlation"), (int, float))
            else f"   ⚠️  测试集相关性: {weighted.get('test_correlation', 'N/A')}"
        )
        if "backtest" in weighted:
            bt = weighted["backtest"]
            print(f"   📈 回测Sharpe: {bt.get('sharpe_ratio', 'N/A')}")
            print(f"   📈 回测胜率: {bt.get('win_rate', 'N/A')}")
    else:
        print("   ❌ 结果未找到")

    print()
    print("3️⃣  对比分析")
    print("-" * 80)

    if baseline and weighted:
        baseline_cv = baseline.get("avg_cv_metric")
        weighted_cv = weighted.get("avg_cv_metric")

        if isinstance(baseline_cv, (int, float)) and isinstance(
            weighted_cv, (int, float)
        ):
            cv_diff = weighted_cv - baseline_cv
            cv_improvement = (
                (cv_diff / abs(baseline_cv) * 100) if baseline_cv != 0 else 0
            )
            print(f"   CV指标变化: {cv_diff:+.4f} ({cv_improvement:+.2f}%)")
            if cv_diff > 0:
                print(f"   ✅ 带权重版本CV指标更高（提升 {cv_improvement:.2f}%）")
            elif cv_diff < 0:
                print(f"   ⚠️  带权重版本CV指标更低（下降 {abs(cv_improvement):.2f}%）")
            else:
                print(f"   ➡️  两个版本CV指标相同")

        baseline_corr = baseline.get("test_correlation")
        weighted_corr = weighted.get("test_correlation")

        if isinstance(baseline_corr, (int, float)) and isinstance(
            weighted_corr, (int, float)
        ):
            corr_diff = weighted_corr - baseline_corr
            print(f"   测试集相关性变化: {corr_diff:+.4f}")
            if corr_diff > 0:
                print(f"   ✅ 带权重版本相关性更高")
            elif corr_diff < 0:
                print(f"   ⚠️  带权重版本相关性更低")
            else:
                print(f"   ➡️  两个版本相关性相同")

    print()
    print("=" * 80)
    print("💡 建议")
    print("=" * 80)
    print("1. 如果CV指标提升，说明样本权重有效，模型更关注高质量信号")
    print("2. 如果相关性提升，说明预测质量更好")
    print("3. 如果两个指标都提升，建议使用带权重版本")
    print("4. 如果指标下降，可能需要调整权重策略参数")


def main():
    parser = argparse.ArgumentParser(description="分析权重对比结果")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/strategies_comparison",
        help="结果目录",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    results = load_results(results_dir)

    if not results:
        print(f"❌ 在 {results_dir} 中未找到结果文件")
        return

    compare_strategies(results)


if __name__ == "__main__":
    main()
