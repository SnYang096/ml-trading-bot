#!/usr/bin/env python3
"""
对比4个策略的性能：二分类（无权重/带权重）+ 回归（无权重/带权重）

Usage:
    python scripts/compare_all_strategies.py --symbol BTCUSDT --timeframe 240T
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_training(
    config_name: str, symbol: str, timeframe: str, data_path: str, output_root: str
) -> Dict[str, Any]:
    """运行单个策略的训练"""
    config_path = f"config/strategies/{config_name}"

    print(f"\n{'='*80}")
    print(f"🚀 训练策略: {config_name}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        "scripts/train_strategy_pipeline.py",
        "--config",
        config_path,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--data-path",
        data_path,
        "--output-root",
        output_root,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 训练失败:")
        print(result.stderr[:500])
        return {"success": False, "error": result.stderr[:500]}

    # 从输出中提取关键指标
    output_lines = result.stdout.split("\n")
    metrics = {}

    for line in output_lines:
        if "Average CV Metric" in line:
            try:
                metric_value = float(line.split(":")[-1].strip())
                metrics["avg_cv_metric"] = metric_value
            except:
                pass
        if "test_correlation" in line or "test correlation" in line.lower():
            try:
                corr_value = float(line.split(":")[-1].strip())
                metrics["test_correlation"] = corr_value
            except:
                pass
        if "Label stats before filtering" in line:
            # 提取训练样本数
            if "Train non-null:" in line:
                try:
                    parts = line.split("Train non-null:")[1].split(",")[0].strip()
                    metrics["n_train_samples"] = int(parts)
                except:
                    pass

    return {"success": True, "metrics": metrics, "output": result.stdout}


def load_results(output_dir: Path, strategy_name: str) -> Dict[str, Any]:
    """从结果文件中加载指标"""
    results_file = output_dir / strategy_name / "results.json"
    if not results_file.exists():
        return {}

    with open(results_file, "r") as f:
        data = json.load(f)

    return data


def main():
    parser = argparse.ArgumentParser(description="对比4个策略的性能")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易符号")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/strategies_comparison_all",
        help="输出根目录",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义4个策略
    strategies = [
        {
            "name": "sr_reversal_long",
            "type": "binary",
            "weighted": False,
            "description": "二分类（无权重，全量扫描）",
        },
        {
            "name": "sr_reversal_long_weighted",
            "type": "binary",
            "weighted": True,
            "description": "二分类（带权重，SR过滤）",
        },
        {
            "name": "sr_reversal_rr_reg_long",
            "type": "regression",
            "weighted": False,
            "description": "回归（无权重，全量扫描）",
        },
        {
            "name": "sr_reversal_rr_reg_long_weighted",
            "type": "regression",
            "weighted": True,
            "description": "回归（带权重，SR过滤）",
        },
    ]

    print("=" * 80)
    print("📊 开始对比4个策略")
    print("=" * 80)
    print()
    print("策略列表:")
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s['name']}: {s['description']}")
    print()

    # 训练所有策略
    results = {}
    for strategy in strategies:
        result = run_training(
            strategy["name"],
            args.symbol,
            args.timeframe,
            args.data_path,
            args.output_root,
        )

        if result["success"]:
            # 尝试从结果文件加载完整数据
            file_data = load_results(output_dir, strategy["name"])
            if file_data:
                results[strategy["name"]] = {
                    **strategy,
                    **file_data,
                    "metrics_from_output": result.get("metrics", {}),
                }
            else:
                results[strategy["name"]] = {**strategy, **result.get("metrics", {})}
            print(f"   ✅ {strategy['name']} 训练完成")
        else:
            print(
                f"   ❌ {strategy['name']} 训练失败: {result.get('error', 'Unknown error')}"
            )
            results[strategy["name"]] = {
                **strategy,
                "success": False,
                "error": result.get("error", "Unknown error"),
            }

    # 生成对比报告
    print("\n" + "=" * 80)
    print("📊 对比结果")
    print("=" * 80)
    print()

    # 二分类对比
    print("1️⃣  二分类策略对比")
    print("-" * 80)
    binary_results = {k: v for k, v in results.items() if v.get("type") == "binary"}
    if binary_results:
        print(f"{'策略':<35} {'训练样本':<12} {'CV指标':<12} {'相关性':<12}")
        print("-" * 80)
        for name, data in binary_results.items():
            if data.get("success", True):
                n_train = data.get(
                    "n_train_samples",
                    data.get("metrics_from_output", {}).get("n_train_samples", "N/A"),
                )
                cv_metric = data.get(
                    "avg_cv_metric",
                    data.get("metrics_from_output", {}).get("avg_cv_metric", "N/A"),
                )
                correlation = data.get("evaluation", {}).get(
                    "test_correlation",
                    data.get("metrics_from_output", {}).get("test_correlation", "N/A"),
                )

                if isinstance(cv_metric, (int, float)):
                    cv_str = f"{cv_metric:.4f}"
                else:
                    cv_str = str(cv_metric)

                if isinstance(correlation, (int, float)):
                    corr_str = f"{correlation:.4f}"
                else:
                    corr_str = str(correlation)

                print(
                    f"{data.get('description', name):<35} {str(n_train):<12} {cv_str:<12} {corr_str:<12}"
                )
            else:
                print(
                    f"{data.get('description', name):<35} {'失败':<12} {'N/A':<12} {'N/A':<12}"
                )

    print()

    # 回归对比
    print("2️⃣  回归策略对比")
    print("-" * 80)
    regression_results = {
        k: v for k, v in results.items() if v.get("type") == "regression"
    }
    if regression_results:
        print(f"{'策略':<35} {'训练样本':<12} {'CV指标':<12} {'相关性':<12}")
        print("-" * 80)
        for name, data in regression_results.items():
            if data.get("success", True):
                n_train = data.get(
                    "n_train_samples",
                    data.get("metrics_from_output", {}).get("n_train_samples", "N/A"),
                )
                cv_metric = data.get(
                    "avg_cv_metric",
                    data.get("metrics_from_output", {}).get("avg_cv_metric", "N/A"),
                )
                correlation = data.get("evaluation", {}).get(
                    "test_correlation",
                    data.get("metrics_from_output", {}).get("test_correlation", "N/A"),
                )

                if isinstance(cv_metric, (int, float)):
                    cv_str = f"{cv_metric:.4f}"
                else:
                    cv_str = str(cv_metric)

                if isinstance(correlation, (int, float)):
                    corr_str = f"{correlation:.4f}"
                else:
                    corr_str = str(correlation)

                print(
                    f"{data.get('description', name):<35} {str(n_train):<12} {cv_str:<12} {corr_str:<12}"
                )
            else:
                print(
                    f"{data.get('description', name):<35} {'失败':<12} {'N/A':<12} {'N/A':<12}"
                )

    print()

    # 保存详细结果
    results_file = output_dir / "comparison_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 详细结果已保存到: {results_file}")
    print()
    print("=" * 80)
    print("✅ 对比完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
