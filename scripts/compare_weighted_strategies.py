#!/usr/bin/env python3
"""
对比带权重和不带权重的策略性能

Usage:
    python scripts/compare_weighted_strategies.py --symbol BTCUSDT --timeframe 240T
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_training(
    config_path: str, symbol: str, timeframe: str, data_path: str, output_root: str
) -> Dict[str, Any]:
    """运行单个策略的训练"""
    print(f"\n{'='*80}")
    print(f"🚀 训练策略: {config_path}")
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
        print(result.stderr)
        return {"success": False, "error": result.stderr}

    # 尝试从输出中提取关键指标
    output_lines = result.stdout.split("\n")
    metrics = {}

    # 查找关键指标
    for line in output_lines:
        if "Average CV Metric" in line:
            try:
                metric_value = float(line.split(":")[-1].strip())
                metrics["avg_cv_metric"] = metric_value
            except:
                pass
        if "Precision" in line or "precision" in line.lower():
            metrics["precision_info"] = line.strip()
        if "Recall" in line or "recall" in line.lower():
            metrics["recall_info"] = line.strip()
        if "F1" in line or "f1" in line.lower():
            metrics["f1_info"] = line.strip()

    return {"success": True, "metrics": metrics, "output": result.stdout}


def load_results(output_dir: Path, strategy_name: str) -> Dict[str, Any]:
    """从结果文件中加载指标"""
    results_file = output_dir / strategy_name / "results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="对比带权重和不带权重的策略")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易符号")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/strategies_comparison",
        help="输出根目录",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    strategies = [
        {
            "name": "sr_reversal_long",
            "config": "config/strategies/sr_reversal_long",
            "description": "全量扫描，无权重",
        },
        {
            "name": "sr_reversal_long_weighted",
            "config": "config/strategies/sr_reversal_long_weighted",
            "description": "全量扫描，带样本权重（result_based_rr）",
        },
    ]

    results = {}

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"📊 策略: {strategy['name']}")
        print(f"   描述: {strategy['description']}")
        print(f"{'='*80}")

        result = run_training(
            strategy["config"],
            args.symbol,
            args.timeframe,
            args.data_path,
            str(output_root),
        )

        results[strategy["name"]] = {
            "description": strategy["description"],
            "training_result": result,
            "file_results": load_results(output_root, strategy["name"]),
        }

    # 打印对比结果
    print(f"\n{'='*80}")
    print("📊 对比结果汇总")
    print(f"{'='*80}")

    for name, data in results.items():
        print(f"\n{name} ({data['description']}):")
        if data["training_result"]["success"]:
            metrics = data["training_result"].get("metrics", {})
            if "avg_cv_metric" in metrics:
                print(f"  ✅ 平均CV指标: {metrics['avg_cv_metric']:.4f}")
            if "precision_info" in metrics:
                print(f"  📈 {metrics['precision_info']}")
            if "recall_info" in metrics:
                print(f"  📈 {metrics['recall_info']}")
            if "f1_info" in metrics:
                print(f"  📈 {metrics['f1_info']}")

            # 从文件结果中提取更多指标
            file_results = data.get("file_results", {})
            if file_results:
                print(f"  📁 详细结果文件: {output_root / name / 'results.json'}")
        else:
            print(
                f"  ❌ 训练失败: {data['training_result'].get('error', 'Unknown error')}"
            )

    # 保存对比结果
    comparison_file = output_root / "comparison_results.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 对比结果已保存到: {comparison_file}")
    print(f"\n✅ 对比完成！")


if __name__ == "__main__":
    main()
