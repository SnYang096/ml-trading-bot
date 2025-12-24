#!/usr/bin/env python3
"""
训练所有6个策略并生成对比报告

6个策略：
1. sr_reversal_long: 二分类（无权重，全量扫描）
2. sr_reversal_long_sr_filter: 二分类（无权重，SR过滤）
3. sr_reversal_long_weighted: 二分类（带权重，SR过滤）
4. sr_reversal_rr_reg_long: 回归（无权重，全量扫描）
5. sr_reversal_rr_reg_long_sr_filter: 回归（无权重，SR过滤）
6. sr_reversal_rr_reg_long_weighted: 回归（带权重，SR过滤）
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

    return {"success": True, "output": result.stdout}


def load_results(output_dir: Path, strategy_name: str) -> Dict[str, Any]:
    """从结果文件中加载指标"""
    # 尝试多个可能的结果文件位置
    possible_paths = [
        output_dir / strategy_name / "results.json",
        output_dir / "sr_reversal" / "results.json",  # 某些策略可能使用这个路径
    ]

    for results_file in possible_paths:
        if results_file.exists():
            with open(results_file, "r") as f:
                data = json.load(f)
            return data

    return {}


def main():
    parser = argparse.ArgumentParser(description="训练所有6个策略并生成对比报告")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易符号")
    parser.add_argument("--timeframe", type=str, default="240T", help="时间周期")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/strategies_comparison_6",
        help="输出根目录",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义6个策略
    strategies = [
        {
            "name": "sr_reversal_long",
            "type": "binary",
            "weighted": False,
            "sr_filter": False,
            "description": "二分类（无权重，全量扫描）",
        },
        {
            "name": "sr_reversal_long_sr_filter",
            "type": "binary",
            "weighted": False,
            "sr_filter": True,
            "description": "二分类（无权重，SR过滤）",
        },
        {
            "name": "sr_reversal_long_weighted",
            "type": "binary",
            "weighted": True,
            "sr_filter": True,
            "description": "二分类（带权重，SR过滤）",
        },
        {
            "name": "sr_reversal_rr_reg_long",
            "type": "regression",
            "weighted": False,
            "sr_filter": False,
            "description": "回归（无权重，全量扫描）",
        },
        {
            "name": "sr_reversal_rr_reg_long_sr_filter",
            "type": "regression",
            "weighted": False,
            "sr_filter": True,
            "description": "回归（无权重，SR过滤）",
        },
        {
            "name": "sr_reversal_rr_reg_long_weighted",
            "type": "regression",
            "weighted": True,
            "sr_filter": True,
            "description": "回归（带权重，SR过滤）",
        },
    ]

    print("=" * 80)
    print("📊 开始训练6个策略")
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
                }
            else:
                results[strategy["name"]] = {
                    **strategy,
                }
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
    print("\n" + "=" * 100)
    print("📊 完整对比结果")
    print("=" * 100)
    print()

    print(
        f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12} {'总收益%':<12} {'夏普比率':<12} {'最大回撤%':<12} {'交易次数':<10}"
    )
    print("-" * 100)

    for strategy in strategies:
        name = strategy["name"]
        data = results.get(name, {})

        if data.get("success", True) and not data.get("error"):
            n_train = data.get("n_train_samples", "N/A")
            cv_metric = data.get("avg_cv_metric", "N/A")
            correlation = data.get("evaluation", {}).get("test_correlation", "N/A")

            bt = data.get("backtest", {})
            total_return = bt.get("total_return_pct", "N/A")
            sharpe = bt.get("sharpe", "N/A")
            max_dd = bt.get("max_drawdown_pct", "N/A")
            trades = bt.get("total_trades", "N/A")

            import numpy as np

            cv_str = (
                f"{cv_metric:.4f}"
                if isinstance(cv_metric, (int, float))
                and not (isinstance(cv_metric, float) and np.isnan(cv_metric))
                else str(cv_metric)
            )
            corr_str = (
                f"{correlation:.4f}"
                if isinstance(correlation, (int, float))
                and not (isinstance(correlation, float) and np.isnan(correlation))
                else str(correlation)
            )
            ret_str = (
                f"{total_return:.2f}%"
                if isinstance(total_return, (int, float))
                and not (isinstance(total_return, float) and np.isnan(total_return))
                else str(total_return)
            )
            sharpe_str = (
                f"{sharpe:.4f}"
                if isinstance(sharpe, (int, float))
                and not (isinstance(sharpe, float) and np.isnan(sharpe))
                else str(sharpe)
            )
            dd_str = (
                f"{max_dd:.2f}%"
                if isinstance(max_dd, (int, float))
                and not (isinstance(max_dd, float) and np.isnan(max_dd))
                else str(max_dd)
            )
            trades_str = (
                str(int(trades))
                if isinstance(trades, (int, float))
                and not (isinstance(trades, float) and np.isnan(trades))
                else str(trades)
            )

            print(
                f"{strategy['description']:<40} {str(n_train):<12} {cv_str:<12} {corr_str:<12} {ret_str:<12} {sharpe_str:<12} {dd_str:<12} {trades_str:<10}"
            )
        else:
            print(
                f"{strategy['description']:<40} {'失败':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )

    print()

    # 保存详细结果
    results_file = output_dir / "all_strategies_comparison.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 详细结果已保存到: {results_file}")
    print()
    print("=" * 100)
    print("✅ 所有策略训练完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
