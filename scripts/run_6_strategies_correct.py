#!/usr/bin/env python3
"""
正确运行6个策略的训练，确保每个策略的结果保存在不同目录
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json
import numpy as np


def run_single_strategy(
    config_name: str, symbol: str, timeframe: str, output_root: str
) -> dict:
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
        "--output-root",
        output_root,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"❌ 训练失败:")
        print(result.stderr[:500])
        return {"success": False, "error": result.stderr[:500]}

    # 读取结果文件
    # 结果文件应该保存在 output_root / strategy_name / results.json
    # 需要从配置文件中读取strategy name
    import yaml

    labels_file = Path(config_path) / "labels.yaml"
    if labels_file.exists():
        with open(labels_file, "r") as f:
            config_data = yaml.safe_load(f)
        strategy_name = config_data.get("name", config_name)
    else:
        strategy_name = config_name

    results_file = Path(output_root) / strategy_name / "results.json"

    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
        return {"success": True, "data": data, "strategy_name": strategy_name}
    else:
        return {"success": True, "data": None, "strategy_name": strategy_name}


def main():
    symbol = "BTCUSDT"
    timeframe = "240T"
    output_root = "results/strategies_comparison_6_correct"

    strategies = [
        "sr_reversal_long",
        "sr_reversal_long_sr_filter",
        "sr_reversal_long_weighted",
        "sr_reversal_rr_reg_long",
        "sr_reversal_rr_reg_long_sr_filter",
        "sr_reversal_rr_reg_long_weighted",
    ]

    strategy_descriptions = {
        "sr_reversal_long": "二分类（无权重，全量扫描）",
        "sr_reversal_long_sr_filter": "二分类（无权重，SR过滤）",
        "sr_reversal_long_weighted": "二分类（带权重，SR过滤）",
        "sr_reversal_rr_reg_long": "回归（无权重，全量扫描）",
        "sr_reversal_rr_reg_long_sr_filter": "回归（无权重，SR过滤）",
        "sr_reversal_rr_reg_long_weighted": "回归（带权重，SR过滤）",
    }

    print("=" * 120)
    print("📊 训练6个策略（确保结果保存在不同目录）")
    print("=" * 120)
    print()

    results = {}
    for strategy in strategies:
        result = run_single_strategy(strategy, symbol, timeframe, output_root)
        results[strategy] = result

    # 生成对比报告
    print("\n" + "=" * 120)
    print("📊 6个策略完整对比（包含夏普比率）")
    print("=" * 120)
    print()
    print(
        f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12} {'总收益%':<12} {'夏普比率':<12} {'最大回撤%':<12} {'交易次数':<10}"
    )
    print("-" * 120)

    for strategy, desc in strategy_descriptions.items():
        result = results.get(strategy, {})

        if result.get("success") and result.get("data"):
            data = result["data"]
            n_train = data.get("n_train_samples", "N/A")
            cv_metric = data.get("avg_cv_metric", "N/A")
            correlation = data.get("evaluation", {}).get("test_correlation", "N/A")

            bt = data.get("backtest", {})
            total_return = bt.get("total_return_pct", "N/A")
            sharpe = bt.get("sharpe", "N/A")
            max_dd = bt.get("max_drawdown_pct", "N/A")
            trades = bt.get("total_trades", "N/A")

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
                f"{desc:<40} {str(n_train):<12} {cv_str:<12} {corr_str:<12} {ret_str:<12} {sharpe_str:<12} {dd_str:<12} {trades_str:<10}"
            )
        else:
            print(
                f"{desc:<40} {'失败':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )

    print()
    print("=" * 120)
    print("✅ 所有策略训练完成")
    print("=" * 120)


if __name__ == "__main__":
    main()
