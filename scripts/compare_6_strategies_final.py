#!/usr/bin/env python3
"""
对比6个策略的完整结果（从结果文件读取）
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def main():
    results_dir = Path("results/strategies_comparison_6_final")

    strategies = {
        "sr_reversal_long": "二分类（无权重，全量扫描）",
        "sr_reversal_long_sr_filter": "二分类（无权重，SR过滤）",
        "sr_reversal_long_weighted": "二分类（带权重，SR过滤）",
        "sr_reversal_rr_reg_long": "回归（无权重，全量扫描）",
        "sr_reversal_rr_reg_long_sr_filter": "回归（无权重，SR过滤）",
        "sr_reversal_rr_reg_long_weighted": "回归（带权重，SR过滤）",
    }

    print("=" * 120)
    print("📊 6个策略完整对比（包含回测）")
    print("=" * 120)
    print()
    print(
        f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12} {'总收益%':<12} {'夏普比率':<12} {'最大回撤%':<12} {'交易次数':<10}"
    )
    print("-" * 120)

    all_results = {}

    for strategy_name, desc in strategies.items():
        # 尝试多个可能的结果文件位置
        possible_paths = [
            results_dir / strategy_name / "results.json",
            results_dir / "sr_reversal" / "results.json",  # 某些策略可能使用这个路径
        ]

        data = {}
        for results_file in possible_paths:
            if results_file.exists():
                with open(results_file, "r") as f:
                    data = json.load(f)
                # 检查策略名称是否匹配
                if data.get("strategy") == strategy_name or strategy_name in str(
                    results_file
                ):
                    all_results[strategy_name] = data
                    break

        if not data:
            print(f"{desc:<40} {'文件不存在':<12}")
            continue

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

    # 保存详细结果
    output_file = results_dir / "all_6_strategies_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print(f"💾 详细结果已保存到: {output_file}")
    print()
    print("=" * 120)
    print("✅ 对比完成")
    print("=" * 120)


if __name__ == "__main__":
    main()
