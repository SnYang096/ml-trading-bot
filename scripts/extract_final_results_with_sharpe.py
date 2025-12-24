#!/usr/bin/env python3
"""
从训练日志和结果文件中提取6个策略的完整结果（包含夏普比率）
"""

import re
import json
from pathlib import Path
import numpy as np


def main():
    log_file = Path("/tmp/run_all_6_weighted.log")
    results_dir = Path("results/strategies_comparison_6_final_weighted")

    strategies = {
        "sr_reversal_long": "二分类（无权重，全量扫描）",
        "sr_reversal_long_sr_filter": "二分类（无权重，SR过滤）",
        "sr_reversal_long_weighted": "二分类（带权重，SR过滤）",
        "sr_reversal_rr_reg_long": "回归（无权重，全量扫描）",
        "sr_reversal_rr_reg_long_sr_filter": "回归（无权重，SR过滤）",
        "sr_reversal_rr_reg_long_weighted": "回归（带权重，SR过滤）",
    }

    log_content = log_file.read_text() if log_file.exists() else ""
    all_results_files = list(results_dir.rglob("results.json"))

    print("=" * 120)
    print("📊 6个策略完整对比（盈利样本权重优先 + 夏普比率）")
    print("=" * 120)
    print()
    print(
        f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12} {'总收益%':<12} {'夏普比率':<12} {'最大回撤%':<12} {'交易次数':<10}"
    )
    print("-" * 120)

    for strategy_name, desc in strategies.items():
        # 从日志提取
        pattern = rf"训练策略: {strategy_name}.*?(?=训练策略:|所有策略训练完成|$)"
        match = re.search(pattern, log_content, re.DOTALL)

        n_train = None
        cv_metric = None
        correlation = None

        if match:
            section = match.group(0)
            train_match = re.search(r"Train non-null: (\d+)", section)
            n_train = int(train_match.group(1)) if train_match else None
            cv_match = re.search(r"Average CV Metric: ([\d.-]+)", section)
            cv_metric = float(cv_match.group(1)) if cv_match else None
            corr_match = re.search(
                r"test_correlation: ([\d.-]+)|pearson_correlation: ([\d.-]+)", section
            )
            correlation = (
                float(corr_match.group(1) or corr_match.group(2))
                if corr_match
                else None
            )

        # 从结果文件读取回测数据
        sharpe = None
        total_return = None
        max_dd = None
        trades = None

        for results_file in all_results_files:
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    # 检查是否是当前策略的结果
                    if (
                        strategy_name in str(results_file)
                        or data.get("strategy") == strategy_name
                    ):
                        bt = data.get("backtest", {})
                        sharpe = bt.get("sharpe", None)
                        total_return = bt.get("total_return_pct", None)
                        max_dd = bt.get("max_drawdown_pct", None)
                        trades = bt.get("total_trades", None)
                        break
            except:
                continue

        n_train_str = str(n_train) if n_train else "N/A"
        cv_str = f"{cv_metric:.4f}" if cv_metric is not None else "N/A"
        corr_str = f"{correlation:.4f}" if correlation is not None else "N/A"
        ret_str = f"{total_return:.2f}%" if total_return is not None else "N/A"
        sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
        dd_str = f"{max_dd:.2f}%" if max_dd is not None else "N/A"
        trades_str = str(int(trades)) if trades is not None else "N/A"

        print(
            f"{desc:<40} {n_train_str:<12} {cv_str:<12} {corr_str:<12} {ret_str:<12} {sharpe_str:<12} {dd_str:<12} {trades_str:<10}"
        )

    print()
    print("=" * 120)
    print("✅ 对比完成")
    print("=" * 120)


if __name__ == "__main__":
    main()
