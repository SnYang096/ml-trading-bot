#!/usr/bin/env python3
"""
从训练日志中提取6个策略的结果
"""

import re
from pathlib import Path

log_file = Path("/tmp/run_all_6_final.log")
if not log_file.exists():
    print("日志文件不存在")
    exit(1)

content = log_file.read_text()

strategies = {
    "sr_reversal_long": "二分类（无权重，全量扫描）",
    "sr_reversal_long_sr_filter": "二分类（无权重，SR过滤）",
    "sr_reversal_long_weighted": "二分类（带权重，SR过滤）",
    "sr_reversal_rr_reg_long": "回归（无权重，全量扫描）",
    "sr_reversal_rr_reg_long_sr_filter": "回归（无权重，SR过滤）",
    "sr_reversal_rr_reg_long_weighted": "回归（带权重，SR过滤）",
}

results = {}

for strategy_name, desc in strategies.items():
    # 找到策略的训练部分
    pattern = rf"训练策略: {strategy_name}.*?(?=训练策略:|所有策略训练完成|$)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        continue

    section = match.group(0)

    # 提取训练样本数
    train_match = re.search(r"Train non-null: (\d+)", section)
    n_train = int(train_match.group(1)) if train_match else None

    # 提取CV指标
    cv_match = re.search(r"Average CV Metric: ([\d.-]+)", section)
    cv_metric = float(cv_match.group(1)) if cv_match else None

    # 提取相关性
    corr_match = re.search(
        r"test_correlation: ([\d.-]+)|pearson_correlation: ([\d.-]+)", section
    )
    correlation = (
        float(corr_match.group(1) or corr_match.group(2)) if corr_match else None
    )

    # 提取回测结果
    backtest_match = re.search(r"Backtest completed", section)
    has_backtest = backtest_match is not None

    # 尝试从结果文件读取回测数据
    backtest_data = {}
    if has_backtest:
        import json

        results_file = Path(
            f"results/strategies_comparison_6_final/{strategy_name}/results.json"
        )
        if not results_file.exists():
            results_file = Path(
                f"results/strategies_comparison_6_final/sr_reversal/results.json"
            )

        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    if data.get("strategy") == strategy_name or strategy_name in str(
                        results_file
                    ):
                        backtest_data = data.get("backtest", {})
            except:
                pass

    results[strategy_name] = {
        "description": desc,
        "n_train_samples": n_train,
        "avg_cv_metric": cv_metric,
        "test_correlation": correlation,
        "backtest": backtest_data,
    }

# 打印结果
print("=" * 120)
print("📊 6个策略完整对比（修复后）")
print("=" * 120)
print()
print(
    f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12} {'总收益%':<12} {'夏普比率':<12} {'最大回撤%':<12} {'交易次数':<10}"
)
print("-" * 120)

for strategy_name, desc in strategies.items():
    data = results.get(strategy_name, {})
    n_train = data.get("n_train_samples", "N/A")
    cv_metric = data.get("avg_cv_metric", "N/A")
    correlation = data.get("test_correlation", "N/A")

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
        f"{desc:<40} {str(n_train):<12} {cv_str:<12} {corr_str:<12} {ret_str:<12} {sharpe_str:<12} {dd_str:<12} {trades_str:<10}"
    )

print()
print("=" * 120)
print("✅ 对比完成")
print("=" * 120)
