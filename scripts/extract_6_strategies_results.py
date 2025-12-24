#!/usr/bin/env python3
"""
从训练日志中提取6个策略的结果
"""

import re
from pathlib import Path

log_file = Path("/tmp/train_all_6_strategies.log")

strategies = {
    "sr_reversal_long": "二分类（无权重，全量扫描）",
    "sr_reversal_long_sr_filter": "二分类（无权重，SR过滤）",
    "sr_reversal_long_weighted": "二分类（带权重，SR过滤）",
    "sr_reversal_rr_reg_long": "回归（无权重，全量扫描）",
    "sr_reversal_rr_reg_long_sr_filter": "回归（无权重，SR过滤）",
    "sr_reversal_rr_reg_long_weighted": "回归（带权重，SR过滤）",
}

results = {}

if log_file.exists():
    content = log_file.read_text()

    # 按策略分割
    for strategy_name, desc in strategies.items():
        # 找到策略的训练部分
        pattern = rf"训练策略: {strategy_name}(.*?)(?=训练策略:|📊 完整对比结果|$)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            section = match.group(1)

            # 提取训练样本数
            train_match = re.search(r"Train non-null: (\d+)", section)
            n_train = int(train_match.group(1)) if train_match else None

            # 提取CV指标
            cv_match = re.search(r"Average CV Metric: ([\d.]+)", section)
            cv_metric = float(cv_match.group(1)) if cv_match else None

            # 提取相关性
            corr_match = re.search(r"test_correlation: ([\d.-]+)", section)
            correlation = float(corr_match.group(1)) if corr_match else None

            results[strategy_name] = {
                "description": desc,
                "n_train_samples": n_train,
                "avg_cv_metric": cv_metric,
                "test_correlation": correlation,
            }

# 打印结果
print("=" * 120)
print("📊 6个策略完整对比（从训练日志提取）")
print("=" * 120)
print()
print(f"{'策略':<40} {'训练样本':<12} {'CV指标':<12} {'相关性':<12}")
print("-" * 120)

for strategy_name, desc in strategies.items():
    data = results.get(strategy_name, {})
    n_train = data.get("n_train_samples", "N/A")
    cv_metric = data.get("avg_cv_metric", "N/A")
    correlation = data.get("test_correlation", "N/A")

    cv_str = (
        f"{cv_metric:.4f}" if isinstance(cv_metric, (int, float)) else str(cv_metric)
    )
    corr_str = (
        f"{correlation:.4f}"
        if isinstance(correlation, (int, float))
        else str(correlation)
    )

    print(f"{desc:<40} {str(n_train):<12} {cv_str:<12} {corr_str:<12}")
