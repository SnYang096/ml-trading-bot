#!/usr/bin/env python3
"""
Failure-first Tree 规则筛选器

🟥 核心功能：
从树模型导出的规则中筛选出"有意义的失败规则"。

⚠️ 筛选标准（不是每个条件只出现1次就有价值）：
1. 出现频率：在多棵树/多子样本中重复出现
2. 覆盖度：覆盖足够多的样本
3. Lift：failure_rate 相对于 baseline 的提升
4. 时间稳定性：在不同时间段都成立

📊 输出格式：
- 筛选后的高质量失败规则
- 每条规则的 failure_rate、lift、覆盖样本数
- 可直接映射到 execution_archetypes.yaml 的 when_then_rules

使用方式：
```bash
python scripts/filter_failure_rules.py \
    --model-path models/bpc/model.pkl \
    --data-path data/parquet_data/BTCUSDT_240T.parquet \
    --output results/bpc_failure_rules.yaml
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml


@dataclass
class FailureRule:
    """
    失败规则数据结构。

    对应 execution_archetypes.yaml 中的 when_then_rules 格式。
    """

    feature: str
    operator: str  # "<=", ">", "between"
    threshold: float
    threshold_high: Optional[float] = None  # 用于 between

    # 统计信息
    occurrence_count: int = 0  # 在多少棵树中出现
    sample_coverage: float = 0.0  # 覆盖样本比例
    failure_rate: float = 0.0  # 该条件下的失败率
    baseline_failure_rate: float = 0.0  # 全局失败率
    lift: float = 0.0  # failure_rate / baseline

    # 分类信息
    feature_family: str = ""  # 特征族（vpin/cvd/volume/etc）
    quality_tier: str = ""  # "high"/"medium"/"low"

    def to_when_then(self) -> dict:
        """转换为 execution_archetypes.yaml 的 when_then 格式。"""
        if self.operator == "<=":
            when_clause = {self.feature: {"value_lte": self.threshold}}
        elif self.operator == ">":
            when_clause = {self.feature: {"value_gt": self.threshold}}
        elif self.operator == "between":
            when_clause = {
                self.feature: {
                    "value_gte": self.threshold,
                    "value_lte": self.threshold_high,
                }
            }
        else:
            when_clause = {self.feature: {"value_lte": self.threshold}}

        return {
            "id": f"failure_{self.feature}_{self.operator}_{self.threshold:.4f}".replace(
                ".", "_"
            ),
            "phase": "exclusions",
            "priority": 2,
            "reason": f"Failure-first: {self.feature} {self.operator} {self.threshold:.4f} (lift={self.lift:.2f})",
            "when": when_clause,
            "then": {"action": "deny"},
            "_stats": {
                "occurrence_count": self.occurrence_count,
                "sample_coverage": self.sample_coverage,
                "failure_rate": self.failure_rate,
                "lift": self.lift,
                "quality_tier": self.quality_tier,
            },
        }


def parse_lightgbm_rules(model) -> List[Tuple[str, str, float]]:
    """
    从 LightGBM 模型中解析分裂条件。

    Returns:
        List of (feature, operator, threshold) tuples
    """
    rules = []

    try:
        # 获取模型的树结构
        booster = model.booster_ if hasattr(model, "booster_") else model
        model_dump = booster.dump_model()

        for tree_info in model_dump.get("tree_info", []):
            tree_structure = tree_info.get("tree_structure", {})
            _extract_splits_recursive(
                tree_structure, model_dump.get("feature_names", []), rules
            )
    except Exception as e:
        print(f"   ⚠️  解析 LightGBM 规则失败: {e}")

    return rules


def _extract_splits_recursive(node: dict, feature_names: list, rules: list):
    """递归提取树节点的分裂条件。"""
    if "split_feature" not in node:
        return  # 叶节点

    feature_idx = node.get("split_feature")
    threshold = node.get("threshold")

    if feature_idx is not None and threshold is not None:
        if isinstance(feature_idx, int) and feature_idx < len(feature_names):
            feature_name = feature_names[feature_idx]
        else:
            feature_name = f"feature_{feature_idx}"

        rules.append((feature_name, "<=", float(threshold)))

    # 递归处理子节点
    if "left_child" in node:
        _extract_splits_recursive(node["left_child"], feature_names, rules)
    if "right_child" in node:
        _extract_splits_recursive(node["right_child"], feature_names, rules)


def count_rule_occurrences(
    rules: List[Tuple[str, str, float]], precision: int = 3
) -> Dict[Tuple[str, str, float], int]:
    """
    统计每条规则的出现次数。

    ⚠️ 关键：将阈值四舍五入到指定精度，合并相似规则。
    """
    counter = defaultdict(int)

    for feature, op, threshold in rules:
        # 四舍五入阈值，合并相似规则
        rounded_threshold = round(threshold, precision)
        key = (feature, op, rounded_threshold)
        counter[key] += 1

    return dict(counter)


def filter_rules_by_frequency(
    rule_counts: Dict[Tuple[str, str, float], int],
    min_occurrences: int = 3,
) -> List[Tuple[str, str, float, int]]:
    """
    按出现频率筛选规则。

    Args:
        rule_counts: 规则计数字典
        min_occurrences: 最小出现次数（默认 3，即至少在 3 棵树中出现）

    Returns:
        List of (feature, op, threshold, count) 按出现次数降序排列
    """
    filtered = [
        (feature, op, threshold, count)
        for (feature, op, threshold), count in rule_counts.items()
        if count >= min_occurrences
    ]

    # 按出现次数降序排列
    filtered.sort(key=lambda x: x[3], reverse=True)

    return filtered


def compute_rule_failure_stats(
    df: pd.DataFrame,
    feature: str,
    operator: str,
    threshold: float,
    failure_col: str = "failure_label",
) -> Dict[str, float]:
    """
    计算单条规则的失败统计。

    Returns:
        {
            "sample_coverage": 规则覆盖的样本比例,
            "failure_rate": 规则区域的失败率,
            "baseline_failure_rate": 全局失败率,
            "lift": failure_rate / baseline,
            "n_samples": 规则覆盖的样本数,
            "n_failures": 规则区域的失败数,
        }
    """
    if feature not in df.columns or failure_col not in df.columns:
        return {"sample_coverage": 0, "failure_rate": 0, "lift": 0}

    valid_mask = df[failure_col].notna() & df[feature].notna()
    df_valid = df[valid_mask]

    if len(df_valid) == 0:
        return {"sample_coverage": 0, "failure_rate": 0, "lift": 0}

    # 全局失败率
    baseline = (df_valid[failure_col] == 1).mean()

    # 规则区域
    if operator == "<=":
        rule_mask = df_valid[feature] <= threshold
    elif operator == ">":
        rule_mask = df_valid[feature] > threshold
    else:
        rule_mask = df_valid[feature] <= threshold

    n_samples = rule_mask.sum()
    if n_samples == 0:
        return {
            "sample_coverage": 0,
            "failure_rate": 0,
            "baseline_failure_rate": float(baseline),
            "lift": 0,
            "n_samples": 0,
            "n_failures": 0,
        }

    n_failures = (df_valid.loc[rule_mask, failure_col] == 1).sum()
    failure_rate = n_failures / n_samples
    lift = failure_rate / baseline if baseline > 0 else 0

    return {
        "sample_coverage": float(n_samples / len(df_valid)),
        "failure_rate": float(failure_rate),
        "baseline_failure_rate": float(baseline),
        "lift": float(lift),
        "n_samples": int(n_samples),
        "n_failures": int(n_failures),
    }


def classify_feature_family(feature: str) -> str:
    """根据特征名称判断特征族。"""
    feature_lower = feature.lower()

    if "vpin" in feature_lower:
        return "vpin"
    elif "cvd" in feature_lower:
        return "cvd"
    elif "volume" in feature_lower or "vol_" in feature_lower:
        return "volume"
    elif "hilbert" in feature_lower:
        return "hilbert"
    elif "bpc" in feature_lower:
        return "bpc"
    elif "rsi" in feature_lower:
        return "momentum"
    elif "atr" in feature_lower:
        return "volatility"
    elif "sr_" in feature_lower:
        return "support_resistance"
    elif "dir" in feature_lower or "consistency" in feature_lower:
        return "direction"
    elif "ofci" in feature_lower or "shd" in feature_lower:
        return "orderflow"
    else:
        return "other"


def classify_rule_quality(
    occurrence_count: int,
    lift: float,
    sample_coverage: float,
    min_high_occurrences: int = 10,
    min_medium_occurrences: int = 5,
    min_high_lift: float = 1.5,
    min_medium_lift: float = 1.2,
    min_coverage: float = 0.01,
) -> str:
    """
    判断规则质量等级。

    高质量规则：
    - 出现次数 >= 10
    - lift >= 1.5
    - 覆盖率 >= 1%

    中等质量规则：
    - 出现次数 >= 5
    - lift >= 1.2
    - 覆盖率 >= 1%
    """
    if sample_coverage < min_coverage:
        return "low"

    if occurrence_count >= min_high_occurrences and lift >= min_high_lift:
        return "high"
    elif occurrence_count >= min_medium_occurrences and lift >= min_medium_lift:
        return "medium"
    else:
        return "low"


def filter_failure_rules(
    model,
    df: pd.DataFrame,
    failure_col: str = "failure_label",
    min_occurrences: int = 3,
    min_lift: float = 1.1,
    min_coverage: float = 0.01,
    max_rules: int = 30,
) -> List[FailureRule]:
    """
    从模型中筛选高质量的失败规则。

    筛选标准：
    1. 出现次数 >= min_occurrences
    2. lift >= min_lift
    3. 覆盖率 >= min_coverage

    Args:
        model: 训练好的 LightGBM 模型
        df: 带有 failure_label 的数据
        failure_col: 失败标签列名
        min_occurrences: 最小出现次数
        min_lift: 最小 lift
        min_coverage: 最小覆盖率
        max_rules: 最多返回的规则数

    Returns:
        List[FailureRule]: 筛选后的高质量失败规则
    """
    # 1. 解析所有分裂条件
    print("   📊 解析树分裂条件...")
    all_rules = parse_lightgbm_rules(model)
    print(f"      总分裂条件数: {len(all_rules)}")

    # 2. 统计出现次数
    rule_counts = count_rule_occurrences(all_rules)
    print(f"      去重后条件数: {len(rule_counts)}")

    # 3. 按频率筛选
    frequent_rules = filter_rules_by_frequency(rule_counts, min_occurrences)
    print(f"      出现 >= {min_occurrences} 次的条件: {len(frequent_rules)}")

    # 4. 计算每条规则的失败统计并筛选
    filtered_rules = []

    for feature, op, threshold, count in frequent_rules:
        stats = compute_rule_failure_stats(df, feature, op, threshold, failure_col)

        # 筛选条件
        if stats["lift"] < min_lift:
            continue
        if stats["sample_coverage"] < min_coverage:
            continue

        # 创建 FailureRule
        rule = FailureRule(
            feature=feature,
            operator=op,
            threshold=threshold,
            occurrence_count=count,
            sample_coverage=stats["sample_coverage"],
            failure_rate=stats["failure_rate"],
            baseline_failure_rate=stats["baseline_failure_rate"],
            lift=stats["lift"],
            feature_family=classify_feature_family(feature),
            quality_tier=classify_rule_quality(
                count, stats["lift"], stats["sample_coverage"]
            ),
        )
        filtered_rules.append(rule)

    # 5. 按 lift * occurrence_count 排序（综合考虑质量和稳定性）
    filtered_rules.sort(key=lambda r: r.lift * r.occurrence_count, reverse=True)

    # 6. 限制返回数量
    filtered_rules = filtered_rules[:max_rules]

    print(f"      筛选后的高质量规则: {len(filtered_rules)}")

    return filtered_rules


def export_to_yaml(
    rules: List[FailureRule],
    output_path: Path,
    archetype_name: str = "BreakoutPullbackContinuation",
) -> None:
    """
    将筛选后的规则导出为 execution_archetypes.yaml 格式。
    """
    when_then_rules = [rule.to_when_then() for rule in rules]

    # 按质量分组
    high_quality = [r for r in when_then_rules if r["_stats"]["quality_tier"] == "high"]
    medium_quality = [
        r for r in when_then_rules if r["_stats"]["quality_tier"] == "medium"
    ]
    low_quality = [r for r in when_then_rules if r["_stats"]["quality_tier"] == "low"]

    output = {
        "version": 1,
        "name": f"{archetype_name}_failure_rules",
        "description": "Failure-first Tree 自动导出的失败规则",
        "baseline_failure_rate": rules[0].baseline_failure_rate if rules else 0,
        "total_rules": len(rules),
        "high_quality_rules": len(high_quality),
        "medium_quality_rules": len(medium_quality),
        "low_quality_rules": len(low_quality),
        "rules_by_family": {},
        "when_then_rules": when_then_rules,
    }

    # 按特征族分组统计
    family_counts = defaultdict(int)
    for rule in rules:
        family_counts[rule.feature_family] += 1
    output["rules_by_family"] = dict(family_counts)

    # 写入 YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            output, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    print(f"   💾 规则已导出到: {output_path}")


def generate_failure_report(
    rules: List[FailureRule],
    output_path: Optional[Path] = None,
) -> str:
    """
    生成失败规则的 Markdown 报告。
    """
    lines = [
        "# Failure-first Tree 规则筛选报告\n",
        f"## 摘要\n",
        f"- 总规则数: {len(rules)}\n",
    ]

    if rules:
        baseline = rules[0].baseline_failure_rate
        lines.append(f"- Baseline 失败率: {baseline:.2%}\n")

        # 按质量分组
        high_count = sum(1 for r in rules if r.quality_tier == "high")
        medium_count = sum(1 for r in rules if r.quality_tier == "medium")
        low_count = sum(1 for r in rules if r.quality_tier == "low")

        lines.append(f"- 高质量规则: {high_count}\n")
        lines.append(f"- 中等质量规则: {medium_count}\n")
        lines.append(f"- 低质量规则: {low_count}\n")

        # 按特征族分组
        lines.append("\n## 按特征族分组\n")
        family_counts = defaultdict(list)
        for rule in rules:
            family_counts[rule.feature_family].append(rule)

        for family, family_rules in sorted(
            family_counts.items(), key=lambda x: -len(x[1])
        ):
            avg_lift = np.mean([r.lift for r in family_rules])
            lines.append(
                f"- **{family}**: {len(family_rules)} 条规则, 平均 lift = {avg_lift:.2f}\n"
            )

        # 详细规则表
        lines.append("\n## 规则详情\n")
        lines.append("| 特征 | 条件 | 出现次数 | 覆盖率 | 失败率 | Lift | 质量 |\n")
        lines.append("|------|------|----------|--------|--------|------|------|\n")

        for rule in rules[:30]:  # 只显示前30条
            lines.append(
                f"| `{rule.feature}` | `{rule.operator} {rule.threshold:.4f}` | "
                f"{rule.occurrence_count} | {rule.sample_coverage:.1%} | "
                f"{rule.failure_rate:.1%} | {rule.lift:.2f} | {rule.quality_tier} |\n"
            )

    report = "".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"   📄 报告已保存到: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Failure-first Tree 规则筛选器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 基础用法
    python scripts/filter_failure_rules.py \\
        --model-path models/bpc/model.pkl \\
        --data-path data/parquet_data/BTCUSDT_240T.parquet \\
        --failure-col failure_label \\
        --output results/bpc_failure_rules.yaml
    
    # 调整筛选参数
    python scripts/filter_failure_rules.py \\
        --model-path models/bpc/model.pkl \\
        --data-path data/parquet_data/BTCUSDT_240T.parquet \\
        --min-occurrences 5 \\
        --min-lift 1.3 \\
        --max-rules 20
        """,
    )

    parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data-path", type=str, required=True, help="数据文件路径")
    parser.add_argument(
        "--failure-col", type=str, default="failure_label", help="失败标签列名"
    )
    parser.add_argument(
        "--output", type=str, default="results/failure_rules.yaml", help="输出路径"
    )
    parser.add_argument("--min-occurrences", type=int, default=3, help="最小出现次数")
    parser.add_argument("--min-lift", type=float, default=1.1, help="最小 lift")
    parser.add_argument("--min-coverage", type=float, default=0.01, help="最小覆盖率")
    parser.add_argument("--max-rules", type=int, default=30, help="最多返回规则数")
    parser.add_argument("--archetype", type=str, default="BPC", help="Archetype 名称")

    args = parser.parse_args()

    # 加载模型
    import joblib

    print(f"📂 加载模型: {args.model_path}")
    model = joblib.load(args.model_path)

    # 加载数据
    print(f"📂 加载数据: {args.data_path}")
    df = pd.read_parquet(args.data_path)

    if args.failure_col not in df.columns:
        print(f"   ⚠️  数据中没有 '{args.failure_col}' 列")
        print(f"      可用列: {list(df.columns)[:20]}...")
        return

    # 筛选规则
    print("\n🔍 筛选失败规则...")
    rules = filter_failure_rules(
        model=model,
        df=df,
        failure_col=args.failure_col,
        min_occurrences=args.min_occurrences,
        min_lift=args.min_lift,
        min_coverage=args.min_coverage,
        max_rules=args.max_rules,
    )

    # 导出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_yaml(rules, output_path, archetype_name=args.archetype)

    # 生成报告
    report_path = output_path.with_suffix(".md")
    generate_failure_report(rules, report_path)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
