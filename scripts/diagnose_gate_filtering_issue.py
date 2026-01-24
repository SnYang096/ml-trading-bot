#!/usr/bin/env python3
"""
诊断Gate过滤过严的原因

分析每个archetype的gate规则触发频率，识别最常触发的规则和阈值，
分析特征分布，找出哪些阈值过于严格。

使用方法:
    python scripts/diagnose_gate_filtering_issue.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --out-dir results/diagnosis/gate_filtering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_gate_reasons(reasons_str: str) -> List[str]:
    """解析gate_reasons字符串为规则列表"""
    if pd.isna(reasons_str) or not reasons_str:
        return []

    reasons_str = str(reasons_str)
    # 格式: gate_deny=['rule1', 'rule2', ...]
    if reasons_str.startswith("gate_deny="):
        reasons_str = reasons_str.replace("gate_deny=", "")
        # 移除方括号和引号
        reasons_str = reasons_str.strip("[]'\"")
        # 分割规则
        rules = [r.strip().strip("'\"") for r in reasons_str.split(",")]
        return rules
    return []


def analyze_rule_trigger_frequency(gated_df: pd.DataFrame) -> Dict[str, Any]:
    """分析规则触发频率"""
    rule_counter = Counter()
    rule_by_archetype = defaultdict(Counter)

    for _, row in gated_df.iterrows():
        if row.get("gate_ok", False):
            continue  # 只分析被阻止的交易

        archetype = row.get("gate_archetype", "UNKNOWN")
        reasons_str = row.get("gate_reasons", "")
        rules = parse_gate_reasons(reasons_str)

        for rule in rules:
            rule_counter[rule] += 1
            rule_by_archetype[archetype][rule] += 1

    return {
        "total_rules": dict(rule_counter),
        "by_archetype": {k: dict(v) for k, v in rule_by_archetype.items()},
    }


def analyze_feature_distributions(
    gated_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """分析特征分布，找出哪些阈值过于严格"""
    # 合并数据
    merged = gated_df.merge(
        raw_df,
        on=["symbol", "timestamp"],
        how="left",
        suffixes=("", "_raw"),
    )

    # 分析被阻止和允许的交易的特征分布
    vetoed = merged[merged["gate_ok"] == False]
    allowed = merged[merged["gate_ok"] == True]

    # 关键特征列表（从gate规则中提取）
    key_features = [
        "jump_risk_pct",
        "atr_percentile",
        "path_efficiency_pct",
        "path_length_pct",
        "cvd_change_5_pct",
        "volume_ratio_pct",
        "bb_width_normalized_pct",
        "adx",
        "tc_score",
        "te_score",
        "fr_score",
        "et_score",
    ]

    feature_stats = {}
    for feature in key_features:
        if feature not in merged.columns:
            continue

        vetoed_vals = vetoed[feature].dropna()
        allowed_vals = allowed[feature].dropna()

        if len(vetoed_vals) == 0 and len(allowed_vals) == 0:
            continue

        feature_stats[feature] = {
            "vetoed": {
                "count": len(vetoed_vals),
                "mean": float(vetoed_vals.mean()) if len(vetoed_vals) > 0 else None,
                "median": float(vetoed_vals.median()) if len(vetoed_vals) > 0 else None,
                "p25": (
                    float(vetoed_vals.quantile(0.25)) if len(vetoed_vals) > 0 else None
                ),
                "p75": (
                    float(vetoed_vals.quantile(0.75)) if len(vetoed_vals) > 0 else None
                ),
                "min": float(vetoed_vals.min()) if len(vetoed_vals) > 0 else None,
                "max": float(vetoed_vals.max()) if len(vetoed_vals) > 0 else None,
            },
            "allowed": {
                "count": len(allowed_vals),
                "mean": float(allowed_vals.mean()) if len(allowed_vals) > 0 else None,
                "median": (
                    float(allowed_vals.median()) if len(allowed_vals) > 0 else None
                ),
                "p25": (
                    float(allowed_vals.quantile(0.25))
                    if len(allowed_vals) > 0
                    else None
                ),
                "p75": (
                    float(allowed_vals.quantile(0.75))
                    if len(allowed_vals) > 0
                    else None
                ),
                "min": float(allowed_vals.min()) if len(allowed_vals) > 0 else None,
                "max": float(allowed_vals.max()) if len(allowed_vals) > 0 else None,
            },
        }

    return feature_stats


def analyze_archetype_distribution(gated_df: pd.DataFrame) -> Dict[str, Any]:
    """分析archetype分布"""
    total = len(gated_df)
    vetoed = gated_df[gated_df["gate_ok"] == False]
    allowed = gated_df[gated_df["gate_ok"] == True]

    return {
        "total": total,
        "vetoed": len(vetoed),
        "allowed": len(allowed),
        "veto_rate": len(vetoed) / total if total > 0 else 0,
        "archetype_distribution": {
            "all": gated_df["gate_archetype"].value_counts().to_dict(),
            "vetoed": vetoed["gate_archetype"].value_counts().to_dict(),
            "allowed": allowed["gate_archetype"].value_counts().to_dict(),
        },
    }


def generate_diagnosis_report(
    rule_stats: Dict[str, Any],
    feature_stats: Dict[str, Any],
    archetype_stats: Dict[str, Any],
    out_dir: Path,
) -> None:
    """生成诊断报告"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成Markdown报告
    report_path = out_dir / "gate_filtering_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Gate过滤过严诊断报告\n\n")

        f.write("## 1. 总体统计\n\n")
        f.write(f"- 总决策数: {archetype_stats['total']}\n")
        f.write(
            f"- 被阻止: {archetype_stats['vetoed']} ({archetype_stats['veto_rate']:.2%})\n"
        )
        f.write(f"- 允许: {archetype_stats['allowed']}\n\n")

        f.write("## 2. Archetype分布\n\n")
        f.write("### 所有决策\n")
        for arch, count in archetype_stats["archetype_distribution"]["all"].items():
            f.write(f"- {arch}: {count}\n")
        f.write("\n### 被阻止的决策\n")
        for arch, count in archetype_stats["archetype_distribution"]["vetoed"].items():
            f.write(f"- {arch}: {count}\n")
        f.write("\n### 允许的决策\n")
        for arch, count in archetype_stats["archetype_distribution"]["allowed"].items():
            f.write(f"- {arch}: {count}\n")
        f.write("\n")

        f.write("## 3. 规则触发频率\n\n")
        f.write("### 总体触发频率（前20个）\n")
        sorted_rules = sorted(
            rule_stats["total_rules"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        for rule, count in sorted_rules:
            f.write(f"- {rule}: {count}次\n")
        f.write("\n")

        f.write("### 按Archetype分类\n")
        for arch, rules in rule_stats["by_archetype"].items():
            f.write(f"\n#### {arch}\n")
            sorted_arch_rules = sorted(rules.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            for rule, count in sorted_arch_rules:
                f.write(f"- {rule}: {count}次\n")
        f.write("\n")

        f.write("## 4. 特征分布分析\n\n")
        f.write("比较被阻止和允许交易的特征分布，找出阈值可能过于严格的特征。\n\n")
        for feature, stats in feature_stats.items():
            f.write(f"### {feature}\n")
            vetoed_stats = stats["vetoed"]
            allowed_stats = stats["allowed"]

            if vetoed_stats["count"] > 0 and allowed_stats["count"] > 0:
                f.write(
                    f"- 被阻止: 均值={vetoed_stats['mean']:.4f}, 中位数={vetoed_stats['median']:.4f}, "
                    f"P25={vetoed_stats['p25']:.4f}, P75={vetoed_stats['p75']:.4f}\n"
                )
                f.write(
                    f"- 允许: 均值={allowed_stats['mean']:.4f}, 中位数={allowed_stats['median']:.4f}, "
                    f"P25={allowed_stats['p25']:.4f}, P75={allowed_stats['p75']:.4f}\n"
                )

                # 检查是否有明显差异
                if (
                    vetoed_stats["mean"] is not None
                    and allowed_stats["mean"] is not None
                ):
                    diff = abs(vetoed_stats["mean"] - allowed_stats["mean"])
                    if diff > 0.1:
                        f.write(
                            f"  ⚠️ **注意**: 被阻止和允许的特征值差异较大 ({diff:.4f})，可能需要调整阈值\n"
                        )
            f.write("\n")

    # 生成JSON报告
    json_path = out_dir / "gate_filtering_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rule_stats": rule_stats,
                "feature_stats": feature_stats,
                "archetype_stats": archetype_stats,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"✅ 诊断报告已生成:")
    print(f"   - Markdown: {report_path}")
    print(f"   - JSON: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="诊断Gate过滤过严的原因",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gated-logs",
        required=True,
        help="Gated logs文件（parquet）",
    )
    parser.add_argument(
        "--raw-logs",
        default=None,
        help="原始logs文件（parquet，可选）",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录",
    )

    args = parser.parse_args()

    # 读取数据
    gated_df = pd.read_parquet(args.gated_logs)
    print(f"✅ 读取gated数据: {len(gated_df)} 行")

    raw_df = None
    if args.raw_logs:
        raw_df = pd.read_parquet(args.raw_logs)
        print(f"✅ 读取原始数据: {len(raw_df)} 行")
    else:
        raw_df = gated_df.copy()
        print("⚠️  未提供原始数据，使用gated数据作为基础")

    # 分析
    print("\n📊 分析规则触发频率...")
    rule_stats = analyze_rule_trigger_frequency(gated_df)

    print("📊 分析特征分布...")
    feature_stats = analyze_feature_distributions(gated_df, raw_df)

    print("📊 分析Archetype分布...")
    archetype_stats = analyze_archetype_distribution(gated_df)

    # 生成报告
    print("\n📝 生成诊断报告...")
    generate_diagnosis_report(
        rule_stats, feature_stats, archetype_stats, Path(args.out_dir)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
