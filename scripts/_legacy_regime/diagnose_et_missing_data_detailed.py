#!/usr/bin/env python3
"""
ET数据缺失深度诊断

全面诊断ET（ExhaustionTurn）没有数据的原因，包括：
1. MEAN_REGIME中ET的候选数量
2. ET被gate rules拒绝的详细原因（哪些规则拒绝了多少样本）
3. ET被evidence rules拒绝的详细原因（哪些evidence未通过）
4. ET在gate决策中的优先级问题
5. 提供具体的优化建议
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.feature_store import FeatureStore, FeatureStoreSpec


def check_required_evidence(
    evidence_flags: Dict[str, bool],
    required_evidence: List[str],
) -> bool:
    """检查是否满足所有required evidence"""
    return all(evidence_flags.get(ev, False) for ev in required_evidence)


def analyze_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """分析基础统计"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    stats = {
        "total_mean_regime": len(mean_regime),
        "et_candidates": 0,
        "fr_candidates": 0,
        "both_candidates": 0,
        "et_passed_gate": 0,
        "et_failed_gate": 0,
        "fr_passed_gate": 0,
        "fr_failed_gate": 0,
    }

    if "gate_archetype" in mean_regime.columns:
        et_mask = mean_regime["gate_archetype"].str.contains("ET", case=False, na=False)
        fr_mask = mean_regime["gate_archetype"].str.contains("FR", case=False, na=False)

        stats["et_candidates"] = et_mask.sum()
        stats["fr_candidates"] = fr_mask.sum()
        stats["both_candidates"] = (et_mask & fr_mask).sum()

        # Gate决策统计
        if "gate_ok" in mean_regime.columns:
            stats["et_passed_gate"] = (et_mask & mean_regime["gate_ok"]).sum()
            stats["et_failed_gate"] = (et_mask & ~mean_regime["gate_ok"]).sum()
            stats["fr_passed_gate"] = (fr_mask & mean_regime["gate_ok"]).sum()
            stats["fr_failed_gate"] = (fr_mask & ~mean_regime["gate_ok"]).sum()

    return stats


def analyze_gate_rules_detailed(
    df: pd.DataFrame,
    et_arch: Any,
    quantiles: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """详细分析ET的gate rules"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    if not et_arch or not hasattr(et_arch, "gate_rules"):
        return {"error": "ET archetype or gate_rules not found"}

    gate_rules = et_arch.gate_rules
    rules_list = gate_rules.get("rules", [])
    deny_if = gate_rules.get("deny_if", [])
    allow_if = gate_rules.get("allow_if", [])
    allow_mode = gate_rules.get("allow_mode", "any")
    default_action = gate_rules.get("default_action", "deny")

    # 统计每个规则的触发情况
    deny_rule_stats = {rule_name: 0 for rule_name in deny_if}
    allow_rule_stats = {rule_name: 0 for rule_name in allow_if}

    # 详细分析每个样本
    sample_details = []
    passed_count = 0
    failed_count = 0
    default_action_rejects = 0

    for idx, row in mean_regime.iterrows():
        features = row.to_dict()
        symbol = features.get("symbol", "ALL")
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        # 计算所有规则的flags
        try:
            flags = compute_execution_evidence(
                features=features,
                rules=rules_list,
                quantiles=symbol_quantiles,
            )
        except Exception as e:
            sample_details.append(
                {
                    "index": idx,
                    "symbol": symbol,
                    "timestamp": features.get("timestamp"),
                    "error": str(e),
                }
            )
            failed_count += 1
            continue

        # 检查deny_if
        deny_hits = [name for name in deny_if if flags.get(name, False)]
        if deny_hits:
            for rule_name in deny_hits:
                deny_rule_stats[rule_name] += 1
            sample_details.append(
                {
                    "index": idx,
                    "symbol": symbol,
                    "timestamp": features.get("timestamp"),
                    "gate_result": "deny",
                    "deny_hits": deny_hits,
                    "allow_hits": [],
                }
            )
            failed_count += 1
            continue

        # 检查allow_if
        allow_hits = [name for name in allow_if if flags.get(name, False)]
        for rule_name in allow_hits:
            allow_rule_stats[rule_name] += 1

        # 根据allow_mode判断
        if allow_mode == "all":
            ok = all(flags.get(name, False) for name in allow_if)
        elif allow_mode.startswith("min") or allow_mode.startswith("at_least_"):
            min_hits = 1
            if allow_mode.startswith("min"):
                raw = allow_mode.replace("min", "").replace(":", "").strip()
            else:
                raw = allow_mode.replace("at_least_", "").strip()
            try:
                min_hits = int(raw) if raw else 1
            except Exception:
                min_hits = 1
            hit_count = len(allow_hits)
            ok = hit_count >= max(1, min_hits)
        else:  # any
            ok = len(allow_hits) > 0

        if ok:
            sample_details.append(
                {
                    "index": idx,
                    "symbol": symbol,
                    "timestamp": features.get("timestamp"),
                    "gate_result": "pass",
                    "deny_hits": [],
                    "allow_hits": allow_hits,
                }
            )
            passed_count += 1
        else:
            if not allow_if:
                # 没有allow_if，使用default_action
                if default_action == "deny":
                    default_action_rejects += 1
            sample_details.append(
                {
                    "index": idx,
                    "symbol": symbol,
                    "timestamp": features.get("timestamp"),
                    "gate_result": "deny",
                    "deny_hits": [],
                    "allow_hits": allow_hits,
                    "reason": (
                        "allow_if_not_met"
                        if allow_if
                        else f"default_action_{default_action}"
                    ),
                }
            )
            failed_count += 1

    return {
        "total_samples": len(mean_regime),
        "passed_gate": passed_count,
        "failed_gate": failed_count,
        "default_action_rejects": default_action_rejects,
        "deny_rule_stats": deny_rule_stats,
        "allow_rule_stats": allow_rule_stats,
        "sample_details": sample_details[:100],  # 只保存前100个样本的详细信息
    }


def analyze_evidence_rules_detailed(
    df: pd.DataFrame,
    et_arch: Any,
    quantiles: Dict[str, Any] | None = None,
    gate_passed_mask: pd.Series | None = None,
) -> Dict[str, Any]:
    """详细分析ET的evidence rules"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    if gate_passed_mask is not None:
        mean_regime = mean_regime[gate_passed_mask].copy()

    if not et_arch:
        return {"error": "ET archetype not found"}

    evidence_rules = getattr(et_arch, "evidence_rules", [])
    required_evidence = getattr(et_arch, "required_evidence", [])

    # 统计每个evidence的通过情况
    evidence_stats = {ev: {"passed": 0, "failed": 0} for ev in required_evidence}
    all_evidence_stats = {}

    passed_count = 0
    failed_count = 0
    sample_details = []

    for idx, row in mean_regime.iterrows():
        features = row.to_dict()
        symbol = features.get("symbol", "ALL")
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        try:
            evidence_flags = compute_execution_evidence(
                features=features,
                rules=evidence_rules,
                quantiles=symbol_quantiles,
            )
        except Exception as e:
            sample_details.append(
                {
                    "index": idx,
                    "symbol": symbol,
                    "timestamp": features.get("timestamp"),
                    "error": str(e),
                }
            )
            failed_count += 1
            continue

        # 统计所有evidence
        for ev_name, ev_passed in evidence_flags.items():
            if ev_name not in all_evidence_stats:
                all_evidence_stats[ev_name] = {"passed": 0, "failed": 0}
            if ev_passed:
                all_evidence_stats[ev_name]["passed"] += 1
            else:
                all_evidence_stats[ev_name]["failed"] += 1

        # 检查required_evidence
        passed_required = []
        failed_required = []
        for ev in required_evidence:
            if evidence_flags.get(ev, False):
                passed_required.append(ev)
                evidence_stats[ev]["passed"] += 1
            else:
                failed_required.append(ev)
                evidence_stats[ev]["failed"] += 1

        all_passed = len(failed_required) == 0

        if all_passed:
            passed_count += 1
        else:
            failed_count += 1

        sample_details.append(
            {
                "index": idx,
                "symbol": symbol,
                "timestamp": features.get("timestamp"),
                "evidence_result": "pass" if all_passed else "fail",
                "passed_required": passed_required,
                "failed_required": failed_required,
                "all_evidence_flags": evidence_flags,
            }
        )

    return {
        "total_samples": len(mean_regime),
        "passed_evidence": passed_count,
        "failed_evidence": failed_count,
        "required_evidence_stats": evidence_stats,
        "all_evidence_stats": all_evidence_stats,
        "sample_details": sample_details[:50],  # 只保存前50个样本的详细信息
    }


def analyze_priority_competition(
    df: pd.DataFrame,
    et_arch: Any,
    fr_arch: Any,
    quantiles: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """分析FR vs ET的竞争情况"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    both_passed = 0
    only_et_passed = 0
    only_fr_passed = 0
    neither_passed = 0

    et_scores = []
    fr_scores = []

    for idx, row in mean_regime.iterrows():
        features = row.to_dict()
        symbol = features.get("symbol", "ALL")
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        # 检查ET
        et_passed = False
        if et_arch and hasattr(et_arch, "gate_rules"):
            try:
                et_ok, _ = apply_gate_rules(
                    gate_rules=et_arch.gate_rules,
                    features=features,
                    quantiles=symbol_quantiles,
                )
                et_passed = et_ok
            except Exception:
                et_passed = False

        # 检查FR
        fr_passed = False
        if fr_arch and hasattr(fr_arch, "gate_rules"):
            try:
                fr_ok, _ = apply_gate_rules(
                    gate_rules=fr_arch.gate_rules,
                    features=features,
                    quantiles=symbol_quantiles,
                )
                fr_passed = fr_ok
            except Exception:
                fr_passed = False

        # 统计
        if et_passed and fr_passed:
            both_passed += 1
        elif et_passed:
            only_et_passed += 1
        elif fr_passed:
            only_fr_passed += 1
        else:
            neither_passed += 1

        # 计算score（如果有相关列）
        if "et_semantic_score" in features or "mean_score" in features:
            et_score = features.get("et_semantic_score") or features.get(
                "mean_score", 0
            )
            fr_score = features.get("fr_semantic_score") or features.get(
                "mean_score", 0
            )
            if et_passed:
                et_scores.append(float(et_score) if pd.notna(et_score) else 0)
            if fr_passed:
                fr_scores.append(float(fr_score) if pd.notna(fr_score) else 0)

    return {
        "both_passed": both_passed,
        "only_et_passed": only_et_passed,
        "only_fr_passed": only_fr_passed,
        "neither_passed": neither_passed,
        "et_scores": {
            "mean": float(np.mean(et_scores)) if et_scores else 0,
            "median": float(np.median(et_scores)) if et_scores else 0,
            "std": float(np.std(et_scores)) if et_scores else 0,
            "count": len(et_scores),
        },
        "fr_scores": {
            "mean": float(np.mean(fr_scores)) if fr_scores else 0,
            "median": float(np.median(fr_scores)) if fr_scores else 0,
            "std": float(np.std(fr_scores)) if fr_scores else 0,
            "count": len(fr_scores),
        },
    }


def analyze_feature_availability(
    df: pd.DataFrame,
    et_arch: Any,
) -> Dict[str, Any]:
    """分析ET需要的特征可用性"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    # 收集ET需要的所有特征
    gate_rules = getattr(et_arch, "gate_rules", {}) or {}
    evidence_rules = getattr(et_arch, "evidence_rules", [])

    gate_features = set()
    for rule in gate_rules.get("rules", []):
        if "key" in rule:
            gate_features.add(rule["key"])

    evidence_features = set()
    for rule in evidence_rules:
        if "key" in rule:
            evidence_features.add(rule["key"])
        if "any_key_contains" in rule:
            # 这些是模式匹配，需要检查实际特征
            patterns = rule.get("any_key_contains", [])
            for pattern in patterns:
                matching_cols = [
                    col for col in df.columns if pattern.lower() in col.lower()
                ]
                evidence_features.update(matching_cols)

    all_required_features = gate_features | evidence_features

    # 统计特征缺失
    feature_stats = {}
    for feat in all_required_features:
        if feat in mean_regime.columns:
            missing_count = mean_regime[feat].isna().sum()
            feature_stats[feat] = {
                "available": len(mean_regime) - missing_count,
                "missing": missing_count,
                "missing_rate": (
                    missing_count / len(mean_regime) if len(mean_regime) > 0 else 0
                ),
            }
        else:
            feature_stats[feat] = {
                "available": 0,
                "missing": len(mean_regime),
                "missing_rate": 1.0,
            }

    return {
        "gate_features": sorted(gate_features),
        "evidence_features": sorted(evidence_features),
        "all_required_features": sorted(all_required_features),
        "feature_stats": feature_stats,
    }


def generate_report(
    basic_stats: Dict[str, Any],
    gate_analysis: Dict[str, Any],
    evidence_analysis: Dict[str, Any],
    priority_analysis: Dict[str, Any],
    feature_analysis: Dict[str, Any],
    output_path: Path,
) -> None:
    """生成Markdown格式的详细报告"""
    lines = []
    lines.append("# ET数据缺失深度诊断报告\n")
    lines.append(f"**生成时间**: {pd.Timestamp.now()}\n")
    lines.append("\n---\n")

    # 执行摘要
    lines.append("## 执行摘要\n")
    lines.append(f"- MEAN_REGIME总样本数: **{basic_stats['total_mean_regime']}**")
    lines.append(f"- ET候选数: **{basic_stats['et_candidates']}**")
    lines.append(f"- FR候选数: **{basic_stats['fr_candidates']}**")
    lines.append(f"- 两者都候选: **{basic_stats['both_candidates']}**")

    if "error" not in gate_analysis:
        lines.append(f"- ET通过gate rules: **{gate_analysis['passed_gate']}**")
        lines.append(f"- ET被gate rules拒绝: **{gate_analysis['failed_gate']}**")

    if "error" not in evidence_analysis:
        lines.append(
            f"- ET通过evidence rules: **{evidence_analysis['passed_evidence']}**"
        )
        lines.append(
            f"- ET被evidence rules拒绝: **{evidence_analysis['failed_evidence']}**"
        )

    lines.append("\n---\n")

    # Gate Rules分析
    lines.append("## Gate Rules详细分析\n")
    if "error" in gate_analysis:
        lines.append(f"⚠️ 错误: {gate_analysis['error']}\n")
    else:
        lines.append(f"### 总体统计\n")
        lines.append(f"- 总样本数: {gate_analysis['total_samples']}")
        lines.append(
            f"- 通过gate: {gate_analysis['passed_gate']} ({gate_analysis['passed_gate']/gate_analysis['total_samples']*100:.1f}%)"
        )
        lines.append(
            f"- 被拒绝: {gate_analysis['failed_gate']} ({gate_analysis['failed_gate']/gate_analysis['total_samples']*100:.1f}%)"
        )
        lines.append(
            f"- default_action拒绝: {gate_analysis['default_action_rejects']}\n"
        )

        lines.append("### deny_if规则统计\n")
        lines.append("| 规则名称 | 触发次数 | 触发率 |")
        lines.append("|---------|---------|--------|")
        for rule_name, count in sorted(
            gate_analysis["deny_rule_stats"].items(), key=lambda x: x[1], reverse=True
        ):
            rate = (
                count / gate_analysis["total_samples"] * 100
                if gate_analysis["total_samples"] > 0
                else 0
            )
            lines.append(f"| {rule_name} | {count} | {rate:.1f}% |")

        lines.append("\n### allow_if规则统计\n")
        lines.append("| 规则名称 | 满足次数 | 满足率 |")
        lines.append("|---------|---------|--------|")
        for rule_name, count in sorted(
            gate_analysis["allow_rule_stats"].items(), key=lambda x: x[1], reverse=True
        ):
            rate = (
                count / gate_analysis["total_samples"] * 100
                if gate_analysis["total_samples"] > 0
                else 0
            )
            lines.append(f"| {rule_name} | {count} | {rate:.1f}% |")

    lines.append("\n---\n")

    # Evidence Rules分析
    lines.append("## Evidence Rules详细分析\n")
    if "error" in evidence_analysis:
        lines.append(f"⚠️ 错误: {evidence_analysis['error']}\n")
    else:
        lines.append(f"### 总体统计\n")
        lines.append(f"- 总样本数（通过gate后）: {evidence_analysis['total_samples']}")
        lines.append(
            f"- 通过evidence: {evidence_analysis['passed_evidence']} ({evidence_analysis['passed_evidence']/evidence_analysis['total_samples']*100:.1f}%)"
        )
        lines.append(
            f"- 被拒绝: {evidence_analysis['failed_evidence']} ({evidence_analysis['failed_evidence']/evidence_analysis['total_samples']*100:.1f}%)\n"
        )

        lines.append("### Required Evidence统计\n")
        lines.append("| Evidence名称 | 通过次数 | 失败次数 | 通过率 |")
        lines.append("|------------|---------|---------|--------|")
        for ev_name, stats in evidence_analysis["required_evidence_stats"].items():
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total * 100 if total > 0 else 0
            lines.append(
                f"| {ev_name} | {stats['passed']} | {stats['failed']} | {rate:.1f}% |"
            )

    lines.append("\n---\n")

    # 优先级分析
    lines.append("## FR vs ET竞争分析\n")
    lines.append("| 情况 | 样本数 |")
    lines.append("|------|--------|")
    lines.append(f"| 两者都通过 | {priority_analysis['both_passed']} |")
    lines.append(f"| 只有ET通过 | {priority_analysis['only_et_passed']} |")
    lines.append(f"| 只有FR通过 | {priority_analysis['only_fr_passed']} |")
    lines.append(f"| 两者都不通过 | {priority_analysis['neither_passed']} |\n")

    lines.append("### Score分布比较\n")
    lines.append(f"**ET Score**:")
    lines.append(f"- 平均: {priority_analysis['et_scores']['mean']:.4f}")
    lines.append(f"- 中位数: {priority_analysis['et_scores']['median']:.4f}")
    lines.append(f"- 标准差: {priority_analysis['et_scores']['std']:.4f}")
    lines.append(f"- 样本数: {priority_analysis['et_scores']['count']}\n")

    lines.append(f"**FR Score**:")
    lines.append(f"- 平均: {priority_analysis['fr_scores']['mean']:.4f}")
    lines.append(f"- 中位数: {priority_analysis['fr_scores']['median']:.4f}")
    lines.append(f"- 标准差: {priority_analysis['fr_scores']['std']:.4f}")
    lines.append(f"- 样本数: {priority_analysis['fr_scores']['count']}\n")

    lines.append("\n---\n")

    # 特征可用性分析
    lines.append("## 特征可用性分析\n")
    lines.append(
        f"### Gate Rules需要的特征 ({len(feature_analysis['gate_features'])}个)\n"
    )
    for feat in feature_analysis["gate_features"]:
        lines.append(f"- `{feat}`")

    lines.append(
        f"\n### Evidence Rules需要的特征 ({len(feature_analysis['evidence_features'])}个)\n"
    )
    for feat in feature_analysis["evidence_features"][:20]:  # 只显示前20个
        lines.append(f"- `{feat}`")
    if len(feature_analysis["evidence_features"]) > 20:
        lines.append(
            f"- ... 还有 {len(feature_analysis['evidence_features']) - 20} 个特征"
        )

    lines.append("\n### 特征缺失统计（Top 10）\n")
    lines.append("| 特征名称 | 可用数 | 缺失数 | 缺失率 |")
    lines.append("|---------|--------|--------|--------|")
    sorted_features = sorted(
        feature_analysis["feature_stats"].items(),
        key=lambda x: x[1]["missing_rate"],
        reverse=True,
    )[:10]
    for feat, stats in sorted_features:
        lines.append(
            f"| `{feat}` | {stats['available']} | {stats['missing']} | {stats['missing_rate']*100:.1f}% |"
        )

    lines.append("\n---\n")

    # 优化建议
    lines.append("## 优化建议\n")

    if "error" not in gate_analysis:
        # 基于gate rules分析的建议
        deny_stats = gate_analysis["deny_rule_stats"]
        allow_stats = gate_analysis["allow_rule_stats"]

        top_deny_rule = (
            max(deny_stats.items(), key=lambda x: x[1]) if deny_stats else None
        )
        if top_deny_rule and top_deny_rule[1] > 0:
            lines.append(
                f"1. **最严格的deny_if规则**: `{top_deny_rule[0]}` 拒绝了 {top_deny_rule[1]} 个样本"
            )
            lines.append(f"   - 建议: 考虑放宽此规则或检查其必要性\n")

        if allow_stats:
            min_allow_rule = (
                min(allow_stats.items(), key=lambda x: x[1]) if allow_stats else None
            )
            if min_allow_rule:
                lines.append(
                    f"2. **最难满足的allow_if规则**: `{min_allow_rule[0]}` 只满足了 {min_allow_rule[1]} 次"
                )
                lines.append(f"   - 建议: 考虑降低此规则的阈值或增加allow_if选项\n")

        if gate_analysis["default_action_rejects"] > 0:
            lines.append(
                f"3. **default_action拒绝**: {gate_analysis['default_action_rejects']} 个样本因为default_action: deny被拒绝"
            )
            lines.append(
                f"   - 建议: 考虑将default_action改为allow，或增加更多allow_if选项\n"
            )

    if "error" not in evidence_analysis:
        # 基于evidence rules分析的建议
        req_ev_stats = evidence_analysis["required_evidence_stats"]
        for ev_name, stats in req_ev_stats.items():
            total = stats["passed"] + stats["failed"]
            if total > 0 and stats["failed"] > stats["passed"]:
                rate = stats["passed"] / total * 100
                lines.append(
                    f"4. **Evidence `{ev_name}`通过率低**: {rate:.1f}% ({stats['passed']}/{total})"
                )
                lines.append(f"   - 建议: 考虑放宽此evidence的要求\n")

    if priority_analysis["both_passed"] > 0:
        lines.append(
            f"5. **FR vs ET竞争**: {priority_analysis['both_passed']} 个样本两者都通过"
        )
        lines.append(f"   - 建议: 检查gate决策逻辑，确保ET有公平的竞争机会\n")

    # 保存报告
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Detailed diagnosis of ET missing data")
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (must contain regime, gate_archetype, etc.)",
    )
    p.add_argument(
        "--gated",
        default=None,
        help="Gated file (optional, for comparison)",
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Execution archetypes config",
    )
    p.add_argument(
        "--evidence-quantiles",
        default=None,
        help="Evidence quantiles JSON file",
    )
    p.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore root directory",
    )
    p.add_argument(
        "--feature-store-layer",
        default=None,
        help="FeatureStore layer name",
    )
    p.add_argument(
        "--timeframe",
        default="240T",
        help="Timeframe",
    )
    p.add_argument(
        "--output-json",
        default="results/et_detailed_diagnosis.json",
        help="Output JSON file",
    )
    p.add_argument(
        "--output-md",
        default="results/et_detailed_diagnosis.md",
        help="Output Markdown report file",
    )
    args = p.parse_args()

    print("=" * 80)
    print("ET数据缺失深度诊断")
    print("=" * 80)

    # 读取数据
    logs_path = Path(args.logs)
    if not logs_path.is_absolute():
        logs_path = PROJECT_ROOT / logs_path

    if logs_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(logs_path)
    else:
        df = pd.read_csv(logs_path)

    print(f"\n总样本数: {len(df)}")

    # 从FeatureStore读取缺失的特征
    if args.feature_store_layer:
        print("\n从FeatureStore读取缺失特征...")
        required_features = [
            "vpin",
            "cvd_change_5",
            "vpvr_lvn_distance",
            "volume_ratio",
            "bb_width_normalized",
            "sr_distance_normalized",
            "adx",
            "sqs",
            "trade_quality",
            "mean_score",
            "atr_percentile",
        ]
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            store = FeatureStore(args.feature_store_root)
            symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []

            parts = []
            for sym in symbols:
                spec = FeatureStoreSpec(
                    layer=args.feature_store_layer,
                    symbol=str(sym),
                    timeframe=args.timeframe,
                )
                sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df
                if not sym_df.empty and "timestamp" in sym_df.columns:
                    start_ts = pd.to_datetime(sym_df["timestamp"].min())
                    end_ts = pd.to_datetime(sym_df["timestamp"].max())
                    feat_df = store.read_range(spec, start=start_ts, end=end_ts)
                    if not feat_df.empty:
                        if "symbol" not in feat_df.columns:
                            feat_df["symbol"] = sym
                        if (
                            "timestamp" not in feat_df.columns
                            and getattr(feat_df.index, "name", None) == "timestamp"
                        ):
                            feat_df = feat_df.reset_index()
                        parts.append(
                            feat_df[
                                ["symbol", "timestamp"]
                                + [f for f in missing_features if f in feat_df.columns]
                            ]
                        )

            if parts:
                feats_df = pd.concat(parts, axis=0, ignore_index=False)
                if "timestamp" in feats_df.columns:
                    feats_df["timestamp"] = pd.to_datetime(
                        feats_df["timestamp"], errors="coerce"
                    )
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                df = df.merge(
                    feats_df,
                    on=["symbol", "timestamp"],
                    how="left",
                    suffixes=("", "_feat"),
                )
                print(
                    f"✅ 从FeatureStore读取了 {len([f for f in missing_features if f in df.columns])} 个特征"
                )

    # 读取archetypes
    archetypes = load_execution_archetypes_registry(args.execution_archetypes)

    # 找到ET和FR archetypes
    et_arch = None
    fr_arch = None
    for arch_name, arch_obj in archetypes.items():
        if "ET" in arch_name.upper() or "ExhaustionTurn" in arch_name:
            et_arch = arch_obj
        if "FR" in arch_name.upper() or "FailureReversion" in arch_name:
            fr_arch = arch_obj

    if not et_arch:
        print("❌ 错误: 未找到ET archetype")
        return 1

    # 读取quantiles
    quantiles = load_evidence_quantiles(args.evidence_quantiles)
    if quantiles is None:
        print("⚠️  警告: 未提供evidence quantiles，将使用特征值直接比较")

    # 1. 基础统计
    print("\n1. 分析基础统计...")
    basic_stats = analyze_basic_stats(df)
    print(f"   MEAN_REGIME样本数: {basic_stats['total_mean_regime']}")
    print(f"   ET候选数: {basic_stats['et_candidates']}")
    print(f"   FR候选数: {basic_stats['fr_candidates']}")

    # 2. Gate Rules详细分析
    print("\n2. 分析Gate Rules...")
    gate_analysis = analyze_gate_rules_detailed(df, et_arch, quantiles)
    if "error" not in gate_analysis:
        print(
            f"   通过gate: {gate_analysis['passed_gate']}/{gate_analysis['total_samples']}"
        )
        print(
            f"   被拒绝: {gate_analysis['failed_gate']}/{gate_analysis['total_samples']}"
        )

    # 3. Evidence Rules详细分析（只对通过gate的样本）
    print("\n3. 分析Evidence Rules...")
    gate_passed_mask = None
    if "error" not in gate_analysis and gate_analysis["passed_gate"] > 0:
        # 创建gate passed mask（简化版，实际应该从gate_analysis的sample_details中提取）
        # 这里我们重新计算一次
        mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()
        gate_passed_indices = []
        for idx, row in mean_regime.iterrows():
            features = row.to_dict()
            symbol = features.get("symbol", "ALL")
            symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None
            try:
                ok, _ = apply_gate_rules(
                    gate_rules=et_arch.gate_rules,
                    features=features,
                    quantiles=symbol_quantiles,
                )
                if ok:
                    gate_passed_indices.append(idx)
            except Exception:
                pass
        gate_passed_mask = pd.Series(False, index=mean_regime.index)
        gate_passed_mask.loc[gate_passed_indices] = True

    evidence_analysis = analyze_evidence_rules_detailed(
        df, et_arch, quantiles, gate_passed_mask
    )
    if "error" not in evidence_analysis:
        print(
            f"   通过evidence: {evidence_analysis['passed_evidence']}/{evidence_analysis['total_samples']}"
        )

    # 4. 优先级分析
    print("\n4. 分析优先级和竞争...")
    priority_analysis = analyze_priority_competition(df, et_arch, fr_arch, quantiles)
    print(f"   两者都通过: {priority_analysis['both_passed']}")
    print(f"   只有ET通过: {priority_analysis['only_et_passed']}")
    print(f"   只有FR通过: {priority_analysis['only_fr_passed']}")

    # 5. 特征可用性分析
    print("\n5. 分析特征可用性...")
    feature_analysis = analyze_feature_availability(df, et_arch)
    print(f"   Gate features: {len(feature_analysis['gate_features'])}")
    print(f"   Evidence features: {len(feature_analysis['evidence_features'])}")

    # 保存JSON结果
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "basic_stats": basic_stats,
        "gate_analysis": gate_analysis,
        "evidence_analysis": evidence_analysis,
        "priority_analysis": priority_analysis,
        "feature_analysis": feature_analysis,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ JSON结果已保存到: {output_json_path}")

    # 生成Markdown报告
    output_md_path = Path(args.output_md)
    generate_report(
        basic_stats,
        gate_analysis,
        evidence_analysis,
        priority_analysis,
        feature_analysis,
        output_md_path,
    )

    print(f"✅ Markdown报告已保存到: {output_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
