#!/usr/bin/env python3
"""
分析ET数据缺失原因

检查：
1. MEAN_REGIME中ET的候选数量
2. ET被gate rules拒绝的原因
3. ET被evidence rules拒绝的原因
4. 提出优化建议
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)


def analyze_et_candidates(df: pd.DataFrame) -> Dict[str, Any]:
    """分析MEAN_REGIME中ET的候选数量"""
    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    stats = {
        "total_mean_regime": len(mean_regime),
        "et_candidates": 0,
        "fr_candidates": 0,
        "both_candidates": 0,
    }

    # 检查是否有ET相关的列
    if "gate_archetype" in mean_regime.columns:
        et_trades = mean_regime[
            mean_regime["gate_archetype"].str.contains("ET", case=False, na=False)
        ]
        fr_trades = mean_regime[
            mean_regime["gate_archetype"].str.contains("FR", case=False, na=False)
        ]
        stats["et_candidates"] = len(et_trades)
        stats["fr_candidates"] = len(fr_trades)
        stats["both_candidates"] = len(
            mean_regime[
                mean_regime["gate_archetype"].str.contains("ET", case=False, na=False)
                & mean_regime["gate_archetype"].str.contains("FR", case=False, na=False)
            ]
        )

    return stats


def analyze_gate_rejection(
    df: pd.DataFrame, archetypes: Dict[str, Any]
) -> Dict[str, Any]:
    """分析ET被gate rules拒绝的原因"""
    et_arch = None
    for arch_name, arch_obj in archetypes.items():
        if "ET" in arch_name.upper() or "ExhaustionTurn" in arch_name:
            et_arch = arch_obj
            break

    if not et_arch:
        return {"error": "ET archetype not found"}

    mean_regime = df[df["regime"] == "MEAN_REGIME"].copy()

    # 尝试应用gate rules
    gate_rules = et_arch.gate_rules if hasattr(et_arch, "gate_rules") else None

    if not gate_rules:
        return {"error": "ET gate_rules not found"}

    results = {
        "total_samples": len(mean_regime),
        "passed_gate": 0,
        "failed_gate": 0,
        "rejection_reasons": {},
    }

    # 这里需要实际应用gate rules，但由于需要quantiles等，先做简单统计
    # 实际分析应该在apply_tree_gate_3action.py的输出中进行

    return results


def analyze_evidence_rejection(
    df: pd.DataFrame, archetypes: Dict[str, Any]
) -> Dict[str, Any]:
    """分析ET被evidence rules拒绝的原因"""
    et_arch = None
    for arch_name, arch_obj in archetypes.items():
        if "ET" in arch_name.upper() or "ExhaustionTurn" in arch_name:
            et_arch = arch_obj
            break

    if not et_arch:
        return {"error": "ET archetype not found"}

    evidence_rules = (
        et_arch.evidence_rules if hasattr(et_arch, "evidence_rules") else []
    )

    results = {
        "total_evidence_rules": len(evidence_rules),
        "required_evidence": getattr(et_arch, "required_evidence", []),
        "evidence_rules": [
            {
                "name": rule.get("name", "unknown"),
                "kind": rule.get("kind", "unknown"),
                "key": rule.get("key", "unknown"),
            }
            for rule in evidence_rules
        ],
    }

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze ET missing data")
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (must contain regime, gate_archetype, etc.)",
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Execution archetypes config",
    )
    p.add_argument(
        "--output",
        default="results/et_missing_data_analysis.json",
        help="Output JSON file",
    )
    args = p.parse_args()

    print("=" * 80)
    print("分析ET数据缺失原因")
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

    # 读取archetypes
    archetypes = load_execution_archetypes_registry(args.execution_archetypes)

    # 1. 分析ET候选数量
    print("\n1. MEAN_REGIME中ET候选数量:")
    print("-" * 80)
    candidate_stats = analyze_et_candidates(df)
    print(f"  MEAN_REGIME总样本数: {candidate_stats['total_mean_regime']}")
    print(f"  ET候选数: {candidate_stats['et_candidates']}")
    print(f"  FR候选数: {candidate_stats['fr_candidates']}")

    # 2. 分析gate拒绝原因
    print("\n2. ET Gate Rules分析:")
    print("-" * 80)
    gate_stats = analyze_gate_rejection(df, archetypes)
    if "error" in gate_stats:
        print(f"  ⚠️  {gate_stats['error']}")
    else:
        print(f"  总样本数: {gate_stats['total_samples']}")

    # 3. 分析evidence拒绝原因
    print("\n3. ET Evidence Rules分析:")
    print("-" * 80)
    evidence_stats = analyze_evidence_rejection(df, archetypes)
    if "error" in evidence_stats:
        print(f"  ⚠️  {evidence_stats['error']}")
    else:
        print(f"  Required evidence: {evidence_stats['required_evidence']}")
        print(f"  Evidence rules数量: {evidence_stats['total_evidence_rules']}")
        print(f"  Evidence rules:")
        for rule in evidence_stats["evidence_rules"]:
            print(f"    - {rule['name']}: {rule['kind']} on {rule['key']}")

    # 4. 提出优化建议
    print("\n4. 优化建议:")
    print("-" * 80)

    if candidate_stats["total_mean_regime"] < 50:
        print("  ⚠️  MEAN_REGIME样本数太少，建议放宽MEAN_REGIME分类条件")

    if candidate_stats["et_candidates"] == 0:
        print("  ⚠️  ET候选数为0，可能原因:")
        print("     1. Gate rules太严格（需要满足allow_if中的至少1个）")
        print("     2. Evidence rules太严格（需要has_orderflow, has_volume_profile等）")
        print("     3. FR优先级更高，ET被跳过")
        print("  💡 建议:")
        print("     1. 放宽ET的gate rules（降低allow_if阈值或增加allow_if选项）")
        print("     2. 放宽ET的evidence rules（降低has_orderflow quantile）")
        print("     3. 检查ET在gate决策中的优先级")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "candidate_stats": candidate_stats,
        "gate_stats": gate_stats,
        "evidence_stats": evidence_stats,
    }

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
