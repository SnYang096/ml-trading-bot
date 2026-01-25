#!/usr/bin/env python3
"""
分析regime作为gate veto的架构优化

分析：
1. 当前regime filter的实现
2. 设计regime作为gate veto的方案
3. 评估迁移成本和收益
4. 提出实施计划
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def analyze_current_regime_filter() -> Dict[str, Any]:
    """分析当前regime filter的实现"""
    gate_script = PROJECT_ROOT / "scripts/apply_archetype_gate.py"

    with open(gate_script, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找regime filter相关的代码
    regime_filter_lines = []
    in_regime_section = False

    for i, line in enumerate(content.split("\n"), 1):
        if "regime" in line.lower() and (
            "filter" in line.lower() or "normalized" in line.lower()
        ):
            regime_filter_lines.append((i, line.strip()))
        if "disable_regime_filter" in line:
            in_regime_section = True
        if in_regime_section and "candidates" in line and "=" in line:
            regime_filter_lines.append((i, line.strip()))
            in_regime_section = False

    return {
        "regime_filter_code_lines": regime_filter_lines[:20],  # 前20行
        "has_regime_filter": "disable_regime_filter" in content,
        "regime_based_candidate_selection": "regime_for_lookup" in content,
    }


def design_regime_as_gate_veto(archetypes: Dict[str, Any]) -> Dict[str, Any]:
    """设计regime作为gate veto的方案"""
    design = {
        "current_regime_mapping": {},
        "proposed_gate_rules": {},
    }

    # 分析当前regime到archetype的映射
    for arch_name, arch_obj in archetypes.items():
        # 从archetype名称推断regime
        if "TC" in arch_name.upper() or "TrendContinuation" in arch_name:
            design["current_regime_mapping"][arch_name] = "TC_REGIME"
        elif "TE" in arch_name.upper() or "TrendExpansion" in arch_name:
            design["current_regime_mapping"][arch_name] = "TE_REGIME"
        elif "FR" in arch_name.upper() or "FailureReversion" in arch_name:
            design["current_regime_mapping"][arch_name] = "MEAN_REGIME"
        elif "ET" in arch_name.upper() or "ExhaustionTurn" in arch_name:
            design["current_regime_mapping"][arch_name] = "MEAN_REGIME"

    # 设计新的gate rules
    for arch_name, regime in design["current_regime_mapping"].items():
        design["proposed_gate_rules"][arch_name] = {
            "regime_veto_rule": {
                "name": f"{arch_name.lower()}_regime_mismatch",
                "kind": "value_ne",  # value not equal
                "key": "regime",
                "value": regime,
                "on_missing": False,
            },
            "add_to_deny_if": True,
        }

    return design


def evaluate_migration_cost() -> Dict[str, Any]:
    """评估迁移成本"""
    return {
        "code_changes": [
            "scripts/apply_archetype_gate.py: 移除regime filter逻辑（~30行）",
            "config/nnmultihead/execution_archetypes.yaml: 为每个archetype添加regime veto规则",
            "config/nnmultihead/live/meta_router_live_config.yaml: 更新enabled_archetypes语义（可选）",
        ],
        "testing_required": [
            "验证所有archetype的regime veto规则正确工作",
            "验证gate决策逻辑与之前一致",
            "验证KPI报告正确性",
        ],
        "risk_level": "medium",
        "backward_compatibility": "需要保持向后兼容（通过--disable-regime-filter flag）",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze regime as gate veto architecture")
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Execution archetypes config",
    )
    p.add_argument(
        "--output",
        default="results/regime_as_gate_veto_analysis.json",
        help="Output JSON file",
    )
    args = p.parse_args()

    print("=" * 80)
    print("分析regime作为gate veto的架构优化")
    print("=" * 80)

    # 1. 分析当前regime filter实现
    print("\n1. 当前regime filter实现:")
    print("-" * 80)
    current_impl = analyze_current_regime_filter()
    print(f"  有regime filter: {current_impl['has_regime_filter']}")
    print(f"  基于regime的候选选择: {current_impl['regime_based_candidate_selection']}")
    print(f"  相关代码行数: {len(current_impl['regime_filter_code_lines'])}")

    # 2. 设计新方案
    print("\n2. 设计regime作为gate veto的方案:")
    print("-" * 80)
    archetypes = load_execution_archetypes_registry(args.execution_archetypes)
    design = design_regime_as_gate_veto(archetypes)

    print("  当前regime映射:")
    for arch_name, regime in design["current_regime_mapping"].items():
        print(f"    {arch_name} -> {regime}")

    print("\n  建议的gate rules（regime veto）:")
    for arch_name, rule_info in design["proposed_gate_rules"].items():
        rule = rule_info["regime_veto_rule"]
        print(f"    {arch_name}:")
        print(f"      - name: {rule['name']}")
        print(f"      - kind: {rule['kind']}")
        print(f"      - key: {rule['key']}")
        print(f"      - value: {rule['value']}")
        print(f"      - 添加到deny_if: {rule_info['add_to_deny_if']}")

    # 3. 评估迁移成本
    print("\n3. 评估迁移成本:")
    print("-" * 80)
    migration = evaluate_migration_cost()
    print(f"  风险级别: {migration['risk_level']}")
    print(f"  代码修改:")
    for change in migration["code_changes"]:
        print(f"    - {change}")
    print(f"  测试需求:")
    for test in migration["testing_required"]:
        print(f"    - {test}")
    print(f"  向后兼容: {migration['backward_compatibility']}")

    # 4. 收益分析
    print("\n4. 收益分析:")
    print("-" * 80)
    print("  ✅ 优势:")
    print("    1. 每个archetype可以定义自己的regime要求（更灵活）")
    print("    2. Regime和gate rules统一管理（更容易维护）")
    print("    3. 更符合regime定义（市场状态 vs 执行策略）")
    print("    4. 更容易找到平稳参数（可以单独优化每个archetype的regime要求）")
    print("    5. 后续优化evidence时，regime和evidence可以独立调参")
    print("\n  ⚠️  注意事项:")
    print("    1. 需要为每个archetype添加regime veto规则")
    print("    2. 需要验证gate决策逻辑与之前一致")
    print("    3. 需要更新相关文档")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "current_implementation": current_impl,
        "proposed_design": design,
        "migration_cost": migration,
        "benefits": {
            "flexibility": "每个archetype可以定义自己的regime要求",
            "maintainability": "Regime和gate rules统一管理",
            "semantics": "更符合regime定义（市场状态 vs 执行策略）",
            "optimization": "更容易找到平稳参数，可以单独优化每个archetype",
        },
    }

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
