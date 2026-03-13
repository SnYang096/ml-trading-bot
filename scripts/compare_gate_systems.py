#!/usr/bin/env python3
"""
Gate系统一致性测试脚本
比较向量回测和事件回测中的gate评估行为
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.archetype.loader import (
    load_strategy_archetype,
    _evaluate_when_clause,
)
from src.time_series_model.live.tree_gate import (
    apply_gate_rules,
    apply_when_then_rules,
    _eval_when_clause as tree_gate_eval_when,
)


def load_gate_configs():
    """加载所有策略的gate配置"""
    configs = {}
    strategies = ["bpc", "fer", "me-long"]

    for strategy in strategies:
        # 加载事件回测使用的hard_gates格式配置
        hard_gates_path = Path(f"config/strategies/{strategy}/archetypes/gate.yaml")
        if hard_gates_path.exists():
            with open(hard_gates_path, "r") as f:
                hard_gates_config = yaml.safe_load(f)
                configs[f"{strategy}_hard_gates"] = hard_gates_config

        # 向量回测使用的是StrategyArchetype的gate配置转换为when_then_rules格式
        # 我们通过StrategyArchetype的gate_rules属性来访问这个格式
        try:
            archetype = load_strategy_archetype(strategy)
            configs[f"{strategy}_when_then_rules"] = archetype.gate_rules
        except Exception as e:
            print(f"Warning: Could not load archetype for {strategy}: {e}")

    return configs


def create_sample_features():
    """创建样本特征数据用于测试"""
    sample_features = {
        # BPC相关特征
        "bpc_cvd_z": 1.2,
        "bpc_bb_compression": 0.3,
        "atr": 100.0,
        # FER相关特征
        "fer_trapped": -0.5,
        "fer_ma_distance": 0.1,
        # ME相关特征
        "me_atr_pct": 0.02,
        "me_volatility_ratio": 1.5,
        # 通用特征
        "volume": 1000000,
        "price": 40000.0,
        "rsi": 65.0,
        "macd": 0.5,
        # 时间相关特征
        "hour": 14,
        "day_of_week": 2,
    }
    return sample_features


def test_archetype_gate_evaluation():
    """测试StrategyArchetype的gate评估"""
    print("=== 测试 StrategyArchetype gate评估 ===")

    strategies = ["bpc", "fer", "me-long"]
    sample_features = create_sample_features()

    for strategy in strategies:
        print(f"\n--- 测试 {strategy.upper()} 策略 ---")

        try:
            # 加载StrategyArchetype实例
            archetype = load_strategy_archetype(strategy)

            # 测试gate评估
            gate_result = archetype.apply_gate(sample_features)
            print(f"  Gate评估结果: {gate_result}")

            # 显示具体的gate规则数量
            print(f"  总规则数量: {len(archetype.gate.all_rules)}")
            print(f"  System safety规则: {len(archetype.gate.system_safety)}")
            print(f"  Hard gate规则: {len(archetype.gate.hard_gates)}")
            print(f"  Guardrail规则: {len(archetype.gate.guardrails)}")

            # 测试每条规则
            for i, rule in enumerate(archetype.gate.all_rules):
                matched = _evaluate_when_clause(rule.when, sample_features, None)
                action = rule.then.get("action", "deny")
                print(
                    f"    Rule {i+1} ({rule.tag}): when={rule.when}, matched={matched}, action={action}"
                )

        except Exception as e:
            print(f"  Error loading archetype for {strategy}: {e}")


def test_tree_gate_evaluation():
    """测试tree_gate系统的gate评估"""
    print("\n=== 测试 tree_gate gate评估 ===")

    strategies = ["bpc", "fer", "me-long"]
    sample_features = create_sample_features()

    for strategy in strategies:
        print(f"\n--- 测试 {strategy.upper()} 策略 (tree_gate) ---")

        try:
            # 获取StrategyArchetype的when_then_rules格式配置
            archetype = load_strategy_archetype(strategy)
            gate_rules = archetype.gate_rules

            print(
                f"  When-then rules数量: {len(gate_rules.get('when_then_rules', []))}"
            )

            # 使用apply_gate_rules进行评估
            gate_result = apply_gate_rules(
                gate_rules=gate_rules, features=sample_features, quantiles=None
            )
            print(f"  Gate评估结果: {gate_result}")

            # 如果有规则，尝试详细评估每个规则
            for i, rule in enumerate(gate_rules.get("when_then_rules", [])):
                when_clause = rule.get("when", {})
                try:
                    rule_result = tree_gate_eval_when(
                        when_clause, features=sample_features, quantiles=None
                    )
                    print(
                        f"    Rule {i+1} ({rule.get('reason', rule.get('id', 'unnamed'))}): when={when_clause}, matched={rule_result}"
                    )
                except Exception as e:
                    print(f"    Rule {i+1} error: {e}")

        except Exception as e:
            print(f"  Error applying gate rules for {strategy}: {e}")


def compare_gate_operators():
    """比较两种gate系统支持的操作符"""
    print("\n=== 比较Gate操作符支持 ===")

    # StrategyArchetype支持的操作符 (_evaluate_when_clause函数)
    archetype_operators = [
        "value_eq",
        "value_ne",
        "value_lt",
        "value_lte",
        "value_le",
        "value_gt",
        "value_gte",
        "value_ge",
        "value_in",
        "value_not_in",
        "value_between",
        "value_not_between",
        "value_contains",
        "value_not_contains",
        "quantile_lt",
        "quantile_lte",
        "quantile_gt",
        "quantile_gte",
    ]

    # tree_gate系统支持的操作符 (_eval_when_clause函数)
    tree_gate_operators = [
        "value_eq",
        "value_ne",
        "value_lt",
        "value_lte",
        "value_gt",
        "value_gte",
        "value_in",
        "value_not_in",
        "value_between",
        "value_not_between",
        "value_contains",
        "value_not_contains",
        "quantile_lt",
        "quantile_lte",
        "quantile_gt",
        "quantile_gte",
        "all_of",
        "any_of",
        "not",
        "any_key_contains",
    ]

    print(f"StrategyArchetype operators: {archetype_operators}")
    print(f"Tree gate operators: {tree_gate_operators}")

    # 检查差异
    archetype_only = set(archetype_operators) - set(tree_gate_operators)
    tree_only = set(tree_gate_operators) - set(archetype_operators)

    print(f"仅StrategyArchetype支持: {archetype_only}")
    print(f"仅tree_gate支持: {tree_only}")


def analyze_gate_yaml_contents():
    """分析gate.yaml文件内容"""
    print("\n=== 分析Gate YAML配置内容 ===")

    strategies = ["bpc", "fer", "me-long"]

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 配置 ---")

        # 分析hard_gates格式
        hard_gates_path = Path(f"config/strategies/{strategy}/archetypes/gate.yaml")
        if hard_gates_path.exists():
            with open(hard_gates_path, "r") as f:
                hard_gates_config = yaml.safe_load(f)

            print(f"  Hard gates配置结构:")
            for key, value in hard_gates_config.items():
                if isinstance(value, list):
                    print(f"    {key}: {len(value)} 条规则")
                    for i, rule in enumerate(value[:3]):  # 只显示前3个
                        if isinstance(rule, dict):
                            tag = rule.get("tag", "unnamed")
                            when = rule.get("when", {})
                            then = rule.get("then", {})
                            print(f"      {i+1}. {tag}: when={when}, then={then}")
                else:
                    print(f"    {key}: {type(value).__name__ if value else value}")


def direct_comparison_test():
    """直接比较两个系统的评估结果"""
    print("\n=== 直接比较测试 ===")

    strategies = ["bpc", "fer", "me-long"]
    sample_features = create_sample_features()

    print(f"使用特征样例: {dict(list(sample_features.items())[:5])}...")

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 直接比较 ---")

        try:
            # 加载archetype
            archetype = load_strategy_archetype(strategy)

            # 通过StrategyArchetype评估
            arch_result = archetype.apply_gate(sample_features)
            print(f"  StrategyArchetype结果: {arch_result}")

            # 通过tree_gate系统评估 (使用相同的when_then_rules)
            gate_rules = archetype.gate_rules
            tree_result = apply_gate_rules(
                gate_rules=gate_rules, features=sample_features, quantiles=None
            )
            print(f"  Tree gate结果: {tree_result}")

            # 比较结果是否一致
            arch_passed = arch_result[0]
            tree_passed = tree_result[0]

            if arch_passed == tree_passed:
                print(f"  ✅ 一致性: 两个系统评估结果相同")
            else:
                print(
                    f"  ❌ 不一致: StrategyArchetype={arch_passed}, Tree gate={tree_passed}"
                )

        except Exception as e:
            print(f"  Error comparing {strategy}: {e}")


def main():
    print("Gate系统一致性测试")
    print("=" * 50)

    # 加载配置
    configs = load_gate_configs()
    print(f"加载到 {len(configs)} 个配置文件")

    # 分析YAML内容
    analyze_gate_yaml_contents()

    # 比较操作符支持
    compare_gate_operators()

    # 测试StrategyArchetype评估
    test_archetype_gate_evaluation()

    # 测试tree_gate评估
    test_tree_gate_evaluation()

    # 直接比较测试
    direct_comparison_test()

    print("\n" + "=" * 50)
    print("测试完成")


if __name__ == "__main__":
    main()
