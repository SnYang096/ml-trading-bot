#!/usr/bin/env python3
"""
调试Gate系统边缘案例
"""

import yaml
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.archetype.loader import (
    load_strategy_archetype,
    _evaluate_when_clause,
)
from src.time_series_model.live.tree_gate import (
    apply_gate_rules,
    _eval_when_clause as tree_gate_eval_when,
)


def test_edge_cases():
    """测试边缘案例"""
    print("=== 测试边缘案例 ===")

    strategies = ["bpc", "fer", "me-long"]

    # 边缘案例1: 特征值正好等于阈值
    edge_case_features = {
        "bpc_bb_compression": 0.0965,  # 正好等于阈值
        "bpc_cvd_z": -2.0018,  # 正好等于阈值
        "bpc_volume_compression_pct": 0.1503,  # 正好等于阈值
        "atr_percentile": 0.8167,  # 正好等于阈值
        "me_atr_pct": 0.01,  # 正好等于阈值
        "me_cvd_alignment": 1.0,  # 正好等于阈值
        "cvd_divergence_score": -0.0306,  # 正好等于阈值
        "roc_20": 1.2654,  # 正好等于阈值
        "spectrum_cvd_high_freq_ratio": 0.381,  # 正好等于阈值
        "vp_width_ratio": 0.5814,  # 正好等于阈值
        "evt_var_99": 0.6829,  # 正好等于阈值
        "evt_scale": 0.601,  # 正好等于阈值
        "fer_trapped_longs_score": 3.4965,  # 正好等于阈值
        "fer_trapped_shorts_score": 3.7941,  # 正好等于阈值
    }

    print(f"边缘案例特征: 使用精确阈值测试")

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 边缘案例测试 ---")

        try:
            # 加载archetype
            archetype = load_strategy_archetype(strategy)

            # 通过StrategyArchetype评估
            arch_result = archetype.apply_gate(edge_case_features)
            print(f"  StrategyArchetype结果: {arch_result}")

            # 通过tree_gate系统评估
            gate_rules = archetype.gate_rules
            tree_result = apply_gate_rules(
                gate_rules=gate_rules, features=edge_case_features, quantiles=None
            )
            print(f"  Tree gate结果: {tree_result}")

            # 比较结果
            arch_passed = arch_result[0]
            tree_passed = tree_result[0]

            if arch_passed == tree_passed:
                print(f"  ✅ 一致性: 两个系统评估结果相同")
            else:
                print(
                    f"  ❌ 不一致: StrategyArchetype={arch_passed}, Tree gate={tree_passed}"
                )

                # 详细分析每条规则
                print(f"  详细分析:")
                for i, rule in enumerate(archetype.gate.all_rules):
                    # StrategyArchetype评估
                    arch_matched = _evaluate_when_clause(
                        rule.when, edge_case_features, None
                    )

                    # Tree gate评估
                    tree_matched = tree_gate_eval_when(
                        rule.when, features=edge_case_features, quantiles=None
                    )

                    print(
                        f"    Rule {i+1} ({rule.tag}): SA={arch_matched}, TG={tree_matched}, Diff={arch_matched != tree_matched}"
                    )

        except Exception as e:
            print(f"  Error comparing {strategy}: {e}")


def test_specific_operators():
    """测试特定操作符的行为"""
    print("\n=== 测试特定操作符 ===")

    # 测试value_le和value_ge操作符
    test_cases = [
        {
            "name": "value_le测试 - 小于阈值",
            "features": {"test_value": 0.5},
            "condition": {"test_value": {"value_le": 1.0}},
            "expected": True,
        },
        {
            "name": "value_le测试 - 等于阈值",
            "features": {"test_value": 1.0},
            "condition": {"test_value": {"value_le": 1.0}},
            "expected": True,
        },
        {
            "name": "value_le测试 - 大于阈值",
            "features": {"test_value": 1.5},
            "condition": {"test_value": {"value_le": 1.0}},
            "expected": False,
        },
        {
            "name": "value_ge测试 - 大于阈值",
            "features": {"test_value": 1.5},
            "condition": {"test_value": {"value_ge": 1.0}},
            "expected": True,
        },
        {
            "name": "value_ge测试 - 等于阈值",
            "features": {"test_value": 1.0},
            "condition": {"test_value": {"value_ge": 1.0}},
            "expected": True,
        },
        {
            "name": "value_ge测试 - 小于阈值",
            "features": {"test_value": 0.5},
            "condition": {"test_value": {"value_ge": 1.0}},
            "expected": False,
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")

        # StrategyArchetype评估
        arch_matched = _evaluate_when_clause(case["condition"], case["features"], None)

        # Tree gate评估
        tree_matched = tree_gate_eval_when(
            case["condition"], features=case["features"], quantiles=None
        )

        print(f"  特征: {case['features']}")
        print(f"  条件: {case['condition']}")
        print(f"  预期: {case['expected']}")
        print(f"  SA结果: {arch_matched}, TG结果: {tree_matched}")

        if arch_matched == tree_matched == case["expected"]:
            print(f"  ✅ 正确: 两个系统结果一致且符合预期")
        elif arch_matched == tree_matched:
            print(
                f"  ⚠️  一致但不符合预期: 两个系统结果一致({arch_matched})但不是预期({case['expected']})"
            )
        else:
            print(f"  ❌ 不一致: SA={arch_matched}, TG={tree_matched}")


def test_complex_conditions():
    """测试复杂条件"""
    print("\n=== 测试复杂条件 ===")

    # 测试all_of条件
    complex_features = {"a": 1.0, "b": 2.0, "c": 3.0}

    complex_condition = {
        "all_of": [
            {"a": {"value_le": 2.0}},
            {"b": {"value_ge": 1.0}},
            {"c": {"value_gt": 2.0}},
        ]
    }

    print(f"复杂条件测试: {complex_condition}")
    print(f"特征值: {complex_features}")

    # StrategyArchetype评估
    arch_matched = _evaluate_when_clause(complex_condition, complex_features, None)

    # Tree gate评估
    tree_matched = tree_gate_eval_when(
        complex_condition, features=complex_features, quantiles=None
    )

    print(f"SA结果: {arch_matched}, TG结果: {tree_matched}")

    if arch_matched == tree_matched:
        print(f"✅ 一致: 两个系统结果相同")
    else:
        print(f"❌ 不一致: SA={arch_matched}, TG={tree_matched}")


def main():
    print("Gate系统边缘案例调试")
    print("=" * 50)

    test_edge_cases()
    test_specific_operators()
    test_complex_conditions()

    print("\n" + "=" * 50)
    print("调试完成")


if __name__ == "__main__":
    main()
