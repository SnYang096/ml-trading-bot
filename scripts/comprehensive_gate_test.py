#!/usr/bin/env python3
"""
综合Gate系统测试脚本
验证修复后的gate系统一致性
"""

import yaml
from pathlib import Path
import sys
import pandas as pd

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


def comprehensive_operator_test():
    """全面测试各种操作符的一致性"""
    print("=== 全面操作符一致性测试 ===")

    # 测试数据
    test_cases = [
        {
            "name": "value_le 小于情况",
            "features": {"test_val": 0.5},
            "condition": {"test_val": {"value_le": 1.0}},
            "expected": True,
        },
        {
            "name": "value_le 等于情况",
            "features": {"test_val": 1.0},
            "condition": {"test_val": {"value_le": 1.0}},
            "expected": True,
        },
        {
            "name": "value_le 大于情况",
            "features": {"test_val": 1.5},
            "condition": {"test_val": {"value_le": 1.0}},
            "expected": False,
        },
        {
            "name": "value_ge 小于情况",
            "features": {"test_val": 0.5},
            "condition": {"test_val": {"value_ge": 1.0}},
            "expected": False,
        },
        {
            "name": "value_ge 等于情况",
            "features": {"test_val": 1.0},
            "condition": {"test_val": {"value_ge": 1.0}},
            "expected": True,
        },
        {
            "name": "value_ge 大于情况",
            "features": {"test_val": 1.5},
            "condition": {"test_val": {"value_ge": 1.0}},
            "expected": True,
        },
        {
            "name": "value_lte 小于情况",
            "features": {"test_val": 0.5},
            "condition": {"test_val": {"value_lte": 1.0}},
            "expected": True,
        },
        {
            "name": "value_lte 等于情况",
            "features": {"test_val": 1.0},
            "condition": {"test_val": {"value_lte": 1.0}},
            "expected": True,
        },
        {
            "name": "value_lte 大于情况",
            "features": {"test_val": 1.5},
            "condition": {"test_val": {"value_lte": 1.0}},
            "expected": False,
        },
        {
            "name": "value_gte 小于情况",
            "features": {"test_val": 0.5},
            "condition": {"test_val": {"value_gte": 1.0}},
            "expected": False,
        },
        {
            "name": "value_gte 等于情况",
            "features": {"test_val": 1.0},
            "condition": {"test_val": {"value_gte": 1.0}},
            "expected": True,
        },
        {
            "name": "value_gte 大于情况",
            "features": {"test_val": 1.5},
            "condition": {"test_val": {"value_gte": 1.0}},
            "expected": True,
        },
    ]

    all_passed = True

    for case in test_cases:
        # 测试两个系统
        sa_result = _evaluate_when_clause(case["condition"], case["features"], None)
        tg_result = tree_gate_eval_when(
            case["condition"], features=case["features"], quantiles=None
        )

        # 检查一致性
        consistent = sa_result == tg_result == case["expected"]

        status = "✅" if consistent else "❌"
        print(
            f"{status} {case['name']}: SA={sa_result}, TG={tg_result}, Expected={case['expected']}"
        )

        if not consistent:
            all_passed = False

    return all_passed


def real_world_gate_test():
    """真实世界的gate规则测试"""
    print("\n=== 真实世界Gate规则测试 ===")

    strategies = ["bpc", "fer", "me-long"]
    all_passed = True

    # 创建测试特征数据，覆盖各种边界情况
    test_features_sets = [
        {
            "name": "正常市场条件",
            "features": {
                "bpc_bb_compression": 0.15,
                "bpc_cvd_z": 1.5,
                "bpc_volume_compression_pct": 0.25,
                "atr_percentile": 0.85,
                "me_atr_pct": 0.015,
                "me_cvd_alignment": 1.2,
                "fer_trapped_longs_score": 2.5,
                "fer_trapped_shorts_score": 2.0,
                "roc_20": 0.8,
                "cvd_divergence_score": -0.01,
                "spectrum_cvd_high_freq_ratio": 0.35,
                "vp_width_ratio": 0.65,
                "evt_var_99": 0.75,
                "evt_scale": 0.65,
            },
        },
        {
            "name": "边界条件（等于阈值）",
            "features": {
                "bpc_bb_compression": 0.0965,  # 等于BPC阈值
                "bpc_cvd_z": -2.0018,  # 等于BPC阈值
                "bpc_volume_compression_pct": 0.1503,  # 等于BPC阈值
                "atr_percentile": 0.8167,  # 等于ME阈值
                "me_atr_pct": 0.01,  # 等于ME阈值
                "me_cvd_alignment": 1.0,  # 等于ME阈值
                "fer_trapped_longs_score": 3.4965,  # 等于FER阈值
                "fer_trapped_shorts_score": 3.7941,  # 等于FER阈值
                "roc_20": 1.2654,  # 等于FER阈值
                "cvd_divergence_score": -0.0306,  # 等于FER阈值
                "spectrum_cvd_high_freq_ratio": 0.381,  # 等于FER阈值
            },
        },
    ]

    for features_set in test_features_sets:
        print(f"\n--- {features_set['name']} ---")

        for strategy in strategies:
            try:
                # 加载archetype
                archetype = load_strategy_archetype(strategy)

                # 通过StrategyArchetype评估
                sa_result = archetype.apply_gate(features_set["features"])

                # 通过tree_gate系统评估
                gate_rules = archetype.gate_rules
                tg_result = apply_gate_rules(
                    gate_rules=gate_rules,
                    features=features_set["features"],
                    quantiles=None,
                )

                # 比较结果
                sa_passed = sa_result[0]
                tg_passed = tg_result[0]

                consistent = sa_passed == tg_passed
                status = "✅" if consistent else "❌"

                print(f"  {status} {strategy.upper()}: SA={sa_passed}, TG={tg_passed}")

                if not consistent:
                    all_passed = False
                    print(f"    SA详情: {sa_result}")
                    print(f"    TG详情: {tg_result}")

            except Exception as e:
                print(f"  ⚠️  {strategy.upper()}: Error - {e}")
                all_passed = False

    return all_passed


def edge_case_test():
    """边缘案例测试"""
    print("\n=== 边缘案例测试 ===")

    # 测试复合条件
    complex_features = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 0.5}

    complex_conditions = [
        {
            "name": "all_of with value_le/ge",
            "condition": {
                "all_of": [
                    {"a": {"value_le": 1.5}},
                    {"b": {"value_ge": 1.5}},
                    {"c": {"value_gt": 2.5}},
                    {"d": {"value_lt": 1.0}},
                ]
            },
            "expected": True,
        },
        {
            "name": "nested conditions",
            "condition": {
                "all_of": [{"a": {"value_le": 0.5}}, {"b": {"value_ge": 1.5}}]  # False
            },
            "expected": False,
        },
    ]

    all_passed = True

    for case in complex_conditions:
        # 测试两个系统
        sa_result = tree_gate_eval_when(
            case["condition"], features=complex_features, quantiles=None
        )
        tg_result = tree_gate_eval_when(
            case["condition"], features=complex_features, quantiles=None
        )

        # 注意：这里两个都是用tree_gate_eval_when，因为复杂条件只在tree_gate中处理
        # 我们主要测试tree_gate自身的逻辑是否正确
        consistent = sa_result == tg_result == case["expected"]

        status = "✅" if consistent else "❌"
        print(
            f"{status} {case['name']}: Result={sa_result}, Expected={case['expected']}"
        )

        if not consistent:
            all_passed = False

    return all_passed


def main():
    print("综合Gate系统测试")
    print("=" * 50)

    # 运行所有测试
    operator_test_passed = comprehensive_operator_test()
    real_world_test_passed = real_world_gate_test()
    edge_case_test_passed = edge_case_test()

    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"操作符一致性测试: {'✅ 通过' if operator_test_passed else '❌ 失败'}")
    print(f"真实世界规则测试: {'✅ 通过' if real_world_test_passed else '❌ 失败'}")
    print(f"边缘案例测试: {'✅ 通过' if edge_case_test_passed else '❌ 失败'}")

    overall_pass = (
        operator_test_passed and real_world_test_passed and edge_case_test_passed
    )
    print(f"\n总体结果: {'✅ 全部通过' if overall_pass else '❌ 存在失败'}")

    if overall_pass:
        print("\n🎉 修复验证成功！Gate系统现在完全一致。")
        print("您现在可以放心运行回测，应该能看到BPC、FER、ME策略在事件回测中的交易数")
        print("与向量回测更加一致。")
    else:
        print("\n⚠️  修复存在问题，请检查代码。")

    return overall_pass


if __name__ == "__main__":
    main()
