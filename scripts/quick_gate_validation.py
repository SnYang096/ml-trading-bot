#!/usr/bin/env python3
"""
快速Gate修复验证脚本
专门验证BPC策略中value_le操作符的修复效果
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.archetype.loader import load_strategy_archetype
from src.time_series_model.live.tree_gate import apply_gate_rules


def test_bpc_gate_fix():
    """测试BPC策略的gate修复效果"""
    print("🔍 验证BPC策略Gate修复效果")
    print("-" * 40)

    # 加载BPC archetype
    bpc_archetype = load_strategy_archetype("bpc")

    # 检查BPC gate规则中使用的操作符
    print("📋 BPC Gate规则分析:")
    for i, rule in enumerate(bpc_archetype.gate.all_rules):
        print(f"  规则 {i+1}: {rule.tag}")
        print(f"    条件: {rule.when}")
        print(f"    动作: {rule.then}")

    print("\n🧪 测试特征值刚好满足阈值的情况:")

    # 创建刚好满足阈值的特征数据
    edge_case_features = {
        "bpc_bb_compression": 0.0965,  # 正好等于阈值 (value_le: 0.0965)
        "bpc_cvd_z": -2.0018,  # 正好等于阈值 (value_le: -2.0018)
        "bpc_volume_compression_pct": 0.1503,  # 正好等于阈值 (value_gt: 0.1503)
    }

    print(f"特征值: {edge_case_features}")

    # 测试StrategyArchetype评估
    sa_result = bpc_archetype.apply_gate(edge_case_features)
    print(f"StrategyArchetype结果: {sa_result}")

    # 测试TreeGate评估
    gate_rules = bpc_archetype.gate_rules
    tg_result = apply_gate_rules(
        gate_rules=gate_rules, features=edge_case_features, quantiles=None
    )
    print(f"TreeGate结果: {tg_result}")

    # 检查一致性
    sa_passed = sa_result[0]
    tg_passed = tg_result[0]

    print(f"\n✅ 一致性检查: {sa_passed == tg_passed}")

    if sa_passed == tg_passed:
        print("🎉 修复成功！两个系统现在对BPC策略的评估结果一致")
        if sa_passed:
            print("📈 BPC策略现在可以通过Gate，应该能在事件回测中产生交易")
        else:
            print("📊 BPC策略被拒绝是因为其他规则，不是因为value_le问题")
    else:
        print("❌ 修复失败！两个系统结果仍然不一致")

    return sa_passed == tg_passed


def test_original_problem():
    """测试原始问题是否解决"""
    print("\n🔍 验证原始问题是否解决")
    print("-" * 40)

    # 创建一个特征值刚好满足BPC条件的数据
    # 让bpc_bb_compression稍微大于阈值，这样就不会被value_le规则拒绝
    test_features = {
        "bpc_bb_compression": 0.1,  # 大于0.0965，不会被拒绝
        "bpc_cvd_z": -1.5,  # 大于-2.0018，不会被拒绝
        "bpc_volume_compression_pct": 0.2,  # 大于0.1503，不会被拒绝
        "atr_percentile": 0.9,  # 用于其他策略
        "me_atr_pct": 0.02,  # 用于ME策略
        "fer_trapped_longs_score": 2.0,  # 用于FER策略
    }

    print(f"测试特征: {test_features}")

    # 测试所有策略
    strategies = ["bpc", "fer", "me-long"]

    for strategy in strategies:
        archetype = load_strategy_archetype(strategy)

        sa_result = archetype.apply_gate(test_features)
        tg_result = apply_gate_rules(
            gate_rules=archetype.gate_rules, features=test_features, quantiles=None
        )

        sa_passed = sa_result[0]
        tg_passed = tg_result[0]

        consistent = sa_passed == tg_passed
        status = "✅" if consistent else "❌"

        print(
            f"{status} {strategy.upper()}: SA={sa_passed}, TG={tg_passed}, 一致={consistent}"
        )

        if not consistent:
            print(f"    SA详情: {sa_result}")
            print(f"    TG详情: {tg_result}")

    print("\n🎯 总结:")
    print("如果所有策略都显示✅，说明Gate系统修复成功")
    print("BPC、FER、ME策略现在在向量回测和事件回测中应该有一致的表现")


def main():
    print("🚀 Gate系统修复验证")
    print("=" * 50)

    bpc_test_passed = test_bpc_gate_fix()
    test_original_problem()

    print("\n" + "=" * 50)
    if bpc_test_passed:
        print("✅ Gate修复验证通过！")
        print("您现在可以运行事件回测，应该能看到BPC策略产生交易了。")
    else:
        print("❌ Gate修复存在问题，需要进一步检查。")

    return bpc_test_passed


if __name__ == "__main__":
    main()
