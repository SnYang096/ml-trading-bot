#!/usr/bin/env python3
"""
测试6种archetype的gate规则是否正确加载和应用
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules


def test_archetype_loading():
    """测试archetype配置加载"""
    print("=" * 60)
    print("测试1: Archetype配置加载")
    print("=" * 60)

    arches = load_execution_archetypes_registry(
        "config/nnmultihead/execution_archetypes.yaml"
    )

    expected = [
        "BreakoutPullbackContinuation",
        "HTFBiasLTFEntry",
        "MomentumExpansion",
        "FailedBreakoutFade",
        "LiquiditySweepRejection",
        "AuctionExhaustionReversal",
    ]

    print(f"\n✅ 成功加载 {len(arches)} 个archetype")
    for name in expected:
        if name in arches:
            arch = arches[name]
            print(f"  ✅ {name}:")
            print(f"     - 规则数量: {len(arch.when_then_rules)}")
            print(f"     - default_action: {arch.default_action}")
            print(f"     - 执行约束: {arch.execution_constraints}")

            # 统计各phase的规则数量
            phase_counts = {}
            for rule in arch.when_then_rules:
                phase = rule.get("phase", "unknown")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            print(f"     - 规则分布: {phase_counts}")
        else:
            print(f"  ❌ {name}: 未找到")

    return arches


def test_gate_rules_structure(arches):
    """测试gate规则结构"""
    print("\n" + "=" * 60)
    print("测试2: Gate规则结构验证")
    print("=" * 60)

    for name, arch in arches.items():
        print(f"\n📋 {name}:")

        # 检查when_then_rules结构
        if not arch.when_then_rules:
            print(f"  ⚠️  没有when_then_rules")
            continue

        # 验证每个规则的必要字段
        for i, rule in enumerate(arch.when_then_rules):
            rule_id = rule.get("id", f"rule_{i}")
            phase = rule.get("phase", "unknown")
            reason = rule.get("reason", "")
            when = rule.get("when", {})
            then = rule.get("then", {})
            action = then.get("action", "unknown")

            issues = []
            if not rule_id:
                issues.append("缺少id")
            if not phase:
                issues.append("缺少phase")
            if not when:
                issues.append("缺少when")
            if not then or not action:
                issues.append("缺少then.action")

            if issues:
                print(f"  ⚠️  规则 {rule_id}: {', '.join(issues)}")
            else:
                print(
                    f"  ✅ 规则 {rule_id}: phase={phase}, action={action}, reason={reason[:50]}..."
                )


def test_gate_application_sample(arches):
    """测试gate规则应用（使用示例数据）"""
    print("\n" + "=" * 60)
    print("测试3: Gate规则应用（示例数据）")
    print("=" * 60)

    # 创建一个示例feature字典（所有特征都在合理范围内）
    sample_features = {
        "path_efficiency_pct": 0.7,  # 高
        "price_dir_consistency_pct": 0.7,  # 高
        "vpin": 0.6,  # 中等
        "vp_poc_deviation": 0.3,  # 中等
        "cvd_change_5_pct": 0.5,  # 中等
        "sr_distance_normalized": 0.4,  # 中等
        "jump_risk_pct": 0.5,  # 中等
        "atr_percentile": 0.6,  # 中等
        "bb_width_normalized_pct": 0.7,  # 高
        "volume_ratio_pct": 0.6,  # 中等
        "shd_pct": 0.5,  # 中等（低于deny阈值）
        "ofci_pct": 0.5,  # 中等（低于deny阈值）
        "path_length_pct": 0.7,  # 高
    }

    # 空的quantiles（不使用quantile规则）
    quantiles = {}

    print("\n使用示例特征测试每个archetype:")
    for name, arch in sorted(arches.items()):
        gate_cfg = {
            "when_then_rules": arch.when_then_rules,
            "default_action": arch.default_action,
        }

        ok, reasons = apply_gate_rules(
            gate_rules=gate_cfg,
            features=sample_features,
            quantiles=quantiles,
        )

        status = "✅ PASS" if ok else "❌ VETO"
        print(f"  {status} {name}: {reasons if reasons else '无原因'}")


def main():
    print("🧪 测试6种Archetype Gate规则")
    print("=" * 60)

    # 测试1: 加载配置
    arches = test_archetype_loading()

    # 测试2: 验证结构
    test_gate_rules_structure(arches)

    # 测试3: 应用测试
    test_gate_application_sample(arches)

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
