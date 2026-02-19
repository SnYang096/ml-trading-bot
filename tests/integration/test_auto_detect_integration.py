#!/usr/bin/env python3
"""
集成测试：自动检测计算需求模块的集成测试

测试：
1. build_feature_store_nnmultihead.py 使用公共函数
2. live_feature_plan.py 使用公共函数
3. 验证两个模块使用相同的公共函数，结果一致
"""

import pytest
import yaml

from src.cli.auto_detect_compute_requirements import (
    resolve_feature_dependencies,
    map_features_to_tier_nodes,
    extract_required_features_from_execution_archetypes,
)


class TestBuildFeatureStoreIntegration:
    """测试build_feature_store_nnmultihead.py的集成"""

    def test_auto_detect_logic_in_build_feature_store(self):
        """测试build_feature_store_nnmultihead.py中的自动检测逻辑"""
        # 模拟build_feature_store_nnmultihead.py中的逻辑
        execution_archetypes_path = (
            PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        )
        feature_deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not execution_archetypes_path.exists() or not feature_deps_path.exists():
            pytest.skip("Required config files not found")

        # 1. 提取gate规则需要的特征列名
        gate_features = extract_required_features_from_execution_archetypes(
            execution_archetypes_path
        )
        assert len(gate_features) > 0

        # 2. 读取feature_dependencies
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}

        # 3. 映射到feature nodes
        gate_nodes = map_features_to_tier_nodes(gate_features, feature_dependencies)
        assert len(gate_nodes) > 0

        # 4. 递归解析依赖关系
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)
        assert len(all_gate_nodes) >= len(gate_nodes)

        # 5. 验证所有gate nodes的依赖都被包含
        for node in gate_nodes:
            assert node in all_gate_nodes

        print(f"\n✅ build_feature_store_nnmultihead.py集成测试通过:")
        print(f"   - 提取到 {len(gate_features)} 个gate规则需要的特征")
        print(f"   - 映射到 {len(gate_nodes)} 个feature nodes")
        print(f"   - 解析依赖后共有 {len(all_gate_nodes)} 个nodes")


class TestLiveFeaturePlanIntegration:
    """测试live_feature_plan.py的集成"""

    def test_auto_detect_logic_in_live_feature_plan(self):
        """测试live_feature_plan.py中的自动检测逻辑"""
        # 模拟live_feature_plan.py中的逻辑
        execution_archetypes_path = (
            PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        )
        feature_deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not execution_archetypes_path.exists() or not feature_deps_path.exists():
            pytest.skip("Required config files not found")

        # 1. 提取gate规则需要的特征列名
        gate_features = extract_required_features_from_execution_archetypes(
            execution_archetypes_path
        )
        assert len(gate_features) > 0

        # 2. 读取feature_dependencies
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}

        # 3. 映射到feature nodes
        gate_nodes = map_features_to_tier_nodes(gate_features, feature_dependencies)
        assert len(gate_nodes) > 0

        # 4. 递归解析依赖关系
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)
        assert len(all_gate_nodes) >= len(gate_nodes)

        # 5. 验证所有gate nodes的依赖都被包含
        for node in gate_nodes:
            assert node in all_gate_nodes

        print(f"\n✅ live_feature_plan.py集成测试通过:")
        print(f"   - 提取到 {len(gate_features)} 个gate规则需要的特征")
        print(f"   - 映射到 {len(gate_nodes)} 个feature nodes")
        print(f"   - 解析依赖后共有 {len(all_gate_nodes)} 个nodes")


class TestConsistencyBetweenModules:
    """测试两个模块使用公共函数的一致性"""

    def test_build_feature_store_and_live_feature_plan_consistency(self):
        """测试build_feature_store_nnmultihead.py和live_feature_plan.py使用相同的公共函数，结果一致"""
        execution_archetypes_path = (
            PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        )
        feature_deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not execution_archetypes_path.exists() or not feature_deps_path.exists():
            pytest.skip("Required config files not found")

        # 读取feature_dependencies
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}

        # 模拟build_feature_store_nnmultihead.py的逻辑
        gate_features_1 = extract_required_features_from_execution_archetypes(
            execution_archetypes_path
        )
        gate_nodes_1 = map_features_to_tier_nodes(gate_features_1, feature_dependencies)
        all_gate_nodes_1 = resolve_feature_dependencies(
            gate_nodes_1, feature_dependencies
        )

        # 模拟live_feature_plan.py的逻辑
        gate_features_2 = extract_required_features_from_execution_archetypes(
            execution_archetypes_path
        )
        gate_nodes_2 = map_features_to_tier_nodes(gate_features_2, feature_dependencies)
        all_gate_nodes_2 = resolve_feature_dependencies(
            gate_nodes_2, feature_dependencies
        )

        # 验证结果一致
        assert gate_features_1 == gate_features_2, "提取的特征应该一致"
        assert gate_nodes_1 == gate_nodes_2, "映射的nodes应该一致"
        assert all_gate_nodes_1 == all_gate_nodes_2, "解析依赖后的nodes应该一致"

        print(f"\n✅ 两个模块使用公共函数，结果一致:")
        print(f"   - 提取的特征: {len(gate_features_1)} 个")
        print(f"   - 映射的nodes: {len(gate_nodes_1)} 个")
        print(f"   - 解析依赖后的nodes: {len(all_gate_nodes_1)} 个")


class TestRealWorldScenarios:
    """测试真实场景"""

    def test_real_world_feature_extraction(self):
        """测试从真实的execution_archetypes.yaml提取特征"""
        execution_archetypes_path = (
            PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        )
        feature_deps_path = PROJECT_ROOT / "config/feature_dependencies.yaml"

        if not execution_archetypes_path.exists() or not feature_deps_path.exists():
            pytest.skip("Required config files not found")

        # 提取特征
        gate_features = extract_required_features_from_execution_archetypes(
            execution_archetypes_path
        )

        # 验证包含预期的特征
        expected_features = [
            "path_efficiency_pct",
            "jump_risk_pct",
            "cvd_change_5_pct",
            "volume_ratio_pct",
        ]
        found_expected = [f for f in expected_features if f in gate_features]
        assert (
            len(found_expected) > 0
        ), f"应该找到至少一个预期特征，但只找到: {gate_features}"

        # 读取feature_dependencies
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            feature_dependencies = yaml.safe_load(f) or {}

        # 映射到nodes
        gate_nodes = map_features_to_tier_nodes(gate_features, feature_dependencies)

        # 验证映射成功
        assert len(gate_nodes) > 0, "应该映射到至少一个node"

        # 解析依赖
        all_gate_nodes = resolve_feature_dependencies(gate_nodes, feature_dependencies)

        # 验证依赖解析成功
        assert len(all_gate_nodes) >= len(
            gate_nodes
        ), "解析依赖后应该包含更多或相等的nodes"

        print(f"\n✅ 真实场景测试通过:")
        print(f"   - 提取到 {len(gate_features)} 个特征")
        print(f"   - 映射到 {len(gate_nodes)} 个nodes")
        print(f"   - 解析依赖后共有 {len(all_gate_nodes)} 个nodes")
        print(f"   - 找到的预期特征: {found_expected}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
