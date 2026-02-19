#!/usr/bin/env python3
"""
End-to-End Tests for Archetype Module

验证三层配置架构 (Gate / Evidence / Execution) 的完整流程
"""

import pandas as pd
import numpy as np
import pytest


class TestArchetypeLoading:
    """测试 Archetype 加载"""

    def test_load_bpc_strategy(self):
        """测试加载 BPC 策略"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        assert arch is not None
        assert arch.name == "bpc"
        assert arch.gate is not None
        assert arch.evidence is not None
        assert arch.execution is not None

    def test_gate_config_structure(self):
        """测试 Gate 配置结构"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        # 检查 hard gates
        assert len(arch.gate.hard_gates) >= 1
        for rule in arch.gate.hard_gates:
            assert rule.id is not None
            assert rule.tag is not None
            assert rule.phase in ("system_safety", "hard_gate")
            assert rule.is_hard is True

    def test_evidence_config_structure(self):
        """测试 Evidence 配置结构"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        assert len(arch.evidence.features) >= 1
        for feat in arch.evidence.features:
            assert feat.id is not None
            assert feat.feature is not None
            assert len(feat.quantile_bins) == 4  # 4 个边界定义 5 档
            assert len(feat.quantile_labels) == 5  # 5 档语义标签

    def test_execution_config_structure(self):
        """测试 Execution 配置结构"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        assert arch.execution.stop_loss_r > 0
        assert arch.execution.take_profit_r > 0
        assert arch.execution.direction_source in ("structure", "model", "hybrid")


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_gate_rules_property(self):
        """测试 gate_rules 属性 (向后兼容)"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        gate_rules = arch.gate_rules
        assert isinstance(gate_rules, dict)
        assert "when_then_rules" in gate_rules
        assert "default_action" in gate_rules
        assert isinstance(gate_rules["when_then_rules"], list)

    def test_when_then_rules_property(self):
        """测试 when_then_rules 属性 (向后兼容)"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        rules = arch.when_then_rules
        assert isinstance(rules, list)
        assert len(rules) >= 1

    def test_default_action_property(self):
        """测试 default_action 属性 (向后兼容)"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        assert arch.default_action in ("allow", "deny")

    def test_direction_policy_property(self):
        """测试 direction_policy 属性 (向后兼容)"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        policy = arch.direction_policy
        assert isinstance(policy, dict)
        assert "direction_source" in policy

    def test_execution_constraints_property(self):
        """测试 execution_constraints 属性 (向后兼容)"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        constraints = arch.execution_constraints
        assert isinstance(constraints, dict)


class TestGateApplication:
    """测试 Gate 规则应用"""

    def test_apply_gate_pass(self):
        """测试 Gate 应用 - 通过场景"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        # 模拟特征 - 应该通过
        features = {
            "vpin_percentile": 0.5,
            "direction_commitment_pct": 0.8,  # 高于阈值
            "volume_expansion_pct": 0.7,
            "price_zone_score": 0.5,
            "pullback_depth_pct": 0.4,  # 合理的回调深度
        }

        passed, reasons, weight = arch.apply_gate(features)

        # 应该通过 (假设没有触发规则)
        assert isinstance(passed, bool)
        assert isinstance(reasons, list)
        assert isinstance(weight, float)
        assert 0 <= weight <= 1.0

    def test_apply_gate_with_missing_features(self):
        """测试 Gate 应用 - 缺失特征场景"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        # 空特征
        features = {}

        passed, reasons, weight = arch.apply_gate(features)

        # 不应该崩溃
        assert isinstance(passed, bool)
        assert isinstance(reasons, list)
        assert isinstance(weight, float)


class TestEvidenceScoring:
    """测试 Evidence 评分"""

    def test_compute_evidence_score(self):
        """测试 Evidence 评分计算"""
        from src.time_series_model.archetype import load_strategy_archetype

        arch = load_strategy_archetype("bpc")

        # 模拟特征值
        features = {
            "sr_strength_max": 0.8,  # 高值
            "vpin_ma20": 0.5,
            "vol_slope_20": 0.3,
        }

        score, breakdown = arch.compute_evidence_score(features)

        assert isinstance(score, float)
        assert 0 <= score <= 1.0
        assert isinstance(breakdown, dict)

    def test_evidence_score_semantic_labels(self):
        """测试 Evidence 语义标签映射"""
        from src.time_series_model.archetype import EvidenceFeature

        # 创建测试特征 - 使用完整参数
        ef = EvidenceFeature(
            id="test_feature",
            feature="test_col",
            rank=1,
            quantile_bins=[0.2, 0.4, 0.6, 0.8],
            quantile_labels=["suppress", "downweight", "neutral", "favor", "amplify"],
            split_count=0,
            usage_hint="test",
            affects="position_size",
            threshold_examples={},
            distribution_hint="uniform",
        )

        # 测试不同分位数的标签
        assert ef.compute_label(0.1, None) == "suppress"
        assert ef.compute_label(0.3, None) == "downweight"
        assert ef.compute_label(0.5, None) == "neutral"
        assert ef.compute_label(0.7, None) == "favor"
        assert ef.compute_label(0.9, None) == "amplify"

        # 测试分数
        assert ef.compute_score(0.1, None) == 0.0
        assert ef.compute_score(0.5, None) == 0.5
        assert ef.compute_score(0.9, None) == 1.0


class TestLoadAllStrategies:
    """测试加载所有策略"""

    def test_load_all_strategy_archetypes(self):
        """测试加载所有策略"""
        from src.time_series_model.archetype import load_all_strategy_archetypes

        archetypes = load_all_strategy_archetypes()

        assert isinstance(archetypes, dict)
        # BPC 应该存在
        assert "bpc" in archetypes

        # 检查每个策略
        for name, arch in archetypes.items():
            assert arch.name == name
            assert arch.gate is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
