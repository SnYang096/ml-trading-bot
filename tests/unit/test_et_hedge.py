#!/usr/bin/env python3
"""
单元测试：ET对冲模块

测试：
1. ET风险评分计算
2. ET激活条件判断
3. ET仓位计算
4. ET风险指标计算
"""

import pytest
import numpy as np

from src.time_series_model.execution.et_hedge import (
    compute_et_risk_score,
    should_activate_et,
    compute_et_position,
    compute_et_risk_metrics,
)


class TestComputeETRiskScore:
    """测试ET风险评分计算"""

    def test_compute_et_risk_score_basic(self):
        """测试risk_score公式（0.4*ofci_p + 0.35*shd_p + 0.25*vol_spike_p）"""
        ofci_p = 0.8
        shd_p = 0.7
        vol_spike_p = 0.6

        risk_score = compute_et_risk_score(ofci_p, shd_p, vol_spike_p)

        expected = 0.4 * 0.8 + 0.35 * 0.7 + 0.25 * 0.6
        assert abs(risk_score - expected) < 1e-6
        assert 0.0 <= risk_score <= 1.0

    def test_compute_et_risk_score_all_high(self):
        """测试所有指标都很高的情况"""
        risk_score = compute_et_risk_score(1.0, 1.0, 1.0)

        assert risk_score == 1.0

    def test_compute_et_risk_score_all_low(self):
        """测试所有指标都很低的情况"""
        risk_score = compute_et_risk_score(0.0, 0.0, 0.0)

        assert risk_score == 0.0

    def test_compute_et_risk_score_clipping(self):
        """测试边界值裁剪"""
        # 测试超过1.0的情况（应该被裁剪到1.0）
        risk_score = compute_et_risk_score(1.5, 1.5, 1.5)
        assert risk_score == 1.0

        # 测试负数情况（应该被裁剪到0.0）
        risk_score = compute_et_risk_score(-0.5, -0.5, -0.5)
        assert risk_score == 0.0


class TestShouldActivateET:
    """测试ET激活条件判断"""

    def test_should_activate_et_no_exposure(self):
        """测试激活条件（无TC/TE暴露时返回False）"""
        should_activate, k = should_activate_et(
            tc_position=0.0,
            te_position=0.0,
            ofci_p=0.9,
            shd_p=0.9,
            vol_spike_p=0.9,
        )

        assert should_activate is False
        assert k == 0.0

    def test_should_activate_et_with_exposure(self):
        """测试有TC/TE暴露时应该激活"""
        should_activate, k = should_activate_et(
            tc_position=1.0,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        assert should_activate is True
        assert 0.0 <= k <= 0.8  # k_max = 0.8

    def test_should_activate_et_risk_score_scaling(self):
        """测试risk_score影响k值"""
        # 低风险评分
        _, k_low = should_activate_et(
            tc_position=1.0,
            te_position=0.0,
            ofci_p=0.1,
            shd_p=0.1,
            vol_spike_p=0.1,
        )

        # 高风险评分
        _, k_high = should_activate_et(
            tc_position=1.0,
            te_position=0.0,
            ofci_p=0.9,
            shd_p=0.9,
            vol_spike_p=0.9,
        )

        assert k_low < k_high
        assert k_high <= 0.8

    def test_should_activate_et_both_positions(self):
        """测试TC和TE都有仓位的情况"""
        should_activate, k = should_activate_et(
            tc_position=0.5,
            te_position=0.5,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        assert should_activate is True
        assert k > 0.0


class TestComputeETPosition:
    """测试ET仓位计算"""

    def test_compute_et_position_no_exposure(self):
        """测试无暴露时仓位为0"""
        et_position = compute_et_position(
            tc_position=0.0,
            te_position=0.0,
            ofci_p=0.9,
            shd_p=0.9,
            vol_spike_p=0.9,
        )

        assert et_position == 0.0

    def test_compute_et_position_long_exposure(self):
        """测试多头暴露时ET应该做空"""
        et_position = compute_et_position(
            tc_position=1.0,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # ET应该做空（负数）
        assert et_position < 0.0

    def test_compute_et_position_short_exposure(self):
        """测试空头暴露时ET应该做多"""
        et_position = compute_et_position(
            tc_position=-1.0,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # ET应该做多（正数）
        assert et_position > 0.0

    def test_compute_et_position_proportional(self):
        """测试仓位与暴露成正比"""
        # 小暴露
        et_pos_small = compute_et_position(
            tc_position=0.5,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 大暴露
        et_pos_large = compute_et_position(
            tc_position=2.0,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 大暴露的绝对值应该更大
        assert abs(et_pos_large) > abs(et_pos_small)

    def test_compute_et_position_opposite_directions(self):
        """测试TC和TE方向相反的情况"""
        # TC做多，TE做空，大小相等
        et_position = compute_et_position(
            tc_position=1.0,
            te_position=-1.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 如果方向相反且大小相等，应该不做对冲
        assert et_position == 0.0


class TestComputeETRiskMetrics:
    """测试ET风险指标计算"""

    def test_compute_et_risk_metrics_no_exposure(self):
        """测试无暴露时的风险指标"""
        metrics = compute_et_risk_metrics(
            tc_position=0.0,
            te_position=0.0,
            ofci_p=0.9,
            shd_p=0.9,
            vol_spike_p=0.9,
        )

        assert metrics["should_activate"] is False
        assert metrics["risk_score"] == 0.0
        assert metrics["k"] == 0.0
        assert metrics["directional_exposure"] == 0.0
        assert metrics["et_position"] == 0.0

    def test_compute_et_risk_metrics_with_exposure(self):
        """测试有暴露时的风险指标"""
        metrics = compute_et_risk_metrics(
            tc_position=1.0,
            te_position=0.5,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        assert metrics["should_activate"] is True
        assert 0.0 < metrics["risk_score"] <= 1.0
        assert 0.0 < metrics["k"] <= 0.8
        assert metrics["directional_exposure"] == 1.5
        assert metrics["et_position"] < 0.0  # 应该做空

    def test_compute_et_risk_metrics_contributions(self):
        """测试各指标的贡献度"""
        metrics = compute_et_risk_metrics(
            tc_position=1.0,
            te_position=0.0,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 检查贡献度
        assert metrics["ofci_contribution"] == 0.4 * 0.8
        assert metrics["shd_contribution"] == 0.35 * 0.7
        assert metrics["vol_spike_contribution"] == 0.25 * 0.6


class TestEdgeCases:
    """测试边界情况"""

    def test_extreme_risk_scores(self):
        """测试极端风险评分"""
        # 极端高风险
        risk_score = compute_et_risk_score(1.0, 1.0, 1.0)
        assert risk_score == 1.0

        # 极端低风险
        risk_score = compute_et_risk_score(0.0, 0.0, 0.0)
        assert risk_score == 0.0

    def test_negative_positions(self):
        """测试负仓位（空头）"""
        et_position = compute_et_position(
            tc_position=-1.0,
            te_position=-0.5,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 空头暴露时，ET应该做多（正数）
        assert et_position > 0.0

    def test_mixed_positions(self):
        """测试混合仓位（TC多，TE空）"""
        et_position = compute_et_position(
            tc_position=1.0,
            te_position=-0.5,
            ofci_p=0.8,
            shd_p=0.7,
            vol_spike_p=0.6,
        )

        # 净暴露为正，ET应该做空
        assert et_position < 0.0
