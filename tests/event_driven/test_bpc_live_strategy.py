"""
BPCLiveStrategy 单元测试

测试范围:
  1. select_tier() — 4 档 tier 映射 + 边界 + default fallback
  2. BPCLiveStrategy.__init__() — 属性初始化
  3. BPCLiveStrategy.load_configs() — 配置加载
  4. BPCLiveStrategy._evaluate_entry_signal() — 5 步决策管线
     - 方向判断 (bpc_breakout_direction)
     - Gate 检查 (hard deny + soft weight)
     - Entry Filter 检查 (OR 组合)
     - Evidence Score + Tier 选择
  5. BPCLiveStrategy.decide() — 产出 TradeIntent
  6. BPCLiveStrategy.reset() — 重置状态
  7. 端到端集成: 从真实配置加载 → 决策管线 → 输出验证
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.time_series_model.live.bpc_live_strategy import (
    BPCLiveStrategy,
    select_tier,
)
from src.time_series_model.core.meta_router_core import TradeIntent
from src.time_series_model.archetype.loader import (
    StrategyArchetype,
    load_strategy_archetype,
    GateConfig,
    GateRule,
    EvidenceConfig,
    EvidenceFeature,
    ExecutionConfig,
)
from src.time_series_model.execution.entry_filter import (
    DerivedEntryFeatureState,
    check_entry_filters_or_single,
    load_entry_filters_config,
)


# ================================================================
# Fixtures
# ================================================================

# 标准 execution.yaml tiers 配置（与真实 BPC 一致）
TIERS_CFG = {
    "enabled": True,
    "levels": [
        {
            "name": "强证据",
            "evidence_min": 0.70,
            "stop_loss": {
                "initial_r": 0.8,
                "trailing": {"activation_r": 0.5, "trail_r": 0.5},
            },
            "size_multiplier": 1.2,
            "time_stop_bars": 200,
        },
        {
            "name": "中等证据",
            "evidence_min": 0.50,
            "stop_loss": {
                "initial_r": 1.0,
                "trailing": {"activation_r": 1.0, "trail_r": 0.8},
            },
            "size_multiplier": 1.0,
            "time_stop_bars": 150,
        },
        {
            "name": "弱证据",
            "evidence_min": 0.30,
            "stop_loss": {
                "initial_r": 1.2,
                "trailing": {"activation_r": 1.5, "trail_r": 1.0},
            },
            "size_multiplier": 0.8,
            "time_stop_bars": 100,
        },
        {
            "name": "边缘证据",
            "evidence_min": 0.10,
            "stop_loss": {
                "initial_r": 1.5,
                "trailing": {"activation_r": 2.0, "trail_r": 1.5},
            },
            "size_multiplier": 0.5,
            "time_stop_bars": 80,
        },
    ],
}

EXEC_CONFIG = {
    "stop_loss": {
        "initial_r": 2.0,
        "trailing": {"activation_r": 1.0, "trail_r": 1.5},
    },
    "holding": {"time_stop_bars": 50},
}


# ================================================================
# Test: select_tier()
# ================================================================


class TestSelectTier:
    """select_tier() 辅助函数测试"""

    def test_tier_strong(self):
        """score=0.85 → 强证据"""
        t = select_tier(0.85, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "强证据"
        assert t["initial_r"] == 0.8
        assert t["activation_r"] == 0.5
        assert t["trail_r"] == 0.5
        assert t["size_multiplier"] == 1.2
        assert t["time_stop_bars"] == 200

    def test_tier_medium(self):
        """score=0.55 → 中等证据"""
        t = select_tier(0.55, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "中等证据"
        assert t["initial_r"] == 1.0
        assert t["size_multiplier"] == 1.0

    def test_tier_weak(self):
        """score=0.35 → 弱证据"""
        t = select_tier(0.35, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "弱证据"
        assert t["initial_r"] == 1.2
        assert t["size_multiplier"] == 0.8

    def test_tier_edge(self):
        """score=0.15 → 边缘证据"""
        t = select_tier(0.15, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "边缘证据"
        assert t["initial_r"] == 1.5
        assert t["size_multiplier"] == 0.5

    def test_tier_boundary_exact(self):
        """score 恰好在边界上 → 应匹配该 tier"""
        t = select_tier(0.70, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "强证据"

        t = select_tier(0.50, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "中等证据"

        t = select_tier(0.30, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "弱证据"

        t = select_tier(0.10, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "边缘证据"

    def test_tier_default_fallback(self):
        """score=0.05 → 低于所有 tier → fallback 到全局默认"""
        t = select_tier(0.05, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "default"
        assert t["initial_r"] == 2.0
        assert t["activation_r"] == 1.0
        assert t["trail_r"] == 1.5
        assert t["time_stop_bars"] == 50
        assert t["size_multiplier"] == 1.0

    def test_tier_zero_score(self):
        """score=0.0 → default"""
        t = select_tier(0.0, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "default"

    def test_tier_perfect_score(self):
        """score=1.0 → 强证据"""
        t = select_tier(1.0, TIERS_CFG, EXEC_CONFIG)
        assert t["tier_name"] == "强证据"

    def test_tier_empty_levels(self):
        """无 levels → default"""
        t = select_tier(0.85, {"levels": []}, EXEC_CONFIG)
        assert t["tier_name"] == "default"

    def test_tier_empty_tiers_cfg(self):
        """空 tiers_cfg → default"""
        t = select_tier(0.85, {}, EXEC_CONFIG)
        assert t["tier_name"] == "default"

    def test_tier_descending_order_independence(self):
        """levels 不需要预先排序（函数内部排序）"""
        reversed_cfg = {"levels": list(reversed(TIERS_CFG["levels"]))}
        t = select_tier(0.75, reversed_cfg, EXEC_CONFIG)
        assert t["tier_name"] == "强证据"


# ================================================================
# Test: BPCLiveStrategy 初始化
# ================================================================


class TestBPCLiveStrategyInit:
    """BPCLiveStrategy 初始化测试"""

    def test_default_init(self):
        """默认初始化参数正确"""
        s = BPCLiveStrategy()
        assert s._primary_timeframe == "240T"
        assert s._archetype is None
        assert s._entry_cfg == {}
        assert s._exec_config == {}
        assert s._tiers_cfg == {}
        assert s._holding_cfg == {}
        assert s._ef_state is None
        assert s._last_tier_params is None
        assert s._holding_yaml_path is None

    def test_custom_timeframe(self):
        """自定义 primary_timeframe"""
        s = BPCLiveStrategy(primary_timeframe="60T")
        assert s._primary_timeframe == "60T"

    def test_custom_holding_path(self):
        """自定义 holding_yaml_path"""
        s = BPCLiveStrategy(holding_yaml_path="/tmp/custom_holding.yaml")
        assert s._holding_yaml_path == "/tmp/custom_holding.yaml"

    def test_trade_size(self):
        """trade_size 参数"""
        s = BPCLiveStrategy(trade_size=0.5)
        assert s.trade_size == 0.5

    def test_bar_minutes(self):
        """bar_minutes 参数"""
        s = BPCLiveStrategy(bar_minutes=60)
        assert s._bar_minutes == 60


# ================================================================
# Test: BPCLiveStrategy._evaluate_entry_signal()
# ================================================================


class TestEvaluateEntrySignal:
    """_evaluate_entry_signal() 决策管线测试"""

    @pytest.fixture
    def strategy(self):
        """创建 mock strategy 实例"""
        s = BPCLiveStrategy()

        # 设置 mock archetype
        s._archetype = MagicMock(spec=StrategyArchetype)
        s._archetype.apply_gate.return_value = (True, [], 1.0)
        s._archetype.compute_evidence_score.return_value = (0.6, {"feat1": 0.7})

        # 设置 entry filter config (简单 OR 过滤)
        s._entry_cfg = {
            "filters": [
                {
                    "id": "test_filter",
                    "enabled": True,
                    "conditions": [
                        {
                            "feature": "bpc_was_in_pullback",
                            "operator": "==",
                            "value": 1,
                        },
                    ],
                }
            ]
        }

        # 设置 ef_state
        s._ef_state = DerivedEntryFeatureState()

        # 设置 tiers
        s._tiers_cfg = TIERS_CFG
        s._exec_config = EXEC_CONFIG

        return s

    def test_empty_features(self, strategy):
        """空特征 → 不入场"""
        should, info = strategy._evaluate_entry_signal({})
        assert should is False
        assert info == {}

    def test_no_direction(self, strategy):
        """bpc_breakout_direction=0 → 不入场"""
        should, info = strategy._evaluate_entry_signal({"bpc_breakout_direction": 0})
        assert should is False
        assert info.get("reject_reason") == "no_direction"

    def test_missing_direction(self, strategy):
        """缺少 bpc_breakout_direction → 不入场"""
        should, info = strategy._evaluate_entry_signal({"atr": 100, "vpin": 0.5})
        assert should is False
        assert info.get("reject_reason") == "no_direction"

    def test_buy_direction(self, strategy):
        """direction=+1 → BUY, 通过全管线"""
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "bb_width_normalized_pct": 0.3,
            "vol_percentile_approx": 0.1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True
        assert info["side"] == "BUY"
        assert info["direction"] == 1
        assert "tier" in info
        assert "evidence_score" in info

    def test_sell_direction(self, strategy):
        """direction=-1 → SELL"""
        features = {
            "bpc_breakout_direction": -1,
            "bpc_was_in_pullback": 1,
            "bb_width_normalized_pct": 0.3,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True
        assert info["side"] == "SELL"
        assert info["direction"] == -1

    def test_gate_hard_deny(self, strategy):
        """Gate hard deny → 不入场"""
        strategy._archetype.apply_gate.return_value = (
            False,
            ["HARD_WPT_IGNITION_FAIL"],
            0.0,
        )
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is False
        assert info["reject_reason"] == "gate_deny"
        assert "HARD_WPT_IGNITION_FAIL" in info["gate_reasons"]

    def test_gate_soft_weight(self, strategy):
        """Gate soft filter → evidence_score 被 gate_weight 降权"""
        strategy._archetype.apply_gate.return_value = (True, [], 0.7)
        strategy._archetype.compute_evidence_score.return_value = (0.8, {"feat1": 0.8})
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True
        # evidence_score = 0.8 × 0.7 = 0.56
        assert abs(info["evidence_score"] - 0.56) < 1e-6
        assert info["gate_weight"] == 0.7

    def test_entry_filter_deny(self, strategy):
        """Entry Filter 不通过 → 不入场"""
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 0,  # filter 要求 == 1
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is False
        assert info["reject_reason"] == "entry_filter_deny"

    def test_entry_filter_no_config(self, strategy):
        """无 entry filter 配置 → 全放行"""
        strategy._entry_cfg = {}
        features = {
            "bpc_breakout_direction": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True

    def test_evidence_score_affects_tier(self, strategy):
        """不同 evidence_score 选择不同 tier"""
        # 高 evidence → 强证据
        strategy._archetype.compute_evidence_score.return_value = (0.85, {})
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert info["tier"]["tier_name"] == "强证据"

        # 低 evidence → 边缘证据
        strategy._archetype.compute_evidence_score.return_value = (0.15, {})
        should, info = strategy._evaluate_entry_signal(features)
        assert info["tier"]["tier_name"] == "边缘证据"

    def test_no_archetype_defaults(self, strategy):
        """无 archetype → gate_weight=1.0, evidence=0.5"""
        strategy._archetype = None
        strategy._entry_cfg = {}
        features = {
            "bpc_breakout_direction": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True
        assert info["gate_weight"] == 1.0
        assert info["evidence_score"] == 0.5

    def test_signal_info_contains_bpc_features(self, strategy):
        """signal_info 包含 BPC 特征用于日志"""
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "bpc_score_breakout": 0.8,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert info["bpc_breakout_direction"] == 1
        assert info["bpc_score_breakout"] == 0.8
        assert info["bpc_was_in_pullback"] == 1

    def test_last_tier_params_saved(self, strategy):
        """_evaluate_entry_signal 后 _last_tier_params 被保存"""
        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "atr": 100.0,
        }
        strategy._evaluate_entry_signal(features)
        assert strategy._last_tier_params is not None
        assert "tier_name" in strategy._last_tier_params

    def test_direction_cast_from_float(self, strategy):
        """direction 为 float 1.0 时正确转为 int"""
        features = {
            "bpc_breakout_direction": 1.0,
            "bpc_was_in_pullback": 1,
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is True
        assert info["direction"] == 1

    def test_direction_invalid_type(self, strategy):
        """direction 为非数字 → 当作 0 处理"""
        features = {
            "bpc_breakout_direction": "invalid",
            "atr": 100.0,
        }
        should, info = strategy._evaluate_entry_signal(features)
        assert should is False
        assert info.get("reject_reason") == "no_direction"


# ================================================================
# Test: BPCLiveStrategy.decide() → TradeIntent
# ================================================================


class TestDecide:
    """decide() 输出 TradeIntent 测试"""

    @pytest.fixture
    def strategy(self):
        s = BPCLiveStrategy()
        s._archetype = MagicMock(spec=StrategyArchetype)
        s._archetype.apply_gate.return_value = (True, [], 1.0)
        s._archetype.compute_evidence_score.return_value = (0.6, {"feat1": 0.7})
        s._entry_cfg = {}  # 无 filter，全放行
        s._ef_state = DerivedEntryFeatureState()
        s._tiers_cfg = TIERS_CFG
        s._exec_config = EXEC_CONFIG
        s._holding_cfg = {"breakeven_lock": {"enabled": True, "trigger_r": 1.0}}
        return s

    def test_decide_returns_trade_intent(self, strategy):
        """正常信号 → 返回包含 1 个 TradeIntent 的列表"""
        features = {
            "bpc_breakout_direction": 1,
            "atr": 100.0,
        }
        intents = strategy.decide(features=features, symbol="BTCUSDT")
        assert len(intents) == 1
        intent = intents[0]
        assert isinstance(intent, TradeIntent)
        assert intent.action == "LONG"
        assert intent.symbol == "BTCUSDT"
        assert intent.archetype == "bpc"

    def test_decide_short(self, strategy):
        """direction=-1 → SHORT intent"""
        features = {
            "bpc_breakout_direction": -1,
            "atr": 100.0,
        }
        intents = strategy.decide(features=features, symbol="ETHUSDT")
        assert len(intents) == 1
        assert intents[0].action == "SHORT"
        assert intents[0].symbol == "ETHUSDT"

    def test_decide_no_signal(self, strategy):
        """direction=0 → 返回空列表"""
        features = {"bpc_breakout_direction": 0, "atr": 100.0}
        intents = strategy.decide(features=features, symbol="BTCUSDT")
        assert intents == []

    def test_decide_execution_profile(self, strategy):
        """TradeIntent 包含 execution_profile 和 bpc_position_config"""
        features = {"bpc_breakout_direction": 1, "atr": 100.0}
        intent = strategy.decide(features=features, symbol="BTCUSDT")[0]

        ep = intent.execution_profile
        assert "rr_constraints" in ep
        assert "bpc_position_config" in ep

        rr = ep["rr_constraints"]
        assert rr["stop_loss_r"] > 0
        assert rr["take_profit_r"] > 0
        assert rr["allow_trailing"] is True

        bpc_cfg = ep["bpc_position_config"]
        assert bpc_cfg["breakeven_enabled"] is True
        assert bpc_cfg["breakeven_trigger_r"] == 1.0
        assert bpc_cfg["bar_minutes"] == 240

    def test_decide_size_multiplier(self, strategy):
        """tier size_multiplier 传入 TradeIntent"""
        strategy._archetype.compute_evidence_score.return_value = (0.85, {})
        features = {"bpc_breakout_direction": 1, "atr": 100.0}
        intent = strategy.decide(features=features, symbol="BTCUSDT")[0]
        assert intent.size_multiplier == 1.2  # 强证据

    def test_decide_confidence(self, strategy):
        """confidence = adjusted evidence_score"""
        strategy._archetype.apply_gate.return_value = (True, [], 0.7)
        strategy._archetype.compute_evidence_score.return_value = (0.8, {})
        features = {"bpc_breakout_direction": 1, "atr": 100.0}
        intent = strategy.decide(features=features, symbol="BTCUSDT")[0]
        assert abs(intent.confidence - 0.56) < 1e-6  # 0.8 * 0.7

    def test_decide_execution_tags(self, strategy):
        """TradeIntent 包含 execution_tags"""
        features = {"bpc_breakout_direction": 1, "atr": 100.0}
        intent = strategy.decide(features=features, symbol="BTCUSDT")[0]
        assert "bpc" in intent.execution_tags
        assert "BUY" in intent.execution_tags


# ================================================================
# Test: BPCLiveStrategy.reset()
# ================================================================


class TestReset:
    """reset() 清理测试"""

    def test_reset_clears_state(self):
        s = BPCLiveStrategy()
        s._ef_state = DerivedEntryFeatureState()
        s._last_tier_params = {"tier_name": "强证据"}

        s._ef_state.update({"bpc_was_in_pullback": 1, "bb_width_normalized_pct": 0.5})

        s.reset()

        assert s._last_tier_params is None
        assert s._ef_state._consolidation_count == 0


# ================================================================
# Test: 端到端集成 — 使用真实配置文件
# ================================================================


class TestIntegrationWithRealConfig:
    """使用真实 BPC 配置文件的集成测试"""

    @pytest.fixture
    def real_archetype(self):
        """加载真实 BPC archetype"""
        try:
            return load_strategy_archetype("bpc")
        except Exception:
            pytest.skip("BPC archetype config not found")

    @pytest.fixture
    def real_entry_cfg(self):
        """加载真实 entry filter 配置"""
        cfg = load_entry_filters_config("bpc")
        if not cfg:
            pytest.skip("BPC entry_filters.yaml not found")
        return cfg

    def test_archetype_loads_correctly(self, real_archetype):
        """真实 archetype 加载: 规则数量、特征数量符合预期"""
        assert len(real_archetype.gate.all_rules) >= 10
        assert len(real_archetype.evidence.features) >= 8

    def test_gate_all_pass(self, real_archetype):
        """安全特征值 → Gate 全部通过"""
        safe_features = {
            "bpc_dir_consistency_long": 0.3,
            "wpt_ignition_score": 0.1,
            "wpt_exhaustion_score": 0.5,
            "sr_strength_max": 1.0,
            "vp_absorption_score": 0.02,
            "vpin_max20": 0.3,
            "cvd_change_5_normalized": 0.5,
            "bpc_bb_compression": 0.6,
            "sma_200_position": 0.01,
            "evt_scale_right": 0.5,
            "evt_var_99": 1.0,
            "evt_tail_shape_left": 0.6,
            "price_position": 0.5,
            "bpc_volume_compression_pct": 0.6,
        }
        passed, reasons, weight = real_archetype.apply_gate(safe_features)
        assert passed is True
        assert len(reasons) == 0

    def test_gate_hard_deny_direction_crowded(self, real_archetype):
        """bpc_dir_consistency_long > 0.55 → HARD deny"""
        features = {
            "bpc_dir_consistency_long": 0.8,
            "wpt_ignition_score": 0.1,
            "wpt_exhaustion_score": 0.5,
            "price_position": 0.5,
            "bpc_volume_compression_pct": 0.6,
        }
        passed, reasons, weight = real_archetype.apply_gate(features)
        assert passed is False
        assert "HARD_DIRECTION_NOT_COMMITTED" in reasons

    def test_gate_soft_filter_weight(self, real_archetype):
        """sr_strength_max > 3.204 → soft downweight 0.7"""
        features = {
            "bpc_dir_consistency_long": 0.3,
            "wpt_ignition_score": 0.1,
            "wpt_exhaustion_score": 0.5,
            "sr_strength_max": 5.0,
            "vp_absorption_score": 0.02,
            "vpin_max20": 0.3,
            "cvd_change_5_normalized": 0.5,
            "bpc_bb_compression": 0.6,
            "sma_200_position": 0.01,
            "evt_scale_right": 0.5,
            "evt_var_99": 1.0,
            "evt_tail_shape_left": 0.6,
            "price_position": 0.5,
            "bpc_volume_compression_pct": 0.6,
        }
        passed, reasons, weight = real_archetype.apply_gate(features)
        assert passed is True
        assert weight < 1.0

    def test_evidence_score_range(self, real_archetype):
        """Evidence score 应在 [0, 1] 范围内"""
        features = {
            "macd_signal_atr": 0.5,
            "vp_absorption_score": 0.03,
            "wpt_exhaustion_score": 0.6,
            "sma_200_position": 0.01,
            "vp_exhaustion_score": 0.5,
            "dist_to_nearest_sr": 0.3,
            "bpc_volume_compression_pct": 0.4,
            "evt_scale_right": 0.5,
            "spectrum_price_flatness": 0.5,
        }
        score, breakdown = real_archetype.compute_evidence_score(features)
        assert 0.0 <= score <= 1.0
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_entry_filter_or_logic(self, real_entry_cfg):
        """Entry Filter OR 组合: 任一通过即可"""
        enabled = [
            f for f in real_entry_cfg.get("filters", []) if f.get("enabled", False)
        ]
        assert len(enabled) >= 2, "应有至少 2 个 enabled filter (bb + liq)"

        features_bb = {
            "bpc_was_in_pullback": 1,
            "bpc_pullback_depth": 0.6,
            "bpc_bb_compression": 0.8,
        }
        check_entry_filters_or_single(features_bb, real_entry_cfg)

    def test_entry_filter_none_pass(self, real_entry_cfg):
        """所有条件都不满足 → 不入场"""
        features_bad = {
            "bpc_was_in_pullback": 0,
            "bpc_pullback_depth": 0.1,
            "bpc_bb_compression": 0.1,
            "vol_percentile_approx": 0.9,
        }
        result = check_entry_filters_or_single(features_bad, real_entry_cfg)
        assert result is False

    def test_full_pipeline_decide(self, real_archetype, real_entry_cfg):
        """端到端: 真实配置 → decide() → TradeIntent"""
        s = BPCLiveStrategy()
        s._archetype = real_archetype
        s._entry_cfg = real_entry_cfg
        s._ef_state = DerivedEntryFeatureState()
        s._tiers_cfg = TIERS_CFG
        s._exec_config = EXEC_CONFIG

        features = {
            "bpc_breakout_direction": 1,
            "bpc_was_in_pullback": 1,
            "bpc_pullback_depth": 0.65,
            "bpc_bb_compression": 0.8,
            "bpc_dir_consistency_long": 0.3,
            "wpt_ignition_score": 0.1,
            "wpt_exhaustion_score": 0.5,
            "sr_strength_max": 1.0,
            "vp_absorption_score": 0.02,
            "vpin_max20": 0.3,
            "cvd_change_5_normalized": 0.5,
            "sma_200_position": 0.01,
            "evt_scale_right": 0.5,
            "evt_var_99": 1.0,
            "evt_tail_shape_left": 0.6,
            "price_position": 0.5,
            "bpc_volume_compression_pct": 0.6,
            "macd_signal_atr": 0.5,
            "vp_exhaustion_score": 0.5,
            "dist_to_nearest_sr": 0.3,
            "spectrum_price_flatness": 0.5,
            "bb_width_normalized_pct": 0.3,
            "vol_percentile_approx": 0.1,
            "atr": 100.0,
            "bpc_score_breakout": 0.7,
        }

        intents = s.decide(features=features, symbol="BTCUSDT")
        assert len(intents) == 1
        intent = intents[0]
        assert intent.action == "LONG"
        assert intent.symbol == "BTCUSDT"
        assert intent.archetype == "bpc"
        assert 0.0 <= intent.confidence <= 1.0
        assert intent.execution_profile is not None
        assert "bpc_position_config" in intent.execution_profile


# ================================================================
# Test: DerivedEntryFeatureState 有状态计算
# ================================================================


class TestDerivedEntryFeatureState:
    """DerivedEntryFeatureState 有状态衍生特征测试"""

    def test_ef_vol_regime_shift_initial(self):
        """初始化阶段 ef_vol_regime_shift=0.0"""
        state = DerivedEntryFeatureState()
        result = state.update({"bb_width_normalized_pct": 0.5})
        assert result["ef_vol_regime_shift"] == 0.0

    def test_ef_vol_regime_shift_computed(self):
        """积累足够 bar 后 ef_vol_regime_shift 计算正确"""
        state = DerivedEntryFeatureState(diff_window=3)
        for v in [0.1, 0.2, 0.3, 0.4]:
            result = state.update({"bb_width_normalized_pct": v})
        assert abs(result["ef_vol_regime_shift"] - 0.3) < 1e-6

    def test_ef_liquidity_silence_vol_percentile(self):
        """vol_percentile_approx 优先"""
        state = DerivedEntryFeatureState()
        result = state.update({"vol_percentile_approx": 0.15, "bpc_vol_ratio": 0.8})
        assert result["ef_liquidity_silence"] == 0.15

    def test_ef_liquidity_silence_fallback_vol_ratio(self):
        """vol_percentile_approx 缺失 → fallback 到 bpc_vol_ratio"""
        state = DerivedEntryFeatureState()
        result = state.update({"bpc_vol_ratio": 0.3})
        assert result["ef_liquidity_silence"] == 0.3

    def test_ef_liquidity_silence_default(self):
        """都缺失 → 默认 0.5"""
        state = DerivedEntryFeatureState()
        result = state.update({})
        assert result["ef_liquidity_silence"] == 0.5

    def test_ef_consolidation_bars_accumulates(self):
        """连续 was_in_pullback=1 → consolidation_bars 递增"""
        state = DerivedEntryFeatureState(consolidation_cap=10)
        for i in range(5):
            result = state.update({"bpc_was_in_pullback": 1})
        assert result["ef_consolidation_bars"] == 0.5

    def test_ef_consolidation_bars_resets(self):
        """was_in_pullback=0 → consolidation 计数重置"""
        state = DerivedEntryFeatureState(consolidation_cap=10)
        state.update({"bpc_was_in_pullback": 1})
        state.update({"bpc_was_in_pullback": 1})
        result = state.update({"bpc_was_in_pullback": 0})
        assert result["ef_consolidation_bars"] == 0.0

    def test_ef_consolidation_bars_capped(self):
        """consolidation_bars 不超过 1.0"""
        state = DerivedEntryFeatureState(consolidation_cap=5)
        for _ in range(10):
            result = state.update({"bpc_was_in_pullback": 1})
        assert result["ef_consolidation_bars"] == 1.0

    def test_reset_clears_all(self):
        """reset() 清除所有状态"""
        state = DerivedEntryFeatureState()
        state.update({"bpc_was_in_pullback": 1, "bb_width_normalized_pct": 0.5})
        state.reset()
        assert state._consolidation_count == 0
        assert len(state._bb_width_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
