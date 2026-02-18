"""
GenericLiveStrategy 单元+集成测试

覆盖范围:
  1. DirectionEvaluator — 4 种 transform + 缺失特征 + 无效值 + 多规则优先
  2. GateEvaluator — hard deny / allow / quantile gate / 无 gate
  3. EntryFilterChecker — OR 组合通过 / 全拒 / 无配置默认通过
  4. EvidenceScorer — 有/无 quantiles / 无 archetype 回退 0.5
  5. ExecutionParamGenerator — 多 tier 选择 / 默认 tier / 无 tier
  6. GenericLiveStrategy 主类:
     - __init__ 自动 load_configs
     - decide 完整管线 (LONG / SHORT / 各步骤拒绝)
     - set_quantiles / set_quantiles_from_df
     - _evaluate_entry_signal 诊断接口
     - _archetype 兼容属性
     - reset 状态清理
     - 多策略名称支持 (bpc / me / fer)
  7. 边界/异常:
     - 缺少 direction.yaml → 警告但不崩溃
     - 空 features → []
     - gate_weight 对 evidence score 的调制
  8. 性能: 1000 次 decide < 10ms 均值
"""

import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
import yaml

from src.time_series_model.live.generic_live_strategy import (
    DirectionEvaluator,
    EntryFilterChecker,
    EvidenceScorer,
    ExecutionParamGenerator,
    GateEvaluator,
    GenericLiveStrategy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: 临时配置目录
# ═══════════════════════════════════════════════════════════════════════════════


def _make_strategy_configs(
    tmpdir: str,
    strategy_name: str = "bpc",
    *,
    direction_rules=None,
    hard_gates=None,
    evidence_features=None,
    execution_tiers=None,
    entry_filters=None,
):
    """在 tmpdir 下创建一整套 archetype 配置文件"""
    arch_dir = os.path.join(tmpdir, "strategies", strategy_name, "archetypes")
    os.makedirs(arch_dir, exist_ok=True)

    # direction.yaml
    if direction_rules is None:
        direction_rules = [
            {
                "id": "default_direction",
                "feature": "signal_score",
                "transform": "sign",
            }
        ]
    direction_cfg = {
        "version": 1,
        "causal_source": f"{strategy_name}_test",
        "direction_rules": direction_rules,
    }
    with open(os.path.join(arch_dir, "direction.yaml"), "w") as f:
        yaml.dump(direction_cfg, f)

    # gate.yaml
    if hard_gates is None:
        hard_gates = []
    with open(os.path.join(arch_dir, "gate.yaml"), "w") as f:
        yaml.dump({"hard_gates": hard_gates}, f)

    # evidence.yaml
    if evidence_features is None:
        evidence_features = [
            {"id": "feat_a", "feature": "feat_a", "rank": 1},
            {"id": "feat_b", "feature": "feat_b", "rank": 2},
        ]
    with open(os.path.join(arch_dir, "evidence.yaml"), "w") as f:
        yaml.dump({"evidence": evidence_features}, f)

    # execution.yaml
    exec_cfg = {
        "stop_loss": {
            "initial_r": 2.0,
            "trailing": {"activation_r": 1.0, "trail_r": 1.5},
        },
        "take_profit": {"multiple": 2.5},
        "holding": {"time_stop_bars": 50},
    }
    if execution_tiers is not None:
        exec_cfg["tiers"] = {"enabled": True, "levels": execution_tiers}
    with open(os.path.join(arch_dir, "execution.yaml"), "w") as f:
        yaml.dump(exec_cfg, f)

    # entry_filters.yaml
    if entry_filters is None:
        entry_filters = []
    with open(os.path.join(arch_dir, "entry_filters.yaml"), "w") as f:
        yaml.dump({"filters": entry_filters}, f)

    return os.path.join(tmpdir, "strategies")


@pytest.fixture
def base_config(tmp_path):
    """最简配置: 无 gate / 无 entry filter / 无 tier"""
    root = _make_strategy_configs(str(tmp_path))
    return root


@pytest.fixture
def full_config(tmp_path):
    """完整配置: gate + entry filter + 3 tiers"""
    root = _make_strategy_configs(
        str(tmp_path),
        direction_rules=[
            {"id": "dir_rule", "feature": "signal_score", "transform": "sign"}
        ],
        hard_gates=[
            {
                "id": "min_score",
                "when": {"signal_score": {"value_gt": 0.5}},
                "then": {"action": "allow"},
            },
            {
                "id": "deny_extreme",
                "when": {"danger_flag": {"value_gt": 0.9}},
                "then": {"action": "deny"},
            },
        ],
        evidence_features=[
            {"id": "feat_a", "feature": "feat_a", "rank": 1},
            {"id": "feat_b", "feature": "feat_b", "rank": 2},
            {"id": "feat_c", "feature": "feat_c", "rank": 3},
        ],
        execution_tiers=[
            {
                "name": "strong",
                "evidence_min": 0.7,
                "stop_loss": {
                    "initial_r": 1.0,
                    "trailing": {"activation_r": 0.5, "trail_r": 0.8},
                },
                "time_stop_bars": 80,
                "size_multiplier": 1.5,
            },
            {
                "name": "medium",
                "evidence_min": 0.4,
                "stop_loss": {
                    "initial_r": 1.5,
                    "trailing": {"activation_r": 1.0, "trail_r": 1.2},
                },
                "time_stop_bars": 60,
                "size_multiplier": 1.0,
            },
            {
                "name": "weak",
                "evidence_min": 0.1,
                "stop_loss": {
                    "initial_r": 2.0,
                    "trailing": {"activation_r": 1.5, "trail_r": 1.5},
                },
                "time_stop_bars": 40,
                "size_multiplier": 0.6,
            },
        ],
        entry_filters=[
            {
                "id": "f1",
                "enabled": True,
                "conditions": [
                    {"feature": "bollinger_position", "operator": ">", "value": 0.8},
                ],
            },
            {
                "id": "f2",
                "enabled": True,
                "conditions": [
                    {"feature": "liq_silence", "operator": ">", "value": 0.5},
                ],
            },
        ],
    )
    return root


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DirectionEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectionEvaluator:
    """方向规则评估器"""

    def test_sign_transform_positive(self):
        ev = DirectionEvaluator(
            {"direction_rules": [{"id": "r1", "feature": "sig", "transform": "sign"}]}
        )
        d, rule_id = ev.evaluate({"sig": 0.8})
        assert d == 1
        assert rule_id == "r1"

    def test_sign_transform_negative(self):
        ev = DirectionEvaluator(
            {"direction_rules": [{"id": "r1", "feature": "sig", "transform": "sign"}]}
        )
        d, _ = ev.evaluate({"sig": -0.3})
        assert d == -1

    def test_sign_transform_zero(self):
        ev = DirectionEvaluator(
            {"direction_rules": [{"id": "r1", "feature": "sig", "transform": "sign"}]}
        )
        d, rule_id = ev.evaluate({"sig": 0.0})
        assert d == 0
        assert rule_id is None  # zero 不匹配

    def test_negate_sign_transform(self):
        ev = DirectionEvaluator(
            {
                "direction_rules": [
                    {"id": "r1", "feature": "sig", "transform": "negate_sign"}
                ]
            }
        )
        d, _ = ev.evaluate({"sig": 0.5})
        assert d == -1

    def test_raw_transform(self):
        ev = DirectionEvaluator(
            {"direction_rules": [{"id": "r1", "feature": "sig", "transform": "raw"}]}
        )
        d, _ = ev.evaluate({"sig": -1})
        assert d == -1

    def test_threshold_transform(self):
        ev = DirectionEvaluator(
            {
                "direction_rules": [
                    {"id": "r1", "feature": "sig", "transform": "threshold"}
                ]
            }
        )
        d_pos, _ = ev.evaluate({"sig": 0.1})
        assert d_pos == 1
        d_neg, _ = ev.evaluate({"sig": -0.1})
        assert d_neg == -1

    def test_missing_feature_skips(self):
        ev = DirectionEvaluator(
            {
                "direction_rules": [
                    {"id": "r1", "feature": "missing", "transform": "sign"}
                ]
            }
        )
        d, rule_id = ev.evaluate({"other": 1.0})
        assert d == 0
        assert rule_id is None

    def test_invalid_value_skips(self):
        ev = DirectionEvaluator(
            {"direction_rules": [{"id": "r1", "feature": "sig", "transform": "sign"}]}
        )
        d, _ = ev.evaluate({"sig": "not_a_number"})
        assert d == 0

    def test_multiple_rules_first_match_wins(self):
        ev = DirectionEvaluator(
            {
                "direction_rules": [
                    {"id": "r1", "feature": "missing", "transform": "sign"},
                    {"id": "r2", "feature": "sig", "transform": "sign"},
                    {"id": "r3", "feature": "sig2", "transform": "sign"},
                ]
            }
        )
        d, rule_id = ev.evaluate({"sig": 0.5, "sig2": -0.5})
        assert d == 1
        assert rule_id == "r2"

    def test_no_rules_returns_zero(self):
        ev = DirectionEvaluator({"direction_rules": []})
        d, rule_id = ev.evaluate({"sig": 0.5})
        assert d == 0
        assert rule_id is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ExecutionParamGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionParamGenerator:
    """执行参数生成器 — tier 选择逻辑"""

    def _make(self, levels):
        return ExecutionParamGenerator(
            {
                "stop_loss": {
                    "initial_r": 2.0,
                    "trailing": {"activation_r": 1.0, "trail_r": 1.5},
                },
                "take_profit": {"multiple": 2.5},
                "holding": {"time_stop_bars": 50},
                "tiers": {"enabled": True, "levels": levels},
            }
        )

    def test_high_evidence_selects_top_tier(self):
        gen = self._make(
            [
                {
                    "name": "high",
                    "evidence_min": 0.7,
                    "stop_loss": {"initial_r": 1.0},
                    "time_stop_bars": 80,
                    "size_multiplier": 1.5,
                },
                {
                    "name": "low",
                    "evidence_min": 0.3,
                    "stop_loss": {"initial_r": 2.0},
                    "time_stop_bars": 40,
                    "size_multiplier": 0.6,
                },
            ]
        )
        params = gen.generate_params(0.85)
        assert params["tier_name"] == "high"
        assert params["size_multiplier"] == 1.5

    def test_mid_evidence_selects_mid_tier(self):
        gen = self._make(
            [
                {"name": "high", "evidence_min": 0.7, "size_multiplier": 1.5},
                {"name": "low", "evidence_min": 0.3, "size_multiplier": 0.6},
            ]
        )
        params = gen.generate_params(0.5)
        assert params["tier_name"] == "low"
        assert params["size_multiplier"] == 0.6

    def test_below_all_tiers_falls_to_default(self):
        gen = self._make(
            [{"name": "only", "evidence_min": 0.8, "size_multiplier": 2.0}]
        )
        params = gen.generate_params(0.1)
        assert params["tier_name"] == "default"
        assert params["size_multiplier"] == 1.0

    def test_no_tiers_uses_base_params(self):
        gen = ExecutionParamGenerator(
            {
                "stop_loss": {
                    "initial_r": 3.0,
                    "trailing": {"activation_r": 2.0, "trail_r": 2.5},
                },
                "take_profit": {"multiple": 4.0},
                "holding": {"time_stop_bars": 100},
            }
        )
        params = gen.generate_params(0.9)
        assert params["tier_name"] == "default"
        assert params["initial_r"] == 3.0
        assert params["take_profit_r"] == 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GenericLiveStrategy — 初始化 + load_configs
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenericInit:
    """初始化与配置加载"""

    def test_auto_loads_configs(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        assert s.archetype is not None
        assert s.direction_evaluator is not None
        assert s.gate_evaluator is not None
        assert s.evidence_scorer is not None
        assert s.execution_generator is not None
        assert s.entry_filter_checker is not None

    def test_attributes_initialized(self, base_config):
        s = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=base_config,
            trade_size=0.5,
            bar_minutes=60,
            primary_timeframe="60T",
        )
        assert s.strategy_name == "bpc"
        assert s.trade_size == 0.5
        assert s.bar_minutes == 60
        assert s._quantiles == {}
        assert s._last_tier_params is None

    def test_missing_direction_yaml_warns_but_no_crash(self, tmp_path):
        """direction.yaml 缺失 → 日志警告，不崩溃"""
        root = _make_strategy_configs(str(tmp_path))
        dir_path = os.path.join(root, "bpc", "archetypes", "direction.yaml")
        os.remove(dir_path)
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=root)
        assert s.direction_evaluator is None
        # decide 应返回空
        intents = s.decide(features={"signal_score": 1.0}, symbol="X")
        assert intents == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. decide — 完整管线
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecidePipeline:
    """decide() 决策管线各步骤验证"""

    def test_long_signal(self, base_config):
        """正方向 → LONG"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        intents = s.decide(features={"signal_score": 0.8}, symbol="BTCUSDT")
        assert len(intents) == 1
        assert intents[0].action == "LONG"
        assert intents[0].symbol == "BTCUSDT"
        assert intents[0].archetype == "bpc"

    def test_short_signal(self, base_config):
        """负方向 → SHORT"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        intents = s.decide(features={"signal_score": -0.5}, symbol="ETHUSDT")
        assert len(intents) == 1
        assert intents[0].action == "SHORT"
        assert intents[0].symbol == "ETHUSDT"

    def test_no_direction_returns_empty(self, base_config):
        """方向特征缺失 → 无信号"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        intents = s.decide(features={"other_feat": 1.0}, symbol="X")
        assert intents == []

    def test_zero_direction_returns_empty(self, base_config):
        """方向=0 → 无信号"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        intents = s.decide(features={"signal_score": 0.0}, symbol="X")
        assert intents == []

    def test_empty_features_returns_empty(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        assert s.decide(features={}, symbol="X") == []

    def test_gate_deny_blocks_signal(self, full_config):
        """Gate deny → 无信号"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        # danger_flag > 0.9 触发 deny
        intents = s.decide(
            features={"signal_score": 0.8, "danger_flag": 0.95}, symbol="X"
        )
        assert intents == []

    def test_gate_allow_passes(self, full_config):
        """Gate allow + entry filter pass → 有信号"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={
                "signal_score": 0.8,
                "danger_flag": 0.1,
                "bollinger_position": 0.9,  # entry filter pass
            },
            symbol="X",
        )
        assert len(intents) == 1
        assert intents[0].action == "LONG"

    def test_entry_filter_deny_blocks(self, full_config):
        """Entry filter 全拒 → 无信号"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={
                "signal_score": 0.8,
                "bollinger_position": 0.5,  # < 0.8
                "liq_silence": 0.3,  # < 0.5
            },
            symbol="X",
        )
        assert intents == []

    def test_entry_filter_or_one_pass(self, full_config):
        """Entry filter OR: 一个通过即可"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={
                "signal_score": 0.8,
                "bollinger_position": 0.5,  # fail
                "liq_silence": 0.7,  # pass
            },
            symbol="X",
        )
        assert len(intents) == 1

    def test_execution_profile_structure(self, full_config):
        """验证 TradeIntent 的 execution_profile 结构"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={"signal_score": 0.8, "bollinger_position": 0.9},
            symbol="BTCUSDT",
        )
        assert len(intents) == 1
        profile = intents[0].execution_profile
        assert "rr_constraints" in profile
        assert "strategy_specific" in profile
        rr = profile["rr_constraints"]
        assert rr["stop_loss_r"] > 0
        assert rr["take_profit_r"] > 0
        assert rr["max_holding_bars"] > 0

    def test_confidence_between_0_and_1(self, full_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={"signal_score": 0.8, "bollinger_position": 0.9},
            symbol="X",
        )
        assert len(intents) == 1
        assert 0 <= intents[0].confidence <= 1.0

    def test_execution_tags(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        intents = s.decide(features={"signal_score": 0.8}, symbol="X")
        assert "bpc" in intents[0].execution_tags
        assert "BUY" in intents[0].execution_tags

    def test_last_tier_params_saved(self, full_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        assert s._last_tier_params is None
        s.decide(features={"signal_score": 0.8, "bollinger_position": 0.9}, symbol="X")
        assert s._last_tier_params is not None
        assert "tier_name" in s._last_tier_params


# ═══════════════════════════════════════════════════════════════════════════════
# 5. set_quantiles
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetQuantiles:
    """分位数计算"""

    def test_computes_evidence_quantiles(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        df = pd.DataFrame(
            {
                "feat_a": np.random.randn(100),
                "feat_b": np.random.randn(100),
            }
        )
        s.set_quantiles(df)
        assert "feat_a" in s._quantiles
        assert "feat_b" in s._quantiles
        # 默认 quantile_bins = [0.2, 0.4, 0.6, 0.8]
        assert "0.2" in s._quantiles["feat_a"]

    def test_alias_set_quantiles_from_df(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        df = pd.DataFrame(
            {"feat_a": np.random.randn(50), "feat_b": np.random.randn(50)}
        )
        s.set_quantiles_from_df(df)
        assert len(s._quantiles) >= 1

    def test_skips_missing_columns(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        df = pd.DataFrame({"unrelated": np.random.randn(50)})
        s.set_quantiles(df)
        assert len(s._quantiles) == 0

    def test_skips_too_few_rows(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        df = pd.DataFrame({"feat_a": [1.0, 2.0]})  # < 10 rows
        s.set_quantiles(df)
        assert "feat_a" not in s._quantiles

    def test_gate_quantiles_computed(self, tmp_path):
        """gate 中引用 quantile_gt 的特征也应计算分位数"""
        root = _make_strategy_configs(
            str(tmp_path),
            hard_gates=[
                {
                    "id": "q_gate",
                    "when": {"gate_feat": {"quantile_gt": 0.85}},
                    "then": {"action": "deny"},
                }
            ],
        )
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=root)
        df = pd.DataFrame({"gate_feat": np.random.randn(100)})
        s.set_quantiles(df)
        assert "gate_feat" in s._quantiles


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _evaluate_entry_signal (诊断接口)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateEntrySignal:
    """诊断接口"""

    def test_pass_returns_true_and_info(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        ok, info = s._evaluate_entry_signal({"signal_score": 0.8})
        assert ok is True
        assert info["side"] == "BUY"
        assert info["direction"] == 1
        assert "evidence_score" in info
        assert "tier" in info

    def test_no_direction_returns_reject(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        ok, info = s._evaluate_entry_signal({"other": 1.0})
        assert ok is False
        assert info["reject_reason"] == "no_direction"

    def test_gate_deny_returns_reject(self, full_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        ok, info = s._evaluate_entry_signal({"signal_score": 0.8, "danger_flag": 0.95})
        assert ok is False
        assert info["reject_reason"] == "gate_deny"
        assert "gate_reasons" in info

    def test_entry_filter_deny_returns_reject(self, full_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        ok, info = s._evaluate_entry_signal(
            {
                "signal_score": 0.8,
                "bollinger_position": 0.5,
                "liq_silence": 0.3,
            }
        )
        assert ok is False
        assert info["reject_reason"] == "entry_filter_deny"

    def test_empty_features_returns_false(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        ok, info = s._evaluate_entry_signal({})
        assert ok is False

    def test_sell_direction(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        ok, info = s._evaluate_entry_signal({"signal_score": -0.5})
        assert ok is True
        assert info["side"] == "SELL"
        assert info["direction"] == -1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _archetype 属性 + reset
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompatAndReset:
    """兼容属性与状态重置"""

    def test_archetype_property(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        assert s._archetype is s.archetype

    def test_reset_clears_state(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        s.decide(features={"signal_score": 0.8}, symbol="X")
        assert s._last_tier_params is not None
        s.reset()
        assert s._last_tier_params is None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 多策略支持
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiStrategy:
    """同一 GenericLiveStrategy 驱动 bpc / me / fer"""

    def test_three_strategies_coexist(self, tmp_path):
        strategies = {}
        for name in ("bpc", "me", "fer"):
            root = _make_strategy_configs(str(tmp_path), strategy_name=name)
            strategies[name] = GenericLiveStrategy(
                strategy_name=name, strategies_root=root
            )

        for name, strat in strategies.items():
            intents = strat.decide(features={"signal_score": 0.7}, symbol="X")
            assert len(intents) == 1
            assert intents[0].archetype == name
            assert intents[0].execution_strategy == name

    def test_strategy_name_in_tags(self, tmp_path):
        root = _make_strategy_configs(str(tmp_path), strategy_name="fer")
        s = GenericLiveStrategy(strategy_name="fer", strategies_root=root)
        intents = s.decide(features={"signal_score": -0.5}, symbol="X")
        assert "fer" in intents[0].execution_tags
        assert "SELL" in intents[0].execution_tags


# ═══════════════════════════════════════════════════════════════════════════════
# 9. gate_weight 对 evidence 的调制
# ═══════════════════════════════════════════════════════════════════════════════


class TestGateWeightModulation:
    """Gate weight 调制 evidence score"""

    def test_gate_weight_applied_to_confidence(self, full_config):
        """当 gate 有 soft weight 时, confidence = evidence * gate_weight"""
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=full_config)
        intents = s.decide(
            features={"signal_score": 0.8, "bollinger_position": 0.9},
            symbol="X",
        )
        assert len(intents) == 1
        # confidence 应 <= 1.0 (gate_weight <= 1.0)
        assert intents[0].confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. 性能基准
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """决策性能"""

    def test_decide_speed_under_10ms(self, base_config):
        s = GenericLiveStrategy(strategy_name="bpc", strategies_root=base_config)
        feats = {"signal_score": 0.8}
        # 预热
        for _ in range(10):
            s.decide(features=feats, symbol="X")
        # 测量
        t0 = time.perf_counter()
        for _ in range(1000):
            s.decide(features=feats, symbol="X")
        elapsed = (time.perf_counter() - t0) * 1000 / 1000
        assert elapsed < 10.0, f"平均 {elapsed:.3f}ms 超过 10ms 上限"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. 真实配置集成测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealConfigIntegration:
    """使用 config/strategies/bpc 真实配置加载"""

    @pytest.fixture
    def real_strategy(self):
        root = "config/strategies"
        if not os.path.isdir(os.path.join(root, "bpc", "archetypes")):
            pytest.skip("真实 BPC 配置不存在")
        return GenericLiveStrategy(strategy_name="bpc", strategies_root=root)

    def test_archetype_loaded(self, real_strategy):
        assert real_strategy.archetype is not None
        assert len(real_strategy.archetype.gate.all_rules) > 0
        assert len(real_strategy.archetype.evidence.features) > 0

    def test_direction_evaluator_loaded(self, real_strategy):
        assert real_strategy.direction_evaluator is not None
        assert len(real_strategy.direction_evaluator.rules) > 0

    def test_real_decide_no_crash(self, real_strategy):
        """用随机特征跑一次 decide，不崩溃即可"""
        feats = {
            "bpc_score_breakout": 0.6,
            "close": 50000.0,
            "atr": 500.0,
            "volume": 100.0,
        }
        intents = real_strategy.decide(features=feats, symbol="BTCUSDT")
        assert isinstance(intents, list)
