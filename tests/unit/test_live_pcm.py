"""
测试 LivePCM — 多 archetype 信号仲裁层 (Regime-Aware)

覆盖场景:
1. 单策略透传（等价直挂单策略）
2. 多策略静态优先级（无 regime detector）
3. Regime 动态优先级（NORMAL/HIGH_VOL/HIGH_LEVERAGE）
4. 同优先级比 Evidence Score
5. slot 满时拒绝
6. 策略异常不影响其他策略
7. set_quantiles / load_all_configs 透传
8. RegimeDetector 防抖 + 切换
9. LV override 逻辑
"""

import pytest
from typing import Any, Dict, List, Optional

from time_series_model.core.trade_intent import TradeIntent
from time_series_model.portfolio.live_pcm import (
    LivePCM,
    DEFAULT_ARCHETYPE_PRIORITY,
    RegimeDetector,
    REGIME_NORMAL,
    REGIME_HIGH_VOL,
    REGIME_HIGH_LEVERAGE,
    DEFAULT_REGIME_PRIORITIES,
)

# ── Fake Strategy ──


class FakeStrategy:
    """可配置的 mock 策略"""

    def __init__(self, intents: Optional[List[TradeIntent]] = None):
        self._intents = intents or []
        self.quantiles_set = False
        self.configs_loaded = False

    def decide(self, *, features, symbol, bars=None) -> List[TradeIntent]:
        return list(self._intents)

    def set_quantiles(self, features_df):
        self.quantiles_set = True

    def set_quantiles_from_df(self, features_df):
        self.quantiles_set = True

    def load_configs(self):
        self.configs_loaded = True


class ErrorStrategy:
    """总是抛异常的策略"""

    def decide(self, *, features, symbol, bars=None):
        raise RuntimeError("Strategy crashed!")


# ── Helper ──


def _make_intent(
    archetype: str, action: str = "LONG", confidence: float = 0.8
) -> TradeIntent:
    return TradeIntent(
        action=action,
        symbol="BTCUSDT",
        archetype=archetype,
        execution_strategy=archetype.lower(),
        confidence=confidence,
    )


FEATURES = {"close": 50000.0, "volume": 100.0}

# ── 测试 ──


class TestLivePCMSingleStrategy:
    """单策略场景：行为等价直挂单策略"""

    def test_single_strategy_passthrough(self):
        """单策略 → 透传 decide 结果"""
        intent = _make_intent("bpc")
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0] is intent

    def test_single_strategy_no_signal(self):
        """单策略无信号 → 返回空"""
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert result == []

    def test_no_registered_strategy(self):
        """无注册策略 → 返回空"""
        pcm = LivePCM(max_slots=2)
        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert result == []


class TestLivePCMPriority:
    """多策略场景：静态优先级（无 regime detector）"""

    def test_bpc_beats_me_default(self):
        """默认优先级：BPC 优先于 ME"""
        bpc_intent = _make_intent("BPC", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.9)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_me_beats_fer(self):
        """默认优先级：ME 优先于 FER"""
        me_intent = _make_intent("ME", confidence=0.6)
        fer_intent = _make_intent("FER", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("me", FakeStrategy(intents=[me_intent]))
        pcm.register("fer", FakeStrategy(intents=[fer_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "ME"

    def test_bpc_beats_lv_in_normal(self):
        """常态下 BPC 优先于 LV"""
        bpc_intent = _make_intent("BPC", confidence=0.3)
        lv_intent = _make_intent("LV", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("lv", FakeStrategy(intents=[lv_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_all_four_bpc_wins_default(self):
        """四个 archetype 同时触发 → BPC 胜出（默认优先级）"""
        bpc = _make_intent("BPC", confidence=0.5)
        me = _make_intent("ME", confidence=0.9)
        fer = _make_intent("FER", confidence=1.0)
        lv = _make_intent("LV", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_same_priority_compare_evidence(self):
        """同优先级（不太可能但测试兜底）→ 比 Evidence"""
        # 两个 BPC intent（理论上不会发生，但测排序逻辑正确性）
        bpc_high = _make_intent("BPC", confidence=0.9)
        bpc_low = _make_intent("BPC", confidence=0.3)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc1", FakeStrategy(intents=[bpc_low]))
        pcm.register("bpc2", FakeStrategy(intents=[bpc_high]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_one_strategy_fires_one_silent(self):
        """只有一个策略出信号 → 直接返回那个"""
        bpc_intent = _make_intent("BPC", confidence=0.8)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_confidence_none_defaults_to_half(self):
        """confidence=None → 默认 0.5"""
        bpc_intent = _make_intent("BPC", confidence=0.6)
        me_intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="ME",
            confidence=None,  # → 默认 0.5
        )

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        # BPC priority > ME priority 在默认优先级中，所以 BPC 胜出
        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_custom_priority_order(self):
        """自定义优先级顺序"""
        bpc_intent = _make_intent("BPC", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.5)

        # ME > BPC 的自定义顺序
        pcm = LivePCM(archetype_priority=["ME", "BPC", "FER", "LV"], max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "ME"  # 自定义顺序中 ME 优先

    def test_unknown_archetype_lowest_priority(self):
        """不在优先级列表中的 archetype → 排最后"""
        unknown = _make_intent("FooBar", confidence=1.0)
        bpc = _make_intent("BPC", confidence=0.1)

        pcm = LivePCM(max_slots=2)
        pcm.register("foobar", FakeStrategy(intents=[unknown]))
        pcm.register("bpc", FakeStrategy(intents=[bpc]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"  # BPC 在列表中，优先于未知


class TestLivePCMSlotControl:
    """Slot 控制场景"""

    def test_slot_full_rejects_single(self):
        pcm = LivePCM(max_slots=2, get_open_slot_count=lambda: 2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("bpc")]))
        assert pcm.decide(features=FEATURES, symbol="BTCUSDT") == []

    def test_slot_available_allows(self):
        pcm = LivePCM(max_slots=2, get_open_slot_count=lambda: 1)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("bpc")]))
        assert len(pcm.decide(features=FEATURES, symbol="BTCUSDT")) == 1

    def test_no_slot_callback_always_allows(self):
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("bpc")]))
        assert len(pcm.decide(features=FEATURES, symbol="BTCUSDT")) == 1

    def test_slot_full_rejects_multi_strategy_winner(self):
        pcm = LivePCM(max_slots=2, get_open_slot_count=lambda: 2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("BPC", confidence=0.9)]))
        pcm.register("me", FakeStrategy(intents=[_make_intent("ME", confidence=0.7)]))
        assert pcm.decide(features=FEATURES, symbol="BTCUSDT") == []


class TestLivePCMErrorHandling:
    """异常处理场景"""

    def test_strategy_error_does_not_crash(self):
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("BPC", confidence=0.8)]))
        pcm.register("me", ErrorStrategy())
        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "BPC"

    def test_all_strategies_error_returns_empty(self):
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", ErrorStrategy())
        pcm.register("me", ErrorStrategy())
        assert pcm.decide(features=FEATURES, symbol="BTCUSDT") == []


class TestLivePCMManagement:
    """管理接口场景"""

    def test_register_unregister(self):
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy())
        pcm.register("me", FakeStrategy())
        assert set(pcm.registered_archetypes) == {"bpc", "me"}
        pcm.unregister("bpc")
        assert pcm.registered_archetypes == ["me"]

    def test_set_quantiles_transparent(self):
        bpc, me = FakeStrategy(), FakeStrategy()
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", bpc)
        pcm.register("me", me)
        pcm.set_quantiles(None)
        assert bpc.quantiles_set and me.quantiles_set

    def test_set_quantiles_from_df_transparent(self):
        bpc = FakeStrategy()
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", bpc)
        pcm.set_quantiles_from_df(None)
        assert bpc.quantiles_set

    def test_load_all_configs(self):
        bpc, me = FakeStrategy(), FakeStrategy()
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", bpc)
        pcm.register("me", me)
        pcm.load_all_configs()
        assert bpc.configs_loaded and me.configs_loaded

    def test_archetype_priority_property(self):
        pcm = LivePCM(max_slots=2)
        assert pcm.archetype_priority == list(DEFAULT_ARCHETYPE_PRIORITY)

    def test_default_priority(self):
        assert DEFAULT_ARCHETYPE_PRIORITY == ["BPC", "ME", "FER", "LV"]


# ────────────────────────────────────────
# Regime Detector 测试
# ────────────────────────────────────────


class TestRegimeDetector:
    """RegimeDetector 状态机测试"""

    def test_default_is_normal(self):
        """默认 regime = NORMAL"""
        rd = RegimeDetector()
        assert rd.current_regime == REGIME_NORMAL
        assert rd.current_priority == ["BPC", "ME", "FER", "LV"]

    def test_detect_high_vol(self):
        """atr_percentile > 0.7 → HIGH_VOL"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({"atr_percentile": 0.8})
        assert rd.current_regime == REGIME_HIGH_VOL
        assert rd.current_priority == ["ME", "BPC", "FER", "LV"]

    def test_detect_high_leverage(self):
        """oi_zscore > 1.5 AND funding > 2.0 → HIGH_LEVERAGE"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({"oi_zscore": 2.0, "funding_rate_abs_zscore": 3.0})
        assert rd.current_regime == REGIME_HIGH_LEVERAGE
        assert rd.current_priority == ["LV", "FER", "ME", "BPC"]

    def test_high_leverage_beats_high_vol(self):
        """HIGH_LEVERAGE 优先于 HIGH_VOL（两者都满足时）"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect(
            {
                "atr_percentile": 0.9,
                "oi_zscore": 2.0,
                "funding_rate_abs_zscore": 3.0,
            }
        )
        assert rd.current_regime == REGIME_HIGH_LEVERAGE

    def test_debounce_prevents_instant_switch(self):
        """防抖机制：不会立即切换"""
        rd = RegimeDetector(min_bars_in_regime=3)
        # 第一次检测 HIGH_VOL，但防抖中
        result = rd.detect({"atr_percentile": 0.9})
        assert result == REGIME_NORMAL  # 还在 NORMAL 防抖期

    def test_debounce_allows_switch_after_min_bars(self):
        """超过 min_bars 后允许切换"""
        rd = RegimeDetector(min_bars_in_regime=2)
        # 先等待防抖期
        rd.detect({"atr_percentile": 0.3})  # NORMAL
        rd.detect({"atr_percentile": 0.3})  # NORMAL, bars_in=2
        # 现在可以切换
        result = rd.detect({"atr_percentile": 0.9})
        assert result == REGIME_HIGH_VOL
        assert rd.switch_count == 1

    def test_switch_count_tracks(self):
        """切换计数器"""
        rd = RegimeDetector(min_bars_in_regime=1)
        assert rd.switch_count == 0
        rd.detect({"atr_percentile": 0.9})
        assert rd.switch_count == 1
        rd.detect({"atr_percentile": 0.3})
        assert rd.switch_count == 2

    def test_reset(self):
        """重置状态"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({"atr_percentile": 0.9})
        assert rd.current_regime == REGIME_HIGH_VOL
        rd.reset()
        assert rd.current_regime == REGIME_NORMAL
        assert rd.switch_count == 0

    def test_missing_features_stay_normal(self):
        """缺少特征 → 保持 NORMAL"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({})  # 无特征
        assert rd.current_regime == REGIME_NORMAL

    def test_partial_high_leverage_not_triggered(self):
        """HIGH_LEVERAGE 需要 AND，只满足一个不触发"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({"oi_zscore": 2.0})  # 只有 oi, 缺 funding
        assert rd.current_regime == REGIME_NORMAL


# ────────────────────────────────────────
# LivePCM + Regime 集成测试
# ────────────────────────────────────────


class TestLivePCMWithRegime:
    """LivePCM 集成 RegimeDetector 的场景"""

    def test_no_regime_detector_uses_static_priority(self):
        """无 regime detector → 使用静态优先级"""
        pcm = LivePCM(archetype_priority=["FER", "ME", "BPC"], max_slots=2)
        assert pcm.current_regime == REGIME_NORMAL
        assert pcm.archetype_priority == ["FER", "ME", "BPC"]

    def test_with_regime_detector_high_vol_me_wins(self):
        """HIGH_VOL regime → ME 优先于 BPC"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        bpc = _make_intent("BPC", confidence=0.9)
        me = _make_intent("ME", confidence=0.5)

        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        # HIGH_VOL features → ME 优先
        result = pcm.decide(
            features={**FEATURES, "atr_percentile": 0.9},
            symbol="BTCUSDT",
        )
        assert len(result) == 1
        assert result[0].archetype == "ME"
        assert pcm.current_regime == REGIME_HIGH_VOL

    def test_with_regime_detector_high_leverage_lv_wins(self):
        """HIGH_LEVERAGE regime → LV 优先于所有"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        bpc = _make_intent("BPC", confidence=1.0)
        lv = _make_intent("LV", confidence=0.3)

        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        # HIGH_LEVERAGE features → LV override
        result = pcm.decide(
            features={
                **FEATURES,
                "oi_zscore": 2.0,
                "funding_rate_abs_zscore": 3.0,
            },
            symbol="BTCUSDT",
        )
        assert len(result) == 1
        assert result[0].archetype == "LV"
        assert pcm.current_regime == REGIME_HIGH_LEVERAGE

    def test_regime_switch_changes_winner(self):
        """regime 切换后优胜者变化"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        bpc = _make_intent("BPC", confidence=0.8)
        me = _make_intent("ME", confidence=0.8)

        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        # NORMAL: BPC wins
        r1 = pcm.decide(features={**FEATURES, "atr_percentile": 0.3}, symbol="BTCUSDT")
        assert r1[0].archetype == "BPC"

        # HIGH_VOL: ME wins
        r2 = pcm.decide(features={**FEATURES, "atr_percentile": 0.9}, symbol="BTCUSDT")
        assert r2[0].archetype == "ME"

    def test_get_stats_with_regime(self):
        """get_stats() 返回 regime 信息"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)
        pcm.register("bpc", FakeStrategy())

        pcm.decide(features={**FEATURES, "atr_percentile": 0.9}, symbol="BTCUSDT")

        stats = pcm.get_stats()
        assert stats["current_regime"] == REGIME_HIGH_VOL
        assert stats["regime_switch_count"] == 1
        assert "bpc" in stats["registered_archetypes"]


# ────────────────────────────────────────
# Layer 3: Override 极端信号覆盖 测试
# ────────────────────────────────────────

# ── Override 配置 fixtures ──

OVERRIDE_CONFIG = {
    "LV": {
        "overrides": "ALL",
        "min_evidence": 0.0,
        "conditions": [],
    },
    "FER": {
        "overrides": ["ME"],
        "min_evidence": 0.6,
        "conditions": [],
    },
    "ME": {
        "overrides": ["BPC"],
        "min_evidence": 0.7,
        "conditions": [
            {"feature": "atr_percentile", "operator": ">", "threshold": 0.75},
        ],
        "logic": "AND",
    },
}


class TestLivePCMOverride:
    """Layer 3: Override 极端信号覆盖 场景"""

    def test_lv_overrides_all_in_normal_regime(self):
        """LV 覆盖所有：即使 regime=NORMAL，LV 信号触发即赢"""
        bpc = _make_intent("BPC", confidence=0.9)
        lv = _make_intent("LV", confidence=0.3)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "LV"

    def test_lv_overrides_all_four_archetypes(self):
        """LV 覆盖所有：四个 archetype 同时触发，LV 胜出"""
        bpc = _make_intent("BPC", confidence=1.0)
        me = _make_intent("ME", confidence=1.0)
        fer = _make_intent("FER", confidence=1.0)
        lv = _make_intent("LV", confidence=0.1)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "LV"

    def test_me_overrides_bpc_with_high_evidence_and_atr(self):
        """ME 覆盖 BPC：evidence >= 0.7 且 atr_percentile > 0.75"""
        bpc = _make_intent("BPC", confidence=0.9)
        me = _make_intent("ME", confidence=0.8)  # >= 0.7

        features = {**FEATURES, "atr_percentile": 0.85}  # > 0.75

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        result = pcm.decide(features=features, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "ME"

    def test_me_no_override_low_evidence(self):
        """ME 不覆盖：evidence 不足 0.7 → 回退到普通优先级"""
        bpc = _make_intent("BPC", confidence=0.5)
        me = _make_intent("ME", confidence=0.5)  # < 0.7

        features = {**FEATURES, "atr_percentile": 0.85}

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        result = pcm.decide(features=features, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "BPC"  # 普通优先级 BPC > ME

    def test_me_no_override_low_atr(self):
        """ME 不覆盖：atr_percentile 不足 0.75 → 回退到普通优先级"""
        bpc = _make_intent("BPC", confidence=0.5)
        me = _make_intent("ME", confidence=0.8)

        features = {**FEATURES, "atr_percentile": 0.5}  # < 0.75

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        result = pcm.decide(features=features, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "BPC"  # 普通优先级 BPC > ME

    def test_fer_overrides_me(self):
        """FER 覆盖 ME：反转信号覆盖动能扩张"""
        me = _make_intent("ME", confidence=0.9)
        fer = _make_intent("FER", confidence=0.7)  # >= 0.6

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "FER"

    def test_fer_no_override_low_evidence(self):
        """FER 不覆盖：evidence 不足 0.6 → 回退到普通优先级"""
        me = _make_intent("ME", confidence=0.5)
        fer = _make_intent("FER", confidence=0.4)  # < 0.6

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "ME"  # 普通优先级 ME > FER

    def test_override_not_active_without_config(self):
        """无 override 配置 → 不触发覆盖，维持原始优先级"""
        bpc = _make_intent("BPC", confidence=0.3)
        lv = _make_intent("LV", confidence=1.0)

        pcm = LivePCM(max_slots=2)  # 无 override_config
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "BPC"  # 默认优先级 BPC > LV

    def test_lv_override_respects_slot_limit(self):
        """LV override 仍然受 slot 限制"""
        bpc = _make_intent("BPC", confidence=0.9)
        lv = _make_intent("LV", confidence=0.5)

        pcm = LivePCM(
            max_slots=2,
            get_open_slot_count=lambda: 2,  # slot 已满
            override_config=OVERRIDE_CONFIG,
        )
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert result == []  # slot 满，拒绝

    def test_lv_override_priority_over_fer_override(self):
        """LV 抢占权 > FER：LV 和 FER 同时触发 override，LV 胜"""
        me = _make_intent("ME", confidence=0.8)
        fer = _make_intent("FER", confidence=0.9)
        lv = _make_intent("LV", confidence=0.2)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "LV"  # LV 抢占权最高

    def test_fer_override_only_targets_me(self):
        """FER 只覆盖 ME，不覆盖 BPC"""
        bpc = _make_intent("BPC", confidence=0.5)
        fer = _make_intent("FER", confidence=0.9)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("fer", FakeStrategy(intents=[fer]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "BPC"  # FER 不覆盖 BPC，普通优先级 BPC > FER

    def test_me_override_only_targets_bpc(self):
        """ME 只覆盖 BPC，不覆盖 FER"""
        fer = _make_intent("FER", confidence=0.5)
        me = _make_intent("ME", confidence=0.8)

        features = {**FEATURES, "atr_percentile": 0.85}

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("me", FakeStrategy(intents=[me]))

        result = pcm.decide(features=features, symbol="BTCUSDT")
        assert len(result) == 1
        assert result[0].archetype == "ME"  # ME 普通优先级 > FER

    def test_override_with_regime_detector(self):
        """Override + Regime 共存：Override (Layer 3) 优先于 Regime (Layer 2)"""
        rd = RegimeDetector(min_bars_in_regime=1)
        bpc = _make_intent("BPC", confidence=0.9)
        lv = _make_intent("LV", confidence=0.3)

        # NORMAL regime 下 BPC 优先，但 LV override 应该胜出
        pcm = LivePCM(
            regime_detector=rd,
            max_slots=2,
            override_config=OVERRIDE_CONFIG,
        )
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(
            features={**FEATURES, "atr_percentile": 0.3},  # NORMAL regime
            symbol="BTCUSDT",
        )
        assert len(result) == 1
        assert result[0].archetype == "LV"  # Override 优先于 regime
        assert pcm.current_regime == REGIME_NORMAL

    def test_get_stats_includes_override_info(self):
        """get_stats() 包含 override 信息"""
        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        stats = pcm.get_stats()
        assert stats["override_enabled"] is True
        assert set(stats["override_rules"]) == {"LV", "FER", "ME"}

    def test_get_stats_no_override(self):
        """get_stats() 无 override 时的输出"""
        pcm = LivePCM(max_slots=2)
        stats = pcm.get_stats()
        assert stats["override_enabled"] is False
        assert stats["override_rules"] == []
