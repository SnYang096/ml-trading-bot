"""
测试 LivePCM — 多 archetype 信号仲裁层

覆盖场景:
1. 单策略透传（等价直挂 BPCLiveStrategy）
2. 多策略固定优先级（Reversal > ME > BPC）
3. 同优先级比 Evidence Score
4. slot 满时拒绝
5. 策略异常不影响其他策略
6. set_quantiles / load_all_configs 透传
"""

import pytest
from typing import Any, Dict, List, Optional

from time_series_model.core.trade_intent import TradeIntent
from time_series_model.portfolio.live_pcm import LivePCM, DEFAULT_ARCHETYPE_PRIORITY


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
    """单策略场景：行为等价直接挂 BPCLiveStrategy"""

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
    """多策略场景：固定优先级（Reversal > ME > BPC）"""

    def test_reversal_beats_me(self):
        """Reversal 优先于 ME（即使 ME evidence 更高）"""
        rev_intent = _make_intent("Reversal", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.9)

        pcm = LivePCM(max_slots=2)
        pcm.register("reversal", FakeStrategy(intents=[rev_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "Reversal"

    def test_me_beats_bpc(self):
        """ME 优先于 BPC（即使 BPC evidence 更高）"""
        me_intent = _make_intent("ME", confidence=0.6)
        bpc_intent = _make_intent("BPC", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("me", FakeStrategy(intents=[me_intent]))
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "ME"

    def test_reversal_beats_bpc(self):
        """Reversal 优先于 BPC"""
        rev_intent = _make_intent("Reversal", confidence=0.3)
        bpc_intent = _make_intent("BPC", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("reversal", FakeStrategy(intents=[rev_intent]))
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "Reversal"

    def test_all_three_reversal_wins(self):
        """三个 archetype 同时触发 → Reversal 胜出"""
        rev = _make_intent("Reversal", confidence=0.5)
        me = _make_intent("ME", confidence=0.9)
        bpc = _make_intent("BPC", confidence=1.0)

        pcm = LivePCM(max_slots=2)
        pcm.register("reversal", FakeStrategy(intents=[rev]))
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("bpc", FakeStrategy(intents=[bpc]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "Reversal"

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

        # ME priority > BPC priority，所以 ME 胜出（不看 evidence）
        assert len(result) == 1
        assert result[0].archetype == "ME"

    def test_custom_priority_order(self):
        """自定义优先级顺序"""
        bpc_intent = _make_intent("BPC", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.5)

        # BPC > ME 的自定义顺序
        pcm = LivePCM(archetype_priority=["BPC", "ME", "Reversal"], max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].archetype == "BPC"  # 自定义顺序中 BPC 优先

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
        assert DEFAULT_ARCHETYPE_PRIORITY == ["Reversal", "ME", "BPC"]
