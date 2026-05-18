"""
测试 LivePCM — 多 archetype 信号仲裁层 (Regime-Aware)

覆盖场景:
1. 单策略透传（等价直挂单策略）
2. 多策略静态优先级（无 regime detector）
3. Regime 动态优先级（NORMAL/HIGH_VOL/HIGH_LEVERAGE）
4. 同优先级比 Evidence Score
5. slot 满时拒绝
6. 策略异常不影响其他策略
7. load_all_configs 透传
8. RegimeDetector 防抖 + 切换
9. LV override 逻辑
"""

import json
from pathlib import Path

import pytest
import pandas as pd
from typing import Any, Dict, List, Optional

from time_series_model.core.trade_intent import TradeIntent
from time_series_model.portfolio.live_pcm import (
    LivePCM,
    _calendar_day_utc_str,
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
        self.configs_loaded = False

    def decide(self, *, features, symbol, bars=None) -> List[TradeIntent]:
        return list(self._intents)

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


class TestLivePCMBacktestParity:
    """事件回测 / 实盘一致性行为（日历键、诊断 trace）"""

    def test_calendar_day_utc_str_from_pandas_timestamp(self):
        ts = pd.Timestamp("2024-11-07 08:00:00+00:00")
        assert (
            _calendar_day_utc_str(features={"timestamp": ts}, decision_time=None)
            == "2024-11-07"
        )
        assert (
            _calendar_day_utc_str(features={"close": 1.0}, decision_time=ts)
            == "2024-11-07"
        )

    def test_pcm_daily_throttle_uses_decision_time_not_wall_clock(self):
        """同一仿真日超过 max_new_entries_per_day 后拒单；换仿真日重置。"""
        pcm = LivePCM(max_slots=10)
        pcm._daily_entry_limits["me"] = 2
        pcm.register("me", FakeStrategy(intents=[_make_intent("ME")]))
        base_feat = {"close": 50000.0, "ema_200": 40000.0, "volume": 1.0}
        day_a = pd.Timestamp("2024-11-07 00:00:00+00:00")
        day_b = pd.Timestamp("2024-11-08 00:00:00+00:00")
        for _ in range(2):
            out = pcm.decide(
                features={**base_feat, "timestamp": day_a},
                symbol="BTCUSDT",
                decision_time=day_a,
            )
            assert len(out) == 1
            # 与 intent.archetype 大小写一致，否则 slot 未清、会走 add 意图并绕过日限统计
            pcm.notify_position_closed("BTCUSDT", "ME")
        out3 = pcm.decide(
            features={**base_feat, "timestamp": day_a},
            symbol="BTCUSDT",
            decision_time=day_a,
        )
        assert out3 == []
        assert int(pcm._last_decide_trace.get("drop_daily_limit", 0) or 0) >= 1
        pcm.notify_position_closed("BTCUSDT", "ME")
        out4 = pcm.decide(
            features={**base_feat, "timestamp": day_b},
            symbol="BTCUSDT",
            decision_time=day_b,
        )
        assert len(out4) == 1

    def test_hydrate_slot_evidence_from_constitution_slots_restart_recovery(self):
        """Persisted constitution slots refill PCM memory across fake 'restart'."""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
            SlotRecord,
        )

        pcm = LivePCM(max_slots=10)
        pcm.register("me", FakeStrategy(intents=[_make_intent("me")]))
        st = ConstitutionRuntimeState()
        st.slots.active["ETHUSDT:x1"] = SlotRecord(
            position_id="ETHUSDT:x1",
            symbol="ETHUSDT",
            archetype="me",
        )
        assert pcm._count_archetype_slots("me") == 0
        pcm.hydrate_slot_evidence_from_constitution_slots(st)
        assert pcm._count_archetype_slots("me") == 1
        assert "ETHUSDT:me" in pcm._slot_evidence

    def test_last_decide_trace_keys_after_decide(self):
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("bpc")]))
        pcm.decide(features=FEATURES, symbol="BTCUSDT")
        for k in (
            "all_intents",
            "accepted",
            "drop_direction_policy",
            "drop_family_conflict",
            "drop_daily_limit",
            "drop_slot",
        ):
            assert k in pcm._last_decide_trace

    def test_saved_event_backtest_json_has_pcm_funnel_columns(self):
        """本地 rolling 产物：event_backtest 写出 funnel_per_bar 的 pcm_* 列（无文件则 skip）。"""
        root = Path(__file__).resolve().parents[2]
        p = (
            root
            / "results/me/research_roll.features_on/_rolling_sim/20260413_130313/fast_month_2024-11/me"
            / "event_backtest_me_pcm_trace.json"
        )
        if not p.is_file():
            pytest.skip(f"artifact not present: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = [
            r for r in (data.get("funnel_per_bar") or []) if r.get("strategy") == "me"
        ]
        assert rows
        need = {
            "pcm_n_candidates",
            "pcm_n_accepted",
            "pcm_drop_direction_policy",
            "pcm_drop_family_conflict",
            "pcm_drop_daily_limit",
            "pcm_drop_slot",
        }
        for k in need:
            assert k in rows[0], f"missing funnel_per_bar column {k}"


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
    """多策略场景：per-strategy 独立 slot（无跨策略竞争）"""

    def test_multi_strategy_all_pass(self):
        """多策略同时触发 → 全部返回（各自独立 slot）"""
        bpc_intent = _make_intent("BPC", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.9)

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 2
        archetypes = {r.archetype for r in result}
        assert archetypes == {"BPC", "ME"}

    def test_three_strategies_all_pass(self):
        """三策略同时触发 → 全部返回"""
        me_intent = _make_intent("ME", confidence=0.6)
        fer_intent = _make_intent("FER", confidence=1.0)
        bpc_intent = _make_intent("BPC", confidence=0.5)

        pcm = LivePCM(max_slots=3)
        pcm.register("me", FakeStrategy(intents=[me_intent]))
        pcm.register("fer", FakeStrategy(intents=[fer_intent]))
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 3
        archetypes = {r.archetype for r in result}
        assert archetypes == {"ME", "FER", "BPC"}

    def test_four_strategies_all_pass(self):
        """四个 archetype 同时触发 → 全部返回"""
        bpc = _make_intent("BPC", confidence=0.5)
        me = _make_intent("ME", confidence=0.9)
        fer = _make_intent("FER", confidence=1.0)
        lv = _make_intent("LV", confidence=1.0)

        pcm = LivePCM(max_slots=4)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 4

    def test_same_archetype_evidence_competition(self):
        """同 symbol+archetype 仅保留一条（deterministic 排序后的第一条）"""
        bpc_high = _make_intent("BPC", confidence=0.9)
        bpc_low = _make_intent("BPC", confidence=0.3)

        # 注意: 两个都注册为 archetype "BPC" 的变体
        # 但 per-strategy slot 用 archetype key 限制，所以第二个会被 evidence 竞争
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc1", FakeStrategy(intents=[bpc_low]))
        pcm.register("bpc2", FakeStrategy(intents=[bpc_high]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 1
        # 两条同 archetype/timeframe/stop 时，FIFO: 先注册先保留
        assert result[0].confidence == pytest.approx(0.3)

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
        """confidence=None → 默认 0.5, 不影响独立 slot 分配"""
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

        # 两策略独立 slot, 均返回
        assert len(result) == 2

    def test_custom_priority_order_no_effect(self):
        """自定义优先级顺序不再影响选择（per-strategy 独立）"""
        bpc_intent = _make_intent("BPC", confidence=0.5)
        me_intent = _make_intent("ME", confidence=0.5)

        pcm = LivePCM(archetype_priority=["ME", "BPC", "FER", "LV"], max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[bpc_intent]))
        pcm.register("me", FakeStrategy(intents=[me_intent]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 2

    def test_unknown_archetype_gets_slot(self):
        """不在优先级列表中的 archetype 也能获得独立 slot"""
        unknown = _make_intent("FooBar", confidence=1.0)
        bpc = _make_intent("BPC", confidence=0.1)

        pcm = LivePCM(max_slots=2)
        pcm.register("foobar", FakeStrategy(intents=[unknown]))
        pcm.register("bpc", FakeStrategy(intents=[bpc]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")

        assert len(result) == 2
        archetypes = {r.archetype for r in result}
        assert archetypes == {"FooBar", "BPC"}


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

    def test_same_symbol_same_archetype_becomes_add_position_intent(self):
        """已有 {symbol}:{archetype} slot 时，同策略再次触发会转为 add_position 意图（仍返回 1 条）。"""
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("BPC", confidence=0.8)]))
        first = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        second = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(first) == 1
        assert len(second) == 1
        assert bool(second[0].add_position) is True

    def test_same_symbol_same_archetype_allows_when_marked_add_position(self):
        pcm = LivePCM(max_slots=2)
        add_intent = _make_intent("BPC", confidence=0.8)
        add_intent = TradeIntent(
            action=add_intent.action,
            symbol=add_intent.symbol,
            archetype=add_intent.archetype,
            execution_strategy=add_intent.execution_strategy,
            confidence=add_intent.confidence,
            add_position=True,
        )
        pcm.register("bpc", FakeStrategy(intents=[_make_intent("BPC", confidence=0.8)]))
        assert len(pcm.decide(features=FEATURES, symbol="BTCUSDT")) == 1
        pcm.unregister("bpc")
        pcm.register("bpc", FakeStrategy(intents=[add_intent]))
        out = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(out) == 1


class TestLivePCMDeterministicSelection:

    def test_larger_timeframe_wins_same_symbol_archetype(self):
        i1 = _make_intent("BPC", confidence=0.2)
        i2 = _make_intent("BPC", confidence=0.9)
        pcm = LivePCM(max_slots=2)
        pcm.register("bpc60", FakeStrategy(intents=[i1]), timeframe="60T")
        pcm.register("bpc240", FakeStrategy(intents=[i2]), timeframe="240T")
        out = pcm.decide(
            features=FEATURES,
            symbol="BTCUSDT",
            features_by_timeframe={"60T": FEATURES, "240T": FEATURES},
        )
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.9)

    def test_smaller_effective_stop_pct_wins_when_timeframe_archetype_same(self):
        i1 = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="BPC",
            confidence=0.1,
            execution_profile={
                "rr_constraints": {"stop_loss_r": 3.0, "max_stop_pct": 0.03}
            },
        )
        i2 = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="BPC",
            confidence=0.9,
            execution_profile={
                "rr_constraints": {"stop_loss_r": 1.0, "max_stop_pct": 0.03}
            },
        )
        f = {"close": 10000.0, "atr": 100.0}
        pcm = LivePCM(max_slots=2)
        pcm.register("bpcA", FakeStrategy(intents=[i1]), timeframe="240T")
        pcm.register("bpcB", FakeStrategy(intents=[i2]), timeframe="240T")
        out = pcm.decide(
            features=f,
            symbol="BTCUSDT",
            features_by_timeframe={"240T": f},
        )
        assert len(out) == 1
        # i2 stop_pct=1%, i1=3%
        assert out[0].confidence == pytest.approx(0.9)


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
        assert DEFAULT_ARCHETYPE_PRIORITY == ["LV", "FER", "ME", "BPC"]


# ────────────────────────────────────────
# Regime Detector 测试
# ────────────────────────────────────────


class TestRegimeDetector:
    """RegimeDetector 状态机测试"""

    def test_default_is_normal(self):
        """默认 regime = NORMAL"""
        rd = RegimeDetector()
        assert rd.current_regime == REGIME_NORMAL
        assert rd.current_priority == ["LV", "FER", "ME", "BPC"]

    def test_detect_high_vol(self):
        """atr_percentile > 0.7 → HIGH_VOL"""
        rd = RegimeDetector(min_bars_in_regime=1)
        rd.detect({"atr_percentile": 0.8})
        assert rd.current_regime == REGIME_HIGH_VOL
        assert rd.current_priority == ["LV", "ME", "FER", "BPC"]

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

    def test_with_regime_detector_high_vol_both_pass(self):
        """HIGH_VOL regime → BPC 和 ME 都独立通过（不再竞争）"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        bpc = _make_intent("BPC", confidence=0.9)
        me = _make_intent("ME", confidence=0.5)

        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))

        # HIGH_VOL features
        result = pcm.decide(
            features={**FEATURES, "atr_percentile": 0.9},
            symbol="BTCUSDT",
        )
        assert len(result) == 2
        assert pcm.current_regime == REGIME_HIGH_VOL

    def test_with_regime_detector_high_leverage_both_pass(self):
        """HIGH_LEVERAGE regime → 两策略独立通过"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        bpc = _make_intent("BPC", confidence=1.0)
        lv = _make_intent("LV", confidence=0.3)

        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(
            features={
                **FEATURES,
                "oi_zscore": 2.0,
                "funding_rate_abs_zscore": 3.0,
            },
            symbol="BTCUSDT",
        )
        assert len(result) == 2
        assert pcm.current_regime == REGIME_HIGH_LEVERAGE

    def test_regime_switch_both_pass(self):
        """regime 切换后两策略均独立通过"""
        rd = RegimeDetector(min_bars_in_regime=1)
        pcm = LivePCM(regime_detector=rd, max_slots=2)

        fer = _make_intent("FER", confidence=0.8)
        me = _make_intent("ME", confidence=0.8)

        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("me", FakeStrategy(intents=[me]))

        # NORMAL: 两策略独立通过
        r1 = pcm.decide(features={**FEATURES, "atr_percentile": 0.3}, symbol="BTCUSDT")
        assert len(r1) == 2

        # 模拟上一轮仓位已关闭
        pcm.notify_position_closed("BTCUSDT", "FER")
        pcm.notify_position_closed("BTCUSDT", "ME")

        # HIGH_VOL: 两策略独立通过
        r2 = pcm.decide(features={**FEATURES, "atr_percentile": 0.9}, symbol="BTCUSDT")
        assert len(r2) == 2

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
    """Layer 3: Override 已移除 — per-strategy 独立 slot 模式下无跨策略覆盖"""

    def test_override_config_no_effect_per_strategy(self):
        """即使配置了 override_config，多策略也独立返回"""
        bpc = _make_intent("BPC", confidence=0.9)
        lv = _make_intent("LV", confidence=0.3)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 2  # 两策略独立 slot

    def test_all_four_archetypes_independent(self):
        """四个 archetype 同时触发，全部独立返回"""
        bpc = _make_intent("BPC", confidence=1.0)
        me = _make_intent("ME", confidence=1.0)
        fer = _make_intent("FER", confidence=1.0)
        lv = _make_intent("LV", confidence=0.1)

        pcm = LivePCM(max_slots=4, override_config=OVERRIDE_CONFIG)
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 4

    def test_global_slot_full_rejects_all(self):
        """全局 slot 已满，所有策略均被拒绝"""
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

    def test_fer_and_me_independent(self):
        """FER 和 ME 独立返回，不再互相覆盖"""
        me = _make_intent("ME", confidence=0.9)
        fer = _make_intent("FER", confidence=0.7)

        pcm = LivePCM(max_slots=2, override_config=OVERRIDE_CONFIG)
        pcm.register("me", FakeStrategy(intents=[me]))
        pcm.register("fer", FakeStrategy(intents=[fer]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 2
        archetypes = {r.archetype for r in result}
        assert archetypes == {"ME", "FER"}

    def test_override_not_active_without_config_per_strategy(self):
        """无 override 配置，多策略独立返回"""
        bpc = _make_intent("BPC", confidence=0.3)
        lv = _make_intent("LV", confidence=1.0)

        pcm = LivePCM(max_slots=2)  # 无 override_config
        pcm.register("bpc", FakeStrategy(intents=[bpc]))
        pcm.register("lv", FakeStrategy(intents=[lv]))

        result = pcm.decide(features=FEATURES, symbol="BTCUSDT")
        assert len(result) == 2

    def test_override_with_regime_detector_per_strategy(self):
        """Override + Regime 共存：两策略独立返回"""
        rd = RegimeDetector(min_bars_in_regime=1)
        bpc = _make_intent("BPC", confidence=0.9)
        lv = _make_intent("LV", confidence=0.3)

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
        assert len(result) == 2
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


class TestLivePCMDirectionPolicy:
    def _write_constitution(self, tmp_path):
        p = tmp_path / "constitution.yaml"
        p.write_text(
            """
version: 1
name: "C_TEST"
slots:
  enabled: true
  slot_count: 4
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits: {}
  direction_policy:
    enabled: true
    mode: ema200_single_direction
    close_feature: close
    ema_feature: ema_200
    debounce_bars: 1
    default_side: short
    fer_reverse_allowed: true
    reverse_exempt_families: [fer]
""",
            encoding="utf-8",
        )
        return str(p)

    def test_ema200_blocks_opposite_for_non_fer(self, tmp_path):
        cy = self._write_constitution(tmp_path)
        pcm = LivePCM(constitution_yaml=cy)
        pcm.register(
            "bpc-long-240T",
            FakeStrategy(intents=[_make_intent("bpc-long-240T", "LONG")]),
        )
        pcm.register(
            "bpc-short-240T",
            FakeStrategy(intents=[_make_intent("bpc-short-240T", "SHORT")]),
        )

        out = pcm.decide(
            features={"close": 51000.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )

        assert len(out) == 1
        assert out[0].archetype.lower() == "bpc-long-240t"

    def test_ema200_allows_fer_reverse_exception(self, tmp_path):
        cy = self._write_constitution(tmp_path)
        pcm = LivePCM(constitution_yaml=cy)
        pcm.register(
            "fer-short-240T",
            FakeStrategy(intents=[_make_intent("fer-short-240T", "SHORT")]),
        )
        pcm.register(
            "bpc-short-240T",
            FakeStrategy(intents=[_make_intent("bpc-short-240T", "SHORT")]),
        )

        out = pcm.decide(
            features={"close": 51000.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )

        assert len(out) == 1
        assert out[0].archetype.lower() == "fer-short-240t"

    def test_direction_debounce_blocks_until_confirmed(self, tmp_path):
        p = tmp_path / "constitution.yaml"
        p.write_text(
            """
version: 1
name: "C_TEST"
slots:
  enabled: true
  slot_count: 4
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits: {}
  direction_policy:
    enabled: true
    mode: ema200_single_direction
    close_feature: close
    ema_feature: ema_200
    debounce_bars: 3
    default_side: short
    fer_reverse_allowed: false
    reverse_exempt_families: [fer]
""",
            encoding="utf-8",
        )
        pcm = LivePCM(constitution_yaml=str(p))
        pcm.register(
            "bpc-long-240T",
            FakeStrategy(intents=[_make_intent("bpc-long-240T", "LONG")]),
        )

        # 先用3根空头确认空方向（long 应被拒绝）
        out0 = pcm.decide(
            features={"close": 49000.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        out0b = pcm.decide(
            features={"close": 48900.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        out0c = pcm.decide(
            features={"close": 48800.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        assert out0 == []
        assert out0b == []
        assert out0c == []

        # 刚切到多头，防抖期间不放行
        out1 = pcm.decide(
            features={"close": 51000.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        out2 = pcm.decide(
            features={"close": 51100.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        assert out1 == []
        assert out2 == []

        # 第3根确认后放行
        out3 = pcm.decide(
            features={"close": 51200.0, "ema_200": 50000.0},
            symbol="BTCUSDT",
        )
        assert len(out3) == 1
        assert out3[0].archetype.lower() == "bpc-long-240t"


class TestLivePCMNotionalPolicy:
    def _write_constitution(self, tmp_path):
        p = tmp_path / "constitution.yaml"
        p.write_text(
            """
version: 1
name: "C_TEST"
slots:
  enabled: true
  slot_count: 8
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits:
    bpc-long:
      max_slots: 3
      max_risk_per_trade: 0.02
    me-long:
      max_slots: 3
      max_risk_per_trade: 0.02
  notional_policy:
    enabled: true
    soft_max_total_notional_pct: 10.0
    hard_max_total_notional_pct: 12.0
    winner_priority:
      enabled: true
      allow_families: [bpc]
      require_breakeven_locked: true
      require_min_current_r: 1.0
      block_new_slots_when_soft_cap_hit: true
""",
            encoding="utf-8",
        )
        return str(p)

    def _intent(
        self,
        archetype: str,
        *,
        add: bool = False,
        locked: bool = False,
        cur_r: float = 0.0,
    ):
        return TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype=archetype,
            execution_strategy=archetype,
            confidence=0.8,
            add_position=add,
            locked_profit=locked,
            current_r=cur_r,
            size_multiplier=1.0,
            execution_profile={"rr_constraints": {"stop_loss_r": 2.0}},
        )

    def test_soft_cap_blocks_new_slot_but_allows_winner_add(self, tmp_path):
        cy = self._write_constitution(tmp_path)
        pcm = LivePCM(constitution_yaml=cy)
        f = {"close": 50000.0, "atr": 500.0}  # effective_stop_pct ~= 2%

        pcm.register(
            "bpc-long-240T",
            FakeStrategy(intents=[self._intent("bpc-long-240T")]),
        )
        out1 = pcm.decide(features=f, symbol="BTCUSDT")
        assert len(out1) == 1
        first_frac = pcm._current_total_notional_frac()
        assert first_frac > 0

        me_probe = self._intent("me-long-240T")
        me_delta = pcm._estimate_intent_notional_frac(
            me_probe,
            {"effective_stop_pct": pcm._effective_stop_pct_from_intent(me_probe, f)},
        )
        assert me_delta > 0
        pcm._constitution["notional_policy"]["soft_max_total_notional_pct"] = (
            first_frac + me_delta * 0.1
        )
        pcm._constitution["notional_policy"]["hard_max_total_notional_pct"] = (
            first_frac + me_delta * 2.0
        )

        pcm.unregister("bpc-long-240T")
        pcm.register(
            "me-long-240T",
            FakeStrategy(intents=[self._intent("me-long-240T")]),
        )
        out2 = pcm.decide(features=f, symbol="BTCUSDT")
        assert out2 == []  # soft cap 区间禁止新 slot
        s2 = pcm.get_stats()
        assert s2["notional_runtime"]["reject_counts"]["soft_cap_new_slot"] >= 1

        pcm.unregister("me-long-240T")
        add_probe = self._intent("bpc-long-240T", add=True, locked=True, cur_r=1.2)
        add_delta = pcm._estimate_intent_notional_frac(
            add_probe,
            {"effective_stop_pct": pcm._effective_stop_pct_from_intent(add_probe, f)},
        )
        assert add_delta > 0
        pcm._constitution["notional_policy"]["hard_max_total_notional_pct"] = (
            first_frac + add_delta * 1.2
        )
        pcm.register(
            "bpc-long-240T",
            FakeStrategy(
                intents=[
                    self._intent(
                        "bpc-long-240T",
                        add=True,
                        locked=True,
                        cur_r=1.2,
                    )
                ]
            ),
        )
        out3 = pcm.decide(features=f, symbol="BTCUSDT")
        assert len(out3) == 1  # winner priority 放行
        s3 = pcm.get_stats()
        assert s3["notional_runtime"]["total_notional_frac"] > first_frac
        assert s3["notional_runtime"]["last_snapshot"]["family"] == "bpc"

    def test_hard_cap_blocks_even_winner_add(self, tmp_path):
        cy = self._write_constitution(tmp_path)
        pcm = LivePCM(constitution_yaml=cy)
        f = {"close": 50000.0, "atr": 500.0}

        pcm.register(
            "bpc-long-240T",
            FakeStrategy(intents=[self._intent("bpc-long-240T")]),
        )
        assert len(pcm.decide(features=f, symbol="BTCUSDT")) == 1

        pcm.unregister("bpc-long-240T")
        add_probe = self._intent("bpc-long-240T", add=True, locked=True, cur_r=1.2)
        add_delta = pcm._estimate_intent_notional_frac(
            add_probe,
            {"effective_stop_pct": pcm._effective_stop_pct_from_intent(add_probe, f)},
        )
        assert add_delta > 0
        pcm.register(
            "bpc-long-240T",
            FakeStrategy(
                intents=[
                    self._intent("bpc-long-240T", add=True, locked=True, cur_r=1.2)
                ]
            ),
        )
        assert len(pcm.decide(features=f, symbol="BTCUSDT")) == 1
        total2 = pcm._current_total_notional_frac()
        pcm._constitution["notional_policy"]["hard_max_total_notional_pct"] = (
            total2 + add_delta * 0.5
        )

        pcm.unregister("bpc-long-240T")
        pcm.register(
            "bpc-long-240T",
            FakeStrategy(
                intents=[
                    self._intent("bpc-long-240T", add=True, locked=True, cur_r=1.2)
                ]
            ),
        )
        out = pcm.decide(features=f, symbol="BTCUSDT")
        assert out == []  # 第3次会触发 hard cap 拒绝
        s = pcm.get_stats()
        assert s["notional_runtime"]["reject_counts"]["hard_cap"] >= 1


def test_live_pcm_enforces_single_trend_per_symbol_when_enabled(tmp_path):
    constitution = tmp_path / "constitution.yaml"
    constitution.write_text(
        "\n".join(
            [
                "slots:",
                "  slot_count: 10",
                "  risk_per_slot: 0.01",
                "resource_allocation:",
                "  slot_policy:",
                "    trend_group: trend",
                "    min_trend_slots_per_symbol: 1",
                "    max_trend_slots_per_symbol: 1",
                "  archetype_groups:",
                "    trend: [bpc, tpc, me]",
                "  per_strategy_limits:",
                "    bpc: { max_risk_per_trade: 0.01 }",
                "    tpc: { max_risk_per_trade: 0.01 }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bpc = _make_intent("BPC", confidence=0.8)
    tpc = _make_intent("TPC", confidence=0.7)
    pcm = LivePCM(constitution_yaml=str(constitution), get_open_slot_count=lambda: 0)
    pcm.register("bpc", FakeStrategy(intents=[bpc]))
    pcm.register("tpc", FakeStrategy(intents=[tpc]))

    got = pcm.decide(features=FEATURES, symbol="BTCUSDT")

    assert len(got) == 1
    assert (
        int(pcm._last_decide_trace.get("drop_trend_symbol_slot_conflict", 0) or 0) >= 1
    )


def _write_trend_pool_guard_constitution(
    tmp_path,
    *,
    max_after_unlock: int = 3,
    anchor_symbol: str = "",
    require_anchor_first: bool = False,
    correlation_guard: bool = False,
) -> str:
    constitution = tmp_path / "constitution_guard.yaml"
    anchor_lines = []
    if anchor_symbol:
        anchor_lines = [
            f"      anchor_symbol: {anchor_symbol}",
            f"      require_anchor_first: {str(bool(require_anchor_first)).lower()}",
        ]
    correlation_lines = []
    if correlation_guard:
        correlation_lines = [
            "      symbol_correlation_guard:",
            "        enabled: true",
            "        threshold: 0.8",
            "        same_direction_only: true",
            "        pairs:",
            "          BTCUSDT:",
            "            ETHUSDT: 0.9",
        ]
    constitution.write_text(
        "\n".join(
            [
                "slots:",
                "  slot_count: 10",
                "  risk_per_slot: 0.01",
                "resource_allocation:",
                "  slot_policy:",
                "    trend_group: trend",
                "    min_trend_slots_per_symbol: 1",
                "    max_trend_slots_per_symbol: 1",
                "    trend_pool_guard:",
                "      enabled: true",
                "      max_unprotected_symbols: 1",
                "      unlock_on: breakeven_locked",
                f"      max_symbols_after_unlock: {int(max_after_unlock)}",
                *anchor_lines,
                *correlation_lines,
                "  archetype_groups:",
                "    trend: [bpc, tpc, me]",
                "  per_strategy_limits:",
                "    bpc: { max_risk_per_trade: 0.01 }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(constitution)


def _intent_with_symbol(archetype: str, symbol: str) -> TradeIntent:
    return TradeIntent(
        action="LONG",
        symbol=symbol,
        archetype=archetype,
        execution_strategy=archetype.lower(),
        confidence=0.8,
    )


def test_trend_pool_guard_blocks_new_unprotected_symbol(tmp_path):
    cy = _write_trend_pool_guard_constitution(tmp_path, max_after_unlock=3)

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "breakeven_locked": False,
                "stop_risk_nonnegative": False,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert got == []
    assert (
        int(pcm._last_decide_trace.get("drop_trend_pool_unprotected_cap", 0) or 0) >= 1
    )


def test_trend_pool_guard_unlocks_after_protected_winner(tmp_path):
    cy = _write_trend_pool_guard_constitution(tmp_path, max_after_unlock=3)

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "breakeven_locked": True,
                "stop_risk_nonnegative": True,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert len(got) == 1


def test_trend_pool_guard_respects_post_unlock_symbol_cap(tmp_path):
    cy = _write_trend_pool_guard_constitution(tmp_path, max_after_unlock=1)

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "breakeven_locked": True,
                "stop_risk_nonnegative": True,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert got == []
    assert (
        int(pcm._last_decide_trace.get("drop_trend_pool_post_unlock_cap", 0) or 0) >= 1
    )


def test_trend_pool_guard_anchor_requires_btc_first(tmp_path):
    cy = _write_trend_pool_guard_constitution(
        tmp_path,
        max_after_unlock=3,
        anchor_symbol="BTCUSDT",
        require_anchor_first=True,
    )

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 0,
        get_open_trend_positions=lambda: [],
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert got == []
    assert int(pcm._last_decide_trace.get("drop_trend_pool_anchor_first", 0) or 0) >= 1


def test_trend_pool_guard_anchor_unlocks_after_btc_protected(tmp_path):
    cy = _write_trend_pool_guard_constitution(
        tmp_path,
        max_after_unlock=3,
        anchor_symbol="BTCUSDT",
        require_anchor_first=True,
    )

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "breakeven_locked": True,
                "stop_risk_nonnegative": True,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert len(got) == 1


def test_trend_pool_guard_blocks_correlated_same_direction_symbol(tmp_path):
    cy = _write_trend_pool_guard_constitution(
        tmp_path,
        max_after_unlock=3,
        correlation_guard=True,
    )

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "side": "long",
                "breakeven_locked": True,
                "stop_risk_nonnegative": True,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register("bpc", FakeStrategy(intents=[_intent_with_symbol("BPC", "ETHUSDT")]))
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert got == []
    assert (
        int(pcm._last_decide_trace.get("drop_trend_pool_symbol_correlation", 0) or 0)
        >= 1
    )


def test_trend_pool_guard_allows_correlated_opposite_direction_symbol(tmp_path):
    cy = _write_trend_pool_guard_constitution(
        tmp_path,
        max_after_unlock=3,
        correlation_guard=True,
    )

    def _open_trend_positions():
        return [
            {
                "symbol": "BTCUSDT",
                "archetype": "bpc",
                "side": "long",
                "breakeven_locked": True,
                "stop_risk_nonnegative": True,
            }
        ]

    pcm = LivePCM(
        constitution_yaml=cy,
        get_open_slot_count=lambda: 1,
        get_open_trend_positions=_open_trend_positions,
    )
    pcm.register(
        "bpc",
        FakeStrategy(
            intents=[
                TradeIntent(
                    action="SHORT",
                    symbol="ETHUSDT",
                    archetype="BPC",
                    execution_strategy="bpc",
                    confidence=0.8,
                )
            ]
        ),
    )
    got = pcm.decide(features=FEATURES, symbol="ETHUSDT")
    assert len(got) == 1
