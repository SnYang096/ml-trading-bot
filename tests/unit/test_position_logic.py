"""
position_logic 共享模块单元测试

覆盖:
  1. build_position_dict: 基础构建 / BPC 扩展 / 通用 trailing / 默认值
  2. enforce_position: 7 步持仓管理
     - time_stop
     - breakeven_lock
     - HWM/LWM 更新
     - activation trailing
     - SL hit (LONG/SHORT)
     - TP hit (LONG/SHORT)
     - SL 优先于 TP (同 bar 同时触发)
     - 持仓继续 (无触发)
  3. 实盘 vs 回测调用方式: 实盘传同一 price, 回测传 OHLC
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)


# ─── helpers ───


def _make_intent(
    action="LONG",
    symbol="BTCUSDT",
    archetype="bpc",
    confidence=0.75,
    stop_loss_r=2.0,
    take_profit_r=2.5,
    max_holding_bars=50,
    allow_trailing=False,
    trailing_atr=None,
    bpc_position_config=None,
    strategy_specific=None,
) -> TradeIntent:
    """构造一个带 execution_profile 的 TradeIntent"""
    rr = {
        "stop_loss_r": stop_loss_r,
        "take_profit_r": take_profit_r,
        "max_holding_bars": max_holding_bars,
        "allow_trailing": allow_trailing,
    }
    if trailing_atr is not None:
        rr["trailing_atr"] = trailing_atr
    ep = {"rr_constraints": rr}
    if bpc_position_config is not None:
        ep["bpc_position_config"] = bpc_position_config
    if strategy_specific is not None:
        ep["strategy_specific"] = strategy_specific
    return TradeIntent(
        action=action,
        symbol=symbol,
        archetype=archetype,
        confidence=confidence,
        execution_profile=ep,
    )


def _now():
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
# build_position_dict tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildPositionDict:

    def test_basic_long(self):
        intent = _make_intent(action="LONG", stop_loss_r=2.0, take_profit_r=3.0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["side"] == "LONG"
        assert pos["symbol"] == "BTCUSDT"
        assert pos["entry_price"] == 50000
        assert pos["atr_at_entry"] == 500
        assert pos["bar_minutes"] == 240
        # SL = entry - 2.0 * atr = 50000 - 1000 = 49000
        assert pos["stop_loss_price"] == pytest.approx(49000, abs=1)
        # TP = entry + 3.0 * atr = 50000 + 1500 = 51500
        assert pos["take_profit_price"] == pytest.approx(51500, abs=1)
        assert pos["max_holding_bars"] == 50
        assert pos["evidence_score"] == 0.75
        assert pos["bars_counted"] == 0

    def test_basic_short(self):
        intent = _make_intent(action="SHORT", stop_loss_r=1.5, take_profit_r=2.0)
        pos = build_position_dict(
            intent, entry_price=3000, atr=100, bar_minutes=60, entry_time=_now()
        )
        assert pos["side"] == "SHORT"
        # SHORT SL = entry + 1.5 * atr = 3000 + 150 = 3150
        assert pos["stop_loss_price"] == pytest.approx(3150, abs=1)
        # SHORT TP = entry - 2.0 * atr = 3000 - 200 = 2800
        assert pos["take_profit_price"] == pytest.approx(2800, abs=1)

    def test_no_sl_tp_when_atr_zero(self):
        intent = _make_intent(stop_loss_r=2.0, take_profit_r=2.5)
        pos = build_position_dict(
            intent, entry_price=50000, atr=0, bar_minutes=240, entry_time=_now()
        )
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None

    def test_no_sl_tp_when_rr_zero(self):
        intent = _make_intent(stop_loss_r=0, take_profit_r=0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None

    def test_bpc_trailing_config(self):
        bpc_cfg = {
            "activation_r": 1.0,
            "trail_r": 0.8,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 0.5,
            "bar_minutes": 240,
        }
        intent = _make_intent(bpc_position_config=bpc_cfg)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["activation_r"] == 1.0
        assert pos["trail_r"] == 0.8
        assert pos["trailing_activated"] is False
        assert pos["breakeven_enabled"] is True
        assert pos["breakeven_trigger_r"] == 0.5
        assert pos["breakeven_locked"] is False
        assert pos["high_water_mark"] == 50000  # LONG → HWM = entry

    def test_generic_trailing(self):
        intent = _make_intent(allow_trailing=True, trailing_atr=1.5)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["activation_r"] == 1.5
        assert pos["trail_r"] == 1.5
        assert pos["breakeven_enabled"] is False

    def test_strategy_specific_tier(self):
        intent = _make_intent(strategy_specific={"tier_name": "高证据"})
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["tier_name"] == "高证据"

    def test_entry_time_preserved(self):
        t = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        intent = _make_intent()
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=t
        )
        assert pos["entry_time"] == t

    def test_entry_time_default(self):
        intent = _make_intent()
        pos = build_position_dict(intent, entry_price=50000, atr=500)
        assert isinstance(pos["entry_time"], datetime)
        assert pos["entry_time"].tzinfo is not None

    def test_initial_risk_distance(self):
        """initial_risk_distance = stop_loss_r * atr (用于 R-multiple 归一化)"""
        intent = _make_intent(stop_loss_r=2.0, take_profit_r=3.0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["initial_risk_distance"] == pytest.approx(1000)  # 2.0 * 500

    def test_initial_risk_distance_fallback_to_atr(self):
        """stop_loss_r=0 时 initial_risk_distance 退化为 raw ATR"""
        intent = _make_intent(stop_loss_r=0, take_profit_r=0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["initial_risk_distance"] == pytest.approx(500)

    def test_activation_r_from_rr_constraints(self):
        """rr_constraints.activation_r 优先于 trailing_atr 作为 activation_r"""
        rr = {
            "stop_loss_r": 2.0,
            "take_profit_r": 3.0,
            "max_holding_bars": 50,
            "allow_trailing": True,
            "activation_r": 1.5,
            "trailing_atr": 0.5,
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="test",
            confidence=0.75,
            execution_profile=ep,
        )
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        # activation_r 应该取 rr_constraints.activation_r (1.5)，不是 trailing_atr (0.5)
        assert pos["activation_r"] == pytest.approx(1.5)
        # trail_r 应该取 trailing_atr (0.5)
        assert pos["trail_r"] == pytest.approx(0.5)


# ═════════════════════════════════════════════════════════════════════════════
# enforce_position tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEnforcePositionTimeStop:

    def test_time_stop_triggers(self):
        """持仓超过 max_holding_bars * bar_minutes 后应触发 time_stop"""
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        # 10 bars * 240 min = 2400 min = 40 hours
        future = entry_time + timedelta(hours=41)
        reason, exit_price = enforce_position(
            pos,
            price_high=50100,
            price_low=49900,
            price_close=50050,
            now=future,
        )
        assert reason == "time_stop"
        assert exit_price == 50050  # time stop 用 close

    def test_no_time_stop_within_limit(self):
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        within = entry_time + timedelta(hours=20)
        reason, _ = enforce_position(
            pos,
            price_high=50100,
            price_low=49900,
            price_close=50050,
            now=within,
        )
        assert reason is None


class TestEnforcePositionSL:

    def test_long_stop_loss_hit(self):
        """LONG: low <= SL → 止损"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=50200,
            price_low=48900,  # low < SL (49000)
            price_close=49500,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)  # 精确 SL 价

    def test_short_stop_loss_hit(self):
        """SHORT: high >= SL → 止损"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=3200,  # high > SL (3150)
            price_low=2950,
            price_close=3100,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(3150)

    def test_long_sl_not_hit(self):
        """LONG: low > SL → 不触发"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,  # 远离 TP
        }
        reason, _ = enforce_position(
            pos,
            price_high=50500,
            price_low=49100,  # low > SL
            price_close=50200,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None


class TestEnforcePositionTP:

    def test_long_take_profit_hit(self):
        """LONG: high >= TP → 止盈"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=51600,  # high > TP
            price_low=50800,
            price_close=51200,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(51500)

    def test_short_take_profit_hit(self):
        """SHORT: low <= TP → 止盈"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=2950,
            price_low=2790,  # low < TP (2800)
            price_close=2850,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(2800)


class TestEnforcePositionSLPriority:

    def test_sl_priority_over_tp_long(self):
        """同一根 bar SL 和 TP 都触发时, SL 优先 (保守假设)"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        # 极端 bar: low 穿 SL, high 穿 TP
        reason, exit_price = enforce_position(
            pos,
            price_high=52000,
            price_low=48500,
            price_close=50000,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)

    def test_sl_priority_over_tp_short(self):
        """SHORT: 同一 bar SL 和 TP 都触发, SL 优先"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=3200,  # triggers SL
            price_low=2700,  # triggers TP
            price_close=2950,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(3150)


class TestEnforcePositionBreakeven:

    def test_breakeven_lock_triggers(self):
        """当利润 >= breakeven_trigger_r, SL 应锁到入场价"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 52000,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "breakeven_locked": False,
        }
        # profit_r = (51000 - 50000) / 500 = 2.0 >= trigger 1.0
        reason, _ = enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None  # 不关仓
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == 50000  # SL 锁到入场价

    def test_breakeven_not_triggered_below_threshold(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 52000,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "breakeven_locked": False,
        }
        # profit_r = (50200 - 50000) / 500 = 0.4 < trigger 1.0
        enforce_position(
            pos,
            price_high=50200,
            price_low=50000,
            price_close=50100,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is False
        assert pos["stop_loss_price"] == 49000  # 未变


class TestEnforcePositionTrailing:

    def test_activation_trailing_long(self):
        """LONG: 利润达到 activation_r 后, trailing SL 上移"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,
            "activation_r": 1.0,
            "trail_r": 0.8,
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # bar 1: price high=51000 → profit_r = (51000-50000)/500 = 2.0 >= activation 1.0
        # HWM = 51000, trail_sl = 51000 - 0.8*500 = 50600
        enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert pos["trailing_activated"] is True
        assert pos["high_water_mark"] == 51000
        assert pos["stop_loss_price"] == pytest.approx(50600)

        # bar 2: HWM goes to 51500 → trail_sl = 51500 - 400 = 51100
        enforce_position(
            pos,
            price_high=51500,
            price_low=50900,
            price_close=51200,
            now=_now() + timedelta(hours=2),
        )
        assert pos["high_water_mark"] == 51500
        assert pos["stop_loss_price"] == pytest.approx(51100)

    def test_activation_trailing_short(self):
        """SHORT: 利润达到 activation_r 后, trailing SL 下移"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3200,
            "take_profit_price": 2500,
            "activation_r": 1.5,
            "trail_r": 1.0,
            "trailing_activated": False,
            "high_water_mark": None,
            "low_water_mark": 3000,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # profit_r = (3000 - 2800) / 100 = 2.0 >= 1.5
        # LWM = 2800, trail_sl = 2800 + 1.0*100 = 2900
        enforce_position(
            pos,
            price_high=2950,
            price_low=2800,
            price_close=2850,
            now=_now() + timedelta(hours=1),
        )
        assert pos["trailing_activated"] is True
        assert pos["low_water_mark"] == 2800
        assert pos["stop_loss_price"] == pytest.approx(2900)

    def test_trailing_sl_never_moves_backwards_long(self):
        """LONG trailing: SL 只能上移不能下移"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 50600,  # 已经较高
            "take_profit_price": 55000,
            "activation_r": 1.0,
            "trail_r": 0.8,
            "trailing_activated": True,
            "high_water_mark": 51000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # 回调: high=50800, profit_r = (50800-50000)/500 = 1.6 >= 1.0
        # 但 HWM 不变, trail_sl = 51000 - 400 = 50600, 不低于当前 SL
        enforce_position(
            pos,
            price_high=50800,
            price_low=50400,
            price_close=50600,
            now=_now() + timedelta(hours=1),
        )
        # SL 应保持不变 (不会下移)
        assert pos["stop_loss_price"] == pytest.approx(50600)


class TestEnforcePositionLiveMode:
    """验证实盘模式: price_high = price_low = price_close = current_price"""

    def test_live_sl_hit(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        current = 48800  # 低于 SL
        reason, exit_price = enforce_position(
            pos,
            price_high=current,
            price_low=current,
            price_close=current,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)

    def test_live_tp_hit(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        current = 52000
        reason, exit_price = enforce_position(
            pos,
            price_high=current,
            price_low=current,
            price_close=current,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(51500)


class TestEnforcePositionNoClose:

    def test_position_stays_open(self):
        """价格在 SL/TP 之间且未超时 → 无关仓"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, _ = enforce_position(
            pos,
            price_high=50300,
            price_low=49800,
            price_close=50100,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None

    def test_no_sl_tp_prices(self):
        """无 SL/TP 时, 只有 time_stop 能触发"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": None,
            "take_profit_price": None,
        }
        reason, _ = enforce_position(
            pos,
            price_high=40000,
            price_low=30000,
            price_close=35000,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None  # 无 SL/TP 则不触发


class TestEnforcePositionMutatesPos:
    """验证 enforce_position 就地修改 pos 字典 (与实盘行为一致)"""

    def test_hwm_updated(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,
            "high_water_mark": 50500,
            "low_water_mark": None,
        }
        enforce_position(
            pos,
            price_high=51000,
            price_low=50200,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert pos["high_water_mark"] == 51000

    def test_lwm_updated_short(self):
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
            "high_water_mark": None,
            "low_water_mark": 2950,
        }
        enforce_position(
            pos,
            price_high=2980,
            price_low=2900,
            price_close=2920,
            now=_now() + timedelta(hours=1),
        )
        assert pos["low_water_mark"] == 2900
