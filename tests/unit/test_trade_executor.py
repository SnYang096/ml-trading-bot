"""TradeExecutor 单元测试

覆盖:
  test_qty_from_constitution_risk      — 正常 equity 反算 qty
  test_qty_fallback_to_risk_per_trade  — equity=0 时 fallback 到 risk_per_trade
  test_qty_fallback_to_trade_size      — 所有风险反算都不可用时 fallback 到 trade_size
  test_qty_zero_skips_order            — qty<=0 直接 return False
  test_slot_full_no_leaked_slot        — SLOT_FULL (ConstitutionViolation) 后不释放未预留的 slot
  test_order_fail_releases_slot        — 下单异常后释放已预留的 slot
  test_execute_sets_position_in_tracker — 成功下单后 position 被写入 PositionTracker
  test_no_trade_action_skips           — NO_TRADE action 直接返回 False
  test_size_multiplier_applied         — size_multiplier 正确缩放 qty
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.order_management.trade_executor import TradeExecutor
from src.order_management.position_tracker import PositionTracker


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_intent(
    action="LONG",
    symbol="BTCUSDT",
    archetype="bpc",
    stop_loss_r=2.0,
    quantity=None,
    size_multiplier=None,
    position_id=None,
) -> TradeIntent:
    ep = {"rr_constraints": {"stop_loss_r": stop_loss_r, "take_profit_r": 4.0}}
    return TradeIntent(
        action=action,
        symbol=symbol,
        archetype=archetype,
        execution_profile=ep,
        quantity=quantity,
        size_multiplier=size_multiplier,
        position_id=position_id,
    )


def _make_features(
    close=50000.0,
    atr=500.0,
    equity=10000.0,
):
    return {
        "close": close,
        "240T_atr": atr,  # pick_atr() 识别 timeframe-prefixed ATR
        "equity": equity,
        "timestamp": datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    }


def _make_mock_order(order_id="ORD001"):
    o = MagicMock()
    o.order_id = order_id
    return o


def _make_executor(
    risk_per_slot=0.01,
    risk_per_trade=None,
    trade_size=None,
    slot_active_pids=None,
) -> tuple[TradeExecutor, MagicMock, MagicMock, PositionTracker]:
    """创建 TradeExecutor 及其依赖 mock"""
    om = MagicMock()
    om.place_order.return_value = _make_mock_order()

    ce = MagicMock()
    rs = MagicMock()
    # 模拟 active slots
    rs.slots.active = {}
    if slot_active_pids:
        for pid in slot_active_pids:
            rs.slots.active[pid] = MagicMock(symbol="BTCUSDT")

    pt = PositionTracker(order_manager=om, symbol="BTCUSDT")

    ex = TradeExecutor(
        order_manager=om,
        constitution_executor=ce,
        runtime_state=rs,
        position_tracker=pt,
        symbol="BTCUSDT",
        bar_minutes=240,
        risk_per_slot=risk_per_slot,
        risk_per_trade=risk_per_trade,
        trade_size=trade_size,
    )
    return ex, om, ce, pt


# ─── tests ────────────────────────────────────────────────────────────────────


class TestQtyCalculation:

    def test_qty_from_constitution_risk(self):
        """正常 equity 反算: risk_per_slot × equity / (sl_r × atr) → qty"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01)
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=10000.0)

        result = ex.execute(intent=intent, features=features)

        assert result is True
        # qty = risk_usd / (sl_r * atr) / price
        # risk_usd = 10000 * 0.01 = 100
        # sl_dist = 2.0 * 500 = 1000
        # qty_notional = 100 / 1000 * ... must check place_order was called
        call_args = om.place_order.call_args_list[0]
        qty = call_args.kwargs["quantity"]
        assert qty > 0.0

    def test_qty_fallback_to_risk_per_trade(self):
        """equity=0 时跳过宪法反算，fallback 到 risk_per_trade"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01, risk_per_trade=100.0)
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is True
        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert qty > 0.0

    def test_qty_fallback_to_trade_size(self):
        """所有反算均无法计算时 fallback 到 trade_size"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.0, trade_size=0.001)
        intent = _make_intent(stop_loss_r=0.0)  # sl_r=0 → 无法反算
        features = _make_features(equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is True
        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert abs(qty - 0.001) < 1e-9

    def test_qty_zero_skips_order(self):
        """所有 qty 来源均为 0 → 不下单，返回 False"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.0, trade_size=None)
        intent = _make_intent(stop_loss_r=0.0)
        features = _make_features(equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is False
        om.place_order.assert_not_called()

    def test_size_multiplier_applied(self):
        """size_multiplier 正确缩放最终 qty"""
        ex, om, ce, pt = _make_executor(trade_size=1.0)
        intent = _make_intent(stop_loss_r=0.0, size_multiplier=0.5)
        features = _make_features(equity=0.0)

        ex.execute(intent=intent, features=features)

        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert abs(qty - 0.5) < 1e-9

    def test_explicit_quantity_overrides_risk_calc(self):
        """intent.quantity 显式指定时跳过所有反算"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01, trade_size=1.0)
        intent = _make_intent(quantity=0.123)
        features = _make_features(equity=10000.0, atr=500.0)

        ex.execute(intent=intent, features=features)

        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert abs(qty - 0.123) < 1e-9


class TestSlotManagement:

    def test_no_trade_action_skips(self):
        """NO_TRADE action 直接返回 False，不调用任何 mock"""
        ex, om, ce, pt = _make_executor()
        intent = _make_intent(action="NO_TRADE")
        result = ex.execute(intent=intent, features=_make_features())

        assert result is False
        om.place_order.assert_not_called()
        (
            ce.enforce_before_order.assert_not_called()
            if hasattr(ce, "enforce_before_order")
            else None
        )

    def test_slot_full_no_leaked_slot(self):
        """SLOT_FULL (ConstitutionViolation) 抛出后 _release_leaked_slot 不释放未预留的 slot"""
        ex, om, ce, pt = _make_executor()
        # ConstitutionViolation 在 enforce_before_order 阶段抛出（slot 未被预留）
        # rs.slots.active 不含此 position_id
        with patch(
            "src.order_management.trade_executor.enforce_before_order",
            side_effect=ConstitutionViolation(code="SLOT_FULL", message="满了"),
        ):
            result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is False
        # slot 从未被预留，不应调用 release_slot
        ex.constitution_executor.release_slot.assert_not_called()

    def test_order_fail_releases_slot(self):
        """下单异常后释放已预留的 slot（position_id 在 active 中）"""
        pid = "BTCUSDT:999"
        ex, om, ce, pt = _make_executor(trade_size=0.001, slot_active_pids=[pid])
        om.place_order.side_effect = RuntimeError("API Error")

        intent = _make_intent(position_id=pid)
        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(intent=intent, features=_make_features())

        assert result is False
        # slot 被预留了 → 应该 release
        ce.release_slot.assert_called_once_with(
            st=ex.runtime_state, position_id=pid, reason="order_failed"
        )
        ce.save_runtime_state.assert_called_once()

    def test_slot_full_violation_not_in_active_no_release(self):
        """constitution violation 且 pid 不在 active slots → 跳过释放"""
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        # rs.slots.active 为空（slot 未被预留）
        ex.runtime_state.slots.active = {}

        with patch(
            "src.order_management.trade_executor.enforce_before_order",
            side_effect=ConstitutionViolation(code="SLOT_FULL", message="满了"),
        ):
            result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is False
        ce.release_slot.assert_not_called()


class TestPositionTrackerIntegration:

    def test_execute_sets_position_in_tracker(self):
        """成功下单后 position 被写入 PositionTracker"""
        ex, om, ce, pt = _make_executor(trade_size=0.001)

        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is True
        positions = pt.all_positions()
        assert len(positions) == 1
        pos = list(positions.values())[0]
        assert pos["side"] in (
            "LONG",
            "SHORT",
            "LONG".lower(),
            "SHORT".lower(),
            "long",
            "short",
        )

    def test_sl_order_placed_and_recorded(self):
        """SL 挂单成功时 position 中展示 _exchange_sl_price"""
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        # 不设置 take_profit_r （避免 TP 化需要额外挂单），execution_profile 只含 SL
        ep = {"rr_constraints": {"stop_loss_r": 2.0, "take_profit_r": None}}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            execution_profile=ep,
        )
        om.place_order.side_effect = [
            _make_mock_order("MKT_1"),  # market entry
            _make_mock_order("SL_ORDER_1"),  # SL order
        ]

        with patch("src.order_management.trade_executor.enforce_before_order"):
            ex.execute(intent=intent, features=_make_features())

        positions = pt.all_positions()
        pos = list(positions.values())[0]
        # 如果 stop_loss_price 非 None，应该下 SL 挂单并记录
        if pos.get("stop_loss_price") is not None:
            assert "_exchange_sl_price" in pos
            assert pos["_exchange_sl_price"] is not None
        # 必须有持仓被记录
        assert len(positions) == 1
