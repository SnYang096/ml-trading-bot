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
import time
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pandas as pd

from src.order_management.storage import Storage
from src.time_series_model.core.constitution.runtime_state import AddPositionRecord
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
    account_risk_limits=None,
) -> tuple[TradeExecutor, MagicMock, MagicMock, PositionTracker]:
    """创建 TradeExecutor 及其依赖 mock"""
    om = MagicMock()
    om.place_order.return_value = _make_mock_order()

    ce = MagicMock()
    ce.resolve_safety_db_path.side_effect = RuntimeError("no safety db in unit test")
    rs = MagicMock()
    # 模拟 active slots
    rs.slots.active = {}
    rs.add_position.positions = {}
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
        account_risk_limits=account_risk_limits,
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
        """risk_per_slot=0 且无 equity 时 fallback 到 risk_per_trade"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.0, risk_per_trade=100.0)
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is True
        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert qty > 0.0

    def test_constitution_sizing_skips_when_equity_unavailable(self):
        """risk_per_slot>0 且无 equity 时不 silent fallback 到 risk_per_trade"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01, risk_per_trade=100.0)
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is False
        om.place_order.assert_not_called()

    def test_constitution_sizing_rest_equity_when_features_missing(self):
        """features equity 缺失时 REST 刷新后走宪法反算"""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01, risk_per_trade=10.0)
        om.binance_api = MagicMock()
        om.binance_api.get_positions.return_value = []
        om.binance_api.get_account_balance.return_value = {
            "USDT": {"total": 10000.0, "free": 8000.0, "used": 2000.0},
            "info": {"totalMarginBalance": "10000.0"},
        }
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=0.0)

        result = ex.execute(intent=intent, features=features)

        assert result is True
        assert features["equity"] == pytest.approx(10000.0)
        qty = om.place_order.call_args_list[0].kwargs["quantity"]
        assert qty > 0.0
        assert om.binance_api.get_account_balance.call_count == 1

    def test_constitution_sizing_rest_equity_cache_writeback(self):
        """Cached REST equity is written to features for slot enforcement."""
        ex, om, ce, pt = _make_executor(risk_per_slot=0.01, risk_per_trade=10.0)
        om.binance_api = MagicMock()
        om.binance_api.get_positions.return_value = []
        om.binance_api.get_account_balance.return_value = {
            "USDT": {"total": 10000.0, "free": 8000.0, "used": 2000.0},
            "info": {"totalMarginBalance": "10000.0"},
        }
        ex._rest_equity_cache_value = 12000.0
        ex._rest_equity_cache_ts = time.time()
        intent = _make_intent(stop_loss_r=2.0)
        features = _make_features(close=50000.0, atr=500.0, equity=0.0)

        equity = ex._resolve_sizing_equity(features)

        assert equity == pytest.approx(12000.0)
        assert features["equity"] == pytest.approx(12000.0)
        om.binance_api.get_account_balance.assert_not_called()

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


class TestAccountRiskLimits:

    @staticmethod
    def _attach_account_snapshot(om, *, gross_notional=0.0, equity=10000.0):
        om.binance_api = MagicMock()
        om.binance_api.get_positions.return_value = [
            {"symbol": "ETH/USDT:USDT", "notional": gross_notional}
        ]
        om.binance_api.get_account_balance.return_value = {
            "USDT": {"total": equity, "free": equity * 0.8, "used": equity * 0.2},
            "info": {
                "totalMarginBalance": str(equity),
                "availableBalance": str(equity * 0.8),
                "totalPositionInitialMargin": str(equity * 0.2),
            },
        }

    def test_account_gross_leverage_limit_blocks_new_risk(self):
        ex, om, ce, pt = _make_executor(
            risk_per_slot=0.01,
            account_risk_limits={
                "enabled": True,
                "max_gross_leverage": 3.0,
                "fail_closed": True,
            },
        )
        # Normal risk sizing produces ~5000 notional; 29000 + 5000 > 3x equity.
        self._attach_account_snapshot(om, gross_notional=29000.0, equity=10000.0)

        result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is False
        om.place_order.assert_not_called()

    def test_account_risk_limit_allows_order_under_caps(self):
        ex, om, ce, pt = _make_executor(
            risk_per_slot=0.01,
            account_risk_limits={
                "enabled": True,
                "max_gross_leverage": 3.0,
                "max_projected_initial_margin_pct": 0.80,
                "min_projected_available_margin_pct": 0.20,
                "margin_stress_leverage": 5.0,
                "fail_closed": True,
            },
        )
        self._attach_account_snapshot(om, gross_notional=10000.0, equity=10000.0)

        result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is True
        om.place_order.assert_called()

    def test_account_risk_limit_fail_closed_when_snapshot_missing(self):
        ex, om, ce, pt = _make_executor(
            risk_per_slot=0.01,
            account_risk_limits={"enabled": True, "fail_closed": True},
        )
        om.binance_api = MagicMock()
        om.binance_api.get_positions.side_effect = RuntimeError("rate limited")

        result = ex.execute(intent=_make_intent(), features=_make_features())

        assert result is False
        om.place_order.assert_not_called()


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

    def test_execute_persists_software_stop_to_storage(self, tmp_path):
        """开仓成功后 SQLite positions 也能看到软件 SL。"""
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        om.storage = Storage(str(tmp_path / "orders.db"))
        ep = {"rr_constraints": {"stop_loss_r": 2.0, "take_profit_r": None}}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            execution_profile=ep,
            position_id="BTCUSDT:software-sl",
        )

        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(intent=intent, features=_make_features())

        assert result is True
        stored = om.storage.get_position("BTCUSDT:software-sl")
        assert stored is not None
        assert stored.stop_loss_price is not None
        assert stored.status.value == "open"


class TestAddPositionLivePath:

    def test_add_position_uses_runtime_slot_archetype_for_parent_match(self):
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        parent_pid = "BTCUSDT:parent-1"
        pt.add(
            parent_pid,
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "initial_risk_distance": 1000.0,
                "atr_at_entry": 500.0,
            },
        )
        ex.runtime_state.slots.active[parent_pid] = MagicMock(archetype="bpc-long-120T")
        ex.runtime_state.add_position.positions[parent_pid] = MagicMock(add_count=0)
        ce.resolve_add_position_for_strategy.return_value = {
            "enabled": True,
            "max_add_times": 3,
            "require_locked_profit": False,
            "lock_profit_breakeven_trigger_r": 0.0,
            "trigger": {"enabled": False},
            "add_size_multipliers": [1.0],
        }
        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(
                intent=TradeIntent(
                    action="LONG",
                    symbol="BTCUSDT",
                    archetype="bpc-long-120T",
                    position_id="BTCUSDT:add-2",
                    add_position=True,
                    execution_profile={"rr_constraints": {"stop_loss_r": 2.0}},
                ),
                features=_make_features(close=52000.0, atr=500.0, equity=10000.0),
            )
        assert result is True
        ce.validate_add_position.assert_called_once()
        add_pos = pt.get("BTCUSDT:add-2")
        assert add_pos is not None
        assert add_pos.get("_is_add_position") is True
        assert add_pos.get("_parent_pid") == parent_pid
        assert add_pos.get("_share_parent_exit") is True

    def test_add_position_no_parent_is_rejected(self):
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        ce.resolve_add_position_for_strategy.return_value = {
            "enabled": True,
            "max_add_times": 3,
            "require_locked_profit": False,
            "lock_profit_breakeven_trigger_r": 0.0,
            "trigger": {"enabled": False},
            "add_size_multipliers": [1.0],
        }
        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(
                intent=TradeIntent(
                    action="LONG",
                    symbol="BTCUSDT",
                    archetype="bpc-long-120T",
                    add_position=True,
                    execution_profile={"rr_constraints": {"stop_loss_r": 2.0}},
                ),
                features=_make_features(close=52000.0, atr=500.0, equity=10000.0),
            )
        assert result is False
        om.place_order.assert_not_called()

    def test_add_position_min_interval_blocks_second_add_too_fast(self):
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        parent_pid = "BTCUSDT:parent-2"
        pt.add(
            parent_pid,
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "initial_risk_distance": 1000.0,
                "atr_at_entry": 500.0,
            },
        )
        ex.runtime_state.slots.active[parent_pid] = MagicMock(archetype="bpc")
        ref = "2025-06-01T10:00:00+00:00"
        ex.runtime_state.add_position.positions[parent_pid] = AddPositionRecord(
            position_id=parent_pid,
            add_count=1,
            locked_profit=False,
            updated_at=ref,
            last_add_at=ref,
        )
        ce.resolve_add_position_for_strategy.return_value = {
            "enabled": True,
            "max_add_times": 3,
            "require_locked_profit": False,
            "trigger": {"enabled": False},
            "add_size_multipliers": [1.0],
        }
        now = pd.Timestamp("2025-06-01T10:45:00+00:00", tz="UTC")
        ep = {
            "rr_constraints": {"stop_loss_r": 2.0},
            "execution_constraints": {"min_order_interval_minutes": 60},
        }
        with patch(
            "src.order_management.trade_executor.pd.Timestamp.now",
            return_value=now,
        ):
            with patch(
                "src.order_management.trade_executor.enforce_before_order",
            ):
                result = ex.execute(
                    intent=TradeIntent(
                        action="LONG",
                        symbol="BTCUSDT",
                        archetype="bpc",
                        position_id="BTCUSDT:add-interval",
                        add_position=True,
                        execution_profile=ep,
                    ),
                    features=_make_features(close=52000.0, atr=500.0, equity=10000.0),
                )
        assert result is False
        om.place_order.assert_not_called()

    def test_add_position_min_interval_uses_storage_fallback_after_restart(self):
        ex, om, ce, pt = _make_executor(trade_size=0.001)
        parent_pid = "BTCUSDT:parent-3"
        pt.add(
            parent_pid,
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "initial_risk_distance": 1000.0,
                "atr_at_entry": 500.0,
            },
        )
        ex.runtime_state.slots.active[parent_pid] = MagicMock(archetype="bpc")
        # Simulate restart-like state: no add_position runtime record.
        ex.runtime_state.add_position.positions = {}
        ce.resolve_add_position_for_strategy.return_value = {
            "enabled": True,
            "max_add_times": 3,
            "require_locked_profit": False,
            "trigger": {"enabled": False},
            "add_size_multipliers": [1.0],
        }
        om.storage = MagicMock()
        om.storage.get_latest_add_entry_time.return_value = "2025-06-01T10:00:00+00:00"
        now = pd.Timestamp("2025-06-01T10:45:00+00:00", tz="UTC")
        ep = {
            "rr_constraints": {"stop_loss_r": 2.0},
            "execution_constraints": {"min_order_interval_minutes": 60},
        }
        with patch(
            "src.order_management.trade_executor.pd.Timestamp.now",
            return_value=now,
        ):
            with patch(
                "src.order_management.trade_executor.enforce_before_order",
            ):
                result = ex.execute(
                    intent=TradeIntent(
                        action="LONG",
                        symbol="BTCUSDT",
                        archetype="bpc",
                        position_id="BTCUSDT:add-interval-storage",
                        add_position=True,
                        execution_profile=ep,
                    ),
                    features=_make_features(close=52000.0, atr=500.0, equity=10000.0),
                )
        assert result is False
        om.place_order.assert_not_called()


class TestStopGuardrailConsistency:

    def test_resolve_effective_stop_r_clips_and_returns_guardrail_source(self):
        ex, _, _, _ = _make_executor()
        sl_r, atr_pct, eff_pct, source = ex._resolve_effective_stop_r(
            sl_r=3.5,
            atr=100.0,
            entry_price=10000.0,
            rr_constraints={"max_stop_pct": 0.02},
        )
        assert atr_pct == pytest.approx(0.035)
        assert eff_pct == pytest.approx(0.02)
        assert source == "guardrail_clip"
        # 0.02*10000/100 = 2.0
        assert sl_r == pytest.approx(2.0)

    def test_execute_persists_stop_diagnostics_into_position(self):
        ex, om, _, pt = _make_executor(trade_size=0.001)
        ep = {
            "rr_constraints": {
                "stop_loss_r": 3.5,
                "take_profit_r": None,
                "max_stop_pct": 0.02,
            }
        }
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            execution_profile=ep,
        )
        with patch("src.order_management.trade_executor.enforce_before_order"):
            result = ex.execute(
                intent=intent, features=_make_features(close=10000, atr=100)
            )
        assert result is True
        pos = list(pt.all_positions().values())[0]
        assert pos["atr_stop_pct"] == pytest.approx(0.035)
        assert pos["effective_stop_pct"] == pytest.approx(0.02)
        assert pos["sizing_stop_source"] == "guardrail_clip"
