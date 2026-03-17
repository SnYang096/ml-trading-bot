from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.live_data_stream.order_flow_listener import OrderFlowListener
from src.live_data_stream.system_mode import SystemModeManager, SystemMode
from src.order_management.models import Order, OrderType, OrderStatus, OrderSide
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
    SlotRecord,
    AddPositionRecord,
)


def _make_listener():
    om = MagicMock()
    om.binance_api = MagicMock()
    ce = MagicMock()
    rs = ConstitutionRuntimeState()
    mm = SystemModeManager()
    listener = OrderFlowListener(
        symbol="BTCUSDT",
        storage_manager=MagicMock(),
        feature_computer=MagicMock(),
        constitution_executor=ce,
        runtime_state=rs,
        order_manager=om,
        mode_manager=mm,
    )
    return listener, om, ce, rs, mm


def test_exchange_stop_fill_closes_local_position_and_releases_slot():
    listener, om, ce, rs, mm = _make_listener()
    pid = "BTCUSDT:1"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(seconds=5),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )
    rs.add_position.positions[pid] = AddPositionRecord(position_id=pid, add_count=2)
    om.handle_execution_report.return_value = Order(
        order_id="o1",
        position_id=pid,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.FILLED,
    )

    listener.on_execution_report(
        {"symbol": "BTCUSDT", "trade_time": int(datetime.now(timezone.utc).timestamp())}
    )

    assert listener._position_tracker.get(pid) is None
    ce.release_slot.assert_called()
    ce.save_runtime_state.assert_called()
    assert pid not in rs.add_position.positions
    assert mm.get_current_mode() == SystemMode.ABNORMAL


def test_execution_report_without_position_id_reconciles_when_exchange_flat():
    listener, om, ce, rs, _ = _make_listener()
    pid = "BTCUSDT:2"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )
    om.handle_execution_report.return_value = Order(
        order_id="o2",
        position_id=None,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.FILLED,
    )
    om.binance_api.get_position.return_value = {"size": 0.0}

    listener.on_execution_report({"symbol": "BTCUSDT"})

    assert listener._position_tracker.get(pid) is None
    ce.release_slot.assert_called()
