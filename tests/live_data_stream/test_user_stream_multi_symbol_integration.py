"""
User Data Stream ↔ MultiSymbolManager ↔ OrderFlowListener 端到端保护

不启真实 WebSocket：用 BinanceUserStream._handle_message 模拟推送，
用 MultiSymbolManager 的真实回调绑定验证路由与隔离。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.live_data_stream.multi_symbol_manager import MultiSymbolManager
from src.order_management.binance_user_stream import BinanceUserStream


pytest.importorskip("websockets")


def _make_manager(symbols: list[str]) -> MultiSymbolManager:
    om = MagicMock()
    om.binance_api = MagicMock()
    storage = MagicMock()
    return MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        order_manager=om,
        memory_window_hours=0.1,
        feature_compute_interval_minutes=60,
        feature_4h_interval_hours=4,
    )


def _pos_row_normalized(symbol: str, position_amount: float) -> dict:
    """与 BinanceUserStream._normalize_account_update 输出一致。"""
    return {
        "symbol": symbol,
        "position_amount": float(position_amount),
        "entry_price": 0.0,
        "unrealized_pnl": 0.0,
        "position_side": "BOTH",
    }


def test_account_update_broadcasts_equity_snapshot_to_all_listeners():
    manager = _make_manager(["BTCUSDT", "ETHUSDT"])
    btc = manager.listeners["BTCUSDT"]
    eth = manager.listeners["ETHUSDT"]

    manager._on_account_update(
        {
            "wallet_balance": 5000.0,
            "available_balance": 4800.0,
            "unrealized_pnl_total": 12.5,
            "event_time": 1710000000,
            "positions": [],
        }
    )

    assert btc._latest_account_update["wallet_balance"] == 5000.0
    assert eth._latest_account_update["wallet_balance"] == 5000.0
    assert btc._latest_account_update["event_time"] == 1710000000


def test_account_update_zero_position_only_clears_matching_symbol():
    manager = _make_manager(["BTCUSDT", "ETHUSDT"])
    btc = manager.listeners["BTCUSDT"]
    eth = manager.listeners["ETHUSDT"]
    now = datetime.now(timezone.utc)

    btc._position_tracker.add(
        "pid-btc",
        {"symbol": "BTCUSDT", "side": "LONG", "entry_time": now, "qty": 0.01},
    )
    eth._position_tracker.add(
        "pid-eth",
        {"symbol": "ETHUSDT", "side": "LONG", "entry_time": now, "qty": 0.2},
    )

    manager._on_account_update(
        {
            "wallet_balance": 4000.0,
            "available_balance": 3900.0,
            "unrealized_pnl_total": 0.0,
            "positions": [
                _pos_row_normalized("BTCUSDT", 0.0),
                _pos_row_normalized("ETHUSDT", 0.2),
            ],
        }
    )

    assert btc._position_tracker.get("pid-btc") is None
    assert eth._position_tracker.get("pid-eth") is not None


def test_user_stream_handle_message_invokes_manager_account_path():
    manager = _make_manager(["BTCUSDT"])
    assert manager.user_stream is not None

    spy = MagicMock(wraps=manager._on_account_update)
    manager._on_account_update = spy  # type: ignore[method-assign]

    stream = BinanceUserStream(
        binance_api=manager.order_manager.binance_api,
        on_execution_report=manager._on_execution_report,
        on_account_update=manager._on_account_update,
    )

    payload = {
        "e": "ACCOUNT_UPDATE",
        "E": 1710000000000,
        "a": {
            "B": [{"a": "USDT", "wb": "3000", "cw": "2900"}],
            "P": [{"s": "BTCUSDT", "pa": "0", "ep": "0", "up": "0", "ps": "BOTH"}],
        },
    }
    stream._handle_message(json.dumps(payload))

    spy.assert_called_once()
    called = spy.call_args[0][0]
    assert called["wallet_balance"] == 3000.0
    assert (
        manager.listeners["BTCUSDT"]._latest_account_update["wallet_balance"] == 3000.0
    )


def test_user_stream_order_trade_update_routes_to_correct_listener_only():
    manager = _make_manager(["BTCUSDT", "ETHUSDT"])
    btc = manager.listeners["BTCUSDT"]
    eth = manager.listeners["ETHUSDT"]

    btc_m = MagicMock(wraps=btc.on_execution_report)
    eth_m = MagicMock(wraps=eth.on_execution_report)
    btc.on_execution_report = btc_m  # type: ignore[method-assign]
    eth.on_execution_report = eth_m  # type: ignore[method-assign]

    stream = BinanceUserStream(
        binance_api=manager.order_manager.binance_api,
        on_execution_report=manager._on_execution_report,
        on_account_update=manager._on_account_update,
    )

    msg = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1710000001000,
        "o": {
            "s": "ETHUSDT",
            "i": 99,
            "c": "cid1",
            "S": "SELL",
            "o": "MARKET",
            "X": "FILLED",
            "x": "TRADE",
            "l": "0.1",
            "z": "0.1",
            "L": "2000",
            "ap": "2000",
            "T": 1710000001000,
        },
    }
    stream._handle_message(json.dumps(msg))

    btc_m.assert_not_called()
    eth_m.assert_called_once()
