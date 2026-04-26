from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.order_management.grid_execution_adapter import (
    GridExecutionAdapter,
    GridExecutionError,
)
from src.order_management.models import OrderSide, OrderType


def _api(*, hedge_mode: bool = True) -> MagicMock:
    api = MagicMock()
    api.hedge_mode = hedge_mode
    api.place_order.return_value = {
        "order_id": "ex_1",
        "client_order_id": "cg_abc",
        "symbol": "BTCUSDT",
        "status": "open",
    }
    api.cancel_order.return_value = True
    api.get_open_orders.return_value = []
    api.get_positions.return_value = []
    return api


def test_requires_hedge_mode_for_real_execution() -> None:
    with pytest.raises(GridExecutionError, match="requires Binance Hedge Mode"):
        GridExecutionAdapter(_api(hedge_mode=False))


def test_shadow_can_run_without_hedge_mode_when_explicitly_allowed() -> None:
    adapter = GridExecutionAdapter(
        _api(hedge_mode=False), require_hedge_mode=False, shadow=True
    )

    result = adapter.execute_action(
        {
            "action": "place",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.01,
            "price": 99000.0,
            "order_id": "grid_l1",
        }
    )

    assert result.status == "shadow"
    assert result.client_order_id.startswith("cg_")


def test_place_limit_translates_grid_place_action() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": 0.02,
            "price": 101000.0,
            "order_id": "grid_s1",
        }
    )

    api.place_order.assert_called_once()
    kwargs = api.place_order.call_args.kwargs
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["side"] == OrderSide.SELL
    assert kwargs["order_type"] == OrderType.LIMIT
    assert kwargs["quantity"] == 0.02
    assert kwargs["price"] == 101000.0
    assert kwargs["client_order_id"].startswith("cg_")
    assert result.status == "open"


def test_market_exit_uses_reduce_only_opposite_side() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    adapter.execute_action(
        {
            "action": "market_exit",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.03,
            "order_id": "exit_l1",
        }
    )

    kwargs = api.place_order.call_args.kwargs
    assert kwargs["side"] == OrderSide.SELL
    assert kwargs["order_type"] == OrderType.MARKET
    assert kwargs["quantity"] == 0.03
    assert kwargs["reduce_only"] is True


def test_cancel_requires_symbol_and_calls_exchange() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "cancel",
            "symbol": "BTCUSDT",
            "order_id": "ex_123",
        }
    )

    api.cancel_order.assert_called_once_with("ex_123", "BTCUSDT")
    assert result.status == "canceled"


def test_simulation_fill_and_take_profit_are_ignored() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    results = adapter.execute_actions(
        [
            {"action": "fill", "symbol": "BTCUSDT"},
            {"action": "take_profit", "symbol": "BTCUSDT"},
        ]
    )

    assert [r.status for r in results] == [
        "ignored_simulation_event",
        "ignored_simulation_event",
    ]
    api.place_order.assert_not_called()
    api.cancel_order.assert_not_called()


def test_sync_helpers_delegate_to_binance_api() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    assert adapter.sync_open_orders("BTCUSDT") == []
    assert adapter.sync_positions("BTCUSDT") == []
    api.get_open_orders.assert_called_once_with("BTCUSDT")
    api.get_positions.assert_called_once_with("BTCUSDT")
