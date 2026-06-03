from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from src.order_management.grid_execution_adapter import (
    GridExecutionAdapter,
    GridExecutionError,
    MultiLegExecutionAdapter,
    MultiLegExecutionError,
)
from src.order_management.models import OrderSide, OrderType


def _api(*, hedge_mode: bool = True, hedge_mode_probe_error=None) -> MagicMock:
    api = MagicMock()
    api.hedge_mode = hedge_mode
    api.hedge_mode_probe_error = hedge_mode_probe_error
    api.place_order.return_value = {
        "order_id": "ex_1",
        "client_order_id": "cg_abc",
        "symbol": "BTCUSDT",
        "status": "open",
    }
    api.cancel_order.return_value = True
    api.get_open_orders_for_sl_cleanup = None
    api.get_open_orders.return_value = []
    api.get_positions.return_value = []
    return api


def test_requires_hedge_mode_for_real_execution() -> None:
    with pytest.raises(MultiLegExecutionError, match="requires Binance Hedge Mode"):
        GridExecutionAdapter(_api(hedge_mode=False))


def test_probe_failure_is_not_misreported_as_missing_hedge() -> None:
    with pytest.raises(
        MultiLegExecutionError, match="cannot verify Binance USDM hedge mode"
    ):
        GridExecutionAdapter(
            _api(
                hedge_mode=False,
                hedge_mode_probe_error="HTTP 401 ...",
            ),
        )


def test_probe_binance_error_1003_does_not_block_adapter_init() -> None:
    GridExecutionAdapter(
        _api(
            hedge_mode=False,
            hedge_mode_probe_error=(
                "Binance error -1003: Too many requests; current limit "
                "of IP(1.2.3.4) is 2400 requests per minute."
            ),
        ),
    )


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
    assert (result.raw or {}).get("local_order_id") == "grid_s1"


def test_place_marketable_limit_uses_ioc_limit_order() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    adapter.execute_action(
        {
            "action": "place",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.02,
            "price": 101050.0,
            "order_type": "marketable_limit",
            "time_in_force": "IOC",
            "order_id": "dat_l1",
        }
    )

    kwargs = api.place_order.call_args.kwargs
    assert kwargs["side"] == OrderSide.BUY
    assert kwargs["order_type"] == OrderType.LIMIT
    assert kwargs["price"] == 101050.0
    assert kwargs["time_in_force"] == "IOC"


def test_place_market_entry_uses_market_order() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    adapter.execute_action(
        {
            "action": "place",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": 0.02,
            "order_type": "market",
            "order_id": "dat_s1",
        }
    )

    kwargs = api.place_order.call_args.kwargs
    assert kwargs["side"] == OrderSide.SELL
    assert kwargs["order_type"] == OrderType.MARKET
    assert kwargs["price"] is None


def test_multi_leg_adapter_name_is_primary_alias() -> None:
    assert MultiLegExecutionAdapter is GridExecutionAdapter


def test_market_exit_uses_reduce_only_opposite_side() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
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
    assert (result.raw or {}).get("local_order_id") == "exit_l1"


def test_place_stop_loss_protection_uses_explicit_position_side() -> None:
    api = _api()
    adapter = MultiLegExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place_protection",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.03,
            "trigger_price": 99000.0,
            "protection_type": "stop_loss",
            "order_id": "leg_l1_sl",
        }
    )

    kwargs = api.place_order.call_args.kwargs
    assert kwargs["side"] == OrderSide.SELL
    assert kwargs["order_type"] == OrderType.STOP_MARKET
    assert kwargs["quantity"] == 0.03
    assert kwargs["stop_price"] == 99000.0
    assert kwargs["reduce_only"] is True
    assert kwargs["close_position"] is False
    assert kwargs["position_side"] == "LONG"
    assert kwargs["working_type"] == "MARK_PRICE"
    assert kwargs["price_protect"] is True
    assert result.action == "place_protection"


def test_place_take_profit_protection_for_short_uses_post_only_limit_buy() -> None:
    api = _api()
    adapter = MultiLegExecutionAdapter(api)

    adapter.execute_action(
        {
            "action": "place_protection",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "quantity": 0.02,
            "price": 95000.0,
            "trigger_price": 95000.0,
            "order_type": "limit",
            "protection_type": "take_profit",
            "post_only": True,
            "time_in_force": "GTX",
            "order_id": "leg_s1_tp",
        }
    )

    kwargs = api.place_order.call_args.kwargs
    assert kwargs["side"] == OrderSide.BUY
    assert kwargs["order_type"] == OrderType.LIMIT
    assert kwargs["price"] == 95000.0
    assert kwargs["stop_price"] is None
    assert kwargs["reduce_only"] is True
    assert kwargs["position_side"] == "SHORT"
    assert kwargs["working_type"] is None
    assert kwargs["price_protect"] is None
    assert kwargs["post_only"] is True
    assert kwargs["time_in_force"] == "GTX"


def test_place_protection_skips_when_order_already_open() -> None:
    api = _api()
    api.get_order_by_client_id.return_value = {
        "order_id": "ex_existing",
        "client_order_id": "cg_existing",
        "symbol": "BTCUSDT",
        "status": "open",
        "price": 95000.0,
    }
    adapter = MultiLegExecutionAdapter(api)
    action = {
        "action": "place_protection",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "quantity": 0.02,
        "price": 95000.0,
        "trigger_price": 95000.0,
        "order_type": "limit",
        "protection_type": "take_profit",
        "order_id": "leg_s1_tp_supp",
    }

    result = adapter.execute_action(action)

    api.place_order.assert_not_called()
    assert result.status == "open"
    assert result.order_id == "ex_existing"


def test_duplicate_protection_client_id_reuses_live_order() -> None:
    api = _api()
    api.place_order.side_effect = Exception(
        'binance {"code":-4116,"msg":"ClientOrderId is duplicated."}'
    )
    api.get_order_by_client_id.return_value = {
        "order_id": "ex_existing",
        "client_order_id": "cg_existing",
        "symbol": "BTCUSDT",
        "status": "open",
        "price": 95000.0,
    }
    adapter = MultiLegExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place_protection",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "quantity": 0.02,
            "price": 95000.0,
            "trigger_price": 95000.0,
            "order_type": "limit",
            "protection_type": "take_profit",
            "order_id": "leg_s1_tp_supp",
        }
    )

    api.get_order_by_client_id.assert_called_once()
    assert result.status == "open"
    assert result.order_id == "ex_existing"
    assert result.client_order_id == "cg_existing"
    assert (result.raw or {}).get("local_order_id") == "leg_s1_tp_supp"


def test_place_protection_skips_reduce_only_rejected() -> None:
    api = _api()
    api.place_order.side_effect = Exception(
        'binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}'
    )
    adapter = MultiLegExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place_protection",
            "symbol": "BNBUSDT",
            "side": "LONG",
            "quantity": 0.31,
            "price": 643.55,
            "trigger_price": 643.55,
            "order_type": "limit",
            "protection_type": "take_profit",
            "order_id": "BNBUSDT_grid_L1_tp",
        }
    )

    assert result.status == "skipped_no_position"
    assert result.symbol == "BNBUSDT"
    assert (result.raw or {}).get("error", "").find("-2022") >= 0


def test_place_protection_continues_when_lookup_raises_http_400() -> None:
    import requests

    api = _api()
    resp = Mock()
    resp.status_code = 400
    resp.text = ""
    resp.json.side_effect = ValueError("no json")
    api.get_order_by_client_id.side_effect = requests.HTTPError(
        "400 Client Error", response=resp
    )
    api.place_order.return_value = {
        "order_id": "ex_new",
        "client_order_id": "cg_new",
        "symbol": "BNBUSDT",
        "status": "open",
    }
    adapter = MultiLegExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place_protection",
            "symbol": "BNBUSDT",
            "side": "LONG",
            "quantity": 0.31,
            "price": 643.55,
            "trigger_price": 643.55,
            "order_type": "limit",
            "protection_type": "take_profit",
            "order_id": "BNBUSDT_grid_L1_tp",
        }
    )

    api.place_order.assert_called_once()
    assert result.status == "open"
    assert result.order_id == "ex_new"


def test_duplicate_protection_algo_stop_reuses_live_order() -> None:
    api = _api()
    api.place_order.side_effect = Exception(
        'binance {"code":-4116,"msg":"ClientOrderId is duplicated."}'
    )
    api.get_order_by_client_id.return_value = None
    api.get_open_orders_for_sl_cleanup = MagicMock(
        return_value=[
            {
                "order_id": "90535381642",
                "client_order_id": "cg_c6340a24bcce",
                "symbol": "BNBUSDT",
                "status": "new",
                "info": {"clientAlgoId": "cg_c6340a24bcce"},
                "_is_algo_order": True,
            }
        ]
    )
    adapter = MultiLegExecutionAdapter(api)
    action = {
        "action": "place_protection",
        "symbol": "BNBUSDT",
        "side": "LONG",
        "quantity": 0.02,
        "trigger_price": 600.0,
        "protection_type": "stop_loss",
        "order_id": "BNBUSDT_2026-05-19 08:40:00+00:00_S1_sl",
    }
    result = adapter.execute_action(action)

    assert result.status == "new"
    assert result.order_id == "90535381642"
    assert result.client_order_id == "cg_c6340a24bcce"


def test_cancel_requires_symbol_and_calls_exchange() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "cancel",
            "symbol": "BTCUSDT",
            "exchange_order_id": "90489849398",
            "order_id": "BTCUSDT_2026-05-19 08:40:00+00:00_L2",
        }
    )

    api.cancel_order.assert_called_once_with("90489849398", "BTCUSDT")
    assert result.status == "canceled"


def test_place_rejects_below_exchange_min_qty_without_crash() -> None:
    api = _api()
    api.get_symbol_info.return_value = {
        "limits": {
            "amount": {"min": 0.001, "max": None},
            "cost": {"min": 5.0, "max": None},
        }
    }
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "place",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.0003,
            "price": 100000.0,
            "order_id": "BTCUSDT_test_L1",
        }
    )

    api.place_order.assert_not_called()
    assert result.status == "rejected"
    assert "exchange_min_qty" in (result.reason or "")


def test_cancel_skips_local_only_order_id_without_exchange_call() -> None:
    api = _api()
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "cancel",
            "symbol": "BNBUSDT",
            "order_id": "BNBUSDT_2026-05-19 08:40:00+00:00_L2",
            "reason": "regime_or_risk_exit",
        }
    )

    api.cancel_order.assert_not_called()
    assert result.status == "canceled"


def test_cancel_treats_unknown_order_as_already_gone() -> None:
    api = _api()
    api.cancel_order.side_effect = Exception(
        'binance {"code":-2011,"msg":"Unknown order sent."}'
    )
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "cancel",
            "symbol": "BNBUSDT",
            "order_id": "90414533226",
            "reason": "orphan_exchange_order",
        }
    )

    assert result.status == "canceled"


def test_cancel_orphan_algo_order_uses_cancel_algo() -> None:
    api = _api()
    api.cancel_algo_order = MagicMock(return_value=True)
    adapter = GridExecutionAdapter(api)

    result = adapter.execute_action(
        {
            "action": "cancel",
            "symbol": "BNBUSDT",
            "order_id": "2000000972847548",
            "reason": "orphan_exchange_order",
            "is_algo_order": True,
        }
    )

    api.cancel_order.assert_not_called()
    api.cancel_algo_order.assert_called_once_with("2000000972847548", "BNBUSDT")
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
