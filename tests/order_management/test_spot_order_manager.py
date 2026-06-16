"""SpotOrderManager unit tests (mock API + in-memory SQLite, no exchange)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.order_management.spot_order_manager import SpotOrderManager


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "spot_orders.db")


def test_shadow_place_order_persists_without_api(db_path: str) -> None:
    om = SpotOrderManager(db_path=db_path, api=None, shadow=True)

    result = om.place_order(
        symbol="ethusdt",
        side="buy",
        order_type="market",
        quantity=0.01,
    )

    assert result.status == "shadow"
    assert result.symbol == "ETHUSDT"
    assert result.side == "buy"
    assert result.exchange_order_id is None
    assert result.client_order_id.startswith("sa_")

    rows = om.list_orders_for_symbols(["ETHUSDT"])
    assert len(rows) == 1
    assert rows[0]["status"] == "shadow"
    assert rows[0]["quantity"] == pytest.approx(0.01)


def test_live_place_order_calls_api_and_stores_exchange_id(db_path: str) -> None:
    api = MagicMock()
    api.place_order.return_value = {
        "id": "ex_spot_99",
        "status": "FILLED",
        "filled": 0.01,
        "cost": 30.0,
    }
    om = SpotOrderManager(db_path=db_path, api=api, shadow=False)

    result = om.place_order(
        symbol="ETHUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.01,
    )

    assert result.status == "filled"
    assert result.exchange_order_id == "ex_spot_99"
    api.place_order.assert_called_once()
    assert api.place_order.call_args.kwargs["client_order_id"].startswith("sa_")

    rows = om.list_orders_for_symbols(["ETHUSDT"])
    assert rows[0]["exchange_order_id"] == "ex_spot_99"
    assert rows[0]["filled_quantity"] == pytest.approx(0.01)
    assert rows[0]["filled_quote_usdt"] == pytest.approx(30.0)


def test_live_place_order_requires_api(db_path: str) -> None:
    om = SpotOrderManager(db_path=db_path, api=None, shadow=False)
    with pytest.raises(RuntimeError, match="spot api not initialized"):
        om.place_order(
            symbol="ETHUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.01,
        )


def test_update_order_record_and_find_order_id(db_path: str) -> None:
    om = SpotOrderManager(db_path=db_path, api=None, shadow=True)
    placed = om.place_order(
        symbol="BTCUSDT",
        side="buy",
        order_type="limit",
        quantity=0.001,
        price=50_000.0,
    )

    om.update_order_record(
        placed.order_id,
        status="filled",
        filled_quantity=0.001,
        filled_quote_usdt=50.0,
    )

    found = om.find_order_id(client_order_id=placed.client_order_id)
    assert found == placed.order_id

    rows = om.list_orders_for_symbols(["BTCUSDT"], sides=["buy"])
    assert rows[0]["status"] == "filled"
    assert rows[0]["filled_quantity"] == pytest.approx(0.001)


def test_cancel_exchange_order_shadow_noop(db_path: str) -> None:
    om = SpotOrderManager(db_path=db_path, api=None, shadow=True)
    out = om.cancel_exchange_order("BTCUSDT", "ex_1")
    assert out == {"status": "shadow"}
