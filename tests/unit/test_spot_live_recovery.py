from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.order_management.spot_live_recovery import (
    apply_buy_fill_to_position,
    effective_symbol_deployed,
    has_blocking_pending_buy,
    mark_pending_fill_recorded,
    merge_rebuilt_deploy_into_positions,
    new_position_shell,
    normalize_spot_symbol,
    pending_fill_delta,
    pending_buy_age_hours,
    rebuild_positions_from_filled_orders,
    set_pending_buy,
)


def test_effective_deploy_includes_pending_reserved() -> None:
    pos = new_position_shell("BTCUSDT", profit_take_ladder_cfg={"enabled": True})
    pos["_spot_quote_deployed"] = 100.0
    set_pending_buy(
        pos,
        local_order_id="local1",
        exchange_order_id="ex1",
        client_order_id="sa_1",
        quantity=0.01,
        price=50000.0,
        quote_reserved=250.0,
        placed_at="2026-01-01T00:00:00+00:00",
    )
    assert effective_symbol_deployed(pos) == pytest.approx(350.0)
    assert has_blocking_pending_buy(pos)


def test_rebuild_positions_from_filled_buys() -> None:
    ladder = {"enabled": True, "min_profit_multiple": 5.0}
    orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "status": "closed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "quantity": 0.01,
            "price": 40000.0,
            "raw_json": '{"status":"closed","filled":0.01,"cost":400.0}',
        },
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "status": "closed",
            "created_at": "2026-01-02T00:00:00+00:00",
            "quantity": 0.01,
            "price": 50000.0,
            "raw_json": '{"status":"closed","filled":0.01,"cost":500.0}',
        },
    ]
    positions = rebuild_positions_from_filled_orders(
        orders, symbols=["BTCUSDT"], profit_take_ladder_cfg=ladder
    )
    assert positions["BTCUSDT"]["_qty_base"] == pytest.approx(0.02)
    assert positions["BTCUSDT"]["_spot_quote_deployed"] == pytest.approx(900.0)
    assert positions["BTCUSDT"]["profit_take_ladder"]["enabled"] is True


def test_merge_rebuilt_into_empty_local() -> None:
    ladder = {"enabled": True}
    rebuilt = rebuild_positions_from_filled_orders(
        [
            {
                "symbol": "SOLUSDT",
                "side": "buy",
                "status": "filled",
                "created_at": "2026-01-01T00:00:00+00:00",
                "quantity": 1.0,
                "price": 100.0,
                "raw_json": '{"status":"filled","filled":1.0,"cost":100.0}',
            }
        ],
        symbols=["SOLUSDT"],
        profit_take_ladder_cfg=ladder,
    )
    local: dict = {}
    merge_rebuilt_deploy_into_positions(local, rebuilt, profit_take_ladder_cfg=ladder)
    assert local["SOLUSDT"]["_spot_quote_deployed"] == pytest.approx(100.0)


def test_pending_buy_age_hours() -> None:
    pending = {"placed_at": "2026-01-01T00:00:00+00:00"}
    now = datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc)
    assert pending_buy_age_hours(pending, now=now) == pytest.approx(25.0)


def test_apply_buy_fill_accumulates() -> None:
    pos = new_position_shell("BNBUSDT", profit_take_ladder_cfg={})
    q1 = apply_buy_fill_to_position(
        pos, fill_qty=0.5, fill_quote_usdt=100.0, profit_take_ladder_cfg={}
    )
    q2 = apply_buy_fill_to_position(
        pos, fill_qty=0.5, fill_quote_usdt=50.0, profit_take_ladder_cfg={}
    )
    assert q1 == pytest.approx(100.0)
    assert q2 == pytest.approx(50.0)
    assert pos["_qty_base"] == pytest.approx(1.0)
    assert pos["_spot_quote_deployed"] == pytest.approx(150.0)


def test_pending_fill_delta_only_applies_new_fill() -> None:
    pos = new_position_shell("BTCUSDT", profit_take_ladder_cfg={})
    set_pending_buy(
        pos,
        local_order_id="local1",
        exchange_order_id="ex1",
        client_order_id="sa_1",
        quantity=0.02,
        price=50000.0,
        quote_reserved=1000.0,
        placed_at="2026-01-01T00:00:00+00:00",
    )
    pending = pos["_pending_buy"]
    delta_qty, delta_quote = pending_fill_delta(
        pending, filled_qty=0.01, filled_quote=500.0
    )
    assert delta_qty == pytest.approx(0.01)
    assert delta_quote == pytest.approx(500.0)

    mark_pending_fill_recorded(pending, filled_qty=0.01, filled_quote=500.0)
    delta_qty, delta_quote = pending_fill_delta(
        pending, filled_qty=0.01, filled_quote=500.0
    )
    assert delta_qty == pytest.approx(0.0)
    assert delta_quote == pytest.approx(0.0)
    assert effective_symbol_deployed(pos) == pytest.approx(500.0)


def test_normalize_spot_symbol_from_ccxt() -> None:
    assert normalize_spot_symbol("BTC/USDT") == "BTCUSDT"
    assert normalize_spot_symbol("SOL/USDT:USDT") == "SOLUSDT"
