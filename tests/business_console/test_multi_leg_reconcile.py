"""Multi-leg order list / marker reconciliation helpers."""

from __future__ import annotations

from mlbot_console.services.orders_list import collect_orders, multi_leg_orders_list
from mlbot_console.services.trade_markers import (
    _multi_leg_event,
    _multi_leg_take_profit_price,
    collect_markers,
    multi_leg_markers,
)


def test_take_profit_maps_to_tp_not_exit():
    assert _multi_leg_event("take_profit", "") == "tp"
    assert _multi_leg_event("entry", "TAKE_PROFIT_MARKET") == "tp"
    assert (
        _multi_leg_event("place", "LIMIT", local_order_id="g_L2", is_filled=False)
        == "grid"
    )
    assert (
        _multi_leg_event("place", "LIMIT", local_order_id="g_L1", is_filled=True)
        == "entry"
    )
    assert _multi_leg_event("market_exit", "MARKET") == "exit"


def test_l1_entry_shows_tp_from_l1_tp_protection_row(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    from mlbot_console.services.orders_list import collect_orders

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_l1_tp",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.31,
            "price": 637.11,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 637.11,
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1_tp",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "LONG",
            "purpose": "take_profit",
            "quantity": 0.31,
            "price": 643.55,
            "status": "open",
            "filled_quantity": 0.0,
        }
    )
    rows = collect_orders(
        trend_db=multi_leg_db.parent / "missing_trend.db",
        spot_db=multi_leg_db.parent / "missing_spot.db",
        multi_leg_db=multi_leg_db,
        symbol="BNBUSDT",
        scopes=["multi_leg"],
        limit=50,
    )
    l1 = next(r for r in rows if r["order_id"].endswith("_L1"))
    assert l1["take_profit_price"] == 643.55
    assert "tp" in (l1.get("take_profit_hint") or "").lower()


def test_l1_entry_does_not_use_s_grid_as_tp(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    from mlbot_console.services.multileg_order_links import resolve_take_profit_display
    from mlbot_console.services.orders_list import collect_orders

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_l1_s1",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.31,
            "price": 637.11,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 637.11,
            "filled_at": "2026-05-19T14:48:21+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.31,
            "price": 656.42,
            "status": "expired",
            "filled_quantity": 0.0,
        }
    )
    rows = collect_orders(
        trend_db=multi_leg_db.parent / "missing_trend.db",
        spot_db=multi_leg_db.parent / "missing_spot.db",
        multi_leg_db=multi_leg_db,
        symbol="BNBUSDT",
        scopes=["multi_leg"],
        exclude_statuses=["expired", "canceled"],
        limit=50,
    )
    l1 = next(r for r in rows if r["order_id"].endswith("_L1"))
    assert l1.get("take_profit_price") is None or l1.get("take_profit_price") != 656.42


def test_take_profit_price_from_stop_or_short_leg(multi_leg_db):
    row = {
        "purpose": "take_profit",
        "order_type": "TAKE_PROFIT_MARKET",
        "stop_price": 649.99,
        "price": 649.99,
        "side": "SELL",
    }
    assert _multi_leg_take_profit_price(row) == 649.99

    short_leg = {
        "local_order_id": "BNBUSDT_2026-05-19 08:40:00+00:00_S1",
        "purpose": "entry",
        "side": "SELL",
        "price": 656.42,
        "stop_price": None,
    }
    assert _multi_leg_take_profit_price(short_leg) == 656.42


def test_filled_take_profit_marker_is_exit(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_bnb_test",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "bnb_tp_sell",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "order_type": "TAKE_PROFIT_MARKET",
            "quantity": 0.31,
            "price": 649.99,
            "stop_price": 649.99,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 649.99,
            "filled_at": "2026-05-20T14:52:40+00:00",
        }
    )
    markers = multi_leg_markers(multi_leg_db, "BNBUSDT")
    tp = [m for m in markers if m.get("detail", {}).get("purpose") == "take_profit"]
    assert tp
    assert tp[0]["event"] == "tp"
    assert tp[0]["side"] == "short"


def test_exclude_expired_canceled_from_list(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_bnb_filter",
    )
    for lid, st in (
        ("bnb_exp", "expired"),
        ("bnb_can", "canceled"),
        ("bnb_fill", "filled"),
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": lid,
                "symbol": "BNBUSDT",
                "side": "SELL",
                "purpose": "entry",
                "quantity": 0.1,
                "price": 650.0,
                "status": st,
                "filled_quantity": 0.1 if st == "filled" else 0.0,
                "filled_at": "2026-05-19T12:00:00+00:00" if st == "filled" else None,
                "created_at": "2026-05-19T12:00:00+00:00",
            }
        )
    all_rows = multi_leg_orders_list(multi_leg_db, "BNBUSDT", limit=50)
    statuses = {r["order_id"]: r["status"] for r in all_rows}
    assert "bnb_exp" in statuses
    filtered = collect_orders(
        trend_db=multi_leg_db.parent / "missing_trend.db",
        spot_db=multi_leg_db.parent / "missing_spot.db",
        multi_leg_db=multi_leg_db,
        symbol="BNBUSDT",
        scopes=["multi_leg"],
        exclude_statuses=["expired", "canceled"],
        limit=50,
    )
    fids = {r["order_id"] for r in filtered}
    assert "bnb_fill" in fids
    assert "bnb_exp" not in fids
    assert "bnb_can" not in fids


def test_expired_short_not_on_chart_by_default(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_bnb_chart",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "bnb_s1_exp",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.1,
            "price": 656.0,
            "status": "expired",
            "filled_quantity": 0.0,
            "created_at": "2026-05-19T08:40:00+00:00",
        }
    )
    markers = collect_markers(
        trend_db=multi_leg_db.parent / "missing_trend.db",
        spot_db=multi_leg_db.parent / "missing_spot.db",
        multi_leg_db=multi_leg_db,
        symbol="BNBUSDT",
        scopes=["multi_leg"],
        include_pending=False,
    )
    assert not any(
        m.get("detail", {}).get("local_order_id") == "bnb_s1_exp" for m in markers
    )
