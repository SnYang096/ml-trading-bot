"""Multi-leg order list / marker reconciliation helpers."""

from __future__ import annotations

import pytest

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


def test_cg_entry_shows_tp_via_leg_id(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_cg_tp",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_live_entry_1",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "place",
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
            "local_order_id": "cg_live_tp_1",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "price": 643.55,
            "status": "open",
            "created_at": "2026-05-21T07:00:22+00:00",
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
    entry = next(r for r in rows if r["order_id"] == "cg_live_entry_1")
    assert entry["take_profit_price"] == 643.55
    assert entry.get("grid_batch") == group
    assert entry.get("leg_label") == "L1"


def test_s1_tp_shows_s1_inventory_when_entry_order_missing(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_s1_inv",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "entry_price": 653.205,
            "quantity": 0.62,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "purpose": "take_profit",
            "price": 643.5545,
            "status": "open",
            "created_at": "2026-05-21T07:00:22+00:00",
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
    s1 = next(r for r in rows if r.get("order_id") == f"{group}_S1")
    assert not any(r.get("order_id") == f"{group}_S1_tp" for r in rows)
    assert s1["purpose"] == "inventory"
    assert s1["leg_label"] == "S1"
    assert s1["status"] == "filled"
    assert s1.get("take_profit_price") == 643.5545


def test_open_s_grid_leg_does_not_duplicate_tp_price(multi_leg_db):
    from mlbot_console.services.multileg_order_links import resolve_take_profit_display

    row = {
        "order_id": "BNBUSDT_2026-05-19 08:40:00+00:00_S1",
        "purpose": "entry",
        "side": "SELL",
        "price": 643.5545,
        "status": "open",
    }
    tp_px, hint = resolve_take_profit_display(row)
    assert tp_px is None
    assert hint == ""


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


def test_l1_entry_shows_repair_tp_as_l1_tp(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_repair_tp",
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
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 652.67,
            "client_order_id": "cg_repair_long_tp2",
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
    assert not any(r["order_id"].endswith("_L1_tp") for r in rows)
    assert l1["take_profit_price"] == 652.67
    assert "补挂" in (l1.get("take_profit_hint") or "")
    expected = pytest.approx((652.67 - 637.11) * 0.31, rel=1e-4)
    assert l1["pnl_usdt"] == expected


def test_l1_closed_pair_shows_realized_pnl_on_entry_and_tp(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_l1_pnl",
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
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 652.67,
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
    assert not any(r["order_id"].endswith("_L1_tp") for r in rows)
    expected = pytest.approx((652.67 - 637.11) * 0.31, rel=1e-4)
    assert l1["pnl_usdt"] == expected
    assert l1.get("pnl_hint") == "已实现"


def test_open_short_inventory_shows_unrealized_pnl(multi_leg_db):
    from pathlib import Path
    from unittest.mock import patch

    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_s1_upnl",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "entry_price": 653.205,
            "quantity": 0.62,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "purpose": "take_profit",
            "price": 643.55,
            "status": "open",
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
        feature_bus_root=None,
    )
    s1 = next(r for r in rows if r.get("order_id") == f"{group}_S1")
    assert s1.get("pnl_usdt") is None

    with patch(
        "mlbot_console.services.account_summary.latest_close_prices",
        return_value={"BNBUSDT": 640.0},
    ):
        rows2 = collect_orders(
            trend_db=multi_leg_db.parent / "missing_trend.db",
            spot_db=multi_leg_db.parent / "missing_spot.db",
            multi_leg_db=multi_leg_db,
            symbol="BNBUSDT",
            scopes=["multi_leg"],
            exclude_statuses=["expired", "canceled"],
            limit=50,
            feature_bus_root=Path("/tmp"),
        )
    s1b = next(r for r in rows2 if r.get("order_id") == f"{group}_S1")
    assert s1b["pnl_usdt"] == pytest.approx((653.205 - 640.0) * 0.62, rel=1e-4)
    assert s1b.get("pnl_hint") == "浮盈"


def test_s2_filled_without_tp_shows_inferred_missing_tp(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_s2_missing_tp",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.31,
            "price": 649.99,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 649.99,
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "purpose": "take_profit",
            "quantity": 0.31,
            "price": 643.55,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.31,
            "price": 656.42,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 656.42,
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
    assert not any(r["order_id"].endswith("_S1_tp") for r in rows)
    assert not any(r["order_id"].endswith("_S2_tp") for r in rows)
    s2 = next(r for r in rows if r["order_id"].endswith("_S2"))
    assert s2["take_profit_price"] == pytest.approx(649.98, abs=0.02)
    assert "未挂止盈" in (s2.get("take_profit_hint") or "")
