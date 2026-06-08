"""Unit tests for multi-leg per-order PnL enrichment."""

from __future__ import annotations

import pytest

from mlbot_console.services.account_summary import build_order_pnl_maps
from mlbot_console.services.multileg_leg_pnl import (
    attach_multileg_display_pnl,
    multileg_pnl_by_order_id,
)
from mlbot_console.services.orders_list import enrich_orders_pnl


def _seed_l1_closed(storage, *, repair_client: str = "") -> str:
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="ml_pnl_unit",
    )
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
    tp_payload = {
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
    if repair_client:
        tp_payload["client_order_id"] = repair_client
    storage.upsert_order(tp_payload)
    return group


def test_multileg_pnl_by_order_id_maps_entry_and_exit(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    group = _seed_l1_closed(storage)
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "BNBUSDT")
    expected = pytest.approx(4.8236, rel=1e-3)
    assert pnl_map[f"{group}_L1"]["pnl_usdt"] == expected
    assert pnl_map[f"{group}_L1_tp"]["pnl_usdt"] == expected
    assert pnl_map[f"{group}_L1"]["pnl_hint"] == "已实现"
    assert pnl_map[f"{group}_L1_tp"]["realized_pnl"] == expected


def test_multileg_pnl_repair_tp_client_id(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    group = _seed_l1_closed(storage, repair_client="cg_repair_long_tp2")
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "BNBUSDT")
    assert f"{group}_L1" in pnl_map
    assert pnl_map[f"{group}_L1"]["pnl_usdt"] == pytest.approx(4.8236, rel=1e-3)


def test_multileg_short_closed_pair_pnl(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="ml_s_pnl",
    )
    group = "BNBUSDT_2026-05-20 12:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.5,
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 700.0,
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
            "quantity": 0.5,
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 680.0,
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "BNBUSDT")
    assert pnl_map[f"{group}_S1"]["pnl_usdt"] == pytest.approx(10.0, rel=1e-4)
    assert pnl_map[f"{group}_S1_tp"]["pnl_usdt"] == pytest.approx(10.0, rel=1e-4)


def test_build_order_pnl_maps_includes_multileg_legs(
    multi_leg_db, trend_db, spot_db
) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    group = _seed_l1_closed(storage)
    _, _, ml_map = build_order_pnl_maps(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="BNBUSDT",
    )
    assert ml_map[f"{group}_L1"]["pnl_hint"] == "已实现"
    assert ml_map[f"{group}_L1_tp"]["pnl_usdt"] == pytest.approx(4.8236, rel=1e-3)


def test_enrich_orders_pnl_without_feature_bus_still_realizes_multileg(
    multi_leg_db, trend_db, spot_db
) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    group = _seed_l1_closed(storage)
    rows = [
        {
            "scope": "multi_leg",
            "order_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 637.11,
            "purpose": "entry",
            "leg_id": f"{group}_L1",
        }
    ]
    enrich_orders_pnl(
        rows,
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=None,
        symbol="BNBUSDT",
    )
    assert rows[0]["pnl_usdt"] == pytest.approx(4.8236, rel=1e-3)


def test_attach_multileg_display_pnl_on_synthetic_inventory(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="ml_inv_pnl",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "purpose": "take_profit",
            "price": 643.0,
            "status": "open",
        }
    )
    inv_row = {
        "scope": "multi_leg",
        "order_id": f"{group}_S1",
        "symbol": "BNBUSDT",
        "side": "SHORT",
        "status": "filled",
        "purpose": "inventory",
        "order_type": "inventory_leg",
        "filled_quantity": 0.62,
        "average_price": 653.205,
        "leg_id": f"{group}_S1",
        "strategy": "chop_grid",
    }
    attach_multileg_display_pnl(
        [inv_row],
        db_path=multi_leg_db,
        symbol="BNBUSDT",
        mark_prices={"BNBUSDT": 640.0},
    )
    assert inv_row["pnl_usdt"] == pytest.approx((653.205 - 640.0) * 0.62, rel=1e-4)
    assert inv_row["pnl_hint"] == "浮盈"


def test_multileg_pnl_orphan_market_exit_realized(multi_leg_db) -> None:
    import json

    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["XRPUSDT"],
        run_id="ml_pnl_orphan_me",
    )
    group = "XRPUSDT_2026-06-04 02:55:53.371955+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L2",
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 65.1,
            "average_price": 1.1748,
            "filled_at": "2026-06-04 07:20:11+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_b3f1c92377fb",
            "symbol": "XRPUSDT",
            "side": "LONG",
            "position_side": "LONG",
            "purpose": "market_exit",
            "status": "closed",
            "quantity": 65.1,
            "created_at": "2026-06-04 11:11:04+00:00",
            "raw": json.dumps({"filled": 65.1, "average_price": 1.1419}),
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "XRPUSDT")
    expected = pytest.approx((1.1419 - 1.1748) * 65.1, rel=1e-4)
    assert pnl_map[f"{group}_L2"]["pnl_usdt"] == expected
    assert pnl_map[f"{group}_L2"]["pnl_hint"] == "已实现"
