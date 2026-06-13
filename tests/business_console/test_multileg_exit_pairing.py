"""Unified multi-leg entry→exit pairing (TP / SL / market_exit / regime)."""

from __future__ import annotations

import pytest

from mlbot_console.services.multileg_leg_pnl import (
    _order_key,
    exit_kind_for_multileg_row,
    pair_multileg_entry_exits,
)
from mlbot_console.services.trade_links import multi_leg_trade_links


def test_pair_multileg_covers_trend_regime_and_chop_tp(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid", "trend_scalp"],
        symbols=["BNBUSDT"],
        run_id="pair_mix",
    )
    group = "BNBUSDT_grid"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 650.0,
            "filled_at": "2026-06-05 10:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_sl",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "stop_loss",
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 655.0,
            "filled_at": "2026-06-05 11:00:00+00:00",
        }
    )
    segment = "BNBUSDT_2026-06-05 12:00:00+00:00"
    entry_id = f"{segment}_initial_trend_BUY_0_0"
    exit_id = f"{entry_id}_fill0_exit_regime_exit_2026-06-05 13:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 640.0,
            "filled_at": "2026-06-05 12:05:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "BNBUSDT",
            "side": "SELL",
            "position_side": "LONG",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 635.0,
            "filled_at": "2026-06-05 13:00:00+00:00",
        }
    )

    from mlbot_console.services.multileg_leg_pnl import _MULTILEG_ORDER_SQL
    from mlbot_console.services.db import query_rows
    from mlbot_console.services.multileg_order_links import hydrate_multileg_fill_fields

    rows = list(query_rows(multi_leg_db, _MULTILEG_ORDER_SQL, ("BNBUSDT",)))
    for row in rows:
        hydrate_multileg_fill_fields(row)

    pairs = pair_multileg_entry_exits(rows)
    assert len(pairs) == 2
    kinds = {exit_kind_for_multileg_row(ex) for _en, ex in pairs}
    assert kinds == {"stop_loss", "regime_exit"}

    links, _ = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 2
    chop = next(lk for lk in links if lk["strategy"] == "chop_grid")
    trend = next(lk for lk in links if lk["strategy"] == "trend_scalp")
    assert chop["exit_kind"] == "stop_loss"
    assert chop["pnl_usdt"] == pytest.approx((650.0 - 655.0) * 0.5, rel=1e-4)
    assert trend["exit_kind"] == "regime_exit"
    assert trend["pnl_usdt"] == pytest.approx((635.0 - 640.0) * 0.1, rel=1e-4)


def test_chop_grid_shared_hedge_sl_does_not_cross_pair_same_stop(multi_leg_db) -> None:
    """L3 must not inherit L2_sl when both legs share one hedge LONG slot."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["HYPEUSDT"],
        run_id="hype_l3",
    )
    seg = "HYPEUSDT_2026-06-13 00:45:00+00:00"
    for lid, px, ts in (
        (f"{seg}_L2", 58.853, "2026-06-13 03:22:15+00:00"),
        (f"{seg}_L3", 58.657, "2026-06-13 03:22:34+00:00"),
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": lid,
                "leg_id": lid,
                "symbol": "HYPEUSDT",
                "side": "BUY",
                "purpose": "entry",
                "status": "filled",
                "filled_quantity": 82.0,
                "average_price": px,
                "filled_at": ts,
            }
        )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{seg}_L1_sl",
            "leg_id": f"{seg}_L1",
            "symbol": "HYPEUSDT",
            "side": "SELL",
            "purpose": "stop_loss",
            "status": "filled",
            "filled_quantity": 82.0,
            "average_price": 58.26,
            "filled_at": "2026-06-13 04:26:29+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{seg}_L2_sl",
            "leg_id": f"{seg}_L2",
            "symbol": "HYPEUSDT",
            "side": "SELL",
            "purpose": "stop_loss",
            "status": "filled",
            "filled_quantity": 82.0,
            "average_price": 58.042,
            "filled_at": "2026-06-13 04:57:30+00:00",
        }
    )

    from mlbot_console.services.multileg_leg_pnl import _MULTILEG_ORDER_SQL
    from mlbot_console.services.db import query_rows
    from mlbot_console.services.multileg_order_links import hydrate_multileg_fill_fields

    rows = list(query_rows(multi_leg_db, _MULTILEG_ORDER_SQL, ("HYPEUSDT",)))
    for row in rows:
        hydrate_multileg_fill_fields(row)

    pairs = pair_multileg_entry_exits(rows)
    by_entry = {_order_key(en): ex for en, ex in pairs}
    l2_exit = by_entry[f"{seg}_L2"]
    l3_exit = by_entry[f"{seg}_L3"]
    assert _order_key(l2_exit) == f"{seg}_L2_sl"
    assert _order_key(l3_exit) == f"{seg}_L1_sl"

    links, _ = multi_leg_trade_links(multi_leg_db, "HYPEUSDT")
    l3_link = next(lk for lk in links if lk["leg"] == "L3")
    assert l3_link["exit_marker_id"].endswith("_L1_sl")
    assert l3_link["exit_marker_id"] != l3_link["entry_marker_id"].replace(
        "_L3", "_L2_sl"
    )


def test_shared_stop_loss_skips_early_sl_for_later_orphan(multi_leg_db) -> None:
    """First valid SL goes to earlier orphan; later orphan stays open."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["ETHUSDT"],
        run_id="shared_sl_zip",
    )
    seg = "ETHUSDT_2026-06-01 00:00:00+00:00"
    for lid, px, ts in (
        (f"{seg}_L1", 3000.0, "2026-06-01 10:00:00+00:00"),
        (f"{seg}_L2", 2980.0, "2026-06-01 12:00:00+00:00"),
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": lid,
                "leg_id": lid,
                "symbol": "ETHUSDT",
                "side": "BUY",
                "purpose": "entry",
                "status": "filled",
                "filled_quantity": 1.0,
                "average_price": px,
                "filled_at": ts,
            }
        )
    for oid, leg, ts in (
        (f"{seg}_L5_sl", f"{seg}_L5", "2026-06-01 09:00:00+00:00"),
        (f"{seg}_L6_sl", f"{seg}_L6", "2026-06-01 11:00:00+00:00"),
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": oid,
                "leg_id": leg,
                "symbol": "ETHUSDT",
                "side": "SELL",
                "purpose": "stop_loss",
                "status": "filled",
                "filled_quantity": 1.0,
                "average_price": 2950.0,
                "filled_at": ts,
            }
        )

    from mlbot_console.services.multileg_leg_pnl import _MULTILEG_ORDER_SQL
    from mlbot_console.services.db import query_rows
    from mlbot_console.services.multileg_order_links import hydrate_multileg_fill_fields

    rows = list(query_rows(multi_leg_db, _MULTILEG_ORDER_SQL, ("ETHUSDT",)))
    for row in rows:
        hydrate_multileg_fill_fields(row)

    pairs = pair_multileg_entry_exits(rows)
    by_entry = {_order_key(en): ex for en, ex in pairs}
    assert _order_key(by_entry[f"{seg}_L1"]) == f"{seg}_L6_sl"
    assert f"{seg}_L2" not in by_entry


def test_shared_stop_loss_requires_closing_side(multi_leg_db) -> None:
    """LONG entry must not pair to a BUY stop (closes SHORT only)."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BTCUSDT"],
        run_id="shared_sl_side",
    )
    seg = "BTCUSDT_2026-06-01 00:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{seg}_L1",
            "leg_id": f"{seg}_L1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 1.0,
            "average_price": 60000.0,
            "filled_at": "2026-06-01 10:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{seg}_L9_sl",
            "leg_id": f"{seg}_L9",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "purpose": "stop_loss",
            "status": "filled",
            "filled_quantity": 1.0,
            "average_price": 61000.0,
            "filled_at": "2026-06-01 11:00:00+00:00",
        }
    )

    from mlbot_console.services.multileg_leg_pnl import _MULTILEG_ORDER_SQL
    from mlbot_console.services.db import query_rows
    from mlbot_console.services.multileg_order_links import hydrate_multileg_fill_fields

    rows = list(query_rows(multi_leg_db, _MULTILEG_ORDER_SQL, ("BTCUSDT",)))
    for row in rows:
        hydrate_multileg_fill_fields(row)

    pairs = pair_multileg_entry_exits(rows)
    assert pairs == []
