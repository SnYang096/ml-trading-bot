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


def _seed_trend_scalp_segment(
    storage,
    *,
    segment: str,
    side: str,
    entry_qty: float,
    entry_px: float,
    exit_px: float,
    reason: str = "initial_trend",
    seq: int = 0,
) -> tuple[str, str]:
    entry_id = f"{segment}_{reason}_{side}_{seq}_0"
    exit_id = f"{entry_id}_fill0_exit_regime_exit_2026-06-08 00:05:00+00:00"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=[segment.split("_", 1)[0]],
        run_id=f"ts_{segment[-8:]}",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": segment.split("_", 1)[0],
            "side": side,
            "purpose": "entry",
            "status": "closed",
            "filled_quantity": entry_qty,
            "average_price": entry_px,
            "created_at": "2026-06-06 14:00:40",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": segment.split("_", 1)[0],
            "side": "SHORT" if side == "SELL" else "LONG",
            "position_side": "SHORT" if side == "SELL" else "LONG",
            "purpose": "market_exit",
            "status": "closed",
            "filled_quantity": entry_qty,
            "average_price": exit_px,
            "created_at": "2026-06-06 16:01:53",
        }
    )
    return entry_id, exit_id


def test_multileg_pnl_trend_scalp_short_segment(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "ETHUSDT_2026-06-06 14:00:27.526643+00:00"
    entry_id, exit_id = _seed_trend_scalp_segment(
        storage,
        segment=segment,
        side="SELL",
        entry_qty=0.048,
        entry_px=1685.0,
        exit_px=1680.0,
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "ETHUSDT")
    expected = pytest.approx((1685.0 - 1680.0) * 0.048, rel=1e-4)
    assert pnl_map[entry_id]["pnl_usdt"] == expected
    assert pnl_map[exit_id]["pnl_usdt"] == expected
    assert pnl_map[entry_id]["pnl_hint"] == "已实现"


def test_multileg_pnl_trend_scalp_dual_add_long_segment(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "ETHUSDT_2026-06-07 22:19:47.612033+00:00"
    sym = "ETHUSDT"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=[sym],
        run_id="ts_dual_add",
    )
    initial_id = f"{segment}_initial_trend_BUY_0_0"
    add_id = f"{segment}_trend_add_BUY_1_0"
    initial_exit = f"{initial_id}_fill0_exit_regime_exit_2026-06-08 00:05:00+00:00"
    add_exit = f"{add_id}_fill1_exit_regime_exit_2026-06-08 00:05:00+00:00"
    for payload in (
        {
            "local_order_id": initial_id,
            "side": "BUY",
            "filled_quantity": 0.126,
            "average_price": 1685.77,
            "created_at": "2026-06-07 22:19:58",
        },
        {
            "local_order_id": add_id,
            "side": "BUY",
            "filled_quantity": 0.124,
            "average_price": 1686.0,
            "created_at": "2026-06-07 23:05:27",
        },
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "trend_scalp",
                "symbol": sym,
                "purpose": "entry",
                "status": "closed",
                **payload,
            }
        )
    for payload in (
        {
            "local_order_id": initial_exit,
            "filled_quantity": 0.126,
            "average_price": 1682.83,
            "created_at": "2026-06-08 00:06:07",
        },
        {
            "local_order_id": add_exit,
            "filled_quantity": 0.124,
            "average_price": 1682.83,
            "created_at": "2026-06-08 00:06:08",
        },
    ):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "trend_scalp",
                "symbol": sym,
                "side": "LONG",
                "position_side": "LONG",
                "purpose": "market_exit",
                "status": "closed",
                **payload,
            }
        )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, sym)
    assert pnl_map[initial_id]["pnl_usdt"] == pytest.approx(
        (1682.83 - 1685.77) * 0.126, rel=1e-4
    )
    assert pnl_map[add_id]["pnl_usdt"] == pytest.approx(
        (1682.83 - 1686.0) * 0.124, rel=1e-4
    )
    assert pnl_map[initial_exit]["pnl_usdt"] == pnl_map[initial_id]["pnl_usdt"]
    assert pnl_map[add_exit]["pnl_usdt"] == pnl_map[add_id]["pnl_usdt"]


def test_multileg_pnl_trend_scalp_late_fixup_links_entry_and_exit(multi_leg_db) -> None:
    """market_exit_late_fixup must pair with trend entry (no double realized+floating)."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "XRPUSDT_2026-06-05 06:27:23.559332+00:00"
    entry_id = f"{segment}_initial_trend_SELL_0_0"
    exit_id = f"{segment}_market_exit_late_fixup"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=["XRPUSDT"],
        run_id="ts_late_fixup",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "XRPUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 69.1,
            "average_price": 1.1027,
            "created_at": "2026-06-05 06:27:16+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 69.3,
            "average_price": 1.0201,
            "created_at": "2026-06-05 06:27:24+00:00",
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "XRPUSDT")
    expected = pytest.approx((1.1027 - 1.0201) * 69.1, rel=1e-4)
    assert entry_id in pnl_map
    assert exit_id in pnl_map
    assert pnl_map[entry_id]["pnl_hint"] == "已实现"
    assert pnl_map[entry_id]["pnl_usdt"] == expected
    assert pnl_map[exit_id]["pnl_usdt"] == expected
    assert pnl_map[entry_id].get("unrealized_pnl") is None


def test_multileg_pnl_trend_scalp_late_fixup_long_segment(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "XRPUSDT_2026-06-07 09:12:20.523263+00:00"
    entry_id = f"{segment}_initial_trend_BUY_0_0"
    exit_id = f"{segment}_market_exit_late_fixup"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=["XRPUSDT"],
        run_id="ts_late_fixup_long",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 64.3,
            "average_price": 1.1628,
            "created_at": "2026-06-07 09:12:34+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "XRPUSDT",
            "side": "SELL",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 64.3,
            "average_price": 1.1546,
            "created_at": "2026-06-07 09:12:21+00:00",
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "XRPUSDT")
    expected = pytest.approx((1.1546 - 1.1628) * 64.3, rel=1e-4)
    assert pnl_map[entry_id]["pnl_usdt"] == expected
    assert pnl_map[exit_id]["pnl_usdt"] == expected
    assert pnl_map[entry_id]["pnl_hint"] == "已实现"


def test_multileg_pnl_late_fixup_does_not_cross_segments(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    old_seg = "XRPUSDT_2026-06-04 10:00:00+00:00"
    new_seg = "XRPUSDT_2026-06-05 06:27:23.559332+00:00"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=["XRPUSDT"],
        run_id="ts_cross_seg",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": f"{old_seg}_initial_trend_SELL_0_0",
            "symbol": "XRPUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 50.0,
            "average_price": 1.10,
            "created_at": "2026-06-04 10:00:01+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": f"{new_seg}_initial_trend_SELL_0_0",
            "symbol": "XRPUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 69.1,
            "average_price": 1.1027,
            "created_at": "2026-06-05 06:27:16+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": f"{new_seg}_market_exit_late_fixup",
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 69.3,
            "average_price": 1.0201,
            "created_at": "2026-06-05 06:27:24+00:00",
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "XRPUSDT")
    new_entry = f"{new_seg}_initial_trend_SELL_0_0"
    old_entry = f"{old_seg}_initial_trend_SELL_0_0"
    expected = pytest.approx((1.1027 - 1.0201) * 69.1, rel=1e-4)
    assert pnl_map[new_entry]["pnl_usdt"] == expected
    assert old_entry not in pnl_map or pnl_map[old_entry].get("realized_pnl") is None


def test_multileg_pnl_late_fixup_pairs_when_exit_ts_before_entry(multi_leg_db) -> None:
    """late_fixup is often persisted before the entry fill report arrives."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "XRPUSDT_2026-06-07 09:12:20.523263+00:00"
    entry_id = f"{segment}_initial_trend_BUY_0_0"
    exit_id = f"{segment}_market_exit_late_fixup"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=["XRPUSDT"],
        run_id="ts_exit_first",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "XRPUSDT",
            "side": "SELL",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 64.3,
            "average_price": 1.1546,
            "created_at": "2026-06-07 09:12:21+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 64.3,
            "average_price": 1.1628,
            "filled_at": "2026-06-07 09:12:34+00:00",
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "XRPUSDT")
    expected = pytest.approx((1.1546 - 1.1628) * 64.3, rel=1e-4)
    assert pnl_map[entry_id]["pnl_hint"] == "已实现"
    assert pnl_map[entry_id]["pnl_usdt"] == expected
    assert pnl_map[exit_id]["pnl_usdt"] == expected


def test_multileg_pnl_dust_market_exit_uses_partial_fill_qty(multi_leg_db) -> None:
    """HYPE S3_dust: entry 81.96 @ 59.831, exit fill 0.01 @ 60.007."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "HYPEUSDT_2026-06-13 00:45:00+00:00"
    entry_id = f"{segment}_S3"
    exit_id = f"{entry_id}_dust"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["HYPEUSDT"],
        run_id="ml_dust_pnl",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": entry_id,
            "leg_id": entry_id,
            "symbol": "HYPEUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 81.96,
            "average_price": 59.831,
            "filled_at": "2026-06-13 01:16:27+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": exit_id,
            "leg_id": entry_id,
            "symbol": "HYPEUSDT",
            "side": "BUY",
            "position_side": "SHORT",
            "purpose": "market_exit",
            "status": "closed",
            "quantity": 0.01,
            "filled_quantity": 0.01,
            "average_price": 60.007,
            "filled_at": "2026-06-13 14:30:16+00:00",
        }
    )
    pnl_map = multileg_pnl_by_order_id(multi_leg_db, "HYPEUSDT")
    expected = pytest.approx((59.831 - 60.007) * 0.01, rel=1e-4)
    assert pnl_map[entry_id]["pnl_usdt"] == expected
    assert pnl_map[exit_id]["pnl_usdt"] == expected


def test_multileg_pnl_skips_ghost_unrealized_when_positions_table_used(
    multi_leg_db,
) -> None:
    """Filled entries without open multi_leg_positions row must not count as浮盈."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["HYPEUSDT"],
        run_id="ml_ghost_upnl",
    )
    group = "HYPEUSDT_2026-06-15 12:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1",
            "leg_id": f"{group}_L1",
            "symbol": "HYPEUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 10.0,
            "average_price": 60.0,
        }
    )
    storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "leg_id": f"{group}_L1",
            "symbol": "HYPEUSDT",
            "side": "LONG",
            "entry_price": 60.0,
            "quantity": 10.0,
            "status": "closed",
        }
    )

    pnl_map = multileg_pnl_by_order_id(
        multi_leg_db, "HYPEUSDT", mark_prices={"HYPEUSDT": 70.0}
    )
    assert f"{group}_L1" not in pnl_map


# ---------------------------------------------------------------------------
# Orphan-open-position fallback tests
# Tests for _multileg_stats() when multi_leg_positions has open legs
# that have NO corresponding entry rows in multi_leg_orders.
# ---------------------------------------------------------------------------
def _seed_orphan_position(
    multi_leg_db,
    *,
    leg_id: str = "XRPUSDT_2026-06-15 12:58:00+00:00_trend_buy_0",
    symbol: str = "XRPUSDT",
    strategy: str = "trend_scalp",
    side: str = "LONG",
    entry_price: float = 1.2384,
    quantity: float = 8760.9,
    status: str = "open",
) -> None:
    """Insert a position into multi_leg_positions WITHOUT a matching order."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=[strategy],
        symbols=[symbol],
        run_id=f"orphan_{leg_id}",
    )
    storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": strategy,
            "leg_id": leg_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "status": status,
        }
    )


def test_orphan_long_position_computes_unrealized(multi_leg_db) -> None:
    """An open LONG position with no matching order should contribute unrealized PnL."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(
        multi_leg_db,
        entry_price=1.20,
        quantity=1000.0,
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.25},
        since_ts=None,
    )
    # (1.25 - 1.20) * 1000 = 50.0
    assert result["unrealized_pnl"] == pytest.approx(50.0)
    assert result["open_positions"] == 1
    ts = result["by_strategy"]["trend_scalp"]
    assert ts["unrealized_pnl"] == pytest.approx(50.0)
    assert ts["open_positions"] == 1


def test_orphan_short_position_computes_unrealized(multi_leg_db) -> None:
    """An open SHORT position with no matching order should compute (entry - mark) × qty."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(
        multi_leg_db,
        leg_id="XRPUSDT_2026-06-15 13:00:00+00:00_short_0",
        side="SHORT",
        entry_price=1.30,
        quantity=500.0,
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.25},
        since_ts=None,
    )
    # (1.30 - 1.25) * 500 = 25.0
    assert result["unrealized_pnl"] == pytest.approx(25.0)


def test_orphan_position_skipped_when_no_mark_price(multi_leg_db) -> None:
    """Orphan positions should be skipped when mark price is missing or zero."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(multi_leg_db, entry_price=1.20, quantity=1000.0)
    # No mark price for XRPUSDT
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={},
        since_ts=None,
    )
    assert result["unrealized_pnl"] == pytest.approx(0.0)


def test_orphan_position_skipped_when_entry_price_zero(multi_leg_db) -> None:
    """Orphan positions with entry_price=0 should be skipped."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(multi_leg_db, entry_price=0.0, quantity=1000.0)
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.25},
        since_ts=None,
    )
    assert result["unrealized_pnl"] == pytest.approx(0.0)


def test_orphan_position_skipped_when_quantity_zero(multi_leg_db) -> None:
    """Orphan positions with quantity=0 should be skipped."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(multi_leg_db, entry_price=1.20, quantity=0.0)
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.25},
        since_ts=None,
    )
    assert result["unrealized_pnl"] == pytest.approx(0.0)


def test_orphan_closed_position_not_counted(multi_leg_db) -> None:
    """Closed positions should NOT contribute unrealized PnL (only 'open' status)."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(
        multi_leg_db,
        entry_price=1.20,
        quantity=1000.0,
        status="closed",
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.25},
        since_ts=None,
    )
    assert result["unrealized_pnl"] == pytest.approx(0.0)
    assert result["open_positions"] == 0


def test_orphan_not_double_counted_with_order_entry(multi_leg_db) -> None:
    """If a position's leg_id already has a PnL from the order path, it must not be counted again."""
    from src.order_management.multi_leg_storage import MultiLegStorage
    from mlbot_console.services.account_summary import _multileg_stats

    storage = MultiLegStorage(str(multi_leg_db))
    sym = "XRPUSDT"
    leg = "XRPUSDT_2026-06-15 14:00:00+00:00_L1"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=[sym],
        run_id="orphan_dedup",
    )
    # Insert a filled entry order (same leg_id as the position)
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": leg,
            "leg_id": leg,
            "symbol": sym,
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 100.0,
            "average_price": 1.20,
        }
    )
    # Insert matching open position
    storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "leg_id": leg,
            "symbol": sym,
            "side": "LONG",
            "entry_price": 1.20,
            "quantity": 100.0,
            "status": "open",
        }
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol=sym,
        mark_prices={sym: 1.25},
        since_ts=None,
    )
    # Should be computed ONCE: (1.25 - 1.20) * 100 = 5.0
    assert result["unrealized_pnl"] == pytest.approx(5.0)


def test_orphan_multiple_positions_accumulated(multi_leg_db) -> None:
    """Multiple orphan positions should have their unrealized PnL accumulated."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(
        multi_leg_db,
        leg_id="XRPUSDT_2026-06-15 12:00:00+00:00_leg1",
        side="LONG",
        entry_price=1.20,
        quantity=1000.0,
    )
    _seed_orphan_position(
        multi_leg_db,
        leg_id="XRPUSDT_2026-06-15 12:00:00+00:00_leg2",
        side="LONG",
        entry_price=1.25,
        quantity=2000.0,
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.30},
        since_ts=None,
    )
    # leg1: (1.30 - 1.20) * 1000 = 100.0
    # leg2: (1.30 - 1.25) * 2000 = 100.0
    assert result["unrealized_pnl"] == pytest.approx(200.0)
    assert result["open_positions"] == 2


def test_orphan_negative_unrealized_for_losing_position(multi_leg_db) -> None:
    """Orphan position with mark below entry should produce negative unrealized."""
    from mlbot_console.services.account_summary import _multileg_stats

    _seed_orphan_position(
        multi_leg_db,
        entry_price=1.30,
        quantity=1000.0,
    )
    result = _multileg_stats(
        multi_leg_db,
        symbol="XRPUSDT",
        mark_prices={"XRPUSDT": 1.20},
        since_ts=None,
    )
    # (1.20 - 1.30) * 1000 = -100.0
    assert result["unrealized_pnl"] == pytest.approx(-100.0)
