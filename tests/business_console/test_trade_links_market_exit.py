"""market_exit must not duplicate an existing closed entry→TP link."""

from __future__ import annotations

import pytest

from mlbot_console.services.trade_links import multi_leg_trade_links


def test_market_exit_links_entry_when_only_pending_tp(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_me_open",
    )
    group = "BNBUSDT_grid"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_entry_me",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "place",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 640.0,
            "filled_at": "2026-05-19 12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_tp_me",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "price": 646.0,
            "status": "open",
            "created_at": "2026-05-19 12:01:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_flat_me",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 638.0,
            "filled_at": "2026-05-19 13:00:00+00:00",
        }
    )
    links, _ = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 1
    assert links[0]["status"] == "closed"
    assert links[0]["exit_kind"] == "market_exit"
    assert links[0]["exit_price"] == 638.0


def test_trend_scalp_regime_market_exit_link_shows_loss(multi_leg_db):
    """Regime stop exits must appear in 回合 view (not only basket TP wins)."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    segment = "ETHUSDT_2026-06-06 14:00:27.526643+00:00"
    entry_id = f"{segment}_initial_trend_SELL_0_0"
    exit_id = f"{entry_id}_fill0_exit_regime_exit_2026-06-08 00:05:00+00:00"
    run_id = storage.create_run(
        mode="testnet",
        strategies=["trend_scalp"],
        symbols=["ETHUSDT"],
        run_id="ts_loss_link",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "ETHUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 0.048,
            "average_price": 1685.0,
            "filled_at": "2026-06-06 14:00:40+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "ETHUSDT",
            "side": "BUY",
            "position_side": "SHORT",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 0.048,
            "average_price": 1700.0,
            "filled_at": "2026-06-06 16:01:53+00:00",
        }
    )
    links, _ = multi_leg_trade_links(multi_leg_db, "ETHUSDT")
    assert len(links) == 1
    assert links[0]["exit_kind"] == "regime_exit"
    assert links[0]["pnl_usdt"] == pytest.approx((1685.0 - 1700.0) * 0.048, rel=1e-4)


def test_chop_orphan_market_exit_link_via_unified_pairing(multi_leg_db):
    import json

    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["XRPUSDT"],
        run_id="ml_link_orphan",
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
    links, _ = multi_leg_trade_links(multi_leg_db, "XRPUSDT")
    assert len(links) == 1
    assert links[0]["exit_kind"] == "market_exit"
    assert links[0]["pnl_usdt"] == pytest.approx((1.1419 - 1.1748) * 65.1, rel=1e-4)


def test_market_exit_skipped_when_filled_tp_link_exists(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_me_filled",
    )
    group = "BNBUSDT_grid"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_entry_me2",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "place",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 640.0,
            "filled_at": "2026-05-19 12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_tp_me2",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 646.0,
            "filled_at": "2026-05-19 12:30:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "cg_flat_me2",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 638.0,
            "filled_at": "2026-05-19 13:00:00+00:00",
        }
    )
    links, _ = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 1
    assert links[0]["status"] == "closed"
    assert links[0]["exit_kind"] == "take_profit"


def test_orphan_market_exit_links_grid_batch_from_raw_json(multi_leg_db):
    """cg_* market_exit has no grid batch id; fill fields live in raw_json only."""
    import json

    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["XRPUSDT"],
        run_id="mlr_orphan_me",
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
            "local_order_id": f"{group}_L3",
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 65.1,
            "average_price": 1.1709,
            "filled_at": "2026-06-04 07:48:23+00:00",
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
            "filled_quantity": 0.0,
            "average_price": None,
            "quantity": 65.1,
            "created_at": "2026-06-04 11:11:04+00:00",
            "raw": json.dumps(
                {
                    "filled": 65.1,
                    "average_price": 1.1419,
                    "status": "closed",
                }
            ),
        }
    )
    links, _ = multi_leg_trade_links(multi_leg_db, "XRPUSDT")
    assert len(links) == 1
    assert links[0]["leg"] == "L2"
    assert links[0]["status"] == "closed"
    assert links[0]["exit_kind"] == "market_exit"
    assert links[0]["exit_price"] == pytest.approx(1.1419)
