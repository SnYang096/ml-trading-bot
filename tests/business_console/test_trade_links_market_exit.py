"""market_exit must not duplicate an existing closed entry→TP link."""

from __future__ import annotations

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
