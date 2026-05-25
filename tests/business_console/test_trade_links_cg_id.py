"""Trade links when local_order_id is cg_* but leg_id has _L1 suffix."""

from __future__ import annotations

from mlbot_console.services.trade_links import multi_leg_trade_links


def test_cg_client_id_links_via_leg_id(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_cg_link",
    )
    group = "BNBUSDT_grid"
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
            "filled_quantity": 0.2,
            "average_price": 640.0,
            "filled_at": "2026-05-19 12:00:00+00:00",
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
            "price": 646.0,
            "status": "open",
            "created_at": "2026-05-19 12:01:00+00:00",
        }
    )
    links, _ = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert links == []
