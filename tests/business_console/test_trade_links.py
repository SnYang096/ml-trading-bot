"""Trade Map entry→exit links for chop_grid L legs."""

from __future__ import annotations

import pandas as pd

from mlbot_console.services.trade_links import multi_leg_trade_links


def test_open_l1_shows_planned_tp_link_and_exit_marker(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_link_open",
    )
    group = "BNBUSDT_2026-05-19T084000"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "place",
            "quantity": 0.31,
            "price": 637.11,
            "status": "filled",
            "filled_quantity": 0.31,
            "average_price": 637.11,
            "filled_at": "2026-05-19 08:45:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1_tp",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "quantity": 0.31,
            "price": 643.55,
            "status": "open",
            "created_at": "2026-05-19 08:46:00+00:00",
        }
    )
    links, extras = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 1
    assert links[0]["status"] == "open"
    assert links[0]["entry_price"] == 637.11
    assert links[0]["exit_price"] == 643.55
    assert extras == []
    from mlbot_console.services.trade_markers import multi_leg_markers

    markers = multi_leg_markers(multi_leg_db, "BNBUSDT", include_open_orders=True)
    tp_m = [m for m in markers if m.get("event") == "tp"]
    assert len(tp_m) == 1
    assert tp_m[0]["status"] == "pending"


def test_filled_tp_closes_link(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_link_closed",
    )
    group = "BNBUSDT_2026-05-20T120000"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "place",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 640.0,
            "filled_at": "2026-05-20 12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_L1_tp",
            "leg_id": f"{group}_L1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 646.0,
            "filled_at": "2026-05-20 14:00:00+00:00",
        }
    )
    links, extras = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 1
    assert links[0]["status"] == "closed"
    assert links[0]["exit_kind"] == "take_profit"
    assert extras == []


def test_links_outside_window_are_not_clipped_into_view(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_link_window",
    )
    old_group = "BNBUSDT_2026-05-18T120000"
    new_group = "BNBUSDT_2026-05-20T120000"
    for group, day in ((old_group, "2026-05-18"), (new_group, "2026-05-20")):
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": f"{group}_L1",
                "symbol": "BNBUSDT",
                "side": "BUY",
                "purpose": "place",
                "status": "filled",
                "filled_quantity": 0.2,
                "average_price": 640.0,
                "filled_at": f"{day} 12:00:00+00:00",
            }
        )
        storage.upsert_order(
            {
                "run_id": run_id,
                "strategy": "chop_grid",
                "local_order_id": f"{group}_L1_tp",
                "leg_id": f"{group}_L1",
                "symbol": "BNBUSDT",
                "side": "SELL",
                "purpose": "take_profit",
                "status": "filled",
                "filled_quantity": 0.2,
                "average_price": 646.0,
                "filled_at": f"{day} 14:00:00+00:00",
            }
        )

    start_ts = int(pd.Timestamp("2026-05-20T00:00:00Z").timestamp())
    end_ts = int(pd.Timestamp("2026-05-21T00:00:00Z").timestamp())
    links, _ = multi_leg_trade_links(
        multi_leg_db,
        "BNBUSDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert len(links) == 1
    assert links[0]["entry_time"] == int(
        pd.Timestamp("2026-05-20T12:00:00Z").timestamp()
    )
