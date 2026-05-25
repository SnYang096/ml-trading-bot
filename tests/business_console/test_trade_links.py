"""Trade Map entry→exit links for chop_grid L legs."""

from __future__ import annotations

import pandas as pd

from mlbot_console.services.trade_links import (
    collect_trade_links,
    multi_leg_trade_links,
    spot_trade_links,
    trend_trade_links,
)


def test_open_l1_with_pending_tp_emits_no_trade_link(multi_leg_db):
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
    assert links == []
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
    assert links[0]["color"] == "#26a69a"
    assert extras == []


def test_filled_s2_short_tp_link(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_link_s2_closed",
    )
    group = "BNBUSDT_2026-05-21T080000"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "place",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 656.42,
            "filled_at": "2026-05-21 08:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2_tp",
            "leg_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "status": "filled",
            "filled_quantity": 0.2,
            "average_price": 649.98,
            "filled_at": "2026-05-21 10:00:00+00:00",
        }
    )
    links, extras = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    s2_links = [lk for lk in links if lk.get("leg") == "S2"]
    assert len(s2_links) == 1
    assert s2_links[0]["status"] == "closed"
    assert s2_links[0]["exit_kind"] == "take_profit"
    assert s2_links[0]["entry_price"] == 656.42
    assert s2_links[0]["exit_price"] == 649.98
    assert s2_links[0]["entry_price"] > s2_links[0]["exit_price"]
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


def test_trend_open_position_emits_no_trade_link(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_open', 'ETHUSDT', 'long',
            '2024-01-02T10:00:00+00:00', NULL,
            110.0, NULL, NULL, 'open', 'bpc', NULL, NULL, 1.0
        )
        """
    )
    conn.commit()
    conn.close()

    current_time = int(pd.Timestamp("2024-01-02T12:00:00Z").timestamp())
    links = trend_trade_links(
        trend_db,
        "ETHUSDT",
        current_time=current_time,
        current_price=115.0,
    )

    open_links = [lk for lk in links if lk["entry_marker_id"].endswith("p_open:entry")]
    assert open_links == []


def test_trend_add_operation_draws_add_to_exit_link(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO position_operations VALUES (
            'op_add_1', 'p1', 'add', '2024-01-01T12:00:00+00:00',
            0.2, 102.0, 'scale in', NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    links = trend_trade_links(trend_db, "ETHUSDT")
    add_links = [lk for lk in links if lk["entry_marker_id"].endswith("op_add_1")]

    assert len(add_links) == 1
    assert add_links[0]["leg"] == "add"
    assert add_links[0]["entry_price"] == 102.0
    assert add_links[0]["exit_price"] == 105.0
    assert add_links[0]["status"] == "closed"


def test_spot_buy_only_no_trade_link(spot_db):
    links = spot_trade_links(spot_db, "ETHUSDT")
    assert links == []


def test_spot_buy_sell_orders_draw_closed_link(spot_db):
    import sqlite3

    conn = sqlite3.connect(spot_db)
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            's_sell', '2024-01-02T10:00:00+00:00', '2024-01-02T10:05:00+00:00',
            'ETHUSDT', 'sell', 'market', 0.1, 2100.0, 'filled', 0.1, 210.0
        )
        """
    )
    conn.commit()
    conn.close()

    links = spot_trade_links(spot_db, "ETHUSDT")
    assert len(links) == 1
    assert links[0]["strategy"] == "spot_accum_simple"
    assert links[0]["entry_marker_id"] == "spot:spot_orders:s1"
    assert links[0]["exit_marker_id"] == "spot:spot_orders:s_sell"
    assert links[0]["entry_price"] == 2000.0
    assert links[0]["exit_price"] == 2100.0
    assert links[0]["color"] == "#26a69a"


def test_collect_trade_links_includes_spot_but_not_open_trend(
    trend_db, spot_db, multi_leg_db
):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_open2', 'ETHUSDT', 'long',
            '2024-01-02T10:00:00+00:00', NULL,
            110.0, NULL, NULL, 'open', 'bpc', NULL, NULL, 1.0
        )
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(spot_db)
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            's_sell2', '2024-01-02T11:00:00+00:00', '2024-01-02T11:05:00+00:00',
            'ETHUSDT', 'sell', 'market', 0.1, 2110.0, 'filled', 0.1, 211.0
        )
        """
    )
    conn.commit()
    conn.close()

    links, _ = collect_trade_links(
        multi_leg_db=multi_leg_db,
        trend_db=trend_db,
        spot_db=spot_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        current_time=int(pd.Timestamp("2024-01-02T12:00:00Z").timestamp()),
        current_price=115.0,
    )

    assert any(lk["strategy"] == "spot_accum_simple" for lk in links)
    assert not any(lk["entry_marker_id"].endswith("p_open2:entry") for lk in links)
