import pytest

from mlbot_console.services.orders_list import collect_orders, trend_orders


def test_trend_orders_list(trend_db):
    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    assert len(rows) >= 1
    assert rows[0]["scope"] == "trend"
    assert "order_id" in rows[0]
    entry_row = next(r for r in rows if r["order_type"] == "position_entry")
    assert entry_row["stop_loss_price"] == 98.5
    assert entry_row["take_profit_price"] == 106.0
    position_rows = [r for r in rows if str(r["order_id"]).startswith("p1:")]
    assert {r["order_type"] for r in position_rows} >= {
        "position_entry",
        "position_exit",
    }
    exit_row = next(r for r in position_rows if r["order_type"] == "position_exit")
    assert exit_row["side"] == "sell"
    assert exit_row["strategy"] == "tpc"
    assert exit_row["marker_id"] == "trend:positions:p1:exit"


def test_trend_orders_include_position_operations_with_strategy(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO position_operations VALUES (
            'op_add_orders', 'p1', 'add', '2024-01-01T12:00:00+00:00',
            0.2, 102.0, 'scale in', NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    op_row = next(r for r in rows if r["order_id"] == "op_add_orders")
    assert op_row["order_type"] == "position_add"
    assert op_row["strategy"] == "tpc"
    assert op_row["marker_id"] == "trend:position_operations:op_add_orders"


def test_trend_orders_all_symbols(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_btc', 'BTCUSDT', 'BUY', 'filled', 'limit',
            0.01, 50000.0, NULL,
            '2024-01-02T10:00:00+00:00', '2024-01-02T09:00:00+00:00',
            '2024-01-02T10:00:00+00:00', 50000.0, 0.01, NULL
        )
        """
    )
    conn.commit()
    conn.close()
    rows = trend_orders(trend_db, "*", limit=50)
    symbols = {r["symbol"] for r in rows}
    assert "ETHUSDT" in symbols
    assert "BTCUSDT" in symbols


def test_trend_orders_join_survives_positions_created_at_column(tmp_path):
    """Production trend DB has created_at on both orders and positions."""
    import sqlite3

    path = tmp_path / "prod_like_trend.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            realized_pnl REAL,
            status TEXT,
            strategy_id TEXT,
            stop_loss_price REAL,
            take_profit_price REAL,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE position_operations (
            operation_id TEXT PRIMARY KEY,
            position_id TEXT,
            operation_type TEXT,
            operation_time TEXT,
            size REAL,
            price REAL,
            reason TEXT,
            stop_loss_price REAL,
            take_profit_price REAL
        );
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            status TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            stop_price REAL,
            filled_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            average_price REAL,
            filled_quantity REAL,
            position_id TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_bnb', 'BNBUSDT', 'long',
            '2024-01-01T10:00:00+00:00', NULL,
            600.0, NULL, 0.0, 'open', 'tpc', NULL, NULL,
            '2024-01-01T10:00:00+00:00', '2024-01-01T10:00:00+00:00'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_bnb', 'BNBUSDT', 'BUY', 'filled', 'limit',
            0.1, 601.0, NULL,
            '2024-01-01T11:00:00+00:00', '2024-01-01T10:30:00+00:00',
            '2024-01-01T11:00:00+00:00', 601.0, 0.1, 'p_bnb'
        )
        """
    )
    conn.commit()
    conn.close()

    rows = trend_orders(path, "BNBUSDT", limit=50)
    assert any(r["order_id"] == "ord_bnb" for r in rows)
    assert any(r["strategy"] == "tpc" for r in rows)


def test_trend_orders_expose_stop_loss_from_stop_market_row(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_sl_rej', 'ETHUSDT', 'SELL', 'rejected', 'stop_market',
            0.1, 2100.0, 2095.5,
            '2024-01-03T10:00:00+00:00', '2024-01-03T09:00:00+00:00',
            '2024-01-03T10:00:00+00:00', NULL, 0.0, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    sl_row = next(r for r in rows if r["order_id"] == "ord_sl_rej")
    assert sl_row["stop_loss_price"] == 2095.5
    assert sl_row["stop_loss_hint"] == "挂单失败"


def test_trend_orders_exclude_rejected_leaves_filled_visible(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    for i in range(30):
        conn.execute(
            """
            INSERT INTO orders VALUES (
                ?, 'ETHUSDT', 'BUY', 'rejected', 'stop_market',
                0.0, 100.0, NULL,
                '2024-01-02T10:00:00+00:00', '2024-01-02T09:00:00+00:00',
                '2024-01-02T10:00:00+00:00', NULL, 0.0, NULL
            )
            """,
            (f"rej_{i}",),
        )
    conn.commit()
    conn.close()

    rows = trend_orders(trend_db, "ETHUSDT", exclude_statuses=["rejected"], limit=50)
    assert all(r["status"] != "rejected" for r in rows)
    assert any(r["order_type"] == "position_exit" for r in rows)


def test_collect_orders_scopes(trend_db, spot_db, multi_leg_db):
    all_rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot", "multi_leg"],
        limit=50,
    )
    scopes = {r["scope"] for r in all_rows}
    assert "trend" in scopes
    assert "spot" in scopes


def test_collect_orders_entry_row_has_no_trend_exit_pnl(
    trend_db, spot_db, multi_leg_db, bus_root
):
    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend"],
        limit=50,
        feature_bus_root=bus_root,
    )
    entry_row = next(r for r in rows if r["order_id"] == "p1:entry")
    assert entry_row.get("pnl_usdt") is None


def test_trend_filled_order_inherits_stop_loss_without_position_id(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_no_pid', 'ETHUSDT', 'SELL', 'filled', 'limit',
            0.1, 105.0, NULL,
            '2024-01-01T14:00:00+00:00', '2024-01-01T13:30:00+00:00',
            '2024-01-01T14:00:00+00:00', 105.0, 0.1, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    row = next(r for r in rows if r["order_id"] == "ord_no_pid")
    assert row["stop_loss_price"] == 98.5
    assert row["take_profit_price"] == 106.0


def test_collect_orders_multileg_history_survives_exclude_filter(
    trend_db, spot_db, multi_leg_db
):
    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["multi_leg"],
        exclude_statuses=["expired", "canceled", "rejected"],
        limit=50,
    )
    ids = {r["order_id"] for r in rows}
    assert "ml_eth_open_tp" in ids
    assert "ml_eth_entry" in ids
    assert "ml_eth_l2_expired" not in ids


def test_collect_orders_trend_exit_order_gets_pnl(
    trend_db, spot_db, multi_leg_db, bus_root
):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_exit_fill', 'ETHUSDT', 'SELL', 'filled', 'limit',
            0.1, 105.0, NULL,
            '2024-01-01T14:00:00+00:00', '2024-01-01T13:30:00+00:00',
            '2024-01-01T14:00:00+00:00', 105.0, 0.1, 'p1'
        )
        """
    )
    conn.commit()
    conn.close()

    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend"],
        limit=50,
        feature_bus_root=bus_root,
    )
    row = next(r for r in rows if r["order_id"] == "ord_exit_fill")
    assert row["pnl_usdt"] == 12.5
    assert row["stop_loss_price"] == 98.5


def test_collect_orders_attaches_trend_exit_pnl(
    trend_db, spot_db, multi_leg_db, bus_root
):
    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend"],
        limit=50,
        feature_bus_root=bus_root,
    )
    exit_row = next(r for r in rows if r["order_id"] == "p1:exit")
    assert exit_row["realized_pnl"] == 12.5
    assert exit_row["pnl_usdt"] == 12.5
    assert exit_row["pnl_hint"] == "已实现"


def test_collect_orders_open_trend_position_shows_unrealized_pnl(
    trend_db, spot_db, multi_leg_db, bus_root
):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_open', 'ETHUSDT', 'long',
            '2026-05-19T08:00:00+00:00', NULL,
            2100.0, NULL, NULL, 'open', 'tpc', 2095.0, NULL, 0.5
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_open_entry', 'ETHUSDT', 'BUY', 'filled', 'limit',
            0.5, 2100.0, NULL,
            '2026-05-19T08:00:00+00:00', '2026-05-19T08:00:00+00:00',
            '2026-05-19T08:00:00+00:00', 2100.0, 0.5, 'p_open'
        )
        """
    )
    conn.commit()
    conn.close()

    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend"],
        limit=50,
        feature_bus_root=bus_root,
    )
    entry_row = next(r for r in rows if r["order_id"] == "ord_open_entry")
    assert entry_row["pnl_usdt"] is not None
    assert entry_row["unrealized_pnl"] is not None
    assert entry_row["pnl_hint"] == "浮盈"
    pos_entry = next(r for r in rows if r["order_id"] == "p_open:entry")
    assert pos_entry["pnl_usdt"] is not None


def test_collect_orders_trend_stop_loss_pnl_when_realized_null(
    trend_db, spot_db, multi_leg_db, bus_root
):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_sl', 'XRPUSDT', 'long',
            '2026-05-15T10:00:00+00:00', '2026-05-15T12:00:00+00:00',
            2.50, 2.30, NULL, 'closed', 'tpc', 2.30, NULL, 0.0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_sl_exit', 'XRPUSDT', 'SELL', 'filled', 'stop_market',
            1000.0, NULL, 2.30,
            '2026-05-15T12:00:00+00:00', '2026-05-15T11:00:00+00:00',
            '2026-05-15T12:00:00+00:00', 2.30, 1000.0, 'p_sl'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_sl_entry', 'XRPUSDT', 'BUY', 'filled', 'limit',
            1000.0, 2.50, NULL,
            '2026-05-15T10:00:00+00:00', '2026-05-15T10:00:00+00:00',
            '2026-05-15T10:00:00+00:00', 2.50, 1000.0, 'p_sl'
        )
        """
    )
    conn.commit()
    conn.close()

    rows = collect_orders(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="XRPUSDT",
        scopes=["trend"],
        limit=50,
        feature_bus_root=bus_root,
    )
    exit_row = next(r for r in rows if r["order_id"] == "ord_sl_exit")
    assert exit_row["pnl_usdt"] == pytest.approx(-200.0)
    assert exit_row["pnl_hint"] == "已实现"
    pos_exit = next(r for r in rows if r["order_id"] == "p_sl:exit")
    assert pos_exit["pnl_usdt"] == pytest.approx(-200.0)


def test_spot_orders_legacy_schema_without_filled_columns(tmp_path):
    import sqlite3

    from mlbot_console.services.orders_list import spot_orders_list

    path = tmp_path / "legacy_spot.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL,
            status TEXT NOT NULL,
            exchange_order_id TEXT,
            client_order_id TEXT,
            raw_json TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            'legacy_bnb', '2024-01-02T08:00:00+00:00', 'BNBUSDT', 'buy', 'market',
            0.1, 600.0, 'filled', NULL, NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    rows = spot_orders_list(path, "BNBUSDT", limit=10)
    assert len(rows) == 1
    assert rows[0]["order_id"] == "legacy_bnb"
    assert rows[0]["scope"] == "spot"


def test_spot_orders_exclude_status_in_sql_not_post_filter(tmp_path):
    import sqlite3

    from mlbot_console.services.orders_list import spot_orders_list

    path = tmp_path / "spot_exclude.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            symbol TEXT,
            side TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            status TEXT,
            filled_quantity REAL,
            filled_quote_usdt REAL
        );
        """
    )
    for i in range(30):
        conn.execute(
            """
            INSERT INTO spot_orders VALUES (
                ?, '2024-01-02T10:00:00+00:00', '2024-01-02T10:00:00+00:00',
                'BNBUSDT', 'buy', 'limit', 0.1, 600.0, 'expired', 0.0, 0.0
            )
            """,
            (f"exp_{i}",),
        )
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            'bnb_fill', '2024-01-01T08:00:00+00:00', '2024-01-01T08:05:00+00:00',
            'BNBUSDT', 'buy', 'market', 0.1, 590.0, 'filled', 0.1, 59.0
        )
        """
    )
    conn.commit()
    conn.close()

    rows = spot_orders_list(
        path,
        "BNBUSDT",
        exclude_statuses=["expired", "canceled"],
        limit=10,
    )
    assert [r["order_id"] for r in rows] == ["bnb_fill"]
