from mlbot_console.services.orders_list import collect_orders, trend_orders


def test_trend_orders_list(trend_db):
    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    assert len(rows) >= 1
    assert rows[0]["scope"] == "trend"
    assert "order_id" in rows[0]


def test_trend_orders_all_symbols(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_btc', 'BTCUSDT', 'BUY', 'filled', 'limit',
            0.01, 50000.0,
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
