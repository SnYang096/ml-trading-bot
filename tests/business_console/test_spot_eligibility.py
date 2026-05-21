from mlbot_console.services.spot_eligibility import spot_eligibility_summary


def test_spot_eligibility_blockers(bus_root, spot_db):
    data = spot_eligibility_summary(
        feature_bus_root=bus_root,
        spot_db=spot_db,
        symbol="ETHUSDT",
    )
    assert data["symbol"] == "ETHUSDT"
    assert "pending_spot_orders" in data["blockers"]
    assert data["can_buy"] is False


def test_spot_eligibility_positive_ema(tmp_path, bus_root):
    import sqlite3

    spot = tmp_path / "spot.db"
    conn = sqlite3.connect(spot)
    conn.execute(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT,
            symbol TEXT, side TEXT, order_type TEXT, quantity REAL, price REAL,
            status TEXT, filled_quantity REAL, filled_quote_usdt REAL
        )
        """
    )
    conn.commit()
    conn.close()
    data = spot_eligibility_summary(
        feature_bus_root=bus_root,
        spot_db=spot,
        symbol="ETHUSDT",
    )
    assert data["weekly_ema_200_position"] is not None
    assert data["can_buy"] is True
