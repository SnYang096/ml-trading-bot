from app.services.orders_list import collect_orders, trend_orders


def test_trend_orders_list(trend_db):
    rows = trend_orders(trend_db, "ETHUSDT", limit=50)
    assert len(rows) >= 1
    assert rows[0]["scope"] == "trend"
    assert "order_id" in rows[0]


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
