from app.services.trade_markers import collect_markers


def test_pending_excluded_by_default(trend_db, spot_db, multi_leg_db):
    m = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        include_pending=False,
    )
    assert all(x.get("status") != "pending" for x in m)


def test_pending_included(trend_db, spot_db, multi_leg_db):
    m = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        include_pending=True,
    )
    assert any(x.get("status") == "pending" for x in m)
