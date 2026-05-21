from mlbot_console.services.trade_markers import collect_markers, multi_leg_markers


def test_multi_leg_markers(multi_leg_db):
    markers = multi_leg_markers(multi_leg_db, "ETHUSDT")
    scopes = {m["scope"] for m in markers}
    assert "multi_leg" in scopes or len(markers) >= 1
    assert any(m["scope"] == "multi_leg" for m in markers)


def test_collect_includes_multi_leg(multi_leg_db, trend_db, spot_db):
    all_m = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["multi_leg"],
    )
    assert len(all_m) >= 1
