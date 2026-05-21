"""Trade marker extraction from synthetic SQLite stores."""

from __future__ import annotations

from mlbot_console.services.trade_markers import (
    collect_markers,
    spot_markers,
    trend_markers,
)


def test_trend_markers_entry_exit(trend_db):
    markers = trend_markers(trend_db, "ETHUSDT")
    events = {m["event"] for m in markers}
    assert "entry" in events
    assert "exit" in events
    exit_m = next(m for m in markers if m["event"] == "exit")
    assert exit_m["pnl_usdt"] == 12.5
    assert exit_m["strategy"] == "tpc"


def test_spot_markers_buy(spot_db):
    markers = spot_markers(spot_db, "ETHUSDT")
    filled = [m for m in markers if m.get("status") == "filled"]
    assert len(filled) == 1
    assert filled[0]["scope"] == "spot"
    assert filled[0]["event"] == "entry"
    assert filled[0]["strategy"] == "spot_accum_simple"


def test_collect_markers_missing_db(tmp_path):
    markers = collect_markers(
        trend_db=tmp_path / "missing.db",
        spot_db=tmp_path / "missing2.db",
        multi_leg_db=tmp_path / "missing3.db",
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
    )
    assert markers == []


def test_collect_markers_since_filter(trend_db, spot_db, multi_leg_db):
    all_m = trend_markers(trend_db, "ETHUSDT")
    since = all_m[0]["time"]
    inc = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend"],
        since_ts=since,
    )
    assert all(m["time"] > since for m in inc)
