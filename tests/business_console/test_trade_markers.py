"""Trade marker extraction from synthetic SQLite stores."""

from __future__ import annotations

from mlbot_console.services.trade_markers import (
    collect_markers,
    multi_leg_markers,
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


def test_trend_order_markers_inherit_position_strategy(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_filled_p1', 'ETHUSDT', 'SELL', 'filled', 'market',
            0.1, 105.0, NULL,
            '2024-01-01T14:00:00+00:00', '2024-01-01T14:00:00+00:00',
            '2024-01-01T14:00:00+00:00', 105.0, 0.1, 'p1'
        )
        """
    )
    conn.commit()
    conn.close()

    markers = trend_markers(trend_db, "ETHUSDT")
    order_marker = next(m for m in markers if m["id"] == "trend:orders:ord_filled_p1")
    assert order_marker["strategy"] == "tpc"
    assert order_marker["event"] == "exit"
    assert order_marker["side"] == "short"


def test_trend_operation_markers_inherit_position_strategy(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO position_operations VALUES (
            'op_add_marker', 'p1', 'add', '2024-01-01T12:00:00+00:00',
            0.2, 102.0, 'scale in', NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    markers = trend_markers(trend_db, "ETHUSDT")
    op_marker = next(
        m for m in markers if m["id"] == "trend:position_operations:op_add_marker"
    )
    assert op_marker["strategy"] == "tpc"
    assert op_marker["is_add"] is True


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


def _ts(iso: str) -> int:
    from datetime import datetime, timezone

    s = iso.replace("Z", "+00:00")
    return int(datetime.fromisoformat(s).timestamp())


def test_trend_markers_window_filter(trend_db):
    start = _ts("2024-01-02T00:00:00+00:00")
    end = _ts("2024-01-03T00:00:00+00:00")
    markers = trend_markers(trend_db, "ETHUSDT", start_ts=start, end_ts=end)
    assert markers == []


def test_trend_markers_sql_pushdown_keeps_in_window_under_limit(trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    for i in range(5100):
        conn.execute(
            """
            INSERT INTO positions VALUES (
                ?, 'ETHUSDT', 'long',
                '2020-01-01T00:00:00+00:00', '2020-01-01T01:00:00+00:00',
                1.0, 1.1, 0.0, 'closed', 'tpc', NULL, NULL, 1.0
            )
            """,
            (f"old_{i}",),
        )
    conn.commit()
    conn.close()

    start = _ts("2024-01-01T00:00:00+00:00")
    end = _ts("2024-01-02T00:00:00+00:00")
    markers = trend_markers(trend_db, "ETHUSDT", start_ts=start, end_ts=end)
    ids = {m["id"] for m in markers}
    assert "trend:positions:p1:entry" in ids
    assert "trend:positions:p1:exit" in ids


def test_multi_leg_execution_report_window_filter(multi_leg_db):
    start = _ts("2024-01-01T12:00:00+00:00")
    end = _ts("2024-01-01T12:10:00+00:00")
    markers = multi_leg_markers(multi_leg_db, "ETHUSDT", start_ts=start, end_ts=end)
    report_ids = {
        m["id"]
        for m in markers
        if m["id"].startswith("multi_leg:multi_leg_execution_reports:")
    }
    assert len(report_ids) == 1


def test_spot_markers_window_excludes_outside(spot_db):
    start = _ts("2024-01-02T07:00:00+00:00")
    end = _ts("2024-01-02T07:30:00+00:00")
    markers = spot_markers(spot_db, "ETHUSDT", start_ts=start, end_ts=end)
    assert markers == []
