"""Marker alignment and scope counts."""

from __future__ import annotations

from mlbot_console.services.trade_markers import (
    align_markers_to_candles,
    marker_scope_counts,
)


def test_align_markers_snaps_filled_to_nearest_bar():
    markers = [
        {"time": 100, "status": "filled", "scope": "trend"},
        {"time": 500, "status": "filled", "scope": "spot"},
    ]
    out = align_markers_to_candles(markers, [200, 300, 400])
    assert len(out) == 2
    assert out[0]["time"] == 200
    assert out[0]["detail"]["order_time"] == 100
    assert out[1]["time"] == 400


def test_align_markers_snaps_in_range_off_bar_time():
    """Filled orders between bar opens must pin to nearest candle (LWC requirement)."""
    markers = [{"time": 250, "status": "filled", "scope": "trend", "event": "entry"}]
    out = align_markers_to_candles(markers, [200, 300, 400])
    assert out[0]["time"] in (200, 300)
    assert out[0]["detail"]["order_time"] == 250


def test_marker_scope_counts():
    markers = [
        {"scope": "trend"},
        {"scope": "trend"},
        {"scope": "spot"},
    ]
    c = marker_scope_counts(markers)
    assert c["trend"] == 2
    assert c["spot"] == 1
    assert c["total"] == 3
