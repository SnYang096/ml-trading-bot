from __future__ import annotations

import pandas as pd

from scripts.event_backtest import (
    _align_feature_index_to_bar_close,
    _iter_update_bars_1min,
    _timeframe_to_timedelta,
)


def test_timeframe_to_timedelta_parses_common_tokens():
    assert _timeframe_to_timedelta("120T") == pd.Timedelta(hours=2)
    assert _timeframe_to_timedelta("4H") == pd.Timedelta(hours=4)
    assert _timeframe_to_timedelta("1D") == pd.Timedelta(days=1)
    assert _timeframe_to_timedelta("bad") is None


def test_align_feature_index_shifts_2h_to_bar_close():
    idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 02:00:00"], utc=True)
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=idx)

    out = _align_feature_index_to_bar_close(df, "120T")

    assert list(out.index) == list(
        pd.to_datetime(["2024-01-01 02:00:00", "2024-01-01 04:00:00"], utc=True)
    )
    assert out["close"].tolist() == [100.0, 101.0]


def test_align_feature_index_keeps_1min_unchanged():
    idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:01:00"], utc=True)
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=idx)

    out = _align_feature_index_to_bar_close(df, "1T")

    assert list(out.index) == list(idx)


def test_alignment_prevents_future_bar_leakage_in_window_filter():
    # Left-labeled 2H bars at 00:00 / 02:00 / 04:00 represent closed bars at 02:00 / 04:00 / 06:00.
    raw_idx = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 02:00:00", "2024-01-01 04:00:00"], utc=True
    )
    raw = pd.DataFrame({"signal": [1, 2, 3]}, index=raw_idx)

    aligned = _align_feature_index_to_bar_close(raw, "120T")
    cutoff = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")

    # Without alignment, [00:00, 02:00] both appear <= 03:00 (leaks second bar's future info).
    assert len(raw[raw.index <= cutoff]) == 2
    # With alignment, only first closed bar (02:00) is visible before 03:00.
    assert len(aligned[aligned.index <= cutoff]) == 1


def test_iter_update_bars_1min_same_for_fast_and_non_fast():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:01:00",
            "2024-01-01 00:02:00",
            "2024-01-01 00:03:00",
            "2024-01-01 00:04:00",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.1, 2.1, 3.1, 4.1],
            "low": [0.9, 1.9, 2.9, 3.9],
            "close": [1.0, 2.0, 3.0, 4.0],
        },
        index=idx,
    )
    prev_ts = pd.Timestamp("2024-01-01 00:01:30", tz="UTC")
    cur_ts = pd.Timestamp("2024-01-01 00:03:30", tz="UTC")

    non_fast_ts = [ts for ts, _ in _iter_update_bars_1min(bars, prev_ts, cur_ts)]
    fast_ts = [
        ts for ts, _ in _iter_update_bars_1min(bars, prev_ts, cur_ts, fast_mode=True)
    ]
    assert non_fast_ts == fast_ts
    assert non_fast_ts == list(
        pd.to_datetime(["2024-01-01 00:02:00", "2024-01-01 00:03:00"], utc=True)
    )
