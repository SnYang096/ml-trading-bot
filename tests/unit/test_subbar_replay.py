import pandas as pd

from src.time_series_model.grid.subbar_replay import (
    merge_signal_features_onto_execution_bars,
    slice_execution_window,
    timeframe_to_timedelta,
)


def test_timeframe_to_timedelta_2h():
    assert timeframe_to_timedelta("2h") == pd.Timedelta(hours=2)


def test_merge_signal_features_onto_execution_bars_asof():
    idx2 = pd.date_range("2024-01-01", periods=3, freq="2h", tz="UTC")
    df_signal = pd.DataFrame(
        {"semantic_chop": [0.5, 0.6, 0.7], "atr14": [1.0, 1.1, 1.2]},
        index=idx2,
    )
    idx1 = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
    ohlc = pd.DataFrame(
        {
            "open": [1, 1, 1, 1, 1],
            "high": [2, 2, 2, 2, 2],
            "low": [0.5, 0.5, 0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5, 1.5, 1.5],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx1,
    )
    out = merge_signal_features_onto_execution_bars(ohlc, df_signal)
    assert "semantic_chop" in out.columns
    assert out.loc[idx1[0], "semantic_chop"] == 0.5
    assert out.loc[idx1[2], "semantic_chop"] == 0.5
    assert out.loc[idx1[3], "semantic_chop"] == 0.5
    assert out.loc[idx1[4], "semantic_chop"] == 0.6


def test_slice_execution_window_inclusive_signal_segment():
    sig = pd.date_range("2024-01-01", periods=4, freq="2h", tz="UTC")
    df_signal = pd.DataFrame({"x": range(4)}, index=sig)
    idx1 = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    df_exec = pd.DataFrame({"close": 1.0}, index=idx1)
    merged = merge_signal_features_onto_execution_bars(df_exec, df_signal)
    delta = pd.Timedelta(hours=2)
    sub = slice_execution_window(merged, sig, 1, 2, delta)
    assert sub.index[0] >= sig[1]
    assert sub.index[-1] < sig[2] + delta
