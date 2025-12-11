import pandas as pd
import numpy as np

from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_full_scan,
)


def _make_trend_df(up=True):
    # simple OHLCV with deterministic ATR (~1.0)
    prices = np.linspace(100, 110, 20) if up else np.linspace(110, 100, 20)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1.0,
        },
        index=pd.date_range("2024-01-01", periods=len(prices), freq="1H"),
    )
    # constant atr around 1
    df["atr"] = 1.0
    return df


def test_full_scan_long_hits_tp_before_sl():
    df = _make_trend_df(up=True)
    labels = compute_sr_reversal_label_full_scan(
        df,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        combine_mode="long_only",
    )
    # Trending up with entry at next bar; should hit +2R before -1R → label=1
    assert (labels.dropna() == 1).all()


def test_full_scan_short_hits_tp_before_sl():
    df = _make_trend_df(up=False)
    labels = compute_sr_reversal_label_full_scan(
        df,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        combine_mode="short_only",
    )
    # Trending down; short should reach TP first → label=1
    assert (labels.dropna() == 1).all()


def test_full_scan_timeout_produces_nan():
    # Flat prices: neither +2R nor -1R within holding window
    df = _make_trend_df(up=True)
    df[["open", "high", "low", "close"]] = 100.0
    labels = compute_sr_reversal_label_full_scan(
        df,
        max_holding_bars=3,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        combine_mode="long_only",
    )
    # Should timeout: expect NaN where holding window exhausted, no +2R/-1R hit
    assert labels.isna().any()
    # No positives; zeros are allowed for bars that finish holding window as failure
    assert (labels.dropna() == 0).all()
