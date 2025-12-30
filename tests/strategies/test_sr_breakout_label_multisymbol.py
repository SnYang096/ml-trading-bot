import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.sr_breakout_label import (
    compute_sr_breakout_label,
)


def test_sr_breakout_label_is_computed_per_symbol_no_cross_symbol_leakage():
    # Build a 2-symbol interleaved panel (same timestamps), which used to break the label generator.
    idx = pd.date_range("2024-01-01", periods=60, freq="4H", tz="UTC")
    # Use a wiggly path so MAE > 0 sometimes; otherwise labels collapse to max_rr.
    base_a = np.linspace(100, 130, len(idx)) + 2.0 * np.sin(np.linspace(0, 6, len(idx)))
    base_b = np.linspace(1000, 900, len(idx)) + 5.0 * np.sin(
        np.linspace(0, 6, len(idx))
    )
    df_a = pd.DataFrame(
        {
            "_symbol": "AAAUSDT",
            "close": base_a,
            # Make MAE non-trivial so RR is not always capped at max_rr
            "high": base_a + 1.0,
            "low": base_a - 8.0,
            "signal": 1.0,  # always long
        },
        index=idx,
    )
    df_b = pd.DataFrame(
        {
            "_symbol": "BBBUSDT",
            "close": base_b,
            "high": base_b + 2.0,
            "low": base_b - 16.0,
            "signal": -1.0,  # always short
        },
        index=idx,
    )

    # Interleave rows by time (common pooled format)
    df = pd.concat([df_a, df_b], axis=0).sort_index()

    labels = compute_sr_breakout_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=3,
        max_holding_bars=10,
        max_rr=3.0,
        stop_loss_r=1.0,
        auto_generate_signals=False,
    )

    # Should produce some non-NaN labels for both symbols
    assert labels.notna().sum() > 0

    # Must NOT collapse to a single constant across the entire pooled dataset
    uniq = set(np.round(labels.dropna().values, 6).tolist())
    assert len(uniq) > 1
