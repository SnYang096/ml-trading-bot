import pandas as pd


def test_sr_breakout_label_autogenerate_signals_produces_non_null_labels():
    from src.time_series_model.strategies.labels.sr_breakout_label import (
        compute_sr_breakout_label,
    )

    # Simple upward drift with enough bars for max_holding_bars logic
    n = 120
    close = pd.Series(range(n), dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )

    labels = compute_sr_breakout_label(
        df,
        auto_generate_signals=True,
        signal_horizon=1,
        signal_threshold_atr=0.0,
        max_holding_bars=10,
    )

    assert labels.notna().sum() > 0
