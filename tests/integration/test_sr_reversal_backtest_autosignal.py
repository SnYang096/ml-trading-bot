import dataclasses

import numpy as np
import pandas as pd
import pytest


def test_sr_reversal_backtest_autogenerates_signal_when_missing():
    """
    Regression test (updated):
    - sr_reversal_long is long_only and uses direction-fixed probability gating
    - signal column may be missing (we should not require it)
    - entries are driven by predict_proba >= entry_threshold

    We expect run_vectorbt_backtest() to produce trades without requiring `signal`.
    """
    try:
        import vectorbt  # noqa: F401
    except Exception:
        pytest.skip("vectorbt not installed")

    from scripts.train_strategy_pipeline import run_vectorbt_backtest
    from src.time_series_model.strategy_config import StrategyConfigLoader

    cfg = StrategyConfigLoader("config/strategies/sr_reversal_long").load()

    # Synthetic OHLCV with 4H frequency
    n = 160
    idx = pd.date_range("2025-01-01", periods=n, freq="4H")
    close = pd.Series(100.0 + np.sin(np.linspace(0, 8 * np.pi, n)), index=idx)
    atr = pd.Series(1.0, index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close).values,
            "high": (close + 0.5 * atr).values,
            "low": (close - 0.5 * atr).values,
            "close": close.values,
            "volume": np.full(n, 1000.0),
            "atr": atr.values,
            # SR-related columns used by SR signal generator
            "sr_strength_max": np.full(n, 1.0),
            "sqs_hal_low": np.full(n, 1.0),
            "sqs_hal_high": np.full(n, 1.0),
        },
        index=idx,
    )

    # IMPORTANT: signal column is missing here on purpose (direction is fixed by strategy).
    assert "signal" not in df.columns

    # Mostly-high probabilities to trigger entries for a long_only strategy
    preds = np.full(n, 0.9, dtype=float)

    # Use the strategy's backtest config, but disable RR exit to keep the test minimal/deterministic.
    params = dict(cfg.backtest.params)
    params["use_rr_exit"] = False
    params["use_signal_direction"] = False
    backtest_cfg = dataclasses.replace(cfg.backtest, params=params)

    out = run_vectorbt_backtest(
        df=df,
        preds=preds,
        backtest_cfg=backtest_cfg,
        task_type="binary",
        strategy_config=cfg,
    )

    assert out is not None
    assert out.get("total_trades", 0) > 0
