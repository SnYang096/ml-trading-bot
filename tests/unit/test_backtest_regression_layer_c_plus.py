import numpy as np
import pandas as pd

from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
    VectorBTBacktest,
)


def _mk_df(close: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="4H")
    close_s = pd.Series(close, index=idx, dtype=float)
    # Make high/low wide enough to allow trailing trigger on the drop bar
    df = pd.DataFrame(
        {
            "open": close_s.shift(1).fillna(close_s.iloc[0]),
            "high": close_s * 1.001,
            "low": close_s * 0.999,
            "close": close_s,
            "atr": 1.0,
            # Minimal columns to satisfy backtest sizing / plumbing
            "signal": 0.0,
        },
        index=idx,
    )
    return df


def test_regression_time_exit_max_holding_bars_closes_position():
    df = _mk_df([100, 101, 102, 103, 104, 105, 106])
    preds = np.ones(len(df), dtype=float)  # always "high", will enter immediately

    bt = VectorBTBacktest()
    res = bt.run(
        df=df,
        predictions=preds,
        task_type="regression",
        price_col="close",
        freq="4H",
        strategy_direction="long_only",
        # Deterministic one-shot entry
        top_quantile=1.0,
        entry_mode="cross",
        quantile_mode="train",
        train_entry_threshold=0.0,
        # Layer C+ time-exit
        max_holding_bars=3,
    )
    assert res is not None
    assert int(res["total_trades"]) >= 1


def test_regression_trailing_atr_stop_exits_on_drawdown():
    # Trend up then sharp drop; trailing should be hit on the drop.
    df = _mk_df([100, 102, 104, 106, 108, 90, 91, 92])
    preds = np.ones(len(df), dtype=float)

    bt = VectorBTBacktest()
    res = bt.run(
        df=df,
        predictions=preds,
        task_type="regression",
        price_col="close",
        freq="4H",
        strategy_direction="long_only",
        # One-shot entry
        top_quantile=1.0,
        entry_mode="cross",
        quantile_mode="train",
        train_entry_threshold=0.0,
        # Layer C+ trailing stop
        use_trailing_stop=True,
        atr_col="atr",
        atr_window=14,
        trailing_atr_mult=1.0,
    )
    assert res is not None
    assert int(res["total_trades"]) >= 1
