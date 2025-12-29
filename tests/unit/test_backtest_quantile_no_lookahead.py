import numpy as np
import pandas as pd

from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
    VectorBTBacktest,
)


def _expected_expanding_threshold(s: pd.Series, q: float) -> pd.Series:
    out = []
    for i in range(len(s)):
        if i == 0:
            out.append(np.nan)
        else:
            out.append(float(s.iloc[:i].quantile(q)))
    return pd.Series(out, index=s.index)


def _expected_rolling_threshold(s: pd.Series, q: float, window: int) -> pd.Series:
    out = []
    for i in range(len(s)):
        if i == 0:
            out.append(np.nan)
        else:
            start = max(0, i - window)
            out.append(float(s.iloc[start:i].quantile(q)))
    return pd.Series(out, index=s.index)


def test_quantile_threshold_series_expanding_is_causal_shifted():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.RangeIndex(5))
    q = 0.8
    got = VectorBTBacktest._quantile_threshold_series(
        s, q=q, mode="expanding", min_periods=1
    )
    exp = _expected_expanding_threshold(s, q)
    pd.testing.assert_series_equal(got, exp)


def test_quantile_threshold_series_rolling_is_causal_shifted():
    s = pd.Series([10.0, 0.0, 5.0, 20.0, 15.0], index=pd.RangeIndex(5))
    q = 0.5
    got = VectorBTBacktest._quantile_threshold_series(
        s, q=q, mode="rolling", window=3, min_periods=1
    )
    exp = _expected_rolling_threshold(s, q, window=3)
    pd.testing.assert_series_equal(got, exp)


def test_multiclass_prob_quantile_entries_use_causal_thresholds():
    # proba[:, long]=2, short=0
    proba = np.array(
        [
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.10, 0.90],
            [0.60, 0.10, 0.30],
        ],
        dtype=float,
    )
    idx = pd.RangeIndex(len(proba))
    le, se, lx, sx = VectorBTBacktest._multiclass_entries_from_proba(
        proba=proba,
        index=idx,
        long_class=2,
        short_class=0,
        neutral_class=1,
        entry_mode="prob_quantile",
        entry_quantile=0.9,
        quantile_mode="expanding",
        quantile_min_periods=1,
    )
    # Causal thresholds:
    # t0: no history => no entries
    # t1: long_thr = quantile([0.10], 0.9)=0.10 => long_p=0.80 enters
    # t2: long_thr = quantile([0.10,0.80],0.9)=0.73 => long_p=0.90 enters
    # t3: short_thr = quantile([0.10,0.10,0.10],0.9)=0.10 => short_p=0.60 enters
    assert bool(le.iloc[0]) is False and bool(se.iloc[0]) is False
    assert bool(le.iloc[1]) is True
    assert bool(le.iloc[2]) is True
    assert bool(se.iloc[3]) is True
    assert lx.sum() == 0 and sx.sum() == 0
