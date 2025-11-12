"""Helper utilities for Alpha101 factor calculations.

Vendored and adapted from the alpha101-crypto project:
https://raw.githubusercontent.com/lansetaowa/alpha101-crypto/main/alpha_utils.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_dataframe(obj: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj


def rank(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_dataframe(df).rank(axis=1, pct=True)


def scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    df = _ensure_dataframe(df)
    denom = df.abs().sum(axis=1, keepdims=True)
    denom = denom.replace(0, np.nan)
    return df.div(denom).mul(k)


def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(_ensure_dataframe(df))


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(_ensure_dataframe(df))


def power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    return _ensure_dataframe(df).pow(exp)


def _sanitize_window(window: float | int) -> int:
    if window <= 1:
        return 1
    return max(int(round(window)), 1)


def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return _ensure_dataframe(df).shift(int(t))


def ts_delta(df: pd.DataFrame, period: float = 1) -> pd.DataFrame:
    return _ensure_dataframe(df).diff(_sanitize_window(period))


def ts_sum(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).sum()


def ts_mean(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).mean()


def ts_std(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).std()


def ts_min(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).min()


def ts_max(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).max()


def ts_argmax(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return (
        _ensure_dataframe(df)
        .rolling(_sanitize_window(window), min_periods=1)
        .apply(np.argmax)
        .add(1)
    )


def ts_argmin(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return (
        _ensure_dataframe(df)
        .rolling(_sanitize_window(window), min_periods=1)
        .apply(np.argmin)
        .add(1)
    )


def ts_rank(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    window = _sanitize_window(window)
    return (
        _ensure_dataframe(df)
        .rolling(window, min_periods=1)
        .apply(lambda x: pd.Series(x).rank().iloc[-1])
    )


def ts_weighted_mean(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    window = _sanitize_window(window)
    df = _ensure_dataframe(df)
    weights = np.arange(1, window + 1, dtype=float)
    denom = weights.sum()

    def _wma(x: np.ndarray) -> float:
        if len(x) < window:
            w = weights[-len(x):]
        else:
            w = weights
        return float(np.dot(x, w[-len(x):]) / w[-len(x):].sum())

    return df.rolling(window, min_periods=1).apply(_wma, raw=True)


def ts_product(df: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    return _ensure_dataframe(df).rolling(_sanitize_window(window), min_periods=1).apply(np.prod)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    window = _sanitize_window(window)
    return (
        _ensure_dataframe(x)
        .rolling(window, min_periods=1)
        .corr(_ensure_dataframe(y))
    )


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: float = 10) -> pd.DataFrame:
    window = _sanitize_window(window)
    return (
        _ensure_dataframe(x)
        .rolling(window, min_periods=1)
        .cov(_ensure_dataframe(y))
    )
