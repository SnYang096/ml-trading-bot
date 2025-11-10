"""
Feature extraction utilities for regime detection.

The functions here provide lightweight, vectorised implementations of the
core indicators referenced in ``docs/行情：Regime Detection（行情状态识别）.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    return series.astype(float)


def rolling_hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the rolling Hurst exponent using a vectorised R/S method.
    """
    series = _validate_series(series, "series")
    if window < 20:
        raise ValueError("window must be at least 20 for Hurst exponent.")

    log_lags = np.log(np.arange(2, window // 2 + 1))

    def hurst(window_values: np.ndarray) -> float:
        if np.allclose(window_values, window_values[0]):
            return 0.5
        rs = []
        for lag in range(2, window // 2 + 1):
            chunk = window_values[:-lag]
            next_chunk = window_values[lag:]
            diff = next_chunk - chunk
            if diff.std() == 0:
                rs.append(0.0)
                continue
            rs.append(np.mean(np.abs(diff)))
        rs = np.array(rs)
        if np.any(rs <= 0):
            return 0.5
        coeffs = np.polyfit(log_lags, np.log(rs), 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))

    return (
        series.rolling(window=window, min_periods=window)
        .apply(hurst, raw=True)
        .fillna(0.5)
    )


def rolling_linear_regression(
    series: pd.Series, window: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling slope and R^2 of a linear regression fit.
    """
    series = _validate_series(series, "series")
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = np.var(x)

    def regression(values: np.ndarray) -> Tuple[float, float]:
        y = values
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        if x_var == 0:
            return 0.0, 0.0
        slope = cov / (window * x_var)
        intercept = y_mean - slope * x_mean
        y_hat = slope * x + intercept
        ss_tot = ((y - y_mean) ** 2).sum()
        ss_res = ((y - y_hat) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(slope), float(np.clip(r2, 0.0, 1.0))

    slopes = []
    r2_scores = []
    values = series.values
    for i in range(len(series)):
        if i + 1 < window:
            slopes.append(np.nan)
            r2_scores.append(np.nan)
            continue
        window_values = values[i + 1 - window : i + 1]
        slope, r2 = regression(window_values)
        slopes.append(slope)
        r2_scores.append(r2)

    return pd.Series(slopes, index=series.index), pd.Series(r2_scores, index=series.index)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int):
    """
    Calculate classic ATR.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def atr_percentile(atr: pd.Series, window: int) -> pd.Series:
    """
    Convert ATR values to a rolling percentile rank [0,1].
    """
    atr = _validate_series(atr, "atr")

    def percentile(values: np.ndarray) -> float:
        if len(values) == 0:
            return np.nan
        last = values[-1]
        rank = (values <= last).sum() - 1
        return rank / max(len(values) - 1, 1)

    return atr.rolling(window=window, min_periods=10).apply(percentile, raw=True)


def bollinger_band_width(series: pd.Series, window: int, num_std: float = 2.0):
    """
    Compute Bollinger Band width normalised by price.
    """
    series = _validate_series(series, "series")
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    width = (2 * num_std * rolling_std) / rolling_mean.replace(0.0, np.nan)
    return width.replace([np.inf, -np.inf], np.nan)


def compression_score(
    close: pd.Series,
    band_width: pd.Series,
    atr_rank: pd.Series,
    window: int,
) -> pd.Series:
    """
    Heuristic compression score combining price tightness and low volatility.
    """
    close = _validate_series(close, "close")
    band_width = _validate_series(band_width, "band_width")
    atr_rank = _validate_series(atr_rank, "atr_rank")

    zscore = (close - close.rolling(window).mean()) / (close.rolling(window).std())
    zscore = zscore.abs().fillna(0.0)

    components = pd.concat(
        [
            1 - band_width.rank(pct=True),
            1 - atr_rank,
            1 - zscore.rank(pct=True),
        ],
        axis=1,
    )
    score = components.mean(axis=1)
    return score.rolling(window=window, min_periods=window // 2).mean().clip(0, 1)


def volume_health(volume: pd.Series, window: int) -> pd.Series:
    volume = _validate_series(volume, "volume")
    z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
    return np.tanh(z.fillna(0.0) / 3)


@dataclass(frozen=True)
class TrendFeatureSet:
    hurst: pd.Series
    slope: pd.Series
    r2: pd.Series
    trend_score: pd.Series


@dataclass(frozen=True)
class VolatilityFeatureSet:
    atr: pd.Series
    atr_percentile: pd.Series
    bollinger_width: pd.Series


@dataclass(frozen=True)
class StructureFeatureSet:
    compression: pd.Series
    volume_health: Optional[pd.Series] = None


