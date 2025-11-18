"""
Utility functions for constructing training labels and applying target transforms.

Centralises the logic for:
- Log-magnitude targets used by the return regression model.
- Rolling quantile classification labels with look-ahead protection.
- Rolling RMS volatility proxy without leaking future information.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def log_return_magnitude(y_return: pd.Series) -> pd.Series:
    """Project raw returns into log-amplitude space (>=0)."""
    log_mag = np.log1p(np.abs(y_return.to_numpy()))
    return pd.Series(log_mag, index=y_return.index, name="log_return_magnitude")


def invert_log_return_magnitude(values: np.ndarray | pd.Series) -> np.ndarray:
    """Invert log-amplitude predictions back to absolute return magnitude."""
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)  # clip tiny negatives introduced by the model
    return np.expm1(arr)


def rolling_rms_volatility(
    y_return: pd.Series,
    window: int = 5,
    min_periods: int = 2,
) -> pd.Series:
    """
    Compute a simple rolling RMS proxy for volatility (uses trailing returns only).
    """

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

    rms = y_return.rolling(window=window, min_periods=min_periods).apply(_rms, raw=True)
    # Fallback to |r| when insufficient history is available
    rms = rms.fillna(np.abs(y_return))
    return rms.rename("future_volatility")


def rolling_quantile_classification_labels(
    y_return: pd.Series,
    window: int = 20,
    lower_quantile: float = 0.4,
    upper_quantile: float = 0.6,
    min_periods: int = 20,
) -> Tuple[pd.Series, np.ndarray, pd.Series, pd.Series]:
    """
    Build symmetric up/down labels using rolling quantile thresholds computed on
    shifted returns to avoid forward-looking leakage.

    Returns:
        Tuple of (y_class, valid_mask, upper, lower):
        - y_class: Full-length Series with binary labels (1=up, 0=down, NaN=invalid)
                  Same index as y_return, preserving alignment
        - valid_mask: Boolean array indicating which samples are valid (not NaN)
        - upper: Upper quantile threshold Series
        - lower: Lower quantile threshold Series
    """
    shifted = y_return.shift(1)
    upper = shifted.rolling(window=window, min_periods=min_periods).quantile(
        upper_quantile
    )
    lower = shifted.rolling(window=window, min_periods=min_periods).quantile(
        lower_quantile
    )

    labels = pd.Series(np.nan, index=y_return.index)
    valid_window = (~upper.isna()) & (~lower.isna())
    labels.loc[valid_window & (y_return > upper)] = 1
    labels.loc[valid_window & (y_return < lower)] = 0

    valid_mask = labels.notna()
    # Return full-length Series (with NaN for invalid samples) to preserve index alignment
    # This ensures the returned Series has the same index as y_return
    y_class = labels.astype("Int64")  # Use nullable integer type to preserve NaN
    return y_class, valid_mask.to_numpy(), upper, lower
