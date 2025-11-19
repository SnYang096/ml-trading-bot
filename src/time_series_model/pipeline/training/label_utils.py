"""
Utility functions for constructing training labels and applying target transforms.

Centralises the logic for:
- Log-magnitude targets used by the return regression model.
- Rolling quantile classification labels with look-ahead protection.
- Rolling RMS volatility proxy without leaking future information.
- Volatility-normalized targets (Sharpe-like targets).
- Historical quantile labels for evaluation and signal generation.
- Tradable mask for filtering low-quality samples.
- Trend strength as sample weights.
"""

from __future__ import annotations

from typing import Tuple, Optional

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
    window: int = 60,
    min_periods: int = 60,
) -> pd.Series:
    """
    Compute a simple rolling RMS proxy for volatility (uses trailing returns only).
    """

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

    rms = y_return.rolling(window=window, min_periods=min_periods).apply(_rms, raw=True)
    # Fallback to |r| when insufficient history is available
    rms = rms.fillna(np.abs(y_return))
    # Return Series with name (but don't use rename to avoid issues in groupby)
    rms.name = "future_volatility"
    return rms


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


def volatility_normalized_target(
    future_return: pd.Series,
    rolling_vol: pd.Series,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Create volatility-normalized target (Sharpe-like target).

    This normalizes future returns by rolling volatility to create a more
    stable and learnable target that adapts to different volatility regimes.

    Args:
        future_return: Future return series
        rolling_vol: Rolling volatility series (same index as future_return)
        eps: Small value to avoid division by zero

    Returns:
        Volatility-normalized target: future_return / (rolling_vol + eps)
    """
    target = future_return / (rolling_vol + eps)
    return target.rename("volatility_normalized_target")


def historical_quantile_label(
    future_return: pd.Series,
    lookback_window: int = 60,
    hold_period: int = 5,
    min_samples: int = 30,
    asset_col: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute historical quantile label: future_return's position in its historical distribution.

    This calculates what percentile the current future_return falls into relative to
    its historical rolling window, which is useful for:
    - Adaptive evaluation (what is "strong" depends on historical context)
    - Signal generation (only trade when return is in extreme quantiles)

    Args:
        future_return: Future return series
        lookback_window: Number of periods to look back for historical distribution
        hold_period: Holding period (to avoid lookahead bias)
        min_samples: Minimum samples required for quantile calculation
        asset_col: Optional asset identifier series for multi-asset data

    Returns:
        Quantile series (0-1), where 1 means current return is in top percentile
    """

    def _compute_quantile(group: pd.Series) -> pd.Series:
        """Compute quantile for a single asset/series."""
        quantiles = []
        returns = group.values

        for i in range(len(group)):
            # Need enough history before this point
            if i < lookback_window + hold_period:
                quantiles.append(np.nan)
                continue

            # Get historical returns (excluding current and future)
            start_idx = max(0, i - lookback_window - hold_period)
            end_idx = i - hold_period
            hist_rets = returns[start_idx:end_idx]
            hist_rets = hist_rets[~np.isnan(hist_rets)]

            if len(hist_rets) < min_samples:
                quantiles.append(np.nan)
                continue

            current = returns[i]
            if np.isnan(current):
                quantiles.append(np.nan)
                continue

            # Compute quantile: what percentile is current return in historical distribution?
            q = (hist_rets < current).mean()
            quantiles.append(q)

        return pd.Series(quantiles, index=group.index)

    if asset_col is not None:
        # Multi-asset: compute quantile within each asset
        result = future_return.groupby(asset_col).apply(_compute_quantile)
        # If result is MultiIndex, drop the asset level
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        result = result.reindex(future_return.index)
    else:
        # Single asset
        result = _compute_quantile(future_return)

    return result.rename("return_quantile")


def tradable_mask(
    future_return: pd.Series,
    rolling_vol: pd.Series,
    return_quantile: pd.Series,
    vol_mult: float = 0.5,
    quantile_lower: float = 0.1,
    quantile_upper: float = 0.9,
) -> pd.Series:
    """
    Create tradable mask: filter low-quality samples.

    Only keep samples where:
    - |future_return| > vol_mult * rolling_vol (signal is strong enough)
    - return_quantile is in [quantile_lower, quantile_upper] (exclude extreme tails)

    This filters out:
    - Noise trading (weak signals)
    - Extreme outliers (likely data errors or black swan events)

    Args:
        future_return: Future return series
        rolling_vol: Rolling volatility series
        return_quantile: Historical quantile label
        vol_mult: Multiplier for volatility threshold
        quantile_lower: Lower bound for quantile (exclude bottom tail)
        quantile_upper: Upper bound for quantile (exclude top tail)

    Returns:
        Boolean series indicating tradable samples
    """
    # Signal strength check: return must be significant relative to volatility
    signal_strong = future_return.abs() > (vol_mult * rolling_vol)

    # Quantile check: exclude extreme tails (likely noise or outliers)
    quantile_valid = return_quantile.between(
        quantile_lower, quantile_upper, inclusive="neither"
    )

    # Both conditions must be true
    tradable = signal_strong & quantile_valid

    return tradable.rename("tradable")


def trend_strength_weight(
    momentum: pd.Series,
    rolling_vol: pd.Series,
    eps: float = 1e-8,
    clip_lower: float = 0.1,
    clip_upper: float = 5.0,
) -> pd.Series:
    """
    Compute trend strength as sample weight.

    Trend strength = |momentum| / rolling_vol

    This gives higher weight to samples with strong trends relative to volatility,
    which helps the model focus on learnable patterns rather than noise.

    Args:
        momentum: Momentum series (e.g., moving average slope or price change)
        rolling_vol: Rolling volatility series
        eps: Small value to avoid division by zero
        clip_lower: Lower bound for clipping weights
        clip_upper: Upper bound for clipping weights

    Returns:
        Sample weight series (higher = stronger trend)
    """
    trend_strength = (momentum.abs() / (rolling_vol + eps)).clip(
        lower=clip_lower, upper=clip_upper
    )
    return trend_strength.rename("trend_strength")


def compute_momentum(
    price: pd.Series,
    window: int = 20,
    diff_period: int = 5,
    asset_col: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute momentum feature from price series.

    If momentum is not available, this provides a simple alternative:
    momentum = (MA(window) - MA(window).shift(diff_period)) / price

    Args:
        price: Price series (e.g., close price)
        window: Moving average window
        diff_period: Period for computing difference
        asset_col: Optional asset identifier for multi-asset data

    Returns:
        Momentum series
    """

    def _compute_momentum(group: pd.Series) -> pd.Series:
        ma = group.rolling(window=window, min_periods=window).mean()
        momentum = ma.diff(diff_period) / group
        return momentum

    if asset_col is not None:
        result = price.groupby(asset_col).apply(_compute_momentum)
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        # Ensure it's a Series
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        result = result.reindex(price.index)
    else:
        result = _compute_momentum(price)

    # Ensure it's a Series and set name
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    result.name = "momentum"
    return result


def smooth_target(
    future_return: pd.Series,
    method: str = "moving_average",
    window: int = 3,
    span: Optional[float] = None,
    asset_col: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Smooth target variable to reduce noise.

    Methods:
    - "moving_average": Simple moving average
    - "ewm": Exponential weighted moving average
    - "quantile": Quantile transformation (maps to normal distribution)

    Args:
        future_return: Future return series
        method: Smoothing method ("moving_average", "ewm", or "quantile")
        window: Window size for moving average
        span: Span for exponential weighted moving average
        asset_col: Optional asset identifier for multi-asset data

    Returns:
        Smoothed target series
    """

    def _smooth_series(series: pd.Series) -> pd.Series:
        if method == "moving_average":
            return series.rolling(window=window, min_periods=1).mean().shift(-1)
        elif method == "ewm":
            span_val = span if span is not None else window
            return series.ewm(span=span_val).mean().shift(-1)
        elif method == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            qt = QuantileTransformer(
                output_distribution="normal", n_quantiles=min(1000, len(series))
            )
            # Only transform non-NaN values
            valid_mask = series.notna()
            if valid_mask.sum() > 0:
                transformed = series.copy()
                transformed[valid_mask] = qt.fit_transform(
                    series[valid_mask].values.reshape(-1, 1)
                ).flatten()
                return transformed
            else:
                return series
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    if asset_col is not None:
        result = future_return.groupby(asset_col).apply(_smooth_series)
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        result = result.reindex(future_return.index)
    else:
        result = _smooth_series(future_return)

    return result.rename(
        f"smoothed_{future_return.name}"
        if hasattr(future_return, "name")
        else "smoothed_target"
    )
