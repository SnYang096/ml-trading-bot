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


def historical_rolling_volatility(
    y_return: pd.Series,
    window: int = 60,
    min_periods: int = 60,
) -> pd.Series:
    """
    Compute historical rolling RMS volatility (for use as a FEATURE).

    This function computes trailing volatility over the past window periods.
    At time t, it returns: RMS(r_{t-window+1}, r_{t-window+2}, ..., r_t)

    ✅ This is safe to use as a FEATURE because it only uses historical data.
    For future volatility LABELS, use `future_volatility_label()` instead.

    Args:
        y_return: Historical return series (e.g., from price.pct_change())
        window: Rolling window size
        min_periods: Minimum periods required for calculation

    Returns:
        Historical rolling RMS volatility series (suitable for features)
    """

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

    rms = y_return.rolling(window=window, min_periods=min_periods).apply(_rms, raw=True)
    # Fallback to |r| when insufficient history is available
    rms = rms.fillna(np.abs(y_return))
    rms.name = "rolling_vol"
    return rms


# Backward compatibility alias
def rolling_rms_volatility(
    y_return: pd.Series,
    window: int = 60,
    min_periods: int = 60,
) -> pd.Series:
    """
    Deprecated alias for historical_rolling_volatility().

    Use historical_rolling_volatility() for clarity.
    """
    return historical_rolling_volatility(y_return, window, min_periods)


def future_volatility_label(
    price_series: pd.Series,
    horizon: int = 24,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute future volatility label: RMS of future single-period returns.

    This function computes the realized volatility over the future H periods
    as a supervision target. At time t, the label is:
        vol[t] = RMS(r_{t+1}, r_{t+2}, ..., r_{t+H})
    where r_{t+i} = (price[t+i] - price[t+i-1]) / price[t+i-1]

    ⚠️  This uses future information, which is correct for LABELS but would
    cause leakage if used as a FEATURE.

    Args:
        price_series: Price series (e.g., 'close')
        horizon: Number of future periods to compute volatility over
        min_periods: Minimum periods required (default: horizon)

    Returns:
        Future volatility label series (aligned to current time index)
    """
    if min_periods is None:
        min_periods = horizon

    # Compute single-period returns
    returns = price_series.pct_change().dropna()

    # Compute future volatility: RMS of returns over [t+1, t+horizon]
    # Since pandas rolling doesn't support "future windows", we compute manually
    future_vol = pd.Series(
        index=price_series.index, dtype=float, name="future_volatility"
    )

    for i in range(len(price_series)):
        # Get future returns: [i+1, i+horizon]
        start_idx = i + 1
        end_idx = min(i + horizon + 1, len(returns))

        if end_idx - start_idx < min_periods:
            future_vol.iloc[i] = np.nan
            continue

        # Extract future returns and compute RMS
        future_rets = returns.iloc[start_idx:end_idx].values
        if len(future_rets) >= min_periods:
            future_vol.iloc[i] = np.sqrt(np.mean(np.square(future_rets)))
        else:
            future_vol.iloc[i] = np.nan

    # Set name for clarity
    future_vol.name = "future_volatility"
    return future_vol


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


def compute_rr_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    rr_ratio: float = 2.0,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    use_continuous_label: bool = False,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
) -> pd.Series:
    """
    计算基于风险回报比（R/R）的标签，匹配 SR 策略的真实交易逻辑。

    策略逻辑：
    - 在高概率区域入场（由信号决定方向）
    - 设定 1R 止损
    - 捕捉至少 2R 的有利运行
    - 动态止盈止损

    标签计算：
    - 二元标签（use_continuous_label=False）：是否实现 ≥2R（1=成功，0=失败）
    - 连续标签（use_continuous_label=True）：实际实现的盈亏比（Realized R/R）

    Args:
        df: DataFrame with OHLCV data, signals, and ATR
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        price_col: Column name for price (used for entry price)
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        rr_ratio: Target risk-reward ratio (default: 2.0)
        max_holding_bars: Maximum holding period
        stop_loss_r: Stop loss in R units (default: 1.0)
        take_profit_r: Take profit in R units (default: 2.0)
        use_continuous_label: If True, return realized R/R; if False, return binary label
        entry_price_col: Column to use for entry price. Default: 'open' if available else price_col.
        entry_offset: Bars to wait after signal before entering (>=0). entry_offset=1 = next bar entry.

    Returns:
        Series with R/R labels (binary or continuous)
    """
    if signal_col not in df.columns:
        # If no signal column, return NaN series
        return pd.Series(np.nan, index=df.index)

    # Ensure ATR exists
    if atr_col not in df.columns:
        # Compute ATR if not available
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            import talib

            high = df["high"].values
            low = df["low"].values
            close = df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            df[atr_col] = pd.Series(atr_values, index=df.index)
        else:
            # Fallback: use a simple volatility proxy
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    signals = df[signal_col]
    atr_series = df[atr_col]

    if entry_offset < 0:
        raise ValueError("entry_offset must be >= 0")

    if entry_price_col and entry_price_col not in df.columns:
        raise ValueError(
            f"entry_price_col='{entry_price_col}' not found in DataFrame columns"
        )

    if entry_price_col:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        # For offset 0, still shift by 0 (current bar price)
        entry_prices = entry_series.copy()

    # Initialize labels with NaN (will be set to 0.0 for valid samples)
    labels = pd.Series(np.nan, index=df.index, dtype=float)

    # Get high/low columns for efficient access
    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    # Pre-extract arrays for faster access (avoid repeated iloc calls)
    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    atr_arr = atr_series.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    # Process each sample
    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        # Skip if no signal
        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]
        atr = atr_arr[i]

        # Skip if invalid entry price or ATR
        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            continue

        # Calculate stop loss and take profit
        if signal > 0:  # Long signal
            stop_loss = entry_price - stop_loss_r * atr
            take_profit = entry_price + take_profit_r * atr
        else:  # Short signal
            stop_loss = entry_price + stop_loss_r * atr
            take_profit = entry_price - take_profit_r * atr

        # Scan future price path (up to max_holding_bars)
        # This is the core logic: check if TP or SL is hit first
        hit_tp = False
        hit_sl = False
        tp_bar = None
        sl_bar = None
        max_favorable = 0.0
        max_adverse = 0.0

        # Scan from first tradable bar after entry
        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                # Check if TP or SL hit (using high/low to simulate intra-bar execution)
                if not hit_tp and high >= take_profit:
                    hit_tp = True
                    tp_bar = j - i  # Bar number relative to entry
                if not hit_sl and low <= stop_loss:
                    hit_sl = True
                    sl_bar = j - i  # Bar number relative to entry

                # Track MFE/MAE for continuous label
                if use_continuous_label:
                    max_favorable = max(max_favorable, (high - entry_price) / atr)
                    max_adverse = max(max_adverse, (entry_price - low) / atr)
            else:  # Short
                # Check if TP or SL hit
                if not hit_tp and low <= take_profit:
                    hit_tp = True
                    tp_bar = j - i
                if not hit_sl and high >= stop_loss:
                    hit_sl = True
                    sl_bar = j - i

                # Track MFE/MAE for continuous label
                if use_continuous_label:
                    max_favorable = max(max_favorable, (entry_price - low) / atr)
                    max_adverse = max(max_adverse, (high - entry_price) / atr)

            # Early exit: if both hit, check which came first
            if hit_tp and hit_sl:
                break

        # Calculate label
        if use_continuous_label:
            # Continuous label: Realized R/R
            # MFE / MAE, truncated at [0, take_profit_r]
            if max_adverse > 0:
                realized_rr = min(max_favorable, take_profit_r) / max(max_adverse, 0.1)
                realized_rr = min(realized_rr, take_profit_r)  # Cap at target R/R
            else:
                realized_rr = 0.0
            labels.iloc[i] = realized_rr
        else:
            # Binary label: Did we achieve ≥2R?
            # Success = hit TP before SL (or hit TP and never hit SL)
            if hit_tp and (
                not hit_sl
                or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
            ):
                labels.iloc[i] = 1.0  # Success: hit TP before SL
            else:
                labels.iloc[i] = (
                    0.0  # Failure: hit SL first, or timeout without hitting TP
                )

    return labels


def classify_sr_reaction(
    df: pd.DataFrame,
    signal_col: str = "signal",
    sr_zone_col: Optional[str] = None,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    lookback_window: int = 5,
    threshold_factor: float = 0.5,
) -> pd.Series:
    """
    对每个 SR 信号，判断后续价格是"反向"（reversal）还是"突破"（breakout）。

    这是 SR 策略建模的关键拆分：
    - 反转型机会：价格回测 SR 区 → 试探（wick 进入）→ 快速反向运行（核心盈利来源）
    - 突破型机会：价格强势穿过 SR 区，继续沿原方向运行（趋势延续，非反转）

    逻辑：
    - Long 信号（期待从需求区反弹）：
        - 如果价格深跌破区域（< zone_price - threshold * ATR）→ 突破型（反向失败）
        - 如果价格快速上涨（> zone_price + 2 * ATR）且未深跌 → 突破型（强势突破）
        - 否则（有下影线试探后上涨）→ 反转型
    - Short 信号（期待从供给区回落）：
        - 如果价格强势突破区域（> zone_price + threshold * ATR）→ 突破型（反向失败）
        - 如果价格快速下跌（< zone_price - 2 * ATR）且未突破 → 突破型（强势突破）
        - 否则（有上影线试探后下跌）→ 反转型

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        sr_zone_col: Column name for SR zone price (if None, will try to infer from features)
        price_col: Column name for price
        high_col: Column name for high
        low_col: Column name for low
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        lookback_window: Number of future bars to observe
        threshold_factor: Factor for determining "deep penetration" (default: 0.5)

    Returns:
        Series with reaction types: 'reversal', 'breakout', or np.nan
    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Ensure ATR exists
    if atr_col not in df.columns:
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            import talib

            high = df["high"].values
            low = df["low"].values
            close = df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            df[atr_col] = pd.Series(atr_values, index=df.index)
        else:
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    signals = df[signal_col]
    atr_series = df[atr_col]

    # Try to find SR zone price
    zone_price = None
    if sr_zone_col and sr_zone_col in df.columns:
        zone_price = df[sr_zone_col]
    else:
        # Try to infer from features
        for col in ["nearest_sr", "dist_to_nearest_sr", "sqs"]:
            if col in df.columns:
                # If we have dist_to_nearest_sr, we can compute zone_price
                if col == "dist_to_nearest_sr" and price_col in df.columns:
                    # dist_to_nearest_sr is normalized distance, need to convert back
                    # For now, use current price as proxy (will be refined)
                    zone_price = df[price_col]
                    break
                elif col == "nearest_sr" and col in df.columns:
                    zone_price = df[col]
                    break

        if zone_price is None:
            # Fallback: use current price as proxy (not ideal, but allows function to run)
            zone_price = df[price_col]

    reactions = pd.Series(np.nan, index=df.index, dtype=object)

    # Pre-extract arrays for faster access
    signals_arr = signals.values
    atr_arr = atr_series.values
    zone_price_arr = (
        zone_price.values if isinstance(zone_price, pd.Series) else zone_price
    )
    high_arr = df[high_col].values
    low_arr = df[low_col].values
    price_arr = df[price_col].values

    max_i = len(df) - lookback_window - 1

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        atr = atr_arr[i]
        zone = (
            zone_price_arr[i]
            if isinstance(zone_price_arr, np.ndarray)
            else zone_price_arr
        )

        if pd.isna(atr) or atr <= 0 or pd.isna(zone):
            continue

        # Observe future price behavior
        end_idx = min(i + 1 + lookback_window, len(df))
        future_highs = high_arr[i + 1 : end_idx]
        future_lows = low_arr[i + 1 : end_idx]

        if len(future_highs) == 0:
            continue

        min_low = np.min(future_lows)
        max_high = np.max(future_highs)
        threshold = threshold_factor * atr

        if signal > 0:  # Long signal (expecting bounce from demand zone)
            # Check for deep penetration below zone (breakout failure)
            if min_low < zone - threshold:
                # Strong breakdown → breakout type (reversal failed)
                reactions.iloc[i] = "breakout"
            # Check for strong upward move without deep penetration
            elif max_high > zone + 2 * atr and min_low > zone - 0.5 * atr:
                # Fast upward move, no deep dip → likely strong breakout (not reversal)
                reactions.iloc[i] = "breakout"
            else:
                # Has lower wick test then bounce → typical reversal
                reactions.iloc[i] = "reversal"
        else:  # Short signal (expecting rejection from supply zone)
            # Check for strong penetration above zone (breakout failure)
            if max_high > zone + threshold:
                # Strong breakout → breakout type (reversal failed)
                reactions.iloc[i] = "breakout"
            # Check for strong downward move without penetration
            elif min_low < zone - 2 * atr and max_high < zone + 0.5 * atr:
                # Fast downward move, no penetration → likely strong breakdown (not reversal)
                reactions.iloc[i] = "breakout"
            else:
                # Has upper wick test then rejection → typical reversal
                reactions.iloc[i] = "reversal"

    return reactions


def compute_rr_label_by_reaction(
    df: pd.DataFrame,
    signal_col: str = "signal",
    reaction_col: str = "sr_reaction",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    rr_ratio: float = 2.0,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    reaction_type: Optional[str] = None,  # 'reversal' or 'breakout' or None (both)
    use_continuous_label: bool = False,
) -> pd.Series:
    """
    计算基于风险回报比（R/R）的标签，支持按反应类型（反转 vs 突破）分类。

    这是 SR 策略精细化建模的关键：将反转型和突破型机会分开处理。

    Args:
        df: DataFrame with OHLCV data, signals, and reaction types
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        reaction_col: Column name for SR reaction type ('reversal' or 'breakout')
        price_col: Column name for price
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        rr_ratio: Target risk-reward ratio (default: 2.0)
        max_holding_bars: Maximum holding period
        stop_loss_r: Stop loss in R units (default: 1.0)
        take_profit_r: Take profit in R units (default: 2.0)
        reaction_type: Filter by reaction type ('reversal', 'breakout', or None for both)
        use_continuous_label: If True, return realized R/R; if False, return binary label

    Returns:
        Series with R/R labels (binary or continuous), NaN for filtered samples
    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Filter by reaction type if specified
    if reaction_type and reaction_col in df.columns:
        mask = df[reaction_col] == reaction_type
        if not mask.any():
            print(f"   ⚠️  Warning: No samples with reaction_type='{reaction_type}'")
            return pd.Series(np.nan, index=df.index)
    else:
        mask = pd.Series(True, index=df.index)

    # Compute R/R label for all samples first
    rr_labels = compute_rr_label(
        df,
        signal_col=signal_col,
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=rr_ratio,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=use_continuous_label,
    )

    # Filter by reaction type
    if reaction_type and reaction_col in df.columns:
        rr_labels = rr_labels.where(mask)

    return rr_labels


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
            # 使用 shift(1) 而不是 shift(-1) 以避免数据泄漏
            # shift(1) 确保 t 时刻的平滑值只使用 t-1 及之前的信息
            return series.rolling(window=window, min_periods=1).mean().shift(1)
        elif method == "ewm":
            span_val = span if span is not None else window
            # 使用 shift(1) 而不是 shift(-1) 以避免数据泄漏
            return series.ewm(span=span_val).mean().shift(1)
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
