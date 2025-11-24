"""
SR 反转策略标签：二元标签（≥2R 成功率）

标签定义：
在 SR 区入场后，50根K线内是否先触达 +2R 止盈 而非 -1R 止损？

R = 1×ATR
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_sr_reversal_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    hold_bars: int = 50,
    rr_ratio: float = 2.0,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
) -> pd.Series:
    """
    计算 SR 反转策略的二元标签

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Signal column (1=Long, -1=Short, 0=Hold)
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        hold_bars: Maximum holding period
        rr_ratio: Risk-reward ratio (default: 2.0)
        stop_loss_r: Stop loss in R units (default: 1.0)
        take_profit_r: Take profit in R units (default: 2.0)

    Returns:
        Series with binary labels (1.0 = success, 0.0 = failure, NaN = no signal)
    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Ensure ATR exists
    if atr_col not in df.columns:
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            try:
                import talib

                high = df["high"].values
                low = df["low"].values
                close = df[price_col].values
                atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
                df[atr_col] = pd.Series(atr_values, index=df.index)
            except ImportError:
                # Fallback: simple ATR calculation
                tr = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df[price_col].shift(1)),
                        abs(df["low"] - df[price_col].shift(1)),
                    ),
                )
                df[atr_col] = tr.rolling(window=atr_window, min_periods=1).mean()
        else:
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    signals = df[signal_col].values
    atr_arr = df[atr_col].values
    high_arr = df[high_col].values
    low_arr = df[low_col].values
    price_arr = df[price_col].values

    labels = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df) - hold_bars - 1):
        signal = signals[i]

        if pd.isna(signal) or signal == 0:
            continue

        atr = atr_arr[i]
        if pd.isna(atr) or atr <= 0:
            continue

        # Entry price: next bar's open (approximated by current close)
        entry_price = price_arr[i]

        # Set TP/SL based on signal direction
        if signal > 0:  # Long
            stop_loss = entry_price - stop_loss_r * atr
            take_profit = entry_price + take_profit_r * atr
        else:  # Short
            stop_loss = entry_price + stop_loss_r * atr
            take_profit = entry_price - take_profit_r * atr

        # Scan future price path
        future_highs = high_arr[i + 1 : i + 1 + hold_bars]
        future_lows = low_arr[i + 1 : i + 1 + hold_bars]

        if len(future_highs) == 0:
            continue

        # Check which is hit first: TP or SL
        hit_tp = False
        hit_sl = False
        tp_bar = None
        sl_bar = None

        for j, (h, l) in enumerate(zip(future_highs, future_lows)):
            if signal > 0:  # Long
                if h >= take_profit:
                    hit_tp = True
                    tp_bar = i + 1 + j
                    break
                if l <= stop_loss:
                    hit_sl = True
                    sl_bar = i + 1 + j
                    break
            else:  # Short
                if l <= take_profit:
                    hit_tp = True
                    tp_bar = i + 1 + j
                    break
                if h >= stop_loss:
                    hit_sl = True
                    sl_bar = i + 1 + j
                    break

        # Label: 1 if TP hit first (or TP hit and SL never hit), 0 otherwise
        if hit_tp and (
            not hit_sl
            or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
        ):
            labels[i] = 1.0  # Success
        elif hit_sl:
            labels[i] = 0.0  # Failure
        # else: timeout (no TP, no SL) -> NaN (exclude from training)

    return pd.Series(labels, index=df.index)
