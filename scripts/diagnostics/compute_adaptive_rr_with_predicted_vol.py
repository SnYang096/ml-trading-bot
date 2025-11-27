"""
使用预测波动率计算自适应R/R标签

这个函数类似于compute_adaptive_rr_label_with_future_vol，但使用预测波动率而不是未来波动率。
适用于实盘交易和回测。
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_adaptive_rr_label_with_predicted_vol(
    df: pd.DataFrame,
    predicted_vol: np.ndarray,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    atr_lower_bound: float = 0.8,
    atr_upper_bound: float = 1.5,
    use_breakeven_stop: bool = True,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
) -> pd.Series:
    """
    基于预测波动率的自适应 R/R 标签

    核心思想：
    - 对于每个信号点，使用预测波动率（而不是未来波动率）
    - 使用该波动率动态调整止盈止损：
      - TP = entry ± (预测波动率 × take_profit_multiplier)
      - SL = entry ± (预测波动率 × stop_loss_multiplier)
    - 预测波动率会被限制在 [ATR × atr_lower_bound, ATR × atr_upper_bound] 范围内
    - 如果 use_breakeven_stop=True，当价格达到 stop_loss_multiplier × 预测波动率时，止损上移到保本

    适用于实盘交易和回测。

    Args:
        df: DataFrame with OHLCV data, signals, and ATR
        predicted_vol: Array of predicted volatility values (same length as df)
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        price_col: Column name for price
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period
        stop_loss_multiplier: Stop loss multiplier relative to predicted volatility
        take_profit_multiplier: Take profit multiplier relative to predicted volatility
        atr_lower_bound: Lower bound for predicted volatility (as multiple of ATR)
        atr_upper_bound: Upper bound for predicted volatility (as multiple of ATR)
        use_breakeven_stop: If True, move stop loss to breakeven when price reaches stop_loss_multiplier × predicted_vol
        entry_price_col: Column to use for entry price
        entry_offset: Bars to wait after signal before entering

    Returns:
        Series with R/R labels (1=success, 0=failure, NaN=no trade)
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

    if entry_price_col and entry_price_col in df.columns:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        entry_prices = entry_series.copy()

    # Clip predicted volatility to ATR bounds
    atr_values = atr_series.values
    final_vol = np.clip(
        predicted_vol,
        atr_values * atr_lower_bound,
        atr_values * atr_upper_bound,
    )

    # Initialize result
    labels = pd.Series(np.nan, index=df.index, dtype=float)

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]

        # 确保pred_vol索引有效
        if i >= len(final_vol):
            continue
        pred_vol = final_vol[i]

        if pd.isna(entry_price) or pd.isna(pred_vol) or pred_vol <= 0:
            continue

        # Calculate adaptive stop loss and take profit based on predicted volatility
        if signal > 0:  # Long signal
            initial_stop_loss = entry_price - stop_loss_multiplier * pred_vol
            take_profit = entry_price + take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price + stop_loss_multiplier * pred_vol
        else:  # Short signal
            initial_stop_loss = entry_price + stop_loss_multiplier * pred_vol
            take_profit = entry_price - take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price - stop_loss_multiplier * pred_vol

        hit_tp_flag = False
        hit_sl_flag = False
        stop_loss = initial_stop_loss
        breakeven_activated_flag = False

        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and high >= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and low <= stop_loss:
                    hit_sl_flag = True
            else:  # Short
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and low <= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and high >= stop_loss:
                    hit_sl_flag = True

            if hit_tp_flag and hit_sl_flag:
                break

        # Determine label
        if hit_tp_flag and (not hit_sl_flag or (hit_tp_flag and not hit_sl_flag)):
            labels.iloc[i] = 1.0
        else:
            labels.iloc[i] = 0.0

    return labels
