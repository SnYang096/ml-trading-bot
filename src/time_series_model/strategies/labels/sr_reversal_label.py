"""
SR 反转策略标签：二元标签（≥2R 成功率）

标签定义：
在 SR 区入场后，动态检查未来是否先触达 +2R 止盈 而非 -1R 止损？
（hold_bars 只是寻找上限，实际可能在更早的 K 线就满足条件）

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
    max_holding_bars: int = 50,  # 只是寻找上限，实际会动态检查
    rr_ratio: float = 2.0,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
) -> pd.Series:
    """
    计算 SR 反转策略的二元标签（动态检查 R/R，而非固定 hold_bars）

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Signal column (1=Long, -1=Short, 0=Hold)
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period (只是寻找上限，实际会动态检查)
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

    # 动态检查：只要满足 R/R 条件就停止，不需要等到 max_holding_bars
    max_i = len(df) - max_holding_bars - 1

    for i in range(max_i):
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

        # 动态扫描未来价格路径（最多 max_holding_bars，但一旦满足条件就停止）
        hit_tp = False
        hit_sl = False
        tp_bar = None
        sl_bar = None

        # 扫描范围：从 i+1 到 i+1+max_holding_bars（但不一定扫完）
        end_idx = min(i + 1 + max_holding_bars, len(df))

        for j in range(i + 1, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                # 检查是否触达止盈（使用 high 模拟 intra-bar 执行）
                if not hit_tp and high >= take_profit:
                    hit_tp = True
                    tp_bar = j - i  # 相对于入场的 bar 数
                    # 一旦触达止盈，立即停止扫描（动态检查的核心）
                    break
                # 检查是否触达止损（使用 low 模拟 intra-bar 执行）
                if not hit_sl and low <= stop_loss:
                    hit_sl = True
                    sl_bar = j - i
                    # 一旦触达止损，立即停止扫描
                    break
            else:  # Short
                # 检查是否触达止盈（使用 low 模拟 intra-bar 执行）
                if not hit_tp and low <= take_profit:
                    hit_tp = True
                    tp_bar = j - i
                    break
                # 检查是否触达止损（使用 high 模拟 intra-bar 执行）
                if not hit_sl and high >= stop_loss:
                    hit_sl = True
                    sl_bar = j - i
                    break

        # Label: 1 if TP hit first (or TP hit and SL never hit), 0 otherwise
        # 如果既没触达 TP 也没触达 SL（timeout），则标记为 NaN（排除训练）
        if hit_tp and (
            not hit_sl
            or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
        ):
            labels[i] = 1.0  # Success
        elif hit_sl:
            labels[i] = 0.0  # Failure
        # else: timeout (no TP, no SL within max_holding_bars) -> NaN (exclude from training)

    return pd.Series(labels, index=df.index)
