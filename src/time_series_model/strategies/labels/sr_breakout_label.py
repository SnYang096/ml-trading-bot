"""
SR 突破策略标签：连续标签（实现 R/R）

标签定义：
突破后最大有利偏移 / 最大不利偏移（MFE/MAE），截断在 [0, 3] 区间

动态检查：
一旦触达止损就停止扫描（因为 MAE 已经确定），max_holding_bars 只是寻找上限
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_sr_breakout_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,  # 只是寻找上限，实际会动态检查
    max_rr: float = 3.0,
    stop_loss_r: float = 1.0,
) -> pd.Series:
    """
    计算 SR 突破策略的连续标签（实现 R/R，动态检查）

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Signal column (1=Long, -1=Short, 0=Hold)
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period (只是寻找上限，实际会动态检查)
        max_rr: Maximum R/R to cap (default: 3.0)
        stop_loss_r: Stop loss in R units (default: 1.0)

    Returns:
        Series with continuous R/R labels (0 to max_rr, NaN for no signal)
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

    # 动态检查：一旦触达止损就停止，不需要等到 max_holding_bars
    max_i = len(df) - max_holding_bars - 1

    for i in range(max_i):
        signal = signals[i]

        if pd.isna(signal) or signal == 0:
            continue

        atr = atr_arr[i]
        if pd.isna(atr) or atr <= 0:
            continue

        # Entry price
        entry_price = price_arr[i]
        stop_loss = (
            entry_price - stop_loss_r * atr
            if signal > 0
            else entry_price + stop_loss_r * atr
        )

        # 动态扫描未来价格路径（最多 max_holding_bars，但一旦触达止损就停止）
        max_favorable = 0.0
        max_adverse = 0.0
        hit_stop_loss = False

        # 扫描范围：从 i+1 到 i+1+max_holding_bars（但不一定扫完）
        end_idx = min(i + 1 + max_holding_bars, len(df))

        for j in range(i + 1, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                # 检查是否触达止损（使用 low 模拟 intra-bar 执行）
                if not hit_stop_loss and low <= stop_loss:
                    hit_stop_loss = True
                    # 一旦触达止损，MAE 已经确定，立即停止扫描
                    max_adverse = entry_price - low
                    break

                # 更新 MFE 和 MAE
                max_favorable = max(max_favorable, high - entry_price)
                max_adverse = max(max_adverse, entry_price - low)
            else:  # Short
                # 检查是否触达止损（使用 high 模拟 intra-bar 执行）
                if not hit_stop_loss and high >= stop_loss:
                    hit_stop_loss = True
                    # 一旦触达止损，MAE 已经确定，立即停止扫描
                    max_adverse = high - entry_price
                    break

                # 更新 MFE 和 MAE
                max_favorable = max(max_favorable, entry_price - low)
                max_adverse = max(max_adverse, high - entry_price)

        # Calculate R/R: MFE / MAE (normalized by ATR)
        if max_adverse > 0:
            realized_rr = (max_favorable / atr) / (max_adverse / atr)
            # Cap at max_rr
            realized_rr = min(realized_rr, max_rr)
            # Floor at 0
            realized_rr = max(realized_rr, 0.0)
        else:
            # No adverse movement (ideal case)
            realized_rr = max_rr

        labels[i] = realized_rr

    return pd.Series(labels, index=df.index)
