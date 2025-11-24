"""
趋势跟踪策略标签：收益率百分位（Rank）

标签定义：
未来 N 根K线收益率在滚动窗口中的分位数（如 90% 分位 = 强趋势）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_trend_following_label(
    df: pd.DataFrame,
    price_col: str = "close",
    horizon: int = 50,
    rank_window: int = 200,
    min_periods: int = 50,
) -> pd.Series:
    """
    计算趋势跟踪策略的百分位标签

    Args:
        df: DataFrame with price data
        price_col: Price column
        horizon: Future return horizon (number of bars)
        rank_window: Rolling window for rank calculation
        min_periods: Minimum periods for rank calculation

    Returns:
        Series with rank labels (0.0 to 1.0, NaN for insufficient data)
    """
    if price_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Calculate future return
    price = df[price_col]
    future_return = (price.shift(-horizon) - price) / price

    # Calculate rolling rank (percentile)
    # For each point, rank its future_return within the rolling window
    labels = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        if i < min_periods or i >= len(df) - horizon:
            continue

        # Rolling window: [i - rank_window, i]
        window_start = max(0, i - rank_window)
        window_end = i + 1

        # Get future returns in the window
        window_returns = future_return.iloc[window_start:window_end]

        # Current future return
        current_return = future_return.iloc[i]

        if pd.isna(current_return):
            continue

        # Calculate rank (percentile)
        valid_returns = window_returns.dropna()
        if len(valid_returns) < min_periods:
            continue

        # Rank: proportion of values <= current_return
        rank = (valid_returns <= current_return).sum() / len(valid_returns)
        labels[i] = rank

    return pd.Series(labels, index=df.index)
