"""
ME (MomentumExpansion) 策略标签

语义：压缩后波动/区间扩张，放量突破
核心逻辑：
1. 检测压缩区（低波动）
2. 检测扩张/突破（高波动 + 方向性）
3. 计算 forward RR
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.sr_reversal_label import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def detect_compression(
    df: pd.DataFrame,
    atr_col: str = "atr",
    lookback: int = 20,
    compression_percentile: float = 30,
) -> pd.Series:
    """
    检测压缩区（低波动区域）。

    压缩定义：当前 ATR 处于近期历史的低百分位

    Returns:
        pd.Series: 布尔 Series，True 表示处于压缩区
    """
    atr = df[atr_col]

    # 计算 ATR 的滚动百分位
    atr_pct = atr.rolling(window=lookback, min_periods=1).apply(
        lambda x: (
            (x.iloc[-1] <= np.percentile(x, compression_percentile))
            if len(x) > 1
            else False
        ),
        raw=False,
    )

    return atr_pct.fillna(False).astype(bool)


def detect_expansion_breakout(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    compression_mask: pd.Series = None,
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    atr_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """
    检测扩张/突破信号。

    突破定义：
    1. 之前处于压缩区
    2. 价格突破近期高/低
    3. 成交量放大
    4. ATR 扩张

    Returns:
        (breakout_up, breakout_down): 布尔 Series
    """
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]
    atr = df[atr_col]

    # 近期高低点
    rolling_high = high.shift(1).rolling(window=breakout_lookback, min_periods=1).max()
    rolling_low = low.shift(1).rolling(window=breakout_lookback, min_periods=1).min()

    # 价格突破
    price_breakout_up = close > rolling_high
    price_breakout_down = close < rolling_low

    # 成交量放大
    if volume_col in df.columns:
        volume = df[volume_col]
        avg_volume = (
            volume.shift(1).rolling(window=breakout_lookback, min_periods=1).mean()
        )
        volume_expansion = volume > (avg_volume * volume_mult)
    else:
        volume_expansion = pd.Series(True, index=df.index)

    # ATR 扩张（波动率上升）
    avg_atr = atr.shift(1).rolling(window=breakout_lookback, min_periods=1).mean()
    atr_expansion = atr > (avg_atr * 1.1)  # ATR 上升 10%

    # 压缩后突破（如果提供了压缩掩码）
    if compression_mask is not None:
        # 检查前 N 个 bar 是否有压缩
        was_compressed = (
            compression_mask.shift(1).rolling(window=5, min_periods=1).sum() > 0
        )
    else:
        was_compressed = pd.Series(True, index=df.index)

    # 综合条件
    breakout_up = price_breakout_up & volume_expansion & was_compressed
    breakout_down = price_breakout_down & volume_expansion & was_compressed

    return breakout_up, breakout_down


def compute_me_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # 压缩检测参数
    compression_lookback: int = 20,
    compression_percentile: float = 30,
    # 突破检测参数
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 ME (MomentumExpansion) 标签。

    逻辑：
    1. 检测压缩区（低波动）
    2. 检测扩张/突破（高波动 + 方向性）
    3. 顺势入场，计算 forward RR

    Args:
        compression_lookback: 压缩检测的回看窗口
        compression_percentile: 压缩判定的 ATR 百分位阈值
        breakout_lookback: 突破检测的回看窗口
        volume_mult: 成交量放大倍数
        combine_mode: "long_only", "short_only", "any_success"

    Returns:
        pd.Series: 连续 RR 标签，无突破处为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 1. 检测压缩
    compression_mask = detect_compression(
        work_df,
        atr_col=atr_col,
        lookback=compression_lookback,
        compression_percentile=compression_percentile,
    )

    # 2. 检测扩张/突破
    breakout_up, breakout_down = detect_expansion_breakout(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        atr_col=atr_col,
        compression_mask=compression_mask,
        breakout_lookback=breakout_lookback,
        volume_mult=volume_mult,
    )

    # 3. 根据 combine_mode 确定信号
    if combine_mode == "long_only":
        signal_mask = breakout_up
        signal_direction = 1.0
    elif combine_mode == "short_only":
        signal_mask = breakout_down
        signal_direction = -1.0
    else:
        # any_success: 两个方向
        signal_mask = breakout_up | breakout_down
        signal_direction = 1.0

    # 4. 计算 RR 标签
    work_df["__signal"] = signal_direction

    rr_series = compute_rr_label(
        work_df,
        signal_col="__signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=True,
        entry_price_col="open",
        entry_offset=1,
    )

    # 应用信号掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series
