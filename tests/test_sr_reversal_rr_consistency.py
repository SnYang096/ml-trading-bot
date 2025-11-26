import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label,
)


def _build_df_for_sr(
    direction: int,
    hit_tp_first: bool,
    atr_value: float = 10.0,
    max_holding_bars: int = 10,
) -> pd.DataFrame:
    """
    构造一段极简价格路径：
    - t=0 有 SR 反转信号（direction=+1 多头 / -1 空头）
    - t=1 以 open 价格入场
    - 之后价格按照预期先触达 TP(±2R) 或 SL(∓1R)
    """

    n = max_holding_bars + 5
    index = pd.RangeIndex(n)

    # 基准价格
    base_price = 100.0
    close = np.full(n, base_price, dtype=float)
    high = close.copy()
    low = close.copy()
    open_ = close.copy()

    # t=0: 信号出现，但不入场
    signal = np.zeros(n, dtype=float)
    signal[0] = float(direction)

    # t=1: 以 open[1] 入场，后续路径我们人工构造
    entry_price = base_price
    open_[1] = entry_price
    close[1] = entry_price
    high[1] = entry_price
    low[1] = entry_price

    # R = atr_value（和 compute_rr_label 中定义一致）
    R = atr_value

    if direction > 0:
        # 多头：TP = entry + 2R, SL = entry - 1R
        tp_level = entry_price + 2 * R
        sl_level = entry_price - 1 * R
        if hit_tp_first:
            # t=2 直接打到 TP（先 TP）
            low[2] = entry_price  # 不打到 SL
            high[2] = tp_level + 1e-6
        else:
            # t=2 先打到 SL
            low[2] = sl_level - 1e-6
            high[2] = entry_price
    else:
        # 空头：TP = entry - 2R, SL = entry + 1R
        tp_level = entry_price - 2 * R
        sl_level = entry_price + 1 * R
        if hit_tp_first:
            # t=2 直接打到 TP（先 TP）
            high[2] = entry_price  # 不打到 SL
            low[2] = tp_level - 1e-6
        else:
            # t=2 先打到 SL
            high[2] = sl_level + 1e-6
            low[2] = entry_price

    # 之后价格保持平稳，不再触发新事件
    for t in range(3, n):
        open_[t] = base_price
        high[t] = base_price
        low[t] = base_price
        close[t] = base_price

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, R, dtype=float),
            "signal": signal,
        },
        index=index,
    )
    return df


def test_sr_reversal_label_long_tp_first():
    """多头：先触达 +2R，应当被打上 label=1（成功）。"""
    df = _build_df_for_sr(direction=1, hit_tp_first=True)

    labels = compute_sr_reversal_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        auto_generate_signals=False,
    )

    # 标签是在信号出现的 t=0 打上的
    assert labels.iloc[0] == 1.0


def test_sr_reversal_label_long_sl_first():
    """多头：先触达 -1R，应当被打上 label=0（失败）。"""
    df = _build_df_for_sr(direction=1, hit_tp_first=False)

    labels = compute_sr_reversal_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        auto_generate_signals=False,
    )

    assert labels.iloc[0] == 0.0


def test_sr_reversal_label_short_tp_first():
    """空头：先触达 +2R（价格向下 2R），应当被打上 label=1。"""
    df = _build_df_for_sr(direction=-1, hit_tp_first=True)

    labels = compute_sr_reversal_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        auto_generate_signals=False,
    )

    assert labels.iloc[0] == 1.0


def test_sr_reversal_label_short_sl_first():
    """空头：先触达 -1R（价格向上 1R），应当被打上 label=0。"""
    df = _build_df_for_sr(direction=-1, hit_tp_first=False)

    labels = compute_sr_reversal_label(
        df,
        signal_col="signal",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        auto_generate_signals=False,
    )

    assert labels.iloc[0] == 0.0
