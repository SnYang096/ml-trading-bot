"""Factor library — 每个函数返回 DataFrame（同 returns 形状），大值=预期更强（多）。

约定：
    - 输入 returns 为对数收益率 DataFrame（列=symbol）
    - 动量类：值越大 -> 越强 -> 做多
    - 反转类：原始收益越高 -> 近期超买 -> 做空，所以取负号保证"大=多"
    - Funding：正 funding 代表多方付钱给空方（多方拥挤）-> 做空该币，取负号
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(returns: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """过去 lookback 根 K 线累计收益（可选跳过最近 skip 根，经典 "12-1 月" 动量）。"""
    cum = returns.rolling(lookback).sum()
    if skip > 0:
        cum = cum.shift(skip)
    return cum


def short_term_reversal(returns: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """短期反转：近 24 小时收益 -> 取负号（涨太多的要空）。"""
    return -returns.rolling(lookback).sum()


def funding_factor(funding: pd.DataFrame, lookback: int = 24 * 3) -> pd.DataFrame:
    """funding 均值：正 funding 意味着多方拥挤（支付空方）-> 做空信号。取负号。"""
    return -funding.rolling(lookback).mean()


def low_vol_factor(returns: pd.DataFrame, lookback: int = 24 * 7) -> pd.DataFrame:
    """低波动因子：波动越低，风险调整后的预期收益越好。取负号让"低 vol=大值"。"""
    vol = returns.rolling(lookback).std()
    return -vol


def vol_normalize(
    factor: pd.DataFrame, returns: pd.DataFrame, lookback: int = 24 * 7
) -> pd.DataFrame:
    """按近期波动率标准化因子值（避免高 vol 币种主导）。"""
    vol = returns.rolling(lookback).std().replace(0, np.nan)
    return factor.div(vol)


def winsorize(
    df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """按每行分位数截尾，抑制极端值。"""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)
