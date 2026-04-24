"""exp05 regime 标签工具：BTC trend + cross-sectional funding mean。

两个维度：
    - trend (by BTC 30d return): bull (>+10%) / bear (<-10%) / range
    - funding (by cross-sectional mean of funding, 7d rolling):
        long_crowd (>0.02%/8h) / short_crowd (<-0.01%/8h) / normal

实验表明稀疏组合样本太少，因此提供聚合函数 collapse_combined_regime()，
把 9 种组合压缩成样本充足的 5 种。
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


TREND_BULL = 0.10
TREND_BEAR = -0.10
FUNDING_LONG_CROWD = 0.0002  # 0.02% per 8h
FUNDING_SHORT_CROWD = -0.0001  # -0.01% per 8h


def label_trend(btc_ret_30d: float) -> str:
    if pd.isna(btc_ret_30d):
        return "range"
    if btc_ret_30d > TREND_BULL:
        return "bull"
    if btc_ret_30d < TREND_BEAR:
        return "bear"
    return "range"


def label_funding(funding_mean_7d: float) -> str:
    if pd.isna(funding_mean_7d):
        return "normal"
    if funding_mean_7d > FUNDING_LONG_CROWD:
        return "long_crowd"
    if funding_mean_7d < FUNDING_SHORT_CROWD:
        return "short_crowd"
    return "normal"


def compute_regime_labels(
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    trend_lookback_bars: int = 24 * 30,
    funding_lookback_bars: int = 24 * 7,
) -> pd.DataFrame:
    """返回 DataFrame(index=时间, columns=['trend', 'funding', 'combined', 'collapsed'])."""
    if "BTCUSDT" not in prices.columns:
        raise ValueError("prices 需要包含 BTCUSDT 列")
    btc = prices["BTCUSDT"]
    btc_ret = btc.pct_change(trend_lookback_bars)
    trend = btc_ret.apply(label_trend).rename("trend")

    fmean = funding.mean(axis=1)
    fmean_7d = fmean.rolling(funding_lookback_bars).mean()
    fund = fmean_7d.apply(label_funding).rename("funding")

    combined = (trend + "_" + fund).rename("combined")
    collapsed = combined.map(collapse_combined_regime).rename("collapsed")
    return pd.concat([trend, fund, combined, collapsed], axis=1)


COLLAPSE_MAP: Dict[str, str] = {
    # 样本充足的 regime 保留
    "bull_long_crowd": "bull_momentum",  # 多头 + 拥挤 = 强趋势延续
    "bull_normal": "bull_normal",
    "range_normal": "range_normal",
    "range_short_crowd": "range_reversal",  # 震荡 + 空头拥挤 = 反弹机会
    "range_long_crowd": "range_reversal",  # 稀疏，并入反转
    "bear_normal": "bear",
    "bear_long_crowd": "bear",
    "bear_short_crowd": "bear",
    "bull_short_crowd": "bull_normal",  # 罕见，按 bull normal 处理
}


def collapse_combined_regime(combined: str) -> str:
    return COLLAPSE_MAP.get(combined, "range_normal")


COLLAPSED_REGIMES = [
    "bull_momentum",
    "bull_normal",
    "range_normal",
    "range_reversal",
    "bear",
]
