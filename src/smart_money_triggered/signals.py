from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


def calc_takebuy_ratio(ticks: pd.DataFrame) -> float:
    """
    takebuy = 主动买量 / 总量
    """
    if ticks.empty:
        return 0.0
    buy_vol = float(ticks.get("buy_volume", pd.Series(dtype=float)).sum())
    total_vol = float(ticks.get("volume", pd.Series(dtype=float)).sum())
    return buy_vol / total_vol if total_vol > 0 else 0.0


def calc_cvd_slope(ticks: pd.DataFrame, window: str = "30min") -> float:
    """
    计算累积成交量差（CVD）的斜率：近 window 内线性回归斜率。
    """
    if ticks.empty:
        return 0.0
    df = ticks.copy()
    df["delta"] = df.get("delta", 0)
    df["cvd"] = df["delta"].cumsum()
    df = df.set_index(pd.to_datetime(df["ts"]))
    recent = df.last(window)
    if recent.shape[0] < 2:
        return 0.0
    x = np.arange(len(recent))
    y = recent["cvd"].to_numpy()
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def calc_cluster_strength(ticks: pd.DataFrame, threshold: float = 0.6) -> float:
    """
    简单成交聚集度指标：buy_ratio 或 sell_ratio 高于阈值的窗口占比。
    """
    if ticks.empty:
        return 0.0
    cond = (ticks["buy_ratio"] >= threshold) | (ticks["sell_ratio"] >= threshold)
    return float(cond.mean())


def calc_vpin(ticks: pd.DataFrame, bucket_size: int = 1000) -> float:
    """
    VPIN 估算：按成交量分桶，计算 |buy - sell| / 总量 的平均。
    """
    if ticks.empty:
        return 0.0

    df = ticks.copy()
    df["cum_vol"] = df["volume"].cumsum()
    df["bucket"] = (df["cum_vol"] // bucket_size).astype(int)

    grouped = df.groupby("bucket").agg(
        buy=("buy_volume", "sum"), sell=("sell_volume", "sum"), vol=("volume", "sum")
    )
    grouped["imbalance"] = (grouped["buy"] - grouped["sell"]).abs()
    grouped["vpin"] = grouped["imbalance"] / grouped["vol"].replace(0, np.nan)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["vpin"])
    if grouped.empty:
        return 0.0
    return float(grouped["vpin"].mean())


def calc_vwap(ticks: pd.DataFrame) -> float:
    """
    VWAP 基于聚合窗口：sum(price*vol)/sum(vol)
    """
    if ticks.empty:
        return 0.0
    num = (ticks["close_price"] * ticks["volume"]).sum()
    denom = ticks["volume"].sum()
    return float(num / denom) if denom > 0 else 0.0


@dataclass
class SignalResult:
    takebuy: float
    cvd_slope: float
    cluster_score: float
    vpin: float
    vwap: float
    current_price: float
    decision: Optional[str]
    debug: Dict[str, float]


def generate_signal(
    ticks: pd.DataFrame,
    threshold: float = 0.0,
    takebuy_min: float = 0.65,
    cluster_min: float = 0.7,
    vpin_max: float = 0.75,
    vwap_discount: float = 0.98,
) -> SignalResult:
    takebuy = calc_takebuy_ratio(ticks)
    cvd_slope = calc_cvd_slope(ticks)
    cluster_score = calc_cluster_strength(ticks)
    vpin = calc_vpin(ticks)
    vwap_val = calc_vwap(ticks)
    current_price = float(ticks["close_price"].iloc[-1]) if not ticks.empty else 0.0

    decision = None
    if (
        takebuy > takebuy_min
        and cvd_slope > threshold
        and cluster_score > cluster_min
        and current_price > vwap_val * vwap_discount
        and vpin < vpin_max
    ):
        decision = "LONG"
    elif (
        takebuy < (1 - takebuy_min)
        and cvd_slope < -threshold
        and cluster_score > cluster_min
        and current_price < vwap_val * (2 - vwap_discount)
        and vpin < vpin_max
    ):
        decision = "SHORT"

    debug = {
        "takebuy": takebuy,
        "cvd_slope": cvd_slope,
        "cluster_score": cluster_score,
        "vpin": vpin,
        "vwap": vwap_val,
        "current_price": current_price,
    }

    return SignalResult(
        takebuy=takebuy,
        cvd_slope=cvd_slope,
        cluster_score=cluster_score,
        vpin=vpin,
        vwap=vwap_val,
        current_price=current_price,
        decision=decision,
        debug=debug,
    )

