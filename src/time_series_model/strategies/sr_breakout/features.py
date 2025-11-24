"""
SR 突破策略专属特征工程

核心特征：
1. 突破质量（突破时成交量/20日均量、VPER、突破后3根K线收盘站稳比例）
2. 动能持续性（WPT 中低频能量比、Hurst 指数、ROC 比值）
3. 真空区识别（突破方向上的 VPVR 低量节点距离、Spectrum 主频迁移）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.features.time_series.utils_hilbert_features import extract_hilbert_features
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_spectrum_features import extract_spectrum_features
from src.features.time_series.utils_liquidity_features import extract_liquidity_features


def build_sr_breakout_features(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    cvd_col: Optional[str] = None,
    tbr_col: Optional[str] = None,
    atr_col: str = "atr",
) -> pd.DataFrame:
    """
    构建 SR 突破策略的专属特征集

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column
        high_col: High column
        low_col: Low column
        volume_col: Volume column
        cvd_col: CVD column (optional)
        tbr_col: Take Buy Ratio column (optional)
        atr_col: ATR column

    Returns:
        DataFrame with SR breakout features added
    """
    df = df.copy()

    # 1. WPT 特征（多尺度能量比）
    print("   📊 Extracting WPT features...")
    df = extract_wpt_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        cvd_col=cvd_col,
        tbr_col=tbr_col,
        wavelet="db4",
        level=4,
    )

    # WPT 中低频能量比（动能持续性）
    if "wpt_price_energy_mid_low_ratio" in df.columns:
        df["wpt_momentum_persistence"] = df["wpt_price_energy_mid_low_ratio"]

    # 2. Hilbert 特征（CVD 相位领先）
    if cvd_col and "wpt_cvd_fluctuation" in df.columns:
        print("   📊 Extracting Hilbert features...")
        df = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation",
        )

    # 3. Hurst 特征（趋势持续性）
    print("   📊 Extracting Hurst features...")
    df = extract_hurst_features(
        df,
        price_col=price_col,
        cvd_col=cvd_col,
        volume_col=volume_col,
        method="dfa",
        rolling_window=50,
    )

    # 4. Spectrum 特征（主频迁移）
    print("   📊 Extracting Spectrum features...")
    df = extract_spectrum_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        cvd_col=cvd_col,
        rolling_window=64,
    )

    # 5. VPVR 特征（空间域：流动性真空区识别）
    print("   📊 Extracting VPVR features...")
    df = extract_liquidity_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        high_col=high_col,
        low_col=low_col,
        atr_col=atr_col,
        feature_type="vpvr",  # 仅提取 VPVR 特征
    )

    # 6. WPT + Volume 能量协同分析（真假突破判断）
    print("   📊 Extracting WPT+Volume energy features...")
    df = extract_liquidity_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        feature_type="energy",  # 仅提取能量特征
    )

    # 7. ZigZag 结构特征（空间域：突破点识别）
    if "zz_high_value" in df.columns and "zz_low_value" in df.columns:
        # 到最近 ZigZag 高点的距离（用于识别突破点）
        if price_col in df.columns:
            df["dist_to_zz_high"] = (df[price_col] - df["zz_high_value"]).abs()
            df["dist_to_zz_low"] = (df[price_col] - df["zz_low_value"]).abs()
            # 归一化距离
            if atr_col in df.columns:
                df["dist_to_zz_high_atr"] = df["dist_to_zz_high"] / df[atr_col].replace(
                    0, np.nan
                )
                df["dist_to_zz_low_atr"] = df["dist_to_zz_low"] / df[atr_col].replace(
                    0, np.nan
                )

    # 8. 突破质量特征
    if volume_col in df.columns:
        # 突破时成交量 / 20日均量
        df["volume_ma_20"] = df[volume_col].rolling(window=20, min_periods=1).mean()
        df["volume_ratio_breakout"] = df[volume_col] / df["volume_ma_20"].replace(
            0, np.nan
        )

        # VPER（Volume-Price Energy Ratio）
        if "wpt_vper" in df.columns:
            df["vper_spike"] = (
                df["wpt_vper"] > df["wpt_vper"].rolling(window=20).quantile(0.8)
            ).astype(float)

    # 9. 动能持续性特征
    # ROC(5) / ROC(20) 比值
    if price_col in df.columns:
        roc_5 = df[price_col].pct_change(5)
        roc_20 = df[price_col].pct_change(20)
        df["roc_ratio"] = roc_5 / roc_20.replace(0, np.nan)

    # Hurst > 0.6 表示强趋势
    if "hurst_price_rolling" in df.columns:
        df["trend_strength_hurst"] = (df["hurst_price_rolling"] > 0.6).astype(float)

    # 10. 真空区识别特征（使用 VPVR LVN）
    # Spectrum 主频迁移（从高频→低频）
    if "spectrum_price_period" in df.columns:
        df["spectrum_period_change"] = df["spectrum_price_period"].pct_change()
        df["spectrum_migrating_to_low"] = (df["spectrum_period_change"] > 0.1).astype(
            float
        )

    # 11. 突破确认特征
    # 突破后 3 根 K 线收盘站稳比例（需要未来数据，这里用历史数据模拟）
    if price_col in df.columns and high_col in df.columns and low_col in df.columns:
        # 使用历史数据计算"站稳"（收盘价在区间内）
        df["price_in_range"] = (
            (df[price_col] >= df[low_col]) & (df[price_col] <= df[high_col])
        ).astype(float)
        df["hold_ratio_3"] = (
            df["price_in_range"].rolling(window=3, min_periods=1).mean()
        )

    # 12. 确保所有特征都有 shift(1) 以避免未来数据
    wpt_cols = [col for col in df.columns if col.startswith("wpt_")]
    hilbert_cols = [col for col in df.columns if col.startswith("hilbert_")]
    hurst_cols = [col for col in df.columns if col.startswith("hurst_")]
    spectrum_cols = [col for col in df.columns if col.startswith("spectrum_")]
    vpvr_cols = [col for col in df.columns if col.startswith("vpvr_")]
    liquidity_cols = [
        col
        for col in df.columns
        if col.startswith("liquidity_")
        or col.startswith("wpt_vper")
        or col.startswith("wpt_energy")
        or col.startswith("wpt_multi_scale")
        or col.startswith("wpt_breakout")
    ]

    for col in (
        wpt_cols
        + hilbert_cols
        + hurst_cols
        + spectrum_cols
        + vpvr_cols
        + liquidity_cols
    ):
        if col in df.columns:
            df[col] = df[col].shift(1)

    return df


def select_sr_breakout_features(
    df: pd.DataFrame,
    all_features: List[str],
) -> List[str]:
    """
    为 SR 突破策略选择特征

    核心特征：
    - 突破动量（速度、强度、持续性）
    - 成交量放大
    - 趋势延续信号
    - 流动性池位置
    - 波动率扩张
    """
    breakout_keywords = [
        # 突破动量
        "breakout_speed",
        "momentum",
        "momentum_decay",
        "follow_through",
        "breakout_momentum",
        "breakout_strength",
        "wpt_momentum",
        # WPT 特征
        "wpt_price",
        "wpt_volume",
        "wpt_cvd",
        "wpt_vper",
        # 成交量
        "volume_ratio",
        "vol_ratio",
        "volume_spike",
        "volume_confirmation",
        "volume_ratio_breakout",
        "vper",
        "order_flow",
        "cvd",
        "taker_buy_ratio",
        # 趋势
        "trend",
        "trend_strength",
        "trend_context",
        "trend_4h",
        "momentum_persistence",
        "trend_strength_hurst",
        # Hurst 特征
        "hurst_price",
        "hurst_cvd",
        # Spectrum 特征
        "spectrum",
        "spectrum_period",
        "spectrum_migrating",
        # 流动性
        "liquidity",
        "liquidity_pool",
        "order_block",
        # VPVR 特征（空间域）
        "vpvr_pvp",
        "vpvr_hvn",
        "vpvr_lvn",
        "vpvr_lvn_distance",
        "vpvr_volume_density",
        "vpvr_price_in_lvn",
        # ZigZag 特征（空间域）
        "zz_high",
        "zz_low",
        "dist_to_zz",
        # WPT + Volume 能量协同
        "wpt_vper",
        "wpt_energy_cascade",
        "wpt_multi_scale_consistency",
        "wpt_breakout_confidence",
        "wpt_false_breakout_risk",
        # 波动率
        "volatility",
        "volatility_regime",
        "atr",
        "volatility_expansion",
        # 价格行为
        "price_action",
        "breakout_status",
        "breakout_confirmation",
        "hold_ratio",
        # ROC 特征
        "roc",
        "roc_ratio",
        # Default 特征（趋势指标）
        "adx",
        "parabolic",
        "ichimoku",
        "sar",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in breakout_keywords):
            selected.append(feat)
        # 排除反转相关特征
        elif "reversal" in feat_lower or "reversed" in feat_lower:
            continue
        else:
            # 保留通用特征
            if any(keyword in feat_lower for keyword in ["atr", "volatility", "vol"]):
                selected.append(feat)

    return selected
