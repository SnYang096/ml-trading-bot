"""
压缩区突破策略专属特征工程

核心特征：
1. 压缩强度（Bollinger Band Width 分位数、ATR/MA(ATR) 比值、Spectrum 频谱平坦度）
2. 突破触发（最近 3 根 K 线 range 扩张率、Volume spike、CVD 净流入突变）
3. 方向确认（突破首根 K 线 body/range 比、Take Buy Ratio 方向一致性、WPT 高频子带能量方向）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.features.time_series.utils_hilbert_features import extract_hilbert_features
from src.features.time_series.utils_spectrum_features import extract_spectrum_features
from src.features.time_series.utils_liquidity_features import extract_liquidity_features


def build_compression_breakout_features(
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
    构建压缩区突破策略的专属特征集

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
        DataFrame with compression breakout features added
    """
    df = df.copy()

    # 1. WPT 特征（压缩度）
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

    # 2. Spectrum 特征（频谱平坦度）
    print("   📊 Extracting Spectrum features...")
    df = extract_spectrum_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        cvd_col=cvd_col,
        rolling_window=64,
    )

    # 3. VPVR 特征（空间域：压缩区内的 SR 密度）
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

    # 4. ZigZag 结构特征（空间域：压缩区间识别）
    if "zz_high_value" in df.columns and "zz_low_value" in df.columns:
        # 压缩区间高度（ZigZag 高点 - 低点）
        if price_col in df.columns:
            df["compression_range_height"] = (
                df["zz_high_value"] - df["zz_low_value"]
            ).abs()
            # 当前价格在压缩区间内的位置（0~1）
            df["price_in_compression_range"] = (
                (df[price_col] >= df["zz_low_value"])
                & (df[price_col] <= df["zz_high_value"])
            ).astype(float)

    # 5. 压缩强度特征
    # Bollinger Band Width 分位数
    if (
        "bb_upper" in df.columns
        and "bb_lower" in df.columns
        and "bb_middle" in df.columns
    ):
        bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(
            0, np.nan
        )
        if len(df) > 100:
            df["bb_width_percentile"] = bb_width.rolling(
                window=100, min_periods=1
            ).apply(
                lambda x: (
                    (x.iloc[-1] <= x.quantile(0.1)).astype(float) if len(x) > 0 else 0.0
                )
            )

    # ATR / MA(ATR) 比值
    if atr_col in df.columns:
        atr_ma = df[atr_col].rolling(window=20, min_periods=1).mean()
        df["atr_ratio"] = df[atr_col] / atr_ma.replace(0, np.nan)
        df["compression_atr"] = (df["atr_ratio"] < 0.8).astype(float)

    # Spectrum 频谱平坦度（越低越压缩）
    if "spectrum_price_flatness" in df.columns:
        df["compression_spectrum"] = (df["spectrum_price_flatness"] < 0.3).astype(float)

    # 6. 突破触发特征
    # 最近 3 根 K 线 range 扩张率
    if high_col in df.columns and low_col in df.columns:
        price_range = df[high_col] - df[low_col]
        range_ma_3 = price_range.rolling(window=3, min_periods=1).mean()
        range_ma_10 = price_range.rolling(window=10, min_periods=1).mean()
        df["range_expansion_ratio"] = range_ma_3 / range_ma_10.replace(0, np.nan)
        df["range_expanding"] = (df["range_expansion_ratio"] > 1.5).astype(float)

    # Volume spike（>2σ）
    if volume_col in df.columns:
        volume_ma = df[volume_col].rolling(window=20, min_periods=1).mean()
        volume_std = df[volume_col].rolling(window=20, min_periods=1).std()
        df["volume_spike"] = (
            (df[volume_col] > volume_ma + 2 * volume_std.replace(0, np.nan))
        ).astype(float)

    # CVD 净流入突变（Z-score > 2）
    if cvd_col and cvd_col in df.columns:
        cvd_diff = df[cvd_col].diff()
        cvd_ma = cvd_diff.rolling(window=20, min_periods=1).mean()
        cvd_std = cvd_diff.rolling(window=20, min_periods=1).std()
        cvd_zscore = (cvd_diff - cvd_ma) / cvd_std.replace(0, np.nan)
        df["cvd_spike"] = (cvd_zscore.abs() > 2.0).astype(float)

    # 7. 方向确认特征
    # 突破首根 K 线 body/range 比
    if price_col in df.columns and high_col in df.columns and low_col in df.columns:
        body = abs(df[price_col] - df[price_col].shift(1))
        price_range = df[high_col] - df[low_col]
        df["body_range_ratio"] = body / price_range.replace(0, np.nan)
        df["strong_body"] = (df["body_range_ratio"] > 0.6).astype(float)

    # Take Buy Ratio 方向一致性
    if tbr_col and tbr_col in df.columns:
        tbr_ma = df[tbr_col].rolling(window=5, min_periods=1).mean()
        df["tbr_direction"] = (df[tbr_col] > tbr_ma).astype(float) - (
            df[tbr_col] < tbr_ma
        ).astype(float)
        df["tbr_consistent"] = (df["tbr_direction"].abs() > 0.5).astype(float)

    # WPT 高频子带能量方向
    if "wpt_price_energy_high_ratio" in df.columns:
        energy_high_change = df["wpt_price_energy_high_ratio"].diff()
        df["wpt_high_energy_increasing"] = (energy_high_change > 0).astype(float)

    # 8. 压缩区内 SR 密度（使用 VPVR HVN 数量）
    if "vpvr_hvn_count" in df.columns:
        df["compression_sr_density"] = df["vpvr_hvn_count"].fillna(0.0)

    # 9. 确保所有特征都有 shift(1) 以避免未来数据
    wpt_cols = [col for col in df.columns if col.startswith("wpt_")]
    spectrum_cols = [col for col in df.columns if col.startswith("spectrum_")]
    compression_cols = [
        col
        for col in df.columns
        if "compression" in col.lower() or "range_expansion" in col.lower()
    ]
    vpvr_cols = [col for col in df.columns if col.startswith("vpvr_")]

    for col in wpt_cols + spectrum_cols + compression_cols + vpvr_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

    return df


def select_compression_breakout_features(
    df: pd.DataFrame,
    all_features: List[str],
) -> List[str]:
    """
    为压缩区突破策略选择特征

    核心特征：
    - 压缩指标（Bollinger Band 宽度、ATR 相对值）
    - 突破动量（速度、强度）
    - 成交量放大（突破时成交量）
    - 价格位置（在压缩区内的位置）
    - 突破确认（是否站稳）
    """
    compression_keywords = [
        # 压缩相关
        "compression",
        "compression_score",
        "compression_confidence",
        "compression_atr",
        "compression_spectrum",
        "compression_sr",
        "bb_width",
        "bb_position",
        "volatility",
        # 突破相关
        "breakout",
        "breakout_speed",
        "breakout_momentum",
        "breakout_strength",
        "follow_through",
        "momentum_persistence",
        # 成交量
        "volume",
        "vol_ratio",
        "volume_spike",
        "volume_confirmation",
        "volume_ratio_breakout",
        "order_flow",
        "cvd",
        "taker_buy_ratio",
        "cvd_spike",
        # 价格行为
        "price_action",
        "candlestick",
        "body_range",
        "strong_body",
        # 范围扩张
        "range_expansion",
        "range_expanding",
        # Spectrum 特征
        "spectrum",
        "spectrum_flatness",
        "spectrum_period",
        # WPT 特征
        "wpt_price",
        "wpt_volume",
        "wpt_energy",
        # VPVR 特征（空间域）
        "vpvr_pvp",
        "vpvr_hvn",
        "vpvr_lvn",
        "vpvr_volume_density",
        # ZigZag 特征（空间域）
        "zz_high",
        "zz_low",
        "compression_range",
        "price_in_compression",
        # Default 特征（压缩区检测）
        "bbands",
        "bollinger",
        "keltner",
        "donchian",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in compression_keywords):
            selected.append(feat)
        # 保留通用特征
        elif any(keyword in feat_lower for keyword in ["atr", "volatility", "vol"]):
            selected.append(feat)

    return selected
