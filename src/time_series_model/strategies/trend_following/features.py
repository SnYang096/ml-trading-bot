"""
趋势跟踪策略专属特征工程

核心特征：
1. 趋势强度（Hurst 指数、ADX、WPT 低频子带斜率）
2. 资金验证（滚动 5D CVD 趋势、MFE/MAE 比、Volume 与 price 同向率）
3. 节奏感知（Hilbert 瞬时频率稳定性、Spectrum 主频周期长度）
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


def build_trend_following_features(
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
    构建趋势跟踪策略的专属特征集

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
        DataFrame with trend following features added
    """
    df = df.copy()

    # 1. WPT 特征（低频子带斜率）
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

    # WPT 低频子带斜率（趋势强度）
    if "wpt_price_trend" in df.columns:
        df["wpt_trend_slope"] = df["wpt_price_trend"].diff()
        # 线性拟合斜率（滚动窗口）
        if len(df) > 20:
            slopes = []
            for i in range(len(df)):
                if i < 20:
                    slopes.append(0.0)
                else:
                    window_trend = df["wpt_price_trend"].iloc[i - 20 : i]
                    if len(window_trend) > 1:
                        x = np.arange(len(window_trend))
                        coeffs = np.polyfit(x, window_trend.values, 1)
                        slopes.append(coeffs[0])
                    else:
                        slopes.append(0.0)
            df["wpt_trend_slope_20"] = slopes

    # 2. Hurst 特征（趋势持续性）
    print("   📊 Extracting Hurst features...")
    df = extract_hurst_features(
        df,
        price_col=price_col,
        cvd_col=cvd_col,
        volume_col=volume_col,
        method="dfa",
        rolling_window=50,
    )

    # Hurst > 0.6 表示强趋势
    if "hurst_price_rolling" in df.columns:
        df["trend_strength_hurst"] = (df["hurst_price_rolling"] > 0.6).astype(float)
        df["trend_strength_hurst_value"] = df["hurst_price_rolling"]

    # 3. Hilbert 特征（瞬时频率稳定性）
    print("   📊 Extracting Hilbert features...")
    if "wpt_price_fluctuation" in df.columns:
        df = extract_hilbert_features(
            df,
            price_fluctuation_col="wpt_price_fluctuation",
            cvd_fluctuation_col="wpt_cvd_fluctuation" if cvd_col else None,
        )

        # 瞬时频率稳定性（高频震荡 vs 趋势运行）
        if "hilbert_price_freq" in df.columns:
            df["hilbert_freq_stability"] = 1.0 / (
                1.0 + df["hilbert_price_freq"].rolling(window=10, min_periods=1).std()
            )

    # 4. Spectrum 特征（主频周期长度）
    print("   📊 Extracting Spectrum features...")
    df = extract_spectrum_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        cvd_col=cvd_col,
        rolling_window=64,
    )

    # 5. VPVR 特征（空间域：趋势中的流动性聚集区）
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

    # 6. ZigZag 特征（空间域：趋势结构锚点）
    if "zz_high_value" in df.columns and "zz_low_value" in df.columns:
        # 趋势方向上的 ZigZag 结构
        if price_col in df.columns:
            # 当前价格相对于 ZigZag 结构的位置
            df["price_above_zz_high"] = (df[price_col] > df["zz_high_value"]).astype(
                float
            )
            df["price_below_zz_low"] = (df[price_col] < df["zz_low_value"]).astype(
                float
            )

    # 7. 资金验证特征
    # 滚动 5D CVD 趋势（斜率）
    if cvd_col and cvd_col in df.columns:
        if len(df) > 5:
            cvd_slopes = []
            for i in range(len(df)):
                if i < 5:
                    cvd_slopes.append(0.0)
                else:
                    window_cvd = df[cvd_col].iloc[i - 5 : i]
                    if len(window_cvd) > 1:
                        x = np.arange(len(window_cvd))
                        coeffs = np.polyfit(x, window_cvd.values, 1)
                        cvd_slopes.append(coeffs[0])
                    else:
                        cvd_slopes.append(0.0)
            df["cvd_trend_slope_5"] = cvd_slopes

    # Volume 与 price 同向率
    if volume_col in df.columns and price_col in df.columns:
        price_change = df[price_col].pct_change()
        volume_change = df[volume_col].pct_change()
        df["volume_price_alignment"] = (
            (price_change > 0) & (volume_change > 0)
            | (price_change < 0) & (volume_change < 0)
        ).astype(float)
        df["volume_price_alignment_ratio"] = (
            df["volume_price_alignment"].rolling(window=20, min_periods=1).mean()
        )

    # 8. 趋势强度特征（ADX 等，如果有）
    if "adx" in df.columns:
        df["trend_strength_adx"] = (df["adx"] > 25).astype(float)

    # ROC 特征
    if price_col in df.columns:
        roc_20 = df[price_col].pct_change(20)
        df["roc_20"] = roc_20
        df["trend_direction"] = (roc_20 > 0).astype(float) - (roc_20 < 0).astype(float)

    # 9. 收益率百分位（Rank Label 的天然特征）
    if price_col in df.columns:
        returns = df[price_col].pct_change()
        if len(df) > 200:
            df["return_percentile"] = returns.rolling(window=200, min_periods=50).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
            )

    # 10. 确保所有特征都有 shift(1) 以避免未来数据
    wpt_cols = [col for col in df.columns if col.startswith("wpt_")]
    hilbert_cols = [col for col in df.columns if col.startswith("hilbert_")]
    hurst_cols = [col for col in df.columns if col.startswith("hurst_")]
    spectrum_cols = [col for col in df.columns if col.startswith("spectrum_")]
    vpvr_cols = [col for col in df.columns if col.startswith("vpvr_")]

    for col in wpt_cols + hilbert_cols + hurst_cols + spectrum_cols + vpvr_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

    return df


def select_trend_following_features(
    df: pd.DataFrame,
    all_features: List[str],
) -> List[str]:
    """
    为趋势跟踪策略选择特征

    核心特征：
    - 趋势强度（Hurst、ADX、WPT 低频斜率）
    - 资金验证（CVD 趋势、MFE/MAE 比、Volume 与 price 同向率）
    - 节奏感知（Hilbert 瞬时频率稳定性、Spectrum 主频周期长度）
    """
    trend_keywords = [
        # 趋势强度
        "trend",
        "trend_strength",
        "trend_direction",
        "hurst_price",
        "hurst_cvd",
        "trend_strength_hurst",
        "adx",
        "trend_strength_adx",
        # WPT 特征
        "wpt_price",
        "wpt_trend",
        "wpt_trend_slope",
        # 资金验证
        "cvd",
        "cvd_trend",
        "cvd_slope",
        "volume_price_alignment",
        "volume_price_alignment_ratio",
        # 节奏感知
        "hilbert_freq",
        "hilbert_freq_stability",
        "spectrum",
        "spectrum_period",
        "spectrum_dominant",
        # ROC 特征
        "roc",
        "roc_20",
        # 收益率百分位
        "return_percentile",
        # VPVR 特征（空间域）
        "vpvr_pvp",
        "vpvr_hvn",
        "vpvr_volume_density",
        # ZigZag 特征（空间域）
        "zz_high",
        "zz_low",
        "price_above_zz",
        "price_below_zz",
        # Default 特征（趋势指标）
        "adx",
        "parabolic",
        "ichimoku",
        "sar",
        "ma",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in trend_keywords):
            selected.append(feat)
        # 排除局部 SR 特征（趋势中 SR 经常被无视）
        elif "sr_" in feat_lower or "sqs" in feat_lower:
            continue
        else:
            # 保留通用特征
            if any(keyword in feat_lower for keyword in ["atr", "volatility", "vol"]):
                selected.append(feat)

    return selected
