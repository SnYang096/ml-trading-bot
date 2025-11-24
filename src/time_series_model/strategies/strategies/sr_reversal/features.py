"""
SR 反转策略专属特征工程

核心特征：
1. SR 强度（WPT 低频 POC、SR 重叠数、VPVR 高量节点）
2. 流动性验证（CVD 在 SR 区的净买入斜率、Hilbert 相位差、Take Buy Ratio）
3. 波动状态（ATR/Close、Bollinger Band Width）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.features.time_series.utils_hilbert_features import extract_hilbert_features
from src.features.time_series.utils_hurst_features import extract_hurst_features


def build_sr_reversal_features(
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
    构建 SR 反转策略的专属特征集

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
        DataFrame with SR reversal features added
    """
    df = df.copy()

    # 1. WPT 特征（多尺度 SR 结构）
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

    # 4. SR 强度特征（如果已有 SR 相关特征）
    if "sqs" in df.columns or "dist_to_nearest_sr" in df.columns:
        # SR 重叠密度
        if "sqs" in df.columns:
            df["sr_strength_combined"] = df["sqs"].fillna(0.0)

        # SR 距离（标准化）
        if "dist_to_nearest_sr" in df.columns:
            df["sr_distance_normalized"] = df["dist_to_nearest_sr"].fillna(0.0)

    # 5. 流动性验证特征
    if cvd_col and cvd_col in df.columns:
        # CVD 在 SR 区的净买入斜率（滚动 5 根）
        if len(df) > 5:
            df["cvd_slope_5"] = (
                df[cvd_col]
                .rolling(window=5, min_periods=1)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0
                )
            )

    # 6. 波动状态特征
    if atr_col in df.columns and price_col in df.columns:
        df["atr_ratio"] = df[atr_col] / df[price_col].replace(0, np.nan)

    # Bollinger Band Width（压缩度）
    if (
        "bb_upper" in df.columns
        and "bb_lower" in df.columns
        and "bb_middle" in df.columns
    ):
        bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(
            0, np.nan
        )
        df["bb_width_ratio"] = bb_width
        df["compression_score"] = 1.0 / (1.0 + bb_width)

    # 7. Take Buy Ratio 特征（如果有）
    if tbr_col and tbr_col in df.columns:
        df["tbr_ma_5"] = df[tbr_col].rolling(window=5, min_periods=1).mean()
        df["tbr_spike"] = (df[tbr_col] > df["tbr_ma_5"] * 1.5).astype(float)

    # 8. 确保所有特征都有 shift(1) 以避免未来数据
    wpt_cols = [col for col in df.columns if col.startswith("wpt_")]
    hilbert_cols = [col for col in df.columns if col.startswith("hilbert_")]
    hurst_cols = [col for col in df.columns if col.startswith("hurst_")]

    for col in wpt_cols + hilbert_cols + hurst_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

    return df


def select_sr_reversal_features(
    df: pd.DataFrame,
    all_features: List[str],
) -> List[str]:
    """
    为 SR 反转策略选择特征

    核心特征：
    - SR 区域强度（SQS、距离、密度）
    - 反转信号（价格回测、试探、反向运行）
    - 成交量确认（萎缩、异常）
    - 多时间框架结构对齐
    - 压缩区域质量
    """
    reversal_keywords = [
        # SR 区域相关
        "sqs",
        "dist_to_nearest_sr",
        "sr_density",
        "sr_confluence",
        "nearest_sr",
        "direction_to_nearest_sr",
        "sr_strength",
        # WPT 特征
        "wpt_price",
        "wpt_volume",
        "wpt_cvd",
        "wpt_vper",
        # Hilbert 特征（相位领先）
        "hilbert_phase",
        "hilbert_cvd_leads",
        "hilbert_envelope",
        # Hurst 特征
        "hurst_price",
        "hurst_cvd",
        # 反转信号相关
        "reversal",
        "reversed",
        "fake_breakout",
        "breakout_status",
        "price_reversed_before_sr",
        # 成交量确认
        "volume_ratio",
        "vol_ratio",
        "volume_confirmation",
        "tbr",
        # 压缩和结构
        "compression",
        "compression_score",
        "compression_confidence",
        "bb_width",
        # 多时间框架
        "trend_context",
        "trend_4h",
        "volatility_regime",
        # 边界质量
        "breakout_quality",
        "breakout_confirmation",
        "role_flip",
        # 价格行为
        "price_action",
        "candlestick",
        # Default 特征（技术指标，可能对反转有用）
        "rsi",
        "macd",
        "stochastic",
        "williams",
    ]

    selected = []
    for feat in all_features:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in reversal_keywords):
            selected.append(feat)
        # 排除突破相关特征
        elif "breakout_speed" in feat_lower or "momentum_decay" in feat_lower:
            continue
        else:
            # 保留通用特征
            if any(
                keyword in feat_lower
                for keyword in ["atr", "volatility", "vol", "trend_strength"]
            ):
                selected.append(feat)

    return selected
