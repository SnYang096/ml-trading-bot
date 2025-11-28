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

# 注意：基础特征（WPT, Hilbert, Hurst, VPVR, VPIN, Interaction）现在通过配置文件加载
# 这些导入已不再需要，但保留用于向后兼容（如果直接调用函数）
# from src.features.time_series.utils_wpt_features import extract_wpt_features
# from src.features.time_series.utils_hilbert_features import extract_hilbert_features
# from src.features.time_series.utils_hurst_features import extract_hurst_features
# from src.features.time_series.utils_liquidity_features import extract_liquidity_features
# from src.features.time_series.utils_order_flow_features import extract_order_flow_features
# from src.features.time_series.utils_interaction_features import extract_interaction_features


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
    构建 SR 反转策略的专属特征集（组合和衍生特征）

    注意：基础特征（WPT, Hilbert, Hurst, VPVR, VPIN, Interaction）应该通过配置文件加载
    此函数只做：
    1. 特征组合（如 SR 强度组合）
    2. 衍生特征（如 SR 距离归一化、ZigZag 距离）
    3. 简单的滚动统计（如 CVD 斜率、TBR 移动平均）
    4. shift(1) 避免未来数据

    Args:
        df: DataFrame with OHLCV data (应该已经包含基础特征)
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

    # 注意：通用组合特征（sr_strength_combined, sr_distance_normalized, dist_to_zz_*,
    # cvd_slope_5, atr_ratio, bb_width_ratio, compression_score, tbr_ma_5, tbr_spike）
    # 现在通过配置文件加载，不需要在这里计算

    # 这里只做策略特定的组合特征（如果有的话）
    # 目前 SR Reversal 策略没有策略特定的组合特征，所有组合特征都是通用的

    # 确保所有特征都有 shift(1) 以避免未来数据
    # 注意：所有特征（基础特征、交互特征、衍生特征）都应该已经在配置文件中计算
    # 这里统一 shift 所有特征列
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(
            (
                "wpt_",
                "hilbert_",
                "hurst_",
                "vpvr_",
                "vpin",
                "_x_",
                "_rank",
            )
        )
        or col
        in [
            # 通用组合特征（现在通过配置文件加载）
            "sr_strength_combined",
            "sr_distance_normalized",
            "dist_to_zz_high",
            "dist_to_zz_low",
            "dist_to_zz_high_atr",
            "dist_to_zz_low_atr",
            "cvd_slope_5",
            "atr_ratio",
            "bb_width_ratio",
            "compression_score",
            "tbr_ma_5",
            "tbr_spike",
        ]
    ]

    for col in feature_cols:
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
        # VPVR 特征（空间域）
        "vpvr_pvp",
        "vpvr_hvn",
        "vpvr_lvn",
        "vpvr_volume_density",
        # ZigZag 特征（空间域）
        "zz_high",
        "zz_low",
        "dist_to_zz",
        # Hilbert 特征（相位领先 + 背离信号）
        "hilbert_phase",
        "hilbert_cvd_leads",
        "hilbert_envelope",
        "hilbert_price_env",
        "hilbert_cvd_env",
        "hilbert_cvd_price_env_ratio",
        "hilbert_volume_env",
        "hilbert_env_price_vol_ratio",
        "hilbert_triple_divergence",
        "hilbert_price_env_qnorm",
        "hilbert_cvd_env_qnorm",
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
        # VPIN 特征（订单流不平衡）
        "vpin",
        # K 线形态特征（来自 talib）
        "cdl_",  # talib candlestick patterns
        # 特征交互项
        "_x_",  # interaction features
        "_rank",  # rank-transformed features
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
