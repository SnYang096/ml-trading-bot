"""
组合特征包装函数：将独立的组合特征计算函数包装为 DataFrame 输入/输出格式

包含两类组合特征：
1. 交互特征（Interaction）：两个特征的乘积（状态 × 动量）
2. 衍生特征（Derived）：单个特征的变换或两个特征的其他运算（比值、差值、归一化等）

这些函数用于 feature_dependencies.yaml 中的特征定义
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any

from src.features.time_series.utils_interaction_features import (
    # 交互特征（两个特征的乘积）
    compute_liquidity_void_x_wpt_risk,
    compute_compression_energy_x_ofi_short,
    compute_hurst_x_trend_r2,
    compute_evt_x_trend_r2,
    compute_vpin_x_compression,
    compute_sma_slope_x_price_pos,
    compute_vpin_x_wick_upper,
    compute_vpin_x_wick_lower,
    # 衍生特征（单个特征的变换或两个特征的其他运算）
    compute_sr_strength_combined,
    compute_sr_distance_normalized,
    compute_dist_to_zz_high,
    compute_dist_to_zz_low,
    compute_dist_to_zz_high_atr,
    compute_dist_to_zz_low_atr,
    compute_cvd_slope,
    compute_atr_ratio,
    compute_bb_width_ratio,
    compute_compression_score,
    compute_tbr_ma,
    compute_tbr_spike,
)


def _wrap_derived_to_df(
    df: pd.DataFrame,
    compute_func,
    output_col: str,
    **kwargs
) -> pd.DataFrame:
    """
    将组合特征计算函数包装为 DataFrame 输入/输出格式
    
    Args:
        df: Input DataFrame
        compute_func: Derived feature computation function
        output_col: Output column name
        **kwargs: Additional arguments for compute_func
    
    Returns:
        DataFrame with derived feature added
    """
    result = df.copy()
    series = compute_func(df, **kwargs)
    result[output_col] = series
    return result


def compute_sr_strength_combined_wrapper(
    df: pd.DataFrame,
    sqs_col: str = "sqs",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 sr_strength_combined"""
    return _wrap_derived_to_df(
        df,
        compute_sr_strength_combined,
        "sr_strength_combined",
        sqs_col=sqs_col,
    )


def compute_sr_distance_normalized_wrapper(
    df: pd.DataFrame,
    dist_col: str = "dist_to_nearest_sr",
    atr_col: str = "atr",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 sr_distance_normalized"""
    return _wrap_derived_to_df(
        df,
        compute_sr_distance_normalized,
        "sr_distance_normalized",
        dist_col=dist_col,
        atr_col=atr_col,
    )


def compute_dist_to_zz_high_wrapper(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_high_col: str = "zz_high_value",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 dist_to_zz_high"""
    return _wrap_derived_to_df(
        df,
        compute_dist_to_zz_high,
        "dist_to_zz_high",
        price_col=price_col,
        zz_high_col=zz_high_col,
    )


def compute_dist_to_zz_low_wrapper(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_low_col: str = "zz_low_value",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 dist_to_zz_low"""
    return _wrap_derived_to_df(
        df,
        compute_dist_to_zz_low,
        "dist_to_zz_low",
        price_col=price_col,
        zz_low_col=zz_low_col,
    )


def compute_dist_to_zz_high_atr_wrapper(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_high",
    atr_col: str = "atr",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 dist_to_zz_high_atr"""
    return _wrap_derived_to_df(
        df,
        compute_dist_to_zz_high_atr,
        "dist_to_zz_high_atr",
        dist_col=dist_col,
        atr_col=atr_col,
    )


def compute_dist_to_zz_low_atr_wrapper(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_low",
    atr_col: str = "atr",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 dist_to_zz_low_atr"""
    return _wrap_derived_to_df(
        df,
        compute_dist_to_zz_low_atr,
        "dist_to_zz_low_atr",
        dist_col=dist_col,
        atr_col=atr_col,
    )


def compute_cvd_slope_wrapper(
    df: pd.DataFrame,
    cvd_col: str = "cvd",
    window: int = 5,
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 cvd_slope"""
    return _wrap_derived_to_df(
        df,
        compute_cvd_slope,
        f"cvd_slope_{window}",
        cvd_col=cvd_col,
        window=window,
    )


def compute_atr_ratio_wrapper(
    df: pd.DataFrame,
    atr_col: str = "atr",
    price_col: str = "close",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 atr_ratio"""
    return _wrap_derived_to_df(
        df,
        compute_atr_ratio,
        "atr_ratio",
        atr_col=atr_col,
        price_col=price_col,
    )


def compute_bb_width_ratio_wrapper(
    df: pd.DataFrame,
    bb_upper_col: str = "bb_upper",
    bb_lower_col: str = "bb_lower",
    bb_middle_col: str = "bb_middle",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 bb_width_ratio"""
    return _wrap_derived_to_df(
        df,
        compute_bb_width_ratio,
        "bb_width_ratio",
        bb_upper_col=bb_upper_col,
        bb_lower_col=bb_lower_col,
        bb_middle_col=bb_middle_col,
    )


def compute_compression_score_wrapper(
    df: pd.DataFrame,
    bb_width_ratio_col: str = "bb_width_ratio",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 compression_score"""
    return _wrap_derived_to_df(
        df,
        compute_compression_score,
        "compression_score",
        bb_width_ratio_col=bb_width_ratio_col,
    )


def compute_tbr_ma_wrapper(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    window: int = 5,
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 tbr_ma"""
    return _wrap_derived_to_df(
        df,
        compute_tbr_ma,
        f"tbr_ma_{window}",
        tbr_col=tbr_col,
        window=window,
    )


def compute_tbr_spike_wrapper(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    tbr_ma_col: str = "tbr_ma_5",
    spike_threshold: float = 1.5,
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 tbr_spike"""
    return _wrap_derived_to_df(
        df,
        compute_tbr_spike,
        "tbr_spike",
        tbr_col=tbr_col,
        tbr_ma_col=tbr_ma_col,
        spike_threshold=spike_threshold,
    )

