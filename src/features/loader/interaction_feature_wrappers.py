"""
交互特征包装函数：将独立的交互特征计算函数包装为 DataFrame 输入/输出格式

这些函数用于 feature_dependencies.yaml 中的特征定义
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any

from src.features.time_series.utils_interaction_features import (
    compute_liquidity_void_x_wpt_risk,
    compute_compression_energy_x_ofi_short,
    compute_hurst_x_trend_r2,
    compute_evt_x_trend_r2,
    compute_vpin_x_compression,
    compute_sma_slope_x_price_pos,
    compute_vpin_x_wick_upper,
    compute_vpin_x_wick_lower,
    compute_vpin_x_trade_cluster_max_buy_run,
    compute_vpin_zscore_x_trade_cluster_max_buy_run,
    compute_vpin_signed_imbalance_x_trade_cluster_imbalance,
    compute_vpin_x_trade_cluster_entropy,
    apply_rank_transform_to_interaction,
)


def _wrap_interaction_to_df(
    df: pd.DataFrame,
    compute_func,
    output_col: str,
    **kwargs
) -> pd.DataFrame:
    """
    将交互特征计算函数包装为 DataFrame 输入/输出格式
    
    Args:
        df: Input DataFrame
        compute_func: Interaction computation function
        output_col: Output column name
        **kwargs: Additional arguments for compute_func
    
    Returns:
        DataFrame with interaction feature added
    """
    result = df.copy()
    series = compute_func(df, **kwargs)
    result[output_col] = series
    return result


def compute_liquidity_void_x_wpt_risk_wrapper(
    df: pd.DataFrame,
    liquidity_void_col: str = "liquidity_void_detected",
    wpt_risk_col: str = "wpt_false_breakout_risk",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 liquidity_void_x_wpt_risk"""
    return _wrap_interaction_to_df(
        df,
        compute_liquidity_void_x_wpt_risk,
        "liquidity_void_x_wpt_risk",
        liquidity_void_col=liquidity_void_col,
        wpt_risk_col=wpt_risk_col,
    )


def compute_compression_energy_x_ofi_short_wrapper(
    df: pd.DataFrame,
    compression_col: str = "compression_energy",
    ofi_col: str = "ofi_short",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 compression_energy_x_ofi_short"""
    return _wrap_interaction_to_df(
        df,
        compute_compression_energy_x_ofi_short,
        "compression_energy_x_ofi_short",
        compression_col=compression_col,
        ofi_col=ofi_col,
    )


def compute_hurst_x_trend_r2_wrapper(
    df: pd.DataFrame,
    hurst_col: str = "hurst_close_rolling",
    trend_r2_col: str = "trend_r2_20",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 hurst_x_trend_r2"""
    return _wrap_interaction_to_df(
        df,
        compute_hurst_x_trend_r2,
        "hurst_x_trend_r2",
        hurst_col=hurst_col,
        trend_r2_col=trend_r2_col,
    )


def compute_evt_x_trend_r2_wrapper(
    df: pd.DataFrame,
    evt_col: str = "evt_tail_shape",
    trend_r2_col: str = "trend_r2_20",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 evt_x_trend_r2"""
    return _wrap_interaction_to_df(
        df,
        compute_evt_x_trend_r2,
        "evt_x_trend_r2",
        evt_col=evt_col,
        trend_r2_col=trend_r2_col,
    )


def compute_vpin_x_compression_wrapper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    compression_col: str = "compression_energy",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_x_compression"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_x_compression,
        "vpin_x_compression",
        vpin_col=vpin_col,
        compression_col=compression_col,
    )


def compute_sma_slope_x_price_pos_wrapper(
    df: pd.DataFrame,
    sma_slope_col: str = "sma_200_slope",
    sma_col: str = "sma_200",
    close_col: str = "close",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 sma_slope_x_price_pos"""
    return _wrap_interaction_to_df(
        df,
        compute_sma_slope_x_price_pos,
        "sma_slope_x_price_pos",
        sma_slope_col=sma_slope_col,
        sma_col=sma_col,
        close_col=close_col,
    )


def compute_vpin_x_wick_upper_wrapper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_upper_ratio",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_x_wick_upper"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_x_wick_upper,
        "vpin_x_wick_upper",
        vpin_col=vpin_col,
        wick_col=wick_col,
    )


def compute_vpin_x_wick_lower_wrapper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_lower_ratio",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_x_wick_lower"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_x_wick_lower,
        "vpin_x_wick_lower",
        vpin_col=vpin_col,
        wick_col=wick_col,
    )


def compute_vpin_x_trade_cluster_max_buy_run_wrapper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    cluster_col: str = "trade_cluster_max_buy_run",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_x_trade_cluster_max_buy_run"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_x_trade_cluster_max_buy_run,
        "vpin_x_trade_cluster_max_buy_run",
        vpin_col=vpin_col,
        cluster_col=cluster_col,
    )


def compute_vpin_zscore_x_trade_cluster_max_buy_run_wrapper(
    df: pd.DataFrame,
    vpin_zscore_col: str = "vpin_zscore_20",
    cluster_col: str = "trade_cluster_max_buy_run",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_zscore_x_trade_cluster_max_buy_run"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_zscore_x_trade_cluster_max_buy_run,
        "vpin_zscore_x_trade_cluster_max_buy_run",
        vpin_zscore_col=vpin_zscore_col,
        cluster_col=cluster_col,
    )


def compute_vpin_signed_imbalance_x_trade_cluster_imbalance_wrapper(
    df: pd.DataFrame,
    vpin_signed_col: str = "vpin_signed_imbalance",
    cluster_imbalance_col: str = "trade_cluster_imbalance_ratio",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_signed_imbalance_x_trade_cluster_imbalance"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_signed_imbalance_x_trade_cluster_imbalance,
        "vpin_signed_imbalance_x_trade_cluster_imbalance",
        vpin_signed_col=vpin_signed_col,
        cluster_imbalance_col=cluster_imbalance_col,
    )


def compute_vpin_x_trade_cluster_entropy_wrapper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    entropy_col: str = "trade_cluster_directional_entropy",
    **kwargs
) -> pd.DataFrame:
    """包装函数：计算 vpin_x_trade_cluster_entropy"""
    return _wrap_interaction_to_df(
        df,
        compute_vpin_x_trade_cluster_entropy,
        "vpin_x_trade_cluster_entropy",
        vpin_col=vpin_col,
        entropy_col=entropy_col,
    )


def apply_rank_transform_to_interaction_wrapper(
    df: pd.DataFrame,
    interaction_col: str,
    groupby_col: str = None,
    **kwargs
) -> pd.DataFrame:
    """包装函数：对交互特征应用 rank transform"""
    result = df.copy()
    rank_series = apply_rank_transform_to_interaction(
        df,
        interaction_col=interaction_col,
        groupby_col=groupby_col,
    )
    result[f"{interaction_col}_rank"] = rank_series
    return result

