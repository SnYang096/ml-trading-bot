"""
特征计算包装函数

为一些需要特殊参数处理的函数创建包装函数，使其能够通过配置文件直接调用
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.time_series.utils_liquidity_features import (
    extract_liquidity_features,
    build_wpt_denoised_vpvr,
)


def compute_sqs_hal_high(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "resistance",
    **kwargs
) -> pd.DataFrame:
    """
    计算 HAL high 的 SQS（Structure Quality Score）
    
    包装函数：
    1. 一次性计算 HAL（滚动窗口，每个时间点有一个 HAL 值）
    2. 对每个时间点，使用当前时间点的 HAL 价格作为 sr_price 计算 SQS
    
    注意：HAL 是滚动计算的，每个时间点都有一个值。SQS 使用当前时间点的 HAL 价格
    作为支撑阻力价格，评估这个价格的历史质量。
    
    Args:
        df: DataFrame with required columns: high, low, close, volume, atr
        window: SQS 计算窗口（用于评估 SR 质量的历史窗口）
        tolerance_factor: ATR 容忍带系数
        sr_type: SR 类型（'resistance' for HAL high）
        **kwargs: 其他参数（如 poc_window）
    
    Returns:
        DataFrame with 'sqs_hal_high' column added
    """
    result = df.copy()
    
    # 1. 一次性计算 HAL（如果还没有计算）
    # HAL 是滚动窗口计算的，每个时间点都有一个值
    if "hal_high" not in result.columns:
        poc_window = kwargs.get("poc_window", 160)
        price_col = kwargs.get("price_col", None)
        result = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
            result,
            required_features={"hal_high"},
            poc_window=poc_window,
            price_col=price_col,
        )
    
    if "hal_high" not in result.columns:
        # 如果 HAL 计算失败，返回全 0
        result["sqs_hal_high"] = 0.0
        return result
    
    # 2. 对每个时间点计算 SQS
    # 使用当前时间点的 HAL 价格作为 sr_price（支撑阻力价格）
    # 评估这个价格在历史窗口内的质量
    sqs_values = []
    hal_high_series = result["hal_high"]
    
    for i in range(len(result)):
        if i < window:
            sqs_values.append(0.0)
            continue
        
        # 获取当前时间点的 HAL high 价格（作为支撑阻力价格）
        sr_price = hal_high_series.iloc[i]
        if pd.isna(sr_price) or sr_price <= 0:
            sqs_values.append(0.0)
            continue
        
        # 获取历史数据窗口（不含未来信息）
        # 只取窗口内的数据，用于评估这个 SR 价格的历史质量
        start_idx = max(0, i - window + 1)
        hist_df = result.iloc[start_idx:i+1].copy()
        
        # 确保有足够的列
        required_cols = ["high", "low", "close", "atr", "volume"]
        if not all(col in hist_df.columns for col in required_cols):
            sqs_values.append(0.0)
            continue
        
        # 计算 SQS：评估 sr_price 在历史窗口内的质量
        try:
            sqs = BaselineFeatureEngineer.calculate_sqs(
                sr_price=sr_price,  # 使用当前时间点的 HAL 价格作为 SR 价格
                df=hist_df,  # 历史数据窗口，用于评估质量
                window=min(window, len(hist_df)),
                tolerance_factor=tolerance_factor,
                sr_type=sr_type,
            )
            sqs_values.append(float(sqs) if not np.isnan(sqs) else 0.0)
        except Exception:
            sqs_values.append(0.0)
    
    result["sqs_hal_high"] = pd.Series(sqs_values, index=result.index)
    return result


def compute_sqs_hal_low(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "support",
    **kwargs
) -> pd.DataFrame:
    """
    计算 HAL low 的 SQS（Structure Quality Score）
    
    包装函数：
    1. 一次性计算 HAL（滚动窗口，每个时间点有一个 HAL 值）
    2. 对每个时间点，使用当前时间点的 HAL 价格作为 sr_price 计算 SQS
    
    注意：HAL 是滚动计算的，每个时间点都有一个值。SQS 使用当前时间点的 HAL 价格
    作为支撑阻力价格，评估这个价格的历史质量。
    
    Args:
        df: DataFrame with required columns: high, low, close, volume, atr
        window: SQS 计算窗口（用于评估 SR 质量的历史窗口）
        tolerance_factor: ATR 容忍带系数
        sr_type: SR 类型（'support' for HAL low）
        **kwargs: 其他参数（如 poc_window）
    
    Returns:
        DataFrame with 'sqs_hal_low' column added
    """
    result = df.copy()
    
    # 1. 一次性计算 HAL（如果还没有计算）
    # HAL 是滚动窗口计算的，每个时间点都有一个值
    if "hal_low" not in result.columns:
        poc_window = kwargs.get("poc_window", 160)
        price_col = kwargs.get("price_col", None)
        result = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
            result,
            required_features={"hal_low"},
            poc_window=poc_window,
            price_col=price_col,
        )
    
    if "hal_low" not in result.columns:
        # 如果 HAL 计算失败，返回全 0
        result["sqs_hal_low"] = 0.0
        return result
    
    # 2. 对每个时间点计算 SQS
    # 使用当前时间点的 HAL 价格作为 sr_price（支撑阻力价格）
    # 评估这个价格在历史窗口内的质量
    sqs_values = []
    hal_low_series = result["hal_low"]
    
    for i in range(len(result)):
        if i < window:
            sqs_values.append(0.0)
            continue
        
        # 获取当前时间点的 HAL low 价格（作为支撑阻力价格）
        sr_price = hal_low_series.iloc[i]
        if pd.isna(sr_price) or sr_price <= 0:
            sqs_values.append(0.0)
            continue
        
        # 获取历史数据窗口（不含未来信息）
        # 只取窗口内的数据，用于评估这个 SR 价格的历史质量
        start_idx = max(0, i - window + 1)
        hist_df = result.iloc[start_idx:i+1].copy()
        
        # 确保有足够的列
        required_cols = ["high", "low", "close", "atr", "volume"]
        if not all(col in hist_df.columns for col in required_cols):
            sqs_values.append(0.0)
            continue
        
        # 计算 SQS：评估 sr_price 在历史窗口内的质量
        try:
            sqs = BaselineFeatureEngineer.calculate_sqs(
                sr_price=sr_price,  # 使用当前时间点的 HAL 价格作为 SR 价格
                df=hist_df,  # 历史数据窗口，用于评估质量
                window=min(window, len(hist_df)),
                tolerance_factor=tolerance_factor,
                sr_type=sr_type,
            )
            sqs_values.append(float(sqs) if not np.isnan(sqs) else 0.0)
        except Exception:
            sqs_values.append(0.0)
    
    result["sqs_hal_low"] = pd.Series(sqs_values, index=result.index)
    return result


def compute_sr_strength_max(
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    **kwargs
) -> pd.DataFrame:
    """
    计算最大 SR 强度
    
    包装函数：自动获取边界定义，然后计算强度
    
    Args:
        df: DataFrame with required columns
        window: 计算窗口
        tolerance_factor: ATR 容忍带系数
        **kwargs: 其他参数
    
    Returns:
        DataFrame with 'sr_strength_max' column added
    """
    result = df.copy()
    
    # 1. 获取边界定义
    boundaries = BaselineFeatureEngineer._get_sr_boundary_definitions(result)
    
    if not boundaries:
        result["sr_strength_max"] = 0.0
        return result
    
    # 2. 计算边界强度
    compression_series = result.get("compression_confidence")
    boundary_strengths = BaselineFeatureEngineer._compute_boundary_strengths(
        data=result,
        boundaries=boundaries,
        window=window,
        tolerance_factor=tolerance_factor,
        compression_series=compression_series,
    )
    
    # 3. 找到最大强度
    if not boundary_strengths:
        result["sr_strength_max"] = 0.0
        return result
    
    # 合并所有强度序列，取每行的最大值
    strength_df = pd.DataFrame(boundary_strengths)
    result["sr_strength_max"] = strength_df.max(axis=1).fillna(0.0)
    
    return result


def compute_wpt_vpvr(
    df: pd.DataFrame,
    wavelet: str = "db4",
    level: int = 4,
    vpvr_window: int = 100,
    bins: int = 50,
    feature_type: str = "vpvr",
    **kwargs
) -> pd.DataFrame:
    """
    计算 WPT 降噪的 VPVR 特征
    
    包装函数：传递 vpvr_window 参数给 build_wpt_denoised_vpvr
    
    Args:
        df: DataFrame with OHLCV data
        wavelet: Wavelet function
        level: WPT decomposition level
        vpvr_window: VPVR 计算窗口
        bins: 价格分箱数
        feature_type: 特征类型（'vpvr'）
        **kwargs: 其他参数
    
    Returns:
        DataFrame with VPVR features added
    """
    result = df.copy()
    
    # 如果只需要 VPVR 特征，直接调用 build_wpt_denoised_vpvr
    if feature_type == "vpvr":
        result = build_wpt_denoised_vpvr(
            result,
            price_col=kwargs.get("price_col", "close"),
            volume_col=kwargs.get("volume_col", "volume"),
            high_col=kwargs.get("high_col", "high"),
            low_col=kwargs.get("low_col", "low"),
            wavelet=wavelet,
            level=level,
            vpvr_window=vpvr_window,
            bins=bins,
        )
    else:
        # 使用 extract_liquidity_features（不支持 vpvr_window）
        result = extract_liquidity_features(
            result,
            price_col=kwargs.get("price_col", "close"),
            volume_col=kwargs.get("volume_col", "volume"),
            high_col=kwargs.get("high_col", "high"),
            low_col=kwargs.get("low_col", "low"),
            atr_col=kwargs.get("atr_col", "atr"),
            wavelet=wavelet,
            level=level,
            feature_type=feature_type,
        )
    
    return result
