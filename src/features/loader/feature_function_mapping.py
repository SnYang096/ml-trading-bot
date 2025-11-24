"""
特征计算函数映射表

将配置文件中的 compute_func 字符串映射到实际的函数
"""

from typing import Dict, Callable, Optional

# Baseline 特征
from src.features.time_series.baseline_features import BaselineFeatureEngineer

# Enhanced 特征工具函数
from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.features.time_series.utils_hilbert_features import extract_hilbert_features
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_spectrum_features import extract_spectrum_features
from src.features.time_series.utils_liquidity_features import extract_liquidity_features

# 策略专属特征
from src.time_series_model.strategies.sr_reversal.features import (
    build_sr_reversal_features,
)
from src.time_series_model.strategies.sr_breakout.features import (
    build_sr_breakout_features,
)
from src.time_series_model.strategies.compression_breakout.features import (
    build_compression_breakout_features,
)
from src.time_series_model.strategies.trend_following.features import (
    build_trend_following_features,
)

FEATURE_FUNCTION_MAP: Dict[str, Callable] = {
    # ========================================================================
    # Baseline 特征（基础技术指标）
    # ========================================================================
    "BaselineFeatureEngineer._compute_atr": BaselineFeatureEngineer._compute_atr,
    "BaselineFeatureEngineer.compute_rsi": BaselineFeatureEngineer.compute_rsi,
    "BaselineFeatureEngineer.compute_macd": BaselineFeatureEngineer.compute_macd,
    "BaselineFeatureEngineer.compute_bollinger_bands": BaselineFeatureEngineer.compute_bollinger_bands,
    "BaselineFeatureEngineer.compute_atr": BaselineFeatureEngineer.compute_atr,
    
    # Baseline SR 特征（注意：这些是静态方法，需要特殊处理）
    "BaselineFeatureEngineer.calculate_sqs": BaselineFeatureEngineer.calculate_sqs,
    "BaselineFeatureEngineer._compute_boundary_strengths": BaselineFeatureEngineer._compute_boundary_strengths,
    "BaselineFeatureEngineer._compute_breakout_confirmation_and_role_flip": BaselineFeatureEngineer._compute_breakout_confirmation_and_role_flip,
    "BaselineFeatureEngineer._add_breakout_quality_features": BaselineFeatureEngineer._add_breakout_quality_features,
    "BaselineFeatureEngineer._compute_boundary_volume_confirmations": BaselineFeatureEngineer._compute_boundary_volume_confirmations,
    "BaselineFeatureEngineer._add_price_action_features": BaselineFeatureEngineer._add_price_action_features,
    
    # Baseline 基础指标添加函数
    "BaselineFeatureEngineer.add_basic_indicators": BaselineFeatureEngineer.add_basic_indicators,
    "BaselineFeatureEngineer.add_zigzag_dimensionless_features": BaselineFeatureEngineer.add_zigzag_dimensionless_features,
    "BaselineFeatureEngineer.add_poc_hal_dimensionless_features": BaselineFeatureEngineer.add_poc_hal_dimensionless_features,
    "BaselineFeatureEngineer.add_swing_dimensionless_features": BaselineFeatureEngineer.add_swing_dimensionless_features,
    "BaselineFeatureEngineer.add_ols_channel_features": BaselineFeatureEngineer.add_ols_channel_features,
    "BaselineFeatureEngineer.add_price_volume_relative_features": BaselineFeatureEngineer.add_price_volume_relative_features,
    "BaselineFeatureEngineer.add_common_derived_features": BaselineFeatureEngineer.add_common_derived_features,
    
    # ========================================================================
    # Enhanced 特征（WPT, Hilbert, Hurst, Spectrum, Liquidity）
    # ========================================================================
    "extract_wpt_features": extract_wpt_features,
    "extract_hilbert_features": extract_hilbert_features,
    "extract_hurst_features": extract_hurst_features,
    "extract_spectrum_features": extract_spectrum_features,
    "extract_liquidity_features": extract_liquidity_features,
    
    # ========================================================================
    # 策略专属特征构建函数
    # ========================================================================
    "build_sr_reversal_features": build_sr_reversal_features,
    "build_sr_breakout_features": build_sr_breakout_features,
    "build_compression_breakout_features": build_compression_breakout_features,
    "build_trend_following_features": build_trend_following_features,
}


def get_compute_func(func_name: str) -> Optional[Callable]:
    """
    根据函数名获取实际函数
    
    Args:
        func_name: 函数名（字符串）
    
    Returns:
        compute_func: 实际函数，如果不存在则返回 None
    
    Raises:
        ValueError: 如果函数名不存在
    """
    if func_name in FEATURE_FUNCTION_MAP:
        return FEATURE_FUNCTION_MAP[func_name]
    else:
        raise ValueError(
            f"Unknown compute function: {func_name}. "
            f"Available functions: {list(FEATURE_FUNCTION_MAP.keys())}"
        )

