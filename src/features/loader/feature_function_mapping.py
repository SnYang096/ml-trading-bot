"""
特征计算函数映射表

将配置文件中的 compute_func 字符串映射到实际的函数
"""

from typing import Dict, Callable, Optional

# Baseline 特征
from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.time_series.baseline_features import (
    compute_acceleration_3_from_series,
    compute_volume_anomaly_from_series,
    compute_trend_r2_20_from_series,
    compute_trend_r2_50_from_series,
    compute_slope_consistency_score_from_series,
    compute_volatility_reversal_score_from_series,
    compute_atr_percentile_from_series,
    compute_trend_volatility_alignment_from_series,
    compute_compression_to_breakout_prob_from_series,
)

# Enhanced 特征工具函数
from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.features.time_series.utils_hilbert_features import extract_hilbert_features
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_spectrum_features import extract_spectrum_features_from_series
from src.features.time_series.utils_liquidity_features import extract_liquidity_features
from src.features.time_series.utils_order_flow_features import extract_order_flow_features
from src.features.time_series.utils_order_flow_features import (
    select_order_flow_features,
    select_vpin_block_features,
    select_trade_cluster_block_features,
)
from src.features.time_series.utils_order_flow_features import (
    compute_vpin_derived_features_from_base,
    compute_vpin_ma_max_features_from_base,
    compute_vpin_change_features_from_base,
    compute_vpin_zscore_features_from_base,
    compute_vpin_ma_max_features_from_series,
    compute_vpin_change_features_from_series,
    compute_vpin_zscore_features_from_series,
    compute_vpin_quantile_rank_features_from_base,
    compute_vpin_volatility_features_from_base,
    compute_vpin_spike_features_from_base,
    compute_vpin_momentum_features_from_base,
    compute_vpin_signed_zscore_features_from_base,
    compute_vpin_quantile_rank_features_from_series,
    compute_vpin_volatility_features_from_series,
    compute_vpin_spike_features_from_series,
    compute_vpin_momentum_features_from_series,
    compute_vpin_signed_zscore_features_from_series,
    compute_vpin_base_aligned_features_from_series,
    compute_trade_cluster_derived_features_from_base,
    compute_trade_cluster_ratio_features_from_base,
    compute_trade_cluster_buy_sell_ratio_features_from_base,
    compute_trade_cluster_max_run_ratio_features_from_base,
    compute_trade_cluster_buy_sell_max_ratio_features_from_base,
    compute_trade_cluster_avg_run_ratio_features_from_base,
    compute_trade_cluster_buy_sell_avg_ratio_features_from_base,
    compute_trade_cluster_run_length_features_from_base,
    compute_trade_cluster_net_runs_counts_features_from_base,
    compute_trade_cluster_net_runs_ratio_features_from_counts,
    compute_trade_cluster_entropy_features_from_base,
    compute_trade_cluster_entropy_ma_change_features_from_base,
    compute_trade_cluster_entropy_zscore_features_from_base,
    compute_trade_cluster_max_buy_run_ma_features_from_base,
    compute_trade_cluster_imbalance_ratio_ma_features_from_base,
    compute_trade_cluster_net_runs_ma_features_from_base,
    compute_trade_cluster_total_runs_ma_features_from_base,
    compute_trade_cluster_imbalance_zscore_features_from_base,
    compute_trade_cluster_net_runs_zscore_features_from_base,
    compute_trade_cluster_max_buy_run_zscore_features_from_base,
    compute_trade_cluster_max_sell_run_zscore_features_from_base,
    compute_trade_cluster_ratio_features_from_series,
    compute_trade_cluster_buy_sell_ratio_features_from_series,
    compute_trade_cluster_max_run_ratio_features_from_series,
    compute_trade_cluster_buy_sell_max_ratio_features_from_series,
    compute_trade_cluster_avg_run_ratio_features_from_series,
    compute_trade_cluster_buy_sell_avg_ratio_features_from_series,
    compute_trade_cluster_run_length_features_from_series,
    compute_trade_cluster_net_runs_counts_features_from_series,
    compute_trade_cluster_net_runs_ratio_features_from_series,
    compute_trade_cluster_entropy_features_from_series,
    compute_trade_cluster_entropy_ma_change_features_from_series,
    compute_trade_cluster_entropy_zscore_features_from_series,
    compute_trade_cluster_max_buy_run_ma_features_from_series,
    compute_trade_cluster_imbalance_ratio_ma_features_from_series,
    compute_trade_cluster_net_runs_ma_features_from_series,
    compute_trade_cluster_total_runs_ma_features_from_series,
    compute_trade_cluster_imbalance_zscore_features_from_series,
    compute_trade_cluster_net_runs_zscore_features_from_series,
    compute_trade_cluster_max_buy_run_zscore_features_from_series,
    compute_trade_cluster_max_sell_run_zscore_features_from_series,
    compute_trade_cluster_base_aligned_features_from_series,
)
from src.features.time_series.utils_garch_features import (
    extract_garch_features_from_series,
)
from src.features.time_series.utils_dtw_features import (
    extract_dtw_features,
    extract_dtw_features_from_series,
)
from src.features.time_series.utils_evt_features import extract_evt_features_from_series
from src.features.time_series.utils_volatility_features import (
    extract_extended_volatility_features,
    extract_volume_profile_volatility_features_from_series,
    compute_vol_raw_features_from_series,
    compute_vol_atr_features_from_series,
    compute_vol_lag_features_from_series,
    compute_vol_trend_features_from_series,
    compute_vol_ma_features_from_series,
    compute_vol_regime_features_from_series,
    compute_vol_range_features_from_series,
    compute_vol_mom_features_from_series,
    select_extended_volatility_features,
)
# 独立DTW特征提取器（支持多窗口）
from src.features.time_series.utils_dtw_individual import (
    extract_dtw_hammer,
    extract_dtw_head_shoulder_bottom,
    extract_dtw_double_bottom,
    extract_dtw_bullish_engulfing,
    extract_dtw_shooting_star,
    extract_dtw_head_shoulder_top,
    extract_dtw_double_top,
    extract_dtw_bearish_engulfing,
    extract_dtw_bull_flag,
    extract_dtw_bear_flag,
    extract_dtw_triangle,
    extract_dtw_decline_consolidation,
)
# 交互特征包装函数（用于配置文件）
from src.features.loader.interaction_feature_wrappers import (
    compute_liquidity_void_x_wpt_risk_wrapper,
    compute_compression_energy_x_ofi_short_wrapper,
    compute_hurst_x_trend_r2_wrapper,
    compute_evt_x_trend_r2_wrapper,
    compute_vpin_x_compression_wrapper,
    compute_sma_slope_x_price_pos_wrapper,
    compute_vpin_x_wick_upper_wrapper,
    compute_vpin_x_wick_lower_wrapper,
    compute_vpin_x_trade_cluster_max_buy_run_wrapper,
    compute_vpin_zscore_x_trade_cluster_max_buy_run_wrapper,
    compute_vpin_signed_imbalance_x_trade_cluster_imbalance_wrapper,
    compute_vpin_x_trade_cluster_entropy_wrapper,
    apply_rank_transform_to_interaction_wrapper,
)

# 特征包装函数
from src.features.loader.feature_wrappers import (
    compute_unified_volume_profile,
    compute_footprint_features,
)

from src.features.time_series.utils_volume_profile import compute_wpt_vpvr_from_series

# Baseline narrow wrappers (kept in baseline_features.py for readability)
from src.features.time_series.baseline_features import (
    compute_bb_width_features_from_series,
    compute_roc_5_from_series,
    compute_range_ratio_5bar_from_series,
    compute_price_range_symmetry_from_series,
    compute_wick_ratios_from_series,
    compute_poc_hal_features_from_series,
    compute_sqs_hal_high_from_series,
    compute_sqs_hal_low_from_series,
    compute_sr_strength_max_from_series,
)

from src.features.time_series.utils_liquidity_features import (
    compute_liquidity_void_features_from_series,
    compute_wpt_volume_energy_features_from_series,
)

from src.features.loader.selector_utils import select_columns_from_series

# 组合特征包装函数（交互特征 + 衍生特征）
from src.features.loader.common_derived_feature_wrappers import (
    compute_sr_strength_combined_wrapper,
    compute_sr_distance_normalized_wrapper,
    compute_dist_to_zz_high_wrapper,
    compute_dist_to_zz_low_wrapper,
    compute_dist_to_zz_high_atr_wrapper,
    compute_dist_to_zz_low_atr_wrapper,
    compute_cvd_slope_wrapper,
    compute_atr_ratio_wrapper,
    compute_bb_width_ratio_wrapper,
    compute_compression_score_wrapper,
    compute_tbr_ma_wrapper,
    compute_tbr_spike_wrapper,
)

from src.features.time_series.utils_interaction_features import compute_atr_ratio_from_series
from src.features.time_series.utils_interaction_features import (
    compute_bb_width_ratio_from_series,
    compute_compression_score_from_series,
    compute_liquidity_void_x_vpin_from_series,
    compute_liquidity_void_x_wpt_risk_from_series,
    compute_compression_energy_x_ofi_short_from_series,
    compute_hurst_x_trend_r2_from_series,
    compute_evt_x_trend_r2_from_series,
    compute_vpin_x_compression_from_series,
    compute_vpin_x_trade_cluster_max_buy_run_from_series,
    compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series,
    compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series,
    compute_vpin_x_trade_cluster_entropy_from_series,
    compute_sma_slope_x_price_pos_from_series,
    apply_rank_transform_to_interaction_from_series,
    compute_sr_strength_combined_from_series,
    compute_sr_distance_normalized_from_series,
    compute_dist_to_zz_high_from_series,
    compute_dist_to_zz_low_from_series,
    compute_dist_to_zz_high_atr_from_series,
    compute_dist_to_zz_low_atr_from_series,
    compute_cvd_slope_from_series,
)
from src.features.time_series.utils_interaction_features import (
    compute_tbr_ma_from_series,
    compute_tbr_spike_from_series,
)

# TA-Lib 与 DL 特征包装函数
from src.features.loader.talib_feature_wrappers import (
    compute_talib_indicator,
    compute_talib_indicator_from_series,
    compute_talib_sma,
    compute_talib_ema,
    compute_talib_rsi,
    compute_talib_macd,
)
from src.features.loader.dl_feature_wrappers import compute_dl_sequence_features

# 策略专属特征（已移除，现在通过统一的 YAML 配置加载）
# 所有特征现在通过 feature_dependencies.yaml 统一管理

FEATURE_FUNCTION_MAP: Dict[str, Callable] = {
    # ========================================================================
    # Baseline 特征（基础技术指标）
    # ========================================================================
    # Backward-compatible alias: some versions expose _compute_atr, others only compute_atr.
    "BaselineFeatureEngineer._compute_atr": getattr(BaselineFeatureEngineer, "_compute_atr", BaselineFeatureEngineer.compute_atr),
    "BaselineFeatureEngineer.compute_rsi": BaselineFeatureEngineer.compute_rsi,
    "BaselineFeatureEngineer.compute_macd": BaselineFeatureEngineer.compute_macd,
    "BaselineFeatureEngineer.compute_bollinger_bands": BaselineFeatureEngineer.compute_bollinger_bands,
    "BaselineFeatureEngineer.compute_atr": BaselineFeatureEngineer.compute_atr,
    "BaselineFeatureEngineer.compute_price_range_symmetry": BaselineFeatureEngineer.compute_price_range_symmetry,
    "BaselineFeatureEngineer.compute_wick_ratios": BaselineFeatureEngineer.compute_wick_ratios,
    "BaselineFeatureEngineer.compute_trend_r2_20": BaselineFeatureEngineer.compute_trend_r2_20,
    "BaselineFeatureEngineer.compute_trend_r2_50": BaselineFeatureEngineer.compute_trend_r2_50,
    "BaselineFeatureEngineer.compute_acceleration_3": BaselineFeatureEngineer.compute_acceleration_3,
    "BaselineFeatureEngineer.compute_atr_percentile": BaselineFeatureEngineer.compute_atr_percentile,
    "BaselineFeatureEngineer.compute_bb_width_features": BaselineFeatureEngineer.compute_bb_width_features,
    "compute_bb_width_features_from_series": compute_bb_width_features_from_series,
    "BaselineFeatureEngineer.compute_compression_to_breakout_prob": BaselineFeatureEngineer.compute_compression_to_breakout_prob,
    "BaselineFeatureEngineer.compute_range_ratio_5bar": BaselineFeatureEngineer.compute_range_ratio_5bar,
    "compute_range_ratio_5bar_from_series": compute_range_ratio_5bar_from_series,
    # Prefer narrow-input implementation for pipeline; keep old DF-based method for compatibility.
    "BaselineFeatureEngineer.compute_roc_5": BaselineFeatureEngineer.compute_roc_5,
    "compute_roc_5_from_series": compute_roc_5_from_series,
    "compute_acceleration_3_from_series": compute_acceleration_3_from_series,
    "compute_volume_anomaly_from_series": compute_volume_anomaly_from_series,
    "compute_trend_r2_20_from_series": compute_trend_r2_20_from_series,
    "compute_trend_r2_50_from_series": compute_trend_r2_50_from_series,
    "compute_slope_consistency_score_from_series": compute_slope_consistency_score_from_series,
    "compute_volatility_reversal_score_from_series": compute_volatility_reversal_score_from_series,
    "compute_atr_percentile_from_series": compute_atr_percentile_from_series,
    "compute_trend_volatility_alignment_from_series": compute_trend_volatility_alignment_from_series,
    "compute_compression_to_breakout_prob_from_series": compute_compression_to_breakout_prob_from_series,
    "compute_price_range_symmetry_from_series": compute_price_range_symmetry_from_series,
    "compute_wick_ratios_from_series": compute_wick_ratios_from_series,
    "BaselineFeatureEngineer.compute_slope_consistency_score": BaselineFeatureEngineer.compute_slope_consistency_score,
    "BaselineFeatureEngineer.compute_trend_volatility_alignment": BaselineFeatureEngineer.compute_trend_volatility_alignment,
    "BaselineFeatureEngineer.compute_volatility_reversal_score": BaselineFeatureEngineer.compute_volatility_reversal_score,
    "BaselineFeatureEngineer.compute_volume_anomaly": BaselineFeatureEngineer.compute_volume_anomaly,
    
    # Baseline SR 特征（注意：这些是静态方法，需要特殊处理）
    "BaselineFeatureEngineer.calculate_sqs": BaselineFeatureEngineer.calculate_sqs,
    "BaselineFeatureEngineer._compute_boundary_strengths": BaselineFeatureEngineer._compute_boundary_strengths,
    
    # SR 特征（narrow IO，用于配置文件直接调用）
    "compute_poc_hal_features_from_series": compute_poc_hal_features_from_series,
    "compute_sqs_hal_high_from_series": compute_sqs_hal_high_from_series,
    "compute_sqs_hal_low_from_series": compute_sqs_hal_low_from_series,
    "compute_sr_strength_max_from_series": compute_sr_strength_max_from_series,
    "compute_wpt_vpvr_from_series": compute_wpt_vpvr_from_series,
    "compute_unified_volume_profile": compute_unified_volume_profile,  # 新的统一实现
    "compute_footprint_features": compute_footprint_features,
    "compute_liquidity_void_features_from_series": compute_liquidity_void_features_from_series,
    "compute_wpt_volume_energy_features_from_series": compute_wpt_volume_energy_features_from_series,
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
    # Enhanced 特征（WPT, Hilbert, Hurst, Spectrum, Liquidity, Order Flow, GARCH, DTW, EVT）
    # ========================================================================
    "extract_wpt_features": extract_wpt_features,
    "extract_hilbert_features": extract_hilbert_features,
    "extract_hurst_features": extract_hurst_features,
    "extract_spectrum_features_from_series": extract_spectrum_features_from_series,
    "extract_liquidity_features": extract_liquidity_features,
    "extract_order_flow_features": extract_order_flow_features,
    "compute_vpin_derived_features_from_base": compute_vpin_derived_features_from_base,
    "compute_vpin_ma_max_features_from_base": compute_vpin_ma_max_features_from_base,
    "compute_vpin_change_features_from_base": compute_vpin_change_features_from_base,
    "compute_vpin_zscore_features_from_base": compute_vpin_zscore_features_from_base,
    "compute_vpin_ma_max_features_from_series": compute_vpin_ma_max_features_from_series,
    "compute_vpin_change_features_from_series": compute_vpin_change_features_from_series,
    "compute_vpin_zscore_features_from_series": compute_vpin_zscore_features_from_series,
    "compute_vpin_quantile_rank_features_from_base": compute_vpin_quantile_rank_features_from_base,
    "compute_vpin_volatility_features_from_base": compute_vpin_volatility_features_from_base,
    "compute_vpin_spike_features_from_base": compute_vpin_spike_features_from_base,
    "compute_vpin_momentum_features_from_base": compute_vpin_momentum_features_from_base,
    "compute_vpin_signed_zscore_features_from_base": compute_vpin_signed_zscore_features_from_base,
    "compute_vpin_quantile_rank_features_from_series": compute_vpin_quantile_rank_features_from_series,
    "compute_vpin_volatility_features_from_series": compute_vpin_volatility_features_from_series,
    "compute_vpin_spike_features_from_series": compute_vpin_spike_features_from_series,
    "compute_vpin_momentum_features_from_series": compute_vpin_momentum_features_from_series,
    "compute_vpin_signed_zscore_features_from_series": compute_vpin_signed_zscore_features_from_series,
    "compute_vpin_base_aligned_features_from_series": compute_vpin_base_aligned_features_from_series,
    "compute_trade_cluster_derived_features_from_base": compute_trade_cluster_derived_features_from_base,
    "compute_trade_cluster_ratio_features_from_base": compute_trade_cluster_ratio_features_from_base,
    "compute_trade_cluster_buy_sell_ratio_features_from_base": compute_trade_cluster_buy_sell_ratio_features_from_base,
    "compute_trade_cluster_max_run_ratio_features_from_base": compute_trade_cluster_max_run_ratio_features_from_base,
    "compute_trade_cluster_buy_sell_max_ratio_features_from_base": compute_trade_cluster_buy_sell_max_ratio_features_from_base,
    "compute_trade_cluster_avg_run_ratio_features_from_base": compute_trade_cluster_avg_run_ratio_features_from_base,
    "compute_trade_cluster_buy_sell_avg_ratio_features_from_base": compute_trade_cluster_buy_sell_avg_ratio_features_from_base,
    "compute_trade_cluster_run_length_features_from_base": compute_trade_cluster_run_length_features_from_base,
    "compute_trade_cluster_net_runs_counts_features_from_base": compute_trade_cluster_net_runs_counts_features_from_base,
    "compute_trade_cluster_net_runs_ratio_features_from_counts": compute_trade_cluster_net_runs_ratio_features_from_counts,
    "compute_trade_cluster_entropy_features_from_base": compute_trade_cluster_entropy_features_from_base,
    "compute_trade_cluster_entropy_ma_change_features_from_base": compute_trade_cluster_entropy_ma_change_features_from_base,
    "compute_trade_cluster_entropy_zscore_features_from_base": compute_trade_cluster_entropy_zscore_features_from_base,
    "compute_trade_cluster_max_buy_run_ma_features_from_base": compute_trade_cluster_max_buy_run_ma_features_from_base,
    "compute_trade_cluster_imbalance_ratio_ma_features_from_base": compute_trade_cluster_imbalance_ratio_ma_features_from_base,
    "compute_trade_cluster_net_runs_ma_features_from_base": compute_trade_cluster_net_runs_ma_features_from_base,
    "compute_trade_cluster_total_runs_ma_features_from_base": compute_trade_cluster_total_runs_ma_features_from_base,
    "compute_trade_cluster_imbalance_zscore_features_from_base": compute_trade_cluster_imbalance_zscore_features_from_base,
    "compute_trade_cluster_net_runs_zscore_features_from_base": compute_trade_cluster_net_runs_zscore_features_from_base,
    "compute_trade_cluster_max_buy_run_zscore_features_from_base": compute_trade_cluster_max_buy_run_zscore_features_from_base,
    "compute_trade_cluster_max_sell_run_zscore_features_from_base": compute_trade_cluster_max_sell_run_zscore_features_from_base,
    "compute_trade_cluster_ratio_features_from_series": compute_trade_cluster_ratio_features_from_series,
    "compute_trade_cluster_buy_sell_ratio_features_from_series": compute_trade_cluster_buy_sell_ratio_features_from_series,
    "compute_trade_cluster_max_run_ratio_features_from_series": compute_trade_cluster_max_run_ratio_features_from_series,
    "compute_trade_cluster_buy_sell_max_ratio_features_from_series": compute_trade_cluster_buy_sell_max_ratio_features_from_series,
    "compute_trade_cluster_avg_run_ratio_features_from_series": compute_trade_cluster_avg_run_ratio_features_from_series,
    "compute_trade_cluster_buy_sell_avg_ratio_features_from_series": compute_trade_cluster_buy_sell_avg_ratio_features_from_series,
    "compute_trade_cluster_run_length_features_from_series": compute_trade_cluster_run_length_features_from_series,
    "compute_trade_cluster_net_runs_counts_features_from_series": compute_trade_cluster_net_runs_counts_features_from_series,
    "compute_trade_cluster_net_runs_ratio_features_from_series": compute_trade_cluster_net_runs_ratio_features_from_series,
    "compute_trade_cluster_entropy_features_from_series": compute_trade_cluster_entropy_features_from_series,
    "compute_trade_cluster_entropy_ma_change_features_from_series": compute_trade_cluster_entropy_ma_change_features_from_series,
    "compute_trade_cluster_entropy_zscore_features_from_series": compute_trade_cluster_entropy_zscore_features_from_series,
    "compute_trade_cluster_max_buy_run_ma_features_from_series": compute_trade_cluster_max_buy_run_ma_features_from_series,
    "compute_trade_cluster_imbalance_ratio_ma_features_from_series": compute_trade_cluster_imbalance_ratio_ma_features_from_series,
    "compute_trade_cluster_net_runs_ma_features_from_series": compute_trade_cluster_net_runs_ma_features_from_series,
    "compute_trade_cluster_total_runs_ma_features_from_series": compute_trade_cluster_total_runs_ma_features_from_series,
    "compute_trade_cluster_imbalance_zscore_features_from_series": compute_trade_cluster_imbalance_zscore_features_from_series,
    "compute_trade_cluster_net_runs_zscore_features_from_series": compute_trade_cluster_net_runs_zscore_features_from_series,
    "compute_trade_cluster_max_buy_run_zscore_features_from_series": compute_trade_cluster_max_buy_run_zscore_features_from_series,
    "compute_trade_cluster_max_sell_run_zscore_features_from_series": compute_trade_cluster_max_sell_run_zscore_features_from_series,
    "compute_trade_cluster_base_aligned_features_from_series": compute_trade_cluster_base_aligned_features_from_series,
    "select_order_flow_features": select_order_flow_features,
    "select_vpin_block_features": select_vpin_block_features,
    "select_trade_cluster_block_features": select_trade_cluster_block_features,
    "extract_garch_features_from_series": extract_garch_features_from_series,
    "extract_dtw_features": extract_dtw_features,
    "extract_dtw_features_from_series": extract_dtw_features_from_series,
    "extract_evt_features_from_series": extract_evt_features_from_series,
    "extract_extended_volatility_features": extract_extended_volatility_features,
    "compute_vol_raw_features_from_series": compute_vol_raw_features_from_series,
    "compute_vol_atr_features_from_series": compute_vol_atr_features_from_series,
    "compute_vol_lag_features_from_series": compute_vol_lag_features_from_series,
    "compute_vol_trend_features_from_series": compute_vol_trend_features_from_series,
    "compute_vol_ma_features_from_series": compute_vol_ma_features_from_series,
    "compute_vol_regime_features_from_series": compute_vol_regime_features_from_series,
    "compute_vol_range_features_from_series": compute_vol_range_features_from_series,
    "compute_vol_mom_features_from_series": compute_vol_mom_features_from_series,
    "select_extended_volatility_features": select_extended_volatility_features,
    "extract_volume_profile_volatility_features_from_series": extract_volume_profile_volatility_features_from_series,
    "select_columns_from_series": select_columns_from_series,
    
    # ========================================================================
    # 交互特征（每个交互特征独立计算函数）
    # ========================================================================
    "compute_liquidity_void_x_wpt_risk": compute_liquidity_void_x_wpt_risk_wrapper,
    "compute_compression_energy_x_ofi_short": compute_compression_energy_x_ofi_short_wrapper,
    "compute_hurst_x_trend_r2": compute_hurst_x_trend_r2_wrapper,
    "compute_evt_x_trend_r2": compute_evt_x_trend_r2_wrapper,
    "compute_vpin_x_compression": compute_vpin_x_compression_wrapper,
    "compute_sma_slope_x_price_pos": compute_sma_slope_x_price_pos_wrapper,
    "compute_vpin_x_wick_upper": compute_vpin_x_wick_upper_wrapper,
    "compute_vpin_x_wick_lower": compute_vpin_x_wick_lower_wrapper,
    "compute_vpin_x_trade_cluster_max_buy_run": compute_vpin_x_trade_cluster_max_buy_run_wrapper,
    "compute_vpin_zscore_x_trade_cluster_max_buy_run": compute_vpin_zscore_x_trade_cluster_max_buy_run_wrapper,
    "compute_vpin_signed_imbalance_x_trade_cluster_imbalance": compute_vpin_signed_imbalance_x_trade_cluster_imbalance_wrapper,
    "compute_vpin_x_trade_cluster_entropy": compute_vpin_x_trade_cluster_entropy_wrapper,
    "apply_rank_transform_to_interaction": apply_rank_transform_to_interaction_wrapper,
    
    # ========================================================================
    # 组合特征（交互特征 + 衍生特征，所有策略可用）
    # ========================================================================
    "compute_sr_strength_combined": compute_sr_strength_combined_wrapper,
    "compute_sr_distance_normalized": compute_sr_distance_normalized_wrapper,
    "compute_dist_to_zz_high": compute_dist_to_zz_high_wrapper,
    "compute_dist_to_zz_low": compute_dist_to_zz_low_wrapper,
    "compute_dist_to_zz_high_atr": compute_dist_to_zz_high_atr_wrapper,
    "compute_dist_to_zz_low_atr": compute_dist_to_zz_low_atr_wrapper,
    "compute_cvd_slope": compute_cvd_slope_wrapper,
    "compute_sr_strength_combined_from_series": compute_sr_strength_combined_from_series,
    "compute_sr_distance_normalized_from_series": compute_sr_distance_normalized_from_series,
    "compute_dist_to_zz_high_from_series": compute_dist_to_zz_high_from_series,
    "compute_dist_to_zz_low_from_series": compute_dist_to_zz_low_from_series,
    "compute_dist_to_zz_high_atr_from_series": compute_dist_to_zz_high_atr_from_series,
    "compute_dist_to_zz_low_atr_from_series": compute_dist_to_zz_low_atr_from_series,
    "compute_cvd_slope_from_series": compute_cvd_slope_from_series,
    "compute_atr_ratio": compute_atr_ratio_wrapper,
    "compute_atr_ratio_from_series": compute_atr_ratio_from_series,
    "compute_bb_width_ratio": compute_bb_width_ratio_wrapper,
    "compute_bb_width_ratio_from_series": compute_bb_width_ratio_from_series,
    "compute_compression_score": compute_compression_score_wrapper,
    "compute_compression_score_from_series": compute_compression_score_from_series,
    "compute_tbr_ma": compute_tbr_ma_wrapper,
    "compute_tbr_ma_from_series": compute_tbr_ma_from_series,
    "compute_tbr_spike": compute_tbr_spike_wrapper,
    "compute_tbr_spike_from_series": compute_tbr_spike_from_series,
    "compute_liquidity_void_x_vpin_from_series": compute_liquidity_void_x_vpin_from_series,
    "compute_liquidity_void_x_wpt_risk_from_series": compute_liquidity_void_x_wpt_risk_from_series,
    "compute_compression_energy_x_ofi_short_from_series": compute_compression_energy_x_ofi_short_from_series,
    "compute_hurst_x_trend_r2_from_series": compute_hurst_x_trend_r2_from_series,
    "compute_evt_x_trend_r2_from_series": compute_evt_x_trend_r2_from_series,
    "compute_vpin_x_compression_from_series": compute_vpin_x_compression_from_series,
    "compute_sma_slope_x_price_pos_from_series": compute_sma_slope_x_price_pos_from_series,
    "compute_vpin_x_trade_cluster_max_buy_run_from_series": compute_vpin_x_trade_cluster_max_buy_run_from_series,
    "compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series": compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series,
    "compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series": compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series,
    "compute_vpin_x_trade_cluster_entropy_from_series": compute_vpin_x_trade_cluster_entropy_from_series,
    "apply_rank_transform_to_interaction_from_series": apply_rank_transform_to_interaction_from_series,
    
    # ========================================================================
    # TA-Lib 特征（按需计算单个指标）
    # ========================================================================
    "compute_talib_indicator": compute_talib_indicator,  # 通用函数，支持所有 talib 指标
    "compute_talib_indicator_from_series": compute_talib_indicator_from_series,
    "compute_talib_sma": compute_talib_sma,
    "compute_talib_ema": compute_talib_ema,
    "compute_talib_rsi": compute_talib_rsi,
    "compute_talib_macd": compute_talib_macd,
    "compute_dl_sequence_features": compute_dl_sequence_features,
    
    # ========================================================================
    # 策略专属特征构建函数（已移除，现在通过统一的 YAML 配置加载）
    # ========================================================================
    # 注意：所有特征现在通过 feature_dependencies.yaml 统一管理
    # 不再需要策略特定的 build_*_features 和 select_*_features 函数
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
