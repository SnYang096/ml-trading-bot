"""
频谱分析特征工程

核心功能（5个稳健特征）：
1. Spectral Flatness - 频谱平坦度（信号稀疏度/压缩度）
2. High-Freq Energy Ratio - 高频能量占比（噪声强度/流动性碎片化）
3. Low-Freq Energy Ratio - 低频能量占比（趋势/慢速资金主导）
4. Spectral Entropy - 谱熵（系统有序性）
5. Spectral Centroid - 频谱重心（能量集中在低频还是高频）

策略分配建议：
- Strategy 1 (Noise-Adaptive): High-Freq Energy Ratio, Spectral Entropy
- Strategy 2 (Regime Detection): Spectral Flatness, Low-Freq Energy Ratio
- Strategy 3 (Volatility Forecasting): High-Freq Energy Ratio, Spectral Centroid
- Strategy 4 (ML Quality Control): Spectral Flatness, Spectral Entropy

设计理念：
- 金融收益率通常无显著主频，直接提取"周期"容易误导
- 频谱的分布形态（平坦度、能量分布、熵）更能反映市场状态
- 适用于：噪声水平评估、趋势强度代理、极端事件预警、市场状态识别
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy import signal


def compute_spectrum_features(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> Dict[str, float]:
    """
    计算频谱特征（专注于稳健的频谱统计量，而非误导性的"主频"）
    
    Args:
        signal: 输入信号（建议使用收益率、差分等平稳序列）
        fs: 采样频率
        nperseg: 分段长度（默认根据信号长度动态设置）
    
    Returns:
        Dict with spectrum features:
        - has_dominant_freq: 是否存在显著主频（布尔，0/1），而非主频值本身
        - spectral_flatness: 频谱平坦度（0-1），越低表示能量越集中（趋势/共振）
        - high_freq_energy_ratio: 高频能量占比（0-1），越高表示噪声/流动性碎片化越强
        - low_freq_energy_ratio: 低频能量占比（0-1），越高表示长期驱动/宏观因素主导
        - spectral_entropy: 谱熵（0-1），越低表示系统有序性越高（如闪崩前的同步）
        - spectral_centroid: 频谱重心（Hz），能量集中在低频还是高频
    """
    # 提高最小长度要求，确保 Welch 方法有意义
    if len(signal) < 8:
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # 动态设置 nperseg，确保在合理范围内
    # 要求：8 <= nperseg <= min(len(signal), 64)
    if nperseg is None:
        nperseg = min(max(8, len(signal) // 2), 64)
    
    # 确保 nperseg 不超过信号长度，且至少为 4（Welch 最小要求）
    nperseg = min(nperseg, len(signal))
    if nperseg < 4:
        # 信号太短，返回默认值
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # Welch's method (更稳健的功率谱估计)
    try:
        freqs, psd = signal.welch(signal, fs=fs, nperseg=nperseg, scaling='density')
    except Exception:
        # 异常时返回默认值
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # 主频显著性检查（布尔特征，而非主频值）
    # 仅当主频处 PSD 显著高于均值时才认为存在显著主频
    psd_mean = np.mean(psd)
    psd_std = np.std(psd)
    dominant_freq_idx = np.argmax(psd)
    has_dominant_freq = float(psd[dominant_freq_idx] > (psd_mean + 2 * psd_std))
    
    # 频谱平坦度（越低越压缩，表示存在短暂趋势或共振）
    # 几何平均 / 算术平均
    psd_positive = psd[psd > 0]
    if len(psd_positive) > 0:
        geometric_mean = np.exp(np.mean(np.log(psd_positive)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = (
            geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0
        )
    else:
        spectral_flatness = 1.0
    
    # 频率分段：低频（0 ~ fs/8）、中频（fs/8 ~ fs/4）、高频（fs/4 ~ fs/2）
    nyquist = fs / 2
    low_freq_threshold = nyquist / 4  # fs/8
    mid_freq_threshold = nyquist / 2  # fs/4
    
    low_freq_mask = freqs <= low_freq_threshold
    high_freq_mask = freqs > mid_freq_threshold
    
    low_freq_energy = np.sum(psd[low_freq_mask])
    high_freq_energy = np.sum(psd[high_freq_mask])
    total_energy = np.sum(psd)
    
    low_freq_energy_ratio = (
        low_freq_energy / total_energy if total_energy > 0 else 0.0
    )
    high_freq_energy_ratio = (
        high_freq_energy / total_energy if total_energy > 0 else 0.0
    )
    
    # 谱熵（Spectral Entropy）：衡量频谱能量分布的均匀性
    # 越低表示能量越集中（系统有序性高），越高表示能量越分散（随机性强）
    # 公式：H = -Σ(p_i * log(p_i))，其中 p_i = PSD[i] / Σ(PSD)
    psd_normalized = psd / (total_energy + 1e-12)  # 归一化，避免除零
    # 只对非零值计算熵
    psd_nonzero = psd_normalized[psd_normalized > 1e-12]
    if len(psd_nonzero) > 0:
        spectral_entropy = -np.sum(psd_nonzero * np.log(psd_nonzero + 1e-12))
        # 归一化到 [0, 1]：除以最大可能熵 log(N)
        max_entropy = np.log(len(psd_normalized) + 1e-12)
        spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 1.0
    else:
        spectral_entropy = 1.0
    
    # 频谱重心（Spectral Centroid）：能量集中在低频还是高频
    # 公式：C = Σ(f_i * PSD_i) / Σ(PSD_i)
    # 值越大表示能量越集中在高频（噪声/冲击），值越小表示能量越集中在低频（趋势/慢速）
    if total_energy > 1e-12:
        spectral_centroid = np.sum(freqs * psd) / total_energy
    else:
        spectral_centroid = 0.0
    
    return {
        "has_dominant_freq": has_dominant_freq,
        "spectral_flatness": float(spectral_flatness),
        "high_freq_energy_ratio": float(high_freq_energy_ratio),
        "low_freq_energy_ratio": float(low_freq_energy_ratio),
        "spectral_entropy": float(spectral_entropy),
        "spectral_centroid": float(spectral_centroid),
    }


def extract_spectrum_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: Optional[str] = None,
    cvd_col: Optional[str] = None,
    rolling_window: int = 64,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取频谱特征
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        volume_col: Volume column name (optional)
        cvd_col: CVD column name (optional)
        rolling_window: Rolling window for spectrum calculation
    
    Returns:
        DataFrame with spectrum features added
    """
    df = df.copy()
    
    # 价格收益率频谱
    if price_col in df.columns:
        price_returns = df[price_col].pct_change().fillna(0).values
        
        # 滚动频谱特征（5个核心特征）
        has_dominant_freqs = []
        spectral_flatness = []
        high_freq_ratios = []
        low_freq_ratios = []
        spectral_entropy = []
        spectral_centroid = []
        
        for i in range(len(df)):
            if i < rolling_window:
                has_dominant_freqs.append(0.0)
                spectral_flatness.append(1.0)
                high_freq_ratios.append(0.0)
                low_freq_ratios.append(0.0)
                spectral_entropy.append(1.0)
                spectral_centroid.append(0.0)
            else:
                window_returns = price_returns[i - rolling_window : i]
                spec_features = compute_spectrum_features(window_returns)
                has_dominant_freqs.append(spec_features["has_dominant_freq"])
                spectral_flatness.append(spec_features["spectral_flatness"])
                high_freq_ratios.append(spec_features["high_freq_energy_ratio"])
                low_freq_ratios.append(spec_features["low_freq_energy_ratio"])
                spectral_entropy.append(spec_features["spectral_entropy"])
                spectral_centroid.append(spec_features["spectral_centroid"])
        
        df["spectrum_price_has_dominant_freq"] = has_dominant_freqs
        df["spectrum_price_flatness"] = spectral_flatness
        df["spectrum_price_high_freq_ratio"] = high_freq_ratios
        df["spectrum_price_low_freq_ratio"] = low_freq_ratios
        df["spectrum_price_entropy"] = spectral_entropy
        df["spectrum_price_centroid"] = spectral_centroid
    
    # 成交量频谱（滚动窗口）
    if volume_col and volume_col in df.columns:
        volume = df[volume_col].values
        volume_diff = np.diff(volume, prepend=volume[0])
        
        df["spectrum_volume_flatness"] = np.nan
        df["spectrum_volume_high_freq_ratio"] = np.nan
        df["spectrum_volume_low_freq_ratio"] = np.nan
        df["spectrum_volume_entropy"] = np.nan
        df["spectrum_volume_centroid"] = np.nan
        
        for i in range(rolling_window, len(df)):
            window_diff = volume_diff[i - rolling_window : i]
            spec_features = compute_spectrum_features(window_diff)
            df.iloc[i, df.columns.get_loc("spectrum_volume_flatness")] = (
                spec_features["spectral_flatness"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_volume_high_freq_ratio")] = (
                spec_features["high_freq_energy_ratio"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_volume_low_freq_ratio")] = (
                spec_features["low_freq_energy_ratio"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_volume_entropy")] = (
                spec_features["spectral_entropy"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_volume_centroid")] = (
                spec_features["spectral_centroid"]
            )
    
    # CVD 频谱（滚动窗口）
    if cvd_col and cvd_col in df.columns:
        cvd = df[cvd_col].values
        cvd_diff = np.diff(cvd, prepend=cvd[0])
        
        df["spectrum_cvd_flatness"] = np.nan
        df["spectrum_cvd_high_freq_ratio"] = np.nan
        df["spectrum_cvd_low_freq_ratio"] = np.nan
        df["spectrum_cvd_entropy"] = np.nan
        df["spectrum_cvd_centroid"] = np.nan
        
        for i in range(rolling_window, len(df)):
            window_diff = cvd_diff[i - rolling_window : i]
            spec_features = compute_spectrum_features(window_diff)
            df.iloc[i, df.columns.get_loc("spectrum_cvd_flatness")] = (
                spec_features["spectral_flatness"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_cvd_high_freq_ratio")] = (
                spec_features["high_freq_energy_ratio"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_cvd_low_freq_ratio")] = (
                spec_features["low_freq_energy_ratio"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_cvd_entropy")] = (
                spec_features["spectral_entropy"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_cvd_centroid")] = (
                spec_features["spectral_centroid"]
            )
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    spectrum_cols = [col for col in df.columns if col.startswith("spectrum_")]
    for col in spectrum_cols:
        df[col] = df[col].shift(1)
    
    # Fill NaN with default values (after shift)
    if "spectrum_price_has_dominant_freq" in df.columns:
        df["spectrum_price_has_dominant_freq"] = df["spectrum_price_has_dominant_freq"].fillna(0.0)
        df["spectrum_price_flatness"] = df["spectrum_price_flatness"].fillna(1.0)
        df["spectrum_price_high_freq_ratio"] = df["spectrum_price_high_freq_ratio"].fillna(0.0)
        df["spectrum_price_low_freq_ratio"] = df["spectrum_price_low_freq_ratio"].fillna(0.0)
        df["spectrum_price_entropy"] = df["spectrum_price_entropy"].fillna(1.0)
        df["spectrum_price_centroid"] = df["spectrum_price_centroid"].fillna(0.0)
    
    if "spectrum_volume_flatness" in df.columns:
        df["spectrum_volume_flatness"] = df["spectrum_volume_flatness"].fillna(1.0)
        df["spectrum_volume_high_freq_ratio"] = df["spectrum_volume_high_freq_ratio"].fillna(0.0)
        df["spectrum_volume_low_freq_ratio"] = df["spectrum_volume_low_freq_ratio"].fillna(0.0)
        df["spectrum_volume_entropy"] = df["spectrum_volume_entropy"].fillna(1.0)
        df["spectrum_volume_centroid"] = df["spectrum_volume_centroid"].fillna(0.0)
    
    if "spectrum_cvd_flatness" in df.columns:
        df["spectrum_cvd_flatness"] = df["spectrum_cvd_flatness"].fillna(1.0)
        df["spectrum_cvd_high_freq_ratio"] = df["spectrum_cvd_high_freq_ratio"].fillna(0.0)
        df["spectrum_cvd_low_freq_ratio"] = df["spectrum_cvd_low_freq_ratio"].fillna(0.0)
        df["spectrum_cvd_entropy"] = df["spectrum_cvd_entropy"].fillna(1.0)
        df["spectrum_cvd_centroid"] = df["spectrum_cvd_centroid"].fillna(0.0)
    
    return df


def add_spectrum_derived_features(
    df: pd.DataFrame,
    prefix: str = "spectrum_price",
    zscore_window: int = 50,
    diff_periods: List[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    为频谱特征添加派生特征（滚动z-score、变化率等），用于策略特定需求
    
    Args:
        df: DataFrame with spectrum features
        prefix: Prefix of spectrum features (e.g., "spectrum_price", "spectrum_volume")
        zscore_window: Window size for rolling z-score normalization
        diff_periods: List of periods for difference features
    
    Returns:
        DataFrame with derived features added:
        - {prefix}_flatness_zscore: Rolling z-score of flatness
        - {prefix}_high_freq_ratio_diff_{period}: Difference of high-freq ratio
        - etc.
    """
    df = df.copy()
    
    # 核心特征列表
    core_features = ["flatness", "high_freq_ratio", "low_freq_ratio", "entropy", "centroid"]
    
    for feature in core_features:
        col_name = f"{prefix}_{feature}"
        if col_name not in df.columns:
            continue
        
        # 1. 滚动 z-score（用于策略2和4：regime detection）
        zscore_col = f"{col_name}_zscore"
        rolling_mean = df[col_name].rolling(window=zscore_window, min_periods=zscore_window//2).mean()
        rolling_std = df[col_name].rolling(window=zscore_window, min_periods=zscore_window//2).std()
        df[zscore_col] = (df[col_name] - rolling_mean) / (rolling_std + 1e-8)
        
        # 2. 变化率特征（用于策略3：volatility forecasting）
        for period in diff_periods:
            diff_col = f"{col_name}_diff_{period}"
            df[diff_col] = df[col_name].diff(period)
        
        # 3. 滚动变化率（用于检测突变）
        change_col = f"{col_name}_change"
        df[change_col] = df[col_name].pct_change()
    
    return df


def get_strategy_spectrum_features(
    df: pd.DataFrame,
    strategy: str,
    prefix: str = "spectrum_price",
) -> pd.DataFrame:
    """
    根据策略类型返回相关的频谱特征
    
    Args:
        df: DataFrame with spectrum features
        strategy: Strategy name ("noise_adaptive", "regime_detection", 
                 "volatility_forecasting", "ml_quality_control")
        prefix: Prefix of spectrum features
    
    Returns:
        DataFrame with selected features for the strategy
    """
    strategy_features = {
        "noise_adaptive": [
            f"{prefix}_high_freq_ratio",
            f"{prefix}_entropy",
            f"{prefix}_high_freq_ratio_zscore",
            f"{prefix}_entropy_zscore",
        ],
        "regime_detection": [
            f"{prefix}_flatness",
            f"{prefix}_low_freq_ratio",
            f"{prefix}_flatness_zscore",
            f"{prefix}_low_freq_ratio_zscore",
        ],
        "volatility_forecasting": [
            f"{prefix}_high_freq_ratio",
            f"{prefix}_centroid",
            f"{prefix}_high_freq_ratio_diff_5",
            f"{prefix}_centroid_diff_5",
        ],
        "ml_quality_control": [
            f"{prefix}_flatness",
            f"{prefix}_entropy",
            f"{prefix}_flatness_zscore",
            f"{prefix}_entropy_zscore",
        ],
    }
    
    if strategy not in strategy_features:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be one of: {list(strategy_features.keys())}"
        )
    
    # 先添加派生特征
    df = add_spectrum_derived_features(df, prefix=prefix)
    
    # 返回策略相关的特征
    available_features = [f for f in strategy_features[strategy] if f in df.columns]
    return df[available_features]

