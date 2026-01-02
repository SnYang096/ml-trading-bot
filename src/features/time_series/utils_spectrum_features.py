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
from scipy import signal as sp_signal

from src.features.registry import register_feature


def compute_spectrum_features(
    x: np.ndarray,
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
    if len(x) < 8:
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
        nperseg = min(max(8, len(x) // 2), 64)
    
    # 确保 nperseg 不超过信号长度，且至少为 4（Welch 最小要求）
    nperseg = min(nperseg, len(x))
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
        freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
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
        # Clamp all bounded/statistical outputs defensively for NN stability.
        "spectral_flatness": float(np.clip(spectral_flatness, 0.0, 1.0)),
        "high_freq_energy_ratio": float(np.clip(high_freq_energy_ratio, 0.0, 1.0)),
        "low_freq_energy_ratio": float(np.clip(low_freq_energy_ratio, 0.0, 1.0)),
        "spectral_entropy": float(np.clip(spectral_entropy, 0.0, 1.0)),
        # centroid is in Hz and depends on fs; keep as-is (unitless in bar-time scale only if fs fixed).
        "spectral_centroid": float(np.clip(spectral_centroid, 0.0, fs / 2 if fs > 0 else 0.5)),
    }


@register_feature("extract_spectrum_features_from_series", category="spectrum")
def extract_spectrum_features_from_series(
    *,
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    cvd: Optional[pd.Series] = None,
    rolling_window: int = 64,
) -> pd.DataFrame:
    """
    Narrow-IO spectrum feature entrypoint for the feature DAG.

    - Always returns the full set of spectrum columns (price + optional volume/cvd),
      leaving optional blocks as NaN if not provided.
    - Designed to be used with YAML `pass_full_df: false` + `column_mappings`.
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    n = len(close)
    idx = close.index

    # Allocate outputs (match extract_spectrum_features defaults)
    price_has_dom = np.zeros(n, dtype=float)
    price_flat = np.ones(n, dtype=float)
    price_high = np.zeros(n, dtype=float)
    price_low = np.zeros(n, dtype=float)
    price_ent = np.ones(n, dtype=float)
    price_cent = np.zeros(n, dtype=float)

    vol_flat = np.full(n, np.nan, dtype=float)
    vol_high = np.full(n, np.nan, dtype=float)
    vol_low = np.full(n, np.nan, dtype=float)
    vol_ent = np.full(n, np.nan, dtype=float)
    vol_cent = np.full(n, np.nan, dtype=float)

    cvd_flat = np.full(n, np.nan, dtype=float)
    cvd_high = np.full(n, np.nan, dtype=float)
    cvd_low = np.full(n, np.nan, dtype=float)
    cvd_ent = np.full(n, np.nan, dtype=float)
    cvd_cent = np.full(n, np.nan, dtype=float)

    # Price rolling spectrum
    price_returns = close.pct_change().fillna(0.0).values
    for i in range(rolling_window, n):
        window_returns = price_returns[i - rolling_window : i]
        spec = compute_spectrum_features(window_returns)
        price_has_dom[i] = spec["has_dominant_freq"]
        price_flat[i] = spec["spectral_flatness"]
        price_high[i] = spec["high_freq_energy_ratio"]
        price_low[i] = spec["low_freq_energy_ratio"]
        price_ent[i] = spec["spectral_entropy"]
        price_cent[i] = spec["spectral_centroid"]

    # Optional: volume rolling spectrum (diff)
    if volume is not None:
        volume = pd.to_numeric(volume, errors="coerce").astype(float)
        v = volume.values
        v_diff = np.diff(v, prepend=v[0] if len(v) else 0.0)
        for i in range(rolling_window, n):
            spec = compute_spectrum_features(v_diff[i - rolling_window : i])
            vol_flat[i] = spec["spectral_flatness"]
            vol_high[i] = spec["high_freq_energy_ratio"]
            vol_low[i] = spec["low_freq_energy_ratio"]
            vol_ent[i] = spec["spectral_entropy"]
            vol_cent[i] = spec["spectral_centroid"]

    # Optional: cvd rolling spectrum (diff)
    if cvd is not None:
        cvd = pd.to_numeric(cvd, errors="coerce").astype(float)
        c = cvd.values
        c_diff = np.diff(c, prepend=c[0] if len(c) else 0.0)
        for i in range(rolling_window, n):
            spec = compute_spectrum_features(c_diff[i - rolling_window : i])
            cvd_flat[i] = spec["spectral_flatness"]
            cvd_high[i] = spec["high_freq_energy_ratio"]
            cvd_low[i] = spec["low_freq_energy_ratio"]
            cvd_ent[i] = spec["spectral_entropy"]
            cvd_cent[i] = spec["spectral_centroid"]

    return pd.DataFrame(
        {
            "spectrum_price_has_dominant_freq": price_has_dom,
            "spectrum_price_flatness": price_flat,
            "spectrum_price_high_freq_ratio": price_high,
            "spectrum_price_low_freq_ratio": price_low,
            "spectrum_price_entropy": price_ent,
            "spectrum_price_centroid": price_cent,
            "spectrum_volume_flatness": vol_flat,
            "spectrum_volume_high_freq_ratio": vol_high,
            "spectrum_volume_low_freq_ratio": vol_low,
            "spectrum_volume_entropy": vol_ent,
            "spectrum_volume_centroid": vol_cent,
            "spectrum_cvd_flatness": cvd_flat,
            "spectrum_cvd_high_freq_ratio": cvd_high,
            "spectrum_cvd_low_freq_ratio": cvd_low,
            "spectrum_cvd_entropy": cvd_ent,
            "spectrum_cvd_centroid": cvd_cent,
        },
        index=idx,
    )


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
