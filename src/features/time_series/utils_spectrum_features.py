"""
频谱分析特征工程

核心功能：
1. 主频提取（周期长度）
2. 频谱平坦度（压缩度）
3. 高频能量占比（噪声强度）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from scipy import signal


def compute_spectrum_features(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> Dict[str, float]:
    """
    计算频谱特征
    
    Args:
        signal: 输入信号
        fs: 采样频率
        nperseg: 分段长度（默认 len(signal)//4）
    
    Returns:
        Dict with spectrum features
    """
    if len(signal) < 10:
        return {
            "dominant_freq": 0.0,
            "period_length": np.inf,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
        }
    
    if nperseg is None:
        nperseg = min(len(signal) // 4, 64)
    
    # Welch's method (更稳健的功率谱估计)
    freqs, psd = signal.welch(signal, fs=fs, nperseg=nperseg)
    
    # 主频
    dominant_freq_idx = np.argmax(psd)
    dominant_freq = freqs[dominant_freq_idx]
    period_length = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
    
    # 频谱平坦度（越低越压缩）
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
    
    # 高频能量占比（噪声强度）
    # 定义高频为 > 中位数频率
    median_freq_idx = len(freqs) // 2
    high_freq_energy = np.sum(psd[median_freq_idx:])
    total_energy = np.sum(psd)
    high_freq_energy_ratio = (
        high_freq_energy / total_energy if total_energy > 0 else 0.0
    )
    
    return {
        "dominant_freq": dominant_freq,
        "period_length": period_length,
        "spectral_flatness": spectral_flatness,
        "high_freq_energy_ratio": high_freq_energy_ratio,
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
        
        # 滚动频谱特征
        dominant_freqs = []
        period_lengths = []
        spectral_flatness = []
        high_freq_ratios = []
        
        for i in range(len(df)):
            if i < rolling_window:
                dominant_freqs.append(0.0)
                period_lengths.append(np.inf)
                spectral_flatness.append(1.0)
                high_freq_ratios.append(0.0)
            else:
                window_returns = price_returns[i - rolling_window : i]
                spec_features = compute_spectrum_features(window_returns)
                dominant_freqs.append(spec_features["dominant_freq"])
                period_lengths.append(spec_features["period_length"])
                spectral_flatness.append(spec_features["spectral_flatness"])
                high_freq_ratios.append(spec_features["high_freq_energy_ratio"])
        
        df["spectrum_price_dominant_freq"] = dominant_freqs
        df["spectrum_price_period"] = period_lengths
        df["spectrum_price_flatness"] = spectral_flatness
        df["spectrum_price_high_freq_ratio"] = high_freq_ratios
    
    # 成交量频谱（滚动窗口）
    if volume_col and volume_col in df.columns:
        volume = df[volume_col].values
        volume_diff = np.diff(volume, prepend=volume[0])
        
        df["spectrum_volume_dominant_freq"] = np.nan
        df["spectrum_volume_flatness"] = np.nan
        
        for i in range(rolling_window, len(df)):
            window_diff = volume_diff[i - rolling_window : i]
            spec_features = compute_spectrum_features(window_diff)
            df.iloc[i, df.columns.get_loc("spectrum_volume_dominant_freq")] = (
                spec_features["dominant_freq"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_volume_flatness")] = (
                spec_features["spectral_flatness"]
            )
    
    # CVD 频谱（滚动窗口）
    if cvd_col and cvd_col in df.columns:
        cvd = df[cvd_col].values
        cvd_diff = np.diff(cvd, prepend=cvd[0])
        
        df["spectrum_cvd_dominant_freq"] = np.nan
        df["spectrum_cvd_flatness"] = np.nan
        
        for i in range(rolling_window, len(df)):
            window_diff = cvd_diff[i - rolling_window : i]
            spec_features = compute_spectrum_features(window_diff)
            df.iloc[i, df.columns.get_loc("spectrum_cvd_dominant_freq")] = (
                spec_features["dominant_freq"]
            )
            df.iloc[i, df.columns.get_loc("spectrum_cvd_flatness")] = (
                spec_features["spectral_flatness"]
            )
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    spectrum_cols = [col for col in df.columns if col.startswith("spectrum_")]
    for col in spectrum_cols:
        df[col] = df[col].shift(1)
    
    # Fill NaN with default values (after shift)
    df["spectrum_price_dominant_freq"] = df["spectrum_price_dominant_freq"].fillna(0.0)
    df["spectrum_price_period"] = df["spectrum_price_period"].fillna(np.inf)
    df["spectrum_price_flatness"] = df["spectrum_price_flatness"].fillna(1.0)
    df["spectrum_price_high_freq_ratio"] = df["spectrum_price_high_freq_ratio"].fillna(0.0)
    
    if "spectrum_volume_dominant_freq" in df.columns:
        df["spectrum_volume_dominant_freq"] = df["spectrum_volume_dominant_freq"].fillna(0.0)
        df["spectrum_volume_flatness"] = df["spectrum_volume_flatness"].fillna(1.0)
    
    if "spectrum_cvd_dominant_freq" in df.columns:
        df["spectrum_cvd_dominant_freq"] = df["spectrum_cvd_dominant_freq"].fillna(0.0)
        df["spectrum_cvd_flatness"] = df["spectrum_cvd_flatness"].fillna(1.0)
    
    return df

