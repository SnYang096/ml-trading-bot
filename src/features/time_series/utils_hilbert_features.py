"""
Hilbert 变换特征工程

核心功能：
1. 对去趋势后的信号做 Hilbert 变换
2. 提取瞬时相位、包络、频率
3. 计算相位差（CVD vs Price）以捕捉领先-滞后关系
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from scipy.signal import hilbert


def hilbert_transform(
    signal: np.ndarray,
    detrend: bool = True,
    trend: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Hilbert 变换，提取解析信号
    
    Args:
        signal: 输入信号
        detrend: 是否去趋势
        trend: 趋势信号（如果 detrend=True 且提供）
    
    Returns:
        Dict with keys: 'analytic', 'phase', 'envelope', 'instantaneous_freq', 'phase_unwrapped'
    """
    # 去趋势
    if detrend:
        if trend is not None:
            signal_detrended = signal - trend
        else:
            # 简单去均值
            signal_detrended = signal - np.mean(signal)
    else:
        signal_detrended = signal
    
    # Hilbert 变换
    analytic = hilbert(signal_detrended)
    
    # 瞬时相位
    phase = np.angle(analytic)
    phase_unwrapped = np.unwrap(phase)  # 解缠绕
    
    # 包络
    envelope = np.abs(analytic)
    
    # 瞬时频率（相位的导数）
    instantaneous_freq = np.diff(phase_unwrapped) / (2 * np.pi)
    # 对齐长度
    instantaneous_freq = np.concatenate([[instantaneous_freq[0]], instantaneous_freq])
    
    return {
        "analytic": analytic,
        "phase": phase,
        "phase_unwrapped": phase_unwrapped,
        "envelope": envelope,
        "instantaneous_freq": instantaneous_freq,
    }


def compute_phase_lead(
    price_fluctuation: np.ndarray,
    cvd_fluctuation: np.ndarray,
    price_trend: Optional[np.ndarray] = None,
    cvd_trend: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    计算 CVD 与 Price 的相位差，判断领先-滞后关系
    
    Args:
        price_fluctuation: 价格波动分量（已去趋势）
        cvd_fluctuation: CVD 波动分量（已去趋势）
        price_trend: 价格趋势（可选，用于验证）
        cvd_trend: CVD 趋势（可选，用于验证）
    
    Returns:
        Dict with phase difference and related features
    """
    # Hilbert 变换
    price_hilbert = hilbert_transform(price_fluctuation, detrend=False)
    cvd_hilbert = hilbert_transform(cvd_fluctuation, detrend=False)
    
    # 相位差（CVD - Price）
    phase_diff = cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
    
    # 包络斜率（波动率加速/减速）
    price_envelope_slope = np.gradient(price_hilbert["envelope"])
    cvd_envelope_slope = np.gradient(cvd_hilbert["envelope"])
    
    # 瞬时频率稳定性（高频震荡 vs 趋势运行）
    price_freq_stability = 1.0 / (1.0 + np.std(price_hilbert["instantaneous_freq"]))
    cvd_freq_stability = 1.0 / (1.0 + np.std(cvd_hilbert["instantaneous_freq"]))
    
    return {
        "phase_diff": phase_diff,
        "phase_diff_positive": phase_diff > 0,  # CVD 是否领先
        "price_envelope": price_hilbert["envelope"],
        "cvd_envelope": cvd_hilbert["envelope"],
        "price_envelope_slope": price_envelope_slope,
        "cvd_envelope_slope": cvd_envelope_slope,
        "price_freq_stability": price_freq_stability,
        "cvd_freq_stability": cvd_freq_stability,
    }


def extract_hilbert_features(
    df: pd.DataFrame,
    price_fluctuation_col: str = "wpt_price_fluctuation",
    cvd_fluctuation_col: Optional[str] = "wpt_cvd_fluctuation",
    price_trend_col: Optional[str] = "wpt_price_trend",
    cvd_trend_col: Optional[str] = "wpt_cvd_trend",
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 Hilbert 特征
    
    Args:
        df: DataFrame with WPT features
        price_fluctuation_col: Price fluctuation column
        cvd_fluctuation_col: CVD fluctuation column
        price_trend_col: Price trend column
        cvd_trend_col: CVD trend column
    
    Returns:
        DataFrame with Hilbert features added
    """
    df = df.copy()
    
    # 对价格波动做 Hilbert
    if price_fluctuation_col in df.columns:
        price_fluc = df[price_fluctuation_col].values
        price_hilbert = hilbert_transform(price_fluc, detrend=False)
        
        df["hilbert_price_phase"] = price_hilbert["phase"]
        df["hilbert_price_envelope"] = price_hilbert["envelope"]
        df["hilbert_price_freq"] = price_hilbert["instantaneous_freq"]
        
        # 包络斜率
        df["hilbert_price_envelope_slope"] = np.gradient(price_hilbert["envelope"])
    
    # 对 CVD 波动做 Hilbert
    if cvd_fluctuation_col and cvd_fluctuation_col in df.columns:
        cvd_fluc = df[cvd_fluctuation_col].values
        cvd_hilbert = hilbert_transform(cvd_fluc, detrend=False)
        
        df["hilbert_cvd_phase"] = cvd_hilbert["phase"]
        df["hilbert_cvd_envelope"] = cvd_hilbert["envelope"]
        df["hilbert_cvd_freq"] = cvd_hilbert["instantaneous_freq"]
        
        # 包络斜率
        df["hilbert_cvd_envelope_slope"] = np.gradient(cvd_hilbert["envelope"])
        
        # 相位差（CVD - Price）
        if price_fluctuation_col in df.columns:
            phase_diff = cvd_hilbert["phase_unwrapped"] - price_hilbert["phase_unwrapped"]
            df["hilbert_phase_diff"] = phase_diff
            df["hilbert_cvd_leads"] = (phase_diff > 0).astype(float)
            
            # 相位差滚动统计
            if len(df) > 20:
                df["hilbert_phase_diff_mean"] = (
                    pd.Series(phase_diff).rolling(window=20, min_periods=1).mean()
                )
                df["hilbert_phase_diff_std"] = (
                    pd.Series(phase_diff).rolling(window=20, min_periods=1).std()
                )
                # CVD 是否持续领先（相位差 > 1 标准差且持续 3 根 K 线）
                df["hilbert_cvd_leads_strong"] = (
                    (phase_diff > df["hilbert_phase_diff_mean"] + df["hilbert_phase_diff_std"])
                    .rolling(window=3, min_periods=1)
                    .min()
                    .astype(float)
                )
    
    return df

