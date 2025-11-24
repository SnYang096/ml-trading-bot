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
from typing import Optional, Dict
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
    window: int = 64,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 Hilbert 特征（滚动窗口，无数据泄露）
    
    Args:
        df: DataFrame with WPT features
        price_fluctuation_col: Price fluctuation column
        cvd_fluctuation_col: CVD fluctuation column
        price_trend_col: Price trend column
        cvd_trend_col: CVD trend column
        window: Rolling window size for Hilbert transform (default: 64)
    
    Returns:
        DataFrame with Hilbert features added
    """
    df = df.copy()
    
    # 初始化特征列
    hilbert_cols = [
        "hilbert_price_phase",
        "hilbert_price_envelope",
        "hilbert_price_freq",
        "hilbert_price_envelope_slope",
        "hilbert_cvd_phase",
        "hilbert_cvd_envelope",
        "hilbert_cvd_freq",
        "hilbert_cvd_envelope_slope",
        "hilbert_phase_diff",
        "hilbert_cvd_leads",
        "hilbert_phase_diff_mean",
        "hilbert_phase_diff_std",
        "hilbert_cvd_leads_strong",
    ]
    for col in hilbert_cols:
        df[col] = np.nan
    
    # 对价格波动做 Hilbert（滚动窗口）
    if price_fluctuation_col in df.columns:
        price_fluc = df[price_fluctuation_col].values
        
        for i in range(window, len(df)):
            # 使用历史窗口数据 [i-window, i)
            window_data = price_fluc[i - window : i]
            
            if len(window_data) < 10:  # 最小长度要求
                continue
            
            price_hilbert = hilbert_transform(window_data, detrend=False)
            
            # 只使用最后一个点的值（当前时刻的特征）
            if len(price_hilbert["phase"]) > 0:
                df.iloc[i, df.columns.get_loc("hilbert_price_phase")] = (
                    price_hilbert["phase"][-1]
                )
                df.iloc[i, df.columns.get_loc("hilbert_price_envelope")] = (
                    price_hilbert["envelope"][-1]
                )
                df.iloc[i, df.columns.get_loc("hilbert_price_freq")] = (
                    price_hilbert["instantaneous_freq"][-1]
                )
                
                # 包络斜率（使用梯度）
                envelope_slope = np.gradient(price_hilbert["envelope"])
                if len(envelope_slope) > 0:
                    df.iloc[i, df.columns.get_loc("hilbert_price_envelope_slope")] = (
                        envelope_slope[-1]
                    )
    
    # 对 CVD 波动做 Hilbert（滚动窗口）
    if cvd_fluctuation_col and cvd_fluctuation_col in df.columns:
        cvd_fluc = df[cvd_fluctuation_col].values
        
        for i in range(window, len(df)):
            window_data = cvd_fluc[i - window : i]
            
            if len(window_data) < 10:
                continue
            
            cvd_hilbert = hilbert_transform(window_data, detrend=False)
            
            if len(cvd_hilbert["phase"]) > 0:
                df.iloc[i, df.columns.get_loc("hilbert_cvd_phase")] = (
                    cvd_hilbert["phase"][-1]
                )
                df.iloc[i, df.columns.get_loc("hilbert_cvd_envelope")] = (
                    cvd_hilbert["envelope"][-1]
                )
                df.iloc[i, df.columns.get_loc("hilbert_cvd_freq")] = (
                    cvd_hilbert["instantaneous_freq"][-1]
                )
                
                envelope_slope = np.gradient(cvd_hilbert["envelope"])
                if len(envelope_slope) > 0:
                    df.iloc[i, df.columns.get_loc("hilbert_cvd_envelope_slope")] = (
                        envelope_slope[-1]
                    )
                
                # 相位差（CVD - Price）
                if price_fluctuation_col in df.columns:
                    price_window = price_fluc[i - window : i]
                    if len(price_window) >= 10:
                        price_hilbert = hilbert_transform(price_window, detrend=False)
                        
                        if (len(cvd_hilbert["phase_unwrapped"]) > 0 and 
                            len(price_hilbert["phase_unwrapped"]) > 0):
                            phase_diff = (
                                cvd_hilbert["phase_unwrapped"][-1] - 
                                price_hilbert["phase_unwrapped"][-1]
                            )
                            df.iloc[i, df.columns.get_loc("hilbert_phase_diff")] = phase_diff
                            df.iloc[i, df.columns.get_loc("hilbert_cvd_leads")] = (
                                1.0 if phase_diff > 0 else 0.0
                            )
    
    # 相位差滚动统计（使用历史数据）
    if "hilbert_phase_diff" in df.columns:
        phase_diff_series = df["hilbert_phase_diff"]
        if len(phase_diff_series.dropna()) > 20:
            df["hilbert_phase_diff_mean"] = (
                phase_diff_series.rolling(window=20, min_periods=1).mean()
            )
            df["hilbert_phase_diff_std"] = (
                phase_diff_series.rolling(window=20, min_periods=1).std()
            )
            
            # CVD 是否持续领先（相位差 > 1 标准差且持续 3 根 K 线）
            df["hilbert_cvd_leads_strong"] = (
                (phase_diff_series > df["hilbert_phase_diff_mean"] + df["hilbert_phase_diff_std"])
                .rolling(window=3, min_periods=1)
                .min()
                .astype(float)
            )
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    for col in hilbert_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    # Fill NaN with 0 (after shift, NaN means insufficient history)
    df[hilbert_cols] = df[hilbert_cols].fillna(0.0)
    
    return df

