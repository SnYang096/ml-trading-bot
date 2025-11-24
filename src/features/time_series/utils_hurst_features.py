"""
Hurst 指数特征工程

核心功能：
1. 计算全序列 Hurst 指数
2. 滚动 Hurst（捕捉市场状态切换）
3. WPT 子带上的 Hurst
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_hurst_rs(
    returns: np.ndarray,
    max_lag: Optional[int] = None,
) -> float:
    """
    使用 R/S 方法计算 Hurst 指数
    
    Args:
        returns: 收益率序列
        max_lag: 最大滞后（默认 len(returns)//4）
    
    Returns:
        Hurst 指数（0-1之间）
    """
    if len(returns) < 10:
        return 0.5  # 默认随机游走
    
    if max_lag is None:
        max_lag = len(returns) // 4
    
    lags = np.logspace(1, np.log10(max_lag), 10).astype(int)
    lags = np.unique(lags)
    lags = lags[lags < len(returns)]
    
    if len(lags) < 2:
        return 0.5
    
    rs_values = []
    
    for lag in lags:
        # 分段
        n_segments = len(returns) // lag
        if n_segments < 2:
            continue
        
        rs_segments = []
        
        for i in range(n_segments):
            segment = returns[i * lag : (i + 1) * lag]
            if len(segment) < 2:
                continue
            
            # 去均值
            mean_segment = np.mean(segment)
            deviations = segment - mean_segment
            
            # 累积偏差
            cumsum = np.cumsum(deviations)
            
            # 范围
            R = np.max(cumsum) - np.min(cumsum)
            
            # 标准差
            S = np.std(segment)
            
            if S > 0:
                rs_segments.append(R / S)
        
        if rs_segments:
            rs_values.append(np.mean(rs_segments))
    
    if len(rs_values) < 2:
        return 0.5
    
    # 线性回归 log(R/S) vs log(lag)
    log_lags = np.log(lags[: len(rs_values)])
    log_rs = np.log(rs_values)
    
    # 简单线性回归
    if len(log_lags) > 1:
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        hurst = np.clip(hurst, 0.0, 1.0)
    else:
        hurst = 0.5
    
    return hurst


def compute_hurst_dfa(
    returns: np.ndarray,
    min_window: int = 4,
    max_window: Optional[int] = None,
) -> float:
    """
    使用 DFA (Detrended Fluctuation Analysis) 计算 Hurst 指数
    
    这是更稳健的方法，适合非平稳序列
    
    Args:
        returns: 收益率序列
        min_window: 最小窗口
        max_window: 最大窗口
    
    Returns:
        Hurst 指数
    """
    if len(returns) < 20:
        return 0.5
    
    if max_window is None:
        max_window = len(returns) // 4
    
    # 累积和
    y = np.cumsum(returns - np.mean(returns))
    
    windows = np.logspace(np.log10(min_window), np.log10(max_window), 10).astype(int)
    windows = np.unique(windows)
    windows = windows[windows < len(returns) // 2]
    
    if len(windows) < 2:
        return 0.5
    
    fluctuations = []
    
    for window in windows:
        n_segments = len(y) // window
        if n_segments < 2:
            continue
        
        fluct_segments = []
        
        for i in range(n_segments):
            segment = y[i * window : (i + 1) * window]
            
            # 去趋势（线性拟合）
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            
            # 波动
            fluct = np.sqrt(np.mean(detrended ** 2))
            fluct_segments.append(fluct)
        
        if fluct_segments:
            fluctuations.append(np.mean(fluct_segments))
    
    if len(fluctuations) < 2:
        return 0.5
    
    # 线性回归 log(F) vs log(window)
    log_windows = np.log(windows[: len(fluctuations)])
    log_fluct = np.log(fluctuations)
    
    if len(log_windows) > 1:
        hurst = np.polyfit(log_windows, log_fluct, 1)[0]
        hurst = np.clip(hurst, 0.0, 1.0)
    else:
        hurst = 0.5
    
    return hurst


def extract_hurst_features(
    df: pd.DataFrame,
    price_col: str = "close",
    cvd_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    method: str = "dfa",  # 'rs' or 'dfa'
    rolling_window: int = 50,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 Hurst 特征
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        cvd_col: CVD column name (optional)
        volume_col: Volume column name (optional)
        method: Hurst calculation method ('rs' or 'dfa')
        rolling_window: Rolling window for dynamic Hurst
    
    Returns:
        DataFrame with Hurst features added
    """
    df = df.copy()
    
    # 计算价格收益率
    if price_col in df.columns:
        price_returns = df[price_col].pct_change().fillna(0).values
        
        # 全序列 Hurst
        if method == "dfa":
            hurst_full = compute_hurst_dfa(price_returns)
        else:
            hurst_full = compute_hurst_rs(price_returns)
        
        df["hurst_price_full"] = hurst_full
        
        # 滚动 Hurst（捕捉市场状态切换）
        if len(df) > rolling_window:
            rolling_hurst = []
            for i in range(len(df)):
                if i < rolling_window:
                    rolling_hurst.append(0.5)  # 默认值
                else:
                    window_returns = price_returns[i - rolling_window : i]
                    if method == "dfa":
                        h = compute_hurst_dfa(window_returns)
                    else:
                        h = compute_hurst_rs(window_returns)
                    rolling_hurst.append(h)
            
            df["hurst_price_rolling"] = rolling_hurst
        else:
            df["hurst_price_rolling"] = 0.5
    
    # CVD Hurst
    if cvd_col and cvd_col in df.columns:
        cvd_values = df[cvd_col].values
        cvd_diff = np.diff(cvd_values, prepend=cvd_values[0])
        
        if method == "dfa":
            hurst_cvd = compute_hurst_dfa(cvd_diff)
        else:
            hurst_cvd = compute_hurst_rs(cvd_diff)
        
        df["hurst_cvd"] = hurst_cvd
    
    # Volume Hurst（对数变换后）
    if volume_col and volume_col in df.columns:
        volume = df[volume_col].values
        volume_log = np.log(volume + 1e-10)
        volume_diff = np.diff(volume_log, prepend=volume_log[0])
        
        if method == "dfa":
            hurst_volume = compute_hurst_dfa(volume_diff)
        else:
            hurst_volume = compute_hurst_rs(volume_diff)
        
        df["hurst_volume"] = hurst_volume
    
    return df

