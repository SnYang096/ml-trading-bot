"""
小波包变换（WPT）特征工程

核心功能：
1. 多尺度分解（price、volume、CVD）
2. 趋势提取（最低频子带）
3. 残差计算（去趋势后的波动）
4. 能量和熵计算
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pywt


def wpt_decompose(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    mode: str = "symmetric",
) -> Dict[str, np.ndarray]:
    """
    小波包变换分解
    
    Args:
        signal: 输入信号
        wavelet: 小波基函数
        level: 分解层数
        mode: 边界处理模式
    
    Returns:
        Dict with keys: 'trend', 'fluctuation', 'subbands', 'energy', 'entropy'
    """
    wp = pywt.WaveletPacket(
        data=signal, wavelet=wavelet, mode=mode, maxlevel=level
    )
    
    # 提取最低频子带（趋势）
    trend_node = wp.get_node("a" * level)  # 'aaaa' for level=4
    trend = trend_node.data if trend_node else np.zeros_like(signal)
    
    # 重构趋势
    wp_trend = pywt.WaveletPacket(
        data=None, wavelet=wavelet, mode=mode, maxlevel=level
    )
    for node in wp.get_level(level, "natural"):
        if node.path == "a" * level:
            wp_trend[node.path] = node.data
        else:
            wp_trend[node.path] = np.zeros_like(node.data)
    
    trend_recon = wp_trend.reconstruct()
    
    # 残差（波动）
    fluctuation = signal - trend_recon
    
    # 提取各子带能量和熵
    subbands = {}
    energy = {}
    entropy = {}
    
    for node in wp.get_level(level, "natural"):
        subband_data = node.data
        subbands[node.path] = subband_data
        
        # 能量
        energy[node.path] = np.sum(subband_data ** 2)
        
        # 熵（归一化后计算）
        if np.sum(np.abs(subband_data)) > 0:
            prob = np.abs(subband_data) / np.sum(np.abs(subband_data))
            prob = prob[prob > 0]  # 避免 log(0)
            entropy[node.path] = -np.sum(prob * np.log(prob + 1e-10))
        else:
            entropy[node.path] = 0.0
    
    return {
        "trend": trend_recon,
        "fluctuation": fluctuation,
        "subbands": subbands,
        "energy": energy,
        "entropy": entropy,
        "wp": wp,  # 保留原始对象用于重构
    }


def extract_wpt_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    cvd_col: Optional[str] = None,
    tbr_col: Optional[str] = None,
    wavelet: str = "db4",
    level: int = 4,
    window: int = 100,
    return_reconstructed_price: bool = False,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 WPT 特征（滚动窗口，无数据泄露）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        cvd_col: CVD column name (optional)
        tbr_col: Take Buy Ratio column name (optional)
        wavelet: Wavelet function
        level: WPT decomposition level
        window: Rolling window size for WPT calculation (default: 100)
        return_reconstructed_price: If True, only return reconstructed price (for multi-scale SR)
    
    Returns:
        DataFrame with WPT features added
    """
    df = df.copy()
    
    # 计算价格（使用典型价格）
    if "high" in df.columns and "low" in df.columns and price_col in df.columns:
        price = (df["high"] + df["low"] + df[price_col]) / 3.0
    else:
        price = df[price_col]
    
    # 初始化特征列
    df["wpt_price_trend"] = np.nan
    df["wpt_price_fluctuation"] = np.nan
    df["wpt_price_reconstructed"] = np.nan
    df["wpt_price_energy_low_ratio"] = np.nan
    df["wpt_price_energy_mid_ratio"] = np.nan
    df["wpt_price_energy_high_ratio"] = np.nan
    df["wpt_price_energy_mid_low_ratio"] = np.nan
    
    # 滚动窗口计算 WPT 特征（只使用历史数据）
    price_values = price.values
    min_length = max(window, 2 ** level)
    
    for i in range(min_length, len(df)):
        # 使用历史窗口数据 [i-window, i)
        window_data = price_values[i - window : i]
        
        if len(window_data) < 2 ** level:
            continue
        
        # 对窗口数据做 WPT
        price_wpt = wpt_decompose(window_data, wavelet=wavelet, level=level)
        
        # 只使用最后一个点的值（当前时刻的特征）
        # 注意：trend 和 fluctuation 是重构后的序列，我们取最后一个值
        trend_series = price_wpt["trend"]
        fluctuation_series = price_wpt["fluctuation"]
        
        if len(trend_series) > 0 and len(fluctuation_series) > 0:
            # 使用最后一个值作为当前时刻的特征
            df.iloc[i, df.columns.get_loc("wpt_price_trend")] = trend_series[-1]
            df.iloc[i, df.columns.get_loc("wpt_price_fluctuation")] = fluctuation_series[-1]
            df.iloc[i, df.columns.get_loc("wpt_price_reconstructed")] = (
                trend_series[-1] + fluctuation_series[-1]
            )
        
        # 计算能量比（使用整个窗口的能量分布）
        energy_low = price_wpt["energy"].get("a" * level, 0.0)
        energy_mid = sum(
            v for k, v in price_wpt["energy"].items() 
            if k.startswith("aa") and k != "a" * level
        )
        energy_high = sum(
            v for k, v in price_wpt["energy"].items() 
            if not k.startswith("aa")
        )
        
        total_energy = sum(price_wpt["energy"].values())
        if total_energy > 0:
            df.iloc[i, df.columns.get_loc("wpt_price_energy_low_ratio")] = (
                energy_low / total_energy
            )
            df.iloc[i, df.columns.get_loc("wpt_price_energy_mid_ratio")] = (
                energy_mid / total_energy
            )
            df.iloc[i, df.columns.get_loc("wpt_price_energy_high_ratio")] = (
                energy_high / total_energy
            )
            if energy_low > 0:
                df.iloc[i, df.columns.get_loc("wpt_price_energy_mid_low_ratio")] = (
                    energy_mid / energy_low
                )
    
    # 如果只需要重构价格，提前返回
    if return_reconstructed_price:
        return pd.DataFrame(
            {"wpt_price_reconstructed": df["wpt_price_reconstructed"]}, 
            index=df.index
        )
    
    # 对成交量做 WPT（滚动窗口）
    if volume_col in df.columns:
        volume = df[volume_col].values
        df["wpt_volume_trend"] = np.nan
        df["wpt_volume_fluctuation"] = np.nan
        df["wpt_volume_energy_low_ratio"] = np.nan
        
        for i in range(min_length, len(df)):
            window_data = volume[i - window : i]
            if len(window_data) < 2 ** level:
                continue
            
            volume_wpt = wpt_decompose(window_data, wavelet=wavelet, level=level)
            trend_series = volume_wpt["trend"]
            fluctuation_series = volume_wpt["fluctuation"]
            
            if len(trend_series) > 0 and len(fluctuation_series) > 0:
                df.iloc[i, df.columns.get_loc("wpt_volume_trend")] = trend_series[-1]
                df.iloc[i, df.columns.get_loc("wpt_volume_fluctuation")] = fluctuation_series[-1]
            
            volume_total_energy = sum(volume_wpt["energy"].values())
            if volume_total_energy > 0:
                df.iloc[i, df.columns.get_loc("wpt_volume_energy_low_ratio")] = (
                    volume_wpt["energy"].get("a" * level, 0.0) / volume_total_energy
                )
    
    # 对 CVD 做 WPT（滚动窗口）
    if cvd_col and cvd_col in df.columns:
        cvd = df[cvd_col].values
        df["wpt_cvd_trend"] = np.nan
        df["wpt_cvd_fluctuation"] = np.nan
        df["wpt_cvd_energy_low_ratio"] = np.nan
        
        for i in range(min_length, len(df)):
            window_data = cvd[i - window : i]
            if len(window_data) < 2 ** level:
                continue
            
            cvd_wpt = wpt_decompose(window_data, wavelet=wavelet, level=level)
            trend_series = cvd_wpt["trend"]
            fluctuation_series = cvd_wpt["fluctuation"]
            
            if len(trend_series) > 0 and len(fluctuation_series) > 0:
                df.iloc[i, df.columns.get_loc("wpt_cvd_trend")] = trend_series[-1]
                df.iloc[i, df.columns.get_loc("wpt_cvd_fluctuation")] = fluctuation_series[-1]
            
            cvd_total_energy = sum(cvd_wpt["energy"].values())
            if cvd_total_energy > 0:
                df.iloc[i, df.columns.get_loc("wpt_cvd_energy_low_ratio")] = (
                    cvd_wpt["energy"].get("a" * level, 0.0) / cvd_total_energy
                )
    
    # VPER (Volume-Price Energy Ratio) - 需要滚动窗口计算
    if volume_col in df.columns:
        df["wpt_vper"] = np.nan
        for i in range(min_length, len(df)):
            price_window = price_values[i - window : i]
            volume_window = volume[i - window : i]
            
            if len(price_window) < 2 ** level or len(volume_window) < 2 ** level:
                continue
            
            price_wpt = wpt_decompose(price_window, wavelet=wavelet, level=level)
            volume_wpt = wpt_decompose(volume_window, wavelet=wavelet, level=level)
            
            price_total_energy = sum(price_wpt["energy"].values())
            volume_total_energy = sum(volume_wpt["energy"].values())
            
            if price_total_energy > 0 and volume_total_energy > 0:
                df.iloc[i, df.columns.get_loc("wpt_vper")] = (
                    volume_total_energy / price_total_energy
                )
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    wpt_cols = [col for col in df.columns if col.startswith("wpt_")]
    for col in wpt_cols:
        df[col] = df[col].shift(1)
    
    # Fill NaN with 0 (after shift, NaN means insufficient history)
    df[wpt_cols] = df[wpt_cols].fillna(0.0)
    
    return df


def wpt_reconstruct_subband(
    wp: pywt.WaveletPacket,
    subband_path: str,
    level: int,
) -> np.ndarray:
    """
    重构指定子带
    
    Args:
        wp: WaveletPacket 对象
        subband_path: 子带路径（如 'aaaa', 'aaad'）
        level: 分解层数
    
    Returns:
        重构后的信号
    """
    wp_recon = pywt.WaveletPacket(
        data=None, wavelet=wp.wavelet, mode=wp.mode, maxlevel=level
    )
    
    for node in wp.get_level(level, "natural"):
        if node.path == subband_path:
            wp_recon[node.path] = node.data
        else:
            wp_recon[node.path] = np.zeros_like(node.data)
    
    return wp_recon.reconstruct()

