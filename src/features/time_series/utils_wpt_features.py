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
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 WPT 特征
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        cvd_col: CVD column name (optional)
        tbr_col: Take Buy Ratio column name (optional)
        wavelet: Wavelet function
        level: WPT decomposition level
    
    Returns:
        DataFrame with WPT features added
    """
    df = df.copy()
    
    # 计算价格（使用典型价格）
    if "high" in df.columns and "low" in df.columns and price_col in df.columns:
        price = (df["high"] + df["low"] + df[price_col]) / 3.0
    else:
        price = df[price_col]
    
    # 对价格做 WPT
    price_wpt = wpt_decompose(price.values, wavelet=wavelet, level=level)
    df["wpt_price_trend"] = price_wpt["trend"]
    df["wpt_price_fluctuation"] = price_wpt["fluctuation"]
    
    # 价格子带能量比
    energy_low = price_wpt["energy"].get("a" * level, 0.0)
    energy_mid = sum(
        v for k, v in price_wpt["energy"].items() if k.startswith("aa") and k != "a" * level
    )
    energy_high = sum(
        v for k, v in price_wpt["energy"].items() if not k.startswith("aa")
    )
    
    total_energy = sum(price_wpt["energy"].values())
    if total_energy > 0:
        df["wpt_price_energy_low_ratio"] = energy_low / total_energy
        df["wpt_price_energy_mid_ratio"] = energy_mid / total_energy
        df["wpt_price_energy_high_ratio"] = energy_high / total_energy
        df["wpt_price_energy_mid_low_ratio"] = (
            energy_mid / energy_low if energy_low > 0 else 0.0
        )
    
    # 对成交量做 WPT
    if volume_col in df.columns:
        volume = df[volume_col].values
        volume_wpt = wpt_decompose(volume, wavelet=wavelet, level=level)
        df["wpt_volume_trend"] = volume_wpt["trend"]
        df["wpt_volume_fluctuation"] = volume_wpt["fluctuation"]
        
        # 成交量能量
        volume_total_energy = sum(volume_wpt["energy"].values())
        if volume_total_energy > 0:
            df["wpt_volume_energy_low_ratio"] = (
                volume_wpt["energy"].get("a" * level, 0.0) / volume_total_energy
            )
    
    # 对 CVD 做 WPT
    if cvd_col and cvd_col in df.columns:
        cvd = df[cvd_col].values
        cvd_wpt = wpt_decompose(cvd, wavelet=wavelet, level=level)
        df["wpt_cvd_trend"] = cvd_wpt["trend"]
        df["wpt_cvd_fluctuation"] = cvd_wpt["fluctuation"]
        
        # CVD 能量
        cvd_total_energy = sum(cvd_wpt["energy"].values())
        if cvd_total_energy > 0:
            df["wpt_cvd_energy_low_ratio"] = (
                cvd_wpt["energy"].get("a" * level, 0.0) / cvd_total_energy
            )
    
    # VPER (Volume-Price Energy Ratio)
    if volume_col in df.columns and total_energy > 0:
        volume_total_energy = sum(volume_wpt["energy"].values())
        if volume_total_energy > 0:
            df["wpt_vper"] = volume_total_energy / total_energy
    
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

