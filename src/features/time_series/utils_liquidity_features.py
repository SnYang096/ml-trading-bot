"""
流动性真空区与 WPT+Volume 能量协同分析特征

核心功能：
1. 流动性真空区识别（Liquidity Void / Gap Detection）
2. WPT + Volume 能量协同分析（VPER、能量下移、真假突破判断）
3. WPT 降噪的 VPVR（Volume Profile Visible Range）

关键认知：
- 流动性真空区 ≠ 历史低成交量区域，而是当前订单簿深度缺失
- 价格快速通过真空区 ≠ 真突破，需结合多尺度能量验证
- WPT 降噪后的 VPVR 能更清晰地识别关键支撑/阻力位
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pywt
from scipy import stats


def build_wpt_denoised_vpvr(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    wavelet: str = "db4",
    level: int = 4,
    vpvr_window: int = 100,
    bins: int = 50,
    drop_high_freq: bool = True,
) -> pd.DataFrame:
    """
    构建 WPT 降噪后的 VPVR（Volume Profile Visible Range）
    
    核心思想：
    1. 使用 WPT 对价格进行降噪，剔除高频噪声
    2. 基于降噪后的价格构建 VPVR，使成交量分布更聚焦于真实交易区间
    3. 识别 PVP（Point of Control）和 LVN（Low Volume Node）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        high_col: High column name
        low_col: Low column name
        wavelet: Wavelet function
        level: WPT decomposition level
        vpvr_window: Rolling window for VPVR calculation
        bins: Number of price bins for VPVR
        drop_high_freq: Whether to drop highest frequency subband
    
    Returns:
        DataFrame with VPVR features added:
        - vpvr_pvp: Point of Control (price with highest volume)
        - vpvr_hvn_count: High Volume Node count
        - vpvr_lvn_count: Low Volume Node count
        - vpvr_lvn_distance: Distance to nearest LVN
        - vpvr_volume_density: Volume density at current price
        - vpvr_price_in_lvn: Whether current price is in LVN (1.0) or not (0.0)
    """
    df = df.copy()
    
    # 使用典型价格 (H+L+C)/3
    if high_col in df.columns and low_col in df.columns:
        typical_price = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    else:
        typical_price = df[price_col]
    
    # 初始化输出列
    df["vpvr_pvp"] = np.nan
    df["vpvr_hvn_count"] = 0.0
    df["vpvr_lvn_count"] = 0.0
    df["vpvr_lvn_distance"] = np.nan
    df["vpvr_volume_density"] = 0.0
    df["vpvr_price_in_lvn"] = 0.0
    
    # 滚动窗口计算 VPVR
    for i in range(vpvr_window, len(df)):
        window_slice = df.iloc[i - vpvr_window : i]
        price_window = typical_price.iloc[i - vpvr_window : i].values
        volume_window = df[volume_col].iloc[i - vpvr_window : i].values
        
        if len(price_window) < 10:  # 数据不足
            continue
        
        # Step 1: WPT 降噪
        try:
            wp = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            
            if drop_high_freq:
                # 清除最高频子带（如 'dddd' for level=4）
                for node in wp.get_level(level, "freq"):
                    if all(c == "d" for c in node.path):  # 全高频路径
                        wp[node.path].data = np.zeros_like(wp[node.path].data)
            
            price_denoised = wp.reconstruct()
        except Exception:
            # WPT 失败时使用原始价格
            price_denoised = price_window
        
        # Step 2: 构建 VPVR
        # 过滤 NaN 值
        valid_mask = ~np.isnan(price_denoised) & ~np.isnan(volume_window) & (volume_window > 0)
        if not np.any(valid_mask):
            continue
        
        price_denoised_valid = price_denoised[valid_mask]
        volume_window_valid = volume_window[valid_mask]
        
        if len(price_denoised_valid) < 10:  # 有效数据不足
            continue
        
        price_min = price_denoised_valid.min()
        price_max = price_denoised_valid.max()
        
        if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
            continue
        
        # 计算每个价格档位的成交量
        hist, edges = np.histogram(
            price_denoised_valid, bins=bins, range=(price_min, price_max), weights=volume_window_valid
        )
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Step 3: 识别 PVP（最高成交量对应的价格）
        if np.sum(hist) > 0:
            pvp_idx = np.argmax(hist)
            df.iloc[i, df.columns.get_loc("vpvr_pvp")] = centers[pvp_idx]
        
        # Step 4: 识别 HVN 和 LVN
        volume_mean = np.mean(hist[hist > 0]) if np.any(hist > 0) else 0
        volume_std = np.std(hist[hist > 0]) if np.any(hist > 0) else 0
        
        if volume_std > 0:
            # HVN: 成交量 > mean + 0.5*std
            hvn_mask = hist > (volume_mean + 0.5 * volume_std)
            df.iloc[i, df.columns.get_loc("vpvr_hvn_count")] = np.sum(hvn_mask)
            
            # LVN: 成交量 < mean - 0.5*std
            lvn_mask = hist < (volume_mean - 0.5 * volume_std)
            df.iloc[i, df.columns.get_loc("vpvr_lvn_count")] = np.sum(lvn_mask)
            
            # Step 5: 计算当前价格到最近 LVN 的距离
            current_price = typical_price.iloc[i]
            lvn_prices = centers[lvn_mask]
            
            if len(lvn_prices) > 0:
                lvn_distances = np.abs(lvn_prices - current_price)
                nearest_lvn_idx = np.argmin(lvn_distances)
                nearest_lvn_price = lvn_prices[nearest_lvn_idx]
                nearest_lvn_distance = lvn_distances[nearest_lvn_idx]
                
                # 归一化距离（相对于价格范围）
                price_range = price_max - price_min
                if price_range > 0:
                    df.iloc[i, df.columns.get_loc("vpvr_lvn_distance")] = (
                        nearest_lvn_distance / price_range
                    )
                
                # 判断当前价格是否在 LVN 中
                bin_width = (price_max - price_min) / bins
                if np.abs(current_price - nearest_lvn_price) < bin_width:
                    df.iloc[i, df.columns.get_loc("vpvr_price_in_lvn")] = 1.0
            
            # Step 6: 计算当前价格的成交量密度
            current_price_bin = np.digitize(current_price, edges) - 1
            current_price_bin = np.clip(current_price_bin, 0, len(hist) - 1)
            
            if current_price_bin < len(hist):
                current_volume = hist[current_price_bin]
                max_volume = np.max(hist) if len(hist) > 0 else 1.0
                df.iloc[i, df.columns.get_loc("vpvr_volume_density")] = (
                    current_volume / max_volume if max_volume > 0 else 0.0
                )
    
    # 前向填充缺失值
    df["vpvr_pvp"] = df["vpvr_pvp"].ffill()
    df["vpvr_lvn_distance"] = df["vpvr_lvn_distance"].fillna(0.0)
    
    return df


def compute_liquidity_void_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    lookback_window: int = 20,
    speed_threshold_multiplier: float = 2.0,
    volume_quantile: float = 0.2,
) -> pd.DataFrame:
    """
    识别流动性真空区（Liquidity Void / Gap）
    
    核心逻辑：
    1. 价格快速穿越某区域（速度 > 阈值）
    2. 该区域成交量显著低于前后区间（VPVR 低量节点）
    3. 突破后 1~3 根 K 线内迅速回撤 >50%
    
    注意：流动性真空区 ≠ 历史低成交量区域
    - 历史低成交量：过去交易少（可能因无人关注）
    - 流动性真空：当前挂单少（即使过去交易多，也可能因订单撤走而变真空）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        high_col: High column name
        low_col: Low column name
        atr_col: ATR column name
        lookback_window: Window for calculating price speed and volume reference
        speed_threshold_multiplier: Multiplier for price speed threshold
        volume_quantile: Quantile threshold for low volume detection
    
    Returns:
        DataFrame with liquidity void features:
        - liquidity_void_detected: 1.0 if void detected, 0.0 otherwise
        - liquidity_void_speed: Price speed (normalized by ATR)
        - liquidity_void_volume_ratio: Volume ratio (current vs. reference)
        - liquidity_void_retracement: Retracement within 3 bars (if applicable)
        - liquidity_void_false_breakout_risk: Risk score (0-1) for false breakout
    """
    df = df.copy()
    
    # 初始化输出列
    df["liquidity_void_detected"] = 0.0
    df["liquidity_void_speed"] = 0.0
    df["liquidity_void_volume_ratio"] = 1.0
    df["liquidity_void_retracement"] = 0.0
    df["liquidity_void_false_breakout_risk"] = 0.0
    
    if atr_col not in df.columns:
        # 如果没有 ATR，使用价格变化的标准差作为替代
        df[atr_col] = df[price_col].rolling(window=14).std()
    
    # 计算价格速度（归一化）
    price_change = df[price_col].pct_change()
    price_speed = price_change.rolling(window=3).mean()  # 3-bar average speed
    price_speed_normalized = price_speed / (df[atr_col] / df[price_col] + 1e-8)
    
    # 计算成交量比率
    volume_ma = df[volume_col].rolling(window=lookback_window).mean()
    volume_ratio = df[volume_col] / (volume_ma + 1e-8)
    
    # 计算成交量分位数（用于识别低量区）
    volume_quantile_value = df[volume_col].rolling(window=lookback_window).quantile(
        volume_quantile
    )
    is_low_volume = df[volume_col] < volume_quantile_value
    
    # 计算价格速度阈值
    speed_threshold = (
        price_speed_normalized.rolling(window=lookback_window).mean() * speed_threshold_multiplier
    )
    
    # 检测流动性真空区
    for i in range(lookback_window, len(df)):
        # 条件1: 价格速度异常高
        speed_high = price_speed_normalized.iloc[i] > speed_threshold.iloc[i]
        
        # 条件2: 成交量低（相对于历史）
        volume_low = is_low_volume.iloc[i] or volume_ratio.iloc[i] < 0.8
        
        if speed_high and volume_low:
            df.iloc[i, df.columns.get_loc("liquidity_void_detected")] = 1.0
            df.iloc[i, df.columns.get_loc("liquidity_void_speed")] = price_speed_normalized.iloc[
                i
            ]
            df.iloc[i, df.columns.get_loc("liquidity_void_volume_ratio")] = volume_ratio.iloc[i]
            
            # 条件3: 检查后续回撤（如果有足够数据）
            if i + 3 < len(df):
                current_price = df[price_col].iloc[i]
                future_prices = df[price_col].iloc[i + 1 : i + 4]
                
                if len(future_prices) > 0:
                    max_future_price = future_prices.max()
                    min_future_price = future_prices.min()
                    
                    # 计算回撤（相对于当前价格）
                    if current_price > 0:
                        retracement_up = (max_future_price - current_price) / current_price
                        retracement_down = (current_price - min_future_price) / current_price
                        retracement = max(retracement_up, retracement_down)
                        
                        df.iloc[i, df.columns.get_loc("liquidity_void_retracement")] = (
                            retracement
                        )
                        
                        # 如果回撤 > 50%，增加假突破风险
                        if retracement > 0.5:
                            df.iloc[i, df.columns.get_loc("liquidity_void_false_breakout_risk")] = (
                                0.8
                            )
                        elif retracement > 0.3:
                            df.iloc[i, df.columns.get_loc("liquidity_void_false_breakout_risk")] = (
                                0.5
                            )
                        else:
                            df.iloc[i, df.columns.get_loc("liquidity_void_false_breakout_risk")] = (
                                0.2
                            )
    
    return df


def compute_wpt_volume_energy_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    wavelet: str = "db4",
    level: int = 4,
    lookback_window: int = 20,
) -> pd.DataFrame:
    """
    计算 WPT + Volume 能量协同分析特征
    
    核心指标：
    1. VPER (Volume-Price Energy Ratio): 量价能量比
    2. 能量下移（Energy Cascade）: 高频能量向中低频转移
    3. 多尺度一致性验证: 至少两个中低频子带同时出现能量上升
    4. 真假突破评分: 基于多尺度能量和 VPER 的综合评分
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        wavelet: Wavelet function
        level: WPT decomposition level
        lookback_window: Window for calculating reference values
    
    Returns:
        DataFrame with WPT+Volume energy features:
        - wpt_vper_low: VPER for low-frequency subband
        - wpt_vper_mid: VPER for mid-frequency subbands
        - wpt_vper_high: VPER for high-frequency subband
        - wpt_energy_cascade: Energy cascade indicator (high -> mid -> low)
        - wpt_multi_scale_consistency: Multi-scale consistency score (0-1)
        - wpt_breakout_confidence: Breakout confidence score (0-1)
        - wpt_false_breakout_risk: False breakout risk score (0-1)
    """
    df = df.copy()
    
    # 初始化输出列
    df["wpt_vper_low"] = 0.0
    df["wpt_vper_mid"] = 0.0
    df["wpt_vper_high"] = 0.0
    df["wpt_energy_cascade"] = 0.0
    df["wpt_multi_scale_consistency"] = 0.0
    df["wpt_breakout_confidence"] = 0.0
    df["wpt_false_breakout_risk"] = 0.0
    
    # 提取价格和成交量
    price = df[price_col].values
    volume = df[volume_col].values
    
    # 滚动窗口计算
    for i in range(lookback_window, len(df)):
        window_start = max(0, i - lookback_window)
        price_window = price[window_start : i + 1]
        volume_window = volume[window_start : i + 1]
        
        if len(price_window) < 10:
            continue
        
        try:
            # Step 1: 对价格和成交量分别做 WPT 分解
            wp_price = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            wp_volume = pywt.WaveletPacket(
                data=volume_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            
            # Step 2: 提取各子带能量
            price_energy = {}
            volume_energy = {}
            
            for node in wp_price.get_level(level, "natural"):
                price_energy[node.path] = np.sum(node.data ** 2)
            
            for node in wp_volume.get_level(level, "natural"):
                volume_energy[node.path] = np.sum(node.data ** 2)
            
            # Step 3: 分类子带（低频、中频、高频）
            low_freq_paths = ["a" * level]  # 'aaaa' for level=4
            mid_freq_paths = [
                k for k in price_energy.keys() if k.startswith("aa") and k != "a" * level
            ]
            high_freq_paths = [k for k in price_energy.keys() if not k.startswith("aa")]
            
            # Step 4: 计算各频带的能量和 VPER
            price_energy_low = sum(price_energy.get(p, 0) for p in low_freq_paths)
            price_energy_mid = sum(price_energy.get(p, 0) for p in mid_freq_paths)
            price_energy_high = sum(price_energy.get(p, 0) for p in high_freq_paths)
            
            volume_energy_low = sum(volume_energy.get(p, 0) for p in low_freq_paths)
            volume_energy_mid = sum(volume_energy.get(p, 0) for p in mid_freq_paths)
            volume_energy_high = sum(volume_energy.get(p, 0) for p in high_freq_paths)
            
            # VPER = Volume Energy / Price Energy
            eps = 1e-8
            vper_low = volume_energy_low / (price_energy_low + eps)
            vper_mid = volume_energy_mid / (price_energy_mid + eps)
            vper_high = volume_energy_high / (price_energy_high + eps)
            
            df.iloc[i, df.columns.get_loc("wpt_vper_low")] = vper_low
            df.iloc[i, df.columns.get_loc("wpt_vper_mid")] = vper_mid
            df.iloc[i, df.columns.get_loc("wpt_vper_high")] = vper_high
            
            # Step 5: 计算能量下移（Energy Cascade）
            # 理想情况：高频能量下降，中低频能量上升
            total_price_energy = price_energy_low + price_energy_mid + price_energy_high
            if total_price_energy > 0:
                energy_ratio_low = price_energy_low / total_price_energy
                energy_ratio_mid = price_energy_mid / total_price_energy
                energy_ratio_high = price_energy_high / total_price_energy
                
                # 能量下移指标：中低频能量占比增加
                energy_cascade = (energy_ratio_low + energy_ratio_mid) - energy_ratio_high
                df.iloc[i, df.columns.get_loc("wpt_energy_cascade")] = energy_cascade
            
            # Step 6: 多尺度一致性验证
            # 检查至少两个中低频子带是否同时出现能量上升
            if i > 0:
                prev_price_window = price[max(0, i - lookback_window - 1) : i]
                if len(prev_price_window) >= 10:
                    try:
                        wp_price_prev = pywt.WaveletPacket(
                            data=prev_price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
                        )
                        
                        prev_price_energy = {}
                        for node in wp_price_prev.get_level(level, "natural"):
                            prev_price_energy[node.path] = np.sum(node.data ** 2)
                        
                        # 计算中低频子带的能量变化
                        mid_freq_energy_changes = []
                        for path in mid_freq_paths:
                            curr_energy = price_energy.get(path, 0)
                            prev_energy = prev_price_energy.get(path, 0)
                            if prev_energy > 0:
                                energy_change = (curr_energy - prev_energy) / prev_energy
                                mid_freq_energy_changes.append(energy_change)
                        
                        # 如果至少两个中低频子带能量上升，则一致性高
                        positive_changes = sum(1 for chg in mid_freq_energy_changes if chg > 0.1)
                        consistency_score = min(positive_changes / max(len(mid_freq_paths), 1), 1.0)
                        df.iloc[i, df.columns.get_loc("wpt_multi_scale_consistency")] = (
                            consistency_score
                        )
                    except Exception:
                        pass
            
            # Step 7: 真假突破评分
            # 真突破条件：
            # 1. 中低频能量上升
            # 2. VPER 中频处于高位
            # 3. 多尺度一致性高
            
            # 计算 VPER 中频的历史分位数
            if i >= lookback_window * 2:
                vper_mid_history = df["wpt_vper_mid"].iloc[i - lookback_window : i]
                if len(vper_mid_history) > 0 and vper_mid_history.notna().any():
                    vper_mid_quantile = stats.percentileofscore(
                        vper_mid_history.dropna(), vper_mid
                    ) / 100.0
                else:
                    vper_mid_quantile = 0.5
            else:
                vper_mid_quantile = 0.5
            
            # 突破置信度 = (能量下移 + VPER分位数 + 多尺度一致性) / 3
            consistency = df.iloc[i, df.columns.get_loc("wpt_multi_scale_consistency")]
            breakout_confidence = (
                (energy_cascade + 1) / 2 * 0.4 + vper_mid_quantile * 0.3 + consistency * 0.3
            )
            df.iloc[i, df.columns.get_loc("wpt_breakout_confidence")] = breakout_confidence
            
            # 假突破风险 = 1 - 突破置信度，但如果高频能量突增而中频无增长，风险更高
            if price_energy_high > price_energy_mid * 2 and vper_mid < 0.5:
                false_breakout_risk = min(breakout_confidence + 0.3, 1.0)
            else:
                false_breakout_risk = 1.0 - breakout_confidence
            
            df.iloc[i, df.columns.get_loc("wpt_false_breakout_risk")] = false_breakout_risk
            
        except Exception as e:
            # WPT 计算失败时跳过
            continue
    
    # 前向填充缺失值
    for col in [
        "wpt_vper_low",
        "wpt_vper_mid",
        "wpt_vper_high",
        "wpt_energy_cascade",
        "wpt_multi_scale_consistency",
        "wpt_breakout_confidence",
        "wpt_false_breakout_risk",
    ]:
        df[col] = df[col].ffill().fillna(0.0)
    
    return df


def extract_liquidity_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    wavelet: str = "db4",
    level: int = 4,
    feature_type: str = "all",
) -> pd.DataFrame:
    """
    提取完整的流动性相关特征（整合所有子功能）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        high_col: High column name
        low_col: Low column name
        atr_col: ATR column name
        wavelet: Wavelet function
        level: WPT decomposition level
        feature_type: Feature type to extract ("vpvr", "void", "energy", "all")
    
    Returns:
        DataFrame with all liquidity features added
    """
    df = df.copy()
    
    # 根据 feature_type 选择要提取的特征
    if feature_type in ["vpvr", "all"]:
        # 1. WPT 降噪的 VPVR
        df = build_wpt_denoised_vpvr(
            df,
            price_col=price_col,
            volume_col=volume_col,
            high_col=high_col,
            low_col=low_col,
            wavelet=wavelet,
            level=level,
        )
    
    if feature_type in ["void", "all"]:
        # 2. 流动性真空区识别
        df = compute_liquidity_void_features(
            df,
            price_col=price_col,
            volume_col=volume_col,
            high_col=high_col,
            low_col=low_col,
            atr_col=atr_col,
        )
    
    if feature_type in ["energy", "all"]:
        # 3. WPT + Volume 能量协同分析
        df = compute_wpt_volume_energy_features(
            df,
            price_col=price_col,
            volume_col=volume_col,
            wavelet=wavelet,
            level=level,
        )
    
    return df

