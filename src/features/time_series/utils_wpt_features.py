"""
小波包变换（WPT）特征工程

核心功能：
1. 多尺度分解（price、volume、CVD）
2. 趋势提取（最低频子带）
3. 残差计算（去趋势后的波动）
4. 能量和熵计算

重要说明：
- 所有 WPT 特征具有约 window//2 的相位滞后，适用于中低频策略（日线/4H）
- 滚动窗口 + shift(1) 有效避免了未来函数问题
- 推荐小波："db4"（平滑）、"haar"（快速，适合突变）、"sym4"（对称）

性能说明：
- 对于4H/日线等低频K线：WPT计算速度完全足够，单次WPT约0.8-5ms
  → 每根K线都计算WPT特征完全可行（update_step=1，默认值）
- 对于1min/tick等高频K线：可设置update_step=5-10以减少计算量
  → 每分钟计算一次WPT可能过于频繁，每5-10分钟更新一次更合理

实测参考（普通笔记本）：
- len=100, level=4, pywt.WaveletPacket → 约 0.8ms / 次
- 3个信号（price/volume/CVD）× 0.8ms = 2.4ms / 根4H K线
- 即使不做任何优化，也远快于4小时！
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pywt

from src.features.registry import register_feature

# 性能优化：缓存小波对象（避免重复初始化）
# 注意：对于4H级别，此优化非必需，但可以略微提升性能
_WAVELET_CACHE: Dict[str, pywt.Wavelet] = {}


def _get_wavelet(name: str) -> pywt.Wavelet:
    """
    获取小波对象（带缓存）
    
    Args:
        name: 小波函数名称
    
    Returns:
        Wavelet对象
    """
    if name not in _WAVELET_CACHE:
        _WAVELET_CACHE[name] = pywt.Wavelet(name)
    return _WAVELET_CACHE[name]


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
    # 动态限制 level，防止超过最大允许分解层数
    # 先验证小波函数是否有效
    try:
        # 使用缓存的小波对象（性能优化）
        wavelet_obj = _get_wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(signal), wavelet_obj)
        actual_level = min(level, max_level) if max_level > 0 else 1
    except (ValueError, RuntimeError, TypeError):
        # 无效小波函数，返回零趋势和原始信号作为波动
        return {
            "trend": np.zeros_like(signal),
            "fluctuation": signal,
            "subbands": {},
            "energy": {},
            "entropy": {},
            "wp": None,
        }
    
    if actual_level < 1:
        # 无法分解，返回零趋势和原始信号作为波动
        return {
            "trend": np.zeros_like(signal),
            "fluctuation": signal,
            "subbands": {},
            "energy": {},
            "entropy": {},
            "wp": None,
        }
    
    try:
        wp = pywt.WaveletPacket(
            data=signal, wavelet=wavelet, mode=mode, maxlevel=actual_level
        )
        
        # 提取最低频子带（趋势）
        trend_path = "a" * actual_level
        trend_node = next(
            (node for node in wp.get_level(actual_level, "natural") if node.path == trend_path),
            None,
        )
        trend = trend_node.data if trend_node else np.zeros_like(signal)
        
        # 重构趋势
        wp_trend = pywt.WaveletPacket(
            data=None, wavelet=wavelet, mode=mode, maxlevel=actual_level
        )
        for node in wp.get_level(actual_level, "natural"):
            if node.path == trend_path:
                wp_trend[node.path] = node.data
            else:
                wp_trend[node.path] = np.zeros_like(node.data)
        
        trend_recon = wp_trend.reconstruct()
        
        # 修复：强制对齐长度（WPT 重建后长度可能不一致）
        if len(trend_recon) != len(signal):
            if len(trend_recon) > len(signal):
                # 截断到原始长度（最安全）
                trend_recon = trend_recon[:len(signal)]
            else:
                # 如果重建后长度更短，使用原始信号（更安全）
                trend_recon = signal
                trend = np.zeros_like(signal)
    except (ValueError, RuntimeError, TypeError):
        # 只捕获预期的小波相关异常
        trend_recon = signal
        trend = np.zeros_like(signal)
        wp = None
    
    # 残差（波动）
    fluctuation = signal - trend_recon
    
    # 提取各子带能量和熵
    subbands = {}
    energy = {}
    entropy = {}
    
    if wp is None:
        return {
            "trend": trend,
            "fluctuation": fluctuation,
            "subbands": subbands,
            "energy": energy,
            "entropy": entropy,
            "wp": None,
        }
    
    for node in wp.get_level(actual_level, "natural"):
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


@register_feature("extract_wpt_features", category="wpt")
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
    update_step: int = 1,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 WPT 特征（滚动窗口，无数据泄露）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        cvd_col: CVD column name (optional)
        tbr_col: Take Buy Ratio column name (optional)
        wavelet: Wavelet function (推荐："db4"平滑、"haar"快速、"sym4"对称)
        level: WPT decomposition level
        window: Rolling window size for WPT calculation (default: 100)
        return_reconstructed_price: If True, only return reconstructed price (for multi-scale SR)
        update_step: 每多少根K线更新一次WPT特征（默认1，即每根K线都更新）
                     
                     性能说明：
                     - 对于4H/日线等低频K线：保持默认值1即可，WPT计算速度完全足够
                       （单次WPT约0.8-5ms，远快于4小时/1天的K线频率）
                     - 对于1min/tick等高频K线：可设置为5-10以减少计算量
                       （每分钟计算一次WPT可能过于频繁，每5-10分钟更新一次更合理）
                     
                     使用场景：
                     - 4H/日线策略：update_step=1（推荐）
                     - 1min策略：update_step=5-10（可选优化）
                     - 资源受限环境：根据实际情况调整
        use_log_returns: If True, apply np.diff(np.log(price)) before WPT decomposition.
                        This converts non-stationary price to stationary returns,
                        improving energy distribution across frequency bands.
                        Recommended for energy ratio features.
    
    Returns:
        DataFrame with WPT features added
        
    Note:
        - 所有 WPT 特征具有约 window//2 的相位滞后，适用于中低频策略（日线/4H）
        - 滚动窗口 + shift(1) 有效避免了未来函数问题
        - 当 update_step > 1 时，特征值会广播到后续 update_step 根K线
        - 当 use_log_returns=True 时，能量比特征更有意义（避免趋势能量主导低频带）
    """
    df = df.copy()
    
    # 计算价格（使用典型价格）
    if "high" in df.columns and "low" in df.columns and price_col in df.columns:
        price = (df["high"] + df["low"] + df[price_col]) / 3.0
    else:
        price = df[price_col]
    
    # 初始化特征列（确保所有列都存在，即使某些列可能全为 NaN）
    # 这对于按月计算时的列对齐很重要
    df["wpt_price_trend"] = np.nan
    df["wpt_price_fluctuation"] = np.nan
    df["wpt_price_reconstructed"] = np.nan
    df["wpt_price_energy_low_ratio"] = np.nan
    df["wpt_price_energy_mid_ratio"] = np.nan
    df["wpt_price_energy_high_ratio"] = np.nan
    df["wpt_price_energy_mid_low_ratio"] = np.nan
    
    # 初始化 volume 相关列（即使 volume_col 不存在，也要创建列以确保合并时列对齐）
    df["wpt_volume_trend"] = np.nan
    df["wpt_volume_fluctuation"] = np.nan
    df["wpt_volume_energy_low_ratio"] = np.nan
    
    # 初始化 CVD 相关列（即使 cvd_col 不存在，也要创建列以确保合并时列对齐）
    df["wpt_cvd_trend"] = np.nan
    df["wpt_cvd_fluctuation"] = np.nan
    df["wpt_cvd_energy_low_ratio"] = np.nan
    
    # 初始化 VPER 列
    df["wpt_vper"] = np.nan
    
    # 滚动窗口计算 WPT 特征（只使用历史数据）
    # 性能优化：每 update_step 根K线更新一次，减少计算量
    price_values = price.values
    min_length = max(window, 2 ** level)
    
    for i in range(min_length, len(df), update_step):
        # 使用历史窗口数据 [i-window, i)
        window_data_raw = price_values[i - window : i]
        
        if len(window_data_raw) < 2 ** level:
            continue
        
        # 对价格做预处理：如果 use_log_returns=True，转换为 log returns
        # 这将非平稳价格转换为平稳收益率，使能量分布更均匀
        if use_log_returns:
            # log returns: r_t = log(p_t) - log(p_{t-1})
            window_data = np.diff(np.log(np.maximum(window_data_raw, 1e-10)))
            if len(window_data) < 2 ** level:
                continue
        else:
            window_data = window_data_raw
        
        # 对窗口数据做 WPT
        price_wpt = wpt_decompose(window_data, wavelet=wavelet, level=level)
        
        # 只使用最后一个点的值（当前时刻的特征）
        # 注意：trend 和 fluctuation 是重构后的序列，我们取最后一个值
        # 注意：WPT 重构是全局操作，最后一个点受整个窗口影响，但不代表 t 时刻的真实趋势
        # 更严重：边界效应（symmetric padding）导致末尾点不稳定
        # 建议：此特征有滞后 window//2 周期
        trend_series = price_wpt["trend"]
        fluctuation_series = price_wpt["fluctuation"]
        
        # 计算特征值
        trend_val = trend_series[-1] if len(trend_series) > 0 else np.nan
        fluctuation_val = fluctuation_series[-1] if len(fluctuation_series) > 0 else np.nan
        reconstructed_val = trend_val + fluctuation_val if not (np.isnan(trend_val) or np.isnan(fluctuation_val)) else np.nan
        
        # 计算能量比（使用整个窗口的能量分布）
        # 优化：按路径中 'd' 的数量分类（'d'=detail高频，越多越高频）
        all_paths = list(price_wpt["energy"].keys())
        if all_paths:
            path_freq = {p: p.count('d') for p in all_paths}  # 'd'的数量表示频率
            max_d = max(path_freq.values()) if path_freq else 0
            
            # 按频率分类：low (d <= max_d//3), mid (max_d//3 < d <= 2*max_d//3), high (d > 2*max_d//3)
            low_bands = [p for p, d in path_freq.items() if d <= max_d // 3]
            mid_bands = [p for p, d in path_freq.items() if max_d // 3 < d <= 2 * max_d // 3]
            high_bands = [p for p, d in path_freq.items() if d > 2 * max_d // 3]
            
            energy_low = sum(price_wpt["energy"].get(p, 0.0) for p in low_bands)
            energy_mid = sum(price_wpt["energy"].get(p, 0.0) for p in mid_bands)
            energy_high = sum(price_wpt["energy"].get(p, 0.0) for p in high_bands)
        else:
            energy_low = energy_mid = energy_high = 0.0
        
        total_energy = sum(price_wpt["energy"].values())
        energy_low_ratio = energy_low / total_energy if total_energy > 0 else np.nan
        energy_mid_ratio = energy_mid / total_energy if total_energy > 0 else np.nan
        energy_high_ratio = energy_high / total_energy if total_energy > 0 else np.nan
        energy_mid_low_ratio = energy_mid / energy_low if energy_low > 0 else np.nan
        
        # 广播特征值到后续 update_step 根K线
        end_idx = min(i + update_step, len(df))
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_trend")] = trend_val
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_fluctuation")] = fluctuation_val
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_reconstructed")] = reconstructed_val
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_energy_low_ratio")] = energy_low_ratio
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_energy_mid_ratio")] = energy_mid_ratio
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_energy_high_ratio")] = energy_high_ratio
        df.iloc[i:end_idx, df.columns.get_loc("wpt_price_energy_mid_low_ratio")] = energy_mid_low_ratio
    
    # 如果只需要重构价格，提前返回
    if return_reconstructed_price:
        return pd.DataFrame(
            {"wpt_price_reconstructed": df["wpt_price_reconstructed"]}, 
            index=df.index
        )
    
    # 对成交量做 WPT（滚动窗口）
    # 注意：列已经在上面初始化，这里只需要计算值
    if volume_col in df.columns:
        volume = df[volume_col].values
        
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
    # 注意：列已经在上面初始化，这里只需要计算值
    if cvd_col and cvd_col in df.columns:
        cvd = df[cvd_col].values
        
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
    # 注意：列已经在上面初始化，这里只需要计算值
    if volume_col in df.columns:
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
    # 首先处理重复列名（如果有）
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 获取所有 wpt_ 开头的列，并去重
    wpt_cols = list(set([col for col in df.columns if col.startswith("wpt_")]))
    
    # 逐个处理每列，避免重复列名问题
    for col in wpt_cols:
        if col in df.columns:
            df[col] = df[col].shift(1).fillna(0.0)
    
    return df


@register_feature("extract_wpt_price_features_normalized", category="wpt")
def extract_wpt_price_features_normalized(
    df: pd.DataFrame,
    price_col: str = "close",
    wavelet: str = "db4",
    level: int = 4,
    window: int = 100,
    update_step: int = 1,
    use_log_returns: bool = True,  # Default True for energy ratio features
) -> pd.DataFrame:
    """
    Normalized WPT price features for cross-asset training.

    - Keeps the same output column names as `extract_wpt_features`
      but converts price-unit series to unitless ratios vs close:
        - wpt_price_trend := (trend / close) - 1
        - wpt_price_fluctuation := fluctuation / close
    - Energy ratios are clipped to [0, 1].
    - use_log_returns=True (default) ensures energy is distributed across frequency bands
      rather than concentrating in low-frequency (trend) band.
    """
    out = extract_wpt_features(
        df,
        price_col=price_col,
        volume_col="__unused__",  # unused because df likely doesn't have it; extract_wpt_features guards
        cvd_col=None,
        tbr_col=None,
        wavelet=wavelet,
        level=level,
        window=window,
        return_reconstructed_price=False,
        update_step=update_step,
        use_log_returns=use_log_returns,
    )

    if price_col not in df.columns:
        return out
    close = pd.to_numeric(df[price_col], errors="coerce").astype(float)
    close_safe = close.replace(0.0, np.nan)

    if "wpt_price_trend" in out.columns:
        out["wpt_price_trend"] = (
            pd.to_numeric(out["wpt_price_trend"], errors="coerce").astype(float) / close_safe
        ) - 1.0
    if "wpt_price_fluctuation" in out.columns:
        out["wpt_price_fluctuation"] = (
            pd.to_numeric(out["wpt_price_fluctuation"], errors="coerce").astype(float) / close_safe
        )

    for c in [
        "wpt_price_energy_low_ratio",
        "wpt_price_energy_mid_ratio",
        "wpt_price_energy_high_ratio",
    ]:
        if c in out.columns:
            out[c] = (
                pd.to_numeric(out[c], errors="coerce")
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
            )

    return out


@register_feature("extract_wpt_volatility_features_normalized", category="wpt")
def extract_wpt_volatility_features_normalized(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    cvd_col: Optional[str] = None,
    wavelet: str = "db4",
    level: int = 4,
    window: int = 100,
    update_step: int = 1,
) -> pd.DataFrame:
    """
    Normalized full WPT volatility feature block (cross-asset comparable).

    Converts price-unit WPT series to unitless:
      - wpt_price_trend := (trend/close)-1
      - wpt_price_fluctuation := fluct/close
      - wpt_price_reconstructed := (reconstructed/close)-1

    Keeps energy ratios bounded in [0,1].
    For ratio-like long-tail metrics:
      - wpt_price_energy_mid_low_ratio: robust log scaling (median/IQR)
      - wpt_vper: robust log scaling (median/IQR)
    """
    out = extract_wpt_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        cvd_col=cvd_col,
        wavelet=wavelet,
        level=level,
        window=window,
        return_reconstructed_price=False,
        update_step=update_step,
    )

    if price_col in df.columns:
        close = pd.to_numeric(df[price_col], errors="coerce").astype(float)
        close_safe = close.replace(0.0, np.nan)

        if "wpt_price_trend" in out.columns:
            out["wpt_price_trend"] = (pd.to_numeric(out["wpt_price_trend"], errors="coerce").astype(float) / close_safe) - 1.0
        if "wpt_price_fluctuation" in out.columns:
            out["wpt_price_fluctuation"] = pd.to_numeric(out["wpt_price_fluctuation"], errors="coerce").astype(float) / close_safe
        if "wpt_price_reconstructed" in out.columns:
            out["wpt_price_reconstructed"] = (pd.to_numeric(out["wpt_price_reconstructed"], errors="coerce").astype(float) / close_safe) - 1.0

    # Energy ratios are bounded by construction; clip defensively
    for c in [
        "wpt_price_energy_low_ratio",
        "wpt_price_energy_mid_ratio",
        "wpt_price_energy_high_ratio",
        "wpt_volume_energy_low_ratio",
        "wpt_cvd_energy_low_ratio",
    ]:
        if c in out.columns:
            out[c] = (
                pd.to_numeric(out[c], errors="coerce")
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
            )

    def _log_robust_rolling(s: pd.Series, window: int = 50, min_periods: int = 10) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s = np.log1p(np.abs(s))
        med = s.rolling(window=window, min_periods=min_periods).median()
        q25 = s.rolling(window=window, min_periods=min_periods).quantile(0.25)
        q75 = s.rolling(window=window, min_periods=min_periods).quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        z = (s - med) / (iqr + 1e-8)
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "wpt_price_energy_mid_low_ratio" in out.columns:
        out["wpt_price_energy_mid_low_ratio"] = _log_robust_rolling(out["wpt_price_energy_mid_low_ratio"])
    if "wpt_vper" in out.columns:
        out["wpt_vper"] = _log_robust_rolling(out["wpt_vper"])

    return out


def wpt_reconstruct_subband(
    wp: pywt.WaveletPacket,
    subband_path: str,
    level: int,
    original_length: Optional[int] = None,
) -> np.ndarray:
    """
    重构指定子带
    
    Args:
        wp: WaveletPacket 对象
        subband_path: 子带路径（如 'aaaa', 'aaad'）
        level: 分解层数
        original_length: 原始信号长度（用于对齐，如果提供）
    
    Returns:
        重构后的信号
    """
    try:
        wp_recon = pywt.WaveletPacket(
            data=None, wavelet=wp.wavelet, mode=wp.mode, maxlevel=level
        )
        
        for node in wp.get_level(level, "natural"):
            if node.path == subband_path:
                wp_recon[node.path] = node.data
            else:
                wp_recon[node.path] = np.zeros_like(node.data)
        
        reconstructed = wp_recon.reconstruct()
        
        # 如果提供了原始长度，强制对齐
        if original_length is not None and len(reconstructed) != original_length:
            if len(reconstructed) > original_length:
                reconstructed = reconstructed[:original_length]
            # 如果更短，保持原样（或可以插值，但当前实现保持原样）
        
        return reconstructed
    except (ValueError, RuntimeError, TypeError):
        # 如果重构失败，返回零数组
        if original_length is not None:
            return np.zeros(original_length)
        # 尝试从 wp 获取原始长度
        try:
            original_len = len(wp.data) if hasattr(wp, 'data') and wp.data is not None else 100
            return np.zeros(original_len)
        except:
            return np.zeros(100)  # fallback
