"""
Hilbert 变换特征工程（改进版 + 高级功能）

核心原则：
1. 只提取 Hilbert 包络（envelope）作为特征，放弃瞬时相位和频率
2. 使用滚动窗口计算，避免未来信息泄露
3. EMA 平滑包络序列，提升稳健性
4. 提取实用衍生特征（包络比值、斜率等）

高级功能：
- 自适应窗口：基于局部周期估计，动态匹配市场波动周期
- 分位数标准化：跨品种可比，支持多品种统一模型
- 成交量融合：结合价格、资金流、成交量，识别背离信号

原因：
- 金融波动信号非窄带，相位/频率噪声大、不可靠
- 包络可作为瞬时波动强度的有效代理
- EMA 是因果、高效、金融友好的平滑器
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from scipy.signal import hilbert

from src.features.registry import register_feature


def compute_hilbert_envelope(
    signal: np.ndarray,
) -> np.ndarray:
    """
    计算信号的 Hilbert 包络
    
    Args:
        signal: 输入信号（应已去趋势，如 WPT 波动分量）
    
    Returns:
        包络序列（envelope = |analytic_signal|）
    """
    # Hilbert 变换
    analytic = hilbert(signal)
    
    # 包络
    envelope = np.abs(analytic)
    
    return envelope


def estimate_local_period(
    signal: np.ndarray,
    min_period: int = 8,
    max_period: int = 128,
) -> int:
    """
    基于零穿越率（Zero-Crossing Rate）估计局部周期
    
    对去趋势信号，相邻零点间距 ≈ 半周期 → 可估计局部周期
    
    Args:
        signal: 输入信号（应已去趋势）
        min_period: 最小周期长度
        max_period: 最大周期长度
    
    Returns:
        估计的周期长度（整数）
    """
    # 内部 NaN 过滤：防止外部传入含 NaN 的数组导致符号判断错误
    signal = signal[~np.isnan(signal)]
    
    if len(signal) < 20:
        return min(max_period, len(signal))
    
    # 符号序列（0 视为正）
    signs = np.sign(signal)
    signs[signs == 0] = 1
    
    # 零穿越位置（符号变化处）
    # 注意：仅使用 signal[:-1] 判断穿越，避免用最后一个点
    crossings = np.where(np.diff(signs) != 0)[0] + 1  # +1 因为 diff 缩短了
    
    if len(crossings) < 2:
        return min_period
    
    # 计算最近几个半周期长度（取后 M 个）
    half_periods = np.diff(crossings)[-5:]  # 最近5个半周期
    if len(half_periods) == 0:
        return min_period
    
    avg_half_period = np.median(half_periods)
    estimated_period = int(2 * avg_half_period)
    
    # 限制范围
    estimated_period = np.clip(estimated_period, min_period, max_period)
    return int(estimated_period)


def rolling_quantile_normalize(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    因果滚动分位数标准化：将每个值映射为其在历史 window 中的分位数
    
    优势：
    - 消除量纲差异，支持跨品种可比
    - 自适应市场状态（牛市/熊市波动率不同）
    - 保留相对强弱信息
    
    Args:
        series: 输入序列
        window: 滚动窗口大小（建议：252=日线年化，7*24*60=分钟线周化）
    
    Returns:
        标准化后的序列（值域 [0, 1]）
    """
    def _quantile_rank(x: np.ndarray) -> float:
        """计算当前值在窗口中的分位排名"""
        if len(x) < 10:
            return np.nan
        return (x <= x[-1]).mean()  # 当前值在窗口中的分位
    
    return series.rolling(window=window, min_periods=10).apply(
        _quantile_rank, raw=True
    )


@register_feature("extract_hilbert_features", category="hilbert")
def extract_hilbert_features(
    df: pd.DataFrame,
    price_fluctuation_col: str = "wpt_price_fluctuation",
    cvd_fluctuation_col: Optional[str] = "wpt_cvd_fluctuation",
    price_col: Optional[str] = "close",
    volume_col: Optional[str] = "volume",
    window: int = 64,
    ema_span: int = 10,
    # 高级功能参数
    use_adaptive_window: bool = False,
    base_window_min: int = 32,
    base_window_max: int = 128,
    period_lookback: int = 64,
    use_quantile_normalize: bool = False,
    quantile_window: int = 252,
    use_volume_fusion: bool = False,
    vol_detrend_window: int = 20,
    # Contract-focused output normalization:
    # If enabled, we will compute rolling quantile normalization for envelope-like columns
    # and overwrite the base envelope outputs to make them cross-asset comparable.
    replace_env_with_qnorm: bool = False,
) -> pd.DataFrame:
    """
    从 DataFrame 中提取 Hilbert 包络特征（滚动窗口 + EMA 平滑，无数据泄露）
    
    实现步骤：
    1. 使用 WPT 分解后的高频波动分量作为输入
    2. （可选）估计局部周期，自适应调整窗口长度
    3. 滚动计算 Hilbert 包络（每个时刻 t，仅用历史窗口 [t-W, t) 数据）
    4. 提取包络值（当前时刻值）
    5. EMA 平滑原始包络序列
    6. （可选）分位数标准化，支持跨品种可比
    7. （可选）计算成交量包络，融合多信号
    8. 计算衍生特征（包络比值、斜率等）
    9. shift(1) 确保时间对齐
    
    Args:
        df: DataFrame with WPT features
        price_fluctuation_col: Price fluctuation column (WPT 波动分量)
        cvd_fluctuation_col: CVD fluctuation column (WPT 波动分量，可选)
        price_col: Price column for ratio calculation (可选，用于包络比值)
        volume_col: Volume column for volume envelope fusion (可选)
        window: Rolling window size for Hilbert transform (default: 64)
               建议范围：32 ~ 128，需覆盖至少 2~3 个典型波动周期
        ema_span: EMA smoothing span (default: 10)
                 建议范围：5 ~ 20，可通过交叉验证优化
        use_adaptive_window: 是否使用自适应窗口（基于局部周期估计）
        base_window_min: 自适应窗口的最小值
        base_window_max: 自适应窗口的最大值
        period_lookback: 用于估计周期的历史窗口大小
        use_quantile_normalize: 是否使用分位数标准化（跨品种可比）
        quantile_window: 分位数标准化的滚动窗口（建议：252=日线年化）
        use_volume_fusion: 是否融合成交量包络特征
        vol_detrend_window: 成交量去趋势的滚动窗口（default: 20）
                           不同品种（股票 vs 加密货币）的成交量趋势周期差异大，可调整
    
    Returns:
        DataFrame with Hilbert features added:
        - hilbert_price_env: 价格波动强度（EMA 平滑后）
        - hilbert_cvd_env: CVD 波动强度（EMA 平滑后）
        - hilbert_cvd_price_env_ratio: 资金流 vs 价格强度比
        - hilbert_price_env_slope: 价格波动加速/减速
        - hilbert_cvd_env_slope: CVD 波动加速/减速
        - hilbert_adaptive_window: 自适应窗口长度（如果启用）
        - hilbert_price_env_qnorm: 价格包络分位数标准化（如果启用）
        - hilbert_cvd_env_qnorm: CVD 包络分位数标准化（如果启用）
        - hilbert_volume_env: 成交量波动强度（如果启用）
        - hilbert_env_price_vol_ratio: 价格/成交量包络比（如果启用）
        - hilbert_triple_divergence: 三元背离信号（如果启用）
    """
    df = df.copy()
    
    # 初始化基础特征列
    hilbert_cols = [
        "hilbert_price_env",
        "hilbert_cvd_env",
        "hilbert_cvd_price_env_ratio",
        "hilbert_price_env_slope",
        "hilbert_cvd_env_slope",
    ]
    
    # 添加高级功能特征列
    if use_adaptive_window:
        hilbert_cols.append("hilbert_adaptive_window")
    
    if use_quantile_normalize:
        hilbert_cols.extend([
            "hilbert_price_env_qnorm",
            "hilbert_cvd_env_qnorm",
        ])
    
    if use_volume_fusion:
        hilbert_cols.extend([
            "hilbert_volume_env",
            "hilbert_env_price_vol_ratio",
            "hilbert_triple_divergence",
        ])
    
    for col in hilbert_cols:
        df[col] = np.nan
    
    # 存储原始包络序列（用于 EMA 平滑）
    price_envelope_raw = []
    cvd_envelope_raw = []
    price_envelope_indices = []
    cvd_envelope_indices = []
    
    # 获取输入数据（如果存在）
    price_fluc = df[price_fluctuation_col].values if price_fluctuation_col in df.columns else None
    cvd_fluc = (
        df[cvd_fluctuation_col].values
        if (cvd_fluctuation_col and cvd_fluctuation_col in df.columns)
        else None
    )
    
    # ========== 自适应窗口：估计局部周期 ==========
    adaptive_windows = None
    if use_adaptive_window and price_fluc is not None:
        adaptive_windows = []
        for i in range(len(df)):
            if i < period_lookback:
                adaptive_windows.append(base_window_min)
            else:
                # 使用历史窗口估计周期（因果！）
                window_data = price_fluc[i - period_lookback : i]
                # 检查有效数据
                if np.any(~np.isnan(window_data)) and len(window_data) >= 20:
                    period = estimate_local_period(
                        window_data[~np.isnan(window_data)],
                        min_period=base_window_min // 2,
                        max_period=base_window_max // 2,
                    )
                    # window ≈ 3×周期，确保覆盖完整周期
                    adaptive_windows.append(min(period * 3, base_window_max))
                else:
                    adaptive_windows.append(base_window_min)
        
        # 保存自适应窗口序列
        df["hilbert_adaptive_window"] = adaptive_windows
    
    # 合并循环：同时处理 Price 和 CVD，提升性能
    min_valid_points = max(10, window // 2)  # 最小有效数据点要求
    
    # 确定实际使用的窗口（固定或自适应）
    def get_window(i: int) -> int:
        if adaptive_windows is not None:
            return adaptive_windows[i]
        return window
    
    for i in range(window, len(df)):
        current_window = get_window(i)
        # 确保窗口不超过可用数据
        current_window = min(current_window, i)
        if current_window < min_valid_points:
            continue
        # ========== 处理价格波动 ==========
        if price_fluc is not None:
            # 使用历史窗口数据 [i-current_window, i)
            window_data = price_fluc[i - current_window : i]
            
            # 检查窗口长度
            if window_data.size < min_valid_points:
                continue
            
            # 检查有效数据点数量（非 NaN）
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < min_valid_points:
                continue
            
            # 填充 NaN（仅前向填充，避免未来信息泄露）
            # 注意：在滚动窗口 [t-W, t) 中，只能用更早的历史值填充
            # 禁用 bfill()，因为它会用右侧值填左侧 → 引入未来信息
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()  # 仅前向填充（因果！）
            
            # 如果开头仍是 NaN（无历史有效值），跳过
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                # 计算 Hilbert 包络
                envelope = compute_hilbert_envelope(window_data_clean)
                
                # 只使用最后一个点的值（当前时刻的特征）
                if len(envelope) > 0:
                    price_envelope_raw.append(envelope[-1])
                    price_envelope_indices.append(i)
            except Exception:
                # 如果计算失败，跳过
                continue
        
        # ========== 处理 CVD 波动 ==========
        if cvd_fluc is not None:
            window_data = cvd_fluc[i - current_window : i]
            
            if window_data.size < min_valid_points:
                continue
            
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < min_valid_points:
                continue
            
            # 仅前向填充（因果！）
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()
            
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                envelope = compute_hilbert_envelope(window_data_clean)
                
                if len(envelope) > 0:
                    cvd_envelope_raw.append(envelope[-1])
                    cvd_envelope_indices.append(i)
            except Exception:
                continue
    
    # EMA 平滑包络序列
    if len(price_envelope_raw) > 0 and len(price_envelope_indices) > 0:
        # 创建临时 Series 用于 EMA
        price_env_series = pd.Series(price_envelope_raw, index=price_envelope_indices)
        # adjust=False: 使用递归 EMA 公式 y[t] = α·x[t] + (1−α)·y[t−1]，确保完全因果
        # 这是必须的，因为 adjust=True 会使用非递归公式，可能引入轻微的未来信息
        price_env_smoothed = price_env_series.ewm(span=ema_span, adjust=False).mean()
        
        # 将平滑后的值写回 DataFrame
        for idx, val in price_env_smoothed.items():
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc("hilbert_price_env")] = val
    
    if len(cvd_envelope_raw) > 0 and len(cvd_envelope_indices) > 0:
        # 确保长度匹配
        if len(cvd_envelope_raw) == len(cvd_envelope_indices):
            cvd_env_series = pd.Series(cvd_envelope_raw, index=cvd_envelope_indices)
            # adjust=False: 使用递归 EMA 公式，确保完全因果
            cvd_env_smoothed = cvd_env_series.ewm(span=ema_span, adjust=False).mean()
            
            for idx, val in cvd_env_smoothed.items():
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("hilbert_cvd_env")] = val
    
    # 计算衍生特征
    # 1. 包络比值（CVD / Price）
    if "hilbert_cvd_env" in df.columns and "hilbert_price_env" in df.columns:
        price_env = df["hilbert_price_env"]
        cvd_env = df["hilbert_cvd_env"]
        
        # 避免除零，使用安全除法
        ratio = cvd_env / price_env.replace(0, np.nan)
        # Guard against inf/-inf from degenerate envelopes (flat price / tiny denominators)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["hilbert_cvd_price_env_ratio"] = ratio
    
    # 2. 包络斜率（波动加速/减速）
    if "hilbert_price_env" in df.columns:
        price_env = df["hilbert_price_env"]
        # 使用 diff 计算斜率（因果操作）
        df["hilbert_price_env_slope"] = price_env.diff()
    
    if "hilbert_cvd_env" in df.columns:
        cvd_env = df["hilbert_cvd_env"]
        df["hilbert_cvd_env_slope"] = cvd_env.diff()

    # Final safety: never output inf values for downstream consumers/tests
    for col in [
        "hilbert_price_env",
        "hilbert_cvd_env",
        "hilbert_cvd_price_env_ratio",
        "hilbert_price_env_slope",
        "hilbert_cvd_env_slope",
    ]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # ========== 分位数标准化（跨品种可比）==========
    if use_quantile_normalize:
        if "hilbert_price_env" in df.columns:
            df["hilbert_price_env_qnorm"] = rolling_quantile_normalize(
                df["hilbert_price_env"], window=quantile_window
            )
        
        if "hilbert_cvd_env" in df.columns:
            df["hilbert_cvd_env_qnorm"] = rolling_quantile_normalize(
                df["hilbert_cvd_env"], window=quantile_window
            )

    # Optional: overwrite base envelope outputs with their quantile-normalized versions.
    # This keeps column names stable for downstream strategy configs while enforcing
    # cross-asset comparability (rank transformation).
    if bool(replace_env_with_qnorm):
        if "hilbert_price_env_qnorm" in df.columns:
            df["hilbert_price_env"] = df["hilbert_price_env_qnorm"]
        if "hilbert_cvd_env_qnorm" in df.columns:
            df["hilbert_cvd_env"] = df["hilbert_cvd_env_qnorm"]
    
    # ========== 成交量包络融合 ==========
    if use_volume_fusion and volume_col and volume_col in df.columns:
        # 先对成交量做去趋势（简单滚动均值去趋势）
        # 注意：vol_detrend_window 可配置，适应不同品种的成交量趋势周期
        volume = df[volume_col].values
        volume_trend = pd.Series(volume).rolling(window=vol_detrend_window, min_periods=1).mean()
        volume_fluc = volume - volume_trend.values
        
        # 直接计算成交量包络（避免递归调用，使用内联逻辑）
        vol_window = 32  # 成交量使用较小窗口，因为成交量波动更频繁
        vol_ema_span = 5
        vol_min_valid_points = max(10, vol_window // 2)
        
        vol_envelope_raw = []
        vol_envelope_indices = []
        
        for i in range(vol_window, len(df)):
            window_data = volume_fluc[i - vol_window : i]
            
            if window_data.size < vol_min_valid_points:
                continue
            
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < vol_min_valid_points:
                continue
            
            # 仅前向填充（因果！）
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()
            
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                envelope = compute_hilbert_envelope(window_data_clean)
                if len(envelope) > 0:
                    vol_envelope_raw.append(envelope[-1])
                    vol_envelope_indices.append(i)
            except Exception:
                continue
        
        # EMA 平滑成交量包络
        if len(vol_envelope_raw) > 0 and len(vol_envelope_indices) > 0:
            vol_env_series = pd.Series(vol_envelope_raw, index=vol_envelope_indices)
            vol_env_smoothed = vol_env_series.ewm(span=vol_ema_span, adjust=False).mean()
            
            for idx, val in vol_env_smoothed.items():
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("hilbert_volume_env")] = val

            # If we are enforcing rank-based normalization, quantile-normalize volume envelope too.
            if bool(use_quantile_normalize):
                df["hilbert_volume_env_qnorm"] = rolling_quantile_normalize(
                    df["hilbert_volume_env"], window=quantile_window
                )
                if bool(replace_env_with_qnorm):
                    df["hilbert_volume_env"] = df["hilbert_volume_env_qnorm"]
            
            # 计算价格/成交量包络比
            if "hilbert_price_env" in df.columns:
                price_env = df["hilbert_price_env"]
                vol_env = df["hilbert_volume_env"]
                # 避免除零
                df["hilbert_env_price_vol_ratio"] = (
                    price_env / (vol_env + 1e-8)
                )
                
                # 三元背离信号：价格包络新高，但 CVD 包络未新高，且成交量包络下降
                if "hilbert_cvd_env" in df.columns:
                    cvd_env = df["hilbert_cvd_env"]
                    
                    # 价格包络创新高（滚动20期）
                    price_new_high = (
                        price_env > price_env.rolling(window=20, min_periods=1).max().shift(1)
                    )
                    
                    # CVD 包络未创新高
                    cvd_not_high = (
                        cvd_env < cvd_env.rolling(window=20, min_periods=1).max()
                    )
                    
                    # 成交量包络下降
                    vol_declining = vol_env < vol_env.shift(1)
                    
                    # 三元背离信号
                    df["hilbert_triple_divergence"] = (
                        (price_new_high & cvd_not_high & vol_declining).astype(float)
                    )

    # ------------------------------------------------------------------
    # If we replaced envelopes with quantile-normalized values, refresh the
    # derived features so they are consistent with the final envelope series.
    # Also apply safe transforms for ratio-like outputs to avoid extreme spikes.
    # ------------------------------------------------------------------
    if bool(replace_env_with_qnorm):
        eps = 1e-8

        def _robust_rolling_z(v: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
            v = pd.to_numeric(v, errors="coerce").astype(float)
            med = v.rolling(window=window, min_periods=min_periods).median()
            q25 = v.rolling(window=window, min_periods=min_periods).quantile(0.25)
            q75 = v.rolling(window=window, min_periods=min_periods).quantile(0.75)
            iqr = (q75 - q25).replace(0, np.nan)
            z = (v - med) / (iqr + eps)
            return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Ratio (CVD/Price env): use log-ratio then robust rolling scaling.
        if "hilbert_cvd_env" in df.columns and "hilbert_price_env" in df.columns:
            price_env = pd.to_numeric(df["hilbert_price_env"], errors="coerce").astype(float).fillna(0.0)
            cvd_env = pd.to_numeric(df["hilbert_cvd_env"], errors="coerce").astype(float).fillna(0.0)
            log_ratio = np.log((cvd_env + eps) / (price_env + eps))
            df["hilbert_cvd_price_env_ratio"] = _robust_rolling_z(log_ratio, window=int(quantile_window))

            # Slopes on normalized envelope (bounded diffs)
            df["hilbert_price_env_slope"] = price_env.diff().clip(-1.0, 1.0)
            df["hilbert_cvd_env_slope"] = cvd_env.diff().clip(-1.0, 1.0)

        # Price/Volume env ratio: log-ratio then robust scaling (only when fusion exists)
        if "hilbert_env_price_vol_ratio" in df.columns and "hilbert_volume_env" in df.columns and "hilbert_price_env" in df.columns:
            price_env = pd.to_numeric(df["hilbert_price_env"], errors="coerce").astype(float).fillna(0.0)
            vol_env = pd.to_numeric(df["hilbert_volume_env"], errors="coerce").astype(float).fillna(0.0)
            log_ratio = np.log((price_env + eps) / (vol_env + eps))
            df["hilbert_env_price_vol_ratio"] = _robust_rolling_z(log_ratio, window=int(quantile_window))
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    # 注意：shift(1) 后，NaN 表示历史数据不足，不应 fillna(0.0)
    for col in hilbert_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    # 不填充 NaN：0 波动 ≠ 未知，保留 NaN 让模型知道这是缺失的历史信息
    
    return df
