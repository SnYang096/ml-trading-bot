"""
订单流特征：基于 tick 数据的订单流分析

核心特征：
1. VPIN (Volume-Synchronized Probability of Informed Trading) - 真实实现
2. Trade Clustering（交易聚集性）- 连续同向成交的聚集性
3. 其他订单流衍生特征

Trade Clustering 与 VPIN 互补：
- VPIN：关注 volume-bucketed 的净买卖差，不关心成交顺序
- Trade Clustering：关注成交的时序模式，捕捉连续同向交易的聚集性
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from collections import deque

# 常量定义
TOL = 1e-10  # 浮点比较容差（用于 volume 比较、时间戳对齐等）
MIN_BUCKET_VOLUME_TOL = 1e-9  # 桶体积填充容差（用于判断桶是否填满）

try:
    from scipy.stats import percentileofscore, entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_entropy = None

try:
    from numba import njit  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.data_tools.tick_loader import (
    load_tick_data,
    deserialize_tick_loader_params,
    compute_vpin_from_cached_ticks,
)


# Numba JIT 编译的滚动 MAD 函数（高性能优化版）
# 使用插入排序维护有序窗口，复杂度从 O(w log w) 降至 O(w)
if HAS_NUMBA:
    @njit(cache=True)
    def _rolling_mad_numba_optimized(arr: np.ndarray, window: int) -> tuple:
        """
        使用 numba 加速的滚动中位数绝对偏差（MAD）计算（优化版）
        优化策略：使用插入排序维护有序窗口，避免每次完整排序
        复杂度：从 O(N × w log w) 降至 O(N × w)
        Args:
            arr: 输入数组
            window: 滚动窗口大小
        Returns:
            (rolling_median, rolling_mad) 元组，均为 np.ndarray
        """
        n = len(arr)
        median_result = np.full(n, np.nan, dtype=np.float64)
        mad_result = np.full(n, np.nan, dtype=np.float64)
        if window > n or window < 1:
            return median_result, mad_result
        # 初始化第一个窗口的有序数组
        win = arr[:window].copy()
        # 使用插入排序初始化（对于小窗口，插入排序比快排更快）
        for i in range(1, window):
            key = win[i]
            j = i - 1
            while j >= 0 and win[j] > key:
                win[j + 1] = win[j]
                j -= 1
            win[j + 1] = key
        # 计算第一个窗口的中位数和 MAD
        mid = window // 2
        if window % 2 == 0:
            median = (win[mid - 1] + win[mid]) / 2.0
        else:
            median = win[mid]
        # 计算 MAD
        abs_dev = np.empty(window)
        for j in range(window):
            abs_dev[j] = abs(win[j] - median)
        # 对 abs_dev 排序
        for i in range(1, window):
            key = abs_dev[i]
            j = i - 1
            while j >= 0 and abs_dev[j] > key:
                abs_dev[j + 1] = abs_dev[j]
                j -= 1
            abs_dev[j + 1] = key
        if window % 2 == 0:
            mad = (abs_dev[mid - 1] + abs_dev[mid]) / 2.0
        else:
            mad = abs_dev[mid]
        median_result[window - 1] = median
        mad_result[window - 1] = mad
        # 滑动窗口：对每个后续位置，移除旧元素，插入新元素
        for i in range(window, n):
            # 移除离开窗口的元素 arr[i - window]
            old_val = arr[i - window]
            # 在有序数组 win 中找到 old_val 的位置并删除
            # 使用二分查找定位（更快）
            left, right = 0, window - 1
            pos = -1
            while left <= right:
                mid_idx = (left + right) // 2
                if win[mid_idx] == old_val:
                    pos = mid_idx
                    break
                elif win[mid_idx] < old_val:
                    left = mid_idx + 1
                else:
                    right = mid_idx - 1
            # 如果找到，删除它（左移覆盖）
            if pos >= 0:
                for k in range(pos, window - 1):
                    win[k] = win[k + 1]
            else:
                # 如果没找到（可能由于浮点误差），使用线性搜索
                for j in range(window - 1):
                    if abs(win[j] - old_val) < TOL:
                        for k in range(j, window - 1):
                            win[k] = win[k + 1]
                        break
            # 插入新元素 arr[i]（保持有序）
            new_val = arr[i]
            insert_pos = window - 1
            while insert_pos > 0 and win[insert_pos - 1] > new_val:
                win[insert_pos] = win[insert_pos - 1]
                insert_pos -= 1
            win[insert_pos] = new_val
            # 计算中位数
            if window % 2 == 0:
                median = (win[mid - 1] + win[mid]) / 2.0
            else:
                median = win[mid]
            # 计算 MAD：median(|x - median(x)|)
            # 重新计算 abs_dev（因为 median 变了）
            for j in range(window):
                abs_dev[j] = abs(win[j] - median)
            # 对 abs_dev 排序（使用插入排序，因为窗口小）
            for idx in range(1, window):
                key = abs_dev[idx]
                j = idx - 1
                while j >= 0 and abs_dev[j] > key:
                    abs_dev[j + 1] = abs_dev[j]
                    j -= 1
                abs_dev[j + 1] = key
            if window % 2 == 0:
                mad = (abs_dev[mid - 1] + abs_dev[mid]) / 2.0
            else:
                mad = abs_dev[mid]
            median_result[i] = median
            mad_result[i] = mad
        return median_result, mad_result

    # 保持向后兼容：提供只返回 MAD 的版本
    @njit(cache=True)
    def _rolling_mad_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """向后兼容：只返回 MAD（内部调用优化版）"""
        _, mad = _rolling_mad_numba_optimized(arr, window)
        return mad
else:
    # Fallback: 如果 numba 不可用，定义占位函数
    def _rolling_mad_numba_optimized(arr: np.ndarray, window: int) -> tuple:
        """Fallback 实现（不使用 numba）"""
        raise NotImplementedError("numba is required for _rolling_mad_numba_optimized")

    def _rolling_mad_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """Fallback 实现（不使用 numba）"""
        raise NotImplementedError("numba is required for _rolling_mad_numba")


def compute_vpin_from_ticks(
    ticks: pd.DataFrame,
    bucket_volume: Optional[float] = None,
    n_buckets: int = 50,
    lookback_days: int = 7,
    quantile: float = 0.3,
    adaptive: bool = True,
) -> pd.DataFrame:
    """
    基于逐笔成交数据计算真实 VPIN（向量化实现）
    Args:
        ticks: DataFrame with tick data, must contain:
            - timestamp (datetime index)
            - price (float)
            - volume (float)
            - side (1 for buy, -1 for sell, or 'buy'/'sell')
        bucket_volume: Fixed bucket volume (BTC/USD). If None and adaptive=True, will be calculated
        n_buckets: Number of buckets for rolling average
        lookback_days: Days to look back for adaptive bucket calculation
        quantile: Quantile for adaptive bucket volume (0.2-0.4 recommended)
        adaptive: If True, use adaptive bucket volume based on recent volume
    Returns:
        DataFrame with columns:
            - vpin: VPIN values (0-1 range)
            - signed_imbalance: Signed imbalance (-1 to 1, positive = buy pressure)
        Indexed by timestamp
    """
    if len(ticks) == 0:
        # 统一返回 DataFrame（即使为空）
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    # 标准化 side
    if "side" not in ticks.columns:
        raise ValueError("ticks must contain 'side' column (1/-1 or 'buy'/'sell')")
    ticks = ticks.copy()
    if ticks["side"].dtype == "object":
        ticks["side"] = ticks["side"].map({"buy": 1, "sell": -1, "BUY": 1, "SELL": -1})
    # 过滤无效的 side 值（NaN、0、'unknown' 等）
    valid_side_mask = ticks["side"].isin([1, -1])
    if not valid_side_mask.all():
        invalid_count = (~valid_side_mask).sum()
        if invalid_count > 0:
            print(f"   ⚠️  Filtering {invalid_count} ticks with invalid side values")
        ticks = ticks[valid_side_mask].copy()
    if len(ticks) == 0:
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    # 计算自适应 bucket_volume
    if adaptive and bucket_volume is None:
        # 按小时聚合成交量
        if isinstance(ticks.index, pd.DatetimeIndex):
            hourly_volumes = ticks["volume"].resample("1H").sum()
        else:
            # 如果 index 不是 datetime，假设有 timestamp 列
            if "timestamp" in ticks.columns:
                ticks_temp = ticks.set_index("timestamp")
                hourly_volumes = ticks_temp["volume"].resample("1H").sum()
            else:
                raise ValueError("ticks must have DatetimeIndex or 'timestamp' column")
        # 计算典型小时成交量（使用分位数）
        lookback_hours = lookback_days * 24
        typical_hourly_vol = hourly_volumes.rolling(
            window=lookback_hours, min_periods=1
        ).quantile(quantile)
        # bucket_volume = 典型小时成交量的一部分（如 30%）
        bucket_volume = typical_hourly_vol.iloc[-1] if len(typical_hourly_vol) > 0 else 100.0
        # 设置最小桶体积限制（资产自适应，基于名义价值）
        # 使用典型价格估算，确保最小桶的名义价值 >= min_nominal_value USD
        min_nominal_value = 1000.0  # 最小名义价值（USD），可根据策略调整
        if "price" in ticks.columns and len(ticks) > 0:
            typical_price = ticks["price"].median()
            if typical_price > 0:
                min_bucket_volume = min_nominal_value / typical_price
            else:
                min_bucket_volume = 0.01  # fallback
        else:
            min_bucket_volume = 0.01  # fallback（如果没有价格数据）
        bucket_volume = max(bucket_volume, min_bucket_volume)
    if bucket_volume is None:
        bucket_volume = 100.0  # 默认值
    # 确保按时间排序
    if not isinstance(ticks.index, pd.DatetimeIndex):
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp").sort_index()
        else:
            raise ValueError("ticks must have DatetimeIndex or 'timestamp' column")
    else:
        ticks = ticks.sort_index()
    # 向量化实现：使用 cumsum + searchsorted 划分桶边界
    # 这比 iterrows() 快 10-100 倍
    volumes = ticks["volume"].values
    sides = ticks["side"].values
    timestamps = ticks.index.values
    # 计算累计成交量
    cumvol = np.cumsum(volumes)
    total_volume = cumvol[-1]
    if total_volume < bucket_volume:
        # 总成交量不足一个桶
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    # 生成桶边界（累计成交量阈值）
    bucket_edges = np.arange(bucket_volume, total_volume + bucket_volume, bucket_volume)
    # 找到每个桶边界对应的 tick 索引
    # searchsorted 返回插入位置，即第一个 >= bucket_edge 的 cumvol 位置
    bucket_tick_indices = np.searchsorted(cumvol, bucket_edges, side="right")
    # 过滤超出范围的索引
    valid_mask = bucket_tick_indices < len(ticks)
    bucket_tick_indices = bucket_tick_indices[valid_mask]
    bucket_edges = bucket_edges[valid_mask]
    if len(bucket_tick_indices) == 0:
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    # 计算每个桶的 buy/sell volume（向量化 + 正确处理跨桶分割）
    buckets_data = []
    prev_cumvol = 0.0
    prev_idx = 0
    for i, bucket_edge in enumerate(bucket_edges):
        bucket_end_idx = bucket_tick_indices[i]
        # 计算桶内买卖量
        buy_vol = 0.0
        sell_vol = 0.0
        # 处理完整包含在桶内的 ticks
        if prev_idx < bucket_end_idx:
            # 完整包含的 tick 范围：[prev_idx, bucket_end_idx)
            for j in range(prev_idx, bucket_end_idx):
                if sides[j] == 1:
                    buy_vol += volumes[j]
                else:
                    sell_vol += volumes[j]
        # 处理跨越桶边界的最后一个 tick（如果有）
        # 计算桶还需要多少 volume
        if bucket_end_idx > 0:
            cumvol_at_end = cumvol[bucket_end_idx - 1]
        else:
            cumvol_at_end = 0.0
        remaining_to_fill = bucket_edge - cumvol_at_end
        if remaining_to_fill > MIN_BUCKET_VOLUME_TOL and bucket_end_idx < len(ticks):
            # 需要从 bucket_end_idx 这个 tick 借用部分 volume
            borrow_vol = min(remaining_to_fill, volumes[bucket_end_idx])
            if sides[bucket_end_idx] == 1:
                buy_vol += borrow_vol
            else:
                sell_vol += borrow_vol
        # 计算 VPIN 和 signed imbalance
        imbalance = abs(buy_vol - sell_vol)
        vpin_value = imbalance / bucket_volume
        signed_imbalance = (buy_vol - sell_vol) / bucket_volume
        # 桶的时间戳：使用桶内最后一个 tick 的时间
        # 注意：也可以考虑使用"桶结束时的虚拟时间"（bucket_edge 对应的累计时间），
        # 但当前实现使用最后一个 tick 的时间更直观，且能准确反映事件发生时刻
        # 如果单个 tick 产生多个桶，使用纳秒级递增（避免微秒溢出）
        if bucket_end_idx > 0:
            bucket_timestamp = timestamps[bucket_end_idx - 1]
        else:
            bucket_timestamp = timestamps[0] if len(timestamps) > 0 else pd.Timestamp.now()
        # 检查是否有多个桶共享同一时间戳（同一 tick 产生多个桶）
        if i > 0 and len(buckets_data) > 0:
            last_timestamp = buckets_data[-1]["timestamp"]
            if bucket_timestamp == last_timestamp:
                # 使用纳秒级递增（1 秒 = 1e9 纳秒，足够大，避免溢出）
                # 这确保了每个桶都有唯一时间戳，避免聚合时被覆盖
                bucket_timestamp = bucket_timestamp + pd.Timedelta(nanoseconds=i)
        buckets_data.append({
            "timestamp": bucket_timestamp,
            "vpin": vpin_value,
            "signed_imbalance": signed_imbalance,
        })
        prev_idx = bucket_end_idx
    if len(buckets_data) == 0:
        # 统一返回 DataFrame（即使为空）
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    # 转为 DataFrame 并计算滚动平均
    buckets_df = pd.DataFrame(buckets_data)
    buckets_df = buckets_df.set_index("timestamp")
    # 确保有 signed_imbalance 列（如果没有，设为 0）
    if "signed_imbalance" not in buckets_df.columns:
        buckets_df["signed_imbalance"] = 0.0
    # 滚动平均
    vpin_series = buckets_df["vpin"].rolling(window=n_buckets, min_periods=1).mean()
    signed_series = buckets_df["signed_imbalance"].rolling(
        window=n_buckets, min_periods=1
    ).mean()
    # 统一返回 DataFrame
    result_df = pd.DataFrame({
        "vpin": vpin_series,
        "signed_imbalance": signed_series
    })
    return result_df


# 注意：compute_vpin_from_ohlcv 函数已移除
# VPIN 必须基于 tick 数据计算，不支持 proxy 实现
# 如果只有 OHLCV 数据，请使用 tick 数据或移除 VPIN 特征


def extract_order_flow_features(
    df: pd.DataFrame,
    ticks: Optional[pd.DataFrame] = None,
    ticks_loader_json: Optional[str] = None,
    open_col: str = "open",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    buy_qty_col: Optional[str] = None,
    sell_qty_col: Optional[str] = None,
    vpin_bucket_volume: Optional[float] = None,
    vpin_n_buckets: int = 50,
    vpin_adaptive: bool = True,
    freq: Optional[str] = None,
    include_trade_clustering: bool = True,
    trade_clustering_window: int = 100,
) -> pd.DataFrame:
    """
    提取订单流特征（VPIN 等）
    注意：VPIN 必须基于 tick 数据计算，不支持 proxy 实现。
    如果没有 tick 数据，将抛出 ValueError。
    Args:
        df: DataFrame with OHLCV data
        ticks: Tick data for real VPIN calculation (必需)
        open_col: Open price column (未使用，保留用于兼容)
        close_col: Close price column (未使用，保留用于兼容)
        high_col: High price column (未使用，保留用于兼容)
        low_col: Low price column (未使用，保留用于兼容)
        volume_col: Volume column (未使用，保留用于兼容)
        buy_qty_col: Buy quantity column (未使用，保留用于兼容)
        sell_qty_col: Sell quantity column (未使用，保留用于兼容)
        vpin_bucket_volume: Fixed bucket volume for VPIN
        vpin_n_buckets: Number of buckets for VPIN rolling average
        vpin_adaptive: Whether to use adaptive VPIN
    Returns:
        DataFrame with order flow features added
    Raises:
        ValueError: 如果没有提供 tick 数据或 tick 数据为空
    """
    df = df.copy()
    # 检查 tick 数据
    vpin_series = None
    if ticks is not None and len(ticks) > 0:
        required_tick_cols = ["price", "volume", "side"]
        missing_cols = [col for col in required_tick_cols if col not in ticks.columns]
        if missing_cols:
            raise ValueError(
                f"Tick data must contain columns: {required_tick_cols}. "
                f"Missing columns: {missing_cols}"
            )
        print("   📊 Computing real VPIN from tick data (in-memory)...")
        vpin_series = compute_vpin_from_ticks(
            ticks,
            bucket_volume=vpin_bucket_volume,
            n_buckets=vpin_n_buckets,
            adaptive=vpin_adaptive,
        )
    elif ticks_loader_json:
        loader_params = deserialize_tick_loader_params(ticks_loader_json)
        tick_files = loader_params.get("tick_files", [])
        print(f"   📊 Computing real VPIN from ticks ({len(tick_files)} files)...")
        vpin_series = compute_vpin_from_cached_ticks(
            cache_files=tick_files,
            start_ts=loader_params["start_ts"],
            end_ts=loader_params["end_ts"],
            bucket_volume=vpin_bucket_volume,
            n_buckets=vpin_n_buckets,
            adaptive=vpin_adaptive,
            lookback_minutes=loader_params.get("lookback_minutes", 60),
        )
    else:
        # 如果没有tick数据，直接抛出错误并退出
        # VPIN必须基于tick数据计算，不支持降级处理
        raise ValueError(
            "VPIN calculation requires tick data. "
            "Please provide tick data via the 'ticks' parameter "
            "or configure ticks_loader_json. "
            "VPIN cannot be computed without tick data."
        )
    # 对齐到 df 的时间索引（右对齐，避免未来信息泄露）
    # 性能优化：优先使用 resample，失败时回退到循环（兼容性）
    if isinstance(df.index, pd.DatetimeIndex):
        # 关键原则：VPIN 事件只能影响当前及未来的 K 线，不能影响过去
        # 处理返回值：可能是 Series 或 DataFrame
        if isinstance(vpin_series, pd.DataFrame):
            vpin_events = vpin_series
        else:
            vpin_events = vpin_series.to_frame(name="vpin")
        # 推断 df 的频率（假设 df 是等频 K 线）
        # 改进：优先使用用户提供的 freq，否则自动推断
        if freq is None:
            freq = pd.infer_freq(df.index)
        if freq is None:
            # 如果无法推断，尝试从时间间隔估算
            if len(df.index) > 1:
                # 使用多个样本点计算平均间隔（更可靠）
                if len(df.index) >= 10:
                    # 使用前 10 个间隔的平均值
                    time_diffs = [df.index[i+1] - df.index[i] for i in range(min(10, len(df.index)-1))]
                    freq_td = pd.Timedelta(np.mean([td.total_seconds() for td in time_diffs]), unit='s')
                else:
                    # 使用第一个间隔
                    freq_td = df.index[1] - df.index[0]
                # 尝试转换为标准频率字符串（更宽松的匹配）
                freq = None
                std_freqs = ["1T", "5T", "15T", "30T", "1H", "4H", "1D"]
                for std_freq in std_freqs:
                    std_td = pd.Timedelta(std_freq)
                    # 允许 5% 的误差（处理非标准 K 线）
                    if abs((freq_td - std_td).total_seconds()) < abs(std_td.total_seconds() * 0.05):
                        freq = std_freq
                        break
                # 如果仍无法匹配，使用计算出的 freq_td
                if freq is None:
                    freq_td = freq_td
            else:
                freq = "1T"  # fallback
                freq_td = pd.Timedelta(minutes=1)
        else:
            freq_td = pd.Timedelta(freq) if isinstance(freq, str) else freq
        # 方法1：严格右对齐的向量化实现（极快，O(N log M)）
        aligned_vpin = None
        aligned_signed = None
        # 使用原始事件时间戳进行严格右对齐（不依赖 resample）
        # 关键：VPIN 事件应分配给满足 kline_start <= event_time < kline_end 的 K 线
        try:
            # 获取原始 VPIN 事件时间戳和值
            if isinstance(vpin_events, pd.DataFrame):
                event_times = vpin_events.index.values
                vpin_values = vpin_events["vpin"].values
                if "signed_imbalance" in vpin_events.columns:
                    signed_values = vpin_events["signed_imbalance"].values
                else:
                    signed_values = np.zeros_like(vpin_values)
            else:
                # 兼容旧格式（Series）
                event_times = vpin_events.index.values
                vpin_values = vpin_events.values
                signed_values = np.zeros_like(vpin_values)
            # K 线时间边界
            kline_starts = df.index.values
            kline_ends = (df.index + freq_td).values
            # 严格右对齐：找到每个事件所属的 K 线
            # 使用 searchsorted 找到第一个 > event_time 的 kline_start 位置
            # 则 idx = pos - 1 就是所属 K 线（满足 kline_starts[idx] <= event_time < kline_ends[idx]）
            pos = np.searchsorted(kline_starts, event_times, side="right")
            idx = pos - 1
            # 验证：确保事件时间在 K 线窗口内
            valid_mask = (idx >= 0) & (idx < len(df)) & (event_times < kline_ends[idx])
            if valid_mask.any():
                # 有效的 K 线索引和对应的 VPIN 值
                valid_idx = idx[valid_mask]
                valid_vpin = vpin_values[valid_mask]
                valid_signed = signed_values[valid_mask]
                # 按 K 线索引分组聚合（取均值）
                aligned_vpin = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
                # 使用 pandas groupby 聚合（高效）
                vpin_series = pd.Series(valid_vpin, index=valid_idx)
                signed_series = pd.Series(valid_signed, index=valid_idx)
                vpin_aggregated = vpin_series.groupby(valid_idx).mean()
                signed_aggregated = signed_series.groupby(valid_idx).mean()
                aligned_vpin.iloc[vpin_aggregated.index] = vpin_aggregated.values
                aligned_signed.iloc[signed_aggregated.index] = signed_aggregated.values
            else:
                # 没有有效事件，初始化为 0
                aligned_vpin = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
        except Exception as e:
            # 向量化方法失败，回退到循环方法
            print(f"   ⚠️  Vectorized alignment failed ({e}), falling back to loop method")
            aligned_vpin = None
            aligned_signed = None
        # 方法2：循环方法（兼容性，当向量化方法不可用时）
        if aligned_vpin is None:
            aligned_vpin = pd.Series(index=df.index, dtype=float)
            aligned_signed = pd.Series(index=df.index, dtype=float)
            # 获取事件数据
            if isinstance(vpin_events, pd.DataFrame):
                event_vpin = vpin_events["vpin"]
                event_signed = vpin_events.get("signed_imbalance", pd.Series(0.0, index=vpin_events.index))
            else:
                event_vpin = vpin_events
                event_signed = pd.Series(0.0, index=vpin_events.index)
            for kline_time in df.index:
                window_end = kline_time + freq_td
                # 找到该 K 线时间段内的所有 VPIN 事件（右对齐：[kline_time, kline_time + freq)）
                window_mask = (vpin_events.index >= kline_time) & (
                    vpin_events.index < window_end
                )
                if window_mask.any():
                    aligned_vpin.loc[kline_time] = event_vpin.loc[window_mask].mean()
                    aligned_signed.loc[kline_time] = event_signed.loc[window_mask].mean()
                else:
                    aligned_vpin.loc[kline_time] = 0.0
                    aligned_signed.loc[kline_time] = 0.0
        df["vpin"] = aligned_vpin
        # 对齐 signed_imbalance（已在向量化或循环方法中处理）
        if aligned_signed is None:
            aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
        df["vpin_signed_imbalance"] = aligned_signed
    else:
        # 如果 df 没有 datetime index，使用简单映射（不推荐，但保持兼容）
        vpin_series = vpin_series.reindex(df.index).fillna(0.0)
        df["vpin"] = vpin_series
    # VPIN 的滚动统计
    for w in [5, 10, 20]:
        df[f"vpin_ma{w}"] = df["vpin"].rolling(window=w, min_periods=1).mean()
        df[f"vpin_max{w}"] = df["vpin"].rolling(window=w, min_periods=1).max()
    # VPIN 变化率（捕捉订单流突增）
    df["vpin_change"] = df["vpin"].diff()
    df["vpin_change_pct"] = df["vpin"].pct_change().fillna(0.0)
    # 增强特征：Z-score（识别异常高的订单流不平衡）
    for w in [20, 50]:
        rolling_mean = df["vpin"].rolling(window=w, min_periods=1).mean()
        rolling_std = df["vpin"].rolling(window=w, min_periods=1).std()
        df[f"vpin_zscore_{w}"] = (df["vpin"] - rolling_mean) / (rolling_std + TOL)
    # 增强特征：分位数排名（在滚动窗口中的位置，0~1）
    # 性能优化：使用 scipy.stats.percentileofscore（如果可用）
    for w in [20, 50]:
        if HAS_SCIPY:
            def rolling_quantile_rank(x):
                """高效计算分位数排名"""
                if len(x) == 0:
                    return 0.0
                # percentileofscore 返回 0~100，需除以 100
                return percentileofscore(x, x[-1], kind="mean") / 100.0
            df[f"vpin_quantile_rank_{w}"] = (
                df["vpin"].rolling(window=w, min_periods=1)
                .apply(rolling_quantile_rank, raw=True)
            )
        else:
            # fallback：使用原始方法（较慢）
            df[f"vpin_quantile_rank_{w}"] = (
                df["vpin"].rolling(window=w, min_periods=1)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            )
    # 增强特征：VPIN 波动率（衡量订单流稳定性）
    for w in [10, 20]:
        df[f"vpin_volatility_{w}"] = df["vpin"].rolling(window=w, min_periods=1).std()
    # 增强特征：Spike 标志（VPIN 异常突增）
    # 性能优化：使用 numba 加速的 MAD 计算（优化版，比 pandas apply 快 100+ 倍）
    # 优化：同时计算 median 和 mad，避免重复计算
    for w in [20, 50]:
        # 优化 MAD 计算：优先使用 numba（极快且正确）
        # MAD = median(|x - median(x)|)
        if HAS_NUMBA:
            # 使用 numba JIT 编译的滚动 MAD（优化版：插入排序维护有序窗口）
            try:
                vpin_values = df["vpin"].values
                # 优化版同时返回 median 和 mad，避免重复计算
                rolling_median_values, rolling_mad_values = _rolling_mad_numba_optimized(
                    vpin_values, w
                )
                rolling_median = pd.Series(rolling_median_values, index=df.index)
                rolling_mad = pd.Series(rolling_mad_values, index=df.index)
                # 将前 window-1 个 NaN 填充为第一个有效值（与 rolling 行为一致）
                rolling_median = rolling_median.bfill().fillna(0.0)
                rolling_mad = rolling_mad.bfill().fillna(0.0)
            except Exception as e:
                # numba 计算失败，回退到 pandas apply
                print(f"   ⚠️  Numba MAD calculation failed ({e}), falling back to pandas apply")
                rolling_median = df["vpin"].rolling(window=w, min_periods=1).median()
                rolling_mad = (
                    df["vpin"].rolling(window=w, min_periods=1)
                    .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                )
        else:
            # 无 numba：根据窗口大小选择策略
            rolling_median = df["vpin"].rolling(window=w, min_periods=1).median()
            if w <= 50:
                # 小窗口：使用 pandas apply（精确但较慢）
                rolling_mad = (
                    df["vpin"].rolling(window=w, min_periods=1)
                    .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                )
            else:
                # 大窗口：使用 std 作为近似（更快，牺牲一点鲁棒性）
                # std ≈ 1.4826 * MAD（对于正态分布）
                rolling_std = df["vpin"].rolling(window=w, min_periods=1).std()
                rolling_mad = rolling_std / 1.4826
        threshold = rolling_median + 2 * rolling_mad
        df[f"vpin_spike_flag_{w}"] = (df["vpin"] > threshold).astype(int)
    # 新增特征：VPIN 动量（捕捉不平衡加速）
    df["vpin_momentum"] = df["vpin_ma5"] - df["vpin_ma20"]
    # 新增特征：Signed Imbalance Z-score（识别极端买卖压力）
    if "vpin_signed_imbalance" in df.columns:
        for w in [20, 50]:
            rolling_mean = df["vpin_signed_imbalance"].rolling(window=w, min_periods=1).mean()
            rolling_std = df["vpin_signed_imbalance"].rolling(window=w, min_periods=1).std()
            df[f"vpin_signed_imbalance_zscore_{w}"] = (
                df["vpin_signed_imbalance"] - rolling_mean
            ) / (rolling_std + TOL)
    # Trade Clustering 特征（与 VPIN 互补）
    # VPIN 关注 volume-bucketed 的净买卖差，Trade Clustering 关注连续同向成交的聚集性
    if include_trade_clustering and ticks is not None and len(ticks) > 0:
        print("   📊 Computing trade clustering features...")
        try:
            df = extract_trade_clustering_features(
                df,
                ticks=ticks,
                window_size=trade_clustering_window,
                freq=freq,
            )
        except Exception as e:
            print(f"   ⚠️  Trade clustering feature extraction failed: {e}")
    elif include_trade_clustering and ticks_loader_json:
        # 使用 ticks_loader_json 计算 Trade Clustering
        print("   📊 Computing trade clustering features from tick files...")
        try:
            df = extract_trade_clustering_features(
                df,
                ticks_loader_json=ticks_loader_json,
                window_size=trade_clustering_window,
                freq=freq,
            )
        except Exception as e:
            print(f"   ⚠️  Trade clustering feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
    return df


def compute_trade_clustering_from_ticks(
    ticks: pd.DataFrame,
    window_size: int = 100,
    initial_state: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    计算交易聚集性（Trade Clustering）特征（支持流式处理）
    Trade clustering 是指连续同向成交的聚集性（如连续 10 笔都是 buy）。
    与 VPIN 互补：VPIN 关注 volume-bucketed 的净买卖差，不关心成交顺序；
    Trade clustering 关注成交的时序模式，捕捉连续同向交易的聚集性。
    Args:
        ticks: DataFrame with tick data, must contain:
            - timestamp (datetime index)
            - side (1 for buy, -1 for sell)
            - volume (float, optional, for weighted clustering)
        window_size: 滚动窗口大小（用于计算统计量）
        initial_state: 初始状态（用于跨批次连续性），包含：
            - current_run_side: 当前 run 的方向
            - current_run_length: 当前 run 的长度
            - window_runs: 窗口内的 runs（deque of (side, length) tuples）
            - window_total_ticks: 窗口内总 tick 数
            - buy_runs_in_window: 窗口内所有 buy run 的长度（deque）
            - sell_runs_in_window: 窗口内所有 sell run 的长度（deque）
    Returns:
        tuple: (DataFrame with trade clustering features, final_state)
        - DataFrame indexed by timestamp
        - final_state: 最终状态（可用于下一批次）
    """
    empty_result = pd.DataFrame(columns=[
        "trade_cluster_max_buy_run",
        "trade_cluster_max_sell_run",
        "trade_cluster_avg_buy_run",
        "trade_cluster_avg_sell_run",
        "trade_cluster_buy_run_count",
        "trade_cluster_sell_run_count",
        "trade_cluster_imbalance_ratio",
        "trade_cluster_directional_entropy",
    ], dtype=float)
    
    if len(ticks) == 0:
        # 如果没有数据，返回空结果和当前状态（或初始状态）
        final_state = initial_state.copy() if initial_state else {
            "current_run_side": None,
            "current_run_length": 0,
            "window_runs": deque(),
            "window_total_ticks": 0,
            "buy_runs_in_window": deque(),
            "sell_runs_in_window": deque(),
        }
        return empty_result, final_state
    
    # 确保按时间排序
    if not isinstance(ticks.index, pd.DatetimeIndex):
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp").sort_index()
        else:
            raise ValueError("ticks must have DatetimeIndex or 'timestamp' column")
    else:
        ticks = ticks.sort_index()
    
    # 过滤无效的 side 值
    valid_side_mask = ticks["side"].isin([1, -1])
    if not valid_side_mask.all():
        ticks = ticks[valid_side_mask].copy()
    
    if len(ticks) == 0:
        final_state = initial_state.copy() if initial_state else {
            "current_run_side": None,
            "current_run_length": 0,
            "window_runs": deque(),
            "window_total_ticks": 0,
            "buy_runs_in_window": deque(),
            "sell_runs_in_window": deque(),
        }
        return empty_result, final_state
    
    sides = ticks["side"].values
    timestamps = ticks.index.values
    
    # 性能优化：使用增量更新方法，复杂度从 O(N × W) 降至 O(N)
    # 核心思想：维护一个滑动窗口的 run 列表，动态更新统计量
    cluster_features = []
    
    # 初始化状态（从 initial_state 或默认值）
    # 注意：initial_state 中的 deque 可能被序列化为 list，需要转换回 deque
    if initial_state:
        window_runs_data = initial_state.get("window_runs", [])
        window_runs = deque(window_runs_data) if not isinstance(window_runs_data, deque) else window_runs_data
        current_run_side = initial_state.get("current_run_side")
        current_run_length = initial_state.get("current_run_length", 0)
        window_total_ticks = initial_state.get("window_total_ticks", 0)
        buy_runs_data = initial_state.get("buy_runs_in_window", [])
        buy_runs_in_window = deque(buy_runs_data) if not isinstance(buy_runs_data, deque) else buy_runs_data
        sell_runs_data = initial_state.get("sell_runs_in_window", [])
        sell_runs_in_window = deque(sell_runs_data) if not isinstance(sell_runs_data, deque) else sell_runs_data
    else:
        window_runs = deque()  # 存储 (side, length) 元组，按时间顺序
        current_run_side = None  # 当前 run 的方向（窗口末尾的 run）
        current_run_length = 0   # 当前 run 的长度（窗口末尾的 run）
        window_total_ticks = 0   # 窗口内总 tick 数
        buy_runs_in_window = deque()  # 窗口内所有 buy run 的长度（按时间顺序）
        sell_runs_in_window = deque()  # 窗口内所有 sell run 的长度（按时间顺序）
    for i in range(len(ticks)):
        side = sides[i]
        # 更新当前 run（窗口末尾的 run）
        if side == current_run_side:
            # 与当前 run 同向，增加长度
            current_run_length += 1
        else:
            # 方向改变，结束当前 run，开始新 run
            if current_run_side is not None and current_run_length > 0:
                # 将结束的 run 加入窗口
                window_runs.append((current_run_side, current_run_length))
                window_total_ticks += current_run_length
                # 更新统计列表
                if current_run_side == 1:
                    buy_runs_in_window.append(current_run_length)
                else:
                    sell_runs_in_window.append(current_run_length)
            # 开始新 run
            current_run_side = side
            current_run_length = 1
        # 如果窗口超过大小，移除最旧的 run
        while window_total_ticks + current_run_length > window_size and len(window_runs) > 0:
            old_side, old_length = window_runs.popleft()
            window_total_ticks -= old_length
            # 从统计列表中移除（FIFO，所以直接 pop 即可）
            if old_side == 1:
                if buy_runs_in_window:
                    buy_runs_in_window.popleft()
            else:
                if sell_runs_in_window:
                    sell_runs_in_window.popleft()
        # 计算当前窗口的统计量（包含当前正在进行的 run）
        # 注意：当前 run 可能部分在窗口内（如果窗口已满）
        temp_buy_runs = list(buy_runs_in_window)
        temp_sell_runs = list(sell_runs_in_window)
        # 计算当前 run 在窗口内的部分
        remaining_window = window_size - window_total_ticks
        if remaining_window > 0 and current_run_length > 0:
            # 当前 run 在窗口内的长度
            run_in_window = min(current_run_length, remaining_window)
            if current_run_side == 1:
                temp_buy_runs.append(run_in_window)
            else:
                temp_sell_runs.append(run_in_window)
        # 计算统计量
        max_buy_run = max(temp_buy_runs) if temp_buy_runs else 0.0
        max_sell_run = max(temp_sell_runs) if temp_sell_runs else 0.0
        avg_buy_run = np.mean(temp_buy_runs) if temp_buy_runs else 0.0
        avg_sell_run = np.mean(temp_sell_runs) if temp_sell_runs else 0.0
        buy_run_count = len(temp_buy_runs)
        sell_run_count = len(temp_sell_runs)
        # 不平衡比率
        total_runs = buy_run_count + sell_run_count
        imbalance_ratio = (
            (buy_run_count - sell_run_count) / total_runs
            if total_runs > 0
            else 0.0
        )
        # 方向熵
        if total_runs > 0:
            buy_ratio = buy_run_count / total_runs
            sell_ratio = sell_run_count / total_runs
            if HAS_SCIPY and scipy_entropy is not None:
                entropy_val = scipy_entropy([buy_ratio, sell_ratio], base=2)
                directional_entropy = entropy_val
            else:
                if buy_ratio > 0 and sell_ratio > 0:
                    directional_entropy = -(
                        buy_ratio * np.log2(buy_ratio + TOL) +
                        sell_ratio * np.log2(sell_ratio + TOL)
                    )
                else:
                    directional_entropy = 0.0
        else:
            directional_entropy = 0.0
        cluster_features.append({
            "timestamp": timestamps[i],
            "max_buy_run": max_buy_run,
            "max_sell_run": max_sell_run,
            "avg_buy_run": avg_buy_run,
            "avg_sell_run": avg_sell_run,
            "buy_run_count": buy_run_count,
            "sell_run_count": sell_run_count,
            "imbalance_ratio": imbalance_ratio,
            "directional_entropy": directional_entropy,
        })
    # 转为 DataFrame
    cluster_df = pd.DataFrame(cluster_features)
    cluster_df = cluster_df.set_index("timestamp")
    # 重命名列
    cluster_df.columns = [
        "trade_cluster_max_buy_run",
        "trade_cluster_max_sell_run",
        "trade_cluster_avg_buy_run",
        "trade_cluster_avg_sell_run",
        "trade_cluster_buy_run_count",
        "trade_cluster_sell_run_count",
        "trade_cluster_imbalance_ratio",
        "trade_cluster_directional_entropy",
    ]
    
    # 返回最终状态（用于下一批次）
    final_state = {
        "current_run_side": current_run_side,
        "current_run_length": current_run_length,
        "window_runs": list(window_runs),  # 转为 list 以便序列化
        "window_total_ticks": window_total_ticks,
        "buy_runs_in_window": list(buy_runs_in_window),
        "sell_runs_in_window": list(sell_runs_in_window),
    }
    
    return cluster_df, final_state


def extract_trade_clustering_features(
    df: pd.DataFrame,
    ticks: Optional[pd.DataFrame] = None,
    ticks_loader_json: Optional[str] = None,
    window_size: int = 100,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    提取交易聚集性（Trade Clustering）特征并对齐到 K 线
    Args:
        df: DataFrame with OHLCV data (K线数据)
        ticks: Tick data for trade clustering calculation (必需)
        ticks_loader_json: JSON string for tick loader params (可选)
        window_size: 滚动窗口大小（用于计算统计量）
        freq: K线频率（如 '1T', '5T'），如果提供将跳过自动推断
    Returns:
        DataFrame with trade clustering features added
    Raises:
        ValueError: 如果没有提供 tick 数据或 tick 数据为空
    """
    df = df.copy()
    # 检查 tick 数据
    cluster_series = None
    if ticks is not None and len(ticks) > 0:
        required_tick_cols = ["side"]
        missing_cols = [col for col in required_tick_cols if col not in ticks.columns]
        if missing_cols:
            raise ValueError(
                f"Tick data must contain columns: {required_tick_cols}. "
                f"Missing columns: {missing_cols}"
            )
        print("   📊 Computing trade clustering from tick data (in-memory)...")
        cluster_df, _ = compute_trade_clustering_from_ticks(
            ticks,
            window_size=window_size,
        )
    elif ticks_loader_json:
        # 使用 tick loader 加载数据并计算 Trade Clustering
        # 优化：按月分批处理，避免一次性加载所有数据导致内存不足
        print("   📊 Computing trade clustering from tick files (monthly batches)...")
        loader_params = deserialize_tick_loader_params(ticks_loader_json)
        tick_files = loader_params.get("tick_files", [])
        if not tick_files:
            raise ValueError("No tick files provided in ticks_loader_json for trade clustering.")
        
        # 从 tick_files 推断 ticks_dir（取第一个文件的目录）
        import os
        if tick_files:
            first_file = tick_files[0]
            ticks_dir = os.path.dirname(first_file)
        else:
            ticks_dir = "data/parquet_data"  # 默认值
        
        # 优化内存使用：使用流式处理，每次只加载一个月的 tick 数据
        # 但为了保持 Trade Clustering 的连续性，需要维护一个滑动窗口状态
        start_ts = pd.to_datetime(loader_params["start_ts"])
        end_ts = pd.to_datetime(loader_params["end_ts"])
        lookback_minutes = loader_params.get("lookback_minutes", 60)
        
        # 生成月份范围
        current_month = (start_ts - pd.Timedelta(minutes=lookback_minutes)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month = end_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # 流式处理：按月计算 Trade Clustering，避免一次性加载所有数据
        # 维护跨月连续性状态，确保 Trade Clustering 计算的正确性
        cluster_results = []
        state = None  # 跨月连续性状态
        
        while current_month <= end_month:
            month_start = current_month
            month_end = (current_month + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)
            
            # 调整边界
            if current_month == (start_ts - pd.Timedelta(minutes=lookback_minutes)).replace(day=1, hour=0, minute=0, second=0, microsecond=0):
                month_start = start_ts - pd.Timedelta(minutes=lookback_minutes)
            if month_end > end_ts:
                month_end = end_ts
            
            # 加载该月的 tick 数据（只加载必要的列）
            try:
                month_ticks = load_tick_data(
                    symbol=loader_params["symbol"],
                    start_ts=month_start.isoformat(),
                    end_ts=month_end.isoformat(),
                    ticks_dir=ticks_dir,
                    lookback_minutes=0,
                )
                
                if month_ticks is not None and len(month_ticks) > 0:
                    # 只保留 side 列（Trade Clustering 只需要 side）
                    month_ticks = month_ticks[["side"]].copy()
                    print(f"      ✅ Loaded {month_start.strftime('%Y-%m')}: {len(month_ticks)} ticks")
                    
                    # 计算该月的 Trade Clustering（传入上个月的状态）
                    month_cluster_df, state = compute_trade_clustering_from_ticks(
                        month_ticks,
                        window_size=window_size,
                        initial_state=state,
                    )
                    
                    # 保存该月的结果
                    cluster_results.append(month_cluster_df)
                    print(f"      ✅ Computed {month_start.strftime('%Y-%m')}: {len(month_cluster_df)} features")
                    
                    # 立即释放该月的数据
                    del month_ticks, month_cluster_df
                    # 注意：state 中的 deque 已经被转换为 list（在 compute_trade_clustering_from_ticks 中）
                    # 下一批次使用时会在 compute_trade_clustering_from_ticks 中自动转换回 deque
            except Exception as e:
                print(f"      ⚠️  Failed to process {month_start.strftime('%Y-%m')}: {e}")
            
            # 移动到下一个月
            current_month = current_month + pd.DateOffset(months=1)
        
        if not cluster_results:
            raise ValueError("No trade clustering results computed.")
        
        # 合并所有月份的结果（只合并特征结果，不合并原始 tick 数据）
        print(f"      📊 Merging {len(cluster_results)} months of trade clustering results...")
        cluster_df = pd.concat(cluster_results, axis=0).sort_index()
        del cluster_results  # 释放内存
    else:
        raise ValueError(
            "Trade clustering calculation requires tick data. "
            "Please provide tick data via the 'ticks' parameter."
        )
    # 对齐到 df 的时间索引（右对齐，避免未来信息泄露）
    if isinstance(df.index, pd.DatetimeIndex):
        # 推断 df 的频率
        if freq is None:
            freq = pd.infer_freq(df.index)
            if freq is None:
                if len(df.index) > 1:
                    time_diff = df.index[1] - df.index[0]
                    freq_td = pd.Timedelta(time_diff)
                    # 尝试转换为标准频率字符串
                    for std_freq in ["1T", "5T", "15T", "30T", "1H", "4H", "1D"]:
                        if abs(pd.Timedelta(std_freq) - freq_td) < pd.Timedelta(seconds=1):
                            freq = std_freq
                            break
                    if freq is None:
                        freq_td = freq_td
                else:
                    freq = "1T"
                    freq_td = pd.Timedelta(minutes=1)
            else:
                freq_td = pd.Timedelta(freq) if isinstance(freq, str) else freq
        else:
            freq_td = pd.Timedelta(freq) if isinstance(freq, str) else freq
        # 严格右对齐的向量化实现
        aligned_features = {}
        for col in cluster_df.columns:
            aligned_series = None
            try:
                # 获取特征值
                feature_values = cluster_df[col].values
                feature_times = cluster_df.index.values
                # K 线时间边界
                kline_starts = df.index.values
                kline_ends = (df.index + freq_td).values
                # 严格右对齐：找到每个事件所属的 K 线
                pos = np.searchsorted(kline_starts, feature_times, side="right")
                idx = pos - 1
                # 验证：确保事件时间在 K 线窗口内
                valid_mask = (idx >= 0) & (idx < len(df)) & (feature_times < kline_ends[idx])
                if valid_mask.any():
                    valid_idx = idx[valid_mask]
                    valid_values = feature_values[valid_mask]
                    aligned_series = pd.Series(0.0, index=df.index, dtype=float)
                    # 按 K 线索引分组聚合（取均值）
                    feature_series = pd.Series(valid_values, index=valid_idx)
                    aggregated = feature_series.groupby(valid_idx).mean()
                    aligned_series.iloc[aggregated.index] = aggregated.values
                else:
                    aligned_series = pd.Series(0.0, index=df.index, dtype=float)
            except Exception as e:
                print(f"   ⚠️  Trade clustering alignment failed for {col} ({e})")
                aligned_series = pd.Series(0.0, index=df.index, dtype=float)
            aligned_features[col] = aligned_series
        # 添加到 df
        for col, series in aligned_features.items():
            df[col] = series
    else:
        # 如果 df 没有 datetime index，使用简单映射
        for col in cluster_df.columns:
            if col in cluster_df.columns:
                df[col] = cluster_df[col].reindex(df.index).fillna(0.0)
            else:
                df[col] = 0.0
    # 添加衍生特征
    if "trade_cluster_max_buy_run" in df.columns and "trade_cluster_max_sell_run" in df.columns:
        # 最大连续长度比率
        total_max = df["trade_cluster_max_buy_run"] + df["trade_cluster_max_sell_run"]
        df["trade_cluster_max_run_ratio"] = (
            (df["trade_cluster_max_buy_run"] - df["trade_cluster_max_sell_run"]) / (total_max + TOL)
        )
    if "trade_cluster_avg_buy_run" in df.columns and "trade_cluster_avg_sell_run" in df.columns:
        # 平均连续长度比率
        total_avg = df["trade_cluster_avg_buy_run"] + df["trade_cluster_avg_sell_run"]
        df["trade_cluster_avg_run_ratio"] = (
            (df["trade_cluster_avg_buy_run"] - df["trade_cluster_avg_sell_run"]) / (total_avg + TOL)
        )
    # 方向熵的衍生特征
    if "trade_cluster_directional_entropy" in df.columns:
        # 方向熵的移动平均（捕捉混乱度的趋势）
        for w in [5, 10, 20]:
            df[f"trade_cluster_directional_entropy_ma{w}"] = (
                df["trade_cluster_directional_entropy"].rolling(window=w, min_periods=1).mean()
            )
        # 方向熵的变化率（捕捉混乱度的变化）
        df["trade_cluster_directional_entropy_change"] = (
            df["trade_cluster_directional_entropy"].diff()
        )
        # 方向熵的 Z-score（识别异常混乱或异常聚集）
        for w in [20, 50]:
            rolling_mean = df["trade_cluster_directional_entropy"].rolling(window=w, min_periods=1).mean()
            rolling_std = df["trade_cluster_directional_entropy"].rolling(window=w, min_periods=1).std()
            df[f"trade_cluster_directional_entropy_zscore_{w}"] = (
                (df["trade_cluster_directional_entropy"] - rolling_mean) / (rolling_std + TOL)
            )
    # 滚动统计
    for w in [5, 10, 20]:
        if "trade_cluster_max_buy_run" in df.columns:
            df[f"trade_cluster_max_buy_run_ma{w}"] = (
                df["trade_cluster_max_buy_run"].rolling(window=w, min_periods=1).mean()
            )
        if "trade_cluster_imbalance_ratio" in df.columns:
            df[f"trade_cluster_imbalance_ratio_ma{w}"] = (
                df["trade_cluster_imbalance_ratio"].rolling(window=w, min_periods=1).mean()
            )
    return df
