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
EPS = 1e-9   # 通用极小量，避免分母为 0 产生 inf
MIN_BUCKET_VOLUME_TOL = 1e-9  # 桶体积填充容差（用于判断桶是否填满）

try:
    from scipy.stats import percentileofscore, entropy as scipy_entropy, skew as scipy_skew
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_entropy = None
    scipy_skew = None
    linregress = None

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
    bucket_volume_usd: Optional[float] = None,
) -> pd.DataFrame:
    """
    基于逐笔成交数据计算真实 VPIN（向量化实现）
    Args:
        ticks: DataFrame with tick data, must contain:
            - timestamp (datetime index)
            - price (float) - 必需，用于计算 USD 价值
            - volume (float)
            - side (1 for buy, -1 for sell, or 'buy'/'sell')
        bucket_volume: Fixed bucket volume (数量). If None and adaptive=True, will be calculated
        n_buckets: Number of buckets for rolling average
        lookback_days: Days to look back for adaptive bucket calculation
        quantile: Quantile for adaptive bucket volume (0.2-0.4 recommended)
        adaptive: If True, use adaptive bucket volume based on recent volume
        bucket_volume_usd: Bucket volume in USD (如果提供，使用USD价值计算，所有品种使用相同值)
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
    # 如果使用 USD bucket_volume，需要价格数据
    if bucket_volume_usd is not None:
        if "price" not in ticks.columns:
            raise ValueError("price column is required when using bucket_volume_usd")
        # USD 模式：使用 USD 价值而不是数量
        prices = ticks["price"].values
        volumes = ticks["volume"].values
        values_usd = prices * volumes  # 每个 tick 的 USD 价值
        target_bucket = bucket_volume_usd
    else:
        # 传统模式：使用数量
        if bucket_volume is None:
            bucket_volume = 100.0  # 默认值
        values_usd = None
        target_bucket = bucket_volume
    
    # 确保按时间排序
    if not isinstance(ticks.index, pd.DatetimeIndex):
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp").sort_index()
        else:
            raise ValueError("ticks must have DatetimeIndex or 'timestamp' column")
    else:
        ticks = ticks.sort_index()
    
    # 重新获取排序后的数据
    if bucket_volume_usd is not None:
        prices = ticks["price"].values
        volumes = ticks["volume"].values
        values_usd = prices * volumes
    else:
        volumes = ticks["volume"].values
    
    sides = ticks["side"].values
    timestamps = ticks.index.values
    
    # 计算累计值（USD 价值或数量）
    if bucket_volume_usd is not None:
        cumval = np.cumsum(values_usd)
        total_value = cumval[-1]
    else:
        cumval = np.cumsum(volumes)
        total_value = cumval[-1]
    
    if total_value < target_bucket:
        # 总价值不足一个桶
        return pd.DataFrame(columns=["vpin", "signed_imbalance"], dtype=float)
    
    # 生成桶边界（累计价值阈值）
    bucket_edges = np.arange(target_bucket, total_value + target_bucket, target_bucket)
    # 找到每个桶边界对应的 tick 索引
    # searchsorted 返回插入位置，即第一个 >= bucket_edge 的 cumval 位置
    bucket_tick_indices = np.searchsorted(cumval, bucket_edges, side="right")
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
                if bucket_volume_usd is not None:
                    # USD 模式：使用 USD 价值
                    tick_value = values_usd[j]
                else:
                    # 传统模式：使用数量
                    tick_value = volumes[j]
                
                if sides[j] == 1:
                    buy_vol += tick_value
                else:
                    sell_vol += tick_value
        
        # 处理跨越桶边界的最后一个 tick（如果有）
        # 计算桶还需要多少 value
        if bucket_end_idx > 0:
            cumval_at_end = cumval[bucket_end_idx - 1]
        else:
            cumval_at_end = 0.0
        remaining_to_fill = bucket_edge - cumval_at_end
        if remaining_to_fill > MIN_BUCKET_VOLUME_TOL and bucket_end_idx < len(ticks):
            # 需要从 bucket_end_idx 这个 tick 借用部分 value
            if bucket_volume_usd is not None:
                borrow_value = min(remaining_to_fill, values_usd[bucket_end_idx])
            else:
                borrow_value = min(remaining_to_fill, volumes[bucket_end_idx])
            
            if sides[bucket_end_idx] == 1:
                buy_vol += borrow_value
            else:
                sell_vol += borrow_value
        
        # 计算 VPIN 和 signed imbalance
        # 注意：buy_vol + sell_vol 应该等于 target_bucket（或接近）
        # 但由于浮点数精度和跨桶分割，可能略有差异
        total_vol_in_bucket = buy_vol + sell_vol
        if total_vol_in_bucket > 0:
            # 归一化到实际桶体积（处理浮点数误差）
            imbalance = abs(buy_vol - sell_vol)
            vpin_value = imbalance / total_vol_in_bucket  # 使用实际桶体积，而不是 target_bucket
            signed_imbalance = (buy_vol - sell_vol) / total_vol_in_bucket
        else:
            vpin_value = 0.0
            signed_imbalance = 0.0
        
        # 确保 VPIN 值在 [0, 1] 范围内（防止浮点数误差）
        vpin_value = min(vpin_value, 1.0)
        signed_imbalance = max(-1.0, min(1.0, signed_imbalance))
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
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    vpin_bucket_volume_usd: Optional[float] = None,
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
            bucket_volume_usd=vpin_bucket_volume_usd,
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
            monthly_cache_dir=monthly_cache_dir,
            bucket_volume_usd=vpin_bucket_volume_usd,
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
                # 如果仍无法匹配，使用计算出的 freq_td（freq 保持为 None）
                if freq is None:
                    # freq_td 已经在上面计算好了，直接使用
                    pass
            else:
                freq = "1min"  # fallback（使用新格式）
                freq_td = pd.Timedelta(minutes=1)
        else:
            # freq 不为 None，尝试解析
            if isinstance(freq, str):
                # 处理旧格式（向后兼容）
                if freq == "1T":
                    freq = "1min"
                elif freq == "1S":
                    freq = "1s"
                try:
                    freq_td = pd.Timedelta(freq)
                except ValueError:
                    # 如果解析失败，使用默认值
                    freq_td = pd.Timedelta(minutes=1)
            else:
                freq_td = freq if freq is not None else pd.Timedelta(minutes=1)
        # 方法1：严格右对齐的向量化实现（极快，O(N log M)）
        aligned_vpin = None
        aligned_signed = None
        # 初始化多维统计特征变量
        aligned_vpin_max = None
        aligned_vpin_min = None
        aligned_vpin_std = None
        aligned_vpin_last = None
        aligned_vpin_count = None
        aligned_signed_max = None
        aligned_signed_last = None
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
                valid_times = event_times[valid_mask]
                
                # 创建临时 DataFrame 用于多维统计
                temp_df = pd.DataFrame({
                    'kline_idx': valid_idx,
                    'vpin': valid_vpin,
                    'signed': valid_signed,
                    'timestamp': valid_times
                })
                
                # 按 K 线索引分组，计算多维统计特征
                # 关键改进：不仅计算均值，还保留峰值（max）、最新值（last）、波动性（std）、事件数（count）等信息
                # 使用 lambda 函数获取最后一个值（按时间戳排序）
                aligned_vpin = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_max = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_min = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_std = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_last = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_count = pd.Series(0, index=df.index, dtype=int)
                aligned_vpin_skew = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_trend = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed_max = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed_last = pd.Series(0.0, index=df.index, dtype=float)
                
                # 按 K 线分组计算统计量（valid_idx 是整数位置索引）
                grouped = temp_df.groupby('kline_idx')
                for kline_pos in grouped.groups.keys():
                    group_data = grouped.get_group(kline_pos)
                    
                    # 按时间排序获取最后一个值
                    group_sorted = group_data.sort_values('timestamp')
                    vpin_values = group_data['vpin'].values
                    
                    aligned_vpin.iloc[kline_pos] = group_data['vpin'].mean()
                    aligned_vpin_max.iloc[kline_pos] = group_data['vpin'].max()
                    aligned_vpin_min.iloc[kline_pos] = group_data['vpin'].min()
                    aligned_vpin_std.iloc[kline_pos] = group_data['vpin'].std() if len(group_data) > 1 else 0.0
                    aligned_vpin_last.iloc[kline_pos] = group_sorted['vpin'].iloc[-1]
                    aligned_vpin_count.iloc[kline_pos] = len(group_data)
                    
                    # 计算偏度（需要至少 3 个数据点）
                    if len(vpin_values) >= 3:
                        if HAS_SCIPY and scipy_skew is not None:
                            aligned_vpin_skew.iloc[kline_pos] = scipy_skew(vpin_values, nan_policy='omit')
                        else:
                            # 手动计算偏度：E[(X - μ)^3] / σ^3
                            mean_vpin = np.mean(vpin_values)
                            std_vpin = np.std(vpin_values)
                            if std_vpin > EPS:
                                skew_val = np.mean(((vpin_values - mean_vpin) / std_vpin) ** 3)
                                aligned_vpin_skew.iloc[kline_pos] = skew_val
                    
                    # 计算线性回归斜率（趋势）
                    if len(group_sorted) >= 2:
                        # 使用时间戳作为 x 轴（转换为数值）
                        timestamps = group_sorted['timestamp'].values
                        # 转换为相对于第一个时间戳的秒数
                        time_seconds = (timestamps - timestamps[0]).astype('timedelta64[s]').astype(float)
                        vpin_vals = group_sorted['vpin'].values
                        
                        if HAS_SCIPY and linregress is not None:
                            # 使用 scipy 的线性回归
                            slope, _, _, _, _ = linregress(time_seconds, vpin_vals)
                            aligned_vpin_trend.iloc[kline_pos] = slope
                        else:
                            # 手动计算斜率：最小二乘法
                            if len(time_seconds) > 1 and np.std(time_seconds) > EPS:
                                # slope = cov(x,y) / var(x)
                                cov_xy = np.cov(time_seconds, vpin_vals)[0, 1]
                                var_x = np.var(time_seconds)
                                if var_x > EPS:
                                    slope = cov_xy / var_x
                                    aligned_vpin_trend.iloc[kline_pos] = slope
                    
                    aligned_signed.iloc[kline_pos] = group_data['signed'].mean()
                    aligned_signed_max.iloc[kline_pos] = group_data['signed'].max()
                    aligned_signed_last.iloc[kline_pos] = group_sorted['signed'].iloc[-1]
            else:
                # 没有有效事件，初始化所有特征为 0
                aligned_vpin = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_max = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_min = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_std = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_last = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_count = pd.Series(0, index=df.index, dtype=int)
                aligned_vpin_skew = pd.Series(0.0, index=df.index, dtype=float)
                aligned_vpin_trend = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed_max = pd.Series(0.0, index=df.index, dtype=float)
                aligned_signed_last = pd.Series(0.0, index=df.index, dtype=float)
        except Exception as e:
            # 向量化方法失败，回退到循环方法
            print(f"   ⚠️  Vectorized alignment failed ({e}), falling back to loop method")
            aligned_vpin = None
            aligned_signed = None
            # 清除之前初始化的变量，让循环方法重新初始化
            aligned_vpin_max = None
            aligned_vpin_min = None
            aligned_vpin_std = None
            aligned_vpin_last = None
            aligned_vpin_count = None
            aligned_vpin_skew = None
            aligned_vpin_trend = None
            aligned_signed_max = None
            aligned_signed_last = None
        # 方法2：循环方法（兼容性，当向量化方法不可用时）
        if aligned_vpin is None:
            aligned_vpin = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_max = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_min = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_std = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_last = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_count = pd.Series(0, index=df.index, dtype=int)
            aligned_vpin_skew = pd.Series(0.0, index=df.index, dtype=float)
            aligned_vpin_trend = pd.Series(0.0, index=df.index, dtype=float)
            aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
            aligned_signed_max = pd.Series(0.0, index=df.index, dtype=float)
            aligned_signed_last = pd.Series(0.0, index=df.index, dtype=float)
            # 获取事件数据
            if isinstance(vpin_events, pd.DataFrame):
                event_vpin = vpin_events["vpin"]
                event_signed = vpin_events.get("signed_imbalance", pd.Series(0.0, index=vpin_events.index))
            else:
                event_vpin = vpin_events
                event_signed = pd.Series(0.0, index=vpin_events.index)
            # 确保 vpin_events.index 是 DatetimeIndex
            if not isinstance(vpin_events.index, pd.DatetimeIndex):
                if isinstance(vpin_events, pd.DataFrame):
                    vpin_events = vpin_events.copy()
                    if "timestamp" in vpin_events.columns:
                        vpin_events = vpin_events.set_index("timestamp")
                    else:
                        # 如果索引不是 datetime 且没有 timestamp 列，尝试转换
                        vpin_events.index = pd.to_datetime(vpin_events.index)
                else:
                    # Series
                    vpin_events.index = pd.to_datetime(vpin_events.index)
            
            for kline_time in df.index:
                window_end = kline_time + freq_td
                # 找到该 K 线时间段内的所有 VPIN 事件（右对齐：[kline_time, kline_time + freq)）
                window_mask = (vpin_events.index >= kline_time) & (
                    vpin_events.index < window_end
                )
                if window_mask.any():
                    vpin_window = event_vpin.loc[window_mask]
                    signed_window = event_signed.loc[window_mask]
                    vpin_values = vpin_window.values
                    vpin_times = vpin_window.index.values
                    
                    # 计算多维统计特征
                    aligned_vpin.loc[kline_time] = vpin_window.mean()
                    aligned_vpin_max.loc[kline_time] = vpin_window.max()
                    aligned_vpin_min.loc[kline_time] = vpin_window.min()
                    aligned_vpin_std.loc[kline_time] = vpin_window.std() if len(vpin_window) > 1 else 0.0
                    aligned_vpin_last.loc[kline_time] = vpin_window.iloc[-1]  # 最后一个事件（最新的）
                    aligned_vpin_count.loc[kline_time] = len(vpin_window)
                    
                    # 计算偏度（需要至少 3 个数据点）
                    if len(vpin_values) >= 3:
                        if HAS_SCIPY and scipy_skew is not None:
                            aligned_vpin_skew.loc[kline_time] = scipy_skew(vpin_values, nan_policy='omit')
                        else:
                            # 手动计算偏度
                            mean_vpin = np.mean(vpin_values)
                            std_vpin = np.std(vpin_values)
                            if std_vpin > EPS:
                                skew_val = np.mean(((vpin_values - mean_vpin) / std_vpin) ** 3)
                                aligned_vpin_skew.loc[kline_time] = skew_val
                    
                    # 计算线性回归斜率（趋势）
                    if len(vpin_times) >= 2:
                        # 转换为相对于第一个时间戳的秒数
                        time_seconds = (vpin_times - vpin_times[0]).astype('timedelta64[s]').astype(float)
                        if HAS_SCIPY and linregress is not None:
                            slope, _, _, _, _ = linregress(time_seconds, vpin_values)
                            aligned_vpin_trend.loc[kline_time] = slope
                        else:
                            # 手动计算斜率
                            if len(time_seconds) > 1 and np.std(time_seconds) > EPS:
                                cov_xy = np.cov(time_seconds, vpin_values)[0, 1]
                                var_x = np.var(time_seconds)
                                if var_x > EPS:
                                    slope = cov_xy / var_x
                                    aligned_vpin_trend.loc[kline_time] = slope
                    
                    aligned_signed.loc[kline_time] = signed_window.mean()
                    aligned_signed_max.loc[kline_time] = signed_window.max()
                    aligned_signed_last.loc[kline_time] = signed_window.iloc[-1]
                else:
                    # 没有事件，所有特征保持默认值 0
                    pass
        # 添加基础 VPIN 特征（均值，保持向后兼容）
        # vpin = Mean_VPIN_4H：该 4H 周期内所有 VPIN buckets 的均值（衡量这 4 小时整体的博弈强度）
        df["vpin"] = aligned_vpin
        
        # 添加多维 VPIN 统计特征（关键改进：保留峰值信息，避免均值稀释）
        # 这些特征能够保留 K 线周期内的峰值信号，而不是被均值稀释
        # 如果变量未定义（理论上不应该发生），使用默认值
        if aligned_vpin_last is None:
            aligned_vpin_last = aligned_vpin.copy()
        if aligned_vpin_max is None:
            aligned_vpin_max = aligned_vpin.copy()
        if aligned_vpin_min is None:
            aligned_vpin_min = aligned_vpin.copy()
        if aligned_vpin_std is None:
            aligned_vpin_std = pd.Series(0.0, index=df.index, dtype=float)
        if aligned_vpin_count is None:
            aligned_vpin_count = pd.Series(0, index=df.index, dtype=int)
        if aligned_vpin_skew is None:
            aligned_vpin_skew = pd.Series(0.0, index=df.index, dtype=float)
        if aligned_vpin_trend is None:
            aligned_vpin_trend = pd.Series(0.0, index=df.index, dtype=float)
            
        df["vpin_last"] = aligned_vpin_last  # 最新值（反映最新情绪）
        df["vpin_max"] = aligned_vpin_max  # 峰值（Max_VPIN_4H：该 4H 周期内 VPIN 的最大值）
        df["vpin_min"] = aligned_vpin_min  # 最小值
        df["vpin_std"] = aligned_vpin_std  # 波动性（衡量VPIN在K线内的变化）
        df["vpin_count"] = aligned_vpin_count  # 事件数（代理流动性，区分高VPIN是突发事件还是持续活跃）
        df["vpin_skewness"] = aligned_vpin_skew  # 偏度（VPIN_Skewness：如果偏度为正，说明尾部风险大）
        df["vpin_trend"] = aligned_vpin_trend  # 趋势（VPIN_Trend：斜率为正表示风险在积聚，斜率为负表示风险在释放）
        
        # 对齐 signed_imbalance（已在向量化或循环方法中处理）
        if aligned_signed is None:
            aligned_signed = pd.Series(0.0, index=df.index, dtype=float)
        df["vpin_signed_imbalance"] = aligned_signed
        if aligned_signed_last is None:
            aligned_signed_last = aligned_signed.copy()
        if aligned_signed_max is None:
            aligned_signed_max = aligned_signed.copy()
        df["vpin_signed_imbalance_last"] = aligned_signed_last
        df["vpin_signed_imbalance_max"] = aligned_signed_max
    else:
        # 如果 df 没有 datetime index，使用简单映射（不推荐，但保持兼容）
        vpin_series = vpin_series.reindex(df.index).fillna(0.0)
        df["vpin"] = vpin_series
    # VPIN 的滚动统计
    for w in [5, 10, 20]:
        df[f"vpin_ma{w}"] = df["vpin"].rolling(window=w, min_periods=1).mean()
        df[f"vpin_max{w}"] = df["vpin"].rolling(window=w, min_periods=1).max()
    # VPIN 变化率（捕捉订单流突增）
    vpin_base = df["vpin"].replace([np.inf, -np.inf], np.nan)
    df["vpin_change"] = vpin_base.diff()
    # 手动 pct_change，加 EPS 避免分母为 0
    prev = vpin_base.shift(1)
    df["vpin_change_pct"] = ((vpin_base - prev) / (prev + EPS)).replace([np.inf, -np.inf], np.nan)
    # 增强特征：Z-score（识别异常高的订单流不平衡）
    for w in [20, 50]:
        rolling_mean = df["vpin"].rolling(window=w, min_periods=1).mean()
        vpin_clean = df["vpin"].replace([np.inf, -np.inf], np.nan)
        rolling_std = vpin_clean.rolling(window=w, min_periods=1).std()
        z = (vpin_clean - rolling_mean) / (rolling_std + TOL)
        df[f"vpin_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
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
        vpin_clean = df["vpin"].replace([np.inf, -np.inf], np.nan)
        vol = vpin_clean.rolling(window=w, min_periods=1).std()
        df[f"vpin_volatility_{w}"] = vol.replace([np.inf, -np.inf], np.nan)
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
            vsi_clean = df["vpin_signed_imbalance"].replace([np.inf, -np.inf], np.nan)
            rolling_mean = vsi_clean.rolling(window=w, min_periods=1).mean()
            rolling_std = vsi_clean.rolling(window=w, min_periods=1).std()
            z = (vsi_clean - rolling_mean) / (rolling_std + TOL)
            df[f"vpin_signed_imbalance_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
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
                monthly_cache_dir=monthly_cache_dir,
                merge_batch_size=2,
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
                monthly_cache_dir=monthly_cache_dir,
                merge_batch_size=2,
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
        window_size (int): 滚动窗口大小，单位为 tick 笔数（非时间）
            例如：window_size=100 表示最近 100 笔成交
            
            ⚠️  注意：该设计在低流动性时段可能导致窗口时间跨度极大
            - 低流动性时段：100 笔可能跨越数小时甚至数天
            - 高流动性时段：100 笔可能仅几毫秒
            - 这会导致特征尺度不稳定，难以跨时间/品种比较
            
            未来改进方向：
            - 添加 window_type 参数：`"ticks"` 或 `"time"`
            - 如果 window_type="time"，使用 window_seconds 参数（如 3600 秒）
            - 保持向后兼容：默认 window_type="ticks"
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
    # 注意：initial_state 中的 deque 可能被序列化为 list（如从缓存加载），需要转换回 deque
    # 统一在入口转换，确保后续代码可以安全使用 deque 的方法（如 .popleft()）
    if initial_state:
        window_runs_data = initial_state.get("window_runs", [])
        if isinstance(window_runs_data, list):
            window_runs = deque(window_runs_data)
        elif isinstance(window_runs_data, deque):
            window_runs = window_runs_data
        else:
            window_runs = deque()
        
        current_run_side = initial_state.get("current_run_side")
        current_run_length = initial_state.get("current_run_length", 0)
        window_total_ticks = initial_state.get("window_total_ticks", 0)
        
        buy_runs_data = initial_state.get("buy_runs_in_window", [])
        if isinstance(buy_runs_data, list):
            buy_runs_in_window = deque(buy_runs_data)
        elif isinstance(buy_runs_data, deque):
            buy_runs_in_window = buy_runs_data
        else:
            buy_runs_in_window = deque()
        
        sell_runs_data = initial_state.get("sell_runs_in_window", [])
        if isinstance(sell_runs_data, list):
            sell_runs_in_window = deque(sell_runs_data)
        elif isinstance(sell_runs_data, deque):
            sell_runs_in_window = sell_runs_data
        else:
            sell_runs_in_window = deque()
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
        # 清理 temp_buy_runs 和 temp_sell_runs 中的 inf/NaN 值
        temp_buy_runs_clean = [x for x in temp_buy_runs if np.isfinite(x) and x >= 0]
        temp_sell_runs_clean = [x for x in temp_sell_runs if np.isfinite(x) and x >= 0]
        max_buy_run = max(temp_buy_runs_clean) if temp_buy_runs_clean else 0.0
        max_sell_run = max(temp_sell_runs_clean) if temp_sell_runs_clean else 0.0
        avg_buy_run = np.mean(temp_buy_runs_clean) if temp_buy_runs_clean else 0.0
        avg_sell_run = np.mean(temp_sell_runs_clean) if temp_sell_runs_clean else 0.0
        # 确保结果是有限值
        max_buy_run = max_buy_run if np.isfinite(max_buy_run) else 0.0
        max_sell_run = max_sell_run if np.isfinite(max_sell_run) else 0.0
        avg_buy_run = avg_buy_run if np.isfinite(avg_buy_run) else 0.0
        avg_sell_run = avg_sell_run if np.isfinite(avg_sell_run) else 0.0
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
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    merge_batch_size: int = 4,
    persist_monthly: bool = True,
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
        from pathlib import Path
        from src.data_tools.tick_loader import (
            _get_monthly_trade_clustering_cache_key,
            _load_monthly_trade_clustering_cache,
            _save_monthly_trade_clustering_cache,
        )
        
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
        
        # 按月缓存目录
        cache_dir = Path(monthly_cache_dir) if monthly_cache_dir else None
        
        # 生成月份范围
        current_month = (start_ts - pd.Timedelta(minutes=lookback_minutes)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month = end_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # 流式处理：按月计算 Trade Clustering，避免一次性加载所有数据
        # 维护跨月连续性状态，确保 Trade Clustering 计算的正确性
        cluster_results: list[pd.DataFrame] = []
        cluster_paths: list[Path] = []  # 按月落盘以降低内存
        cluster_df_accum: Optional[pd.DataFrame] = None
        state = None  # 跨月连续性状态
        total_files = len(tick_files)
        cached_count = 0
        computed_count = 0
        
        # 按月份匹配 tick 文件
        month_to_files = {}
        for file_path in tick_files:
            path = Path(file_path)
            # 从文件名提取月份（假设格式为 SYMBOL_YYYY-MM.parquet）
            if "_" in path.stem:
                parts = path.stem.split("_")
                if len(parts) >= 2:
                    month_str = parts[-1]  # 如 "2025-01"
                    try:
                        month_ts = pd.to_datetime(month_str)
                        month_key = month_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        if month_key not in month_to_files:
                            month_to_files[month_key] = []
                        month_to_files[month_key].append(file_path)
                    except:
                        pass
        
        while current_month <= end_month:
            month_start = current_month
            month_end = (current_month + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)
            
            # 调整边界
            if current_month == (start_ts - pd.Timedelta(minutes=lookback_minutes)).replace(day=1, hour=0, minute=0, second=0, microsecond=0):
                month_start = start_ts - pd.Timedelta(minutes=lookback_minutes)
            if month_end > end_ts:
                month_end = end_ts
            
            # 优化：类似 VPIN，如果 state 为 None，尝试从前一个月加载状态
            # 找到该月的 tick 文件
            month_files = month_to_files.get(current_month, [])
            
            # 如果 state 为 None，尝试从前一个月加载状态
            if state is None and current_month > (start_ts - pd.Timedelta(minutes=lookback_minutes)).replace(day=1, hour=0, minute=0, second=0, microsecond=0):
                prev_month = current_month - pd.DateOffset(months=1)
                prev_month_files = month_to_files.get(prev_month, [])
                
                if prev_month_files and cache_dir:
                    # 尝试从缓存加载前一个月的 final_state
                    prev_cache_key = _get_monthly_trade_clustering_cache_key(
                        prev_month_files[0], window_size, initial_state=None
                    )
                    prev_cached_result = _load_monthly_trade_clustering_cache(cache_dir, prev_cache_key)
                    if prev_cached_result is not None:
                        _, prev_state = prev_cached_result
                        if prev_state is not None:
                            state = prev_state
                            print(f"      📥 Loaded prev_month state for {current_month.strftime('%Y-%m')}")
            
            # 生成缓存键（包含 state 信息，如果存在）
            standard_cache_key = None
            state_cache_key = None
            if cache_dir and month_files:
                standard_cache_key = _get_monthly_trade_clustering_cache_key(
                    month_files[0], window_size, initial_state=None
                )
                if state is not None:
                    state_cache_key = _get_monthly_trade_clustering_cache_key(
                        month_files[0], window_size, initial_state=state
                    )
            
            # 尝试从缓存加载
            cached_result = None
            cache_key_used = None
            
            if state is not None and state_cache_key is not None:
                # 如果 state 不为空，先尝试使用状态缓存
                cached_result = _load_monthly_trade_clustering_cache(cache_dir, state_cache_key)
                if cached_result is not None:
                    cache_key_used = state_cache_key
            
            if cached_result is None and standard_cache_key is not None:
                # 如果状态缓存未命中，尝试使用标准缓存
                cached_result = _load_monthly_trade_clustering_cache(cache_dir, standard_cache_key)
                if cached_result is not None:
                    cache_key_used = standard_cache_key
            
            if cached_result is not None:
                # 使用缓存
                month_cluster_df, cached_state = cached_result
                
                if state is None:
                    # state 为空，使用标准缓存
                    if month_cluster_df is not None:
                        # 标准缓存包含 DataFrame，直接使用
                        cluster_results.append(month_cluster_df)
                        cached_count += 1
                        state = cached_state  # 更新 state 供下一个月使用
                        print(f"      ✅ Loaded {month_start.strftime('%Y-%m')} (cached): {len(month_cluster_df)} features")
                    else:
                        # 标准缓存只保存了 state，需要重新计算 DataFrame
                        print(f"      ✅ Computing {month_start.strftime('%Y-%m')} (cached state only, recomputing DataFrame)...")
                        state = cached_state  # 使用缓存的 state
                        cached_result = None  # 继续到计算逻辑
                else:
                    # state 不为空
                    if cache_key_used == state_cache_key:
                        # 状态缓存命中，直接使用
                        if month_cluster_df is not None:
                            cluster_results.append(month_cluster_df)
                            cached_count += 1
                            state = cached_state
                            print(f"      ✅ Loaded {month_start.strftime('%Y-%m')} (cached, with state): {len(month_cluster_df)} features")
                        else:
                            # 状态缓存只保存了 state，需要重新计算 DataFrame
                            print(f"      ✅ Computing {month_start.strftime('%Y-%m')} (cached state only, recomputing DataFrame)...")
                            state = cached_state  # 使用缓存的 state
                            cached_result = None  # 继续到计算逻辑
                    else:
                        # 使用了标准缓存，但 state 不为空，需要重新计算
                        print(f"      ✅ Computing {month_start.strftime('%Y-%m')} (state mismatch, recomputing)...")
                        state = cached_state  # 使用缓存的 state（但需要重新计算 DataFrame）
                        cached_result = None  # 继续到计算逻辑
            
            if cached_result is None:
                # 计算该月
                try:
                    # 转换时间格式（load_tick_data 期望 "YYYY-MM-DD HH:MM:SS" 格式）
                    start_ts_str = month_start.strftime("%Y-%m-%d %H:%M:%S")
                    end_ts_str = month_end.strftime("%Y-%m-%d %H:%M:%S")
                    month_ticks = load_tick_data(
                        symbol=loader_params["symbol"],
                        start_ts=start_ts_str,
                        end_ts=end_ts_str,
                        ticks_dir=ticks_dir,
                        lookback_minutes=0,
                    )
                    
                    if month_ticks is not None and len(month_ticks) > 0:
                        # 确保索引是 DatetimeIndex（load_tick_data 应该已经设置了）
                        if not isinstance(month_ticks.index, pd.DatetimeIndex):
                            if "timestamp" in month_ticks.columns:
                                month_ticks = month_ticks.set_index("timestamp")
                            else:
                                raise ValueError(f"Tick data must have DatetimeIndex or 'timestamp' column")
                        # 只保留 side 列（Trade Clustering 只需要 side，但需要保留索引）
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
                        computed_count += 1
                        print(f"      ✅ Computed {month_start.strftime('%Y-%m')}: {len(month_cluster_df)} features")
                        
                        # 保存缓存
                        # 标准缓存：只保存 final_state（优化：不保存 DataFrame，节省存储空间）
                        if cache_dir and month_files:
                            if standard_cache_key is not None:
                                # 只保存 state（标准缓存）
                                _save_monthly_trade_clustering_cache(
                                    cache_dir, standard_cache_key, (None, state)  # DataFrame 为 None，只保存 state
                                )
                            # 状态缓存：保存完整结果（DataFrame + state）
                            if state_cache_key is not None and state_cache_key != standard_cache_key:
                                _save_monthly_trade_clustering_cache(
                                    cache_dir, state_cache_key, (month_cluster_df, state)
                                )
                        
                        # 立即释放该月的数据
                        del month_ticks
                except Exception as e:
                    print(f"      ⚠️  Failed to process {month_start.strftime('%Y-%m')}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 移动到下一个月
            current_month = current_month + pd.DateOffset(months=1)
            
            # 批次合并，降低一次性 concat 的内存峰值
            if merge_batch_size and len(cluster_results) >= merge_batch_size:
                print(f"      📊 Merging a batch of {len(cluster_results)} months of trade clustering results...")
                if persist_monthly and cache_dir:
                    # 将本批次先落盘为 parquet，再清空内存
                    for df_month in cluster_results:
                        month_start_ts = df_month.index.min()
                        month_str = month_start_ts.strftime("%Y-%m") if pd.notna(month_start_ts) else "unknown"
                        file_path = cache_dir / f"trade_cluster_{month_str}_ws{window_size}.parquet"
                        df_month.to_parquet(file_path)
                        cluster_paths.append(file_path)
                    cluster_results.clear()
                else:
                    if cluster_df_accum is None:
                        cluster_df_accum = pd.concat(cluster_results, axis=0).sort_index()
                    else:
                        cluster_df_accum = (
                            pd.concat([cluster_df_accum] + cluster_results, axis=0)
                            .sort_index()
                        )
                    cluster_results.clear()
        
        # 合并剩余的批次（只合并特征结果，不合并原始 tick 数据）
        if cluster_results:
            print(f"      📊 Merging remaining {len(cluster_results)} months of trade clustering results...")
            if persist_monthly and cache_dir:
                for df_month in cluster_results:
                    month_start_ts = df_month.index.min()
                    month_str = month_start_ts.strftime("%Y-%m") if pd.notna(month_start_ts) else "unknown"
                    file_path = cache_dir / f"trade_cluster_{month_str}_ws{window_size}.parquet"
                    df_month.to_parquet(file_path)
                    cluster_paths.append(file_path)
                cluster_results.clear()
            else:
                if cluster_df_accum is None:
                    cluster_df_accum = pd.concat(cluster_results, axis=0).sort_index()
                else:
                    cluster_df_accum = (
                        pd.concat([cluster_df_accum] + cluster_results, axis=0)
                        .sort_index()
                    )
                cluster_results.clear()

        # 如已落盘，则分批读回并拼接，控制内存峰值
        if persist_monthly and cluster_paths:
            print(f"      📊 Loading and merging {len(cluster_paths)} persisted months of trade clustering results...")
            cluster_paths = sorted(cluster_paths)
            step = merge_batch_size or 1
            for i in range(0, len(cluster_paths), step):
                batch_files = cluster_paths[i : i + step]
                batch_dfs = [pd.read_parquet(p) for p in batch_files]
                batch_df = pd.concat(batch_dfs, axis=0).sort_index()
                if cluster_df_accum is None:
                    cluster_df_accum = batch_df
                else:
                    cluster_df_accum = (
                        pd.concat([cluster_df_accum, batch_df], axis=0)
                        .sort_index()
                    )
                del batch_dfs, batch_df

        # 如果既无在内存的累积结果，又没有任何落盘文件，记录警告并返回原 df
        if cluster_df_accum is None and not cluster_paths:
            print("   ⚠️  Trade clustering produced no results; skipping feature merge.")
            return df

        cluster_df = cluster_df_accum if cluster_df_accum is not None else pd.DataFrame()
        
        # 打印 Trade Clustering 数据统计，用于调试
        if cluster_df is not None and len(cluster_df) > 0:
            print(f"   📊 Trade Clustering raw data: {len(cluster_df)} events")
            print(f"      Time range: {cluster_df.index.min()} to {cluster_df.index.max()}")
            # 检查是否有有效数据
            valid_cols = [col for col in cluster_df.columns if cluster_df[col].notna().any()]
            print(f"      Valid columns: {len(valid_cols)}/{len(cluster_df.columns)}")
            if len(valid_cols) < len(cluster_df.columns):
                nan_cols = [col for col in cluster_df.columns if col not in valid_cols]
                print(f"      ⚠️  All-NaN columns: {nan_cols[:5]}..." if len(nan_cols) > 5 else f"      ⚠️  All-NaN columns: {nan_cols}")
        else:
            print(f"   ⚠️  Trade Clustering: No data computed")
        
        if cache_dir and cached_count > 0:
            print(
                f"   💾 Used {cached_count} cached months, computed {computed_count} new months",
                flush=True,
            )
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
        
        # 打印对齐前的统计信息
        print(f"   📊 Trade Clustering alignment:")
        print(f"      Cluster events: {len(cluster_df)}")
        print(f"      K-line bars: {len(df)}")
        print(f"      K-line time range: {df.index.min()} to {df.index.max()}")
        print(f"      Cluster time range: {cluster_df.index.min()} to {cluster_df.index.max()}")
        
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
                    
                    # 统计对齐结果
                    non_zero_count = (aligned_series != 0.0).sum()
                    print(f"      {col}: {non_zero_count}/{len(df)} bars have values")
                else:
                    aligned_series = pd.Series(0.0, index=df.index, dtype=float)
                    print(f"      ⚠️  {col}: No valid alignment (time range mismatch?)")
            except Exception as e:
                print(f"   ⚠️  Trade clustering alignment failed for {col} ({e})")
                import traceback
                traceback.print_exc()
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
        # 最大连续长度（buy 或 sell 中的较大值）
        df["trade_cluster_max_run"] = df[["trade_cluster_max_buy_run", "trade_cluster_max_sell_run"]].max(axis=1)
        # 买方 vs 卖方最大连续长度比
        # 检查输入数据是否包含 inf/NaN
        max_buy_clean = df["trade_cluster_max_buy_run"].replace([np.inf, -np.inf], np.nan)
        max_sell_clean = df["trade_cluster_max_sell_run"].replace([np.inf, -np.inf], np.nan)
        df["trade_cluster_buy_sell_max_ratio"] = (
            max_buy_clean / (max_sell_clean + TOL)
        ).replace([np.inf, -np.inf], np.nan)
    if "trade_cluster_avg_buy_run" in df.columns and "trade_cluster_avg_sell_run" in df.columns:
        # 平均连续长度比率
        total_avg = df["trade_cluster_avg_buy_run"] + df["trade_cluster_avg_sell_run"]
        df["trade_cluster_avg_run_ratio"] = (
            (df["trade_cluster_avg_buy_run"] - df["trade_cluster_avg_sell_run"]) / (total_avg + TOL)
        )
        # 买方 vs 卖方平均连续长度比
        # 检查输入数据是否包含 inf/NaN
        avg_buy_clean = df["trade_cluster_avg_buy_run"].replace([np.inf, -np.inf], np.nan)
        avg_sell_clean = df["trade_cluster_avg_sell_run"].replace([np.inf, -np.inf], np.nan)
        df["trade_cluster_buy_sell_avg_ratio"] = (
            avg_buy_clean / (avg_sell_clean + TOL)
        ).replace([np.inf, -np.inf], np.nan)
        # 总连续长度（buy + sell）
        if "trade_cluster_buy_run_count" in df.columns and "trade_cluster_sell_run_count" in df.columns:
            df["trade_cluster_total_run_length"] = (
                df["trade_cluster_avg_buy_run"] * df["trade_cluster_buy_run_count"] +
                df["trade_cluster_avg_sell_run"] * df["trade_cluster_sell_run_count"]
            )
            # 平均连续长度（所有 runs）
            if "trade_cluster_buy_run_count" in df.columns and "trade_cluster_sell_run_count" in df.columns:
                total_runs = df["trade_cluster_buy_run_count"] + df["trade_cluster_sell_run_count"]
                df["trade_cluster_avg_run_length"] = (
                    df["trade_cluster_total_run_length"] / (total_runs + TOL)
                )
    # 净 Runs 特征
    if "trade_cluster_buy_run_count" in df.columns and "trade_cluster_sell_run_count" in df.columns:
        # 净 runs（buy - sell）
        df["trade_cluster_net_runs"] = (
            df["trade_cluster_buy_run_count"] - df["trade_cluster_sell_run_count"]
        )
        # 总 runs 数（衡量活跃度）
        df["trade_cluster_total_runs"] = (
            df["trade_cluster_buy_run_count"] + df["trade_cluster_sell_run_count"]
        )
        # 净 runs 比率（标准化）
        df["trade_cluster_net_runs_ratio"] = (
            df["trade_cluster_net_runs"] / (df["trade_cluster_total_runs"] + TOL)
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
        if "trade_cluster_directional_entropy" not in df.columns:
            continue
        # 先清理 inf 值，避免 rolling_std 产生 inf
        entropy_clean = df["trade_cluster_directional_entropy"].replace([np.inf, -np.inf], np.nan)
        rolling_mean = entropy_clean.rolling(window=w, min_periods=1).mean()
        rolling_std = entropy_clean.rolling(window=w, min_periods=1).std()
        # 检查 rolling_std 是否包含 inf（可能由输入数据中的 inf 导致）
        if (~np.isfinite(rolling_std)).any():
            print(f"   ⚠️  trade_cluster_directional_entropy rolling_std contains inf, replacing with NaN")
            rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
        z = (entropy_clean - rolling_mean) / (rolling_std + TOL)
        df[f"trade_cluster_directional_entropy_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
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
        if "trade_cluster_net_runs" in df.columns:
            df[f"trade_cluster_net_runs_ma{w}"] = (
                df["trade_cluster_net_runs"].rolling(window=w, min_periods=1).mean()
            )
        if "trade_cluster_total_runs" in df.columns:
            df[f"trade_cluster_total_runs_ma{w}"] = (
                df["trade_cluster_total_runs"].rolling(window=w, min_periods=1).mean()
            )
    
    # Z-score 特征（识别超买/超卖）
    if "trade_cluster_imbalance_ratio" in df.columns:
        for w in [20, 50]:
            # 先清理 inf 值，避免 rolling_std 产生 inf
            ratio_clean = df["trade_cluster_imbalance_ratio"].replace([np.inf, -np.inf], np.nan)
            rolling_mean = ratio_clean.rolling(window=w, min_periods=1).mean()
            rolling_std = ratio_clean.rolling(window=w, min_periods=1).std()
            # 检查 rolling_std 是否包含 inf
            if (~np.isfinite(rolling_std)).any():
                rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
            z = (ratio_clean - rolling_mean) / (rolling_std + TOL)
            df[f"trade_cluster_imbalance_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
    if "trade_cluster_net_runs" in df.columns:
        for w in [20, 50]:
            # 先清理 inf 值，避免 rolling_std 产生 inf
            net_runs_clean = df["trade_cluster_net_runs"].replace([np.inf, -np.inf], np.nan)
            rolling_mean = net_runs_clean.rolling(window=w, min_periods=1).mean()
            rolling_std = net_runs_clean.rolling(window=w, min_periods=1).std()
            # 检查 rolling_std 是否包含 inf
            if (~np.isfinite(rolling_std)).any():
                rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
            z = (net_runs_clean - rolling_mean) / (rolling_std + TOL)
            df[f"trade_cluster_net_runs_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
    if "trade_cluster_max_buy_run" in df.columns:
        for w in [20, 50]:
            # 先清理 inf 值，避免 rolling_std 产生 inf
            max_buy_clean = df["trade_cluster_max_buy_run"].replace([np.inf, -np.inf], np.nan)
            rolling_mean = max_buy_clean.rolling(window=w, min_periods=1).mean()
            rolling_std = max_buy_clean.rolling(window=w, min_periods=1).std()
            # 检查 rolling_std 是否包含 inf
            if (~np.isfinite(rolling_std)).any():
                rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
            z = (max_buy_clean - rolling_mean) / (rolling_std + TOL)
            df[f"trade_cluster_max_buy_run_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
    if "trade_cluster_max_sell_run" in df.columns:
        for w in [20, 50]:
            # 先清理 inf 值，避免 rolling_std 产生 inf
            max_sell_clean = df["trade_cluster_max_sell_run"].replace([np.inf, -np.inf], np.nan)
            rolling_mean = max_sell_clean.rolling(window=w, min_periods=1).mean()
            rolling_std = max_sell_clean.rolling(window=w, min_periods=1).std()
            # 检查 rolling_std 是否包含 inf
            if (~np.isfinite(rolling_std)).any():
                rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)
            z = (max_sell_clean - rolling_mean) / (rolling_std + TOL)
            df[f"trade_cluster_max_sell_run_zscore_{w}"] = z.replace([np.inf, -np.inf], np.nan)
    
    return df
