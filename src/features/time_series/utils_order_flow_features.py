"""
订单流特征：基于 tick 数据的订单流分析

核心特征：
1. VPIN (Volume-Synchronized Probability of Informed Trading) - 真实实现
2. 其他订单流衍生特征
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_vpin_from_ticks(
    ticks: pd.DataFrame,
    bucket_volume: Optional[float] = None,
    n_buckets: int = 50,
    lookback_days: int = 7,
    quantile: float = 0.3,
    adaptive: bool = True,
) -> pd.Series:
    """
    基于逐笔成交数据计算真实 VPIN
    
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
        Series with VPIN values (0-1 range), indexed by timestamp
    """
    if len(ticks) == 0:
        return pd.Series(dtype=float)
    
    # 标准化 side
    if "side" not in ticks.columns:
        raise ValueError("ticks must contain 'side' column (1/-1 or 'buy'/'sell')")
    
    ticks = ticks.copy()
    if ticks["side"].dtype == "object":
        ticks["side"] = ticks["side"].map({"buy": 1, "sell": -1, "BUY": 1, "SELL": -1})
    
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
        bucket_volume = max(bucket_volume, 1e-6)  # 防止为0
    
    if bucket_volume is None:
        bucket_volume = 100.0  # 默认值
    
    # 按成交量切片
    buckets = []
    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    filled_volume = 0.0
    
    # 确保按时间排序
    if not isinstance(ticks.index, pd.DatetimeIndex):
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp").sort_index()
        else:
            raise ValueError("ticks must have DatetimeIndex or 'timestamp' column")
    else:
        ticks = ticks.sort_index()
    
    for _, row in ticks.iterrows():
        vol = row["volume"]
        side = row["side"]
        
        # 分配到当前桶
        remaining_vol = vol
        while remaining_vol > 0:
            space_left = bucket_volume - filled_volume
            trade_in_bucket = min(remaining_vol, space_left)
            
            if side == 1:
                current_bucket_buy += trade_in_bucket
            else:
                current_bucket_sell += trade_in_bucket
            
            filled_volume += trade_in_bucket
            remaining_vol -= trade_in_bucket
            
            # 桶满了，保存并重置
            if filled_volume >= bucket_volume - 1e-9:
                imbalance = abs(current_bucket_buy - current_bucket_sell)
                vpin_value = imbalance / bucket_volume
                buckets.append({
                    "timestamp": row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                    "vpin": vpin_value,
                })
                
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                filled_volume = 0.0
    
    if len(buckets) == 0:
        return pd.Series(dtype=float)
    
    # 转为 DataFrame 并计算滚动平均
    buckets_df = pd.DataFrame(buckets)
    buckets_df = buckets_df.set_index("timestamp")
    
    # 滚动平均
    vpin_series = buckets_df["vpin"].rolling(window=n_buckets, min_periods=1).mean()
    
    return vpin_series


# 注意：compute_vpin_from_ohlcv 函数已移除
# VPIN 必须基于 tick 数据计算，不支持 proxy 实现
# 如果只有 OHLCV 数据，请使用 tick 数据或移除 VPIN 特征


def extract_order_flow_features(
    df: pd.DataFrame,
    ticks: Optional[pd.DataFrame] = None,
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
    if ticks is None or len(ticks) == 0:
        raise ValueError(
            "VPIN calculation requires tick data. "
            "Please provide tick data via the 'ticks' parameter. "
            "Proxy VPIN from OHLCV is not supported."
        )
    
    # 验证 tick 数据格式
    required_tick_cols = ["price", "volume", "side"]
    missing_cols = [col for col in required_tick_cols if col not in ticks.columns]
    if missing_cols:
        raise ValueError(
            f"Tick data must contain columns: {required_tick_cols}. "
            f"Missing columns: {missing_cols}"
        )
    
    print("   📊 Computing real VPIN from tick data...")
    # 真实 VPIN（基于 tick 数据）
    vpin_series = compute_vpin_from_ticks(
        ticks,
        bucket_volume=vpin_bucket_volume,
        n_buckets=vpin_n_buckets,
        adaptive=vpin_adaptive,
    )
    
    # 对齐到 df 的时间索引
    if isinstance(df.index, pd.DatetimeIndex):
        vpin_series = vpin_series.reindex(df.index, method="ffill").fillna(0.0)
    else:
        # 如果 df 没有 datetime index，尝试合并
        vpin_series = vpin_series.reindex(df.index).fillna(0.0)
    
    df["vpin"] = vpin_series
    
    # VPIN 的滚动统计
    for w in [5, 10, 20]:
        df[f"vpin_ma{w}"] = df["vpin"].rolling(window=w, min_periods=1).mean()
        df[f"vpin_max{w}"] = df["vpin"].rolling(window=w, min_periods=1).max()
    
    # VPIN 变化率（捕捉订单流突增）
    df["vpin_change"] = df["vpin"].diff()
    df["vpin_change_pct"] = df["vpin"].pct_change().fillna(0.0)
    
    return df

