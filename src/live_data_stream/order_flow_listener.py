"""
订单流监听器

使用 Nautilus Trader 监听订单流数据，实现：
1. 实时接收 TradeTick 事件
2. 按1分钟聚合tick数据
3. 每15分钟计算特征并保存
4. 每4小时聚合特征并保存
5. 支持从断线中恢复
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import pandas as pd
import numpy as np

try:
    from nautilus_trader.model import TradeTick, Bar
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.enums import BarAggregation, PriceType

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    TradeTick = None
    Bar = None
    AggressorSide = None
    BarType = None
    InstrumentId = None
    Symbol = None
    Price = None
    Quantity = None
    BarAggregation = None
    PriceType = None

from .feature_storage import StorageManager
from .memory_window import MemoryWindow
from .gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer


class OrderFlowListener:
    """
    订单流监听器
    
    功能：
    1. 监听 TradeTick 事件
    2. 按1分钟聚合tick数据
    3. 维护内存滑动窗口（默认4小时）
    4. 每15分钟计算特征并保存
    5. 每4小时聚合特征并保存
    6. 支持从断线中恢复
    """
    
    def __init__(
        self,
        symbol: str,
        storage_manager: StorageManager,
        feature_computer: Optional[IncrementalFeatureComputer] = None,
        gap_filler: Optional[GapFiller] = None,
        memory_window_hours: float = 4.0,
        feature_compute_interval_minutes: int = 15,
        feature_4h_interval_hours: int = 4,
        storage_base_path: str = "data/live_storage",
        on_bar_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_feature_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            storage_manager: 存储管理器
            feature_computer: 特征计算器（如果为None，会创建默认的）
            memory_window_hours: 内存滑动窗口时长（小时）
            feature_compute_interval_minutes: 特征计算间隔（分钟）
            feature_4h_interval_hours: 4小时特征保存间隔（小时）
            storage_base_path: 存储根目录
            on_bar_callback: 收到新bar时的回调函数
            on_feature_callback: 计算完特征时的回调函数
        """
        self.symbol = symbol
        self.storage_manager = storage_manager
        self.memory_window_hours = memory_window_hours
        self.feature_compute_interval_minutes = feature_compute_interval_minutes
        self.feature_4h_interval_hours = feature_4h_interval_hours
        
        # 特征计算器
        if feature_computer is None:
            self.feature_computer = IncrementalFeatureComputer(
                tick_window_minutes=int(memory_window_hours * 60),
                bar_window_size=int(memory_window_hours * 60),  # 假设1分钟bar
            )
        else:
            self.feature_computer = feature_computer
        
        # 数据补全器
        self.gap_filler = gap_filler
        
        # 内存滑动窗口
        self.memory_window = MemoryWindow(window_hours=memory_window_hours)
        
        # 回调函数
        self.on_bar_callback = on_bar_callback
        self.on_feature_callback = on_feature_callback
        
        # 1分钟聚合状态
        self.current_1min_bar: Optional[Dict[str, Any]] = None
        self.current_1min_start: Optional[pd.Timestamp] = None
        
        # 定时器状态
        self.last_feature_compute_time: Optional[pd.Timestamp] = None
        self.last_4h_save_time: Optional[pd.Timestamp] = None
        
        # 运行状态
        self.is_running = False
        self._stop_event: Optional[asyncio.Event] = None
    
    def on_trade_tick(self, tick: TradeTick | Any) -> None:
        """
        处理 TradeTick 事件
        
        Args:
            tick: Nautilus Trader TradeTick 对象或Mock对象
        """
        # 转换时间戳（支持多种格式）
        if hasattr(tick, 'ts_init'):
            # Nautilus Trader TradeTick使用ts_init（纳秒时间戳）
            tick_ts = pd.Timestamp(tick.ts_init, unit="ns", tz="UTC")
        elif hasattr(tick, 'ts_init_ns'):
            # Mock对象或其他格式
            tick_ts = pd.Timestamp(tick.ts_init_ns, unit="ns", tz="UTC")
        else:
            # 其他格式，尝试直接转换
            tick_ts = pd.Timestamp(getattr(tick, 'timestamp', pd.Timestamp.now()))
        
        # 计算当前1分钟bar的开始时间
        bar_start = tick_ts.floor("1min")
        
        # 如果是新的1分钟bar，完成上一个bar
        if self.current_1min_start is not None and bar_start > self.current_1min_start:
            self._finalize_1min_bar()
        
        # 获取价格和数量
        if hasattr(tick, 'price'):
            price = float(tick.price) if not isinstance(tick.price, (int, float)) else float(tick.price)
        else:
            price = float(getattr(tick, 'price', 0))
        
        if hasattr(tick, 'size'):
            size = float(tick.size) if not isinstance(tick.size, (int, float)) else float(tick.size)
        else:
            size = float(getattr(tick, 'size', 0))
        
        # 初始化或更新当前1分钟bar
        if self.current_1min_bar is None:
            self.current_1min_bar = {
                "timestamp": bar_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "trade_count": 0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "buy_count": 0,
                "sell_count": 0,
            }
            self.current_1min_start = bar_start
        
        # 更新当前bar
        self.current_1min_bar["high"] = max(self.current_1min_bar["high"], price)
        self.current_1min_bar["low"] = min(self.current_1min_bar["low"], price)
        self.current_1min_bar["close"] = price
        self.current_1min_bar["volume"] += size
        self.current_1min_bar["trade_count"] += 1
        
        # 判断买卖方向（支持多种格式）
        if hasattr(tick, 'aggressor_side'):
            aggressor_side = tick.aggressor_side
            if isinstance(aggressor_side, str):
                is_buy = aggressor_side in ("BUY", "BUYER") or (NAUTILUS_AVAILABLE and aggressor_side == AggressorSide.BUYER)
            else:
                is_buy = NAUTILUS_AVAILABLE and aggressor_side == AggressorSide.BUYER
        else:
            # 尝试从其他属性推断
            is_buy = getattr(tick, 'side', 1) == 1
        
        if is_buy:
            self.current_1min_bar["buy_volume"] += size
            self.current_1min_bar["buy_count"] += 1
        else:
            self.current_1min_bar["sell_volume"] += size
            self.current_1min_bar["sell_count"] += 1
        
        # 传递给特征计算器
        # incremental_feature_computer可以直接接受TradeTick对象，会自动转换
        # 如果是Mock对象，传递字典格式（使用ts字段，纳秒时间戳）
        if NAUTILUS_AVAILABLE and isinstance(tick, TradeTick):
            # 直接传递TradeTick对象，incremental_feature_computer会自动处理
            self.feature_computer.on_tick(tick)
        else:
            # Mock对象，传递字典格式
            side_value = 1 if is_buy else -1
            self.feature_computer.on_tick({
                "ts": tick_ts.value,  # 纳秒时间戳
                "price": price,
                "volume": size,  # 使用volume而不是size
                "side": side_value,  # 使用1/-1格式
            })
        
        # 定期保存未完成的bar（用于恢复）
        self._periodic_save_incomplete_bar()
    
    def _finalize_1min_bar(self) -> None:
        """完成当前1分钟bar"""
        if self.current_1min_bar is None:
            return
        
        # 计算订单流指标
        total_volume = self.current_1min_bar["volume"]
        if total_volume > 0:
            self.current_1min_bar["buy_ratio"] = self.current_1min_bar["buy_volume"] / total_volume
            self.current_1min_bar["sell_ratio"] = self.current_1min_bar["sell_volume"] / total_volume
            self.current_1min_bar["delta"] = (
                self.current_1min_bar["buy_volume"] - self.current_1min_bar["sell_volume"]
            )
        else:
            self.current_1min_bar["buy_ratio"] = 0.0
            self.current_1min_bar["sell_ratio"] = 0.0
            self.current_1min_bar["delta"] = 0.0
        
        # 转换为DataFrame并保存
        bar_df = pd.DataFrame([self.current_1min_bar])
        self.storage_manager.save_1min_ticks(
            self.symbol,
            bar_df,
            include_incomplete=False,  # 已完成的bar
        )
        
        # 添加到内存窗口
        self.memory_window.add(self.current_1min_bar.copy())
        
        # 传递给特征计算器（确保bar数据有ts字段，纳秒时间戳）
        bar_for_computer = self.current_1min_bar.copy()
        if "ts" not in bar_for_computer:
            # 添加ts字段（纳秒时间戳）
            bar_for_computer["ts"] = int(pd.Timestamp(bar_for_computer["timestamp"]).value)
        self.feature_computer.on_bar(bar_for_computer, timeframe="1min")
        
        # 回调
        if self.on_bar_callback:
            self.on_bar_callback(self.current_1min_bar)
        
        # 重置当前bar
        self.current_1min_bar = None
        self.current_1min_start = None
    
    def _periodic_save_incomplete_bar(self) -> None:
        """定期保存未完成的bar（每10秒）"""
        # 简化实现：每次tick都保存（实际可以优化为每10秒保存一次）
        if self.current_1min_bar is not None:
            bar_df = pd.DataFrame([self.current_1min_bar])
            self.storage_manager.save_1min_ticks(
                self.symbol,
                bar_df,
                include_incomplete=True,  # 未完成的bar
            )
    
    def _compute_and_save_15min_features(self) -> None:
        """计算并保存15分钟特征"""
        # 获取特征
        features = self.feature_computer.get_features()
        orderflow_features = self.feature_computer.get_orderflow_features(window_minutes=15)
        
        if not features and not orderflow_features:
            return
        
        # 合并特征
        all_features = {**(features or {}), **(orderflow_features or {})}
        
        # 添加时间戳
        now = pd.Timestamp.now(tz="UTC")
        all_features["timestamp"] = now
        
        # 转换为DataFrame
        features_df = pd.DataFrame([all_features])
        
        # 保存
        self.storage_manager.save_15min_features(self.symbol, features_df, now)
        
        # 回调
        if self.on_feature_callback:
            self.on_feature_callback(all_features)
    
    def _aggregate_and_save_4h_features(self) -> None:
        """聚合并保存4小时特征（从15分钟特征聚合）"""
        # 从Parquet加载最近4小时的15分钟特征
        now = pd.Timestamp.now(tz="UTC")
        start_time = now - timedelta(hours=4)
        
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        
        # 加载15分钟特征
        features_15min = self.storage_manager.feature_15min.load_range(
            self.symbol, start_date, end_date
        )
        
        if len(features_15min) == 0:
            # 如果没有15分钟特征，使用当前计算的特征
            features = self.feature_computer.get_features()
            orderflow_features = self.feature_computer.get_orderflow_features(window_minutes=240)
            
            if not features and not orderflow_features:
                return
            
            all_features = {**(features or {}), **(orderflow_features or {})}
            all_features["timestamp"] = now
            features_df = pd.DataFrame([all_features])
        else:
            # 过滤时间范围
            features_15min = features_15min[
                (features_15min["timestamp"] >= start_time) &
                (features_15min["timestamp"] <= now)
            ]
            
            if len(features_15min) == 0:
                return
            
            # 聚合15分钟特征到4小时（取平均值或最后值）
            # 这里简化实现：取最后一条特征
            last_features = features_15min.iloc[-1].to_dict()
            last_features["timestamp"] = now
            features_df = pd.DataFrame([last_features])
        
        # 保存
        self.storage_manager.save_4h_features(self.symbol, features_df, now)
    
    async def _periodic_tasks(self) -> None:
        """定期任务（特征计算和保存）"""
        while not self._stop_event.is_set():
            now = pd.Timestamp.now(tz="UTC")
            
            # 检查是否需要计算15分钟特征
            if (
                self.last_feature_compute_time is None
                or (now - self.last_feature_compute_time).total_seconds() >= self.feature_compute_interval_minutes * 60
            ):
                self._compute_and_save_15min_features()
                self.last_feature_compute_time = now
            
            # 检查是否需要保存4小时特征
            if (
                self.last_4h_save_time is None
                or (now - self.last_4h_save_time).total_seconds() >= self.feature_4h_interval_hours * 3600
            ):
                self._aggregate_and_save_4h_features()
                self.last_4h_save_time = now
            
            # 等待1分钟再检查
            await asyncio.sleep(60)
    
    def warmup(self, days: int = 30, use_gap_filler: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载warmup数据（支持从Feature Store和Parquet加载）
        
        Args:
            days: 加载最近N天的数据
            use_gap_filler: 是否使用GapFiller进行补数据
        
        Returns:
            包含三种数据的字典
        """
        # 如果使用GapFiller，优先从Feature Store加载
        if use_gap_filler and self.gap_filler:
            data = self.gap_filler.warmup(self.symbol, days=days)
        else:
            # 否则直接从存储管理器加载
            data = self.storage_manager.warmup_load(self.symbol, days=days)
        
        # 恢复状态
        self._restore_state(data)
        
        return data
    
    def _restore_state(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        恢复状态（特征计算器和内存窗口）
        
        Args:
            data: warmup数据字典
        """
        # 恢复特征计算器状态（从15分钟特征恢复）
        if len(data.get("features_15min", pd.DataFrame())) > 0:
            features_15min = data["features_15min"]
            # 获取最新的特征时间戳
            latest_ts = features_15min["timestamp"].max()
            self.last_feature_compute_time = pd.Timestamp(latest_ts)
        
        # 恢复4小时特征保存时间
        if len(data.get("features_4h", pd.DataFrame())) > 0:
            features_4h = data["features_4h"]
            latest_ts = features_4h["timestamp"].max()
            self.last_4h_save_time = pd.Timestamp(latest_ts)
        
        # 恢复内存窗口和特征计算器状态（从1分钟tick数据）
        if len(data.get("ticks_1min", pd.DataFrame())) > 0:
            ticks_1min = data["ticks_1min"]
            # 转换为字典列表
            bars = ticks_1min.to_dict("records")
            # 添加到内存窗口和特征计算器（重建状态）
            for bar in bars:
                self.memory_window.add(bar)
                # 传递给特征计算器（重建状态）
                self.feature_computer.on_bar(bar, timeframe="1min")
    
    def get_recovery_state(self) -> Dict[str, Any]:
        """获取恢复状态（用于从断线中恢复）"""
        return self.storage_manager.get_recovery_state(self.symbol)
    
    async def start(self) -> None:
        """启动监听器"""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event = asyncio.Event()
        
        # 启动定期任务
        asyncio.create_task(self._periodic_tasks())
    
    async def stop(self) -> None:
        """停止监听器"""
        if not self.is_running:
            return
        
        # 完成当前bar
        self._finalize_1min_bar()
        
        # 停止定期任务
        if self._stop_event:
            self._stop_event.set()
        
        self.is_running = False
    
    def get_memory_window(self) -> pd.DataFrame:
        """获取内存窗口数据（用于调试）"""
        return self.memory_window.to_dataframe()
    
    def recover_from_interruption(self) -> Dict[str, Any]:
        """
        从断线中恢复
        
        Returns:
            恢复状态信息
        """
        # 获取恢复状态
        recovery_state = self.get_recovery_state()
        
        # 如果有未完成的bar，恢复当前bar状态
        if recovery_state.get("incomplete_bar"):
            incomplete_bar = recovery_state["incomplete_bar"]
            self.current_1min_bar = incomplete_bar
            if "timestamp" in incomplete_bar:
                self.current_1min_start = pd.Timestamp(incomplete_bar["timestamp"])
        
        # 如果有数据缺失，使用GapFiller补数据
        if self.gap_filler and recovery_state.get("latest_1min_timestamp"):
            latest_ts = recovery_state["latest_1min_timestamp"]
            now = pd.Timestamp.now(tz="UTC")
            
            # 如果缺失超过1天，从币安API补数据
            if (now - latest_ts).total_seconds() > 86400:
                print(f"⚠️ 检测到数据缺失超过1天，开始补数据...")
                fill_data = self.gap_filler.fill_from_binance_api(
                    self.symbol,
                    latest_ts + timedelta(minutes=1),
                    now,
                    timeframe="1m",
                )
                
                if len(fill_data) > 0:
                    # 恢复内存窗口和特征计算器状态
                    bars = fill_data.to_dict("records")
                    for bar in bars:
                        self.memory_window.add(bar)
                        self.feature_computer.on_bar(bar, timeframe="1min")
                    
                    # 保存补全的数据
                    self.storage_manager.save_1min_ticks(
                        self.symbol,
                        fill_data,
                        include_incomplete=False,
                    )
        
        return recovery_state
