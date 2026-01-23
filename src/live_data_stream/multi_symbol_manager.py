"""
多Symbol管理器

管理多个OrderFlowListener实例，提供统一接口启动/停止、warmup、状态查询等。
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from .order_flow_listener import OrderFlowListener
from .feature_storage import StorageManager
from .gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class MultiSymbolManager:
    """
    多Symbol管理器
    
    管理多个OrderFlowListener实例，每个symbol一个独立的listener。
    """
    
    def __init__(
        self,
        symbols: List[str],
        storage_manager: StorageManager,
        feature_computer_factory: Optional[callable] = None,
        gap_filler: Optional[GapFiller] = None,
        memory_window_hours: float = 4.0,
        feature_compute_interval_minutes: int = 15,
        feature_4h_interval_hours: int = 4,
    ):
        """
        Args:
            symbols: 交易对符号列表
            storage_manager: 存储管理器（共享）
            feature_computer_factory: 特征计算器工厂函数（如果为None，使用默认创建）
            gap_filler: 数据补全器（共享，可选）
            memory_window_hours: 内存滑动窗口时长（小时）
            feature_compute_interval_minutes: 特征计算间隔（分钟）
            feature_4h_interval_hours: 4小时特征保存间隔（小时）
        """
        self.symbols = symbols
        self.storage_manager = storage_manager
        self.gap_filler = gap_filler
        self.memory_window_hours = memory_window_hours
        self.feature_compute_interval_minutes = feature_compute_interval_minutes
        self.feature_4h_interval_hours = feature_4h_interval_hours
        
        # 为每个symbol创建独立的OrderFlowListener
        self.listeners: Dict[str, OrderFlowListener] = {}
        
        for symbol in symbols:
            # 为每个symbol创建独立的特征计算器
            if feature_computer_factory:
                feature_computer = feature_computer_factory(symbol)
            else:
                feature_computer = IncrementalFeatureComputer(
                    tick_window_minutes=int(memory_window_hours * 60),
                    bar_window_size=int(memory_window_hours * 60),
                )
            
            # 创建OrderFlowListener
            listener = OrderFlowListener(
                symbol=symbol,
                storage_manager=storage_manager,
                feature_computer=feature_computer,
                gap_filler=gap_filler,
                memory_window_hours=memory_window_hours,
                feature_compute_interval_minutes=feature_compute_interval_minutes,
                feature_4h_interval_hours=feature_4h_interval_hours,
            )
            
            self.listeners[symbol] = listener
    
    def get_listener(self, symbol: str) -> Optional[OrderFlowListener]:
        """
        获取指定symbol的listener
        
        Args:
            symbol: 交易对符号
        
        Returns:
            OrderFlowListener实例或None
        """
        return self.listeners.get(symbol)
    
    def on_trade_tick(self, symbol: str, tick: Any) -> None:
        """
        处理指定symbol的tick数据
        
        Args:
            symbol: 交易对符号
            tick: TradeTick对象
        """
        listener = self.listeners.get(symbol)
        if listener:
            listener.on_trade_tick(tick)
        else:
            raise ValueError(f"Unknown symbol: {symbol}")
    
    async def warmup_all(
        self,
        days: int = 30,
        use_gap_filler: bool = True,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        为所有symbol执行warmup
        
        Args:
            days: 加载最近N天的数据
            use_gap_filler: 是否使用GapFiller进行补数据
        
        Returns:
            包含每个symbol的warmup数据的字典
        """
        results = {}
        for symbol, listener in self.listeners.items():
            try:
                warmup_data = listener.warmup(days=days, use_gap_filler=use_gap_filler)
                results[symbol] = warmup_data
            except Exception as e:
                print(f"⚠️ Warmup failed for {symbol}: {e}")
                results[symbol] = {}
        
        return results
    
    async def start_all(self) -> None:
        """启动所有listener"""
        tasks = []
        for symbol, listener in self.listeners.items():
            tasks.append(listener.start())
        
        await asyncio.gather(*tasks)
    
    async def stop_all(self) -> None:
        """停止所有listener"""
        tasks = []
        for symbol, listener in self.listeners.items():
            tasks.append(listener.stop())
        
        await asyncio.gather(*tasks)
    
    def get_all_recovery_states(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有symbol的恢复状态
        
        Returns:
            包含每个symbol恢复状态的字典
        """
        states = {}
        for symbol, listener in self.listeners.items():
            states[symbol] = listener.get_recovery_state()
        
        return states
    
    def recover_all_from_interruption(self) -> Dict[str, Dict[str, Any]]:
        """
        为所有symbol执行恢复
        
        Returns:
            包含每个symbol恢复状态的字典
        """
        states = {}
        for symbol, listener in self.listeners.items():
            try:
                state = listener.recover_from_interruption()
                states[symbol] = state
            except Exception as e:
                print(f"⚠️ Recovery failed for {symbol}: {e}")
                states[symbol] = {}
        
        return states
    
    def get_all_memory_windows(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有symbol的内存窗口数据
        
        Returns:
            包含每个symbol内存窗口数据的字典
        """
        windows = {}
        for symbol, listener in self.listeners.items():
            windows[symbol] = listener.get_memory_window()
        
        return windows
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        获取所有symbol的状态摘要
        
        Returns:
            状态摘要字典
        """
        summary = {
            "symbols": list(self.listeners.keys()),
            "listeners": {},
        }
        
        for symbol, listener in self.listeners.items():
            memory_window = listener.get_memory_window()
            recovery_state = listener.get_recovery_state()
            
            summary["listeners"][symbol] = {
                "is_running": listener.is_running,
                "memory_window_size": len(memory_window),
                "latest_1min_timestamp": recovery_state.get("latest_1min_timestamp"),
                "latest_15min_timestamp": recovery_state.get("latest_15min_timestamp"),
                "has_incomplete_bar": recovery_state.get("incomplete_bar") is not None,
            }
        
        return summary
