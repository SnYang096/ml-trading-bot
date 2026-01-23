"""
内存滑动窗口

实现可配置时长的内存滑动窗口，支持自动清理过期数据。
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
import pandas as pd
import numpy as np


class MemoryWindow:
    """
    内存滑动窗口
    
    功能：
    1. 维护固定时长的数据窗口（如4小时）
    2. 自动清理过期数据
    3. 支持按时间范围查询
    4. 支持转换为DataFrame
    """
    
    def __init__(
        self,
        window_hours: float = 4.0,
        max_items: Optional[int] = None,
    ):
        """
        Args:
            window_hours: 窗口时长（小时）
            max_items: 最大条目数（可选，用于额外限制）
        """
        self.window_hours = window_hours
        self.window_duration = timedelta(hours=window_hours)
        self.max_items = max_items
        
        # 使用deque存储数据（按时间排序）
        self._data: Deque[Dict[str, Any]] = deque(maxlen=max_items)
    
    def add(self, item: Dict[str, Any]) -> None:
        """
        添加数据项
        
        Args:
            item: 数据项字典，必须包含timestamp字段
        """
        if "timestamp" not in item:
            raise ValueError("item must contain 'timestamp' field")
        
        # 转换为Timestamp
        if isinstance(item["timestamp"], str):
            timestamp = pd.Timestamp(item["timestamp"])
        elif isinstance(item["timestamp"], datetime):
            timestamp = pd.Timestamp(item["timestamp"])
        else:
            timestamp = item["timestamp"]
        
        # 更新timestamp
        item = item.copy()
        item["timestamp"] = timestamp
        
        # 添加数据
        self._data.append(item)
        
        # 清理过期数据
        self._cleanup_expired()
    
    def add_batch(self, items: List[Dict[str, Any]]) -> None:
        """
        批量添加数据项
        
        Args:
            items: 数据项列表
        """
        for item in items:
            self.add(item)
    
    def _cleanup_expired(self) -> None:
        """清理过期数据"""
        if len(self._data) == 0:
            return
        
        # 获取当前时间（使用最新数据的时间戳，或当前时间）
        if len(self._data) > 0:
            latest_timestamp = self._data[-1]["timestamp"]
        else:
            latest_timestamp = pd.Timestamp.now(tz="UTC")
        
        # 计算过期时间点
        cutoff_time = latest_timestamp - self.window_duration
        
        # 移除过期数据（从左侧开始）
        while len(self._data) > 0:
            if self._data[0]["timestamp"] < cutoff_time:
                self._data.popleft()
            else:
                break
    
    def get_range(
        self,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取时间范围内的数据
        
        Args:
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
        
        Returns:
            数据项列表
        """
        result = []
        
        for item in self._data:
            timestamp = item["timestamp"]
            
            # 检查时间范围
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            
            result.append(item)
        
        return result
    
    def get_latest(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        获取最新的N条数据
        
        Args:
            n: 数量
        
        Returns:
            数据项列表
        """
        return list(self._data)[-n:]
    
    def get_oldest(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        获取最旧的N条数据
        
        Args:
            n: 数量
        
        Returns:
            数据项列表
        """
        return list(self._data)[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为DataFrame
        
        Returns:
            DataFrame
        """
        if len(self._data) == 0:
            return pd.DataFrame()
        
        # 清理过期数据
        self._cleanup_expired()
        
        # 转换为DataFrame
        df = pd.DataFrame(list(self._data))
        
        # 确保timestamp列存在且为datetime类型
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def clear(self) -> None:
        """清空窗口"""
        self._data.clear()
    
    def size(self) -> int:
        """获取当前数据量"""
        return len(self._data)
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self._data) == 0
    
    def get_time_range(self) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        获取时间范围
        
        Returns:
            (最早时间, 最晚时间) 或 None（如果为空）
        """
        if len(self._data) == 0:
            return None
        
        timestamps = [item["timestamp"] for item in self._data]
        return (min(timestamps), max(timestamps))
    
    def get_latest_timestamp(self) -> Optional[pd.Timestamp]:
        """获取最新时间戳"""
        if len(self._data) == 0:
            return None
        return self._data[-1]["timestamp"]
    
    def get_oldest_timestamp(self) -> Optional[pd.Timestamp]:
        """获取最旧时间戳"""
        if len(self._data) == 0:
            return None
        return self._data[0]["timestamp"]
