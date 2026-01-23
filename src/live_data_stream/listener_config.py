"""
订单流监听器配置

提供可配置的参数，使内存保留时长、聚合窗口、特征计算间隔等参数可配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class OrderFlowListenerConfig:
    """
    订单流监听器配置
    """
    
    # 基本配置
    symbol: str
    storage_base_path: str = "data/live_storage"
    
    # 内存窗口配置
    memory_window_hours: float = 4.0
    
    # 特征计算配置
    feature_compute_interval_minutes: int = 15
    feature_4h_interval_hours: int = 4
    
    # Feature Store配置（可选）
    feature_store_dir: Optional[str] = None
    feature_store_layer: Optional[str] = None
    
    # 数据补全配置
    gap_fill_enabled: bool = True
    gap_fill_threshold_hours: float = 24.0  # 超过24小时从币安API获取
    
    # 特征计算器配置
    tick_window_minutes: Optional[int] = None  # 如果为None，使用memory_window_hours * 60
    bar_window_size: Optional[int] = None  # 如果为None，使用memory_window_hours * 60
    
    def __post_init__(self):
        """后处理：设置默认值"""
        if self.tick_window_minutes is None:
            self.tick_window_minutes = int(self.memory_window_hours * 60)
        
        if self.bar_window_size is None:
            self.bar_window_size = int(self.memory_window_hours * 60)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "OrderFlowListenerConfig":
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
        
        Returns:
            配置对象
        """
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            "symbol": self.symbol,
            "storage_base_path": self.storage_base_path,
            "memory_window_hours": self.memory_window_hours,
            "feature_compute_interval_minutes": self.feature_compute_interval_minutes,
            "feature_4h_interval_hours": self.feature_4h_interval_hours,
            "feature_store_dir": self.feature_store_dir,
            "feature_store_layer": self.feature_store_layer,
            "gap_fill_enabled": self.gap_fill_enabled,
            "gap_fill_threshold_hours": self.gap_fill_threshold_hours,
            "tick_window_minutes": self.tick_window_minutes,
            "bar_window_size": self.bar_window_size,
        }
