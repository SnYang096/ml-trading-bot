"""
WebSocket重连管理器

提供指数退避、重连次数限制、连接状态管理等功能。
"""

from __future__ import annotations

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ReconnectionStats:
    """重连统计信息"""
    total_reconnects: int = 0
    successful_reconnects: int = 0
    failed_reconnects: int = 0
    last_reconnect_time: Optional[float] = None
    last_reconnect_delay: float = 0.0
    consecutive_failures: int = 0
    first_failure_time: Optional[float] = None


@dataclass
class ReconnectionConfig:
    """重连配置"""
    initial_delay: float = 5.0  # 初始重连延迟（秒）
    max_delay: float = 60.0  # 最大重连延迟（秒）
    backoff_multiplier: float = 2.0  # 退避倍数
    max_retries: Optional[int] = None  # 最大重连次数（None=无限）
    reset_after_success: bool = True  # 成功后重置延迟
    jitter: bool = True  # 是否添加随机抖动


class ReconnectionManager:
    """
    WebSocket重连管理器
    
    特性：
    - 指数退避策略
    - 最大重连次数限制
    - 连接状态管理
    - 重连统计
    - 重连回调
    """
    
    def __init__(
        self,
        config: Optional[ReconnectionConfig] = None,
        on_reconnect_success: Optional[Callable[[], None]] = None,
        on_reconnect_failure: Optional[Callable[[Exception], None]] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
    ):
        """
        Args:
            config: 重连配置
            on_reconnect_success: 重连成功回调
            on_reconnect_failure: 重连失败回调
            on_state_change: 状态变化回调
        """
        self.config = config or ReconnectionConfig()
        self.on_reconnect_success = on_reconnect_success
        self.on_reconnect_failure = on_reconnect_failure
        self.on_state_change = on_state_change
        
        self.state = ConnectionState.DISCONNECTED
        self.stats = ReconnectionStats()
        self._current_delay = self.config.initial_delay
        self._retry_count = 0
        self._lock = asyncio.Lock()
    
    def _set_state(self, new_state: ConnectionState) -> None:
        """设置连接状态"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"Connection state changed: {old_state.value} -> {new_state.value}")
            
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    def _calculate_delay(self) -> float:
        """计算重连延迟（指数退避）"""
        delay = min(self._current_delay, self.config.max_delay)
        
        # 添加随机抖动（±20%）
        if self.config.jitter:
            import random
            jitter = delay * 0.2 * (random.random() * 2 - 1)  # -20% to +20%
            delay = max(0.1, delay + jitter)
        
        return delay
    
    async def wait_before_reconnect(self) -> bool:
        """
        等待重连前的延迟
        
        Returns:
            True if should continue reconnecting, False if should stop
        """
        async with self._lock:
            # 检查是否超过最大重连次数
            if self.config.max_retries is not None:
                if self._retry_count >= self.config.max_retries:
                    logger.warning(
                        f"Max retries ({self.config.max_retries}) reached. "
                        f"Stopping reconnection attempts."
                    )
                    self._set_state(ConnectionState.FAILED)
                    return False
            
            self._set_state(ConnectionState.RECONNECTING)
            self._retry_count += 1
            self.stats.total_reconnects += 1
            
            delay = self._calculate_delay()
            self.stats.last_reconnect_delay = delay
            self.stats.last_reconnect_time = time.time()
            
            logger.info(
                f"Reconnecting (attempt {self._retry_count}) "
                f"after {delay:.2f}s delay..."
            )
            
            # 记录首次失败时间
            if self.stats.first_failure_time is None:
                self.stats.first_failure_time = time.time()
        
        await asyncio.sleep(delay)
        return True
    
    def on_connection_success(self) -> None:
        """连接成功回调"""
        async def _handle():
            async with self._lock:
                self._set_state(ConnectionState.CONNECTED)
                self.stats.successful_reconnects += 1
                self.stats.consecutive_failures = 0
                self.stats.first_failure_time = None
                
                # 重置延迟（如果配置了）
                if self.config.reset_after_success:
                    self._current_delay = self.config.initial_delay
                    self._retry_count = 0
                
                logger.info(
                    f"Connection successful. "
                    f"Total reconnects: {self.stats.total_reconnects}, "
                    f"Successful: {self.stats.successful_reconnects}"
                )
                
                if self.on_reconnect_success:
                    try:
                        self.on_reconnect_success()
                    except Exception as e:
                        logger.error(f"Error in reconnect success callback: {e}")
        
        # 创建任务但不等待（避免阻塞）
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_handle())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_handle())
    
    def on_connection_failure(self, error: Exception) -> None:
        """连接失败回调"""
        async def _handle():
            async with self._lock:
                self.stats.failed_reconnects += 1
                self.stats.consecutive_failures += 1
                
                # 增加延迟（指数退避）
                self._current_delay = min(
                    self._current_delay * self.config.backoff_multiplier,
                    self.config.max_delay
                )
                
                logger.warning(
                    f"Connection failed: {error}. "
                    f"Consecutive failures: {self.stats.consecutive_failures}, "
                    f"Next delay: {self._current_delay:.2f}s"
                )
                
                if self.on_reconnect_failure:
                    try:
                        self.on_reconnect_failure(error)
                    except Exception as e:
                        logger.error(f"Error in reconnect failure callback: {e}")
        
        # 创建任务但不等待（避免阻塞）
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_handle())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_handle())
    
    def reset(self) -> None:
        """重置重连管理器状态"""
        async def _reset():
            async with self._lock:
                self._set_state(ConnectionState.DISCONNECTED)
                self._current_delay = self.config.initial_delay
                self._retry_count = 0
                self.stats.consecutive_failures = 0
                self.stats.first_failure_time = None
                logger.debug("Reconnection manager reset")
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_reset())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_reset())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取重连统计信息"""
        return {
            "state": self.state.value,
            "total_reconnects": self.stats.total_reconnects,
            "successful_reconnects": self.stats.successful_reconnects,
            "failed_reconnects": self.stats.failed_reconnects,
            "consecutive_failures": self.stats.consecutive_failures,
            "current_delay": self._current_delay,
            "retry_count": self._retry_count,
            "last_reconnect_time": self.stats.last_reconnect_time,
            "last_reconnect_delay": self.stats.last_reconnect_delay,
        }
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.state == ConnectionState.CONNECTED
    
    def is_reconnecting(self) -> bool:
        """检查是否正在重连"""
        return self.state == ConnectionState.RECONNECTING
    
    def should_continue(self) -> bool:
        """检查是否应该继续重连"""
        if self.config.max_retries is None:
            return True
        return self._retry_count < self.config.max_retries
