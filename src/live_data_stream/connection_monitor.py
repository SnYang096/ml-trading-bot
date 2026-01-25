"""
WebSocket连接监控器

提供心跳检测、超时检测、健康状态评估等功能。
"""

from __future__ import annotations

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 延迟较高但仍在工作
    UNHEALTHY = "unhealthy"  # 心跳超时或连接异常
    DEAD = "dead"  # 连接已断开


@dataclass
class ConnectionHealth:
    """连接健康信息"""
    status: HealthStatus = HealthStatus.HEALTHY
    last_heartbeat_time: Optional[float] = None
    last_message_time: Optional[float] = None
    heartbeat_missed_count: int = 0
    message_count: int = 0
    latency_ms: Optional[float] = None


class ConnectionMonitor:
    """
    WebSocket连接监控器
    
    特性：
    - 心跳检测
    - 超时检测
    - 健康状态评估
    - 消息统计
    """
    
    def __init__(
        self,
        heartbeat_timeout: float = 60.0,  # 心跳超时（秒）
        health_check_interval: float = 30.0,  # 健康检查间隔（秒）
        on_health_change: Optional[Callable[[HealthStatus], None]] = None,
        on_timeout: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            heartbeat_timeout: 心跳超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
            on_health_change: 健康状态变化回调
            on_timeout: 超时回调
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.health_check_interval = health_check_interval
        self.on_health_change = on_health_change
        self.on_timeout = on_timeout
        
        self.health = ConnectionHealth()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitoring:
            logger.warning("Connection monitor is already running")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.debug("Connection monitor started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                # 尝试等待任务完成，但如果event loop已经在运行，则跳过
                try:
                    loop = asyncio.get_running_loop()
                    # 如果event loop正在运行，只取消任务，不等待
                    pass
                except RuntimeError:
                    # 如果没有运行中的event loop，可以等待
                    try:
                        asyncio.get_event_loop().run_until_complete(self._monitor_task)
                    except (asyncio.CancelledError, RuntimeError):
                        pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Error stopping monitor: {e}")
            self._monitor_task = None
        logger.debug("Connection monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    async def _check_health(self) -> None:
        """检查连接健康状态"""
        async with self._lock:
            current_time = time.time()
            old_status = self.health.status
            
            # 检查心跳超时
            if self.health.last_heartbeat_time is not None:
                time_since_heartbeat = current_time - self.health.last_heartbeat_time
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    self.health.status = HealthStatus.DEAD
                    self.health.heartbeat_missed_count += 1
                    
                    logger.warning(
                        f"Heartbeat timeout: {time_since_heartbeat:.2f}s "
                        f"(threshold: {self.heartbeat_timeout}s)"
                    )
                    
                    if self.on_timeout:
                        try:
                            self.on_timeout()
                        except Exception as e:
                            logger.error(f"Error in timeout callback: {e}")
                elif time_since_heartbeat > self.heartbeat_timeout * 0.7:
                    # 接近超时，状态降级
                    self.health.status = HealthStatus.DEGRADED
                else:
                    # 心跳正常
                    if self.health.status == HealthStatus.DEGRADED:
                        self.health.status = HealthStatus.HEALTHY
                    self.health.heartbeat_missed_count = 0
            
            # 检查消息接收
            if self.health.last_message_time is not None:
                time_since_message = current_time - self.health.last_message_time
                
                # 如果超过2倍心跳超时没有消息，认为连接异常
                if time_since_message > self.heartbeat_timeout * 2:
                    if self.health.status != HealthStatus.DEAD:
                        self.health.status = HealthStatus.UNHEALTHY
                        logger.warning(
                            f"No messages received for {time_since_message:.2f}s"
                        )
            
            # 触发状态变化回调
            if old_status != self.health.status and self.on_health_change:
                try:
                    self.on_health_change(self.health.status)
                except Exception as e:
                    logger.error(f"Error in health change callback: {e}")
    
    def record_heartbeat(self) -> None:
        """记录心跳"""
        async def _record():
            async with self._lock:
                self.health.last_heartbeat_time = time.time()
                if self.health.status == HealthStatus.DEAD:
                    self.health.status = HealthStatus.HEALTHY
                    self.health.heartbeat_missed_count = 0
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_record())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_record())
    
    def record_message(self, latency_ms: Optional[float] = None) -> None:
        """记录消息接收"""
        async def _record():
            async with self._lock:
                self.health.last_message_time = time.time()
                self.health.message_count += 1
                if latency_ms is not None:
                    self.health.latency_ms = latency_ms
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_record())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_record())
    
    def get_health(self) -> Dict[str, Any]:
        """获取健康状态信息"""
        async def _get():
            async with self._lock:
                return {
                    "status": self.health.status.value,
                    "last_heartbeat_time": self.health.last_heartbeat_time,
                    "last_message_time": self.health.last_message_time,
                    "heartbeat_missed_count": self.health.heartbeat_missed_count,
                    "message_count": self.health.message_count,
                    "latency_ms": self.health.latency_ms,
                }
        
        # 同步获取（简化实现）
        return {
            "status": self.health.status.value,
            "last_heartbeat_time": self.health.last_heartbeat_time,
            "last_message_time": self.health.last_message_time,
            "heartbeat_missed_count": self.health.heartbeat_missed_count,
            "message_count": self.health.message_count,
            "latency_ms": self.health.latency_ms,
        }
    
    def reset(self) -> None:
        """重置监控器状态"""
        async def _reset():
            async with self._lock:
                self.health = ConnectionHealth()
                logger.debug("Connection monitor reset")
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_reset())
        except RuntimeError:
            # 如果没有运行中的event loop，同步执行
            import asyncio as aio
            aio.run(_reset())
    
    def is_healthy(self) -> bool:
        """检查连接是否健康"""
        return self.health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
