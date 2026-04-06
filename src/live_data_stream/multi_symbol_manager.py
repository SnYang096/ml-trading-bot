"""
多Symbol管理器

管理多个OrderFlowListener实例，提供统一接口启动/停止、warmup、状态查询等。
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import logging

from .order_flow_listener import OrderFlowListener
from .feature_storage import StorageManager
from .gap_filler import GapFiller
from .order_manager_factory import init_order_manager_from_env
from .system_mode import SystemMode, SystemModeManager, ModeDecision
from src.order_management.binance_user_stream import BinanceUserStream
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

logger = logging.getLogger(__name__)

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
        orderflow_window_minutes: Optional[int] = None,
        feature_4h_interval_hours: int = 4,
        order_manager: Optional[Any] = None,
    ):
        """
        Args:
            symbols: 交易对符号列表
            storage_manager: 存储管理器（共享）
            feature_computer_factory: 特征计算器工厂函数（如果为None，使用默认创建）
            gap_filler: 数据补全器（共享，可选）
            memory_window_hours: 内存滑动窗口时长（小时）
            feature_compute_interval_minutes: 特征计算间隔（分钟）
            orderflow_window_minutes: 订单流特征窗口（分钟）
            feature_4h_interval_hours: 4小时特征保存间隔（小时）
            order_manager: 订单管理器（可选，默认从环境变量初始化）
        """
        self.symbols = symbols
        self.storage_manager = storage_manager
        self.gap_filler = gap_filler
        self.memory_window_hours = memory_window_hours
        self.feature_compute_interval_minutes = feature_compute_interval_minutes
        self.orderflow_window_minutes = orderflow_window_minutes
        self.feature_4h_interval_hours = feature_4h_interval_hours
        self.order_manager = order_manager or init_order_manager_from_env()
        self.user_stream: Optional[BinanceUserStream] = None

        # 系统模式管理器
        self.mode_manager = SystemModeManager()

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
                orderflow_window_minutes=orderflow_window_minutes,
                feature_4h_interval_hours=feature_4h_interval_hours,
                order_manager=self.order_manager,
                mode_manager=self.mode_manager,  # 传入模式管理器
            )

            self.listeners[symbol] = listener

        # 账户级 User Data Stream（一个账户只需一个，按 symbol 分发）
        try:
            api = getattr(self.order_manager, "binance_api", None)
            if api is not None:
                self.user_stream = BinanceUserStream(
                    binance_api=api,
                    on_execution_report=self._on_execution_report,
                    on_account_update=self._on_account_update,
                    keepalive_interval=30 * 60,
                )
        except Exception as e:
            logger.warning("User Data Stream 初始化失败，降级为定时同步: %s", e)

    def _on_execution_report(self, report: Dict[str, Any]) -> None:
        symbol = str(report.get("symbol") or "").upper().strip()
        listener = self.listeners.get(symbol)
        if listener is not None:
            listener.on_execution_report(report)

    def _on_account_update(self, update: Dict[str, Any]) -> None:
        # 账户事件是全账户级别：所有 listener 都更新权益快照；
        # position 仅由各 listener 过滤本 symbol。
        for listener in self.listeners.values():
            listener.on_account_update(update)

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
        max_retries: int = 3,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        为所有symbol执行warmup（支持重试）

        Args:
            days: 加载最近N天的数据
            use_gap_filler: 是否使用GapFiller进行补数据
            max_retries: 最大重试次数

        Returns:
            包含每个symbol的warmup数据的字典
        """
        results = {}

        for symbol, listener in self.listeners.items():
            success = False
            last_exception = None

            # 重试机制（指数退避）
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"🔄 Warmup {symbol} (attempt {attempt + 1}/{max_retries})..."
                    )
                    warmup_data = listener.warmup(
                        days=days, use_gap_filler=use_gap_filler
                    )
                    results[symbol] = warmup_data
                    success = True
                    logger.info(f"✅ Warmup {symbol} succeeded")
                    break

                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"⚠️ Warmup {symbol} failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )

                    # 如果还有重试机会，等待后重试（指数退避）
                    if attempt < max_retries - 1:
                        backoff_seconds = 2**attempt  # 1s, 2s, 4s
                        logger.info(f"   Retrying in {backoff_seconds}s...")
                        time.sleep(backoff_seconds)

            # 如果所有重试都失败，记录空结果
            if not success:
                logger.error(
                    f"❌ Warmup {symbol} failed after {max_retries} attempts: {last_exception}"
                )
                results[symbol] = {}

        return results

    def decide_startup_mode(
        self, warmup_results: Dict[str, Dict[str, pd.DataFrame]]
    ) -> ModeDecision:
        """根据warmup结果决定启动模式

        Args:
            warmup_results: warmup_all返回的结果

        Returns:
            ModeDecision对象
        """
        # 合并所有symbol的数据进行决策（选择最严格的模式）
        decisions = []

        for symbol, data in warmup_results.items():
            decision = self.mode_manager.decide_mode(data)
            decisions.append((symbol, decision))
            logger.info(
                f"  {symbol}: {decision.mode.value} ({decision.bar_count} bars, "
                f"{decision.data_coverage_hours:.2f}h)"
            )

        # 如果任何symbol返回OFFLINE，整体OFFLINE
        if any(d[1].mode == SystemMode.OFFLINE for d in decisions):
            offline_symbols = [
                d[0] for d in decisions if d[1].mode == SystemMode.OFFLINE
            ]
            return ModeDecision(
                mode=SystemMode.OFFLINE,
                reason=f"Insufficient data for symbols: {', '.join(offline_symbols)}",
                bar_count=min(d[1].bar_count for d in decisions),
                data_coverage_hours=min(d[1].data_coverage_hours for d in decisions),
            )

        # 如果任何symbol返回DEGRADED，整体DEGRADED
        if any(d[1].mode == SystemMode.DEGRADED for d in decisions):
            degraded_symbols = [
                d[0] for d in decisions if d[1].mode == SystemMode.DEGRADED
            ]
            return ModeDecision(
                mode=SystemMode.DEGRADED,
                reason=f"Incomplete data for symbols: {', '.join(degraded_symbols)}",
                bar_count=min(d[1].bar_count for d in decisions),
                data_coverage_hours=min(d[1].data_coverage_hours for d in decisions),
            )

        # 所有symbol都NORMAL
        return ModeDecision(
            mode=SystemMode.NORMAL,
            reason="All symbols have complete data",
            bar_count=min(d[1].bar_count for d in decisions),
            data_coverage_hours=min(d[1].data_coverage_hours for d in decisions),
        )

    def get_current_mode(self) -> SystemMode:
        """获取当前系统模式"""
        return self.mode_manager.get_current_mode()

    def is_trading_allowed(self) -> bool:
        """是否允许交易"""
        return self.mode_manager.is_trading_allowed()

    def manual_reset_to_normal(
        self, reason: str = "CI/CD restart manual reset"
    ) -> None:
        """人工复位系统模式到 NORMAL（用于修复后放行）"""
        self.mode_manager.reset_to_normal(reason=reason)

    async def start_all(self) -> None:
        """启动所有listener"""
        tasks = []
        for symbol, listener in self.listeners.items():
            tasks.append(listener.start())
        if self.user_stream is not None:
            tasks.append(self.user_stream.start())

        await asyncio.gather(*tasks)

    async def stop_all(self) -> None:
        """停止所有listener"""
        tasks = []
        for symbol, listener in self.listeners.items():
            tasks.append(listener.stop())
        if self.user_stream is not None:
            tasks.append(self.user_stream.stop())

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
