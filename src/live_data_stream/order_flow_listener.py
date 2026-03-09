"""订单流监听器

实盘数据管线，实现：
1. 实时接收 tick 事件（dict 格式）
2. 按1分钟聚合tick数据
3. 每15分钟计算特征并保存
4. 每4小时聚合特征并保存
5. 可插拔决策路由（GenericLiveStrategy / 自定义 decision_handler）
6. 增强持仓管理（breakeven lock, activation trailing, time stop）
7. 支持从断线中恢复
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Nautilus 已废弃
NAUTILUS_AVAILABLE = False

from .feature_storage import StorageManager
from .memory_window import MemoryWindow
from .gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.trade_intent import TradeIntent
from src.order_management.models import OrderSide, OrderType
from src.time_series_model.live.execution_profile_apply import (
    pick_atr,
    compute_rr_prices,
    holding_expired,
)
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.portfolio.slot_sizing import (
    compute_slot_size_from_risk,
)


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
        orderflow_window_minutes: Optional[int] = None,
        feature_4h_interval_hours: int = 4,
        storage_base_path: str = "data/live_storage",
        on_bar_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_feature_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        constitution_executor: Optional[ConstitutionExecutor] = None,
        runtime_state: Optional[ConstitutionRuntimeState] = None,
        order_manager: Optional[Any] = None,
        trade_size: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
        decision_handler: Optional[Any] = None,
        mode_manager: Optional[Any] = None,
    ):
        """
        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            storage_manager: 存储管理器
            feature_computer: 特征计算器（如果为None，会创建默认的）
            memory_window_hours: 内存滑动窗口时长（小时）
            feature_compute_interval_minutes: 特征计算间隔（分钟）
            orderflow_window_minutes: 订单流特征窗口（分钟）
            feature_4h_interval_hours: 4小时特征保存间隔（小时）
            storage_base_path: 存储根目录
            on_bar_callback: 收到新bar时的回调函数
            on_feature_callback: 计算完特征时的回调函数
            constitution_executor: 宪法执行器（可选）
            runtime_state: 宪法运行时状态（可选）
            order_manager: 订单管理器（可选）
            trade_size: 默认下单数量（可选，已废弃，建议用 risk_per_trade）
            risk_per_trade: 每笔交易风险金额（美元），基于止损距离反算仓位
            decision_handler: 可插拔决策路由器（可选），需实现
                decide(*, features, symbol, bars=None) -> List[TradeIntent]
                如 GenericLiveStrategy 或任何自定义决策引擎。
            mode_manager: 系统模式管理器（可选），用于检查是否允许交易
        """
        self.symbol = symbol
        self.storage_manager = storage_manager
        self.memory_window_hours = memory_window_hours
        self.feature_compute_interval_minutes = feature_compute_interval_minutes
        self.orderflow_window_minutes = (
            int(orderflow_window_minutes)
            if orderflow_window_minutes is not None
            else int(feature_compute_interval_minutes)
        )
        self.feature_4h_interval_hours = feature_4h_interval_hours
        self.mode_manager = mode_manager  # 模式管理器

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

        # Optional trading pipeline
        self.constitution_executor = constitution_executor
        self.runtime_state = runtime_state
        self.order_manager = order_manager
        self.trade_size = trade_size
        self.risk_per_trade = risk_per_trade
        self.risk_per_slot: float = 0.0  # 从宪法注入 (equity 的比例, 如 0.01 = 1%)
        self.per_strategy_limits: Dict[str, Any] = {}  # 从宪法注入
        self.decision_handler = decision_handler
        self.stats_collector = None  # 可选: StatsCollector 实例，由外部注入
        self.extra_feature_computers: Dict[str, "IncrementalFeatureComputer"] = {}
        self._open_positions: Dict[str, Dict[str, Any]] = {}

        # 1分钟聚合状态（bar级别）
        self.current_1min_bar: Optional[Dict[str, Any]] = None
        self.current_1min_start: Optional[pd.Timestamp] = None

        # 1分钟tick聚合缓冲区（tick级别，按买卖分离）
        self.tick_1min_buffer: Dict[str, Any] = {
            "start_time": None,
            "buy_ticks": [],  # 买方tick列表
            "sell_ticks": [],  # 卖方tick列表
        }

        # 定时器状态
        self.last_feature_compute_time: Optional[pd.Timestamp] = None
        self.last_4h_save_time: Optional[pd.Timestamp] = None

        # 心跳计数器
        self._tick_count: int = 0
        self._bar_count: int = 0

        # 节流：上次保存未完成bar的时间
        self._last_incomplete_save_time: float = 0.0
        self._incomplete_save_interval: float = 10.0  # 每10秒保存一次
        self._last_storage_cleanup_date: Optional[str] = None  # 每天最多清理一次

        # 运行状态
        self.is_running = False
        self._stop_event: Optional[asyncio.Event] = None

    def on_trade_tick(self, tick: Any) -> None:
        """
        处理 tick 事件

        Args:
            tick: dict 或 SimpleNamespace，需有 price/size/side 字段
        """
        # 转换时间戳（支持多种格式）
        if hasattr(tick, "ts_init"):
            # Nautilus Trader TradeTick使用ts_init（纳秒时间戳）
            tick_ts = pd.Timestamp(tick.ts_init, unit="ns", tz="UTC")
        elif hasattr(tick, "ts_init_ns"):
            # Mock对象或其他格式
            tick_ts = pd.Timestamp(tick.ts_init_ns, unit="ns", tz="UTC")
        else:
            # 其他格式，尝试直接转换
            tick_ts = pd.Timestamp(getattr(tick, "timestamp", pd.Timestamp.now()))

        # 计算当前1分钟bar的开始时间
        bar_start = tick_ts.floor("1min")

        # 如果是新的1分钟bar，完成上一个bar
        if self.current_1min_start is not None and bar_start > self.current_1min_start:
            self._finalize_1min_bar()

        # 获取价格和数量
        if hasattr(tick, "price"):
            price = (
                float(tick.price)
                if not isinstance(tick.price, (int, float))
                else float(tick.price)
            )
        else:
            price = float(getattr(tick, "price", 0))

        if hasattr(tick, "size"):
            size = (
                float(tick.size)
                if not isinstance(tick.size, (int, float))
                else float(tick.size)
            )
        else:
            size = float(getattr(tick, "size", 0))

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
        if hasattr(tick, "aggressor_side"):
            aggressor_side = tick.aggressor_side
            is_buy = str(aggressor_side) in ("BUY", "BUYER")
        else:
            # 尝试从其他属性推断
            is_buy = getattr(tick, "side", 1) == 1

        if is_buy:
            self.current_1min_bar["buy_volume"] += size
            self.current_1min_bar["buy_count"] += 1
        else:
            self.current_1min_bar["sell_volume"] += size
            self.current_1min_bar["sell_count"] += 1

        # 传递给特征计算器（统一 dict 格式）
        side_value = 1 if is_buy else -1
        self.feature_computer.on_tick(
            {
                "ts": tick_ts.value,  # 纳秒时间戳
                "price": price,
                "volume": size,
                "side": side_value,
            }
        )
        self._tick_count += 1

        # 新增：缓存tick到1分钟缓冲区（按买卖分离）
        bar_start = tick_ts.floor("1min")
        if self.tick_1min_buffer["start_time"] != bar_start:
            # 新的一分钟，保存上一分钟的tick
            if self.tick_1min_buffer["start_time"] is not None:
                self._save_1min_ticks()
            # 重置缓冲区
            self.tick_1min_buffer = {
                "start_time": bar_start,
                "buy_ticks": [],
                "sell_ticks": [],
            }

        # 累加tick（按方向分类）
        tick_record = {
            "timestamp": tick_ts,
            "price": price,
            "volume": size,
        }
        if is_buy:
            self.tick_1min_buffer["buy_ticks"].append(tick_record)
        else:
            self.tick_1min_buffer["sell_ticks"].append(tick_record)

        # 定期保存未完成的bar（用于恢复）
        self._periodic_save_incomplete_bar()

    def _finalize_1min_bar(self) -> None:
        """完成当前1分钟bar"""
        if self.current_1min_bar is None:
            return

        # 计算订单流指标
        total_volume = self.current_1min_bar["volume"]
        if total_volume > 0:
            self.current_1min_bar["buy_ratio"] = (
                self.current_1min_bar["buy_volume"] / total_volume
            )
            self.current_1min_bar["sell_ratio"] = (
                self.current_1min_bar["sell_volume"] / total_volume
            )
            self.current_1min_bar["delta"] = (
                self.current_1min_bar["buy_volume"]
                - self.current_1min_bar["sell_volume"]
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
            bar_for_computer["ts"] = int(
                pd.Timestamp(bar_for_computer["timestamp"]).value
            )
        self.feature_computer.on_bar(bar_for_computer, timeframe="1min")

        # 回调
        if self.on_bar_callback:
            self.on_bar_callback(self.current_1min_bar)

        # 自动升级检查：每收到一条实时 1min bar，通知 mode_manager
        # 策略B：累积足够实时 bar 后自动 DEGRADED → NORMAL
        if self.mode_manager is not None:
            self.mode_manager.on_realtime_bar()

        # 重置当前bar
        self.current_1min_bar = None
        self.current_1min_start = None

    def _save_1min_ticks(self) -> None:
        """保存1分钟聚合tick数据（按买卖分离，与研究pipeline格式一致）

        格式：每1分钟生成2条tick记录（buy和sell分开）
        [timestamp, price, volume, side]
        """
        if not self.tick_1min_buffer.get("start_time"):
            return

        buy_ticks = self.tick_1min_buffer.get("buy_ticks", [])
        sell_ticks = self.tick_1min_buffer.get("sell_ticks", [])
        start_time = self.tick_1min_buffer["start_time"]

        tick_records = []

        # 处理买方ticks：聚合成一条
        if buy_ticks:
            total_volume = sum(t["volume"] for t in buy_ticks)
            # 使用VWAP作为价格
            vwap = sum(t["price"] * t["volume"] for t in buy_ticks) / total_volume
            tick_records.append(
                {
                    "timestamp": start_time,
                    "price": vwap,
                    "volume": total_volume,
                    "side": 1,  # buy
                }
            )

        # 处理卖方ticks：聚合成一条（时间戳稍微错开）
        if sell_ticks:
            total_volume = sum(t["volume"] for t in sell_ticks)
            vwap = sum(t["price"] * t["volume"] for t in sell_ticks) / total_volume
            tick_records.append(
                {
                    "timestamp": start_time
                    + pd.Timedelta(milliseconds=1),  # 错开时间戳
                    "price": vwap,
                    "volume": total_volume,
                    "side": -1,  # sell
                }
            )

        # 保存到存储
        if tick_records:
            tick_df = pd.DataFrame(tick_records)
            trading_date = start_time.strftime("%Y-%m-%d")
            self.storage_manager.ticks.append(self.symbol, trading_date, tick_df)

    def _periodic_save_incomplete_bar(self) -> None:
        """定期保存未完成的bar（每10秒节流，避免高频 I/O 损坏文件）"""
        import time as _time

        now = _time.monotonic()
        if now - self._last_incomplete_save_time < self._incomplete_save_interval:
            return  # 节流：距上次保存不到10秒
        self._last_incomplete_save_time = now

        if self.current_1min_bar is not None:
            try:
                bar_df = pd.DataFrame([self.current_1min_bar])
                self.storage_manager.save_1min_ticks(
                    self.symbol,
                    bar_df,
                    include_incomplete=True,  # 未完成的bar
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("save incomplete bar failed: %s", e)

    def _compute_and_save_15min_features(self) -> None:
        """从磁盘+Buffer批量计算特征（和研发流程一致）

        流程：
        1. 从磁盘读取 90+ 天 1min bars（用于 atr_percentile(540) 等长 lookback）
        2. 从磁盘读取 7 天 1min ticks（用于 VPIN 自适应桶）
        3. 合并内存buffer数据（memory_window + tick_buffer）
        4. 调用 feature_computer.compute_features_batch()
        5. 保存 + 传给决策引擎

        v2 优化 (2026-02-13):
        - 磁盘数据可能有1-2分钟延迟（最新bars还在内存未落盘）
        - 合并buffer确保计算用到最新数据
        """
        now = pd.Timestamp.now(tz="UTC")

        # ── 1. 从磁盘读取数据 (历史主体) ──
        # 1min bars: 150 天（覆盖 atr_percentile window=540 + shift(1) ≈ 541 bars ≈ 90天，留充足余量）
        bar_lookback_days = 150
        bar_start = (now - timedelta(days=bar_lookback_days)).strftime("%Y-%m-%d")
        bar_end = now.strftime("%Y-%m-%d")
        bars_disk = self.storage_manager.bar_1min.load_range(
            self.symbol, bar_start, bar_end
        )

        # 诊断: bars 数据严重不足时提示 warmup
        min_bars_for_features = 240 * 10  # 10 个 4h bars = 2400 1min bars
        if len(bars_disk) < min_bars_for_features:
            bars_path = self.storage_manager.bar_1min.root / self.symbol
            n_files = (
                len(list(bars_path.glob("*.parquet"))) if bars_path.exists() else 0
            )
            logger.error(
                "[%s] ⚠️ bars 数据严重不足: disk=%d 条 (需要>=%d), "
                "bars目录=%s 含 %d 个文件。"
                "请运行: rsync -avz live/highcap/data/bars/ remote:live/highcap/data/bars/ "
                "或 bash live/scripts/prepare_warmup_ticks.sh highcap 6",
                self.symbol,
                len(bars_disk),
                min_bars_for_features,
                bars_path,
                n_files,
            )

        # 1min ticks: 8 天（覆盖 VPIN 7 天滚动窗口）
        # 如果近期数据有缺口，向前扩展查找（最多100天）
        tick_lookback_days = 8
        tick_start = (now - timedelta(days=tick_lookback_days)).strftime("%Y-%m-%d")
        ticks_disk = self.storage_manager.ticks.load_range(
            self.symbol, tick_start, bar_end
        )

        # 如果ticks不足（VPIN需要7天×1440×2=20160条），向前查找更多数据
        # 临时降低阈值以适应周末/假期数据不足的情况
        min_ticks_required = int(
            os.getenv("MLBOT_MIN_TICKS_REQUIRED", "15000")
        )  # 默认15000
        recent_ticks_count = len(ticks_disk)
        if recent_ticks_count < min_ticks_required:
            # 尝试加载更早的数据（从100天前开始）
            extended_tick_start = (now - timedelta(days=100)).strftime("%Y-%m-%d")
            ticks_disk_extended = self.storage_manager.ticks.load_range(
                self.symbol, extended_tick_start, bar_end
            )
            if len(ticks_disk_extended) > len(ticks_disk):
                ticks_disk = ticks_disk_extended
                logger.info(
                    "[%s] 扩展tick加载范围: %s ~ %s, 共%d条",
                    self.symbol,
                    extended_tick_start,
                    bar_end,
                    len(ticks_disk),
                )

        # 检测最近几天数据缺口：如果近8天数据不足，但100天有足够数据，说明中间有缺口
        if (
            recent_ticks_count < min_ticks_required
            and len(ticks_disk) >= min_ticks_required
        ):
            # 计算缺口天数：近8天应有 8*1440*2=23040条，实际只有recent_ticks_count
            expected_recent = 8 * 1440 * 2
            gap_ratio = 1 - (recent_ticks_count / expected_recent)
            gap_days = int(gap_ratio * 8)

            logger.warning(
                "[%s] ⚠️ 最近7天数据有缺口（约%d天），已用历史数据替代。"
                "建议运行: bash live/scripts/prepare_warmup_ticks.sh %s 1 --fill-gap",
                self.symbol,
                gap_days,
                "highcap",
            )

        # 如果扩展后仍不足，报错退出
        if len(ticks_disk) < min_ticks_required:
            logger.error(
                "[%s] ❌ tick数据不足（需要%d条，实际%d条），VPIN无法计算",
                self.symbol,
                min_ticks_required,
                len(ticks_disk),
            )
            raise RuntimeError(
                f"tick数据不足 (symbol={self.symbol}, 需要{min_ticks_required}条, 实际{len(ticks_disk)}条)。"
                f"请运行: bash live/scripts/prepare_warmup_ticks.sh highcap 1 --fill-gap"
            )

        # ── 2. 从内存buffer读取数据 (最新补充) ──
        bars_buffer = self.memory_window.to_dataframe()
        ticks_buffer = self._get_tick_buffer_df()

        # ── 3. 验证磁盘数据 ──
        if bars_disk.empty:
            logger.error(
                "[%s] ❌ 磁盘bars数据为空，需要先执行warmup准备历史数据",
                self.symbol,
            )
            raise RuntimeError(
                f"磁盘bars数据为空 (symbol={self.symbol})。"
                "请先执行warmup准备历史数据，或检查存储路径配置。"
            )

        # ── 4. 合并 + 去重 ──
        bars_merged = self._merge_bars(bars_disk, bars_buffer)
        ticks_merged = self._merge_ticks(ticks_disk, ticks_buffer)

        # 注入 _symbol 列 — OI join 等特征需要识别 symbol
        if "_symbol" not in bars_merged.columns:
            bars_merged["_symbol"] = self.symbol

        logger.info(
            "[%s] 批量计算: bars=%d (disk=%d + buffer=%d), ticks=%d (disk=%d + buffer=%d)",
            self.symbol,
            len(bars_merged),
            len(bars_disk),
            len(bars_buffer),
            len(ticks_merged),
            len(ticks_disk),
            len(ticks_buffer),
        )

        # ── 5. 批量计算 (primary timeframe) ──
        primary_tf = self.feature_computer.primary_timeframe or "240T"
        self.feature_computer._current_symbol = self.symbol  # for health report
        features = self.feature_computer.compute_features_batch(
            bars_1min=bars_merged,
            ticks_1min=ticks_merged,
            primary_timeframe=primary_tf,
        )

        if not features:
            logger.info("[%s] 特征计算跳过（无可用数据）", self.symbol)
            return

        # ── 5b. 额外时间框架特征 (多策略多 timeframe) ──
        features_by_timeframe = {primary_tf: dict(features)}
        for tf, extra_fc in self.extra_feature_computers.items():
            try:
                extra_fc._current_symbol = self.symbol  # for health report
                extra_features = extra_fc.compute_features_batch(
                    bars_1min=bars_merged,
                    ticks_1min=ticks_merged,
                    primary_timeframe=tf,
                )
                if extra_features:
                    features_by_timeframe[tf] = extra_features
                    logger.info(
                        "[%s] 额外时间框架 %s: %d 个特征",
                        self.symbol,
                        tf,
                        len(extra_features),
                    )
            except Exception as e:
                logger.warning(
                    "[%s] 额外时间框架 %s 特征计算失败: %s", self.symbol, tf, e
                )

        # ── 6. 保存 + 决策 ──
        all_features = dict(features)
        all_features["timestamp"] = now
        features_df = pd.DataFrame([all_features])
        self.storage_manager.save_15min_features(self.symbol, features_df, now)

        n_feat = len([k for k in all_features if k != "timestamp"])
        n_tf = len(features_by_timeframe)
        logger.info(
            "[%s] 特征计算完成: %d 个特征, %d 个时间框架, bars_disk=%d, ticks_disk=%d",
            self.symbol,
            n_feat,
            n_tf,
            len(bars_disk),
            len(ticks_disk),
        )

        self._handle_features(
            all_features,
            features_by_timeframe=features_by_timeframe if n_tf > 1 else None,
        )

    def _get_tick_buffer_df(self) -> pd.DataFrame:
        """从incrementalFeatureComputer.tick_buffer提取最近ticks为DataFrame

        Returns:
            DataFrame with columns: ts, timestamp, price, volume, side
        """
        if (
            not hasattr(self.feature_computer, "tick_buffer")
            or not self.feature_computer.tick_buffer
        ):
            return pd.DataFrame()

        ticks = list(self.feature_computer.tick_buffer)
        if not ticks:
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        # 转换ts (纳秒) -> timestamp
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ns", utc=True)

        return df

    def _merge_bars(
        self, disk_df: pd.DataFrame, buffer_df: pd.DataFrame
    ) -> pd.DataFrame:
        """合并磁盘和内存buffer的bars，按timestamp去重

        Args:
            disk_df: 从磁盘加载的1min bars
            buffer_df: 从memory_window提取的bars

        Returns:
            合并后的DataFrame（按timestamp排序，无重复）
        """
        if disk_df.empty and buffer_df.empty:
            return pd.DataFrame()
        if disk_df.empty:
            return buffer_df
        if buffer_df.empty:
            return disk_df

        # 统一timestamp格式
        disk_df = disk_df.copy()
        buffer_df = buffer_df.copy()

        if "timestamp" in disk_df.columns:
            disk_df["timestamp"] = pd.to_datetime(disk_df["timestamp"], utc=True)
        if "timestamp" in buffer_df.columns:
            buffer_df["timestamp"] = pd.to_datetime(buffer_df["timestamp"], utc=True)

        # 合并 + 去重 (keep='last' 保留buffer的最新数据)
        merged = pd.concat([disk_df, buffer_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        return merged

    def _merge_ticks(
        self, disk_df: pd.DataFrame, buffer_df: pd.DataFrame
    ) -> pd.DataFrame:
        """合并磁盘和内存buffer的ticks

        Args:
            disk_df: 从磁盘加载的1min聚合ticks
            buffer_df: 从tick_buffer提取的ticks

        Returns:
            合并后的DataFrame（按timestamp排序）

        Note:
            tick可以有重复时间戳（同一毫秒多笔交易），不去重
        """
        if disk_df.empty and buffer_df.empty:
            return pd.DataFrame()
        if disk_df.empty:
            return buffer_df
        if buffer_df.empty:
            return disk_df

        # 统一timestamp
        disk_df = disk_df.copy()
        buffer_df = buffer_df.copy()

        if "timestamp" in disk_df.columns:
            disk_df["timestamp"] = pd.to_datetime(disk_df["timestamp"], utc=True)
        if "timestamp" in buffer_df.columns:
            buffer_df["timestamp"] = pd.to_datetime(buffer_df["timestamp"], utc=True)

        # 合并 + 按时间排序
        merged = pd.concat([disk_df, buffer_df], ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # tick可以有重复时间戳，不去重
        return merged

    def _aggregate_and_save_4h_features(self) -> None:
        """保存4小时特征（从最近15分钟特征取最后一条）

        15min 特征已经是在 4h bar 上计算的（compute_features_batch 重采样为 4h），
        4h 特征直接取最近一条 15min 特征即可。
        """
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
            return

        # 取最近 4h 内的最后一条
        features_15min = features_15min[features_15min["timestamp"] >= start_time]
        if len(features_15min) == 0:
            return

        last_features = features_15min.iloc[-1].to_dict()
        last_features["timestamp"] = now
        features_df = pd.DataFrame([last_features])
        self.storage_manager.save_4h_features(self.symbol, features_df, now)

    def _handle_features(
        self,
        all_features: Dict[str, Any],
        *,
        features_by_timeframe: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """处理计算完的特征 — 路由决策 + 执行 + 持仓管理

        Args:
            all_features: 主时间框架特征 (flat dict)
            features_by_timeframe: 多时间框架特征 {timeframe: features_dict}
                用于 LivePCM 多策略路由，可选。
        """
        if self.on_feature_callback:
            self.on_feature_callback(all_features)

        # 是否允许实际下单
        trading_enabled = self.order_manager is not None
        if self.mode_manager is not None:
            if not self.mode_manager.is_trading_allowed():
                mode = self.mode_manager.get_current_mode()
                logger.info("[%s] 当前模式=%s，仅观察", self.symbol, mode.value)
                trading_enabled = False

        intents = []

        # 使用 decision_handler（GenericLiveStrategy / LivePCM 等）
        # 即使不交易也执行决策，以收集漏斗统计
        if self.decision_handler is not None:
            try:
                intents = self.decision_handler.decide(
                    features=all_features,
                    symbol=self.symbol,
                    bars=(
                        self.memory_window.get_latest(240) if self.memory_window else []
                    ),
                    features_by_timeframe=features_by_timeframe,
                )
            except TypeError:
                # 后向兼容: handler 不支持 features_by_timeframe (如单策略 GenericLiveStrategy)
                intents = self.decision_handler.decide(
                    features=all_features,
                    symbol=self.symbol,
                    bars=(
                        self.memory_window.get_latest(240) if self.memory_window else []
                    ),
                )

        if not intents:
            logger.info("[%s] 无交易信号", self.symbol)
        elif trading_enabled:
            for intent in intents:
                logger.info("[%s] 交易信号: %s", self.symbol, intent)
                self._execute_intent(intent=intent, features=all_features)
        else:
            for intent in intents:
                logger.info("[%s] 交易信号(观察模式，不下单): %s", self.symbol, intent)

        # 持仓管理 (仅在交易模式下)
        if trading_enabled:
            self._enforce_open_positions(features=all_features)

        # 每 15min 决策周期结束后 flush 统计 (始终执行)
        self._flush_stats()

    def _execute_intent(self, intent: TradeIntent, features: Dict[str, Any]) -> None:
        if intent.action == "NO_TRADE":
            return

        try:
            self._execute_intent_inner(intent, features)
        except ConstitutionViolation as cv:
            logger.warning("[%s] 宪法拒绝: %s (%s)", self.symbol, cv.code, cv.message)
            return
        except Exception as exc:
            # 🐛 Fix: 下单失败时释放已预留的 slot，防止 slot 被永久占满。
            #   enforce_before_order() 在 place_order() 之前预留 slot，
            #   如果 place_order() 失败（API key/余额/权限），slot 泄漏。
            logger.error("[%s] 下单异常: %s", self.symbol, exc)
            # 尝试释放刚预留的 slot
            if (
                self.constitution_executor is not None
                and self.runtime_state is not None
            ):
                _pid = intent.position_id or f"{self.symbol}:"
                try:
                    self.constitution_executor.release_slot(
                        st=self.runtime_state,
                        position_id=_pid,
                        reason="order_failed",
                    )
                    self.constitution_executor.save_runtime_state(self.runtime_state)
                    logger.warning(
                        "[%s] 已释放因下单失败而泄漏的 slot: %s", self.symbol, _pid
                    )
                except Exception:
                    pass
            return

    def _execute_intent_inner(
        self, intent: TradeIntent, features: Dict[str, Any]
    ) -> None:
        side = OrderSide.BUY if intent.action == "LONG" else OrderSide.SELL

        # ── 仓位计算 ──
        # 优先级: intent.quantity > risk_per_strategy(宪法) × equity 反算
        #         > risk_per_trade(固定美元) > trade_size(固定数量)
        qty = 0.0
        qty_source = "none"

        exec_profile = intent.execution_profile or {}
        rr_constraints = exec_profile.get("rr_constraints") or {}
        sl_r = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
        atr = pick_atr(features) or 0.0
        entry_price = self._resolve_entry_price(features)

        # 解析每策略风险: min(risk_per_slot, strategy.max_risk_per_trade)
        arch_key = str(intent.archetype or "").strip().lower()
        effective_risk = self.risk_per_slot
        if effective_risk > 0 and arch_key and self.per_strategy_limits:
            strat_cfg = self.per_strategy_limits.get(arch_key) or {}
            strat_risk = strat_cfg.get("max_risk_per_trade")
            if strat_risk is not None:
                effective_risk = min(effective_risk, float(strat_risk))

        if intent.quantity is not None:
            qty = float(intent.quantity)
            qty_source = "intent.quantity"
        elif (
            self.risk_per_slot > 0
            and sl_r > 0
            and atr > 0
            and entry_price
            and entry_price > 0
        ):
            # 宪法风险反算: risk_usd = equity × effective_risk (per-strategy capped)
            equity = float(features.get("equity", 0.0) or 0.0)
            if equity <= 0:
                # features 中没有 equity，fallback 到固定 risk_per_trade
                equity = 0.0
            if equity > 0:
                result = compute_slot_size_from_risk(
                    equity_usd=equity,
                    risk_frac=effective_risk,
                    price=entry_price,
                    atr=atr,
                    stop_atr=sl_r,
                    max_leverage=3.0,
                )
                qty = result.qty
                qty_source = "constitution_risk"
                logger.info(
                    "[%s] 宪法风险反算: equity=$%.0f, risk_pct=%.2f%% (strategy=%s), "
                    "risk_usd=$%.1f, SL=%.1fR*ATR=%.2f, entry=%.2f, qty=%.6f",
                    self.symbol,
                    equity,
                    effective_risk * 100,
                    arch_key,
                    equity * effective_risk,
                    sl_r,
                    atr,
                    entry_price,
                    qty,
                )

        # fallback: 固定美元风险 (MLBOT_RISK_PER_TRADE 环境变量)
        if qty <= 0 and self.risk_per_trade and self.risk_per_trade > 0:
            if sl_r > 0 and atr > 0 and entry_price and entry_price > 0:
                result = compute_slot_size_from_risk(
                    equity_usd=self.risk_per_trade / 0.01,  # 反推: risk_usd / risk_frac
                    risk_frac=0.01,
                    price=entry_price,
                    atr=atr,
                    stop_atr=sl_r,
                    max_leverage=10.0,
                )
                qty = result.qty
                qty_source = "risk_per_trade_usd"
                logger.info(
                    "[%s] 固定风险反算: risk=$%.1f, SL=%.1fR*ATR=%.2f, "
                    "entry=%.2f, qty=%.6f",
                    self.symbol,
                    self.risk_per_trade,
                    sl_r,
                    atr,
                    entry_price,
                    qty,
                )
            else:
                logger.warning(
                    "[%s] 风险反算缺少参数 (sl_r=%.2f, atr=%.2f, price=%s)",
                    self.symbol,
                    sl_r,
                    atr,
                    entry_price,
                )

        # fallback: 固定数量 (trade_size)
        if qty <= 0 and self.trade_size and self.trade_size > 0:
            qty = float(self.trade_size)
            qty_source = "trade_size"

        # 最小开仓量检查: 如果风险反算的 qty 太小，fallback 到 trade_size
        if (
            qty_source in ("constitution_risk", "risk_per_trade_usd")
            and self.trade_size
            and self.trade_size > 0
            and qty < self.trade_size
        ):
            logger.warning(
                "[%s] 风险反算 qty=%.6f < 最小开仓 trade_size=%.6f, "
                "fallback 到 trade_size",
                self.symbol,
                qty,
                self.trade_size,
            )
            qty = float(self.trade_size)
            qty_source = "trade_size_min_fallback"
        size_mult = float(intent.size_multiplier or 1.0)
        qty *= max(0.0, size_mult)
        if intent.pcm_budget:
            per_symbol = (intent.pcm_budget.get("per_symbol_budget") or {}).get(
                self.symbol
            )
            if per_symbol is not None:
                try:
                    qty *= max(0.0, float(per_symbol))
                except Exception:
                    pass
        if qty <= 0:
            return
        position_id = intent.position_id
        if not position_id:
            position_id = f"{self.symbol}:{int(pd.Timestamp.now(tz='UTC').value)}"

        if intent.add_position and intent.parent_position_id:
            # 无风险加仓: 所有同 direction 的活跃仓位必须 breakeven locked
            #   确保任意时刻最大风险 = 最新一仓的 1R
            _new_side = (
                "LONG" if str(intent.action).upper() in ("LONG", "BUY") else "SHORT"
            )
            _same_dir = [
                p for p in self._open_positions.values() if p.get("side") == _new_side
            ]
            _all_locked = _same_dir and all(
                p.get("breakeven_locked", False) for p in _same_dir
            )
            if not _all_locked:
                logger.info("[add_position] 拒绝: 存在未 breakeven locked 的同向仓位")
                return
            self.constitution_executor.validate_add_position(
                st=self.runtime_state,
                position_id=intent.parent_position_id,
                archetype=intent.archetype,
                current_r=intent.current_r,
                locked_profit=True,  # 已通过上方 _all_locked 检查
            )

        enforce_before_order(
            executor=self.constitution_executor,
            runtime_state=self.runtime_state,
            position_id=position_id,
            symbol=self.symbol,
            archetype=str(intent.archetype),
            execution_strategy=str(intent.execution_strategy or intent.archetype),
            execution_tags=intent.execution_tags,
            execution_evidence=intent.execution_evidence,
            equity=features.get("equity"),
            drawdown=features.get("drawdown"),
            daily_loss=float(features.get("daily_loss", 0.0)),
            weekly_loss=float(features.get("weekly_loss", 0.0)),
            monthly_loss=float(features.get("monthly_loss", 0.0)),
            daily_cost_mean=features.get("daily_cost_mean"),
            daily_turnover_mean=features.get("daily_turnover_mean"),
            hard_violation=bool(features.get("hard_violation", False)),
            data_bad=bool(features.get("data_bad", False)),
            evt_risk_flag=features.get("evt_risk_flag"),
            pcm_budget=intent.pcm_budget,
        )
        # 🐛 Fix: 记录 position_id，供 _execute_intent 的 except 分支释放 slot
        intent.position_id = position_id

        # 检查 order_manager 是否可用
        if self.order_manager is None:
            raise RuntimeError(
                "order_manager is None — 检查 MLBOT_ORDER_MANAGER_ENABLED "
                "和 BINANCE_API_KEY/BINANCE_API_SECRET 环境变量"
            )

        # 使用共享模块构建持仓字典 (与回测完全一致)
        entry_price = self._resolve_entry_price(features)
        atr = pick_atr(features) or 0.0
        default_bar_minutes = int(self.feature_4h_interval_hours * 60)

        pos = build_position_dict(
            intent=intent,
            entry_price=float(entry_price) if entry_price else 0.0,
            atr=atr,
            bar_minutes=default_bar_minutes,
            entry_time=datetime.now(timezone.utc),
        )

        # 从共享字典读取 SL/TP/trailing 配置 (用于下单)
        stop_loss_price = pos.get("stop_loss_price")
        take_profit_price = pos.get("take_profit_price")
        allow_trailing = pos.get("allow_trailing", False)

        self.order_manager.place_order(
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            position_id=position_id,
        )

        # Place protective orders — 始终下 STOP_MARKET 兜底 (含 trailing 策略)
        close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        if take_profit_price is not None:
            try:
                tp_order = self.order_manager.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=qty,
                    stop_price=take_profit_price,
                    reduce_only=True,
                    close_position=True,
                    position_id=position_id,
                )
                pos["_exchange_tp_order_id"] = tp_order.order_id
            except Exception:
                logger.warning("[%s] 下 TP 挂单失败, 软件 TP 仍生效", self.symbol)
        if stop_loss_price is not None:
            try:
                sl_order = self.order_manager.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=qty,
                    stop_price=stop_loss_price,
                    reduce_only=True,
                    close_position=True,
                    position_id=position_id,
                )
                pos["_exchange_sl_order_id"] = sl_order.order_id
                pos["_exchange_sl_price"] = stop_loss_price
            except Exception:
                logger.warning("[%s] 下 SL 挂单失败, 软件 SL 仍生效", self.symbol)

        # 记录下单到统计收集器
        if self.stats_collector is not None:
            archetype_name = str(intent.archetype or "unknown").lower()
            self.stats_collector.record_order_placed(
                symbol=self.symbol,
                strategy=archetype_name,
            )

        # 存储持仓 (共享字典 + qty)
        pos["qty"] = float(qty)
        self._open_positions[position_id] = pos

        if intent.add_position and intent.parent_position_id:
            self.constitution_executor.record_add_position(
                st=self.runtime_state,
                position_id=intent.parent_position_id,
                current_r=intent.current_r,
                locked_profit=intent.locked_profit,
            )
            self.constitution_executor.save_runtime_state(self.runtime_state)

    def _flush_stats(self) -> None:
        """将 stats_collector 当前窗口数据 flush 到 SQLite

        额外传入:
          - symbol: 当前币种
          - system_health: 数据健康指标 (tick_count, bar_count, memory_window_size)
        """
        if self.stats_collector is None:
            return
        try:
            positions = {
                pid: {"side": p["side"], "qty": p["qty"]}
                for pid, p in self._open_positions.items()
            }
            # 数据健康指标
            data_health: Dict[str, Any] = {
                "tick_count": self._tick_count,
                "bar_count": self._bar_count,
                "memory_window_size": (
                    self.memory_window.size() if self.memory_window else 0
                ),
            }
            self.stats_collector.flush(
                symbol=self.symbol,
                positions=positions,
                system_health=data_health,
            )
        except Exception:
            logger.exception("[%s] stats_collector flush 失败", self.symbol)

        # 每天触发一次 feature 文件清理 (仅当 auto_cleanup=True)
        if getattr(self.stats_collector, "auto_cleanup", False):
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._last_storage_cleanup_date != today:
                self._last_storage_cleanup_date = today
                try:
                    sm = self.storage_manager
                    sm.feature_15min.cleanup_old_files(days=30)
                    sm.feature_4h.cleanup_old_files(days=30)
                except Exception:
                    logger.exception("[%s] feature storage cleanup 失败", self.symbol)

    def _resolve_entry_price(self, features: Dict[str, Any]) -> Optional[float]:
        for key in ("close", "price", "last_price", "mark_price"):
            if key in features and features.get(key) is not None:
                try:
                    return float(features.get(key))
                except Exception:
                    pass
        if self.current_1min_bar:
            try:
                return float(self.current_1min_bar.get("close"))
            except Exception:
                pass
        bars = self.memory_window.get_latest(1) if self.memory_window else []
        if bars:
            try:
                return float(bars[-1].get("close"))
            except Exception:
                return None
        return None

    def _enforce_open_positions(self, features: Dict[str, Any]) -> None:
        """管理已有持仓 — 调用共享 enforce_position (与回测同一份代码)"""
        if not self._open_positions:
            return
        now = features.get("timestamp")
        if isinstance(now, str):
            try:
                now = datetime.fromisoformat(str(now))
            except Exception:
                now = None
        if not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
        current_price = self._resolve_entry_price(features)
        if current_price is None:
            return
        default_bar_minutes = int(self.feature_4h_interval_hours * 60)
        to_close: List[str] = []

        for pid, pos in self._open_positions.items():
            # structural exit: 从 features 获取 EMA200 当前值
            _structural_price = None
            if pos.get("structural_exit") == "ema200":
                _structural_price = features.get("ema_200")
                if _structural_price is not None:
                    try:
                        _structural_price = float(_structural_price)
                    except (TypeError, ValueError):
                        _structural_price = None

            # 实盘用单一 current_price (high=low=close=current_price)
            close_reason, _exit_price = enforce_position(
                pos,
                price_high=current_price,
                price_low=current_price,
                price_close=current_price,
                now=now,
                default_bar_minutes=default_bar_minutes,
                structural_price=_structural_price,
            )

            # ── 交易所 SL 同步: trailing 更新了 SL 价格时 cancel+replace ──
            if close_reason is None:
                new_sl = pos.get("stop_loss_price")
                old_exchange_sl = pos.get("_exchange_sl_price")
                if (
                    new_sl is not None
                    and old_exchange_sl is not None
                    and abs(new_sl - old_exchange_sl) > 1e-8
                ):
                    self._sync_exchange_sl(pid, pos)

            if close_reason:
                self._close_position(
                    position_id=pid,
                    side=str(pos.get("side")),
                    qty=float(pos.get("qty") or 0.0),
                    reason=close_reason,
                )
                to_close.append(pid)

        for pid in to_close:
            self._open_positions.pop(pid, None)

    def _sync_exchange_sl(self, position_id: str, pos: Dict[str, Any]) -> None:
        """将软件侧更新的 SL 价格同步到交易所 STOP_MARKET 挂单 (cancel+replace)"""
        if self.order_manager is None:
            return

        new_sl = pos.get("stop_loss_price")
        if new_sl is None:
            return

        # 1. Cancel 旧 SL 挂单
        old_order_id = pos.get("_exchange_sl_order_id")
        if old_order_id:
            try:
                self.order_manager.cancel_order(old_order_id)
            except Exception:
                logger.warning(
                    "[%s] cancel 旧 SL 挂单失败 (可能已触发): %s",
                    self.symbol,
                    old_order_id,
                )

        # 2. Place 新 SL 挂单
        side_str = str(pos.get("side", "")).upper()
        close_side = OrderSide.SELL if side_str in {"LONG", "BUY"} else OrderSide.BUY
        qty = float(pos.get("qty") or 0.0)
        if qty <= 0:
            return

        try:
            new_order = self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                order_type=OrderType.STOP_MARKET,
                quantity=qty,
                stop_price=new_sl,
                reduce_only=True,
                close_position=True,
                position_id=position_id,
            )
            old_sl_price = pos.get("_exchange_sl_price", 0)
            pos["_exchange_sl_order_id"] = new_order.order_id
            pos["_exchange_sl_price"] = new_sl
            logger.info(
                "[%s] 交易所 SL 同步: %.4f → %.4f, order=%s",
                self.symbol,
                old_sl_price,
                new_sl,
                new_order.order_id,
            )
        except Exception:
            logger.error(
                "[%s] place 新 SL 挂单失败 (%.4f), 软件 SL 仍生效",
                self.symbol,
                new_sl,
            )

    def _close_position(
        self, *, position_id: str, side: str, qty: float, reason: str
    ) -> None:
        if qty <= 0 or self.order_manager is None:
            return

        # ── 清理交易所 SL/TP 挂单 (平仓前 cancel, 避免重复触发) ──
        pos = self._open_positions.get(position_id, {})
        for key in ("_exchange_sl_order_id", "_exchange_tp_order_id"):
            oid = pos.get(key)
            if oid:
                try:
                    self.order_manager.cancel_order(oid)
                except Exception:
                    pass  # 可能已触发 / 不存在, 忽略

        close_side = (
            OrderSide.SELL if str(side).upper() in {"LONG", "BUY"} else OrderSide.BUY
        )
        try:
            self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=float(qty),
                reduce_only=True,
                close_position=True,
                position_id=position_id,
            )
        except Exception:
            logger.warning(
                "[%s] 软件平仓失败 (reason=%s), 交易所挂单可能已触发平仓",
                self.symbol,
                reason,
            )

    async def _periodic_tasks(self) -> None:
        """定期任务（特征计算和保存）"""
        _TIME_SYNC_INTERVAL = 30 * 60  # 30 分钟同步一次 Binance 时间
        _last_time_sync = 0.0
        import time as _time_mod

        while not self._stop_event.is_set():
            now = pd.Timestamp.now(tz="UTC")

            # 心跳日志：每60秒打印一次
            price_str = ""
            if self.current_1min_bar is not None:
                price_str = f", price={self.current_1min_bar['close']:.2f}"
            bars_in_window = self.memory_window.size() if self.memory_window else 0
            logger.info(
                "[%s] ❤ ticks=%d, bars=%d%s",
                self.symbol,
                self._tick_count,
                bars_in_window,
                price_str,
            )

            # ── Binance 时间同步 + Slot 持仓同步 (每 30 分钟) ──
            _now_mono = _time_mod.monotonic()
            if _now_mono - _last_time_sync >= _TIME_SYNC_INTERVAL:
                try:
                    api = getattr(
                        getattr(self, "order_manager", None), "binance_api", None
                    )
                    if api is not None and hasattr(api, "_check_time_sync"):
                        api._check_time_sync()
                except Exception as _e:
                    logger.warning("[%s] Binance 定期时间同步失败: %s", self.symbol, _e)
                # Slot 同步: 释放服务端无持仓的 stale slot
                try:
                    _ce = getattr(self, "constitution_executor", None)
                    _rs = getattr(self, "runtime_state", None)
                    _om = getattr(self, "order_manager", None)
                    if _ce is not None and _rs is not None and _om is not None:
                        _api = getattr(_om, "binance_api", None)
                        if _api is not None:
                            _active = dict(_rs.slots.active)
                            if _active:
                                # get_positions() 已只返回 contracts!=0 的持仓
                                # ccxt symbol 格式 BTC/USDT:USDT → 转换为 BTCUSDT
                                _positions = _api.get_positions()
                                _live_syms = set()
                                for _p in _positions:
                                    _raw = (
                                        _p.get("symbol", "")
                                        .replace("/", "")
                                        .split(":")[0]
                                    )
                                    if _raw:
                                        _live_syms.add(_raw)
                                _freed = 0
                                for _pid, _rec in _active.items():
                                    _ssym = getattr(_rec, "symbol", None) or ""
                                    if _ssym and _ssym not in _live_syms:
                                        _ce.release_slot(
                                            st=_rs,
                                            position_id=_pid,
                                            reason="stale_sync",
                                        )
                                        _freed += 1
                                        logger.warning(
                                            "[%s] 🗑️ 定期同步释放 stale slot: %s (%s)",
                                            self.symbol,
                                            _pid,
                                            _ssym,
                                        )
                                if _freed > 0:
                                    _ce.save_runtime_state(_rs)
                except Exception as _e:
                    logger.warning("[%s] 定期 slot 同步失败: %s", self.symbol, _e)
                _last_time_sync = _now_mono

            # 检查是否需要计算15分钟特征
            if (
                self.last_feature_compute_time is None
                or (now - self.last_feature_compute_time).total_seconds()
                >= self.feature_compute_interval_minutes * 60
            ):
                self._compute_and_save_15min_features()
                self.last_feature_compute_time = now

            # 检查是否需要保存4小时特征
            if (
                self.last_4h_save_time is None
                or (now - self.last_4h_save_time).total_seconds()
                >= self.feature_4h_interval_hours * 3600
            ):
                self._aggregate_and_save_4h_features()
                self.last_4h_save_time = now

            # 等待1分钟再检查
            await asyncio.sleep(60)

    def warmup(
        self, days: int = 30, use_gap_filler: bool = True
    ) -> Dict[str, pd.DataFrame]:
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
        恢复状态（简化版：只恢复时间戳 + memory_window）

        特征计算已改为磁盘批量模式 (compute_features_batch)，
        不再需要通过回放 bars/ticks 重建流式状态。

        Args:
            data: warmup数据字典，可包含：
                - ticks_1min: 1分钟聚合tick
                - bars_1min: 1分钟 OHLCV bar
                - features_15min: 15分钟特征
                - features_4h: 4小时特征
        """
        import pandas as pd

        # 恢复特征计算时间戳
        if len(data.get("features_15min", pd.DataFrame())) > 0:
            features_15min = data["features_15min"]
            latest_ts = features_15min["timestamp"].max()
            self.last_feature_compute_time = pd.Timestamp(latest_ts)

        # 恢复4小时特征保存时间
        if len(data.get("features_4h", pd.DataFrame())) > 0:
            features_4h = data["features_4h"]
            latest_ts = features_4h["timestamp"].max()
            self.last_4h_save_time = pd.Timestamp(latest_ts)

        # 恢复 memory_window（BPC 决策引擎需要近期 bars）
        bars_1min = data.get("bars_1min", pd.DataFrame())
        if len(bars_1min) > 0:
            logger.info("  → Restoring memory_window: %d bars", len(bars_1min))
            for row in bars_1min.itertuples(index=False):
                bar_data = {
                    "timestamp": row.timestamp,
                    "open": float(getattr(row, "open", 0)),
                    "high": float(getattr(row, "high", 0)),
                    "low": float(getattr(row, "low", 0)),
                    "close": float(getattr(row, "close", 0)),
                    "volume": float(getattr(row, "volume", 0)),
                }
                self.memory_window.add(bar_data)
            logger.info(
                "  → memory_window restored: %d bars", self.memory_window.size()
            )

        # NOTE: 不再回放 ticks/bars 到 feature_computer
        # 特征计算现在通过 compute_features_batch() 从磁盘直接读取
        ticks_count = len(data.get("ticks_1min", pd.DataFrame()))
        if ticks_count > 0:
            logger.info(
                "  → Skip tick replay (%d ticks on disk, batch compute)", ticks_count
            )

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
                logger.warning("⚠️ 检测到数据缺失超过1天，开始补数据...")
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
