"""订单流监听器

实盘数据管线，实现：
1. 实时接收 tick 事件（dict 格式）
2. 按1分钟聚合tick数据
3. 每15分钟计算特征并保存
4. 每4小时聚合特征并保存
5. 可插拔决策路由（BPCLiveStrategy / 自定义 decision_handler）
6. 增强持仓管理（breakeven lock, activation trailing, time stop）
7. 支持从断线中恢复
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import pandas as pd
import numpy as np

# Nautilus 已废弃
NAUTILUS_AVAILABLE = False

from .feature_storage import StorageManager
from .memory_window import MemoryWindow
from .gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
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
        decision_handler: Optional[Any] = None,
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
            trade_size: 默认下单数量（可选）
            decision_handler: 可插拔决策路由器（可选），需实现
                decide(*, features, symbol, bars=None) -> List[TradeIntent]
                如 BPCLiveStrategy 或任何自定义决策引擎。
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
        self.decision_handler = decision_handler
        self._open_positions: Dict[str, Dict[str, Any]] = {}
        
        # 1分钟聚合状态
        self.current_1min_bar: Optional[Dict[str, Any]] = None
        self.current_1min_start: Optional[pd.Timestamp] = None
        
        # 定时器状态
        self.last_feature_compute_time: Optional[pd.Timestamp] = None
        self.last_4h_save_time: Optional[pd.Timestamp] = None
        
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
            is_buy = str(aggressor_side) in ("BUY", "BUYER")
        else:
            # 尝试从其他属性推断
            is_buy = getattr(tick, 'side', 1) == 1
        
        if is_buy:
            self.current_1min_bar["buy_volume"] += size
            self.current_1min_bar["buy_count"] += 1
        else:
            self.current_1min_bar["sell_volume"] += size
            self.current_1min_bar["sell_count"] += 1
        
        # 传递给特征计算器（统一 dict 格式）
        side_value = 1 if is_buy else -1
        self.feature_computer.on_tick({
            "ts": tick_ts.value,  # 纳秒时间戳
            "price": price,
            "volume": size,
            "side": side_value,
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
        orderflow_features = self.feature_computer.get_orderflow_features(
            window_minutes=self.orderflow_window_minutes
        )
        
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
        
        self._handle_features(all_features)
    
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
            orderflow_features = self.feature_computer.get_orderflow_features(
                window_minutes=max(self.orderflow_window_minutes, 240)
            )
            
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

    def _handle_features(self, all_features: Dict[str, Any]) -> None:
        """处理计算完的特征 — 路由决策 + 执行 + 持仓管理"""
        if self.on_feature_callback:
            self.on_feature_callback(all_features)

        # 检查下单所需依赖
        if self.order_manager is None:
            return

        intents = []

        # 使用 decision_handler（BPCLiveStrategy 等）
        if self.decision_handler is not None:
            intents = self.decision_handler.decide(
                features=all_features,
                symbol=self.symbol,
                bars=self.memory_window.get_latest(240) if self.memory_window else [],
            )
        else:
            return

        if not intents:
            # 即使没有新 intent，仍需管理已有持仓
            self._enforce_open_positions(features=all_features)
            return

        for intent in intents:
            self._execute_intent(intent=intent, features=all_features)
        self._enforce_open_positions(features=all_features)

    def _execute_intent(self, intent: TradeIntent, features: Dict[str, Any]) -> None:
        if intent.action == "NO_TRADE":
            return
        side = OrderSide.BUY if intent.action == "LONG" else OrderSide.SELL
        qty = (
            float(intent.quantity)
            if intent.quantity is not None
            else float(self.trade_size or 0.0)
        )
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
            self.constitution_executor.validate_add_position(
                st=self.runtime_state,
                position_id=intent.parent_position_id,
                archetype=intent.archetype,
                current_r=intent.current_r,
                locked_profit=intent.locked_profit,
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

        exec_profile = intent.execution_profile or {}
        rr_constraints = exec_profile.get("rr_constraints") or {}
        entry_price = self._resolve_entry_price(features)
        atr = pick_atr(features) or 0.0
        stop_loss_r = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
        take_profit_r = float(rr_constraints.get("take_profit_r", 0.0) or 0.0)
        allow_trailing = bool(rr_constraints.get("allow_trailing", False))
        trailing_atr = rr_constraints.get("trailing_atr")
        max_holding_bars = rr_constraints.get("max_holding_bars")

        stop_loss_price = None
        take_profit_price = None
        if entry_price is not None and atr > 0 and stop_loss_r > 0 and take_profit_r > 0:
            stop_loss_price, take_profit_price = compute_rr_prices(
                side=intent.action,
                entry_price=float(entry_price),
                atr=float(atr),
                stop_loss_r=stop_loss_r,
                take_profit_r=take_profit_r,
            )

        self.order_manager.place_order(
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            position_id=position_id,
        )

        # Place protective orders if configured (skip SL if trailing is enabled).
        close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        if take_profit_price is not None:
            self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                order_type=OrderType.TAKE_PROFIT_MARKET,
                quantity=qty,
                stop_price=take_profit_price,
                reduce_only=True,
                close_position=True,
                position_id=position_id,
            )
        if stop_loss_price is not None and not allow_trailing:
            self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                order_type=OrderType.STOP_MARKET,
                quantity=qty,
                stop_price=stop_loss_price,
                reduce_only=True,
                close_position=True,
                position_id=position_id,
            )

        self._open_positions[position_id] = {
            "side": str(intent.action),
            "qty": float(qty),
            "entry_price": entry_price,
            "entry_time": datetime.now(timezone.utc),
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "allow_trailing": allow_trailing,
            "trailing_atr": trailing_atr,
            "max_holding_bars": max_holding_bars,
            # BPC 扩展字段（从 execution_profile.bpc_position_config 读取）
            "atr_at_entry": atr,
        }
        # 写入 BPC 持仓管理配置（如果有）
        bpc_cfg = exec_profile.get("bpc_position_config") or {}
        if bpc_cfg:
            pos = self._open_positions[position_id]
            pos["activation_r"] = bpc_cfg.get("activation_r")
            pos["trail_r"] = bpc_cfg.get("trail_r")
            pos["trailing_activated"] = False
            pos["high_water_mark"] = entry_price if str(intent.action).upper() in {"LONG", "BUY"} else None
            pos["low_water_mark"] = entry_price if str(intent.action).upper() in {"SHORT", "SELL"} else None
            pos["breakeven_enabled"] = bpc_cfg.get("breakeven_enabled", False)
            pos["breakeven_trigger_r"] = bpc_cfg.get("breakeven_trigger_r", 1.0)
            pos["breakeven_locked"] = False
            pos["bar_minutes"] = bpc_cfg.get("bar_minutes", 240)

        if intent.add_position and intent.parent_position_id:
            self.constitution_executor.record_add_position(
                st=self.runtime_state,
                position_id=intent.parent_position_id,
                current_r=intent.current_r,
                locked_profit=intent.locked_profit,
            )
            self.constitution_executor.save_runtime_state(self.runtime_state)

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
        """管理已有持仓 — time stop / trailing / breakeven / TP/SL hit"""
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
        atr = pick_atr(features) or 0.0
        default_bar_minutes = int(self.feature_4h_interval_hours * 60)
        to_close: List[str] = []

        for pid, pos in self._open_positions.items():
            entry_time = pos.get("entry_time")
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(str(entry_time))
                except Exception:
                    entry_time = None
            if not isinstance(entry_time, datetime):
                entry_time = datetime.now(timezone.utc)

            side_str = str(pos.get("side", "")).upper()
            is_long = side_str in {"LONG", "BUY"}
            entry_price = pos.get("entry_price") or current_price
            pos_atr = pos.get("atr_at_entry") or atr
            bar_minutes = pos.get("bar_minutes", default_bar_minutes)

            close_reason = None

            # ── 1. Time stop ──
            if holding_expired(
                entry_time=entry_time,
                now=now,
                max_holding_bars=pos.get("max_holding_bars"),
                bar_minutes=bar_minutes,
            ):
                close_reason = "max_holding_bars"

            # ── 2. Breakeven lock (BPC) ──
            if (
                close_reason is None
                and pos.get("breakeven_enabled")
                and not pos.get("breakeven_locked")
                and pos_atr > 0
            ):
                if is_long:
                    profit_r = (current_price - entry_price) / pos_atr
                else:
                    profit_r = (entry_price - current_price) / pos_atr
                if profit_r >= pos.get("breakeven_trigger_r", 1.0):
                    pos["breakeven_locked"] = True
                    pos["stop_loss_price"] = entry_price

            # ── 3. Update high/low water mark (BPC) ──
            if is_long and pos.get("high_water_mark") is not None:
                if current_price > pos["high_water_mark"]:
                    pos["high_water_mark"] = current_price
            elif not is_long and pos.get("low_water_mark") is not None:
                if current_price < pos["low_water_mark"]:
                    pos["low_water_mark"] = current_price

            # ── 4. Activation-based trailing (BPC) ──
            if (
                close_reason is None
                and pos.get("activation_r") is not None
                and pos_atr > 0
            ):
                if is_long:
                    profit_r = (current_price - entry_price) / pos_atr
                else:
                    profit_r = (entry_price - current_price) / pos_atr
                activation_r = pos["activation_r"]
                trail_r = pos.get("trail_r", 1.0)
                if profit_r >= activation_r:
                    if not pos.get("trailing_activated"):
                        pos["trailing_activated"] = True
                    if is_long:
                        hwm = pos.get("high_water_mark", current_price)
                        trail_sl = hwm - trail_r * pos_atr
                    else:
                        lwm = pos.get("low_water_mark", current_price)
                        trail_sl = lwm + trail_r * pos_atr
                    old_sl = pos.get("stop_loss_price")
                    if old_sl is not None:
                        if is_long and trail_sl > old_sl:
                            pos["stop_loss_price"] = trail_sl
                        elif not is_long and trail_sl < old_sl:
                            pos["stop_loss_price"] = trail_sl
                    else:
                        pos["stop_loss_price"] = trail_sl


            # ── 5. Check SL hit ──
            if close_reason is None:
                sl = pos.get("stop_loss_price")
                if sl is not None:
                    if is_long and current_price <= float(sl):
                        close_reason = "stop_loss_hit"
                    elif not is_long and current_price >= float(sl):
                        close_reason = "stop_loss_hit"

            # ── 6. Check TP hit ──
            if close_reason is None:
                tp = pos.get("take_profit_price")
                if tp is not None:
                    if is_long and current_price >= float(tp):
                        close_reason = "take_profit_hit"
                    elif not is_long and current_price <= float(tp):
                        close_reason = "take_profit_hit"

            # ── 7. Execute close ──
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

    def _close_position(
        self, *, position_id: str, side: str, qty: float, reason: str
    ) -> None:
        if qty <= 0 or self.order_manager is None:
            return
        close_side = OrderSide.SELL if str(side).upper() in {"LONG", "BUY"} else OrderSide.BUY
        self.order_manager.place_order(
            symbol=self.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=float(qty),
            reduce_only=True,
            close_position=True,
            position_id=position_id,
        )
    
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
