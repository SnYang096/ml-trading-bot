"""交易执行器 — 从 TradeIntent 到实际下单的完整流程

从 order_flow_listener.py 拆分出来的独立模块，职责:
  - qty 计算（constitution_risk → risk_per_trade → trade_size fallback）
  - 调用 enforce_before_order() 预留 slot
  - 调用 order_manager.place_order() 开仓
  - 下单失败时释放预留的 slot（防泄漏）
  - 下 SL/TP 保护挂单
  - 将成功开仓的持仓交给 PositionTracker 管理
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.order_management.models import OrderSide, OrderType
from src.order_management.position_tracker import PositionTracker
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_profile_apply import pick_atr
from src.time_series_model.live.position_logic import build_position_dict
from src.time_series_model.portfolio.slot_sizing import compute_slot_size_from_risk

logger = logging.getLogger(__name__)


class TradeExecutor:
    """交易执行器

    Args:
        order_manager: OrderManager 实例（None 时抛出 RuntimeError）
        constitution_executor: ConstitutionExecutor（slot 预留 / 释放）
        runtime_state: ConstitutionRuntimeState
        position_tracker: PositionTracker（成功开仓后记录持仓）
        symbol: 交易对（如 "BTCUSDT"）
        bar_minutes: 信号时钟分钟数（如 240 = 4h）
        risk_per_slot: 每 slot 风险比例（equity 的分数，如 0.01 = 1%）
        risk_per_trade: 固定风险金额（美元，备用）
        trade_size: 固定下单数量（最终 fallback）
        per_strategy_limits: 各策略风险上限配置 {"bpc": {"max_risk_per_trade": 0.005}}
        stats_collector: 可选，用于记录 order_placed 统计
    """

    def __init__(
        self,
        order_manager: Any,
        constitution_executor: Any,
        runtime_state: Any,
        position_tracker: PositionTracker,
        symbol: str,
        bar_minutes: int = 240,
        risk_per_slot: float = 0.0,
        risk_per_trade: Optional[float] = None,
        trade_size: Optional[float] = None,
        per_strategy_limits: Optional[Dict[str, Any]] = None,
        stats_collector: Optional[Any] = None,
    ) -> None:
        self.order_manager = order_manager
        self.constitution_executor = constitution_executor
        self.runtime_state = runtime_state
        self.position_tracker = position_tracker
        self.symbol = symbol
        self.bar_minutes = bar_minutes
        self.risk_per_slot = risk_per_slot
        self.risk_per_trade = risk_per_trade
        self.trade_size = trade_size
        self.per_strategy_limits = per_strategy_limits or {}
        self.stats_collector = stats_collector

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def execute(self, intent: TradeIntent, features: Dict[str, Any]) -> bool:
        """执行交易意图

        流程:
          1. 分配 position_id（若 intent 没有）
          2. 调用 _execute_inner()
          3. 捕获 ConstitutionViolation / 一般异常，释放泄漏 slot

        Returns:
            True = 成功下单，False = 被拒绝或失败
        """
        if intent.action == "NO_TRADE":
            return False

        # 在 try 块外分配 position_id，确保异常时能释放正确的 slot
        if not intent.position_id:
            intent = dataclasses.replace(
                intent,
                position_id=f"{self.symbol}:{int(pd.Timestamp.now(tz='UTC').value)}",
            )

        try:
            placed = self._execute_inner(intent, features)
            return placed
        except ConstitutionViolation as cv:
            logger.warning("[%s] 宪法拒绝: %s (%s)", self.symbol, cv.code, cv.message)
            self._release_leaked_slot(intent)
            return False
        except Exception as exc:
            logger.error("[%s] 下单异常: %s", self.symbol, exc)
            self._release_leaked_slot(intent)
            return False

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _execute_inner(self, intent: TradeIntent, features: Dict[str, Any]) -> bool:
        """返回 True = 成功下单，False = qty<=0 跳过"""
        side = OrderSide.BUY if intent.action == "LONG" else OrderSide.SELL

        # ── 1. qty 计算 ──
        exec_profile = intent.execution_profile or {}
        rr_constraints = exec_profile.get("rr_constraints") or {}
        sl_r = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
        atr = pick_atr(features) or 0.0
        entry_price = self._resolve_entry_price(features)

        qty = self._calc_qty(intent, features, sl_r, atr, entry_price)
        if qty <= 0:
            return False

        # ── 2. slot 预留（enforce_before_order 在 place_order 前调用）──
        position_id = intent.position_id
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

        # ── 3. order_manager 可用性检查 ──
        if self.order_manager is None:
            raise RuntimeError(
                "order_manager is None — 检查 MLBOT_ORDER_MANAGER_ENABLED "
                "和 BINANCE_API_KEY/BINANCE_API_SECRET 环境变量"
            )

        # ── 4. 构建持仓字典 ──
        pos = build_position_dict(
            intent=intent,
            entry_price=float(entry_price) if entry_price else 0.0,
            atr=atr,
            bar_minutes=self.bar_minutes,
            entry_time=datetime.now(timezone.utc),
        )
        stop_loss_price = pos.get("stop_loss_price")
        take_profit_price = pos.get("take_profit_price")

        # ── 5. 开仓 ──
        self.order_manager.place_order(
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            position_id=position_id,
        )

        # ── 6. 保护挂单（SL / TP）──
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
                logger.warning("[%s] 下 TP 挂单失败，软件 TP 仍生效", self.symbol)

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
                logger.warning("[%s] 下 SL 挂单失败，软件 SL 仍生效", self.symbol)

        # ── 7. 统计 ──
        if self.stats_collector is not None:
            arch = str(intent.archetype or "unknown").lower()
            self.stats_collector.record_order_placed(symbol=self.symbol, strategy=arch)

        # ── 8. 交给 PositionTracker ──
        pos["qty"] = float(qty)
        self.position_tracker.add(position_id, pos)
        return True

    def _calc_qty(
        self,
        intent: TradeIntent,
        features: Dict[str, Any],
        sl_r: float,
        atr: float,
        entry_price: Optional[float],
    ) -> float:
        """计算开仓数量

        优先级（高→低）：
          1. intent.quantity（显式指定）
          2. constitution_risk × equity 反算（risk_per_slot > 0）
          3. risk_per_trade 固定美元反算
          4. trade_size 固定数量

        Returns:
            qty (包含 size_multiplier 和 pcm_budget 调整)，≤0 表示跳过
        """
        qty = 0.0
        qty_source = "none"

        # --- 1. intent.quantity ---
        if intent.quantity is not None:
            qty = float(intent.quantity)
            qty_source = "intent.quantity"

        # --- 2. constitution_risk ---
        if (
            qty <= 0
            and self.risk_per_slot > 0
            and sl_r > 0
            and atr > 0
            and entry_price
            and entry_price > 0
        ):
            arch_key = str(intent.archetype or "").strip().lower()
            effective_risk = self.risk_per_slot
            if arch_key and self.per_strategy_limits:
                strat_cfg = self.per_strategy_limits.get(arch_key) or {}
                strat_risk = strat_cfg.get("max_risk_per_trade")
                if strat_risk is not None:
                    effective_risk = min(effective_risk, float(strat_risk))

            equity = float(features.get("equity", 0.0) or 0.0)
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
                    "[%s] 宪法风险反算: equity=$%.0f risk_pct=%.2f%% arch=%s "
                    "risk_usd=$%.1f SL=%.1fR*ATR=%.2f entry=%.2f qty=%.6f",
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

        # --- 3. risk_per_trade ---
        if qty <= 0 and self.risk_per_trade and self.risk_per_trade > 0:
            if sl_r > 0 and atr > 0 and entry_price and entry_price > 0:
                result = compute_slot_size_from_risk(
                    equity_usd=self.risk_per_trade / 0.01,
                    risk_frac=0.01,
                    price=entry_price,
                    atr=atr,
                    stop_atr=sl_r,
                    max_leverage=10.0,
                )
                qty = result.qty
                qty_source = "risk_per_trade_usd"
                logger.info(
                    "[%s] 固定风险反算: risk=$%.1f SL=%.1fR*ATR=%.2f entry=%.2f qty=%.6f",
                    self.symbol,
                    self.risk_per_trade,
                    sl_r,
                    atr,
                    entry_price,
                    qty,
                )
            else:
                logger.warning(
                    "[%s] 风险反算缺少参数 (sl_r=%.2f atr=%.2f price=%s)",
                    self.symbol,
                    sl_r,
                    atr,
                    entry_price,
                )

        # --- 4. trade_size fallback ---
        if qty <= 0 and self.trade_size and self.trade_size > 0:
            qty = float(self.trade_size)
            qty_source = "trade_size"

        # 若风险反算结果小于最小下单量，回退到 trade_size
        if (
            qty_source in ("constitution_risk", "risk_per_trade_usd")
            and self.trade_size
            and self.trade_size > 0
            and qty < self.trade_size
        ):
            logger.warning(
                "[%s] 风险反算 qty=%.6f < 最小开仓 trade_size=%.6f，fallback",
                self.symbol,
                qty,
                self.trade_size,
            )
            qty = float(self.trade_size)
            qty_source = "trade_size_min_fallback"

        # size_multiplier 调整
        size_mult = float(intent.size_multiplier or 1.0)
        qty *= max(0.0, size_mult)

        # pcm_budget 调整
        if intent.pcm_budget:
            per_symbol = (intent.pcm_budget.get("per_symbol_budget") or {}).get(
                self.symbol
            )
            if per_symbol is not None:
                try:
                    qty *= max(0.0, float(per_symbol))
                except Exception:
                    pass

        return qty

    def _release_leaked_slot(self, intent: TradeIntent) -> None:
        """释放因 enforce_before_order 预留但未完成开仓的 slot

        Bug 修复: SLOT_FULL 场景下新 position_id 从未加入 active，
        此时直接跳过（无需释放），避免误导日志。
        """
        if self.constitution_executor is None or self.runtime_state is None:
            return
        pid = intent.position_id or f"{self.symbol}:"
        # slot 从未被预留（如 SLOT_FULL）→ 跳过
        if pid not in self.runtime_state.slots.active:
            return
        try:
            self.constitution_executor.release_slot(
                st=self.runtime_state,
                position_id=pid,
                reason="order_failed",
            )
            self.constitution_executor.save_runtime_state(self.runtime_state)
            logger.warning("[%s] 已释放因下单失败而泄漏的 slot: %s", self.symbol, pid)
        except Exception:
            pass

    def _resolve_entry_price(self, features: Dict[str, Any]) -> Optional[float]:
        for key in ("close", "price", "last_price", "mark_price"):
            v = features.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None
