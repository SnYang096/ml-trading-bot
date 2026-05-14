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
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_size_multiplier,
    validate_add_position_trigger,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_profile_apply import pick_atr
from src.time_series_model.live.metrics_exporter import METRICS
from src.time_series_model.live.position_logic import build_position_dict
from src.time_series_model.portfolio.slot_sizing import compute_slot_size_from_risk

logger = logging.getLogger(__name__)


def _family_key(archetype: str) -> str:
    s = str(archetype or "").strip().lower()
    if not s:
        return ""
    return s.split("-", 1)[0]


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
        exec_profile = dict(intent.execution_profile or {})
        rr_constraints = dict(exec_profile.get("rr_constraints") or {})
        sl_r_raw = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
        atr = pick_atr(features) or 0.0
        entry_price = self._resolve_entry_price(features)
        sl_r, atr_stop_pct, effective_stop_pct, stop_source = (
            self._resolve_effective_stop_r(
                sl_r=sl_r_raw,
                atr=atr,
                entry_price=entry_price,
                rr_constraints=rr_constraints,
            )
        )
        add_ctx: Optional[Dict[str, Any]] = None
        if bool(intent.add_position):
            intent, add_ctx = self._prepare_add_position(
                intent=intent,
                features=features,
                entry_price=entry_price,
            )
        if abs(sl_r - sl_r_raw) > 1e-9:
            rr_constraints["stop_loss_r"] = float(sl_r)
            exec_profile["rr_constraints"] = rr_constraints
            intent = dataclasses.replace(intent, execution_profile=exec_profile)
        logger.info(
            "[%s] stop guardrail: atr_stop_pct=%.4f effective_stop_pct=%.4f source=%s",
            self.symbol,
            atr_stop_pct,
            effective_stop_pct,
            stop_source,
        )

        qty = self._calc_qty(intent, features, sl_r, atr, entry_price)
        if qty <= 0:
            return False

        # ── 3. order_manager 可用性检查 ──
        if self.order_manager is None:
            raise RuntimeError(
                "order_manager is None — 检查 MLBOT_ORDER_MANAGER_ENABLED "
                "和 BINANCE_API_KEY/BINANCE_API_SECRET 环境变量"
            )
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

        # ── 4. 构建持仓字典 ──
        pos = build_position_dict(
            intent=intent,
            entry_price=float(entry_price) if entry_price else 0.0,
            atr=atr,
            bar_minutes=self.bar_minutes,
            entry_time=datetime.now(timezone.utc),
        )
        if add_ctx is not None and bool(add_ctx.get("inherit_parent_stop", False)):
            parent_sl = add_ctx.get("parent_stop_loss_price")
            if parent_sl is not None:
                # Add position shares parent's current SL anchor.
                pos["stop_loss_price"] = float(parent_sl)
            pos["breakeven_enabled"] = False
            pos["activation_r"] = None
            pos["trailing_activated"] = False
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
        try:
            arch = str(intent.archetype or "unknown").lower()
            side = str(intent.action or "na").lower()
            METRICS.record_strategy_event(
                scope="trend",
                strategy=arch,
                symbol=self.symbol,
                event="entry",
                side=side,
                price=float(entry_price) if entry_price is not None else None,
            )
        except Exception:
            logger.debug(
                "[%s] entry marker metrics update skipped",
                self.symbol,
                exc_info=True,
            )

        # ── 8. 交给 PositionTracker ──
        pos["qty"] = float(qty)
        pos["atr_stop_pct"] = float(atr_stop_pct)
        pos["effective_stop_pct"] = float(effective_stop_pct)
        pos["sizing_stop_source"] = stop_source
        if add_ctx is not None:
            pos["_is_add_position"] = True
            pos["_parent_pid"] = str(add_ctx.get("parent_position_id", ""))
            pos["_add_position_seq"] = int(add_ctx.get("add_position_seq", 1) or 1)
            pos["_share_parent_exit"] = bool(add_ctx.get("share_parent_exit", True))
            pos["_inherit_parent_stop"] = bool(
                add_ctx.get("inherit_parent_stop", False)
            )
            _pse = add_ctx.get("parent_structural_exit")
            if _pse and not pos.get("structural_exit"):
                pos["structural_exit"] = str(_pse)
        self.position_tracker.add(position_id, pos)
        if add_ctx is not None:
            self.constitution_executor.record_add_position(
                st=self.runtime_state,
                position_id=str(add_ctx.get("parent_position_id", "")),
                current_r=add_ctx.get("current_r"),
                locked_profit=add_ctx.get("locked_profit"),
            )
            self.constitution_executor.save_runtime_state(self.runtime_state)
        return True

    def _prepare_add_position(
        self,
        *,
        intent: TradeIntent,
        features: Dict[str, Any],
        entry_price: Optional[float],
    ) -> tuple[TradeIntent, Dict[str, Any]]:
        parent_pid, parent_pos = self._resolve_parent_position(intent)
        if parent_pid is None or parent_pos is None:
            raise ConstitutionViolation(
                code="ADD_POSITION_NO_PARENT",
                message=(
                    f"no parent position for add_position "
                    f"(symbol={self.symbol} archetype={intent.archetype})"
                ),
            )
        current_r = (
            float(intent.current_r)
            if intent.current_r is not None
            else self._compute_current_r(
                side=str(intent.action).upper(),
                parent_pos=parent_pos,
                current_price=entry_price,
            )
        )
        locked_profit = bool(
            intent.locked_profit
            if intent.locked_profit is not None
            else parent_pos.get("breakeven_locked", False)
        )
        self.constitution_executor.validate_add_position(
            st=self.runtime_state,
            position_id=parent_pid,
            archetype=str(intent.archetype),
            current_r=current_r,
            locked_profit=locked_profit,
            position_action=str(intent.action),
        )
        add_rules = dict(
            self.constitution_executor.resolve_add_position_for_strategy(
                str(intent.archetype), position_action=str(intent.action)
            )
        )
        _intent_add = (getattr(intent, "execution_profile", {}) or {}).get(
            "add_position"
        ) or {}
        if _intent_add:
            _trig = dict(add_rules.get("trigger", {}) or {})
            _trig.update(dict(_intent_add.get("trigger", {}) or {}))
            add_rules.update(
                {k: v for k, v in dict(_intent_add).items() if k != "trigger"}
            )
            if _trig:
                add_rules["trigger"] = _trig

        rec = self.runtime_state.add_position.positions.get(parent_pid)
        next_add_no = int(rec.add_count) + 1 if rec is not None else 1
        signal = dict(features or {})
        signal["add_position_seq"] = next_add_no
        _atr_parent = float(parent_pos.get("atr_at_entry", 0.0) or 0.0)
        _risk_parent = float(parent_pos.get("initial_risk_distance", 0.0) or 0.0)
        if _atr_parent > 0 and _risk_parent > 0:
            signal["parent_initial_r"] = _risk_parent / _atr_parent
        if not validate_add_position_trigger(
            archetype=str(intent.archetype),
            direction=1 if str(intent.action).upper() in {"LONG", "BUY"} else -1,
            signal=signal,
            add_position_cfg=add_rules,
            current_r=current_r,
        ):
            raise ConstitutionViolation(
                code="ADD_POSITION_TRIGGER_NOT_MET",
                message=(
                    f"add trigger not met (symbol={self.symbol} "
                    f"archetype={intent.archetype} current_r={current_r:.4f})"
                ),
            )
        add_mult = resolve_add_position_size_multiplier(add_rules, next_add_no, signal)
        total_mult = float(intent.size_multiplier or 1.0) * float(add_mult)
        enriched = dataclasses.replace(
            intent,
            parent_position_id=parent_pid,
            current_r=float(current_r),
            locked_profit=locked_profit,
            size_multiplier=float(total_mult),
        )
        return enriched, {
            "parent_position_id": parent_pid,
            "current_r": float(current_r),
            "locked_profit": locked_profit,
            "add_position_seq": int(next_add_no),
            "share_parent_exit": bool(add_rules.get("share_parent_exit", True)),
            "inherit_parent_stop": bool(add_rules.get("inherit_parent_stop", False)),
            "parent_stop_loss_price": parent_pos.get("stop_loss_price"),
            "parent_structural_exit": parent_pos.get("structural_exit"),
        }

    def _resolve_parent_position(
        self, intent: TradeIntent
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        pos_all = self.position_tracker.all_positions()
        if intent.parent_position_id:
            pid = str(intent.parent_position_id).strip()
            p = pos_all.get(pid)
            if p is not None:
                return pid, p
        target_side = (
            "LONG" if str(intent.action).upper() in {"LONG", "BUY"} else "SHORT"
        )
        target_arch = str(intent.archetype or "").strip().lower()
        target_family = _family_key(target_arch)
        for pid, pos in pos_all.items():
            p_side = str(pos.get("side", "")).upper()
            if p_side not in {"LONG", "SHORT", "BUY", "SELL"}:
                continue
            p_norm_side = "LONG" if p_side in {"LONG", "BUY"} else "SHORT"
            if p_norm_side != target_side:
                continue
            p_arch = str(pos.get("archetype", "") or "").strip().lower()
            slot_arch = ""
            try:
                slot_arch = (
                    str(self.runtime_state.slots.active.get(str(pid)).archetype or "")
                    .strip()
                    .lower()
                )
            except Exception:
                slot_arch = ""
            p_tier = str(pos.get("tier_name", "") or "").strip().lower()
            cand = {p_arch, slot_arch, p_tier}
            if target_arch and target_arch in cand:
                return str(pid), pos
            fams = {_family_key(p_arch), _family_key(slot_arch), _family_key(p_tier)}
            if target_family and target_family in fams:
                return str(pid), pos
        return None, None

    @staticmethod
    def _compute_current_r(
        *,
        side: str,
        parent_pos: Dict[str, Any],
        current_price: Optional[float],
    ) -> float:
        ep = float(parent_pos.get("entry_price", 0.0) or 0.0)
        risk_dist = float(parent_pos.get("initial_risk_distance", 0.0) or 0.0)
        px = float(current_price or 0.0)
        if ep <= 0 or risk_dist <= 1e-9 or px <= 0:
            return 0.0
        if str(side).upper() in {"LONG", "BUY"}:
            return (px - ep) / risk_dist
        return (ep - px) / risk_dist

    def _resolve_effective_stop_r(
        self,
        *,
        sl_r: float,
        atr: float,
        entry_price: Optional[float],
        rr_constraints: Dict[str, Any],
    ) -> tuple[float, float, float, str]:
        sl_r = max(0.0, float(sl_r or 0.0))
        if sl_r <= 0 or atr <= 0 or not entry_price or entry_price <= 0:
            return sl_r, 0.0, 0.0, "invalid_inputs"
        atr_stop_pct = max(0.0, (sl_r * atr) / float(entry_price))
        min_stop_pct = rr_constraints.get("min_stop_pct")
        max_stop_pct = rr_constraints.get("max_stop_pct")
        eff = atr_stop_pct
        if min_stop_pct is not None:
            try:
                eff = max(eff, float(min_stop_pct))
            except Exception:
                pass
        if max_stop_pct is not None:
            try:
                eff = min(eff, float(max_stop_pct))
            except Exception:
                pass
        effective_stop_pct = max(0.0, eff)
        if effective_stop_pct <= 0:
            return sl_r, atr_stop_pct, atr_stop_pct, "atr"
        effective_stop_r = effective_stop_pct * float(entry_price) / float(atr)
        source = (
            "atr"
            if abs(effective_stop_pct - atr_stop_pct) <= 1e-9
            else "guardrail_clip"
        )
        return max(1e-6, effective_stop_r), atr_stop_pct, effective_stop_pct, source

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
