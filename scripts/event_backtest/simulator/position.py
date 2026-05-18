from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

from scripts.account_ledger import AccountLedger
from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.spot.budget import (
    _allocate_spot_accum_leg,
    _spot_entry_fill_price,
    _spot_peer_sims,
    _spot_regime_leg_kwargs,
    _spot_symbol_deploy_legs_today,
    _utc_calendar_day_str,
)
from scripts.event_backtest.types.stats import resolve_add_position_size_multiplier
from scripts.event_backtest.types.trade import ClosedTrade
from src.time_series_model.core.constitution.add_position_rules import (
    add_regime_gate_allows as _shared_add_regime_gate_allows,
    resolve_add_position_max_times as _shared_resolve_add_position_max_times,
    resolve_add_position_min_current_r,
    resolve_float_r_ladder_only as _shared_resolve_float_r_ladder_only,
    validate_add_position_trigger as _shared_validate_add_position_trigger,
)
from src.time_series_model.core.constitution.runtime_state import AddPositionRecord
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.live.spot_accum_simple import (
    apply_partial_sell_to_position,
    is_spot_accum_archetype,
    maybe_spot_simple_partial_sell,
)


def _resolve_add_position_size_multiplier(add_rules, add_number, signal=None):
    return resolve_add_position_size_multiplier(add_rules, add_number, signal)


def _json_safe(value):
    from scripts.event_backtest.reporting.audit import json_safe

    return json_safe(value)


class PositionSimulator:
    """
    持仓模拟器 — 调用 position_logic 共享模块，与实盘完全同一份代码。

    回测用 1min bar OHLC (保守假设 SL 优先):
      LONG: if low <= SL → 止损; elif high >= TP → 止盈
      SHORT: if high >= SL → 止损; elif low <= TP → 止盈
    """

    def __init__(
        self,
        default_bar_minutes: int = 240,
        max_positions: int = 1,
        fee_rate: float = 0.0,
    ):
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.default_bar_minutes = default_bar_minutes
        self.max_positions = max_positions
        self.fee_rate = fee_rate  # 单边手续费率 (如 0.0004 = 0.04% taker)
        self.closed_trades: List[ClosedTrade] = []
        # structural exit: 最新 EMA200 价格 (由主循环在每根 4H bar 到达时更新)
        self._structural_price: Optional[float] = None
        self._macro_tp_vwap_position: Optional[float] = None
        # 与 macro_tp_vwap_1200_position 同根特征行的滚动典型价 VWAP 价格水平；
        # 在两次 primary-TF 收盘之间冻结，供 1m bar 用 (close_1m - level)/close_1m 检测死区穿越
        self._macro_tp_vwap_level: Optional[float] = None
        # EMA1200 结构性出场: 与 VWAP1200 同理，冻结 EMA1200 水平供 1m 重算
        # 统计依据: EMA1200 零点穿越比 VWAP1200 更可靠 (分歧时 EMA 正确率显著更高)
        self._ema_1200_position: Optional[float] = None
        self._ema_1200_level: Optional[float] = None
        # 周线宏观结构退出（weekly close<EMA50 连续 + LL/HL 结构）投影到 primary-TF 的信号
        self._macro_cycle_exit_signal: Optional[float] = None
        self._macro_regime_score: Optional[float] = None
        # order_management 集成 (由 EventBacktester 注入)
        self._om_bridge: Optional["OMBridge"] = None
        self.max_observed_leverage: float = 0.0
        self.max_observed_notional_frac: float = 0.0
        # 记录最近一次加仓失败原因（供 funnel 细分统计）
        self.last_add_reject_reason: str = ""
        self.last_add_attempt_signal: Dict[str, Any] = {}
        self.last_open_reject_reason: str = ""
        self._risk_per_slot_usdt: float = 0.0
        self._account_ledger: Optional[AccountLedger] = None
        # spot_accum constitution 记账：账户权益锚点 + peer simulators + UTC 日历 deploy
        self._spot_capital_budget: Optional[Dict[str, Any]] = None
        self._spot_peer_sims: Optional[List["PositionSimulator"]] = None
        self._spot_daily_deploy_totals: Optional[Dict[str, float]] = None
        self._spot_symbol_daily_leg_counts: Optional[Dict[str, int]] = None
        # SRB 加仓门控（由 EventBacktester 从 execution.yaml 注入）
        self._srb_add_policy: Optional[Dict[str, Any]] = None
        self._primary_bar_count: int = 0
        # 主周期（primary TF）最新收盘 ATR — trailing 可选与入场 ATR 取 max 放宽带宽
        self._primary_tf_atr: Optional[float] = None
        # L3 dynamic trailing 所需：当前 primary bar 的 wide_sr 上下沿价格
        self._wide_sr_upper_px: Optional[float] = None
        self._wide_sr_lower_px: Optional[float] = None
        # Phase D 加仓形态门 recent_momentum 所需：近 N 根 primary close 滚动缓存（净位移用）
        self._primary_close_buffer: List[float] = []
        self._primary_close_buffer_max: int = 16

    @property
    def has_positions(self) -> bool:
        return len(self._positions) > 0

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def slot_position_count(self) -> int:
        """供 PCM 全局 slot 统计：加仓腿不占全局 slot。"""
        return sum(
            1
            for pos in self._positions.values()
            if not bool((pos or {}).get("_is_add_position", False))
        )

    def snapshot_open_positions(self) -> List[Dict[str, Any]]:
        """导出当前未平仓状态 (用于跨月续跑)."""
        rows: List[Dict[str, Any]] = []
        for pid, pos in self._positions.items():
            rows.append({"pid": str(pid), "position": _json_safe(pos)})
        return rows

    def restore_open_positions(self, rows: List[Dict[str, Any]]) -> int:
        """恢复未平仓状态 (由 --resume-state 提供)."""
        loaded = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            raw_pos = row.get("position", {})
            if not isinstance(raw_pos, dict):
                continue
            pos = dict(raw_pos)
            entry_time = pos.get("entry_time")
            if isinstance(entry_time, str):
                try:
                    pos["entry_time"] = pd.Timestamp(entry_time).to_pydatetime()
                except Exception:
                    pos["entry_time"] = datetime.now(timezone.utc)
            elif not isinstance(entry_time, datetime):
                pos["entry_time"] = datetime.now(timezone.utc)
            if pos["entry_time"].tzinfo is None:
                pos["entry_time"] = pos["entry_time"].replace(tzinfo=timezone.utc)
            pid = str(row.get("pid") or str(uuid.uuid4())[:12])
            self._positions[pid] = pos
            loaded += 1
        for pid, pos in self._positions.items():
            if not bool(pos.get("_is_add_position", False)):
                continue
            parent = self._positions.get(str(pos.get("_parent_pid") or ""))
            if not isinstance(parent, dict):
                continue
            se = parent.get("structural_exit")
            if se and not pos.get("structural_exit"):
                pos["structural_exit"] = se
            rlc = parent.get("regime_lifecycle_exit")
            if isinstance(rlc, dict) and rlc and not pos.get("regime_lifecycle_exit"):
                pos["regime_lifecycle_exit"] = dict(rlc)
        return loaded

    def _estimate_entry_notional_usdt(
        self,
        *,
        pos: Dict[str, Any],
        entry_price: float,
        size_multiplier: float,
    ) -> float:
        """Estimate entry notional for non-spot positions.

        Priority:
        1) existing explicit notional in position
        2) risk-per-slot cash with stop distance
        3) conservative fallback from risk budget
        """
        n0 = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
        if n0 > 0.0:
            return n0
        if entry_price <= 0.0:
            return 0.0
        stop_pct = float(pos.get("effective_stop_pct", 0.0) or 0.0)
        rb = max(0.0, float(self._risk_per_slot_usdt or 0.0))
        sm = max(0.0, float(size_multiplier or 0.0))
        if rb > 0.0 and sm > 0.0 and stop_pct > 1e-9:
            return (rb * sm) / stop_pct
        if rb > 0.0 and sm > 0.0:
            return rb * sm
        return max(0.0, 100.0 * sm)

    def _build_close_economics(
        self,
        *,
        pos: Dict[str, Any],
        exit_price: float,
        qty_override: Optional[float] = None,
    ) -> Dict[str, float]:
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)
        if entry_price <= 0.0 or exit_price <= 0.0:
            return {
                "notional_usdt": 0.0,
                "qty_base": 0.0,
                "entry_fee_usdt": 0.0,
                "exit_fee_usdt": 0.0,
                "exit_notional_usdt": 0.0,
                "pnl_usd_realized": 0.0,
            }
        qty_full = float(pos.get("_qty_base", 0.0) or 0.0)
        qty = float(qty_override) if qty_override is not None else qty_full
        notional = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
        if qty <= 0.0 and notional > 0.0:
            qty = notional / entry_price
        if qty <= 0.0:
            sm = float(pos.get("_size_multiplier", 1.0) or 1.0)
            notional = self._estimate_entry_notional_usdt(
                pos=pos,
                entry_price=entry_price,
                size_multiplier=sm,
            )
            if notional > 0.0:
                qty = notional / entry_price
        if notional <= 0.0 and qty > 0.0:
            notional = qty * entry_price
        if qty_override is not None and qty_full > 0.0:
            slice_ratio = min(1.0, max(0.0, qty / qty_full))
            notional = float(pos.get("_entry_notional_usdt", 0.0) or 0.0) * slice_ratio
            if notional <= 0.0:
                notional = qty * entry_price
        entry_fee = float(pos.get("_entry_fee_usdt", 0.0) or 0.0)
        if qty_override is not None and qty_full > 0.0:
            entry_fee *= min(1.0, max(0.0, qty / qty_full))
        exit_notional = max(0.0, qty * float(exit_price))
        exit_fee = exit_notional * max(0.0, float(self.fee_rate or 0.0))
        is_long = str(pos.get("side", "")).upper() in {"LONG", "BUY"}
        gross = (
            (exit_price - entry_price) if is_long else (entry_price - exit_price)
        ) * qty
        realized = gross - entry_fee - exit_fee
        return {
            "notional_usdt": float(notional),
            "qty_base": float(max(0.0, qty)),
            "entry_fee_usdt": float(max(0.0, entry_fee)),
            "exit_fee_usdt": float(max(0.0, exit_fee)),
            "exit_notional_usdt": float(exit_notional),
            "pnl_usd_realized": float(realized),
        }

    def open_position(
        self,
        intent: Any,
        entry_bar: Dict[str, Any],
        features: Dict[str, Any],
        bar_minutes: Optional[int] = None,
    ) -> Optional[str]:
        """从 TradeIntent + 当前 bar 创建虚拟持仓 (调用共享 build_position_dict)"""
        self.last_open_reject_reason = ""
        _arch = str(getattr(intent, "archetype", "") or "").lower().strip()
        _sym = str(getattr(intent, "symbol", "") or "")
        exec_prof = getattr(intent, "execution_profile", {}) or {}
        exec_cons = (
            exec_prof.get("execution_constraints")
            if isinstance(exec_prof.get("execution_constraints"), dict)
            else {}
        )
        _accum_into = bool((exec_cons or {}).get("accumulate_same_archetype", False))

        _intent_side_raw = str(getattr(intent, "action", "") or "").strip().upper()
        _intent_side_bucket = (
            "LONG"
            if _intent_side_raw in {"LONG", "BUY"}
            else ("SHORT" if _intent_side_raw in {"SHORT", "SELL"} else "")
        )

        duplicate_pids: List[str] = []
        for _pid0, _p0 in self._positions.items():
            if (
                _p0.get("symbol", "") == _sym
                and str(_p0.get("archetype", "") or "").lower().strip() == _arch
            ):
                _ps = str(_p0.get("side", "") or "").strip().upper()
                _pb = (
                    "LONG"
                    if _ps in {"LONG", "BUY"}
                    else ("SHORT" if _ps in {"SHORT", "SELL"} else "")
                )
                if _intent_side_bucket and _pb == _intent_side_bucket:
                    duplicate_pids.append(str(_pid0))

        if duplicate_pids:
            if not _accum_into:
                # Same symbol/archetype: default path → caller may try_add_position().
                return None
            duplicate_pids.sort(
                key=lambda pid: bool(
                    self._positions.get(pid, {}).get("_is_add_position", False)
                )
            )
            parent_pid0 = duplicate_pids[0]
            parent_pos = self._positions.get(parent_pid0)
            if not isinstance(parent_pos, dict):
                return None

            # If the matched row is a child add-leg, collapse to its parent book.
            if bool(parent_pos.get("_is_add_position", False)):
                pp = str(parent_pos.get("_parent_pid") or "").strip()
                parent_pid0 = pp
                parent_pos = self._positions.get(pp)
                if not isinstance(parent_pos, dict):
                    return None

            _is_long_merge = _intent_side_bucket in {"", "LONG"}
            entry_price_fill, _fill_rej = _spot_entry_fill_price(
                entry_bar, exec_cons, is_long=_is_long_merge
            )
            if entry_price_fill is None or float(entry_price_fill) <= 0:
                self.last_open_reject_reason = _fill_rej or "spot_entry_no_fill"
                return None
            entry_price_fill = float(entry_price_fill)
            atr_live = (
                float(entry_bar.get("atr", 0) or 0.0)
                or float(features.get("atr", 0) or 0.0)
                or float(parent_pos.get("atr_at_entry", 0) or 0.0)
                or 0.0
            )
            if atr_live <= 0:
                return None

            bar_ts = entry_bar.get("timestamp")
            if isinstance(bar_ts, str):
                now_ts = pd.Timestamp(bar_ts).to_pydatetime()
            elif isinstance(bar_ts, pd.Timestamp):
                now_ts = bar_ts.to_pydatetime()
            elif isinstance(bar_ts, datetime):
                now_ts = bar_ts
            else:
                now_ts = datetime.now(timezone.utc)
            if now_ts.tzinfo is None:
                now_ts = now_ts.replace(tzinfo=timezone.utc)

            _day_key = _utc_calendar_day_str(now_ts)
            _max_legs_day = int(
                (exec_cons or {}).get("max_deploy_legs_per_day", 0) or 0
            )
            if _max_legs_day > 0:
                if (
                    _spot_symbol_deploy_legs_today(self, symbol=_sym, day_key=_day_key)
                    >= _max_legs_day
                ):
                    self.last_open_reject_reason = "spot_budget_daily_leg_cap"
                    return None

            min_gap_m = float(
                (exec_cons or {}).get("min_order_interval_minutes", 0) or 0.0
            )
            last_dep = parent_pos.get("_last_deploy_ts")
            if min_gap_m > 0 and last_dep is not None:
                try:
                    lt = pd.Timestamp(last_dep).to_pydatetime()
                    gap_minutes = (
                        pd.Timestamp(now_ts).to_pydatetime() - lt
                    ).total_seconds() / 60.0
                except Exception:
                    gap_minutes = 1.0e9
                if gap_minutes < min_gap_m:
                    self.last_open_reject_reason = "spot_budget_min_interval"
                    return None

            yaml_ml = int((exec_cons or {}).get("max_deploy_legs", 0) or 0)
            cur_legs = int(parent_pos.get("_deploy_leg_count", 1) or 1)
            cap_list: List[int] = []
            if yaml_ml > 0:
                cap_list.append(yaml_ml)
            if isinstance(self._spot_capital_budget, dict):
                try:
                    _tps = int(
                        self._spot_capital_budget.get("tranches_per_symbol") or 0
                    )
                except (TypeError, ValueError):
                    _tps = 0
                if _tps > 0:
                    cap_list.append(_tps)
                else:
                    try:
                        _tc = int(self._spot_capital_budget.get("tranche_count") or 0)
                    except (TypeError, ValueError):
                        _tc = 0
                    if _tc > 0:
                        cap_list.append(_tc)
            _max_le_eff = min(cap_list) if cap_list else (10**18)
            if cur_legs >= _max_le_eff:
                self.last_open_reject_reason = "spot_budget_tranches"
                return None

            ok_sz, scaled_add_m, leg_usdt, dk_slot, rej = _allocate_spot_accum_leg(
                self,
                archetype_lc=_arch,
                symbol=_sym,
                intent_base_add_m=float(getattr(intent, "size_multiplier", 1.0) or 1.0),
                parent_pos_for_merge=parent_pos,
                now_ts=now_ts,
                **_spot_regime_leg_kwargs(features, exec_cons),
            )
            if not ok_sz:
                self.last_open_reject_reason = rej or "spot_budget_room"
                return None
            add_m = float(scaled_add_m)
            leg_notional = float(max(0.0, leg_usdt))
            leg_qty = (
                float(leg_notional / entry_price_fill)
                if leg_notional > 0.0 and entry_price_fill > 0.0
                else 0.0
            )
            leg_entry_fee = leg_notional * max(0.0, float(self.fee_rate or 0.0))

            if isinstance(self._account_ledger, AccountLedger) and leg_notional > 0.0:
                try:
                    self._account_ledger.merge_lot(
                        lot_id=str(parent_pid0),
                        add_notional_usdt=leg_notional,
                        add_price=float(entry_price_fill),
                        fee_rate=float(self.fee_rate or 0.0),
                        allow_scale_down=False,
                    )
                except Exception:
                    pass

            old_m = float(parent_pos.get("_size_multiplier", 1.0) or 1.0)
            ep_old = float(parent_pos.get("entry_price", entry_price_fill) or 0.0)
            total_units = max(old_m + add_m, 1e-12)
            if is_spot_accum_archetype(str(_arch)) and leg_qty > 0.0:
                old_qty = float(parent_pos.get("_qty_base", 0.0) or 0.0)
                if old_qty <= 0.0:
                    old_notional = float(
                        parent_pos.get("_entry_notional_usdt", 0.0) or 0.0
                    )
                    if old_notional > 0.0 and ep_old > 0.0:
                        old_qty = old_notional / ep_old
                tot_qty = max(1e-12, old_qty + leg_qty)
                parent_pos["entry_price"] = (
                    ep_old * old_qty + entry_price_fill * leg_qty
                ) / tot_qty
                parent_pos["_qty_base"] = float(tot_qty)
                parent_pos["_entry_notional_usdt"] = (
                    float(parent_pos.get("_entry_notional_usdt", 0.0) or 0.0)
                    + leg_notional
                )
                parent_pos["_entry_fee_usdt"] = (
                    float(parent_pos.get("_entry_fee_usdt", 0.0) or 0.0) + leg_entry_fee
                )
            elif ep_old <= 0.0:
                parent_pos["entry_price"] = entry_price_fill
            else:
                parent_pos["entry_price"] = (
                    ep_old * old_m + entry_price_fill * add_m
                ) / total_units
            parent_pos["_size_multiplier"] = total_units
            parent_pos["_deploy_leg_count"] = cur_legs + 1
            parent_pos["_last_deploy_ts"] = now_ts

            parent_pos["_last_deploy_price"] = float(entry_price_fill)
            _record_spot_symbol_deploy_leg(self, symbol=_sym, day_key=_day_key)
            parent_pos["_accumulate_deploys"] = (
                int(parent_pos.get("_accumulate_deploys", 0) or 0) + 1
            )
            parent_pos.setdefault(
                "_first_entry_time",
                parent_pos.get("entry_time", now_ts),
            )

            dm = getattr(self, "_spot_daily_deploy_totals", None)
            if dm is not None and dk_slot and float(leg_usdt) > 0.0:
                dm[str(dk_slot)] = float(dm.get(str(dk_slot), 0.0) or 0.0) + float(
                    leg_usdt
                )
            parent_pos["_spot_quote_deployed"] = float(
                parent_pos.get("_spot_quote_deployed", 0.0) or 0.0
            ) + float(leg_usdt)

            if self._om_bridge:
                try:
                    self._om_bridge.record_open(
                        pid=str(parent_pid0),
                        symbol=str(parent_pos.get("symbol", "")),
                        side=parent_pos.get("side", ""),
                        entry_price=float(entry_price_fill),
                        size=add_m,
                        atr=float(atr_live),
                        stop_loss=None,
                        take_profit=None,
                        archetype=str(parent_pos.get("archetype", "")),
                        entry_time=now_ts,
                    )
                except Exception:
                    pass
            return str(parent_pid0)

        if len(self._positions) >= self.max_positions:
            return None

        pid = str(uuid.uuid4())[:12]

        _is_long_open = _intent_side_bucket in {"", "LONG"}
        entry_price, _fill_rej = _spot_entry_fill_price(
            entry_bar, exec_cons, is_long=_is_long_open
        )
        if entry_price is None or float(entry_price) <= 0:
            self.last_open_reject_reason = _fill_rej or "spot_entry_no_fill"
            return None
        entry_price = float(entry_price)
        # 直接取 "atr" 键 — 不用 pick_atr() 因为它会误匹配 macd_atr 等特征
        atr = float(entry_bar.get("atr", 0)) or float(features.get("atr", 0)) or 0.0

        # ATR=0 时拒绝开仓 — 无法计算止损/R-multiple
        if atr <= 0:
            return None

        # 解析 entry_time
        bar_ts = entry_bar.get("timestamp")
        if isinstance(bar_ts, str):
            entry_time = pd.Timestamp(bar_ts).to_pydatetime()
        elif isinstance(bar_ts, pd.Timestamp):
            entry_time = bar_ts.to_pydatetime()
        elif isinstance(bar_ts, datetime):
            entry_time = bar_ts
        else:
            entry_time = datetime.now(timezone.utc)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        _day_key = _utc_calendar_day_str(entry_time)
        _max_legs_day = int((exec_cons or {}).get("max_deploy_legs_per_day", 0) or 0)
        if _max_legs_day > 0:
            if (
                _spot_symbol_deploy_legs_today(self, symbol=_sym, day_key=_day_key)
                >= _max_legs_day
            ):
                self.last_open_reject_reason = "spot_budget_daily_leg_cap"
                return None

        ok_sz, scaled_add_m, leg_usdt, dk_slot, rej = _allocate_spot_accum_leg(
            self,
            archetype_lc=_arch,
            symbol=_sym,
            intent_base_add_m=float(getattr(intent, "size_multiplier", 1.0) or 1.0),
            parent_pos_for_merge=None,
            now_ts=entry_time,
            **_spot_regime_leg_kwargs(features, exec_cons),
        )
        if not ok_sz:
            self.last_open_reject_reason = rej or "spot_budget_room"
            return None
        scaled_m = float(scaled_add_m)
        if scaled_m <= 0.0:
            scaled_m = 1.0

        # 调用共享模块构建持仓 dict
        pos = build_position_dict(
            intent=intent,
            entry_price=entry_price,
            atr=atr,
            bar_minutes=bar_minutes or self.default_bar_minutes,
            entry_time=entry_time,
        )
        pos["archetype"] = getattr(intent, "archetype", "") or ""
        if str(_arch) == "spot_accum":
            from src.time_series_model.live.position_logic import (
                update_regime_lifecycle_state,
            )

            try:
                _ers = float(features.get("abc_macro_regime_score"))
            except (TypeError, ValueError):
                _ers = None
            update_regime_lifecycle_state(pos, macro_regime_score=_ers)
        # spot_accum_simple: no regime lifecycle state on open
        # 存储 size_multiplier（PCM regime × 缩放）；spot 预算下按 leg_usd / unit_notional 缩放
        pos["_size_multiplier"] = scaled_m
        pos["_deploy_leg_count"] = 1
        pos["_last_deploy_ts"] = entry_time
        if is_spot_accum_archetype(str(_arch)):
            _record_spot_symbol_deploy_leg(self, symbol=_sym, day_key=_day_key)
        _is_spot = is_spot_accum_archetype(str(_arch))
        if _is_spot:
            entry_notional = float(max(0.0, leg_usdt))
        else:
            entry_notional = self._estimate_entry_notional_usdt(
                pos=pos,
                entry_price=float(entry_price),
                size_multiplier=float(scaled_m),
            )
        qty_base = (
            float(entry_notional / entry_price)
            if entry_notional > 0.0 and float(entry_price) > 0.0
            else 0.0
        )
        entry_fee_usdt = entry_notional * max(0.0, float(self.fee_rate or 0.0))

        pos["_spot_quote_deployed"] = float(leg_usdt)
        pos["_entry_notional_usdt"] = float(entry_notional)
        pos["_qty_base"] = float(qty_base)
        pos["_entry_fee_usdt"] = float(entry_fee_usdt)
        dm2 = getattr(self, "_spot_daily_deploy_totals", None)
        if dm2 is not None and dk_slot and float(leg_usdt) > 0.0:
            dm2[str(dk_slot)] = float(dm2.get(str(dk_slot), 0.0) or 0.0) + float(
                leg_usdt
            )

        if isinstance(self._account_ledger, AccountLedger) and entry_notional > 0.0:
            try:
                self._account_ledger.open_lot(
                    lot_id=str(pid),
                    strategy=str(pos.get("archetype", "")),
                    symbol=str(pos.get("symbol", "")),
                    side=str(pos.get("side", "")),
                    notional_usdt=float(entry_notional),
                    entry_price=float(entry_price),
                    fee_rate=float(self.fee_rate or 0.0),
                    opened_at=entry_time,
                    cash_mode=("cash_notional" if _is_spot else "fee_only"),
                    allow_scale_down=False,
                )
            except Exception:
                pass

        self._positions[pid] = pos

        # 写入 order_management DB
        if self._om_bridge:
            self._om_bridge.record_open(
                pid=pid,
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                size=scaled_m,
                atr=atr,
                stop_loss=pos.get("stop_loss"),
                take_profit=pos.get("take_profit"),
                archetype=pos.get("archetype", ""),
                entry_time=entry_time,
            )
        return pid

    def update(self, bar_1min: Dict[str, Any]) -> List[ClosedTrade]:
        """用 1min bar 更新所有持仓 — 调用共享 enforce_position"""
        if not self._positions:
            return []

        bar_high = float(bar_1min.get("high", 0))
        bar_low = float(bar_1min.get("low", 0))
        bar_close = float(bar_1min.get("close", 0))

        bar_ts = bar_1min.get("timestamp")
        if isinstance(bar_ts, str):
            now = pd.Timestamp(bar_ts).to_pydatetime()
        elif isinstance(bar_ts, pd.Timestamp):
            now = bar_ts.to_pydatetime()
        elif isinstance(bar_ts, datetime):
            now = bar_ts
        else:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        closed = []
        to_remove = []
        parent_close_meta: Dict[str, Dict[str, Any]] = {}

        for pid, pos in self._positions.items():
            if bool(pos.get("_is_add_position", False)) and bool(
                pos.get("_inherit_parent_stop", False)
            ):
                parent_pid = str(pos.get("_parent_pid", "") or "")
                parent = self._positions.get(parent_pid)
                if parent is not None and parent.get("stop_loss_price") is not None:
                    # tighten-only：子仓 SL 只向入场有利方向跟随父仓
                    new_sl = float(parent.get("stop_loss_price"))
                    old_sl = pos.get("stop_loss_price")
                    is_long = str(pos.get("side", "")).upper() in {"LONG", "BUY"}
                    if old_sl is None:
                        pos["stop_loss_price"] = new_sl
                    else:
                        try:
                            old_sl_f = float(old_sl)
                            if is_long and new_sl > old_sl_f:
                                pos["stop_loss_price"] = new_sl
                            elif (not is_long) and new_sl < old_sl_f:
                                pos["stop_loss_price"] = new_sl
                        except (TypeError, ValueError):
                            pos["stop_loss_price"] = new_sl

            # vwap1200: 特征表里的 pv 只在 primary-TF 收盘更新；1m 上用冻结的 VWAP 水平对当前 close 重算 pv，
            # 否则价格在两根 2H 之间穿越死区不会触发结构性出场。
            macro_pv: Optional[float] = self._macro_tp_vwap_position
            _lvl = getattr(self, "_macro_tp_vwap_level", None)
            if _lvl is not None and bar_close > 0:
                try:
                    lv = float(_lvl)
                    bc = float(bar_close)
                    if lv == lv and bc == bc and bc > 0.0:
                        live_pv = (bc - lv) / bc
                        if live_pv == live_pv:
                            macro_pv = max(-1.0, min(1.0, float(live_pv)))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # ema1200: 同理冻结 EMA1200 水平，1m 重算 position
            ema_1200_pv: Optional[float] = self._ema_1200_position
            _ema_lvl = getattr(self, "_ema_1200_level", None)
            if _ema_lvl is not None and bar_close > 0:
                try:
                    elv = float(_ema_lvl)
                    bc = float(bar_close)
                    if elv == elv and bc == bc and bc > 0.0:
                        live_ev = (bc - elv) / bc
                        if live_ev == live_ev:
                            ema_1200_pv = max(-1.0, min(1.0, float(live_ev)))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            if is_spot_accum_archetype(str(pos.get("archetype", "") or "")):
                _partial = maybe_spot_simple_partial_sell(
                    pos, price_close=float(bar_close), now=now
                )
                if _partial is not None:
                    sell_qty, part_reason = _partial
                    econ_part = self._build_close_economics(
                        pos=pos,
                        exit_price=float(bar_close),
                        qty_override=float(sell_qty),
                    )
                    entry_price_p = float(pos.get("entry_price", 0.0) or 0.0)
                    risk_p = (
                        float(pos.get("initial_risk_distance", 0.0) or 0.0)
                        or float(pos.get("atr_at_entry", 0.0) or 0.0)
                        or 0.0
                    )
                    is_long_p = pos.get("side") in {"LONG", "BUY"}
                    pnl_usd_p = (
                        (float(bar_close) - entry_price_p) * float(sell_qty)
                        if is_long_p
                        else (entry_price_p - float(bar_close)) * float(sell_qty)
                    )
                    _raw_r_p = pnl_usd_p / risk_p if risk_p > 0 else 0.0
                    trade_p = ClosedTrade(
                        symbol=pos.get("symbol", ""),
                        side=pos["side"],
                        entry_price=entry_price_p,
                        exit_price=float(bar_close),
                        entry_time=pos["entry_time"],
                        exit_time=now,
                        atr_at_entry=pos.get("atr_at_entry", 0),
                        pnl_r=_raw_r_p * float(pos.get("_size_multiplier", 1.0) or 1.0),
                        pnl_usd=pnl_usd_p,
                        exit_reason=part_reason,
                        pnl_usd_realized=float(
                            econ_part.get("pnl_usd_realized", 0.0) or 0.0
                        ),
                        notional_usdt=float(econ_part.get("notional_usdt", 0.0) or 0.0),
                        qty_base=float(sell_qty),
                        entry_fee_usdt=float(
                            econ_part.get("entry_fee_usdt", 0.0) or 0.0
                        ),
                        exit_fee_usdt=float(econ_part.get("exit_fee_usdt", 0.0) or 0.0),
                        exit_notional_usdt=float(
                            econ_part.get("exit_notional_usdt", 0.0) or 0.0
                        ),
                        archetype=pos.get("archetype", ""),
                        bars_held=pos.get("bars_counted", 0),
                        size_multiplier=pos.get("_size_multiplier", 1.0),
                        atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                        effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                        sizing_stop_source=pos.get("sizing_stop_source", ""),
                    )
                    closed.append(trade_p)
                    self.closed_trades.append(trade_p)
                    apply_partial_sell_to_position(
                        pos, sell_qty=float(sell_qty), exit_price=float(bar_close)
                    )
                    pos["_profit_ladder_last_sell_day"] = pd.Timestamp(now).strftime(
                        "%Y-%m-%d"
                    )
                    if float(pos.get("_qty_base", 0.0) or 0.0) <= 1e-12:
                        to_remove.append(pid)
                        continue

            # 调用共享 7 步持仓管理 (structural: EMA200 / vwap1200 / ema1200)
            close_reason, exit_price = enforce_position(
                pos,
                price_high=bar_high,
                price_low=bar_low,
                price_close=bar_close,
                now=now,
                default_bar_minutes=self.default_bar_minutes,
                structural_price=self._structural_price,
                macro_tp_vwap_position=macro_pv,
                ema_1200_position=ema_1200_pv,
                macro_cycle_exit_signal=self._macro_cycle_exit_signal,
                macro_regime_score=self._macro_regime_score,
                primary_tf_atr=self._primary_tf_atr,
                wide_sr_upper_px=getattr(self, "_wide_sr_upper_px", None),
                wide_sr_lower_px=getattr(self, "_wide_sr_lower_px", None),
            )

            if close_reason:
                entry_price = pos["entry_price"]
                # 用 initial_risk_distance (= initial_r × ATR) 归一化 R-multiple
                # 与研究回测 (backtest_execution_layer.py) 保持一致
                risk = (
                    pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
                )
                is_long = pos["side"] in {"LONG", "BUY"}
                pnl_usd = (
                    (exit_price - entry_price)
                    if is_long
                    else (entry_price - exit_price)
                )
                _raw_r = pnl_usd / risk if risk > 0 else 0.0
                # 扣除双边手续费 (开仓+平仓)
                # fee_r = (entry_price + exit_price) × fee_rate / risk
                if self.fee_rate > 0 and risk > 0:
                    fee_r = (entry_price + exit_price) * self.fee_rate / risk
                    _raw_r -= fee_r
                # 应用 position scale — 与向量回测 exec_returns *= _position_scale 对齐
                pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)

                # 归一化 exit_reason: 与向量回测对齐命名
                normalized_reason = close_reason
                if close_reason == "stop_loss":
                    normalized_reason = (
                        "trailing_sl" if pos.get("trailing_activated") else "sl"
                    )
                elif close_reason == "take_profit":
                    normalized_reason = "tp"
                elif close_reason == "time_stop":
                    normalized_reason = "timeout"
                econ = self._build_close_economics(
                    pos=pos,
                    exit_price=float(exit_price),
                )

                trade = ClosedTrade(
                    symbol=pos.get("symbol", ""),
                    side=pos["side"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=pos["entry_time"],
                    exit_time=now,
                    atr_at_entry=pos.get("atr_at_entry", 0),
                    pnl_r=pnl_r,
                    pnl_usd=pnl_usd,
                    exit_reason=normalized_reason,
                    pnl_usd_realized=float(econ.get("pnl_usd_realized", 0.0)),
                    notional_usdt=float(econ.get("notional_usdt", 0.0)),
                    qty_base=float(econ.get("qty_base", 0.0)),
                    entry_fee_usdt=float(econ.get("entry_fee_usdt", 0.0)),
                    exit_fee_usdt=float(econ.get("exit_fee_usdt", 0.0)),
                    exit_notional_usdt=float(econ.get("exit_notional_usdt", 0.0)),
                    archetype=pos.get("archetype", ""),
                    bars_held=pos.get("bars_counted", 0),
                    is_add_position=pos.get("_is_add_position", False),
                    is_reverse=False,
                    size_multiplier=pos.get("_size_multiplier", 1.0),
                    atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                    effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                    sizing_stop_source=pos.get("sizing_stop_source", ""),
                    breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
                )
                closed.append(trade)
                self.closed_trades.append(trade)
                if isinstance(self._account_ledger, AccountLedger):
                    try:
                        self._account_ledger.close_lot(
                            lot_id=str(pid),
                            exit_price=float(exit_price),
                            fee_rate=float(self.fee_rate or 0.0),
                        )
                    except Exception:
                        pass
                if str(pos.get("archetype", "") or "").lower() == "fer":
                    try:
                        from src.time_series_model.live.fer_diagnostics import (
                            record_fer_exit,
                        )

                        record_fer_exit(
                            pos=dict(pos),
                            close_reason_raw=str(close_reason),
                            exit_reason_normalized=str(normalized_reason),
                            exit_price=float(exit_price),
                            now=now,
                            pnl_r=float(pnl_r),
                        )
                    except Exception:
                        pass
                to_remove.append(pid)
                if not bool(pos.get("_is_add_position", False)):
                    parent_close_meta[str(pid)] = {
                        "exit_price": float(exit_price),
                        "normalized_reason": str(normalized_reason),
                        "breakeven_locked": bool(pos.get("breakeven_locked", False)),
                    }

                # 写入 order_management DB
                if self._om_bridge:
                    self._om_bridge.record_close(
                        pid=pid,
                        exit_price=exit_price,
                        exit_time=now,
                        exit_reason=close_reason,
                        pnl_r=pnl_r,
                    )

        # 母仓退出时，默认强制同 bar 带走其加仓子仓（可通过 _share_parent_exit=False 关闭）
        if parent_close_meta:
            for pid, pos in self._positions.items():
                if pid in to_remove:
                    continue
                if not bool(pos.get("_is_add_position", False)):
                    continue
                if not bool(pos.get("_share_parent_exit", True)):
                    continue
                parent_pid = str(pos.get("_parent_pid", "") or "")
                meta = parent_close_meta.get(parent_pid)
                if not meta:
                    continue
                entry_price = float(pos.get("entry_price", 0.0) or 0.0)
                risk = (
                    float(pos.get("initial_risk_distance", 0.0) or 0.0)
                    or float(pos.get("atr_at_entry", 0.0) or 0.0)
                    or 0.0
                )
                is_long = pos.get("side") in {"LONG", "BUY"}
                forced_exit = float(meta.get("exit_price", bar_close) or bar_close)
                pnl_usd = (
                    (forced_exit - entry_price)
                    if is_long
                    else (entry_price - forced_exit)
                )
                _raw_r = pnl_usd / risk if risk > 0 else 0.0
                if self.fee_rate > 0 and risk > 0:
                    fee_r = (entry_price + forced_exit) * self.fee_rate / risk
                    _raw_r -= fee_r
                pnl_r = _raw_r * float(pos.get("_size_multiplier", 1.0) or 1.0)
                econ = self._build_close_economics(
                    pos=pos,
                    exit_price=float(forced_exit),
                )
                trade = ClosedTrade(
                    symbol=pos.get("symbol", ""),
                    side=pos["side"],
                    entry_price=entry_price,
                    exit_price=forced_exit,
                    entry_time=pos["entry_time"],
                    exit_time=now,
                    atr_at_entry=pos.get("atr_at_entry", 0),
                    pnl_r=pnl_r,
                    pnl_usd=pnl_usd,
                    exit_reason=str(meta.get("normalized_reason", "sl")),
                    pnl_usd_realized=float(econ.get("pnl_usd_realized", 0.0)),
                    notional_usdt=float(econ.get("notional_usdt", 0.0)),
                    qty_base=float(econ.get("qty_base", 0.0)),
                    entry_fee_usdt=float(econ.get("entry_fee_usdt", 0.0)),
                    exit_fee_usdt=float(econ.get("exit_fee_usdt", 0.0)),
                    exit_notional_usdt=float(econ.get("exit_notional_usdt", 0.0)),
                    archetype=pos.get("archetype", ""),
                    bars_held=pos.get("bars_counted", 0),
                    is_add_position=True,
                    size_multiplier=pos.get("_size_multiplier", 1.0),
                    atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                    effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                    sizing_stop_source=pos.get("sizing_stop_source", ""),
                    breakeven_locked_at_exit=bool(meta.get("breakeven_locked", False)),
                )
                closed.append(trade)
                self.closed_trades.append(trade)
                if isinstance(self._account_ledger, AccountLedger):
                    try:
                        self._account_ledger.close_lot(
                            lot_id=str(pid),
                            exit_price=float(forced_exit),
                            fee_rate=float(self.fee_rate or 0.0),
                        )
                    except Exception:
                        pass
                to_remove.append(pid)

        for pid in to_remove:
            self._positions.pop(pid, None)

        # 计数存活持仓的 bar 数
        for pos in self._positions.values():
            pos["bars_counted"] = pos.get("bars_counted", 0) + 1

        return closed

    def force_close_all(self, price: float, now: datetime) -> List[ClosedTrade]:
        """回测结束时关闭所有持仓"""
        closed = []
        for pid, pos in list(self._positions.items()):
            is_long = pos["side"] in {"LONG", "BUY"}
            entry_price = pos["entry_price"]
            risk = pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
            pnl_usd = (price - entry_price) if is_long else (entry_price - price)
            _raw_r = pnl_usd / risk if risk > 0 else 0.0
            if self.fee_rate > 0 and risk > 0:
                fee_r = (entry_price + price) * self.fee_rate / risk
                _raw_r -= fee_r
            pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)
            econ = self._build_close_economics(
                pos=pos,
                exit_price=float(price),
            )
            trade = ClosedTrade(
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                exit_price=price,
                entry_time=pos["entry_time"],
                exit_time=now,
                atr_at_entry=pos.get("atr_at_entry", 0),
                pnl_r=pnl_r,
                pnl_usd=pnl_usd,
                exit_reason="end_of_backtest",
                pnl_usd_realized=float(econ.get("pnl_usd_realized", 0.0)),
                notional_usdt=float(econ.get("notional_usdt", 0.0)),
                qty_base=float(econ.get("qty_base", 0.0)),
                entry_fee_usdt=float(econ.get("entry_fee_usdt", 0.0)),
                exit_fee_usdt=float(econ.get("exit_fee_usdt", 0.0)),
                exit_notional_usdt=float(econ.get("exit_notional_usdt", 0.0)),
                archetype=pos.get("archetype", ""),
                bars_held=pos.get("bars_counted", 0),
                is_add_position=pos.get("_is_add_position", False),
                is_reverse=False,
                size_multiplier=pos.get("_size_multiplier", 1.0),
                atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                sizing_stop_source=pos.get("sizing_stop_source", ""),
                breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
            )
            closed.append(trade)
            self.closed_trades.append(trade)
            if isinstance(self._account_ledger, AccountLedger):
                try:
                    self._account_ledger.close_lot(
                        lot_id=str(pid),
                        exit_price=float(price),
                        fee_rate=float(self.fee_rate or 0.0),
                    )
                except Exception:
                    pass

            # 写入 order_management DB
            if self._om_bridge:
                self._om_bridge.record_close(
                    pid=pid,
                    exit_price=price,
                    exit_time=now,
                    exit_reason="end_of_backtest",
                    pnl_r=pnl_r,
                )
        self._positions.clear()
        return closed

    def close_by_archetype(
        self, archetype: str, close_price: float, close_time: datetime
    ) -> List[ClosedTrade]:
        """关闭指定 archetype 的所有仓位 (遗留接口, 竞争驱逐已移除)"""
        closed = []
        to_remove = []
        for pid, pos in self._positions.items():
            if pos.get("archetype", "").lower() != archetype.lower():
                continue
            is_long = pos["side"] in {"LONG", "BUY"}
            entry_price = pos["entry_price"]
            risk = pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
            pnl_usd = (
                (close_price - entry_price) if is_long else (entry_price - close_price)
            )
            _raw_r = pnl_usd / risk if risk > 0 else 0.0
            if self.fee_rate > 0 and risk > 0:
                fee_r = (entry_price + close_price) * self.fee_rate / risk
                _raw_r -= fee_r
            pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)
            econ = self._build_close_economics(
                pos=pos,
                exit_price=float(close_price),
            )
            trade = ClosedTrade(
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                exit_price=close_price,
                entry_time=pos["entry_time"],
                exit_time=close_time,
                atr_at_entry=pos.get("atr_at_entry", 0),
                pnl_r=pnl_r,
                pnl_usd=pnl_usd,
                exit_reason="evicted",
                pnl_usd_realized=float(econ.get("pnl_usd_realized", 0.0)),
                notional_usdt=float(econ.get("notional_usdt", 0.0)),
                qty_base=float(econ.get("qty_base", 0.0)),
                entry_fee_usdt=float(econ.get("entry_fee_usdt", 0.0)),
                exit_fee_usdt=float(econ.get("exit_fee_usdt", 0.0)),
                exit_notional_usdt=float(econ.get("exit_notional_usdt", 0.0)),
                archetype=pos.get("archetype", ""),
                bars_held=pos.get("bars_counted", 0),
                is_add_position=pos.get("_is_add_position", False),
                is_reverse=False,
                size_multiplier=pos.get("_size_multiplier", 1.0),
                atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                sizing_stop_source=pos.get("sizing_stop_source", ""),
                breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
            )
            closed.append(trade)
            self.closed_trades.append(trade)
            if isinstance(self._account_ledger, AccountLedger):
                try:
                    self._account_ledger.close_lot(
                        lot_id=str(pid),
                        exit_price=float(close_price),
                        fee_rate=float(self.fee_rate or 0.0),
                    )
                except Exception:
                    pass
            to_remove.append(pid)
        for pid in to_remove:
            self._positions.pop(pid, None)
        return closed

    def try_add_position(
        self,
        intent: Any,
        entry_bar: Dict[str, Any],
        features: Dict[str, Any],
        executor: ConstitutionExecutor,
        runtime_state: ConstitutionRuntimeState,
        bar_minutes: Optional[int] = None,
        *,
        skip_signal_trigger: bool = False,
    ) -> Optional[str]:
        """加仓模拟: 复用实盘 validate_add_position / record_add_position。

        与实盘 constitution_executor.py 100% 同一份代码:
          1. executor.validate_add_position() — 策略/次数/利润锁定检查
          2. executor.record_add_position() — 更新 ConstitutionRuntimeState

        skip_signal_trigger:
            True 时只检查 min_current_r_by_add（及 ATR 换算），不跑 BPC/ME 等特征 trigger。
            供 execution.add_position.trigger.type=float_r_ladder_only 浮盈阶梯加仓（事件回测）。

        Returns:
            position_id if added, None if rejected
        """
        self.last_add_reject_reason = ""
        self.last_add_attempt_signal = {}
        archetype = getattr(intent, "archetype", "").lower().strip()
        new_side = (
            "LONG"
            if str(getattr(intent, "action", "")).upper() in ("LONG", "BUY")
            else "SHORT"
        )

        # 1. 查找同 symbol 同 side 同 archetype 的已有持仓
        parent_pid = None
        parent_pos = None
        for pid, pos in self._positions.items():
            _pos_arch = str(pos.get("archetype", "") or "").lower().strip()
            if (
                pos.get("symbol", "") == intent.symbol
                and pos["side"] == new_side
                and _pos_arch == archetype
            ):
                parent_pid = pid
                parent_pos = pos
                break

        if parent_pos is None:
            self.last_add_reject_reason = "no_parent_position"
            return None  # 没有已有持仓，不是加仓场景

        _pol = getattr(self, "_srb_add_policy", None)
        if archetype == "srb" and _pol:
            _ok, _why = srb_add_position_allowed(features or {}, _pol)
            if not _ok:
                self.last_add_reject_reason = _why
                return None

        # 2. 计算 current_r (用于 validate_add_position)
        entry_price = parent_pos["entry_price"]
        risk = (
            parent_pos.get("initial_risk_distance")
            or parent_pos.get("atr_at_entry", 0)
            or 1
        )
        is_long = parent_pos["side"] in {"LONG", "BUY"}
        current_price = float(entry_bar.get("close", 0))
        current_r = (
            (
                (current_price - entry_price)
                if is_long
                else (entry_price - current_price)
            )
            / risk
            if risk > 0
            else 0.0
        )

        # 2a-Phase-D: 加仓事后形态门（srb_add_position_policy.post_hoc_shape_gate）。
        # 仅 SRB：若 post_hoc_shape_gate 下任一子项 enabled，追加形态确认。
        if archetype == "srb" and _pol:
            _gate_cfg = _pol.get("post_hoc_shape_gate") or {}
            if any(
                bool((_gate_cfg.get(_k) or {}).get("enabled", False))
                for _k in (
                    "retrace_guard",
                    "recent_momentum",
                    "trend_r2_gate",
                    "wide_sr_expansion",
                    "trend_health_gate",
                )
            ):
                # 计算母仓 MFE R（用 initial_risk_distance 归一化，与退出逻辑对齐）
                _hwm = parent_pos.get("high_water_mark")
                _lwm = parent_pos.get("low_water_mark")
                if is_long and _hwm is not None:
                    _mfe_r = (float(_hwm) - entry_price) / risk if risk > 0 else 0.0
                elif (not is_long) and _lwm is not None:
                    _mfe_r = (entry_price - float(_lwm)) / risk if risk > 0 else 0.0
                else:
                    _mfe_r = max(0.0, current_r)
                # recent_net_move_atr：近 N primary close 净变化 / ATR（与 mother 方向同向为正）
                _shape_feat = dict(features or {})
                _shape_feat["mfe_r"] = _mfe_r
                _shape_feat["current_r"] = current_r
                if "recent_net_move_atr" not in _shape_feat:
                    _rn = (_gate_cfg.get("recent_momentum") or {}).get(
                        "lookback_bars", 6
                    ) or 6
                    try:
                        _rn = int(_rn)
                    except (TypeError, ValueError):
                        _rn = 6
                    _buf = getattr(self, "_primary_close_buffer", None) or []
                    _atr_now = float(
                        features.get("atr") or parent_pos.get("atr_at_entry") or 0.0
                    )
                    if len(_buf) >= 2 and _atr_now > 0:
                        _tail = _buf[-max(2, min(_rn + 1, len(_buf))) :]
                        _net = _tail[-1] - _tail[0]
                        # 保留带符号：正 = 上涨，负 = 下跌；gate 内部按 side 判方向。
                        _shape_feat["recent_net_move_atr"] = _net / _atr_now
                # bars_since_mother_entry（E4）：用 entry_bar 的 timestamp 与 parent.entry_time
                # 的差，按 bar_minutes 转成 primary bar 数。缺失时兜底 0。
                try:
                    _parent_et = parent_pos.get("entry_time")
                    _now_ts = entry_bar.name if hasattr(entry_bar, "name") else None
                    _bm = int(
                        parent_pos.get("bar_minutes")
                        or self._primary_bar_minutes
                        or 240
                    )
                    if _parent_et is not None and _now_ts is not None and _bm > 0:
                        _pt = pd.Timestamp(_parent_et)
                        _nt = pd.Timestamp(_now_ts)
                        if _pt.tzinfo is None:
                            _pt = _pt.tz_localize("UTC")
                        if _nt.tzinfo is None:
                            _nt = _nt.tz_localize("UTC")
                        _delta_min = (_nt - _pt).total_seconds() / 60.0
                        _shape_feat["bars_since_mother_entry"] = max(
                            0.0, _delta_min / float(_bm)
                        )
                except Exception:
                    pass
                # wide_sr_dist_atr：从 features 直接读（已存在）
                _mother_ctx = {
                    "side": parent_pos.get("side"),
                    "entry_wide_sr_dist_atr": parent_pos.get(
                        "_srb_entry_wide_sr_dist_atr"
                    ),
                }
                _rej, _why = should_reject_srb_add_by_shape(
                    _shape_feat, _mother_ctx, _gate_cfg
                )
                if _rej:
                    self.last_add_reject_reason = _why
                    return None

        # 2b. 找出同 symbol + 同 direction 的活跃仓位
        same_sym_dir = [
            p
            for p in self._positions.values()
            if p.get("symbol", "") == intent.symbol and p["side"] == new_side
        ]

        rec = runtime_state.add_position.positions.get(parent_pid)
        next_add_no = int(rec.add_count) + 1 if rec is not None else 1

        # 3. 复用实盘 validate_add_position (raises ConstitutionViolation on failure)
        try:
            executor.validate_add_position(
                st=runtime_state,
                position_id=parent_pid,
                archetype=archetype,
                current_r=current_r,
                locked_profit=parent_pos.get("breakeven_locked", False),
                position_action=new_side,
            )
        except ConstitutionViolation as exc:
            if str(getattr(exc, "code", "")).strip().upper() == (
                "ADD_POSITION_LOCKED_PROFIT_REQUIRED"
            ):
                self.last_add_reject_reason = "locked_profit_required"
            else:
                self.last_add_reject_reason = "constitution_reject"
            return None

        add_rules = dict(
            executor.resolve_add_position_for_strategy(
                archetype, position_action=new_side
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
        if next_add_no > _shared_resolve_add_position_max_times(add_rules):
            self.last_add_reject_reason = "max_add_times"
            return None
        signal = dict(features or {})
        signal["add_position_seq"] = next_add_no
        signal["current_r"] = float(current_r)
        signal["position_action"] = str(new_side)
        signal.setdefault("close", current_price)
        _atr_parent = float(parent_pos.get("atr_at_entry", 0) or 0)
        _risk_parent = float(parent_pos.get("initial_risk_distance", 0) or 0)
        if _atr_parent > 0 and _risk_parent > 0:
            signal["parent_initial_r"] = _risk_parent / _atr_parent
        _risk_frac = float(
            executor.resolve_risk_for_strategy(archetype, position_action=new_side)
        )
        _current_lev = 0.0
        _current_notional_frac = 0.0
        for _pos in same_sym_dir:
            _stop_dist = float(_pos.get("initial_risk_distance", 0.0) or 0.0)
            _ep = float(_pos.get("entry_price", 0.0) or 0.0)
            _stop_pct = _stop_dist / _ep if _ep > 0 else 0.0
            if _stop_pct <= 1e-9:
                continue
            _mult = float(_pos.get("_size_multiplier", 1.0) or 1.0)
            _current_lev += _risk_frac * _mult / _stop_pct
            _current_notional_frac += _risk_frac * _mult
        _parent_ep = float(parent_pos.get("entry_price", 0.0) or 0.0)
        _parent_stop_pct = (_risk_parent / _parent_ep) if _parent_ep > 0 else 0.0
        signal["base_leverage_unit"] = (
            _risk_frac / _parent_stop_pct if _parent_stop_pct > 1e-9 else 1.0
        )
        signal["current_leverage"] = float(_current_lev)
        signal["base_notional_frac"] = float(_risk_frac)
        signal["current_notional_frac"] = float(_current_notional_frac)
        signal["equity_usd"] = float(features.get("equity", 0.0) or 0.0)
        self.last_add_attempt_signal = dict(signal)
        if skip_signal_trigger:
            _thr = resolve_add_position_min_current_r(add_rules, next_add_no, signal)
            if current_r < _thr:
                self.last_add_reject_reason = "trigger_not_met"
                return None
        elif not _shared_validate_add_position_trigger(
            archetype=archetype,
            direction=1 if new_side == "LONG" else -1,
            signal=signal,
            add_position_cfg=add_rules,
            current_r=current_r,
        ):
            _thr = resolve_add_position_min_current_r(add_rules, next_add_no, signal)
            if current_r < _thr:
                self.last_add_reject_reason = "add_min_current_r"
            else:
                _trg = dict(add_rules.get("trigger") or {})
                _tt = str(_trg.get("type", "")).strip().lower()
                if _tt in {"bpc_follow_signal", "follow_signal"}:
                    self.last_add_reject_reason = "add_bpc_breakout_mismatch"
                else:
                    self.last_add_reject_reason = "add_trigger_feature_rules"
            return None
        _ok_gate, _gate_why = _shared_add_regime_gate_allows(signal, add_rules)
        if not _ok_gate:
            self.last_add_reject_reason = f"add_regime_gate:{_gate_why}"
            return None
        add_mult = _resolve_add_position_size_multiplier(add_rules, next_add_no, signal)
        parent_mult = float(parent_pos.get("_size_multiplier", 1.0) or 1.0)
        projected_lev = float(_current_lev) + float(
            signal.get("base_leverage_unit", 1.0)
        ) * float(add_mult)
        projected_notional = float(_current_notional_frac) + float(_risk_frac) * float(
            parent_mult
        ) * float(add_mult)
        self.max_observed_leverage = max(self.max_observed_leverage, projected_lev)
        self.max_observed_notional_frac = max(
            self.max_observed_notional_frac, projected_notional
        )

        # 4. 记录加仓 (更新 ConstitutionRuntimeState — 同实盘 record_add_position)
        executor.record_add_position(
            st=runtime_state,
            position_id=parent_pid,
            current_r=current_r,
            locked_profit=parent_pos.get("breakeven_locked", False),
        )

        # 5. 开加仓仓位
        pid = str(uuid.uuid4())[:12]
        pos = build_position_dict(
            intent=intent,
            entry_price=float(entry_bar.get("close", 0)),
            atr=float(entry_bar.get("atr", 0)) or float(features.get("atr", 0)) or 0.0,
            bar_minutes=bar_minutes or self.default_bar_minutes,
            entry_time=(
                pd.Timestamp(entry_bar.get("timestamp")).to_pydatetime()
                if entry_bar.get("timestamp") is not None
                else datetime.now(timezone.utc)
            ),
        )
        # float_r_ladder_only 等 intent 常仅有 add_position、无 rr_constraints → 无 structural_exit。
        # 否则 vwap1200 只在首仓 enforce，加仓腿仅宽止损/共享止损离场，图上会像「结构止损失效」。
        _p_se = parent_pos.get("structural_exit")
        if _p_se and not pos.get("structural_exit"):
            pos["structural_exit"] = str(_p_se)
        _p_rlc = parent_pos.get("regime_lifecycle_exit")
        if isinstance(_p_rlc, dict) and _p_rlc and not pos.get("regime_lifecycle_exit"):
            pos["regime_lifecycle_exit"] = dict(_p_rlc)
        pos["_is_add_position"] = True
        pos["_parent_pid"] = parent_pid
        pos["_add_position_seq"] = next_add_no
        pos["_share_parent_exit"] = bool(add_rules.get("share_parent_exit", True))
        pos["_inherit_parent_stop"] = bool(add_rules.get("inherit_parent_stop", False))
        if bool(pos["_inherit_parent_stop"]):
            parent_sl = parent_pos.get("stop_loss_price")
            if parent_sl is not None:
                pos["stop_loss_price"] = float(parent_sl)
            pos["breakeven_enabled"] = False
            pos["activation_r"] = None
            pos["trailing_activated"] = False
        # 加仓继承父仓的 regime scale，但风险预算按 add_size_multipliers 缩小
        pos["_size_multiplier"] = parent_mult * add_mult
        self._positions[pid] = pos
        self.last_add_reject_reason = ""
        return pid


def _count_open_add_legs_for_parent(sim: PositionSimulator, parent_pid: str) -> int:
    n = 0
    pp = str(parent_pid)
    for pos in sim._positions.values():
        if (
            bool(pos.get("_is_add_position", False))
            and str(pos.get("_parent_pid") or "") == pp
        ):
            n += 1
    return n


def _rehydrate_add_position_runtime_from_simulator(
    sim: PositionSimulator, st: ConstitutionRuntimeState
) -> None:
    for pid, pos in sim._positions.items():
        if bool(pos.get("_is_add_position", False)):
            continue
        n = _count_open_add_legs_for_parent(sim, str(pid))
        if n <= 0:
            continue
        st.add_position.positions[str(pid)] = AddPositionRecord(
            position_id=str(pid), add_count=n
        )


def _load_add_position_runtime_from_resume(
    resume_blob: Dict[str, Any], st: ConstitutionRuntimeState
) -> int:
    raw = (resume_blob or {}).get("add_position_state") or {}
    positions = raw.get("positions") if isinstance(raw, dict) else None
    if not isinstance(positions, dict):
        return 0
    n = 0
    for pid, row in positions.items():
        if not isinstance(row, dict):
            continue
        st.add_position.positions[str(pid)] = AddPositionRecord(
            position_id=str(row.get("position_id") or pid),
            add_count=int(row.get("add_count", 0) or 0),
            locked_profit=bool(row.get("locked_profit", False)),
            current_r=(
                float(row["current_r"])
                if row.get("current_r") is not None
                and str(row.get("current_r", "")).strip() != ""
                else None
            ),
            updated_at=(
                str(row["updated_at"])
                if isinstance(row.get("updated_at"), str)
                else None
            ),
            last_add_at=(
                str(row["last_add_at"])
                if isinstance(row.get("last_add_at"), str)
                else None
            ),
        )
        n += 1
    return n


def _collect_open_parent_pids(simulators: Mapping[str, PositionSimulator]) -> Set[str]:
    out: Set[str] = set()
    for sim in (simulators or {}).values():
        if sim is None:
            continue
        for pid, pos in sim._positions.items():
            if not isinstance(pos, dict):
                continue
            if not bool(pos.get("_is_add_position", False)):
                out.add(str(pid))
    return out


def _prune_stale_add_position_records(
    st: ConstitutionRuntimeState, open_parent_pids: Set[str]
) -> None:
    ap = st.add_position.positions
    for k in list(ap.keys()):
        if k not in open_parent_pids:
            del ap[k]


def _merge_add_position_runtime_with_open_legs(
    sim: PositionSimulator, st: ConstitutionRuntimeState
) -> None:
    for pid, pos in sim._positions.items():
        if bool(pos.get("_is_add_position", False)):
            continue
        spid = str(pid)
        open_legs = _count_open_add_legs_for_parent(sim, spid)
        rec = st.add_position.positions.get(spid)
        if rec is None:
            if open_legs > 0:
                st.add_position.positions[spid] = AddPositionRecord(
                    position_id=spid, add_count=open_legs
                )
            continue
        rec.add_count = max(int(rec.add_count), open_legs)


def _filter_add_position_dict_for_open_parents(
    full: Dict[str, Any], rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    open_parents: Set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pos = row.get("position") or {}
        if not isinstance(pos, dict) or bool(pos.get("_is_add_position", False)):
            continue
        p = row.get("pid")
        if p is not None:
            open_parents.add(str(p))
    positions = (full or {}).get("positions") or {}
    if not isinstance(positions, dict):
        return {"positions": {}}
    out_pos = {
        k: v for k, v in positions.items() if k in open_parents and isinstance(v, dict)
    }
    return {"positions": out_pos}
