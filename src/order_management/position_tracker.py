"""持仓追踪器 — 管理实盘开仓、SL/TP/trailing/EMA1200/VWAP1200 退出、平仓

从 order_flow_listener.py 拆分出来的独立模块，职责:
  - 维护 _open_positions 状态
  - 每个特征周期调用 enforce_all() 检查退出条件
  - 需要平仓时调用 order_manager.place_order() 并 cancel SL/TP 挂单
  - trailing SL 更新时 cancel+replace 交易所 STOP_MARKET 挂单
"""

from __future__ import annotations

import logging
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.order_management.models import (
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)
from src.time_series_model.live.metrics_exporter import METRICS
from src.time_series_model.live.position_logic import enforce_position

logger = logging.getLogger(__name__)


class PositionTracker:
    """实盘持仓追踪器

    Args:
        order_manager: OrderManager 实例（用于平仓 / cancel+replace SL）
        symbol: 当前交易对（如 "BTCUSDT"）
        default_bar_minutes: 信号时钟分钟数（默认 240 = 4h）
    """

    _exchange_sl_fail_last_log: Dict[str, float] = {}

    def __init__(
        self,
        order_manager: Any,
        symbol: str,
        default_bar_minutes: int = 240,
        state_path: Optional[str | Path] = None,
        truth_sync: Any = None,
    ) -> None:
        self.order_manager = order_manager
        self.symbol = symbol
        self.default_bar_minutes = default_bar_minutes
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.state_path = Path(state_path) if state_path else None
        self._truth_sync = truth_sync

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def add(self, position_id: str, pos: Dict[str, Any]) -> None:
        """记录新开仓

        Args:
            position_id: 唯一持仓 ID
            pos: build_position_dict() 产出的持仓字典（已含 qty）
        """
        self._positions[position_id] = pos
        self._persist_state()
        self._persist_position_record(position_id, pos)
        logger.info(
            "[%s] 记录持仓: %s side=%s entry=%.4f sl=%.4f",
            self.symbol,
            position_id,
            pos.get("side"),
            pos.get("entry_price", 0),
            pos.get("stop_loss_price") or 0,
        )

    def get(self, position_id: str) -> Optional[Dict[str, Any]]:
        return self._positions.get(position_id)

    def all_positions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._positions)

    def restore_from_disk(self, *, live_symbols: Optional[set[str]] = None) -> int:
        """Restore persisted position dictionaries after process restart.

        The tracker stores the full ``build_position_dict`` output, including
        trailing/breakeven/structural-exit state. Restore is intentionally
        conservative: if exchange-backed ``live_symbols`` is supplied and this
        symbol is absent, persisted positions are cleared instead of adopted.
        """
        if self.state_path is None or not self.state_path.exists():
            return 0
        sym = str(self.symbol or "").upper().strip()
        if live_symbols is not None and sym not in live_symbols:
            self._positions = {}
            self._persist_state()
            logger.info("[%s] persisted positions skipped: no exchange position", sym)
            return 0
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[%s] failed to load persisted positions: %s", sym, exc)
            return 0
        if str(raw.get("symbol") or "").upper().strip() not in {"", sym}:
            logger.warning(
                "[%s] persisted positions ignored: symbol mismatch %s",
                sym,
                raw.get("symbol"),
            )
            return 0
        positions_raw = raw.get("positions") or {}
        if not isinstance(positions_raw, dict):
            return 0
        restored: Dict[str, Dict[str, Any]] = {}
        for pid, pos in positions_raw.items():
            if not isinstance(pos, dict) or not str(pid).strip():
                continue
            restored[str(pid)] = self._from_json_safe(pos)
        self._positions = restored
        if restored:
            logger.info("[%s] restored %d persisted position(s)", sym, len(restored))
            for pid, pos in restored.items():
                self._persist_position_record(pid, pos)
        return len(restored)

    def bootstrap_from_exchange_position(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
    ) -> Optional[str]:
        """Create a PositionTracker entry from exchange data when JSON snapshot is missing.

        Used on ``run_live`` restart when the exchange still has a leg but
        ``restore_from_disk()`` returned 0.  Applies conservative defaults
        (1% ATR estimate, 1.5R stop / 3R target) so ``enforce_position`` and
        ``ensure_exchange_stop_losses`` can manage the leg immediately.

        If SQLite already has an open row for the same symbol+side, merges into
        that canonical ``position_id`` (memory + JSON re-keyed to match).

        Returns the canonical position_id, or None if bootstrap is not possible.
        """
        if not symbol or qty <= 0 or entry_price <= 0:
            return None
        sym_base = (
            symbol.upper().rstrip("USDT")
            if symbol.upper().endswith("USDT")
            else symbol.upper()
        )
        now = datetime.now(timezone.utc)
        pid = f"{sym_base}:bootstrap_{int(now.timestamp() * 1e6)}"
        side_norm = side.upper()
        action = "SHORT" if side_norm in {"SHORT", "SELL"} else "LONG"
        entry = float(entry_price)
        qty_f = float(qty)
        # Conservative ATR estimate (1% of entry) and default RR (1.5R stop)
        atr_est = entry * 0.01
        stop_r = 1.5
        sl_price = (
            entry - (stop_r * atr_est)
            if action == "LONG"
            else entry + (stop_r * atr_est)
        )
        tp_price = (
            entry + (2.0 * stop_r * atr_est)
            if action == "LONG"
            else entry - (2.0 * stop_r * atr_est)
        )
        pos: Dict[str, Any] = {
            "position_id": pid,
            "symbol": symbol.upper(),
            "side": action,
            "entry_price": entry,
            "qty": qty_f,
            "entry_time": now.isoformat(),
            "archetype": "tpc",
            "status": "open",
            "atr_at_entry": atr_est,
            "stop_loss_r": stop_r,
            "bar_minutes": 120,
            "stop_loss_price": round(sl_price, 4),
            "take_profit_price": round(tp_price, 4),
            "high_water_mark": entry if action == "LONG" else None,
            "low_water_mark": entry if action == "SHORT" else None,
            "_bootstrap_missing_snapshot": True,
        }
        self._positions[pid] = pos
        self._persist_state()
        canonical = self._persist_position_record(pid, pos)
        pid = self._rekey_in_memory(pid, canonical, pos)
        self._persist_state()
        logger.warning(
            "[%s] bootstrapped from exchange: pid=%s side=%s qty=%.4f entry=%.4f"
            " sl=%.4f (conservative 1%% ATR, 1.5R stop)",
            self.symbol,
            pid,
            action,
            qty_f,
            entry,
            sl_price,
        )
        return pid

    def ensure_exchange_stop_losses(self) -> int:
        """Place exchange STOP_MARKET for positions that have software SL but no exchange SL."""
        placed = 0
        for pid, pos in list(self._positions.items()):
            if pos.get("stop_loss_price") is None:
                continue
            if pos.get("_exchange_sl_price") is not None:
                continue
            before_oid = pos.get("_exchange_sl_order_id")
            self._maybe_sync_exchange_sl(pid, pos)
            if pos.get("_exchange_sl_order_id") and not before_oid:
                placed += 1
        return placed

    def __len__(self) -> int:
        return len(self._positions)

    def enforce_all(self, features: Dict[str, Any]) -> List[str]:
        """检查所有持仓的退出条件，执行需要平仓的仓位

        对每个持仓调用 enforce_position()，如触发退出则调用 close()。
        trailing SL 更新时同步交易所挂单。

        Args:
            features: 当前特征字典（须含 timestamp, close, ema_200 等）

        Returns:
            本次已平仓的 position_id 列表
        """
        if not self._positions:
            return []

        now = self._resolve_now(features)
        current_price = self._resolve_price(features)
        if current_price is None:
            logger.warning("[%s] enforce_all: 无法获取当前价格，跳过", self.symbol)
            return []

        closed: List[str] = []
        close_decisions: Dict[str, Tuple[str, float]] = {}

        for pid, pos in list(self._positions.items()):
            self._sync_child_stop_from_parent(pid, pos)
            structural_price = self._resolve_structural_price(pos, features)
            macro_tp_vwap = self._resolve_macro_tp_vwap_position(pos, features)
            ema_1200_pos = self._resolve_ema_1200_position(pos, features)

            # L3 dynamic trailing 读取当前 feature 中的 wide_sr 价格位。
            _w_up = (
                features.get("wide_sr_upper_px") if isinstance(features, dict) else None
            )
            _w_lo = (
                features.get("wide_sr_lower_px") if isinstance(features, dict) else None
            )
            try:
                _w_up_f = float(_w_up) if _w_up is not None and _w_up == _w_up else None
            except (TypeError, ValueError):
                _w_up_f = None
            try:
                _w_lo_f = float(_w_lo) if _w_lo is not None and _w_lo == _w_lo else None
            except (TypeError, ValueError):
                _w_lo_f = None

            close_reason, exit_price = enforce_position(
                pos,
                price_high=current_price,
                price_low=current_price,
                price_close=current_price,
                now=now,
                default_bar_minutes=self.default_bar_minutes,
                structural_price=structural_price,
                macro_tp_vwap_position=macro_tp_vwap,
                ema_1200_position=ema_1200_pos,
                wide_sr_upper_px=_w_up_f,
                wide_sr_lower_px=_w_lo_f,
            )

            # trailing SL 更新时同步交易所挂单（仅在未触发退出时）
            if close_reason is None:
                if self._maybe_sync_exchange_sl(pid, pos):
                    closed.append(pid)
                elif pid in self._positions:
                    self._persist_position_record(pid, pos)

            if close_reason:
                close_decisions[pid] = (str(close_reason), float(exit_price))

        # 默认行为: 母仓触发退出时，同 bar 强制平掉对应加仓子仓
        parent_close_ids = {
            pid
            for pid in close_decisions
            if not bool(self._positions.get(pid, {}).get("_is_add_position", False))
        }
        if parent_close_ids:
            for pid, pos in list(self._positions.items()):
                if pid in close_decisions:
                    continue
                if not bool(pos.get("_is_add_position", False)):
                    continue
                if not bool(pos.get("_share_parent_exit", True)):
                    continue
                parent_pid = str(pos.get("_parent_pid", "") or "")
                if parent_pid in parent_close_ids:
                    reason, px = close_decisions[parent_pid]
                    close_decisions[pid] = (reason, px)

        for pid, (close_reason, _exit_price) in close_decisions.items():
            qty = float(self._positions.get(pid, {}).get("qty") or 0.0)
            self.close(pid, qty, close_reason)
            closed.append(pid)

        for pid in set(closed):
            self._positions.pop(pid, None)

        self._persist_state()

        return closed

    def _sync_child_stop_from_parent(self, pid: str, pos: Dict[str, Any]) -> None:
        """Child add-position inherits parent's stop in real time — tighten-only.

        子仓 SL 跟随父仓 SL，但只允许向入场有利方向移动（tighten-only），
        避免父仓 breakeven 尚未触发时反而把子仓 SL 放宽。
        """
        if not bool(pos.get("_is_add_position", False)):
            return
        if not bool(pos.get("_inherit_parent_stop", False)):
            return
        parent_pid = str(pos.get("_parent_pid", "") or "")
        if not parent_pid:
            return
        parent = self._positions.get(parent_pid)
        if not parent:
            return
        parent_sl = parent.get("stop_loss_price")
        if parent_sl is None:
            return
        try:
            new_sl = float(parent_sl)
        except (TypeError, ValueError):
            return
        old_sl = pos.get("stop_loss_price")
        is_long = str(pos.get("side", "")).upper() in {"LONG", "BUY"}
        if old_sl is None:
            pos["stop_loss_price"] = new_sl
            return
        try:
            old_sl_f = float(old_sl)
        except (TypeError, ValueError):
            pos["stop_loss_price"] = new_sl
            return
        if is_long and new_sl > old_sl_f:
            pos["stop_loss_price"] = new_sl
        elif (not is_long) and new_sl < old_sl_f:
            pos["stop_loss_price"] = new_sl

    def close(self, position_id: str, qty: float, reason: str) -> bool:
        """平仓：cancel SL/TP 挂单 → market 平仓

        Args:
            position_id: 持仓 ID
            qty: 平仓数量
            reason: 退出原因（用于日志）

        Returns:
            True 当市价平仓单成功提交；失败（异常）返回 False。
        """
        if qty <= 0 or self.order_manager is None:
            return False

        pos = self._positions.get(position_id, {})

        # 1. Cancel 未触发的 SL/TP 挂单（避免平仓后重复触发）
        for key in ("_exchange_sl_order_id", "_exchange_tp_order_id"):
            oid = pos.get(key)
            if oid:
                try:
                    self.order_manager.cancel_order(oid)
                except Exception:
                    pass  # 可能已触发，忽略

        # 2. Market 平仓
        side_str = str(pos.get("side", "")).upper()
        close_side = OrderSide.SELL if side_str in {"LONG", "BUY"} else OrderSide.BUY
        try:
            self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=float(qty),
                reduce_only=True,
                # MARKET + closePosition is invalid on Binance (-4136); use qty+reduceOnly.
                close_position=False,
                position_id=position_id,
            )
            logger.info(
                "[%s] 平仓成功: %s reason=%s qty=%.6f",
                self.symbol,
                position_id,
                reason,
                qty,
            )
            storage_pid = self._storage_position_id(position_id, pos)
            self._persist_position_record(
                storage_pid,
                pos,
                status=PositionStatus.CLOSED,
                exit_reason=reason,
            )
            try:
                METRICS.record_strategy_event(
                    scope="trend",
                    strategy=str(pos.get("archetype") or "unknown").lower(),
                    symbol=self.symbol,
                    event="exit",
                    side=str(pos.get("side") or "na").lower(),
                )
            except Exception:
                logger.debug(
                    "[%s] exit marker metrics update skipped",
                    self.symbol,
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "[%s] 软件平仓失败 reason=%s，交易所挂单可能已触发",
                self.symbol,
                reason,
            )
            return False
        return True

    def close_from_exchange(
        self,
        position_id: str,
        *,
        reason: str,
        exit_price: Optional[float] = None,
    ) -> bool:
        """交易所已成交关闭后，同步移除本地持仓（不再重复下市价平仓单）"""
        pos = self._positions.get(position_id)
        if pos is None:
            return False

        if self.order_manager is not None:
            for key in ("_exchange_sl_order_id", "_exchange_tp_order_id"):
                oid = pos.get(key)
                if oid:
                    try:
                        self.order_manager.cancel_order(oid)
                    except Exception:
                        pass
            try:
                self._cancel_open_close_position_conditionals(
                    position_side=self._position_side_from_pos(pos)
                )
            except Exception:
                logger.debug(
                    "[%s] cancel conditionals on exchange close failed",
                    self.symbol,
                    exc_info=True,
                )

        pos = self._positions.pop(position_id, None)
        if pos is None:
            return False
        self._persist_state()
        # 交易所已触发时，挂单通常已终态；尝试清理本地引用即可。
        pos["_exchange_close_reason"] = reason
        if exit_price is not None:
            pos["_exchange_exit_price"] = float(exit_price)
        storage_pid = self._storage_position_id(position_id, pos)
        self._persist_position_record(
            storage_pid,
            pos,
            status=PositionStatus.CLOSED,
            exit_price=exit_price,
            exit_reason=reason,
        )
        logger.info(
            "[%s] 交易所关闭同步: %s reason=%s exit=%.6f",
            self.symbol,
            position_id,
            reason,
            float(exit_price or 0.0),
        )
        try:
            METRICS.record_strategy_event(
                scope="trend",
                strategy=str(pos.get("archetype") or "unknown").lower(),
                symbol=self.symbol,
                event="exit",
                side=str(pos.get("side") or "na").lower(),
                price=exit_price,
            )
        except Exception:
            logger.debug(
                "[%s] exchange exit marker metrics update skipped",
                self.symbol,
                exc_info=True,
            )
        return True

    def sync_exchange_sl(self, position_id: str) -> None:
        """手动触发指定持仓的交易所 SL 同步（cancel+replace）"""
        pos = self._positions.get(position_id)
        if pos is None:
            return
        self._maybe_sync_exchange_sl(position_id, pos)
        self._persist_state()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _position_side_from_pos(pos: Dict[str, Any]) -> str:
        side_str = str(pos.get("side", "")).upper()
        return "LONG" if side_str in {"LONG", "BUY"} else "SHORT"

    def _should_skip_exchange_sl_sync(self, pid: str, pos: Dict[str, Any]) -> bool:
        """Add legs that inherit parent stop must not place a second closePosition SL."""
        if bool(pos.get("_is_add_position", False)) and bool(
            pos.get("_inherit_parent_stop", False)
        ):
            return True
        # One closePosition SL per symbol+side: only the designated owner may sync.
        side = self._position_side_from_pos(pos)
        owner_pid = self._exchange_sl_owner_pid(side)
        return owner_pid is not None and owner_pid != pid

    @staticmethod
    def _order_info_close_position(order: Dict[str, Any]) -> bool:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        raw = info.get("closePosition", order.get("closePosition"))
        return raw is True or str(raw).lower() == "true"

    def _exchange_sl_owner_pid(self, position_side: str) -> Optional[str]:
        """Pick one position per symbol+side to own the closePosition exchange SL."""
        candidates: List[Tuple[int, str]] = []
        for pid, pos in self._positions.items():
            if self._position_side_from_pos(pos) != position_side:
                continue
            if bool(pos.get("_is_add_position", False)) and bool(
                pos.get("_inherit_parent_stop", False)
            ):
                continue
            rank = 0
            if pos.get("_exchange_sl_order_id"):
                rank -= 10
            if not bool(pos.get("_is_add_position", False)):
                rank -= 5
            candidates.append((rank, str(pid)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _fetch_mark_price(self, position_side: str) -> Optional[float]:
        """Best-effort mark/last for -2021 guard before exchange SL sync."""
        api = (
            getattr(self.order_manager, "binance_api", None)
            if self.order_manager
            else None
        )
        if api is None:
            return None
        get_ticker = getattr(api, "get_ticker_price", None)
        if callable(get_ticker):
            try:
                px = get_ticker(self.symbol)
                if px is None:
                    pass
                else:
                    fpx = float(px)
                    if fpx > 0:
                        return fpx
            except (TypeError, ValueError):
                pass
            except Exception:
                logger.debug(
                    "[%s] mark fetch via ticker failed",
                    self.symbol,
                    exc_info=True,
                )
        get_positions = getattr(api, "get_positions", None)
        if callable(get_positions):
            try:
                for row in get_positions() or []:
                    if not isinstance(row, dict):
                        continue
                    sym = str(row.get("symbol") or "").upper()
                    if sym and sym != str(self.symbol or "").upper():
                        continue
                    ps = str(
                        row.get("position_side") or row.get("positionSide") or ""
                    ).upper()
                    if ps and ps != position_side:
                        continue
                    mark = row.get("mark_price") or row.get("markPrice")
                    if mark is not None:
                        fmark = float(mark)
                        if fmark > 0:
                            return fmark
            except (TypeError, ValueError):
                pass
            except Exception:
                logger.debug(
                    "[%s] mark fetch via positions failed",
                    self.symbol,
                    exc_info=True,
                )
        return None

    @staticmethod
    def _stop_would_immediately_trigger(
        *, mark: float, stop_price: float, is_long: bool
    ) -> bool:
        """True when STOP_MARKET would reject with Binance -2021."""
        if mark <= 0 or stop_price <= 0:
            return False
        if is_long:
            return mark <= stop_price
        return mark >= stop_price

    def _fetch_open_orders_for_sl_cleanup(self) -> List[Dict[str, Any]]:
        api = getattr(self.order_manager, "binance_api", None)
        if api is None:
            return []
        fetch = getattr(api, "get_open_orders_for_sl_cleanup", None)
        if callable(fetch):
            return list(fetch(self.symbol) or [])
        return list(api.get_open_orders(self.symbol) or [])

    @staticmethod
    def _is_stop_or_tp_conditional(order: Dict[str, Any]) -> bool:
        otype = str(order.get("type") or "").lower()
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        raw_type = str(info.get("type") or info.get("orderType") or "").lower()
        combined = f"{otype} {raw_type}"
        return any(
            token in combined
            for token in (
                "stop",
                "take_profit",
                "trailing_stop",
            )
        )

    @staticmethod
    def _client_order_id_from_order(order: Dict[str, Any]) -> str:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        return str(
            order.get("client_order_id")
            or info.get("clientOrderId")
            or info.get("clientAlgoId")
            or ""
        ).strip()

    @staticmethod
    def _is_owned_stop_or_tp_conditional(order: Dict[str, Any]) -> bool:
        """Fallback cleanup only touches bot-managed STOP/TP unless explicitly broadened."""
        if not PositionTracker._is_stop_or_tp_conditional(order):
            return False
        raw = os.getenv("MLBOT_EXCHANGE_SL_CLEAN_ALL_STOP_TP", "0").strip().lower()
        if raw not in {"0", "false", "off", "no"}:
            return True
        cid = PositionTracker._client_order_id_from_order(order)
        if not cid:
            return False
        prefix = (
            os.getenv("MLBOT_LIVE_CLIENT_ORDER_PREFIX", "tl").strip() or "tl"
        ).replace("-", "")
        prefix = "".join(c for c in prefix if str(c).isalnum())[:12] or "tl"
        return cid.startswith(f"{prefix}_")

    def _cancel_open_close_position_conditionals(
        self,
        *,
        position_side: str,
        include_all_stop_tp: bool = False,
    ) -> int:
        """Cancel STOP/TP conditionals so a new closePosition SL can be placed (-4130 guard)."""
        api = getattr(self.order_manager, "binance_api", None)
        if api is None:
            return 0
        close_side = "sell" if position_side == "LONG" else "buy"
        cancelled = 0
        try:
            open_orders = self._fetch_open_orders_for_sl_cleanup()
        except Exception as exc:
            logger.warning(
                "[%s] fetch open orders for SL cleanup failed: %s",
                self.symbol,
                exc,
            )
            return 0
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            if not include_all_stop_tp and not self._order_info_close_position(order):
                continue
            if include_all_stop_tp and not self._is_owned_stop_or_tp_conditional(order):
                continue
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            order_pos_side = str(info.get("positionSide") or "BOTH").upper()
            if order_pos_side not in {"", "BOTH"} and order_pos_side != position_side:
                continue
            if order_pos_side in {"", "BOTH"}:
                if str(order.get("side") or "").lower() != close_side:
                    continue
            ex_id = str(order.get("order_id") or "").strip()
            if not ex_id:
                continue
            try:
                if order.get("_is_algo_order") and hasattr(api, "cancel_algo_order"):
                    api.cancel_algo_order(ex_id, self.symbol)
                else:
                    api.cancel_order(ex_id, self.symbol)
                cancelled += 1
            except Exception as exc:
                logger.warning(
                    "[%s] cancel closePosition conditional %s failed: %s",
                    self.symbol,
                    ex_id,
                    exc,
                )
        if cancelled:
            logger.info(
                "[%s] cleared %d closePosition conditional(s) before SL sync (%s)",
                self.symbol,
                cancelled,
                position_side,
            )
        return cancelled

    def _log_exchange_sl_failure(
        self,
        new_sl: float,
        exc: Optional[Exception],
        *,
        log_key: str,
    ) -> None:
        """Rate-limit repeated -4130 / place SL errors (enforce_all runs often)."""
        try:
            interval = float(os.getenv("MLBOT_EXCHANGE_SL_FAIL_LOG_SECONDS", "300"))
        except ValueError:
            interval = 300.0
        interval = max(30.0, interval)
        now = time.monotonic()
        last = PositionTracker._exchange_sl_fail_last_log.get(log_key, 0.0)
        is_4130 = exc is not None and "-4130" in str(exc)
        if now - last < interval:
            logger.debug(
                "[%s] exchange SL still failing (%.4f) — suppressed repeat (%s)",
                self.symbol,
                new_sl,
                type(exc).__name__ if exc else "unknown",
            )
            return
        PositionTracker._exchange_sl_fail_last_log[log_key] = now
        if is_4130:
            logger.warning(
                "[%s] place 新 SL 挂单失败 (%.4f) — closePosition slot busy; "
                "软件 SL 仍生效",
                self.symbol,
                new_sl,
            )
        else:
            logger.error(
                "[%s] place 新 SL 挂单失败 (%.4f)，软件 SL 仍生效",
                self.symbol,
                new_sl,
            )

    def _maybe_sync_exchange_sl(self, pid: str, pos: Dict[str, Any]) -> bool:
        """Place or refresh exchange STOP_MARKET when software stop_loss_price is set.

        Returns True when the local position was removed (e.g. -2021 market close
        or -4509 ghost reconcile) so callers skip re-persisting as OPEN.
        """
        if self.order_manager is None:
            return False
        if self._should_skip_exchange_sl_sync(pid, pos):
            return False

        new_sl_raw = pos.get("stop_loss_price")
        if new_sl_raw is None:
            return False
        try:
            new_sl = float(new_sl_raw)
        except (TypeError, ValueError):
            return False
        if new_sl <= 0:
            return False
        self._persist_position_record(pid, pos)

        old_sl_raw = pos.get("_exchange_sl_price")
        if old_sl_raw is not None:
            try:
                old_sl = float(old_sl_raw)
            except (TypeError, ValueError):
                old_sl = None
            else:
                if abs(new_sl - old_sl) < 1e-8:
                    return False  # 价格未变，不操作
        else:
            old_sl = None

        position_side = self._position_side_from_pos(pos)
        side_str = str(pos.get("side", "")).upper()
        is_long = side_str in {"LONG", "BUY"}
        qty = float(pos.get("qty") or 0.0)
        if qty <= 0:
            return False

        mark = self._fetch_mark_price(position_side)
        if mark is not None and self._stop_would_immediately_trigger(
            mark=mark, stop_price=new_sl, is_long=is_long
        ):
            logger.warning(
                "[%s] exchange SL skipped (-2021): mark=%.4f beyond stop=%.4f pid=%s "
                "— market close",
                self.symbol,
                mark,
                new_sl,
                pid,
            )
            if not self.close(pid, qty, reason="sl_breached_mark"):
                # 市价平失败：保留本地仓位（软件 SL 下个 bar 继续兜底），
                # 跳过本轮 STOP 挂单（必然 -2021）。
                logger.error(
                    "[%s] -2021 market close failed, keep local position pid=%s",
                    self.symbol,
                    pid,
                )
                return False
            self._positions.pop(pid, None)
            self._persist_state()
            return True

        # cancel 旧挂单（仅更新路径；首次挂单无旧单）
        old_oid = pos.get("_exchange_sl_order_id")
        if old_oid:
            try:
                self.order_manager.cancel_order(old_oid)
            except Exception:
                logger.warning(
                    "[%s] cancel 旧 SL 挂单失败（可能已触发）: %s",
                    self.symbol,
                    old_oid,
                )
        self._cancel_open_close_position_conditionals(position_side=position_side)

        # place 新挂单
        close_side = OrderSide.SELL if is_long else OrderSide.BUY

        new_order = None
        last_exc: Optional[Exception] = None
        _4130_log_key = f"{self.symbol}:{position_side}"
        for attempt in range(2):
            try:
                new_order = self.order_manager.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=qty,
                    stop_price=new_sl,
                    reduce_only=True,
                    close_position=True,
                    position_id=pid,
                    position_side=position_side,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and "-4130" in str(exc):
                    logger.warning(
                        "[%s] exchange SL rejected (-4130), clearing closePosition stops and retrying",
                        self.symbol,
                    )
                    n_cleared = self._cancel_open_close_position_conditionals(
                        position_side=position_side
                    )
                    if n_cleared == 0:
                        n_cleared = self._cancel_open_close_position_conditionals(
                            position_side=position_side,
                            include_all_stop_tp=True,
                        )
                    if n_cleared == 0:
                        logger.warning(
                            "[%s] -4130 retry: no STOP/TP conditional in "
                            "openOrders/openAlgoOrders (%s); check exchange manually",
                            self.symbol,
                            position_side,
                        )
                    else:
                        time.sleep(0.25)
                    continue
                # -4509: closePosition STOP requires an open position. If the
                # exchange is actually flat on this side, the local position is a
                # ghost; clear it so we stop retrying (and log spamming) forever.
                if "-4509" in str(
                    exc
                ) and self._reconcile_local_position_if_exchange_flat(
                    pid, position_side
                ):
                    return True
                self._log_exchange_sl_failure(new_sl, last_exc, log_key=_4130_log_key)
                return False
        if new_order is None:
            if last_exc is not None:
                self._log_exchange_sl_failure(new_sl, last_exc, log_key=_4130_log_key)
            return False
        if old_sl is None:
            logger.info(
                "[%s] 交易所 SL 首次挂单: %.4f order=%s pid=%s",
                self.symbol,
                new_sl,
                new_order.order_id,
                pid,
            )
        else:
            logger.info(
                "[%s] 交易所 SL 同步: %.4f → %.4f order=%s",
                self.symbol,
                old_sl,
                new_sl,
                new_order.order_id,
            )
        pos["_exchange_sl_order_id"] = new_order.order_id
        pos["_exchange_sl_price"] = new_sl
        self._persist_position_record(pid, pos)
        self._persist_state()
        return False

    def _reconcile_local_position_if_exchange_flat(
        self, pid: str, position_side: str
    ) -> bool:
        """Return True iff the exchange has no position on ``position_side`` and the
        local ghost position was dropped via :meth:`close_from_exchange`.

        Only clears when the exchange read succeeds and confirms flatness; any read
        error keeps the local position so the next cycle retries (fail-safe)."""
        api = getattr(self.order_manager, "binance_api", None)
        if api is None:
            return False
        try:
            exchange_positions = api.get_positions(self.symbol) or []
        except Exception:
            logger.warning(
                "[%s] -4509 reconcile: 查询交易所仓位失败，保留本地仓位待下次重试",
                self.symbol,
                exc_info=True,
            )
            return False
        want = str(position_side or "").upper()
        for ep in exchange_positions:
            if abs(float(ep.get("size") or 0.0)) <= 0:
                continue
            ep_side = str(ep.get("side") or "").upper()
            if not want or ep_side == want:
                # Real live position on this side -> -4509 is not a ghost; keep it.
                return False
        logger.warning(
            "[%s] -4509: 交易所该方向无持仓，判定本地为幽灵仓，自动同步关闭 pid=%s side=%s",
            self.symbol,
            pid,
            position_side,
        )
        return self.close_from_exchange(pid, reason="exchange_flat_minus_4509")

    def _storage(self) -> Any:
        if self.order_manager is None:
            return None
        attrs = getattr(self.order_manager, "__dict__", {})
        if isinstance(attrs, dict) and "storage" in attrs:
            return attrs.get("storage")
        return None

    @staticmethod
    def _as_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _rekey_in_memory(
        self, old_pid: str, canonical_pid: str, pos: Dict[str, Any]
    ) -> str:
        """Align in-memory tracker key with SQLite canonical position_id after merge."""
        canonical = str(canonical_pid or old_pid).strip() or old_pid
        if canonical == old_pid:
            return old_pid
        self._positions.pop(old_pid, None)
        pos["position_id"] = canonical
        self._positions[canonical] = pos
        return canonical

    def _storage_position_id(self, position_id: str, pos: Dict[str, Any]) -> str:
        """Resolve SQLite row id (handles bootstrap merge vs in-memory bootstrap pid)."""
        storage = self._storage()
        if storage is None:
            return position_id
        try:
            if storage.get_position(position_id) is not None:
                return position_id
        except Exception:
            pass
        side_raw = str(pos.get("side") or "").upper()
        side = PositionSide.LONG if side_raw in {"LONG", "BUY"} else PositionSide.SHORT
        try:
            for row in storage.get_open_positions(self.symbol) or []:
                if row.side == side and row.status == PositionStatus.OPEN:
                    return str(row.position_id or position_id)
        except Exception:
            pass
        return position_id

    def _persist_position_record(
        self,
        position_id: str,
        pos: Dict[str, Any],
        *,
        status: PositionStatus = PositionStatus.OPEN,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> str:
        """Best-effort mirror of the in-memory software stop into SQLite.

        Delegates to ``TrendPositionTruthSync.project_to_sqlite()`` if
        a truth_sync instance is available; otherwise falls back to the
        legacy in-place logic.

        Returns the canonical ``position_id`` written (may differ after dedup merge).
        """
        if self._truth_sync is not None:
            return self._truth_sync.project_to_sqlite(
                position_id,
                pos,
                status=status,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )
        # ── Legacy fallback (no truth_sync injected) ──
        storage = self._storage()
        if storage is None:
            return position_id
        side_raw = str(pos.get("side") or "").upper()
        side = PositionSide.LONG if side_raw in {"LONG", "BUY"} else PositionSide.SHORT
        try:
            entry_price = float(pos.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        try:
            qty = float(pos.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if not position_id or qty <= 0:
            return position_id
        try:
            existing = storage.get_position(position_id)
        except Exception:
            existing = None
        # ── Dedup: if another OPEN position already exists for the same symbol+side ──
        if existing is None and status == PositionStatus.OPEN:
            try:
                for p in storage.get_open_positions(self.symbol) or []:
                    if p.side == side and p.status == PositionStatus.OPEN:
                        logger.info(
                            "[%s] merged duplicate position %s -> existing %s",
                            self.symbol,
                            position_id,
                            p.position_id,
                        )
                        existing = p
                        position_id = str(p.position_id or position_id)
                        break
            except Exception:
                pass
        record = existing or Position(
            position_id=position_id,
            symbol=self.symbol,
            side=side,
            entry_time=self._as_dt(pos.get("entry_time")),
        )
        record.symbol = self.symbol
        record.side = side
        record.entry_price = entry_price
        record.initial_size = record.initial_size or qty
        record.current_size = 0.0 if status == PositionStatus.CLOSED else qty
        record.total_cost = entry_price * (record.initial_size or qty)
        record.status = status
        record.stop_loss_price = pos.get("stop_loss_price")
        record.take_profit_price = pos.get("take_profit_price")
        record.strategy_id = str(pos.get("archetype") or "") or record.strategy_id
        record.archetype = str(pos.get("archetype") or "") or record.archetype
        record.add_count = int(pos.get("_add_position_seq") or record.add_count or 0)
        record.parent_position_id = pos.get("_parent_pid") or record.parent_position_id
        if status == PositionStatus.CLOSED:
            record.exit_time = datetime.now(timezone.utc)
            record.exit_price = exit_price
            record.exit_reason = exit_reason
        try:
            if existing is None:
                storage.create_position(record)
            else:
                storage.update_position(record)
        except Exception:
            logger.warning(
                "[%s] persist position record skipped: %s",
                self.symbol,
                position_id,
                exc_info=True,
            )
        return position_id

    @staticmethod
    def _resolve_now(features: Dict[str, Any]) -> datetime:
        now = features.get("timestamp")
        if isinstance(now, str):
            try:
                now = datetime.fromisoformat(now)
            except Exception:
                now = None
        if not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
        return now

    @staticmethod
    def _resolve_price(features: Dict[str, Any]) -> Optional[float]:
        for key in ("close", "price", "last_price", "mark_price"):
            v = features.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "symbol": self.symbol,
                "positions": self._to_json_safe(self._positions),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.state_path)
        except Exception:
            logger.warning("[%s] persist position tracker state failed", self.symbol)

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return {"__datetime__": value.isoformat()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_json_safe(v) for v in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return cls._to_json_safe(value.item())
            except Exception:
                pass
        if hasattr(value, "value"):
            try:
                return cls._to_json_safe(value.value)
            except Exception:
                pass
        return str(value)

    @classmethod
    def _from_json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value.keys()) == {"__datetime__"}:
                raw = value.get("__datetime__")
                if isinstance(raw, str):
                    try:
                        dt = datetime.fromisoformat(raw)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except Exception:
                        return raw
            return {str(k): cls._from_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._from_json_safe(v) for v in value]
        return value

    @staticmethod
    def _resolve_structural_price(
        pos: Dict[str, Any], features: Dict[str, Any]
    ) -> Optional[float]:
        """获取 EMA200 价格（仅当 structural_exit=="ema200" 时有意义）"""
        if pos.get("structural_exit") != "ema200":
            return None
        v = features.get("ema_200")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_macro_tp_vwap_position(
        pos: Dict[str, Any], features: Dict[str, Any]
    ) -> Optional[float]:
        if str(pos.get("structural_exit") or "").strip().lower() != "vwap1200":
            return None
        v = features.get("macro_tp_vwap_1200_position")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_ema_1200_position(
        pos: Dict[str, Any], features: Dict[str, Any]
    ) -> Optional[float]:
        if str(pos.get("structural_exit") or "").strip().lower() != "ema1200":
            return None
        v = features.get("ema_1200_position")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
