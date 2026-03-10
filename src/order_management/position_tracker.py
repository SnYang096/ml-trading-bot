"""持仓追踪器 — 管理实盘开仓、SL/TP/trailing/EMA200 退出、平仓

从 order_flow_listener.py 拆分出来的独立模块，职责:
  - 维护 _open_positions 状态
  - 每个特征周期调用 enforce_all() 检查退出条件
  - 需要平仓时调用 order_manager.place_order() 并 cancel SL/TP 挂单
  - trailing SL 更新时 cancel+replace 交易所 STOP_MARKET 挂单
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.order_management.models import OrderSide, OrderType
from src.time_series_model.live.position_logic import enforce_position

logger = logging.getLogger(__name__)


class PositionTracker:
    """实盘持仓追踪器

    Args:
        order_manager: OrderManager 实例（用于平仓 / cancel+replace SL）
        symbol: 当前交易对（如 "BTCUSDT"）
        default_bar_minutes: 信号时钟分钟数（默认 240 = 4h）
    """

    def __init__(
        self,
        order_manager: Any,
        symbol: str,
        default_bar_minutes: int = 240,
    ) -> None:
        self.order_manager = order_manager
        self.symbol = symbol
        self.default_bar_minutes = default_bar_minutes
        self._positions: Dict[str, Dict[str, Any]] = {}

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

        for pid, pos in list(self._positions.items()):
            structural_price = self._resolve_structural_price(pos, features)

            close_reason, exit_price = enforce_position(
                pos,
                price_high=current_price,
                price_low=current_price,
                price_close=current_price,
                now=now,
                default_bar_minutes=self.default_bar_minutes,
                structural_price=structural_price,
            )

            # trailing SL 更新时同步交易所挂单（仅在未触发退出时）
            if close_reason is None:
                self._maybe_sync_exchange_sl(pid, pos)

            if close_reason:
                qty = float(pos.get("qty") or 0.0)
                self.close(pid, qty, close_reason)
                closed.append(pid)

        for pid in closed:
            self._positions.pop(pid, None)

        return closed

    def close(self, position_id: str, qty: float, reason: str) -> None:
        """平仓：cancel SL/TP 挂单 → market 平仓

        Args:
            position_id: 持仓 ID
            qty: 平仓数量
            reason: 退出原因（用于日志）
        """
        if qty <= 0 or self.order_manager is None:
            return

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
        close_side = (
            OrderSide.SELL if side_str in {"LONG", "BUY"} else OrderSide.BUY
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
            logger.info(
                "[%s] 平仓成功: %s reason=%s qty=%.6f",
                self.symbol,
                position_id,
                reason,
                qty,
            )
        except Exception:
            logger.warning(
                "[%s] 软件平仓失败 reason=%s，交易所挂单可能已触发",
                self.symbol,
                reason,
            )

    def sync_exchange_sl(self, position_id: str) -> None:
        """手动触发指定持仓的交易所 SL 同步（cancel+replace）"""
        pos = self._positions.get(position_id)
        if pos is None:
            return
        self._maybe_sync_exchange_sl(position_id, pos)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _maybe_sync_exchange_sl(self, pid: str, pos: Dict[str, Any]) -> None:
        """若 SL 价格发生变化，cancel+replace 交易所 STOP_MARKET 挂单"""
        if self.order_manager is None:
            return

        new_sl = pos.get("stop_loss_price")
        old_sl = pos.get("_exchange_sl_price")
        if new_sl is None or old_sl is None:
            return
        if abs(new_sl - old_sl) < 1e-8:
            return  # 价格未变，不操作

        # cancel 旧挂单
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

        # place 新挂单
        side_str = str(pos.get("side", "")).upper()
        close_side = (
            OrderSide.SELL if side_str in {"LONG", "BUY"} else OrderSide.BUY
        )
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
                position_id=pid,
            )
            logger.info(
                "[%s] 交易所 SL 同步: %.4f → %.4f order=%s",
                self.symbol,
                old_sl,
                new_sl,
                new_order.order_id,
            )
            pos["_exchange_sl_order_id"] = new_order.order_id
            pos["_exchange_sl_price"] = new_sl
        except Exception:
            logger.error(
                "[%s] place 新 SL 挂单失败 (%.4f)，软件 SL 仍生效",
                self.symbol,
                new_sl,
            )

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
