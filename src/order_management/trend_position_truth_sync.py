"""B 层 Trend 持仓 TruthSync — SQLite 投影唯一写入口。

P1 实现：所有 SQLite positions 表写入经 ``project_to_sqlite()``，
消灭「PT 写一处、脚本写一处、CMS 逻辑再猜」的多入口问题。

Not a standalone daemon — imported by existing live processes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.order_management.models import (
    Position,
    PositionSide,
    PositionStatus,
)

logger = logging.getLogger(__name__)


class TrendPositionTruthSync:
    """B 层 Trend 持仓 TruthSync — SQLite 投影唯一写入口。

    Args:
        symbol: 交易对符号（如 "BTCUSDT"）
        storage_factory: callable 返回 Storage 实例（或 None）
    """

    def __init__(
        self,
        symbol: str,
        storage_factory: Callable[[], Any],
    ) -> None:
        self.symbol = symbol
        self._storage_factory = storage_factory

    def _storage(self) -> Any:
        """获取 Storage 实例。"""
        try:
            return self._storage_factory()
        except Exception:
            return None

    @staticmethod
    def _as_dt(value: Any) -> datetime:
        """Convert various datetime representations to a datetime object."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def project_to_sqlite(
        self,
        position_id: str,
        pos: Dict[str, Any],
        *,
        status: PositionStatus = PositionStatus.OPEN,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> str:
        """唯一 SQLite positions 写入口。

        Best-effort mirror of the in-memory position dict into the SQLite
        ``positions`` table.  Includes dedup merge: if another OPEN row
        already exists for the same symbol+side, reuse it instead of
        creating a duplicate.

        Returns the canonical ``position_id`` written (may differ after
        dedup merge).
        """
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

        # ── Dedup: if another OPEN position already exists for the same
        #     symbol+side (e.g. exchange_sync created before bootstrap
        #     JSON loaded), reuse it instead of creating a duplicate row. ──
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
        record.initial_size = pos.get("initial_size") or record.initial_size or qty
        record.current_size = 0.0 if status == PositionStatus.CLOSED else qty
        record.total_cost = entry_price * (record.initial_size or qty)
        record.status = status
        record.stop_loss_price = pos.get("stop_loss_price")
        record.take_profit_price = pos.get("take_profit_price")
        record.strategy_id = str(pos.get("archetype") or "") or record.strategy_id
        record.archetype = str(pos.get("archetype") or "") or record.archetype
        record.add_count = int(pos.get("_add_position_seq") or record.add_count or 0)
        record.parent_position_id = pos.get("_parent_pid") or record.parent_position_id
        record.notes = pos.get("notes") or record.notes
        record.unrealized_pnl = pos.get("unrealized_pnl") or record.unrealized_pnl
        record.realized_pnl = pos.get("realized_pnl") or record.realized_pnl

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

    # ── Convenience: project a Position object directly ──

    def project_position_object(
        self,
        position: Position,
        *,
        status: Optional[PositionStatus] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> str:
        """Convenience wrapper: project an existing ``Position`` object to SQLite.

        Converts the dataclass fields into the pos-dict format expected by
        ``project_to_sqlite()`` so callers don't need to know the internal
        key names.
        """
        if status is None:
            status = position.status
        side_val = (
            position.side.value
            if hasattr(position.side, "value")
            else str(position.side)
        )
        pos: Dict[str, Any] = {
            "side": side_val,
            "entry_price": float(position.entry_price or 0.0),
            "initial_size": float(position.initial_size or 0.0),
            "qty": float(position.current_size or position.initial_size or 0.0),
            "entry_time": position.entry_time,
            "stop_loss_price": position.stop_loss_price,
            "take_profit_price": position.take_profit_price,
            "archetype": position.archetype or position.strategy_id or "",
            "_add_position_seq": position.add_count,
            "_parent_pid": position.parent_position_id,
            "notes": position.notes,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
        }
        return self.project_to_sqlite(
            position.position_id,
            pos,
            status=status,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
