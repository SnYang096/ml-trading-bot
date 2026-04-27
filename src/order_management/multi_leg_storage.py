"""SQLite persistence for standalone multi-leg runtime state.

These tables are intentionally isolated from the classic ``orders`` and
``positions`` tables used by ``OrderManager`` / ``PositionManager``. Multi-leg
engines own per-leg inventory, while this layer stores exchange mappings and
append-only audit events for restart/reconciliation.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class MultiLegStorage:
    """Persistence helper for multi-leg runs, orders, positions, and events."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        schema_file = Path(__file__).parent / "database" / "schema.sql"
        conn = self._connect()
        try:
            conn.executescript(schema_file.read_text(encoding="utf-8"))
            conn.commit()
        finally:
            conn.close()

    def create_run(
        self,
        *,
        mode: str,
        strategies: Iterable[str],
        symbols: Iterable[str],
        account_label: str = "multi_leg",
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        rid = run_id or f"mlr_{uuid.uuid4().hex}"
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_runs (
                    run_id, mode, strategies, symbols, account_label, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    str(mode),
                    ",".join(str(x) for x in strategies),
                    ",".join(str(x) for x in symbols),
                    str(account_label),
                    _json(config or {}),
                ),
            )
            conn.commit()
            return rid
        finally:
            conn.close()

    def finish_run(self, run_id: str, *, status: str = "stopped") -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE multi_leg_runs
                SET status = ?, ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (str(status), str(run_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_order(self, payload: Dict[str, Any]) -> str:
        local_order_id = str(
            payload.get("local_order_id")
            or payload.get("order_id")
            or f"mlo_{uuid.uuid4().hex}"
        )
        row = {
            "local_order_id": local_order_id,
            "run_id": payload.get("run_id"),
            "strategy": payload.get("strategy") or "",
            "symbol": payload.get("symbol") or "",
            "leg_id": payload.get("leg_id") or payload.get("position_id"),
            "side": payload.get("side") or "",
            "position_side": payload.get("position_side") or payload.get("positionSide"),
            "order_type": payload.get("order_type") or payload.get("type") or "",
            "purpose": payload.get("purpose") or payload.get("action"),
            "quantity": float(payload.get("quantity") or payload.get("amount") or 0.0),
            "price": _optional_float(payload.get("price")),
            "stop_price": _optional_float(payload.get("stop_price")),
            "client_order_id": _none_if_blank(payload.get("client_order_id")),
            "exchange_order_id": _none_if_blank(
                payload.get("exchange_order_id") or payload.get("binance_order_id")
            ),
            "status": payload.get("status") or "unknown",
            "raw_json": _json(payload.get("raw") or payload),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_orders (
                    local_order_id, run_id, strategy, symbol, leg_id, side,
                    position_side, order_type, purpose, quantity, price, stop_price,
                    client_order_id, exchange_order_id, status, raw_json
                )
                VALUES (
                    :local_order_id, :run_id, :strategy, :symbol, :leg_id, :side,
                    :position_side, :order_type, :purpose, :quantity, :price,
                    :stop_price, :client_order_id, :exchange_order_id, :status,
                    :raw_json
                )
                ON CONFLICT(local_order_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    strategy = excluded.strategy,
                    symbol = excluded.symbol,
                    leg_id = excluded.leg_id,
                    side = excluded.side,
                    position_side = excluded.position_side,
                    order_type = excluded.order_type,
                    purpose = excluded.purpose,
                    quantity = excluded.quantity,
                    price = excluded.price,
                    stop_price = excluded.stop_price,
                    client_order_id = excluded.client_order_id,
                    exchange_order_id = excluded.exchange_order_id,
                    status = excluded.status,
                    raw_json = excluded.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                row,
            )
            conn.commit()
            return local_order_id
        finally:
            conn.close()

    def upsert_position(self, payload: Dict[str, Any]) -> str:
        leg_id = str(payload.get("leg_id") or payload.get("position_id") or "")
        if not leg_id:
            raise ValueError("multi-leg position requires leg_id")
        row = {
            "leg_id": leg_id,
            "run_id": payload.get("run_id"),
            "strategy": payload.get("strategy") or "",
            "symbol": payload.get("symbol") or "",
            "side": payload.get("side") or "",
            "entry_price": float(payload.get("entry_price") or 0.0),
            "quantity": float(payload.get("quantity") or 0.0),
            "status": payload.get("status") or "open",
            "parent_leg_id": payload.get("parent_leg_id"),
            "protection_order_ids": _json(payload.get("protection_order_ids") or []),
            "raw_json": _json(payload.get("raw") or payload),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_positions (
                    leg_id, run_id, strategy, symbol, side, entry_price, quantity,
                    status, parent_leg_id, protection_order_ids, raw_json
                )
                VALUES (
                    :leg_id, :run_id, :strategy, :symbol, :side, :entry_price,
                    :quantity, :status, :parent_leg_id, :protection_order_ids,
                    :raw_json
                )
                ON CONFLICT(leg_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    strategy = excluded.strategy,
                    symbol = excluded.symbol,
                    side = excluded.side,
                    entry_price = excluded.entry_price,
                    quantity = excluded.quantity,
                    status = excluded.status,
                    parent_leg_id = excluded.parent_leg_id,
                    protection_order_ids = excluded.protection_order_ids,
                    raw_json = excluded.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                row,
            )
            conn.commit()
            return leg_id
        finally:
            conn.close()

    def record_execution_report(self, payload: Dict[str, Any]) -> str:
        event_id = str(payload.get("event_id") or f"mle_{uuid.uuid4().hex}")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_execution_reports (
                    event_id, run_id, strategy, symbol, order_id, client_order_id,
                    status, execution_type, raw_json, event_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    payload.get("run_id"),
                    payload.get("strategy"),
                    payload.get("symbol"),
                    payload.get("order_id"),
                    payload.get("client_order_id"),
                    payload.get("status"),
                    payload.get("execution_type"),
                    _json(payload.get("raw") or payload),
                    payload.get("event_time") or payload.get("trade_time"),
                ),
            )
            conn.commit()
            return event_id
        finally:
            conn.close()

    def record_reconciliation_snapshot(self, payload: Dict[str, Any]) -> str:
        snapshot_id = str(payload.get("snapshot_id") or f"mlr_{uuid.uuid4().hex}")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_reconciliation_snapshots (
                    snapshot_id, run_id, strategy, symbol, ok, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    payload.get("run_id"),
                    payload.get("strategy"),
                    payload.get("symbol"),
                    1 if bool(payload.get("ok", False)) else 0,
                    _json(payload.get("raw") or payload),
                ),
            )
            conn.commit()
            return snapshot_id
        finally:
            conn.close()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _none_if_blank(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
