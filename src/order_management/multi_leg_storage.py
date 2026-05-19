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
        schema_file = Path(__file__).parent / "database" / "schema_multi_leg.sql"
        conn = self._connect()
        try:
            conn.executescript(schema_file.read_text(encoding="utf-8"))
            self._ensure_multi_leg_order_columns(conn)
            conn.commit()
        finally:
            conn.close()

    def _ensure_multi_leg_order_columns(self, conn: sqlite3.Connection) -> None:
        """Backfill new multi_leg_orders columns for existing DB files."""
        cursor = conn.execute("PRAGMA table_info(multi_leg_orders)")
        existing = {str(row[1]) for row in cursor.fetchall()}
        alters = [
            ("filled_quantity", "REAL DEFAULT 0"),
            ("average_price", "REAL"),
            ("commission", "REAL DEFAULT 0"),
            ("commission_asset", "TEXT"),
            ("filled_at", "TIMESTAMP"),
            ("canceled_at", "TIMESTAMP"),
            ("error_message", "TEXT"),
        ]
        for col, ddl in alters:
            if col not in existing:
                conn.execute(f"ALTER TABLE multi_leg_orders ADD COLUMN {col} {ddl}")

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
            "position_side": payload.get("position_side")
            or payload.get("positionSide"),
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
            "filled_quantity": float(payload.get("filled_quantity") or 0.0),
            "average_price": _optional_float(payload.get("average_price")),
            "commission": float(payload.get("commission") or 0.0),
            "commission_asset": _none_if_blank(payload.get("commission_asset")),
            "filled_at": payload.get("filled_at"),
            "canceled_at": payload.get("canceled_at"),
            "error_message": _none_if_blank(payload.get("error_message")),
            "raw_json": _json(payload.get("raw") or payload),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO multi_leg_orders (
                    local_order_id, run_id, strategy, symbol, leg_id, side,
                    position_side, order_type, purpose, quantity, price, stop_price,
                    client_order_id, exchange_order_id, status,
                    filled_quantity, average_price, commission, commission_asset,
                    filled_at, canceled_at, error_message, raw_json
                )
                VALUES (
                    :local_order_id, :run_id, :strategy, :symbol, :leg_id, :side,
                    :position_side, :order_type, :purpose, :quantity, :price,
                    :stop_price, :client_order_id, :exchange_order_id, :status,
                    :filled_quantity, :average_price, :commission, :commission_asset,
                    :filled_at, :canceled_at, :error_message, :raw_json
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
                    filled_quantity = excluded.filled_quantity,
                    average_price = excluded.average_price,
                    commission = excluded.commission,
                    commission_asset = excluded.commission_asset,
                    filled_at = excluded.filled_at,
                    canceled_at = excluded.canceled_at,
                    error_message = excluded.error_message,
                    raw_json = excluded.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                row,
            )
            conn.commit()
            return local_order_id
        finally:
            conn.close()

    def get_recent_orders_for_backfill(
        self,
        *,
        lookback_hours: int = 168,
        limit: int = 200,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent multi-leg rows that likely need REST status refresh.

        Candidates include:
        - non-terminal rows with exchange ids (still expected to evolve),
        - filled rows missing avg/filled_at.

        Terminal cancels often have no exchange error text. Treat them as done;
        otherwise REST snapshots with a null reason are written every interval.
        """
        conn = self._connect()
        try:
            where_extra = ""
            params: list[Any] = [max(1, int(lookback_hours))]
            if strategy:
                where_extra += " AND strategy = ? "
                params.append(str(strategy))
            if symbol:
                where_extra += " AND symbol = ? "
                params.append(str(symbol).upper())
            params.append(max(1, int(limit)))
            cur = conn.execute(
                f"""
                SELECT
                    local_order_id, run_id, strategy, symbol, status,
                    exchange_order_id, client_order_id, filled_quantity,
                    average_price, filled_at, canceled_at, error_message,
                    created_at, updated_at
                FROM multi_leg_orders
                WHERE
                    created_at >= datetime('now', '-' || ? || ' hours')
                    AND exchange_order_id IS NOT NULL
                    AND (
                        LOWER(TRIM(COALESCE(status, ''))) IN (
                            'submitted', 'open', 'pending',
                            'partially_filled', 'unknown', 'new'
                        )
                        OR (
                            LOWER(TRIM(COALESCE(status, ''))) = 'filled'
                            AND (average_price IS NULL OR filled_at IS NULL)
                        )
                    )
                    {where_extra}
                ORDER BY COALESCE(updated_at, created_at) ASC
                LIMIT ?
                """,
                tuple(params),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def apply_execution_report(self, payload: Dict[str, Any]) -> int:
        """Update multi_leg_orders status/fill fields from user-stream report."""
        exchange_order_id = _none_if_blank(payload.get("order_id"))
        client_order_id = _none_if_blank(payload.get("client_order_id"))
        if exchange_order_id is None and client_order_id is None:
            return 0

        status_raw = str(payload.get("status") or "").strip()
        status = status_raw.lower() if status_raw else "unknown"
        filled_qty = _optional_float(payload.get("filled_qty"))
        avg_price = _optional_float(payload.get("avg_price"))
        if avg_price is None:
            lp = _optional_float(payload.get("last_filled_price"))
            if lp is not None and lp > 0:
                avg_price = lp
        commission = _optional_float(payload.get("commission"))
        commission_asset = _none_if_blank(payload.get("commission_asset"))
        evt_time = payload.get("trade_time") or payload.get("event_time")
        is_terminal = status in {"filled", "canceled", "rejected", "expired"}
        filled_at = evt_time if status == "filled" else None
        canceled_at = (
            evt_time if status in {"canceled", "rejected", "expired"} else None
        )
        error_message = _none_if_blank(
            payload.get("error_message") or payload.get("reject_reason")
        )

        conn = self._connect()
        try:
            clauses = []
            params: list[Any] = []
            if exchange_order_id is not None:
                clauses.append("exchange_order_id = ?")
                params.append(exchange_order_id)
            if client_order_id is not None:
                clauses.append("client_order_id = ?")
                params.append(client_order_id)
            where = " OR ".join(clauses)
            cur = conn.execute(
                f"""
                SELECT status, filled_quantity, average_price, error_message
                FROM multi_leg_orders
                WHERE {where}
                """,
                tuple(params),
            )
            existing = cur.fetchone()
            if existing is None:
                changed = 0
            else:
                old_status = str(existing["status"] or "").strip().lower()
                old_filled = _optional_float(existing["filled_quantity"])
                old_avg = _optional_float(existing["average_price"])
                old_err = _none_if_blank(existing["error_message"])
                same_status = old_status == status
                same_filled = filled_qty is None or (
                    old_filled is not None
                    and filled_qty is not None
                    and abs(old_filled - filled_qty) < 1e-12
                )
                same_avg = avg_price is None or (
                    old_avg is not None
                    and avg_price is not None
                    and abs(old_avg - avg_price) < 1e-12
                )
                same_err = error_message is None or old_err == error_message
                if same_status and same_filled and same_avg and same_err:
                    return 0
            cur = conn.execute(
                f"""
                UPDATE multi_leg_orders
                SET status = ?,
                    filled_quantity = COALESCE(?, filled_quantity),
                    average_price = COALESCE(?, average_price),
                    commission = COALESCE(?, commission),
                    commission_asset = COALESCE(?, commission_asset),
                    filled_at = CASE
                        WHEN ? IS NOT NULL THEN ?
                        ELSE filled_at
                    END,
                    canceled_at = CASE
                        WHEN ? IS NOT NULL THEN ?
                        ELSE canceled_at
                    END,
                    error_message = COALESCE(?, error_message),
                    updated_at = CURRENT_TIMESTAMP
                WHERE {where}
                """,
                [
                    status,
                    filled_qty,
                    avg_price,
                    commission,
                    commission_asset,
                    filled_at,
                    filled_at,
                    canceled_at if is_terminal else None,
                    canceled_at,
                    error_message,
                    *params,
                ],
            )
            conn.commit()
            changed = int(cur.rowcount or 0)
            if changed == 0:
                # Ensure terminal events are not dropped even if place-result row
                # was missed; use client/exchange id as deterministic local key.
                local_key = (
                    _none_if_blank(payload.get("client_order_id"))
                    or _none_if_blank(payload.get("order_id"))
                    or f"mlo_{uuid.uuid4().hex}"
                )
                self.upsert_order(
                    {
                        "local_order_id": local_key,
                        "run_id": payload.get("run_id"),
                        "strategy": payload.get("strategy") or "unknown",
                        "symbol": payload.get("symbol") or "",
                        "side": "",
                        "order_type": "unknown",
                        "purpose": "execution_report",
                        "quantity": 0.0,
                        "client_order_id": client_order_id,
                        "exchange_order_id": exchange_order_id,
                        "status": status,
                        "filled_quantity": filled_qty or 0.0,
                        "average_price": avg_price,
                        "commission": commission or 0.0,
                        "commission_asset": commission_asset,
                        "filled_at": filled_at,
                        "canceled_at": canceled_at,
                        "error_message": error_message,
                        "raw": payload.get("raw") or payload,
                    }
                )
                return 1
            return changed
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

    def close_absent_positions(
        self,
        *,
        strategy: str,
        symbol: str,
        active_leg_ids: Iterable[str],
        run_id: Optional[str] = None,
    ) -> int:
        """Close open DB rows no longer present in the engine inventory snapshot."""
        active = [str(x) for x in active_leg_ids if str(x)]
        conn = self._connect()
        try:
            params: list[Any] = [run_id, str(strategy), str(symbol), *active]
            not_in = ""
            if active:
                not_in = f"AND leg_id NOT IN ({','.join('?' for _ in active)})"
            cur = conn.execute(
                f"""
                UPDATE multi_leg_positions
                SET status = 'closed',
                    run_id = COALESCE(?, run_id),
                    closed_at = COALESCE(closed_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE strategy = ?
                  AND symbol = ?
                  AND LOWER(TRIM(COALESCE(status, ''))) = 'open'
                  {not_in}
                """,
                params,
            )
            conn.commit()
            return int(cur.rowcount or 0)
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
