from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.order_management.spot_binance_api import SpotBinanceAPI

logger = logging.getLogger(__name__)


@dataclass
class SpotOrderResult:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    exchange_order_id: Optional[str]
    client_order_id: str = ""
    payload: Optional[Dict[str, Any]] = None


class SpotOrderManager:
    """Spot order recorder + optional executor."""

    def __init__(
        self,
        *,
        db_path: str,
        api: Optional[SpotBinanceAPI],
        shadow: bool = False,
        client_prefix: str = "sa",
    ) -> None:
        self.db_path = Path(db_path)
        self.api = api
        self.shadow = bool(shadow)
        self.client_prefix = (
            "".join(ch for ch in client_prefix if ch.isalnum())[:12] or "sa"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if self.shadow:
            logger.info("SpotOrderManager: shadow mode enabled")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spot_orders (
                    order_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    exchange_order_id TEXT,
                    client_order_id TEXT,
                    raw_json TEXT
                )
                """)
            self._ensure_order_columns(conn)
            # Additional indexes (2026-06-17 review)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_spot_orders_symbol_status ON spot_orders(symbol, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_spot_orders_exchange_order_id ON spot_orders(exchange_order_id)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_spot_orders_client_order_id ON spot_orders(client_order_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_spot_orders_created_at ON spot_orders(created_at)"
            )
            conn.commit()

    def _ensure_order_columns(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(spot_orders)")
        existing = {str(row[1]) for row in cursor.fetchall()}
        alters = [
            ("filled_quantity", "REAL DEFAULT 0"),
            ("filled_quote_usdt", "REAL DEFAULT 0"),
            ("updated_at", "TEXT"),
        ]
        for col, ddl in alters:
            if col not in existing:
                conn.execute(f"ALTER TABLE spot_orders ADD COLUMN {col} {ddl}")

    def _client_order_id(self) -> str:
        return f"{self.client_prefix}_{uuid.uuid4().hex}"[:36]

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> SpotOrderResult:
        oid = f"spot_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        status = "shadow" if self.shadow else "submitted"
        payload: Dict[str, Any] = {}
        exchange_order_id: Optional[str] = None
        client_order_id = self._client_order_id()

        if not self.shadow:
            if self.api is None:
                raise RuntimeError("spot api not initialized while shadow=false")
            payload = self.api.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
            )
            status = str(payload.get("status") or "submitted").lower()
            exchange_order_id = str(payload.get("id") or payload.get("orderId") or "")
            if not exchange_order_id:
                exchange_order_id = None

        filled_qty = 0.0
        filled_quote = 0.0
        if payload:
            try:
                filled_qty = float(payload.get("filled") or 0.0)
            except (TypeError, ValueError):
                filled_qty = 0.0
            try:
                cost = payload.get("cost")
                filled_quote = float(cost) if cost is not None else 0.0
            except (TypeError, ValueError):
                filled_quote = 0.0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO spot_orders (
                    order_id, created_at, symbol, side, order_type, quantity, price,
                    status, exchange_order_id, client_order_id, raw_json,
                    filled_quantity, filled_quote_usdt, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    oid,
                    now,
                    symbol.upper(),
                    side.lower(),
                    order_type.lower(),
                    float(quantity),
                    float(price) if price is not None else None,
                    status,
                    exchange_order_id,
                    client_order_id,
                    (
                        json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                        if payload
                        else None
                    ),
                    filled_qty,
                    filled_quote,
                    now,
                ),
            )
            conn.commit()

        return SpotOrderResult(
            order_id=oid,
            symbol=symbol.upper(),
            side=side.lower(),
            order_type=order_type.lower(),
            quantity=float(quantity),
            price=float(price) if price is not None else None,
            status=status,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            payload=payload,
        )

    def update_order_record(
        self,
        local_order_id: str,
        *,
        status: str,
        filled_quantity: Optional[float] = None,
        filled_quote_usdt: Optional[float] = None,
        raw_json: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE spot_orders
                SET status = ?, filled_quantity = COALESCE(?, filled_quantity),
                    filled_quote_usdt = COALESCE(?, filled_quote_usdt),
                    raw_json = COALESCE(?, raw_json),
                    updated_at = ?
                WHERE order_id = ?
                """,
                (
                    str(status).lower(),
                    filled_quantity,
                    filled_quote_usdt,
                    raw_json,
                    now,
                    local_order_id,
                ),
            )
            conn.commit()

    def find_order_id(
        self,
        *,
        exchange_order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[str]:
        clauses = []
        params: List[Any] = []
        if exchange_order_id:
            clauses.append("exchange_order_id = ?")
            params.append(str(exchange_order_id))
        if client_order_id:
            clauses.append("client_order_id = ?")
            params.append(str(client_order_id))
        if not clauses:
            return None
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT order_id FROM spot_orders WHERE {' OR '.join(clauses)} "
                "ORDER BY created_at DESC LIMIT 1",
                params,
            ).fetchone()
        return str(row["order_id"]) if row else None

    def list_orders_for_symbols(
        self,
        symbols: Iterable[str],
        *,
        sides: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        syms = [str(s).upper() for s in symbols]
        if not syms:
            return []
        placeholders = ",".join("?" for _ in syms)
        params: List[Any] = list(syms)
        side_clause = ""
        if sides is not None:
            side_list = [str(s).lower() for s in sides]
            if side_list:
                side_clause = f" AND side IN ({','.join('?' for _ in side_list)})"
                params.extend(side_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT order_id, created_at, symbol, side, order_type, quantity, price,
                       status, exchange_order_id, client_order_id, raw_json,
                       filled_quantity, filled_quote_usdt, updated_at
                FROM spot_orders
                WHERE symbol IN ({placeholders}){side_clause}
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def cancel_exchange_order(
        self, symbol: str, exchange_order_id: str
    ) -> Dict[str, Any]:
        if self.shadow or self.api is None:
            return {"status": "shadow"}
        return self.api.cancel_order(symbol, exchange_order_id)
