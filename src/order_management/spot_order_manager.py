from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
    payload: Dict[str, Any]


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
        self.client_prefix = "".join(ch for ch in client_prefix if ch.isalnum())[:12] or "sa"
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
            conn.execute(
                """
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
                """
            )
            conn.commit()

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

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO spot_orders (
                    order_id, created_at, symbol, side, order_type, quantity, price,
                    status, exchange_order_id, client_order_id, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(payload) if payload else None,
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
            payload=payload,
        )
