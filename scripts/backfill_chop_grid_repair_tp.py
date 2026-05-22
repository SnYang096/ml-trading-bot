#!/usr/bin/env python3
"""Backfill manual chop_grid repair TP orders into multi_leg_orders as canonical *_L{n}_tp rows.

Use when a leg was closed via cg_repair_* client order but never recorded as L1_tp / S1_tp.

Example (known repair on mainnet):
  python scripts/backfill_chop_grid_repair_tp.py \\
    --symbol BNBUSDT \\
    --group "BNBUSDT_2026-05-19 08:40:00+00:00" \\
    --leg L1 \\
    --client-order-id cg_repair_long_tp2 \\
    --exchange-order-id 90488444017 \\
    --execute

Dry-run (default): prints payload only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.order_management.binance_api import BinanceAPI
from src.order_management.multi_leg_storage import MultiLegStorage

logger = logging.getLogger(__name__)


def _api_from_env() -> BinanceAPI:
    key = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_KEY", "") or os.getenv(
        "MULTI_LEG_BINANCE_API_KEY", ""
    )
    secret = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_SECRET", "") or os.getenv(
        "MULTI_LEG_BINANCE_API_SECRET", ""
    )
    if not key or not secret:
        raise SystemExit("Set MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET")
    return BinanceAPI(key, secret, testnet=False)


def _fetch_order(
    api: BinanceAPI, symbol: str, *, exchange_order_id: str
) -> Dict[str, Any]:
    if not exchange_order_id:
        raise SystemExit("--exchange-order-id is required (BinanceAPI.get_order)")
    row = api.get_order(str(exchange_order_id), symbol)
    if not row:
        raise SystemExit(f"Order not found: {exchange_order_id}")
    return row


def _leg_entry_id(group: str, leg: str) -> str:
    leg = leg.strip().upper()
    if not leg.startswith(("L", "S")):
        raise SystemExit("--leg must be like L1 or S1")
    return f"{group}_{leg}"


def _position_side(leg: str) -> str:
    return "LONG" if leg.upper().startswith("L") else "SHORT"


def build_repair_tp_payload(
    *,
    group: str,
    leg: str,
    symbol: str,
    client_order_id: str,
    exchange_order: Dict[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    entry_id = _leg_entry_id(group, leg)
    local_tp = f"{entry_id}_tp"
    status = str(exchange_order.get("status") or "filled").lower()
    filled_qty = float(
        exchange_order.get("executed_qty")
        or exchange_order.get("filled_quantity")
        or exchange_order.get("quantity")
        or 0
    )
    avg_px = float(
        exchange_order.get("avg_price")
        or exchange_order.get("average_price")
        or exchange_order.get("price")
        or 0
    )
    limit_px = float(exchange_order.get("price") or 0)
    return {
        "run_id": run_id,
        "strategy": "chop_grid",
        "local_order_id": local_tp,
        "leg_id": entry_id,
        "symbol": symbol.upper(),
        "side": _position_side(leg),
        "purpose": "take_profit",
        "order_type": "limit",
        "quantity": filled_qty,
        "filled_quantity": filled_qty,
        "price": limit_px or avg_px,
        "average_price": avg_px,
        "status": status,
        "client_order_id": client_order_id
        or str(exchange_order.get("client_order_id") or ""),
        "exchange_order_id": str(
            exchange_order.get("order_id")
            or exchange_order.get("exchange_order_id")
            or ""
        ),
        "filled_at": exchange_order.get("update_time")
        or exchange_order.get("filled_at"),
        "created_at": exchange_order.get("time") or exchange_order.get("created_at"),
        "_repair_tp": True,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BNBUSDT")
    p.add_argument(
        "--group",
        required=True,
        help="Grid batch key, e.g. BNBUSDT_2026-05-19 08:40:00+00:00",
    )
    p.add_argument("--leg", required=True, help="Entry leg label: L1, S1, …")
    p.add_argument("--client-order-id", default="")
    p.add_argument("--exchange-order-id", default="")
    p.add_argument("--db", default="data/multi_leg_order_management.db")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    symbol = str(args.symbol).upper()
    api = _api_from_env()
    ex = _fetch_order(api, symbol, exchange_order_id=str(args.exchange_order_id or ""))
    storage = MultiLegStorage(str(args.db))
    run_id = storage.create_run(
        mode="backfill",
        strategies=["chop_grid"],
        symbols=[symbol],
        run_id="backfill_repair_tp",
    )
    payload = build_repair_tp_payload(
        group=str(args.group),
        leg=str(args.leg),
        symbol=symbol,
        client_order_id=str(args.client_order_id or ""),
        exchange_order=ex,
        run_id=run_id,
    )
    logger.info("payload: %s", json.dumps(payload, default=str, indent=2))
    if not args.execute:
        logger.info("Dry-run; pass --execute to upsert.")
        return
    storage.upsert_order(payload)
    storage.finish_run(run_id, status="done")
    logger.info(
        "Upserted %s (client=%s)", payload["local_order_id"], payload["client_order_id"]
    )


if __name__ == "__main__":
    main()
