#!/usr/bin/env python3
"""One-off: restore falsely-expired chop_grid entry fills from Binance REST.

Also reports short TP coverage (S1_tp qty vs inventory).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.order_management.binance_api import BinanceAPI
from src.order_management.multi_leg_order_backfill import normalize_rest_order_status
from src.order_management.multi_leg_storage import MultiLegStorage

logger = logging.getLogger(__name__)

GROUP = "BNBUSDT_2026-05-19 08:40:00+00:00"
LEGS = ("L1", "L2", "S1", "S2")


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BNBUSDT")
    p.add_argument("--group", default=GROUP)
    p.add_argument("--db", default="data/multi_leg_order_management.db")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    symbol = str(args.symbol).upper()
    group = str(args.group)
    api = _api_from_env()
    storage = MultiLegStorage(str(args.db))
    run_id = storage.create_run(
        mode="backfill",
        strategies=["chop_grid"],
        symbols=[symbol],
        run_id="backfill_entry_fills",
    )

    short_filled_qty = 0.0
    tp_cover_qty = 0.0

    for leg in LEGS:
        local_id = f"{group}_{leg}"
        row = (
            storage.get_order_by_local_id(local_id)
            if hasattr(storage, "get_order_by_local_id")
            else None
        )
        if row is None:
            import sqlite3

            conn = sqlite3.connect(str(args.db))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM multi_leg_orders WHERE local_order_id = ?",
                (local_id,),
            ).fetchone()
            conn.close()
            row = dict(row) if row else None
        if not row:
            logger.warning("missing local row %s", local_id)
            continue
        ex_id = str(row.get("exchange_order_id") or "")
        cid = str(row.get("client_order_id") or "")
        snap = (
            api.get_order(ex_id, symbol, client_order_id=cid) if ex_id or cid else None
        )
        if not snap:
            logger.info("%s: no exchange snapshot", leg)
            continue
        status = normalize_rest_order_status(snap.get("status"))
        filled = float(snap.get("filled") or 0)
        avg = snap.get("average_price")
        payload = {
            "run_id": run_id,
            "strategy": "chop_grid",
            "symbol": symbol,
            "order_id": str(snap.get("order_id") or ex_id),
            "client_order_id": snap.get("client_order_id") or cid,
            "status": status,
            "filled_qty": filled,
            "avg_price": avg,
            "event_time": snap.get("update_time") or snap.get("timestamp"),
            "trade_time": snap.get("update_time") or snap.get("timestamp"),
            "raw": snap,
        }
        logger.info(
            "%s local=%s -> exchange status=%s filled=%s avg=%s",
            leg,
            local_id,
            status,
            filled,
            avg,
        )
        if leg in {"S1", "S2"} and status == "filled" and filled > 0:
            short_filled_qty += filled
        if args.execute and status in {"filled", "canceled", "open"}:
            storage.apply_execution_report(payload)

    tp_row = None
    import sqlite3

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    tp_row = conn.execute(
        "SELECT * FROM multi_leg_orders WHERE local_order_id = ?",
        (f"{group}_S1_tp",),
    ).fetchone()
    conn.close()
    if tp_row:
        tp_row = dict(tp_row)
        ex_id = str(tp_row.get("exchange_order_id") or "")
        tp_snap = api.get_order(ex_id, symbol) if ex_id else None
        if tp_snap and str(tp_snap.get("status") or "").lower() == "open":
            tp_cover_qty = float(
                tp_snap.get("quantity") or tp_snap.get("remaining") or 0
            )

    logger.info(
        "SHORT filled legs qty=%.4f open S1_tp cover qty=%.4f",
        short_filled_qty,
        tp_cover_qty,
    )
    if short_filled_qty > tp_cover_qty + 1e-6:
        logger.warning(
            "S1_tp only covers part of short inventory; run repair_chop_grid_protection "
            "to place missing reduce-only TP (e.g. S2_tp qty=%.4f).",
            short_filled_qty - tp_cover_qty,
        )

    state_path = Path("data/multi_leg_live/state") / f"chop_grid_{symbol}.json"
    if state_path.is_file():
        st = json.loads(state_path.read_text())
        logger.info("engine inventory: %s", st.get("inventory"))

    if not args.execute:
        logger.info("Dry-run; pass --execute to write DB rows.")
        return
    storage.finish_run(run_id, status="done")
    logger.info(
        "DB updated. Refresh CMS; consider repair_chop_grid_protection --execute."
    )


if __name__ == "__main__":
    main()
