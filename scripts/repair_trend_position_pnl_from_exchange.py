#!/usr/bin/env python3
"""Backfill positions.exit_price / realized_pnl from Binance allOrders + userTrades.

Use when local closes exist (often exchange_sync_flat) but realized_pnl=0 or
exit_price=entry_price, and/or no filled close order in SQLite — e.g. exchange
SL triggered without being linked to position_id in orders table.

Dry-run (default):

  python3 scripts/repair_trend_position_pnl_from_exchange.py \\
    --db /app/data/order_management.db --since 2026-05-01

Apply:

  python3 scripts/repair_trend_position_pnl_from_exchange.py --db ... --apply
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from scripts.reconcile_trend_closes_with_exchange import (  # noqa: E402
    _fetch_all_orders,
    _fetch_user_trades,
    _ms_to_iso,
    _parse_ts,
    _since_ms,
)
from src.order_management.binance_api import BinanceAPI  # noqa: E402

_WINDOW_AFTER_EXIT_MS = 14 * 24 * 3600 * 1000


def _pos_side_norm(side: str) -> str:
    s = str(side or "long").lower()
    return "short" if s in {"short", "sell"} else "long"


def _close_order_side(pos_side: str) -> str:
    return "BUY" if _pos_side_norm(pos_side) == "short" else "SELL"


def _realized_pnl(pos_side: str, entry: float, exit_px: float, qty: float) -> float:
    if qty <= 0 or entry <= 0 or exit_px <= 0:
        return 0.0
    if _pos_side_norm(pos_side) == "short":
        return (entry - exit_px) * qty
    return (exit_px - entry) * qty


def _order_position_side(row: Dict[str, Any]) -> str:
    ps = str(row.get("positionSide") or "BOTH").upper()
    if ps == "SHORT":
        return "short"
    if ps == "LONG":
        return "long"
    side = str(row.get("side") or "").upper()
    if side == "SELL":
        return "short"
    if side == "BUY":
        return "long"
    return "long"


def _needs_repair(row: Dict[str, Any]) -> bool:
    if str(row.get("status") or "").lower() != "closed":
        return False
    if not row.get("exit_time"):
        return False
    reason = str(row.get("exit_reason") or "")
    if reason.startswith("exchange_sync"):
        return True
    try:
        rpnl = float(row.get("realized_pnl") or 0.0)
        entry = float(row.get("entry_price") or 0.0)
        exit_px = float(row.get("exit_price") or 0.0)
    except (TypeError, ValueError):
        return True
    if abs(rpnl) > 1e-9 and abs(entry - exit_px) > 1e-9:
        return False
    return abs(rpnl) <= 1e-9 or abs(entry - exit_px) <= 1e-9


def _position_qty(row: Dict[str, Any], entry_qty: float) -> float:
    for key in ("current_size", "initial_size"):
        try:
            v = float(row.get(key) or 0.0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return entry_qty


def _entry_qty_from_orders(conn: sqlite3.Connection, pid: str, pos_side: str) -> float:
    entry_side = "sell" if _pos_side_norm(pos_side) == "short" else "buy"
    row = conn.execute(
        """
        SELECT filled_quantity, quantity FROM orders
        WHERE position_id = ? AND lower(side) = ? AND lower(status) = 'filled'
        ORDER BY COALESCE(filled_at, created_at) ASC LIMIT 1
        """,
        (pid, entry_side),
    ).fetchone()
    if not row:
        return 0.0
    try:
        return float(row[0] or row[1] or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _has_filled_close_order(conn: sqlite3.Connection, pid: str, pos_side: str) -> bool:
    close_side = _close_order_side(pos_side).lower()
    row = conn.execute(
        """
        SELECT 1 FROM orders
        WHERE position_id = ? AND lower(side) = ? AND lower(status) = 'filled'
        LIMIT 1
        """,
        (pid, close_side),
    ).fetchone()
    return row is not None


def _pick_exchange_close(
    orders: List[Dict[str, Any]],
    *,
    pos_side: str,
    entry_ms: int,
    exit_ms: Optional[int],
) -> Optional[Dict[str, Any]]:
    close_side = _close_order_side(pos_side)
    end_ms = (exit_ms or entry_ms) + _WINDOW_AFTER_EXIT_MS
    candidates: List[Dict[str, Any]] = []
    for row in orders:
        if str(row.get("status") or "").upper() != "FILLED":
            continue
        if str(row.get("side") or "").upper() != close_side:
            continue
        if _order_position_side(row) != _pos_side_norm(pos_side):
            continue
        t_ms = int(row.get("updateTime") or row.get("time") or 0)
        if t_ms < entry_ms or t_ms > end_ms:
            continue
        candidates.append(row)
    if not candidates:
        return None

    def _score(row: Dict[str, Any]) -> Tuple[int, int, int]:
        typ = str(row.get("type") or "").upper()
        stop_rank = 0 if "STOP" in typ else 1
        t_ms = int(row.get("updateTime") or row.get("time") or 0)
        if exit_ms is not None:
            dist = abs(t_ms - exit_ms)
        else:
            dist = t_ms - entry_ms
        return (stop_rank, dist, -t_ms)

    return sorted(candidates, key=_score)[0]


def _trades_for_order(
    trades: List[Dict[str, Any]], order_id: str
) -> List[Dict[str, Any]]:
    oid = str(order_id or "")
    return [t for t in trades if str(t.get("orderId") or "") == oid]


def _repair_positions_for_symbol(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    all_orders: List[Dict[str, Any]],
    user_trades: List[Dict[str, Any]],
    dry_run: bool,
    record_order: bool,
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    rows = conn.execute(
        """
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, exit_reason,
               current_size, initial_size, strategy_id
        FROM positions
        WHERE symbol = ? AND lower(status) = 'closed'
        ORDER BY entry_time
        """,
        (sym,),
    ).fetchall()
    cols = [
        "position_id",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "realized_pnl",
        "status",
        "exit_reason",
        "current_size",
        "initial_size",
        "strategy_id",
    ]
    applied: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for raw in rows:
        pos = dict(zip(cols, raw))
        if not _needs_repair(pos):
            continue
        pid = str(pos["position_id"])
        pos_side = str(pos.get("side") or "long")
        entry_ms = (_parse_ts(pos.get("entry_time")) or 0) * 1000
        exit_ms = (_parse_ts(pos.get("exit_time")) or 0) * 1000 or None
        entry_qty = _entry_qty_from_orders(conn, pid, pos_side)
        qty = _position_qty(pos, entry_qty)

        ex = _pick_exchange_close(
            all_orders,
            pos_side=pos_side,
            entry_ms=entry_ms,
            exit_ms=exit_ms,
        )
        if ex is None:
            applied.append(
                {
                    "position_id": pid,
                    "status": "no_exchange_match",
                    "exit_reason": pos.get("exit_reason"),
                }
            )
            continue

        exit_px = float(ex.get("avgPrice") or 0.0)
        ex_qty = float(ex.get("executedQty") or qty or 0.0)
        if ex_qty > 0:
            qty = min(qty, ex_qty) if qty > 0 else ex_qty
        trades = _trades_for_order(user_trades, str(ex.get("orderId") or ""))
        if trades:
            wsum = sum(
                float(t.get("price") or 0) * float(t.get("qty") or 0) for t in trades
            )
            tqty = sum(float(t.get("qty") or 0) for t in trades)
            if tqty > 0:
                exit_px = wsum / tqty
                qty = tqty
            rpnl_trade = sum(float(t.get("realizedPnl") or 0) for t in trades)
        else:
            rpnl_trade = None

        entry_px = float(pos.get("entry_price") or 0.0)
        rpnl = (
            float(rpnl_trade)
            if rpnl_trade is not None and abs(float(rpnl_trade)) > 1e-12
            else _realized_pnl(pos_side, entry_px, exit_px, qty)
        )
        ex_ms = int(ex.get("updateTime") or ex.get("time") or 0)
        ex_iso = _ms_to_iso(ex_ms) if ex_ms else str(pos.get("exit_time"))

        detail = {
            "position_id": pid,
            "status": "would_update" if dry_run else "updated",
            "exit_reason": pos.get("exit_reason"),
            "entry_price": entry_px,
            "old_exit_price": float(pos.get("exit_price") or 0.0),
            "new_exit_price": exit_px,
            "qty": qty,
            "realized_pnl": round(rpnl, 6),
            "exchange_order_id": str(ex.get("orderId") or ""),
            "exchange_type": str(ex.get("type") or ""),
            "exchange_time": ex_iso,
        }
        applied.append(detail)

        if dry_run:
            continue

        conn.execute(
            """
            UPDATE positions SET
                exit_time = ?,
                exit_price = ?,
                realized_pnl = ?,
                current_size = 0
            WHERE position_id = ?
            """,
            (ex_iso, exit_px, rpnl, pid),
        )

        ex_oid = str(ex.get("orderId") or "")
        if record_order and ex_oid and not _has_filled_close_order(conn, pid, pos_side):
            exists = conn.execute(
                "SELECT 1 FROM orders WHERE binance_order_id = ? LIMIT 1", (ex_oid,)
            ).fetchone()
            if not exists:
                local_oid = f"order_{uuid.uuid4().hex}"
                conn.execute(
                    """
                    INSERT INTO orders (
                        order_id, binance_order_id, position_id, symbol, side,
                        order_type, quantity, average_price, filled_quantity,
                        status, created_at, updated_at, filled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, ?)
                    """,
                    (
                        local_oid,
                        ex_oid,
                        pid,
                        sym,
                        _close_order_side(pos_side).lower(),
                        str(ex.get("type") or "market").lower(),
                        qty,
                        exit_px,
                        qty,
                        ex_iso,
                        now,
                        ex_iso,
                    ),
                )
                detail["inserted_close_order"] = local_oid

    if not dry_run:
        conn.commit()
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--since", default="2026-05-01")
    parser.add_argument("--symbol", default="", help="Single symbol or empty = all")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--record-order",
        action="store_true",
        help="Insert filled close row into orders when missing",
    )
    args = parser.parse_args()
    if not args.db.is_file():
        raise SystemExit(f"DB not found: {args.db}")

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
        "BINANCE_FUTURES_API_SECRET", ""
    )
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")

    conn = sqlite3.connect(str(args.db))
    if args.symbol:
        symbols = [str(args.symbol).upper()]
    else:
        symbols = sorted(
            {
                str(r[0] or "").upper()
                for r in conn.execute(
                    """
                    SELECT DISTINCT symbol FROM positions
                    WHERE lower(status) = 'closed'
                    """
                )
                if str(r[0] or "").strip()
            }
        )

    api = BinanceAPI(api_key, api_secret, testnet=False)
    start_ms = _since_ms(str(args.since))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] repair_trend_position_pnl_from_exchange since={args.since}")

    total = 0
    for sym in symbols:
        print(f"\n=== {sym} ===")
        print("Fetching allOrders...")
        all_orders = _fetch_all_orders(api, sym, start_ms)
        print("Fetching userTrades...")
        user_trades = _fetch_user_trades(api, sym, start_ms)
        print(f"  orders={len(all_orders)} trades={len(user_trades)}")
        rows = _repair_positions_for_symbol(
            conn,
            symbol=sym,
            all_orders=all_orders,
            user_trades=user_trades,
            dry_run=not args.apply,
            record_order=bool(args.record_order),
        )
        for row in rows:
            if row.get("status") in {"updated", "would_update"}:
                total += 1
                print(
                    f"  {row['position_id']}: "
                    f"{row.get('old_exit_price')} -> {row.get('new_exit_price')} "
                    f"pnl={row.get('realized_pnl')} "
                    f"ex={row.get('exchange_type')} @{row.get('exchange_time')} "
                    f"orderId={row.get('exchange_order_id')}"
                )
            elif row.get("status") == "no_exchange_match":
                print(
                    f"  {row['position_id']}: no exchange close match ({row.get('exit_reason')})"
                )

    print(f"\n[{mode}] repaired_or_candidate={total}")


if __name__ == "__main__":
    main()
