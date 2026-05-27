#!/usr/bin/env python3
"""Backfill trend positions.realized_pnl from filled orders (repair + dry-run).

Use when console order rows show no PnL because:
- closed positions were never persisted, or
- positions were closed without realized_pnl / exit_price.

Does not touch rejected/pending stop orders.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _side_is_entry(side: str, pos_side: str = "long") -> bool:
    s = str(side or "").lower()
    ps = str(pos_side or "long").lower()
    if ps == "short":
        return s in {"sell", "short"}
    return s in {"buy", "long"}


def _realized_pnl(pos_side: str, entry_px: float, exit_px: float, qty: float) -> float:
    if qty <= 0 or entry_px <= 0 or exit_px <= 0:
        return 0.0
    if str(pos_side or "long").lower() == "short":
        return (entry_px - exit_px) * qty
    return (exit_px - entry_px) * qty


@dataclass
class _Leg:
    order_id: str
    side: str
    price: float
    qty: float
    ts: str


def _order_ts(row: Dict[str, Any]) -> str:
    for key in ("filled_at", "updated_at", "created_at"):
        v = row.get(key)
        if v:
            return str(v)
    return ""


def _filled_legs(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Optional[_Leg], Optional[_Leg]]:
    legs: List[_Leg] = []
    for row in rows:
        if str(row.get("status") or "").lower() != "filled":
            continue
        px = float(row.get("average_price") or 0.0)
        qty = float(row.get("filled_quantity") or row.get("quantity") or 0.0)
        if px <= 0 or qty <= 0:
            continue
        legs.append(
            _Leg(
                order_id=str(row.get("order_id") or ""),
                side=str(row.get("side") or ""),
                price=px,
                qty=qty,
                ts=_order_ts(row),
            )
        )
    if not legs:
        return None, None
    legs.sort(key=lambda x: x.ts)
    entry = None
    exit_leg = None
    for leg in legs:
        if _side_is_entry(leg.side):
            if entry is None:
                entry = leg
        elif entry is not None:
            exit_leg = leg
    return entry, exit_leg


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _fifo_pair_pnl_by_symbol(
    order_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Match filled buys/sells per symbol (FIFO) when position_id differs between legs."""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for row in order_rows:
        if str(row.get("status") or "").lower() != "filled":
            continue
        sym = str(row.get("symbol") or "").upper()
        if sym:
            by_sym.setdefault(sym, []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for sym, rows in by_sym.items():
        rows = sorted(rows, key=_order_ts)
        buys: List[Dict[str, Any]] = []
        for row in rows:
            side = str(row.get("side") or "").lower()
            px = float(row.get("average_price") or 0.0)
            qty = float(row.get("filled_quantity") or row.get("quantity") or 0.0)
            if px <= 0 or qty <= 0:
                continue
            if side in {"buy", "long"}:
                buys.append(row)
                continue
            if side not in {"sell", "short"} or not buys:
                continue
            entry = buys.pop(0)
            entry_px = float(entry.get("average_price") or 0.0)
            exit_px = px
            match_qty = min(
                float(entry.get("filled_quantity") or entry.get("quantity") or 0.0),
                qty,
            )
            pnl = _realized_pnl("long", entry_px, exit_px, match_qty)
            rec = {
                "pnl_usdt": float(pnl),
                "realized_pnl": float(pnl),
                "unrealized_pnl": None,
                "pnl_hint": "已实现",
            }
            exit_oid = str(row.get("order_id") or "")
            if exit_oid:
                out[exit_oid] = rec
            entry_oid = str(entry.get("order_id") or "")
            entry_pid = str(entry.get("position_id") or "")
            exit_pid = str(row.get("position_id") or "")
            if entry_pid:
                out[f"{entry_pid}:exit"] = rec
            if exit_pid and exit_pid != entry_pid:
                out[f"{exit_pid}:exit"] = rec
            if entry_oid:
                out[entry_oid] = rec
    return out


def backfill(
    db_path: Path,
    *,
    dry_run: bool = True,
) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    pos_cols = _table_columns(conn, "positions")

    pids = [
        str(r[0])
        for r in conn.execute(
            """
            SELECT DISTINCT position_id FROM orders
            WHERE coalesce(position_id, '') != ''
              AND lower(status) = 'filled'
            """
        )
    ]

    stats = {
        "position_ids_scanned": len(pids),
        "inserted": 0,
        "updated": 0,
        "skipped_open": 0,
        "skipped_no_pair": 0,
        "details": [],
    }

    now = datetime.now(timezone.utc).isoformat()

    for pid in pids:
        order_rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT order_id, symbol, side, status, filled_quantity, quantity,
                       average_price, created_at, filled_at, updated_at
                FROM orders
                WHERE position_id = ?
                ORDER BY COALESCE(filled_at, created_at) ASC
                """,
                (pid,),
            )
        ]
        entry, exit_leg = _filled_legs(order_rows)
        if entry is None:
            stats["skipped_no_pair"] += 1
            continue

        sym = str(order_rows[0].get("symbol") or "").upper()
        existing = conn.execute(
            "SELECT * FROM positions WHERE position_id = ?", (pid,)
        ).fetchone()

        if exit_leg is None:
            stats["skipped_open"] += 1
            if existing is None and not dry_run:
                conn.execute(
                    """
                    INSERT INTO positions (
                        position_id, symbol, side, entry_time, entry_price,
                        initial_size, current_size, total_cost, realized_pnl,
                        status, strategy_id, created_at, updated_at
                    ) VALUES (?, ?, 'long', ?, ?, ?, ?, ?, 0, 'open', 'tpc', ?, ?)
                    """,
                    (
                        pid,
                        sym,
                        entry.ts or now,
                        entry.price,
                        entry.qty,
                        entry.qty,
                        entry.price * entry.qty,
                        now,
                        now,
                    ),
                )
                stats["inserted"] += 1
            continue

        qty = min(entry.qty, exit_leg.qty)
        rpnl = _realized_pnl("long", entry.price, exit_leg.price, qty)
        detail = {
            "position_id": pid,
            "symbol": sym,
            "entry_price": entry.price,
            "exit_price": exit_leg.price,
            "qty": qty,
            "realized_pnl": round(rpnl, 4),
            "exit_order_id": exit_leg.order_id,
        }
        stats["details"].append(detail)

        if dry_run:
            continue

        if existing is None:
            conn.execute(
                """
                INSERT INTO positions (
                    position_id, symbol, side, entry_time, exit_time,
                    entry_price, exit_price, initial_size, current_size,
                    total_cost, realized_pnl, status, strategy_id,
                    exit_reason, created_at, updated_at
                ) VALUES (?, ?, 'long', ?, ?, ?, ?, ?, 0, ?, ?, 'closed', 'tpc',
                          'backfill_from_orders', ?, ?)
                """,
                (
                    pid,
                    sym,
                    entry.ts or now,
                    exit_leg.ts or now,
                    entry.price,
                    exit_leg.price,
                    entry.qty,
                    entry.qty * entry.price,
                    rpnl,
                    now,
                    now,
                ),
            )
            stats["inserted"] += 1
        else:
            sets = [
                "exit_time = COALESCE(exit_time, ?)",
                "exit_price = COALESCE(NULLIF(exit_price, 0), ?)",
                "realized_pnl = ?",
                "current_size = 0",
                "status = 'closed'",
                "updated_at = ?",
            ]
            params: List[Any] = [
                exit_leg.ts or now,
                exit_leg.price,
                rpnl,
                now,
            ]
            if "exit_reason" in pos_cols:
                sets.append(
                    "exit_reason = COALESCE(exit_reason, 'backfill_from_orders')"
                )
            params.append(pid)
            conn.execute(
                f"UPDATE positions SET {', '.join(sets)} WHERE position_id = ?",
                tuple(params),
            )
            stats["updated"] += 1

    fifo_rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT order_id, symbol, side, status, position_id,
                   filled_quantity, quantity, average_price,
                   created_at, filled_at, updated_at
            FROM orders
            WHERE lower(status) = 'filled'
            ORDER BY COALESCE(filled_at, created_at) ASC
            """
        )
    ]
    fifo_map = _fifo_pair_pnl_by_symbol(fifo_rows)
    stats["fifo_pairs"] = len(
        {k for k in fifo_map if not k.endswith(":exit") and "order_" in k}
    )

    if not dry_run:
        by_sym: Dict[str, List[Dict[str, Any]]] = {}
        for row in fifo_rows:
            sym = str(row.get("symbol") or "").upper()
            if sym:
                by_sym.setdefault(sym, []).append(row)
        for sym, rows in by_sym.items():
            rows = sorted(rows, key=_order_ts)
            buys: List[Dict[str, Any]] = []
            for row in rows:
                side = str(row.get("side") or "").lower()
                px = float(row.get("average_price") or 0.0)
                qty = float(row.get("filled_quantity") or row.get("quantity") or 0.0)
                if px <= 0 or qty <= 0:
                    continue
                if side in {"buy", "long"}:
                    buys.append(row)
                    continue
                if side not in {"sell", "short"} or not buys:
                    continue
                entry = buys.pop(0)
                entry_px = float(entry.get("average_price") or 0.0)
                match_qty = min(
                    float(entry.get("filled_quantity") or entry.get("quantity") or 0.0),
                    qty,
                )
                rpnl = _realized_pnl("long", entry_px, px, match_qty)
                for pid in {
                    str(entry.get("position_id") or ""),
                    str(row.get("position_id") or ""),
                }:
                    if not pid:
                        continue
                    existing = conn.execute(
                        "SELECT position_id FROM positions WHERE position_id = ?",
                        (pid,),
                    ).fetchone()
                    entry_ts = _order_ts(entry) or now
                    exit_ts = _order_ts(row) or now
                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO positions (
                                position_id, symbol, side, entry_time, exit_time,
                                entry_price, exit_price, initial_size, current_size,
                                total_cost, realized_pnl, status, strategy_id,
                                exit_reason, created_at, updated_at
                            ) VALUES (?, ?, 'long', ?, ?, ?, ?, ?, 0, ?, ?, 'closed',
                                      'tpc', 'backfill_fifo', ?, ?)
                            """,
                            (
                                pid,
                                sym,
                                entry_ts,
                                exit_ts,
                                entry_px,
                                px,
                                match_qty,
                                entry_px * match_qty,
                                rpnl,
                                now,
                                now,
                            ),
                        )
                        stats["inserted"] += 1
                    else:
                        conn.execute(
                            """
                            UPDATE positions SET
                                exit_time = COALESCE(exit_time, ?),
                                exit_price = COALESCE(NULLIF(exit_price, 0), ?),
                                realized_pnl = ?,
                                current_size = 0,
                                status = 'closed',
                                updated_at = ?
                            WHERE position_id = ?
                            """,
                            (exit_ts, px, rpnl, now, pid),
                        )
                        stats["updated"] += 1
        conn.commit()
    conn.close()
    stats["fifo_sell_pnl"] = fifo_map
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/order_management.db"),
        help="Trend order_management.db path",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default: dry-run only)",
    )
    args = parser.parse_args()
    if not args.db.is_file():
        raise SystemExit(f"DB not found: {args.db}")

    stats = backfill(args.db, dry_run=not args.apply)
    mode = "DRY-RUN" if not args.apply else "APPLIED"
    print(
        f"[{mode}] scanned={stats['position_ids_scanned']} "
        f"inserted={stats['inserted']} updated={stats['updated']} "
        f"open_no_exit={stats['skipped_open']} no_entry={stats['skipped_no_pair']} "
        f"fifo_sells={stats.get('fifo_pairs', 0)}"
    )
    for row in stats["details"]:
        print(
            f"  {row['symbol']} {row['position_id']}: "
            f"{row['entry_price']:.4f} -> {row['exit_price']:.4f} "
            f"qty={row['qty']:.6f} pnl={row['realized_pnl']:+.4f} USDT"
        )
    fifo = stats.get("fifo_sell_pnl") or {}
    for oid, rec in sorted(fifo.items()):
        if not str(oid).startswith("order_"):
            continue
        print(f"  fifo {oid}: pnl={rec.get('pnl_usdt'):+.4f} USDT")


if __name__ == "__main__":
    main()
