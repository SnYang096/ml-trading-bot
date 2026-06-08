#!/usr/bin/env python3
"""One-off: backfill multi_leg_orders.filled_quantity from persisted raw_json.

Trend scalp market fills were often zeroed when user-stream events carried z=0
after the place response had already stored the real fill in raw_json. Console
hydrate fixes display; this script repairs the DB column for analytics/history.

Default is dry-run; pass --apply to write.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _parse_raw_json_blob(raw_json: Any) -> Dict[str, Any]:
    if isinstance(raw_json, dict):
        return raw_json
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_fill_fields_from_raw(
    raw_json: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Mirror console hydrate: read cumulative fill/qty and avg price from raw_json."""
    raw = _parse_raw_json_blob(raw_json)
    if not raw:
        return None, None
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}

    filled: Optional[float] = None
    for key in ("filled", "filled_quantity", "executedQty"):
        val = raw.get(key) if key != "executedQty" else info.get("executedQty")
        if val is None:
            continue
        try:
            qty = float(val)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            filled = qty
            break

    avg_px: Optional[float] = None
    for key in ("average_price", "price", "avgPrice"):
        val = raw.get(key) if key != "avgPrice" else info.get("avgPrice")
        if val is None:
            continue
        try:
            px = float(val)
        except (TypeError, ValueError):
            continue
        if px > 0:
            avg_px = px
            break
    return filled, avg_px


def backfill(
    db_path: Path,
    *,
    strategy: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE (filled_quantity IS NULL OR filled_quantity <= 0) AND raw_json IS NOT NULL"
        params: list[Any] = []
        if strategy:
            where += " AND strategy = ?"
            params.append(str(strategy))
        rows = conn.execute(
            f"""
            SELECT local_order_id, strategy, symbol, filled_quantity, average_price, raw_json
            FROM multi_leg_orders
            {where}
            """,
            tuple(params),
        ).fetchall()
        scanned = len(rows)
        updated = 0
        skipped = 0
        for row in rows:
            filled, avg_px = extract_fill_fields_from_raw(row["raw_json"])
            if filled is None and avg_px is None:
                skipped += 1
                continue
            if not dry_run:
                conn.execute(
                    """
                    UPDATE multi_leg_orders
                    SET filled_quantity = COALESCE(?, filled_quantity),
                        average_price = COALESCE(?, average_price),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE local_order_id = ?
                    """,
                    (filled, avg_px, row["local_order_id"]),
                )
            updated += 1
        if not dry_run:
            conn.commit()
        return {
            "scanned": scanned,
            "updated": updated,
            "skipped_no_raw_fill": skipped,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/multi_leg_order_management.db"),
        help="multi_leg_order_management.db path",
    )
    parser.add_argument(
        "--strategy",
        default="",
        help="Optional filter, e.g. trend_scalp or chop_grid",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default: dry-run only)",
    )
    args = parser.parse_args()
    if not args.db.is_file():
        raise SystemExit(f"DB not found: {args.db}")

    stats = backfill(
        args.db,
        strategy=str(args.strategy).strip() or None,
        dry_run=not args.apply,
    )
    mode = "DRY-RUN" if not args.apply else "APPLIED"
    strat = f" strategy={args.strategy}" if args.strategy else ""
    print(
        f"[{mode}]{strat} scanned={stats['scanned']} "
        f"would_update={stats['updated']} skipped_no_raw_fill={stats['skipped_no_raw_fill']}"
    )
    if not args.apply and stats["updated"]:
        print("Pass --apply to write filled_quantity / average_price from raw_json.")


if __name__ == "__main__":
    main()
