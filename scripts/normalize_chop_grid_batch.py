#!/usr/bin/env python3
"""Normalize a chop_grid batch to the canonical 12-order model.

Usage:
  python scripts/normalize_chop_grid_batch.py --symbol BNBUSDT --grid-id BNBUSDT_2026-05-19_08:40:00+00:00
"""

import argparse
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_batch(db_path: Path, symbol: str, grid_id: str, dry_run: bool = True):
    if not db_path.exists():
        logger.error(f"DB not found: {db_path}")
        return

    con = sqlite3.connect(str(db_path))

    # 1. Cancel _supp orders
    supp_like = f"{grid_id}%_supp"
    logger.info(f"Finding _supp orders like {supp_like}")
    supp_rows = con.execute(
        "SELECT local_order_id, status FROM multi_leg_orders WHERE symbol=? AND local_order_id LIKE ?",
        (symbol, supp_like),
    ).fetchall()

    for row in supp_rows:
        logger.info(f"Found supp order: {row[0]} (status: {row[1]})")
        if not dry_run:
            con.execute(
                "UPDATE multi_leg_orders SET status='canceled' WHERE local_order_id=?",
                (row[0],),
            )
            logger.info(f"  -> Marked {row[0]} as canceled")

    # 2. Reopen L2 if it was expired
    l2_id = f"{grid_id}_L2"
    l2_row = con.execute(
        "SELECT local_order_id, status FROM multi_leg_orders WHERE symbol=? AND local_order_id=?",
        (symbol, l2_id),
    ).fetchone()

    if l2_row:
        logger.info(f"Found L2 order: {l2_row[0]} (status: {l2_row[1]})")
        if l2_row[1] == "expired":
            if not dry_run:
                con.execute(
                    "UPDATE multi_leg_orders SET status='open' WHERE local_order_id=?",
                    (l2_id,),
                )
                logger.info(f"  -> Marked {l2_id} as open")
    else:
        logger.info(f"L2 order not found: {l2_id}")

    if not dry_run:
        con.commit()
    con.close()
    logger.info("Done.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--grid-id", required=True)
    p.add_argument("--db", default="data/multi_leg_order_management.db")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    normalize_batch(
        Path(args.db), args.symbol.upper(), args.grid_id, dry_run=not args.execute
    )


if __name__ == "__main__":
    main()
