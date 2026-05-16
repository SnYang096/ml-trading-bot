#!/usr/bin/env python3
"""Move legacy multi-leg tables out of classical order_management.db.

Older builds applied one monolithic schema to whichever SQLite path opened, so a
PCM ``order_management.db`` could contain ``multi_leg_*`` tables and an optional
standalone multi-leg DB could contain classical tables.

This script copies ``multi_leg_*`` rows from the classic DB into the multi-leg DB
(and ensures that DB has schema via ``MultiLegStorage``), then DROPs those
tables from the classic file.

Always back up databases first."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MULTI_LEG_TABLES_ORDERED = (
    "multi_leg_runs",
    "multi_leg_orders",
    "multi_leg_positions",
    "multi_leg_execution_reports",
    "multi_leg_reconciliation_snapshots",
)

TREND_TABLES_ORDERED = (
    "positions",
    "position_operations",
    "orders",
    "stop_loss_trailing",
    "performance_metrics",
    "safety_state",
    "slots_state",
    "add_position_state",
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _columns(
    conn: sqlite3.Connection, table: str, *, schema: str = "main"
) -> list[str]:
    rows = conn.execute(f'PRAGMA {schema}.table_info("{table}")').fetchall()
    return [str(row[1]) for row in rows]


def _backup(path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    dst = Path(str(path) + f".bak_{ts}")
    shutil.copy2(path, dst)
    return dst


def prune_classic_tables_from_multi_leg_db(
    *,
    multi_leg: Path,
    dry_run: bool,
) -> int:
    """Drop positions/orders/... from file if old builds created them."""
    if not multi_leg.is_file():
        print(f"multi-leg db missing: {multi_leg}")
        return 2

    conn = sqlite3.connect(str(multi_leg))
    try:
        tables = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            if r[0]
        ]
        drop_candidates = [t for t in TREND_TABLES_ORDERED if t in tables]
        if not drop_candidates:
            print("multi-leg DB has no stray classical tables.")
            return 0
        print(f"Classic tables seen in multi-leg DB ({multi_leg}): {drop_candidates}")
        if dry_run:
            print("Dry run; would DROP these tables.")
            return 0
        _backup(multi_leg)
        conn.execute("PRAGMA foreign_keys=OFF")
        for t in sorted(drop_candidates, reverse=True):
            conn.execute(f'DROP TABLE IF EXISTS "{t}"')
            conn.commit()
            print(f"  dropped {t}")
        print("Cleanup complete.")
        return 0
    finally:
        conn.close()


def _non_multi_tables(conn: sqlite3.Connection) -> list[str]:
    out: list[str] = []
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        if name.startswith("sqlite_"):
            continue
        if not str(name).startswith("multi_leg_"):
            out.append(str(name))
    return out


def migrate(*, classic: Path, multi_leg: Path, dry_run: bool) -> int:
    if not classic.is_file():
        print(f"classic db missing: {classic}")
        return 2

    multi_leg.parent.mkdir(parents=True, exist_ok=True)

    conn_c = sqlite3.connect(str(classic))
    try:
        present_ml = [t for t in MULTI_LEG_TABLES_ORDERED if _table_exists(conn_c, t)]
        if not present_ml:
            print("No multi-leg tables in classic DB; nothing to migrate.")
            return 0

        counts = {t: _count(conn_c, t) for t in present_ml}

        extras_on_ml_side: list[str] = []
        if multi_leg.is_file():
            conn_m = sqlite3.connect(str(multi_leg))
            try:
                extras_on_ml_side = _non_multi_tables(conn_m)
            finally:
                conn_m.close()

        print(f"Classic: {classic.resolve()}")
        print(f"Multi-leg target: {multi_leg.resolve()} (exists={multi_leg.is_file()})")
        print(f"Tables to move: {', '.join(present_ml)}")
        print(f"Row counts (classic): {counts}")
        if extras_on_ml_side:
            print(
                "Note: multi-leg DB already has non-multi_leg tables "
                f"({extras_on_ml_side}); INSERT OR REPLACE may touch those too if names clash."
            )

        if dry_run:
            print("Dry run; no changes.")
            return 0

        bak_c = _backup(classic)
        print(f"Backed up classic → {bak_c}")
        if multi_leg.is_file():
            bak_m = _backup(multi_leg)
            print(f"Backed up multi-leg → {bak_m}")

        # Ensure multi-leg DDL only on target file (no classic tables beyond any legacy bleed).
        from src.order_management.multi_leg_storage import MultiLegStorage

        MultiLegStorage(str(multi_leg))

        conn_c.execute("ATTACH DATABASE ? AS ml", (str(multi_leg),))
        try:
            conn_c.execute("PRAGMA foreign_keys=OFF")
            for tbl in present_ml:
                try:
                    before_dst = int(
                        conn_c.execute(f'SELECT COUNT(*) FROM ml."{tbl}"').fetchone()[0]
                    )
                except sqlite3.DatabaseError:
                    before_dst = 0
                print(f'  merging ml."{tbl}": rows in ml before={before_dst}')
                src_cols = _columns(conn_c, tbl)
                dst_cols = _columns(conn_c, tbl, schema="ml")
                common_cols = [c for c in dst_cols if c in src_cols]
                if not common_cols:
                    print(f'  skipped ml."{tbl}": no shared columns')
                    continue
                quoted = ", ".join(f'"{c}"' for c in common_cols)
                conn_c.execute(
                    f'INSERT OR REPLACE INTO ml."{tbl}" ({quoted}) '
                    f'SELECT {quoted} FROM main."{tbl}"'
                )
                conn_c.commit()
                after_ml = int(
                    conn_c.execute(f'SELECT COUNT(*) FROM ml."{tbl}"').fetchone()[0]
                )
                print(f'  merged ml."{tbl}": rows after merge={after_ml}')
        finally:
            conn_c.execute("DETACH DATABASE ml")
            conn_c.commit()

    finally:
        conn_c.close()

    conn_drop = sqlite3.connect(str(classic))
    try:
        conn_drop.execute("PRAGMA foreign_keys=OFF")
        # Drop dependents first — reverse FK graph (runs last).
        drop_order = list(reversed(list(MULTI_LEG_TABLES_ORDERED)))
        for tbl in drop_order:
            if _table_exists(conn_drop, tbl):
                conn_drop.execute(f'DROP TABLE IF EXISTS "{tbl}"')
                conn_drop.commit()
                print(f"  dropped classic table {tbl}")
    finally:
        conn_drop.close()

    print("Migration complete.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strip-classic-from-multi-leg-db",
        action="store_true",
        help=(
            "Only remove positions/orders/... from multi-leg DB "
            "(legacy bleed). Use after backing up."
        ),
    )
    p.add_argument(
        "--classic-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "order_management.db",
        help="Trend / PCM classical SQLite path",
    )
    p.add_argument(
        "--multi-leg-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "multi_leg_order_management.db",
        help="Dedicated multi-leg SQLite path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only",
    )
    args = p.parse_args()
    ml_path = args.multi_leg_db.resolve()

    if args.strip_classic_from_multi_leg_db:
        code = prune_classic_tables_from_multi_leg_db(
            multi_leg=ml_path,
            dry_run=bool(args.dry_run),
        )
        raise SystemExit(code)

    code = migrate(
        classic=args.classic_db.resolve(),
        multi_leg=ml_path,
        dry_run=bool(args.dry_run),
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
