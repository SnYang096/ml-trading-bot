"""Ground-truth open legs from ``multi_leg_positions`` (live hedge reconcile)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Set

from mlbot_console.services.db import query_rows
from mlbot_console.services.symbols import is_all_symbols


def multileg_open_leg_ids(db_path: Path, symbol: Optional[str]) -> Set[str]:
    """Leg ids with status=open in multi_leg_positions."""
    if not db_path.is_file():
        return set()
    where = "WHERE lower(trim(coalesce(status, ''))) = 'open'"
    params: tuple[Any, ...] = ()
    if symbol and not is_all_symbols(symbol):
        where += " AND symbol = ?"
        params = (symbol.upper(),)
    rows = query_rows(
        db_path,
        f"""
        SELECT leg_id
        FROM multi_leg_positions
        {where}
        """,
        params,
    )
    return {
        str(row.get("leg_id") or "").strip()
        for row in rows
        if str(row.get("leg_id") or "").strip()
    }


def multileg_positions_table_used(db_path: Path, symbol: Optional[str]) -> bool:
    """True when multi_leg_positions has rows (live hedge persists inventory here)."""
    if not db_path.is_file():
        return False
    where = ""
    params: tuple[Any, ...] = ()
    if symbol and not is_all_symbols(symbol):
        where = "WHERE symbol = ?"
        params = (symbol.upper(),)
    rows = query_rows(
        db_path,
        f"SELECT 1 FROM multi_leg_positions {where} LIMIT 1",
        params,
    )
    return bool(rows)
