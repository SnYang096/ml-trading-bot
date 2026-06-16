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


def multileg_closed_leg_ids(db_path: Path, symbol: Optional[str]) -> Set[str]:
    """Leg ids with status=closed in multi_leg_positions."""
    if not db_path.is_file():
        return set()
    where = "WHERE lower(trim(coalesce(status, ''))) = 'closed'"
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


def leg_key_matches_open_position_legs(leg_key: str, active_leg_ids: Set[str]) -> bool:
    """True when *leg_key* corresponds to an open ``multi_leg_positions`` row.

    Matches exact leg_id, or trend_scalp inventory suffixes where the position
    row uses ``{order_leg_id}_fill{N}`` while ``multi_leg_orders`` keeps the
    bare entry leg_id.
    """
    key = str(leg_key or "").strip()
    if not key or not active_leg_ids:
        return False
    if key in active_leg_ids:
        return True
    fill_prefix = key + "_fill"
    return any(str(al).startswith(fill_prefix) for al in active_leg_ids)


def leg_key_is_pruned_ghost(
    leg_key: str,
    *,
    positions_table_used: bool,
    active_leg_ids: Set[str],
    closed_leg_ids: Set[str],
) -> bool:
    """True when ``multi_leg_positions`` ground truth says hide this entry leg.

    * Still open (exact or trend_scalp ``_fill{N}`` suffix) → not a ghost.
    * Explicitly closed in positions table → ghost.
    * Other legs still open but this leg is not among them → ghost.
    * Positions table in use, no open legs globally, no closed row for this leg
      → not a ghost (order-based fallback after restart / missing row).
    """
    if not positions_table_used:
        return False
    if leg_key_matches_open_position_legs(leg_key, active_leg_ids):
        return False
    key = str(leg_key or "").strip()
    if not key:
        return False
    if key in closed_leg_ids:
        return True
    fill_prefix = key + "_fill"
    if any(str(cl).startswith(fill_prefix) for cl in closed_leg_ids):
        return True
    if active_leg_ids:
        return True
    return False
