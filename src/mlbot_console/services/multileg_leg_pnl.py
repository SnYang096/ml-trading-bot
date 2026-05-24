"""Per-order realized / unrealized PnL for chop_grid multi-leg rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services.account_summary import _link_pnl_usdt
from mlbot_console.services.db import query_rows
from mlbot_console.services.multileg_order_links import (
    _is_filled_row,
    _pick_filled_tp,
    _pick_planned_tp,
    _price,
    _protection_tp_rows,
    build_leg_link_index,
    entry_link_id,
    is_entry_row,
)
from mlbot_console.services.multileg_repair_tp import pick_repair_filled_tp

_MULTILEG_ORDER_SQL = """
    SELECT local_order_id AS order_id, local_order_id, symbol, side, status, purpose,
           order_type, quantity, filled_quantity, average_price, price, strategy, leg_id,
           client_order_id, filled_at, created_at
    FROM multi_leg_orders
    WHERE symbol = ?
"""


def _order_key(row: Dict[str, Any]) -> str:
    return str(row.get("order_id") or row.get("local_order_id") or "")


def _display_row_as_raw(row: Dict[str, Any]) -> Dict[str, Any]:
    oid = _order_key(row)
    return {
        "order_id": oid,
        "local_order_id": oid,
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "status": row.get("status"),
        "purpose": row.get("purpose") or row.get("order_type"),
        "order_type": row.get("order_type"),
        "quantity": row.get("quantity"),
        "filled_quantity": row.get("filled_quantity"),
        "average_price": row.get("average_price"),
        "price": row.get("price"),
        "strategy": row.get("strategy"),
        "leg_id": row.get("leg_id"),
        "client_order_id": row.get("client_order_id"),
        "filled_at": row.get("filled_at"),
        "created_at": row.get("created_at"),
    }


def _unrealized_pnl_usdt(
    entry_row: Dict[str, Any], mark_px: float
) -> Optional[float]:
    qty = float(entry_row.get("filled_quantity") or entry_row.get("quantity") or 0.0)
    if qty <= 0 or mark_px <= 0:
        return None
    entry_px = float(entry_row.get("average_price") or entry_row.get("price") or 0.0)
    if entry_px <= 0:
        return None
    side = str(entry_row.get("side") or "").lower()
    if side in {"buy", "long"}:
        return (mark_px - entry_px) * qty
    return (entry_px - mark_px) * qty


def _pnl_rec(
    *,
    pnl: float,
    hint: str,
    unrealized: bool = False,
) -> Dict[str, Any]:
    if unrealized:
        return {
            "pnl_usdt": pnl,
            "unrealized_pnl": pnl,
            "realized_pnl": None,
            "pnl_hint": hint,
        }
    return {
        "pnl_usdt": pnl,
        "realized_pnl": pnl,
        "unrealized_pnl": None,
        "pnl_hint": hint,
    }


def _filled_exit_row(
    group_rows: List[Dict[str, Any]], entry_row: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    eid = entry_link_id(entry_row)
    oid = _order_key(entry_row)
    tp_rows = _protection_tp_rows(group_rows, eid)
    if not tp_rows and oid and oid != eid:
        tp_rows = _protection_tp_rows(group_rows, oid)
    exit_row = _pick_filled_tp(tp_rows)
    if exit_row is None:
        exit_row = pick_repair_filled_tp(group_rows, eid)
    return exit_row


def multileg_pnl_by_order_id(
    db_path: Path,
    symbol: str,
    *,
    extra_rows: Optional[List[Dict[str, Any]]] = None,
    mark_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map local_order_id -> pnl fields for filled entry/exit legs (L and S)."""
    if not db_path.is_file():
        return {}
    sym = str(symbol).upper()
    if sym in {"", "*", "ALL", "__ALL__"}:
        return {}

    raw: List[Dict[str, Any]] = list(query_rows(db_path, _MULTILEG_ORDER_SQL, (sym,)))
    known = {_order_key(r) for r in raw if _order_key(r)}
    for row in extra_rows or []:
        key = _order_key(row)
        if key and key not in known:
            raw.append(_display_row_as_raw(row))
            known.add(key)

    by_group = build_leg_link_index(raw)
    mark = float((mark_prices or {}).get(sym) or 0.0)
    out: Dict[str, Dict[str, Any]] = {}

    for group_rows in by_group.values():
        for entry in group_rows:
            if not is_entry_row(entry) or not _is_filled_row(entry):
                continue
            entry_key = _order_key(entry)
            if not entry_key:
                continue
            exit_row = _filled_exit_row(group_rows, entry)
            if exit_row is not None:
                pnl = _link_pnl_usdt(entry, exit_row)
                if pnl is None:
                    continue
                rec = _pnl_rec(pnl=pnl, hint="已实现")
                out[entry_key] = rec
                exit_key = _order_key(exit_row)
                if exit_key:
                    out[exit_key] = dict(rec)
                continue
            if mark <= 0:
                continue
            upnl = _unrealized_pnl_usdt(entry, mark)
            if upnl is None:
                continue
            rec = _pnl_rec(pnl=upnl, hint="浮盈", unrealized=True)
            out[entry_key] = rec

    return out


def attach_multileg_display_pnl(
    rows: List[Dict[str, Any]],
    *,
    db_path: Path,
    symbol: str,
    mark_prices: Optional[Dict[str, float]] = None,
) -> None:
    """Fill pnl_* on multi_leg display rows (includes synthetic inventory legs)."""
    ml_display = [r for r in rows if str(r.get("scope") or "") == "multi_leg"]
    if not ml_display:
        return
    sym = str(symbol).upper()
    if sym in {"", "*", "ALL", "__ALL__"}:
        sym = str(ml_display[0].get("symbol") or "").upper()
    if not sym:
        return
    pnl_map = multileg_pnl_by_order_id(
        db_path, sym, extra_rows=ml_display, mark_prices=mark_prices
    )
    for row in ml_display:
        if row.get("pnl_usdt") is not None:
            continue
        rec = pnl_map.get(_order_key(row))
        if not rec:
            continue
        for key in ("pnl_usdt", "realized_pnl", "unrealized_pnl", "pnl_hint"):
            if rec.get(key) is not None:
                row[key] = rec[key]
