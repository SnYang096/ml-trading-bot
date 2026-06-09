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
    filled_quantity,
    hydrate_multileg_fill_fields,
    is_entry_row,
    is_l_entry_row,
    is_s_entry_row,
    is_trend_entry_row,
    leg_group_key,
    row_group_key,
    late_fixup_entry_segment_matches,
    market_exit_closing_position_side,
    trend_entry_position_side,
    trend_exit_entry_id,
)
from mlbot_console.services.multileg_repair_tp import pick_repair_filled_tp

_MULTILEG_ORDER_SQL = """
    SELECT local_order_id AS order_id, local_order_id, symbol, side, position_side, status,
           purpose, order_type, quantity, filled_quantity, average_price, price, strategy,
           leg_id, client_order_id, filled_at, created_at, raw_json
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


def _entry_position_side(row: Dict[str, Any]) -> Optional[str]:
    if is_l_entry_row(row):
        return "LONG"
    if is_s_entry_row(row):
        return "SHORT"
    return trend_entry_position_side(row)


def _ts_row(row: Dict[str, Any]) -> Optional[int]:
    from mlbot_console.services.trade_markers import _parse_ts

    ts = _parse_ts(row.get("filled_at")) or _parse_ts(row.get("created_at"))
    return int(ts) if ts is not None else None


def _trend_exit_for_entry(
    group_rows: List[Dict[str, Any]], entry_row: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not is_trend_entry_row(entry_row):
        return None
    entry_id = entry_link_id(entry_row)
    if not entry_id:
        return None
    entry_ts = _ts_row(entry_row)
    best: Optional[Dict[str, Any]] = None
    best_ts = -1
    for row in group_rows:
        purpose = str(row.get("purpose") or "").lower()
        if "market_exit" not in purpose or not _is_filled_row(row):
            continue
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        if trend_exit_entry_id(oid) != entry_id:
            continue
        if _price(row) is None:
            continue
        exit_ts = _ts_row(row) or 0
        if entry_ts is not None and exit_ts < entry_ts:
            continue
        if exit_ts >= best_ts:
            best = row
            best_ts = exit_ts
    return best


def _orphan_market_exit_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        purpose = str(row.get("purpose") or "").lower()
        if "market_exit" not in purpose or not _is_filled_row(row):
            continue
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        if leg_group_key(oid):
            continue
        if trend_exit_entry_id(oid):
            continue
        if _price(row) is None:
            continue
        out.append(row)
    out.sort(key=lambda r: (_ts_row(r) or 0, _order_key(r)))
    return out


def _filled_exit_row(
    group_rows: List[Dict[str, Any]],
    entry_row: Dict[str, Any],
    *,
    orphan_market_exits: Optional[List[Dict[str, Any]]] = None,
    used_market_exit_ids: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    eid = entry_link_id(entry_row)
    oid = _order_key(entry_row)
    tp_rows = _protection_tp_rows(group_rows, eid)
    if not tp_rows and oid and oid != eid:
        tp_rows = _protection_tp_rows(group_rows, oid)
    exit_row = _pick_filled_tp(tp_rows)
    if exit_row is None:
        exit_row = pick_repair_filled_tp(group_rows, eid)
    if exit_row is not None:
        return exit_row

    exit_row = _trend_exit_for_entry(group_rows, entry_row)
    if exit_row is not None:
        return exit_row

    entry_ts = _ts_row(entry_row)
    if entry_ts is None:
        return None
    ent_side = _entry_position_side(entry_row)
    ent_qty = filled_quantity(entry_row)
    if ent_side is None or ent_qty <= 0:
        return None
    used = used_market_exit_ids if used_market_exit_ids is not None else set()
    for mex in orphan_market_exits or []:
        mex_id = _order_key(mex)
        if not mex_id or mex_id in used:
            continue
        exit_ts = _ts_row(mex)
        if exit_ts is None or exit_ts < entry_ts:
            continue
        if market_exit_closing_position_side(mex) != ent_side:
            continue
        if not late_fixup_entry_segment_matches(mex_id, _order_key(entry_row)):
            continue
        mex_qty = filled_quantity(mex)
        if mex_qty <= 0 or ent_qty > mex_qty * 1.02:
            continue
        used.add(mex_id)
        return mex
    return None


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
    for row in raw:
        hydrate_multileg_fill_fields(row)
    known = {_order_key(r) for r in raw if _order_key(r)}
    for row in extra_rows or []:
        key = _order_key(row)
        if key and key not in known:
            raw.append(_display_row_as_raw(row))
            known.add(key)

    by_group = build_leg_link_index(raw)
    orphan_exits = _orphan_market_exit_rows(raw)
    used_market_exit_ids: set[str] = set()
    mark = float((mark_prices or {}).get(sym) or 0.0)
    out: Dict[str, Dict[str, Any]] = {}

    pending_entries: List[Dict[str, Any]] = []
    for group_rows in by_group.values():
        for entry in group_rows:
            if is_entry_row(entry) and _is_filled_row(entry) and _order_key(entry):
                pending_entries.append(entry)
    pending_entries.sort(key=lambda r: (_ts_row(r) or 0, _order_key(r)))

    for entry in pending_entries:
        entry_key = _order_key(entry)
        gk = row_group_key(entry)
        group_rows = by_group.get(gk or "", [entry])
        if not entry_key:
            continue
        exit_row = _filled_exit_row(
            group_rows,
            entry,
            orphan_market_exits=orphan_exits,
            used_market_exit_ids=used_market_exit_ids,
        )
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
