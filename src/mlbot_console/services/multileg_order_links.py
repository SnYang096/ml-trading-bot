"""Pair chop_grid / trend_scalp L/S leg orders for console take-profit display."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from mlbot_console.services.trade_markers import _OPEN_ORDER_STATUSES, _multi_leg_take_profit_price

_LEG_SUFFIX_RE = re.compile(r"_(L|S)(\d+)$", re.I)
_TP_SUFFIX_RE = re.compile(r"_(L|S)(\d+)_tp$", re.I)


def leg_group_key(order_id: str) -> Optional[str]:
    oid = str(order_id or "")
    m = _LEG_SUFFIX_RE.search(oid)
    if not m:
        m = _TP_SUFFIX_RE.search(oid)
        if m:
            return oid[: m.start()]
        return None
    return oid[: m.start()]


def leg_side_kind(order_id: str) -> Optional[str]:
    m = _LEG_SUFFIX_RE.search(str(order_id or ""))
    if not m:
        return None
    return m.group(1).upper()


def leg_suffix(order_id: str) -> str:
    m = _LEG_SUFFIX_RE.search(str(order_id or ""))
    if not m:
        m = _TP_SUFFIX_RE.search(str(order_id or ""))
        if not m:
            return ""
        return f"{m.group(1)}{m.group(2)}_tp"
    return f"{m.group(1)}{m.group(2)}"


def leg_index(order_id: str) -> int:
    m = _LEG_SUFFIX_RE.search(str(order_id or ""))
    if not m:
        m = _TP_SUFFIX_RE.search(str(order_id or ""))
    if not m:
        return 0
    try:
        return int(m.group(2))
    except ValueError:
        return 0


def _price(row: Dict[str, Any]) -> Optional[float]:
    for key in ("average_price", "price", "stop_price"):
        val = row.get(key)
        if val is not None and val == val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _is_filled_row(row: Dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    qty = float(row.get("filled_quantity") or 0)
    return status in {"filled", "closed"} or qty > 0


def _entry_leg_id(row: Dict[str, Any]) -> str:
    oid = str(row.get("order_id") or "")
    leg = str(row.get("leg_id") or "")
    if leg and not leg.endswith("_tp"):
        return leg
    m = _TP_SUFFIX_RE.search(oid)
    if m:
        return oid[: m.start()] + f"_{m.group(1)}{m.group(2)}"
    return oid


def _protection_tp_rows(
    legs: List[Dict[str, Any]], entry_order_id: str
) -> List[Dict[str, Any]]:
    eid = str(entry_order_id or "")
    out: List[Dict[str, Any]] = []
    for row in legs:
        oid = str(row.get("order_id") or "")
        purpose = str(row.get("purpose") or "").lower()
        leg = str(row.get("leg_id") or "")
        if oid.startswith(f"{eid}_tp") or (
            "take_profit" in purpose and leg == eid
        ):
            out.append(row)
    return out


def _pick_planned_tp(tp_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = sorted(tp_rows, key=leg_index)
    for row in ordered:
        st = str(row.get("status") or "").lower()
        if st in _OPEN_ORDER_STATUSES:
            return row
    for row in ordered:
        if _price(row) is not None:
            return row
    return None


def _pick_filled_tp(tp_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in tp_rows:
        if _is_filled_row(row):
            return row
    return None


def annotate_leg_group(legs: List[Dict[str, Any]]) -> None:
    """Mutate raw multi_leg_orders rows with _link_* fields for entry legs."""
    l_legs = [r for r in legs if is_l_entry_row(r)]
    s_legs = [
        r
        for r in legs
        if leg_side_kind(str(r.get("order_id") or "")) == "S"
        or leg_side_kind(str(r.get("leg_id") or "")) == "S"
    ]
    if not l_legs and not s_legs:
        return

    for row in l_legs + s_legs:
        eid = str(row.get("order_id") or "")
        tp_rows = _protection_tp_rows(legs, eid)
        planned = _pick_planned_tp(tp_rows)
        exit_row = _pick_filled_tp(tp_rows)
        if planned is not None:
            row["_link_tp_price"] = _price(planned)
            row["_link_tp_leg"] = leg_suffix(str(planned.get("order_id") or ""))
            row["_link_tp_status"] = str(planned.get("status") or "")
        if exit_row is not None:
            row["_link_exit_price"] = _price(exit_row)
            row["_link_exit_leg"] = str(exit_row.get("order_id") or "")
            row["_link_exit_status"] = str(exit_row.get("status") or "")


def row_group_key(row: Dict[str, Any]) -> Optional[str]:
    """Group key for chop_grid legs (cg_* ids use leg_id for L1/L2 grouping)."""
    for field in ("order_id", "local_order_id", "leg_id"):
        gk = leg_group_key(str(row.get(field) or ""))
        if gk:
            return gk
    lid = str(row.get("leg_id") or "")
    m = _LEG_SUFFIX_RE.search(lid)
    if m:
        return lid[: m.start()]
    return None


def is_l_entry_row(row: Dict[str, Any]) -> bool:
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" in purpose or "market_exit" in purpose:
        return False
    for field in ("order_id", "local_order_id", "leg_id"):
        if leg_side_kind(str(row.get(field) or "")) == "L":
            return True
    return False


def entry_link_id(row: Dict[str, Any]) -> str:
    lid = str(row.get("leg_id") or "").strip()
    if lid:
        return lid
    return str(row.get("local_order_id") or row.get("order_id") or "")


def build_leg_link_index(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        gk = row_group_key(row)
        if not gk:
            continue
        by_group.setdefault(gk, []).append(row)
    for group in by_group.values():
        annotate_leg_group(group)
    return by_group


def resolve_take_profit_display(row: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Return (price, hint) for UI: *_tp protection orders, not grid S entry legs.
    """
    exit_px = row.get("_link_exit_price")
    if exit_px is not None and exit_px == exit_px:
        hint = "已平仓"
        leg = row.get("_link_exit_leg")
        if leg:
            hint = f"已平·{str(leg)[-6:]}"
        return float(exit_px), hint

    direct = _multi_leg_take_profit_price(row)
    if direct is not None:
        return direct, ""

    link_px = row.get("_link_tp_price")
    if link_px is not None and link_px == link_px:
        hint = "挂单"
        st = str(row.get("_link_tp_status") or "").lower()
        leg = row.get("_link_tp_leg") or ""
        if leg and st:
            hint = f"{leg}·{st}"
        elif leg:
            hint = str(leg)
        elif st:
            hint = st
        return float(link_px), hint

    return None, ""


def enrich_multileg_rows_for_symbol(
    db_path,
    symbol: str,
    rows: List[Dict[str, Any]],
) -> None:
    """Load all legs for symbol so L rows get TP even when S legs are filtered out."""
    from mlbot_console.services.db import query_rows

    sym = str(symbol).upper()
    if sym in {"", "*", "ALL", "__ALL__"}:
        build_leg_link_index(rows)
        return
    sql = """
        SELECT local_order_id AS order_id, symbol, side, status, order_type, purpose,
               quantity, price, stop_price, filled_quantity, average_price, created_at,
               filled_at, strategy, leg_id
        FROM multi_leg_orders
        WHERE symbol = ?
    """
    all_rows = query_rows(db_path, sql, (sym,))
    build_leg_link_index(all_rows)
    by_oid = {str(r.get("order_id") or ""): r for r in all_rows}
    for item in rows:
        oid = str(item.get("order_id") or "")
        src = by_oid.get(oid)
        if not src:
            continue
        for key in (
            "_link_tp_price",
            "_link_tp_leg",
            "_link_tp_status",
            "_link_exit_price",
            "_link_exit_leg",
            "_link_exit_status",
        ):
            if key in src:
                item[key] = src[key]
