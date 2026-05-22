"""Recognize manual chop_grid repair TP orders (cg_repair_* client ids)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mlbot_console.services.multileg_order_links import (
    _LEG_SUFFIX_RE,
    _is_filled_row,
    entry_link_id,
    is_l_entry_row,
    is_s_entry_row,
    leg_group_key,
    leg_side_kind,
    row_group_key,
)

_REPAIR_TP_RE = re.compile(r"^cg_repair_(long|short)_tp(\d+)$", re.I)


def repair_client_order_id(row: Dict[str, Any]) -> str:
    for field in ("client_order_id", "order_id", "local_order_id"):
        val = str(row.get(field) or "").strip()
        if _REPAIR_TP_RE.match(val):
            return val
    return ""


def is_repair_tp_row(row: Dict[str, Any]) -> bool:
    if repair_client_order_id(row):
        return True
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" not in purpose:
        return False
    return bool(str(row.get("_repair_tp") or ""))


def repair_position_side(row: Dict[str, Any]) -> Optional[str]:
    cid = repair_client_order_id(row)
    if cid:
        m = _REPAIR_TP_RE.match(cid)
        if m:
            return "LONG" if m.group(1).lower() == "long" else "SHORT"
    side = str(row.get("side") or row.get("position_side") or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    return None


def repair_display_leg_label(row: Dict[str, Any]) -> str:
    """Canonical leg badge e.g. L1_tp for repair / backfilled rows."""
    leg = str(row.get("leg_id") or "").strip()
    if leg:
        m = _LEG_SUFFIX_RE.search(leg)
        if m:
            return f"{m.group(1)}{m.group(2)}_tp"
    for field in ("order_id", "local_order_id"):
        oid = str(row.get(field) or "")
        m = _LEG_SUFFIX_RE.search(oid)
        if m and oid.endswith("_tp"):
            return f"{m.group(1)}{m.group(2)}_tp"
    pos = repair_position_side(row)
    if pos == "LONG":
        return "L1_tp"
    if pos == "SHORT":
        return "S1_tp"
    return "tp"


def _entry_side_kind(row: Dict[str, Any]) -> Optional[str]:
    for field in ("order_id", "local_order_id", "leg_id"):
        kind = leg_side_kind(str(row.get(field) or ""))
        if kind:
            return kind
    return None


def _qty(row: Dict[str, Any]) -> float:
    for key in ("filled_quantity", "quantity"):
        try:
            val = float(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return 0.0


def repair_tp_matches_entry(
    repair_row: Dict[str, Any],
    entry_order_id: str,
    legs: List[Dict[str, Any]],
) -> bool:
    if not is_repair_tp_row(repair_row):
        return False
    eid = str(entry_order_id or "").strip()
    if not eid:
        return False
    leg = str(repair_row.get("leg_id") or "").strip()
    if leg and leg == eid:
        return True
    entry = next(
        (
            r
            for r in legs
            if str(r.get("order_id") or "") == eid
            or entry_link_id(r) == eid
        ),
        None,
    )
    if entry is None:
        return False
    want = _entry_side_kind(entry)
    rside = repair_position_side(repair_row)
    if want == "L" and rside != "LONG":
        return False
    if want == "S" and rside != "SHORT":
        return False
    gk = row_group_key(entry) or leg_group_key(eid)
    if gk and row_group_key(repair_row) and row_group_key(repair_row) != gk:
        return False
    rq = _qty(repair_row)
    eq = _qty(entry)
    if rq > 0 and eq > 0 and abs(rq - eq) > 1e-6:
        return False
    return True


def repair_tp_rows_for_entry(
    legs: List[Dict[str, Any]], entry_order_id: str
) -> List[Dict[str, Any]]:
    eid = str(entry_order_id or "")
    out: List[Dict[str, Any]] = []
    for row in legs:
        if repair_tp_matches_entry(row, eid, legs):
            out.append(row)
    return out


def pick_repair_filled_tp(
    legs: List[Dict[str, Any]], entry_order_id: str
) -> Optional[Dict[str, Any]]:
    for row in repair_tp_rows_for_entry(legs, entry_order_id):
        if _is_filled_row(row):
            return row
    return None
