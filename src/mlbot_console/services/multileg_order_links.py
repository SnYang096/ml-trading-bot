"""Pair chop_grid / trend_scalp L/S leg orders for console take-profit display."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from mlbot_console.services.trade_markers import _OPEN_ORDER_STATUSES, _multi_leg_take_profit_price

_LEG_SUFFIX_RE = re.compile(r"_(L|S)(\d+)$", re.I)
_TP_SUFFIX_RE = re.compile(r"_(L|S)(\d+)_tp$", re.I)
_PROT_SUFFIX_RE = re.compile(r"_(L|S)(\d+)_(tp|sl)(?:_supp)?$", re.I)
_TREND_ENTRY_RE = re.compile(
    r"^.+_(?:initial_trend|trend_add)_(?:BUY|SELL)_\d+_\d+$",
    re.I,
)
_TREND_EXIT_RE = re.compile(
    r"^(.+_(?:initial_trend|trend_add)_(?:BUY|SELL)_\d+_\d+)_fill\d+_exit_",
    re.I,
)
_TREND_SEGMENT_RE = re.compile(r"^(.+)_(?:initial_trend|trend_add)_", re.I)
_LATE_FIXUP_SUFFIX = "_market_exit_late_fixup"


def trend_segment_key(order_id: str) -> Optional[str]:
    """Segment id prefix for trend_scalp entry / add / market_exit rows."""
    m = _TREND_SEGMENT_RE.match(str(order_id or ""))
    if not m:
        return None
    return m.group(1)


def trend_exit_entry_id(order_id: str) -> Optional[str]:
    """Entry local_order_id embedded in a trend market_exit id."""
    m = _TREND_EXIT_RE.match(str(order_id or ""))
    if not m:
        return None
    return m.group(1)


def is_trend_entry_row(row: Dict[str, Any]) -> bool:
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" in purpose or "market_exit" in purpose or "stop_loss" in purpose:
        return False
    for field in ("order_id", "local_order_id", "leg_id"):
        if _TREND_ENTRY_RE.match(str(row.get(field) or "")):
            return True
    return False


def trend_entry_position_side(row: Dict[str, Any]) -> Optional[str]:
    for field in ("order_id", "local_order_id", "leg_id"):
        oid = str(row.get(field) or "")
        if not _TREND_ENTRY_RE.match(oid):
            continue
        if "_BUY_" in oid.upper():
            return "LONG"
        if "_SELL_" in oid.upper():
            return "SHORT"
    return None


def late_fixup_segment_prefix(order_id: str) -> Optional[str]:
    """Segment batch id embedded in ``{segment}_market_exit_late_fixup`` rows."""
    oid = str(order_id or "")
    if oid.endswith(_LATE_FIXUP_SUFFIX):
        return oid[: -len(_LATE_FIXUP_SUFFIX)]
    return None


def late_fixup_entry_segment_matches(
    exit_order_id: str, entry_order_id: str
) -> bool:
    """When exit is a late_fixup row, require the same trend segment prefix."""
    seg = late_fixup_segment_prefix(exit_order_id)
    if not seg:
        return True
    return trend_segment_key(entry_order_id) == seg


def market_exit_closing_position_side(row: Dict[str, Any]) -> Optional[str]:
    """Position reduced by a market_exit row (hedge: BUY closes SHORT, SELL closes LONG)."""
    ps = str(row.get("position_side") or "").upper()
    if ps in {"LONG", "SHORT"}:
        return ps
    side = str(row.get("side") or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    if side == "BUY":
        return "SHORT"
    if side == "SELL":
        return "LONG"
    return None


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
    oid = str(order_id or "")
    m = _PROT_SUFFIX_RE.search(oid)
    if m:
        label = f"{m.group(1)}{m.group(2)}_{m.group(3).lower()}"
        if oid.endswith("_supp"):
            label += "_supp"
        return label
    m = _LEG_SUFFIX_RE.search(oid)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = _TP_SUFFIX_RE.search(oid)
    if m:
        return f"{m.group(1)}{m.group(2)}_tp"
    return ""


def leg_index(order_id: str) -> int:
    oid = str(order_id or "")
    m = _PROT_SUFFIX_RE.search(oid) or _LEG_SUFFIX_RE.search(oid) or _TP_SUFFIX_RE.search(oid)
    if not m:
        return 0
    try:
        return int(m.group(2))
    except ValueError:
        return 0


def _parse_raw_json_blob(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("raw_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str) and parsed.strip():
            try:
                nested = json.loads(parsed)
                if isinstance(nested, dict):
                    return nested
            except json.JSONDecodeError:
                return {}
    return {}


def hydrate_multileg_fill_fields(row: Dict[str, Any]) -> None:
    """Backfill filled_quantity / average_price from persisted exchange raw_json.

    Idempotent: parses ``raw_json`` at most once per row (memoized via a private
    flag) since ``_price`` may be called many times on the same row.
    """
    if row.get("_fill_hydrated"):
        return
    row["_fill_hydrated"] = True
    raw = _parse_raw_json_blob(row)
    if not raw:
        return
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}

    if float(row.get("filled_quantity") or 0) <= 0:
        for key in ("filled", "filled_quantity", "executedQty"):
            val = raw.get(key) if key != "executedQty" else info.get("executedQty")
            if val is None:
                continue
            try:
                qty = float(val)
            except (TypeError, ValueError):
                continue
            if qty > 0:
                row["filled_quantity"] = qty
                break

    avg = row.get("average_price")
    if avg is None or avg != avg:
        for key in ("average_price", "price", "avgPrice"):
            val = raw.get(key) if key != "avgPrice" else info.get("avgPrice")
            if val is None:
                continue
            try:
                px = float(val)
            except (TypeError, ValueError):
                continue
            if px > 0:
                row["average_price"] = px
                break


def _price(row: Dict[str, Any]) -> Optional[float]:
    hydrate_multileg_fill_fields(row)
    for key in ("average_price", "price", "stop_price"):
        val = row.get(key)
        if val is not None and val == val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def filled_quantity(row: Dict[str, Any]) -> float:
    hydrate_multileg_fill_fields(row)
    qty = float(row.get("filled_quantity") or 0)
    if qty > 0:
        return qty
    return float(row.get("quantity") or 0)


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
    seen: set[str] = set()
    for row in legs:
        oid = str(row.get("order_id") or "")
        purpose = str(row.get("purpose") or "").lower()
        leg = str(row.get("leg_id") or "")
        if oid.startswith(f"{eid}_tp") or (
            "take_profit" in purpose and leg == eid
        ):
            if oid not in seen:
                seen.add(oid)
                out.append(row)
    from mlbot_console.services.multileg_repair_tp import repair_tp_rows_for_entry

    for row in repair_tp_rows_for_entry(legs, eid):
        oid = str(row.get("order_id") or "")
        if oid and oid not in seen:
            seen.add(oid)
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


def _protection_sl_rows(
    legs: List[Dict[str, Any]], entry_order_id: str
) -> List[Dict[str, Any]]:
    eid = str(entry_order_id or "")
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in legs:
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        purpose = str(row.get("purpose") or "").lower()
        leg = str(row.get("leg_id") or "")
        if oid.startswith(f"{eid}_sl") or (
            "stop_loss" in purpose and leg == eid
        ):
            if oid not in seen:
                seen.add(oid)
                out.append(row)
    return out


def _pick_filled_sl(sl_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _pick_filled_tp(sl_rows)


def is_s_entry_row(row: Dict[str, Any]) -> bool:
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" in purpose or "market_exit" in purpose or "stop_loss" in purpose:
        return False
    for field in ("order_id", "local_order_id", "leg_id"):
        oid = str(row.get(field) or "")
        if _PROT_SUFFIX_RE.search(oid) or _TP_SUFFIX_RE.search(oid):
            return False
        if leg_side_kind(oid) == "S":
            return True
    return False


def is_entry_row(row: Dict[str, Any]) -> bool:
    return is_l_entry_row(row) or is_s_entry_row(row) or is_trend_entry_row(row)


def _entry_side_kind(row: Dict[str, Any]) -> Optional[str]:
    for field in ("order_id", "local_order_id", "leg_id"):
        kind = leg_side_kind(str(row.get(field) or ""))
        if kind:
            return kind
    return None


def _infer_chop_grid_spacing(legs: List[Dict[str, Any]], side_kind: str) -> Optional[float]:
    """Spacing from an entry↔TP pair on the same side, else from adjacent entry prices."""
    spacings: List[float] = []
    indexed: List[Tuple[int, float]] = []
    for row in legs:
        if _entry_side_kind(row) != side_kind or not is_entry_row(row):
            continue
        entry_px = _price(row)
        idx = leg_index(
            str(row.get("order_id") or row.get("local_order_id") or row.get("leg_id") or "")
        )
        if entry_px is not None and idx > 0:
            indexed.append((idx, entry_px))
        eid = entry_link_id(row)
        oid = str(row.get("order_id") or "")
        tp_rows = _protection_tp_rows(legs, eid)
        if not tp_rows and oid and oid != eid:
            tp_rows = _protection_tp_rows(legs, oid)
        for tp in tp_rows:
            tp_px = _price(tp)
            if entry_px is not None and tp_px is not None:
                spacings.append(abs(entry_px - tp_px))
    if spacings:
        return spacings[0]
    if len(indexed) >= 2:
        indexed.sort(key=lambda x: x[0])
        i0, p0 = indexed[0]
        i1, p1 = indexed[-1]
        if i1 != i0:
            return abs(p1 - p0) / abs(i1 - i0)
    return None


def _planned_tp_price(entry_px: float, spacing: float, side_kind: str) -> float:
    if side_kind == "L":
        return entry_px + spacing
    return entry_px - spacing


def annotate_leg_group(legs: List[Dict[str, Any]]) -> None:
    """Mutate raw multi_leg_orders rows with _link_* fields for entry legs."""
    l_legs = [r for r in legs if is_l_entry_row(r)]
    s_legs = [r for r in legs if is_s_entry_row(r)]
    if not l_legs and not s_legs:
        return

    for row in l_legs + s_legs:
        eid = entry_link_id(row)
        tp_rows = _protection_tp_rows(legs, eid)
        oid = str(row.get("order_id") or "")
        if not tp_rows and oid and oid != eid:
            tp_rows = _protection_tp_rows(legs, oid)
        planned = _pick_planned_tp(tp_rows)
        exit_row = _pick_filled_tp(tp_rows)
        if exit_row is None:
            from mlbot_console.services.multileg_repair_tp import pick_repair_filled_tp

            exit_row = pick_repair_filled_tp(legs, eid)
        if planned is not None:
            from mlbot_console.services.multileg_repair_tp import (
                is_repair_tp_row,
                repair_display_leg_label,
            )

            planned_oid = str(planned.get("order_id") or "")
            row["_link_tp_price"] = _price(planned)
            row["_link_tp_leg"] = leg_suffix(planned_oid)
            row["_link_tp_status"] = str(planned.get("status") or "")
            row["_link_tp_order_id"] = planned_oid
            row["_link_tp_leg_label"] = (
                repair_display_leg_label(planned)
                if is_repair_tp_row(planned)
                else leg_suffix(planned_oid)
            )
            row["_link_tp_is_repair"] = is_repair_tp_row(planned)
        elif not tp_rows and _is_filled_row(row):
            side = _entry_side_kind(row)
            entry_px = _price(row)
            spacing = _infer_chop_grid_spacing(legs, side) if side else None
            if side and entry_px is not None and spacing is not None and spacing > 0:
                row["_link_tp_price"] = _planned_tp_price(entry_px, spacing, side)
                row["_link_tp_status"] = "missing"
        if exit_row is not None:
            from mlbot_console.services.multileg_repair_tp import (
                is_repair_tp_row,
                repair_display_leg_label,
            )

            exit_oid = str(exit_row.get("order_id") or "")
            row["_link_exit_price"] = _price(exit_row)
            row["_link_exit_leg"] = exit_oid
            row["_link_exit_status"] = str(exit_row.get("status") or "")
            row["_link_exit_is_repair"] = is_repair_tp_row(exit_row)
            row["_link_tp_order_id"] = exit_oid
            row["_link_tp_leg_label"] = (
                repair_display_leg_label(exit_row)
                if is_repair_tp_row(exit_row)
                else leg_suffix(exit_oid)
            )
            row["_link_tp_is_repair"] = is_repair_tp_row(exit_row)


def row_group_key(row: Dict[str, Any]) -> Optional[str]:
    """Group key for chop_grid legs and trend_scalp segment batches."""
    for field in ("order_id", "local_order_id", "leg_id"):
        gk = leg_group_key(str(row.get(field) or ""))
        if gk:
            return gk
    for field in ("order_id", "local_order_id", "leg_id"):
        sk = trend_segment_key(str(row.get(field) or ""))
        if sk:
            return sk
    lid = str(row.get("leg_id") or "")
    m = _LEG_SUFFIX_RE.search(lid)
    if m:
        return lid[: m.start()]
    return None


def is_l_entry_row(row: Dict[str, Any]) -> bool:
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" in purpose or "market_exit" in purpose or "stop_loss" in purpose:
        return False
    for field in ("order_id", "local_order_id", "leg_id"):
        oid = str(row.get(field) or "")
        if _PROT_SUFFIX_RE.search(oid) or _TP_SUFFIX_RE.search(oid):
            return False
        if leg_side_kind(oid) == "L":
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


def _is_tp_protection_row(row: Dict[str, Any]) -> bool:
    from mlbot_console.services.multileg_repair_tp import is_repair_tp_row

    if is_repair_tp_row(row):
        return True
    purpose = str(row.get("purpose") or "").lower()
    if "take_profit" in purpose:
        return True
    for field in ("order_id", "local_order_id", "leg_id"):
        oid = str(row.get(field) or "")
        if _TP_SUFFIX_RE.search(oid) or (
            _PROT_SUFFIX_RE.search(oid) and "_tp" in oid.lower()
        ):
            return True
    return False


def resolve_take_profit_display(row: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Return (price, hint) for UI: *_tp protection orders, not grid S entry legs.
    """
    purpose = str(row.get("purpose") or "").lower()
    oid = str(row.get("order_id") or row.get("local_order_id") or "")
    if "stop_loss" in purpose or oid.endswith("_sl") or "_sl_" in oid:
        return None, ""

    exit_px = row.get("_link_exit_price")
    if exit_px is not None and exit_px == exit_px:
        hint = "已平仓"
        leg = row.get("_link_exit_leg")
        if row.get("_link_exit_is_repair"):
            hint = "补挂止盈"
        elif leg:
            hint = f"已平·{str(leg)[-6:]}"
        return float(exit_px), hint

    link_px = row.get("_link_tp_price")
    if link_px is not None and link_px == link_px:
        st = str(row.get("_link_tp_status") or "").lower()
        if st == "missing":
            return float(link_px), "未挂止盈"
        hint = "挂单"
        leg = row.get("_link_tp_leg") or ""
        if leg and st:
            hint = f"{leg}·{st}"
        elif leg:
            hint = str(leg)
        elif st:
            hint = st
        return float(link_px), hint

    if _is_tp_protection_row(row):
        direct = _multi_leg_take_profit_price(row)
        if direct is not None:
            return direct, "止盈单"

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
               filled_at, strategy, leg_id, client_order_id
        FROM multi_leg_orders
        WHERE symbol = ?
    """
    all_rows = query_rows(db_path, sql, (sym,))
    known = {str(r.get("order_id") or "") for r in all_rows}
    for item in rows:
        oid = str(item.get("order_id") or "")
        if oid and oid not in known:
            all_rows.append(item)
            known.add(oid)
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
            "_link_tp_order_id",
            "_link_tp_leg_label",
            "_link_tp_is_repair",
            "_link_exit_price",
            "_link_exit_leg",
            "_link_exit_status",
            "_link_exit_is_repair",
        ):
            if key in src:
                item[key] = src[key]
