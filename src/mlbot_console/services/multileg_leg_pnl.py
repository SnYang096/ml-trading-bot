"""Per-order realized / unrealized PnL for chop_grid multi-leg rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services.account_summary import _link_pnl_usdt
from mlbot_console.services.db import query_rows
from mlbot_console.services.multileg_order_links import (
    _LATE_FIXUP_SUFFIX,
    _is_filled_row,
    _pick_filled_sl,
    _pick_filled_tp,
    _pick_planned_tp,
    _price,
    _protection_sl_rows,
    _protection_tp_rows,
    build_leg_link_index,
    entry_link_id,
    filled_quantity,
    hydrate_multileg_fill_fields,
    is_entry_row,
    is_l_entry_row,
    is_pairable_market_exit_row,
    is_s_entry_row,
    is_trend_entry_row,
    leg_group_key,
    leg_suffix,
    row_group_key,
    late_fixup_entry_segment_matches,
    market_exit_closing_position_side,
    trend_entry_position_side,
    trend_exit_entry_id,
    trend_segment_key,
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


def exit_kind_for_multileg_row(
    exit_row: Dict[str, Any],
    *,
    entry_row: Optional[Dict[str, Any]] = None,
) -> str:
    """Map a filled exit order to a console ``exit_kind`` label."""
    exit_strat = str(exit_row.get("strategy") or "").lower()
    purpose = str(exit_row.get("purpose") or "").lower()
    if (
        entry_row is not None
        and is_trend_entry_row(entry_row)
        and exit_strat == "chop_grid"
        and "market_exit" in purpose
        and _is_filled_row(exit_row)
    ):
        return "cross_strategy_exit"
    oid = _order_key(exit_row).lower()
    if "take_profit" in purpose or "_tp" in oid:
        return "take_profit"
    if "stop_loss" in purpose or "_sl" in oid:
        return "stop_loss"
    if "market_exit" in purpose:
        if "regime" in oid or "regime_exit" in oid:
            return "regime_exit"
        if "basket_tp" in oid or "basket" in oid:
            return "take_profit"
        if "_market_exit_late_fixup" in oid:
            return "market_exit"
        return "market_exit"
    return "exit"


def leg_label_for_multileg_entry(entry_row: Dict[str, Any]) -> str:
    oid = _order_key(entry_row)
    ekey = entry_link_id(entry_row)
    if is_trend_entry_row(entry_row):
        if "trend_add" in oid:
            return "add"
        if "initial_trend" in oid:
            return "init"
    return leg_suffix(ekey) or leg_suffix(oid) or ""


def _stop_loss_matches_entry(
    row: Dict[str, Any], entry_order_id: str, entry_key: str
) -> bool:
    """Per-leg SL must match the entry id — not any same-side hedge stop."""
    eid = str(entry_order_id or "")
    ekey = str(entry_key or eid)
    exit_lid = str(row.get("leg_id") or "").strip()
    exit_oid = str(row.get("order_id") or row.get("local_order_id") or "")
    if exit_lid and (exit_lid == ekey or exit_lid == eid):
        return True
    if exit_oid in {f"{ekey}_sl", f"{eid}_sl"}:
        return True
    base = exit_lid[:-3] if exit_lid.endswith("_sl") else ""
    return bool(base and base in {ekey, eid})


def _chop_grid_exit_for_entry(
    group_rows: List[Dict[str, Any]], entry_row: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """In-group chop_grid market_exit / stop_loss (L/S legs), excluding basket TP."""
    if is_trend_entry_row(entry_row) or not is_entry_row(entry_row):
        return None
    entry_ts = _ts_row(entry_row)
    if entry_ts is None:
        return None
    ent_side = _entry_position_side(entry_row)
    if ent_side is None:
        return None
    ekey = entry_link_id(entry_row)
    ent_oid = _order_key(entry_row)
    best: Optional[Dict[str, Any]] = None
    best_ts = -1
    for row in group_rows:
        purpose = str(row.get("purpose") or "").lower()
        if "take_profit" in purpose:
            continue
        if "market_exit" not in purpose and "stop_loss" not in purpose:
            continue
        if not _is_filled_row(row) or _price(row) is None:
            continue
        exit_ts = _ts_row(row) or 0
        if exit_ts < entry_ts:
            continue
        exit_lid = str(row.get("leg_id") or "").strip()
        exit_oid = _order_key(row)
        matched = False
        if exit_lid and (exit_lid == ekey or exit_lid == ent_oid):
            matched = True
        elif "stop_loss" in purpose and _stop_loss_matches_entry(
            row, ent_oid, ekey
        ):
            matched = True
        elif "market_exit" in purpose and market_exit_closing_position_side(row) == ent_side:
            # Batch flatten in segment group (legacy chop_grid path).
            matched = True
        elif exit_oid and ent_oid and exit_oid == ent_oid:
            matched = True
        if not matched:
            continue
        if exit_ts >= best_ts:
            best = row
            best_ts = exit_ts
    return best


def _cross_strategy_flatten_for_trend_entry(
    all_rows: List[Dict[str, Any]],
    entry_row: Dict[str, Any],
    *,
    used_market_exit_ids: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Pair trend entry with chop_grid market_exit that flattened the shared slot."""
    if not is_trend_entry_row(entry_row):
        return None
    entry_ts = _ts_row(entry_row)
    if entry_ts is None:
        return None
    ent_side = _entry_position_side(entry_row)
    ent_qty = filled_quantity(entry_row)
    if ent_side is None or ent_qty <= 0:
        return None
    used = used_market_exit_ids if used_market_exit_ids is not None else set()
    best: Optional[Dict[str, Any]] = None
    best_ts = 2**62
    for row in all_rows:
        if str(row.get("strategy") or "").lower() != "chop_grid":
            continue
        purpose = str(row.get("purpose") or "").lower()
        if "market_exit" not in purpose:
            continue
        if not _is_filled_row(row) or _price(row) is None:
            continue
        oid = _order_key(row)
        if not oid or oid in used:
            continue
        exit_ts = _ts_row(row)
        if exit_ts is None or exit_ts < entry_ts:
            continue
        if market_exit_closing_position_side(row) != ent_side:
            continue
        mex_qty = filled_quantity(row)
        if mex_qty <= 0:
            continue
        qty_ratio = mex_qty / max(ent_qty, 1e-12)
        if qty_ratio < 0.98 or qty_ratio > 1.02:
            continue
        if exit_ts < best_ts:
            best = row
            best_ts = exit_ts
    if best is not None:
        used.add(_order_key(best))
    return best


def _trend_exit_for_entry(
    group_rows: List[Dict[str, Any]], entry_row: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not is_trend_entry_row(entry_row):
        return None
    entry_id = entry_link_id(entry_row)
    if not entry_id:
        return None
    entry_ts = _ts_row(entry_row)
    filled_best: Optional[Dict[str, Any]] = None
    filled_ts = -1
    skipped_best: Optional[Dict[str, Any]] = None
    skipped_ts = -1
    for row in group_rows:
        if not is_pairable_market_exit_row(row):
            continue
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        if trend_exit_entry_id(oid) != entry_id:
            continue
        exit_ts = _ts_row(row) or 0
        if entry_ts is not None and exit_ts < entry_ts:
            continue
        status = str(row.get("status") or "").lower()
        if _is_filled_row(row):
            if exit_ts >= filled_ts:
                filled_best = row
                filled_ts = exit_ts
        elif status in {"skipped_no_position", "rejected"}:
            if exit_ts >= skipped_ts:
                skipped_best = row
                skipped_ts = exit_ts
    if filled_best is not None:
        return filled_best
    return skipped_best


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


def _late_fixup_exit_for_entry(
    entry_row: Dict[str, Any],
    *,
    all_rows: List[Dict[str, Any]],
    orphan_market_exits: Optional[List[Dict[str, Any]]] = None,
    used_market_exit_ids: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Pair ``{segment}_market_exit_late_fixup`` even when exit ts < entry fill ts."""
    entry_id = _order_key(entry_row)
    seg = trend_segment_key(entry_id)
    if not seg:
        return None
    target_id = f"{seg}{_LATE_FIXUP_SUFFIX}"
    ent_side = _entry_position_side(entry_row)
    if ent_side is None:
        return None
    used = used_market_exit_ids if used_market_exit_ids is not None else set()
    ent_qty = filled_quantity(entry_row)
    for row in list(all_rows) + list(orphan_market_exits or []):
        mex_id = _order_key(row)
        if mex_id != target_id or not mex_id or mex_id in used:
            continue
        if not _is_filled_row(row) or _price(row) is None:
            continue
        if market_exit_closing_position_side(row) != ent_side:
            continue
        mex_qty = filled_quantity(row)
        if mex_qty <= 0 or ent_qty > mex_qty * 1.02:
            continue
        used.add(mex_id)
        return row
    return None


def _filled_exit_row(
    group_rows: List[Dict[str, Any]],
    entry_row: Dict[str, Any],
    *,
    orphan_market_exits: Optional[List[Dict[str, Any]]] = None,
    used_market_exit_ids: Optional[set[str]] = None,
    all_rows: Optional[List[Dict[str, Any]]] = None,
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

    sl_rows = _protection_sl_rows(group_rows, eid)
    if not sl_rows and oid and oid != eid:
        sl_rows = _protection_sl_rows(group_rows, oid)
    exit_row = _pick_filled_sl(sl_rows)
    if exit_row is not None:
        return exit_row

    exit_row = _chop_grid_exit_for_entry(group_rows, entry_row)
    if exit_row is not None:
        return exit_row

    exit_row = _cross_strategy_flatten_for_trend_entry(
        all_rows or group_rows,
        entry_row,
        used_market_exit_ids=used_market_exit_ids,
    )
    if exit_row is not None:
        return exit_row

    exit_row = _trend_exit_for_entry(group_rows, entry_row)
    if exit_row is not None:
        return exit_row

    exit_row = _late_fixup_exit_for_entry(
        entry_row,
        all_rows=all_rows or group_rows,
        orphan_market_exits=orphan_market_exits,
        used_market_exit_ids=used_market_exit_ids,
    )
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
        qty_ratio = mex_qty / max(ent_qty, 1e-12)
        if qty_ratio > 1.02:
            continue
        used.add(mex_id)
        return mex
    return None


def _stop_loss_closes_entry(
    sl_row: Dict[str, Any], entry_row: Dict[str, Any]
) -> bool:
    ent_side = _entry_position_side(entry_row)
    if ent_side is None:
        return False
    return market_exit_closing_position_side(sl_row) == ent_side


def _shared_stop_loss_pairs(
    by_group: Dict[str, List[Dict[str, Any]]],
    pending: List[Dict[str, Any]],
    pairs: List[tuple[Dict[str, Any], Dict[str, Any]]],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Map hedge-mode SL fills to legs without a dedicated ``{leg}_sl`` order."""
    paired_entries = {_order_key(entry) for entry, _ in pairs if _order_key(entry)}
    used_sl = {
        _order_key(exit_row)
        for _, exit_row in pairs
        if "stop_loss" in str(exit_row.get("purpose") or "").lower()
        and _order_key(exit_row)
    }
    extra: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for gk, group_rows in by_group.items():
        sl_fills = sorted(
            [
                row
                for row in group_rows
                if "stop_loss" in str(row.get("purpose") or "").lower()
                and _is_filled_row(row)
                and _price(row) is not None
                and _order_key(row)
                and _order_key(row) not in used_sl
            ],
            key=lambda row: (_ts_row(row) or 0, _order_key(row)),
        )
        if not sl_fills:
            continue
        orphans = [
            entry
            for entry in pending
            if row_group_key(entry) == gk
            and is_entry_row(entry)
            and _is_filled_row(entry)
            and not is_trend_entry_row(entry)
            and _order_key(entry)
            and _order_key(entry) not in paired_entries
        ]
        if not orphans:
            continue

        def _orphan_sort_key(entry: Dict[str, Any]) -> tuple:
            side = _entry_position_side(entry)
            px = float(_price(entry) or 0.0)
            ts = _ts_row(entry) or 0
            if side == "LONG":
                return (ts, px)
            return (ts, -px)

        orphans.sort(key=_orphan_sort_key)
        for entry in orphans:
            entry_ts = _ts_row(entry) or 0
            matched_sl: Optional[Dict[str, Any]] = None
            for sl_row in sl_fills:
                sl_key = _order_key(sl_row)
                if not sl_key or sl_key in used_sl:
                    continue
                if (_ts_row(sl_row) or 0) < entry_ts:
                    continue
                if not _stop_loss_closes_entry(sl_row, entry):
                    continue
                matched_sl = sl_row
                break
            if matched_sl is None:
                continue
            sl_key = _order_key(matched_sl)
            if not sl_key:
                continue
            extra.append((entry, matched_sl))
            paired_entries.add(_order_key(entry) or "")
            used_sl.add(sl_key)
    return extra


def pair_multileg_entry_exits(
    rows: List[Dict[str, Any]],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Closed entry→exit pairs for chop_grid + trend_scalp (shared by links and PnL)."""
    by_group = build_leg_link_index(rows)
    orphan_exits = _orphan_market_exit_rows(rows)
    used_market_exit_ids: set[str] = set()
    pending: List[Dict[str, Any]] = []
    for group_rows in by_group.values():
        for entry in group_rows:
            if is_entry_row(entry) and _is_filled_row(entry) and _order_key(entry):
                pending.append(entry)
    pending.sort(key=lambda r: (_ts_row(r) or 0, _order_key(r)))

    pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for entry in pending:
        gk = row_group_key(entry)
        group_rows = by_group.get(gk or "", [entry])
        exit_row = _filled_exit_row(
            group_rows,
            entry,
            orphan_market_exits=orphan_exits,
            used_market_exit_ids=used_market_exit_ids,
            all_rows=rows,
        )
        if exit_row is not None:
            pairs.append((entry, exit_row))
    pairs.extend(_shared_stop_loss_pairs(by_group, pending, pairs))
    return pairs


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
    mark = float((mark_prices or {}).get(sym) or 0.0)
    out: Dict[str, Dict[str, Any]] = {}

    pending_entries: List[Dict[str, Any]] = []
    for group_rows in by_group.values():
        for entry in group_rows:
            if is_entry_row(entry) and _is_filled_row(entry) and _order_key(entry):
                pending_entries.append(entry)
    pending_entries.sort(key=lambda r: (_ts_row(r) or 0, _order_key(r)))

    for entry, exit_row in pair_multileg_entry_exits(raw):
        entry_key = _order_key(entry)
        if not entry_key:
            continue
        pnl = _link_pnl_usdt(entry, exit_row)
        if pnl is None:
            continue
        rec = _pnl_rec(pnl=pnl, hint="已实现")
        out[entry_key] = rec
        exit_key = _order_key(exit_row)
        if exit_key:
            out[exit_key] = dict(rec)

    for entry in pending_entries:
        entry_key = _order_key(entry)
        if not entry_key or entry_key in out:
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
