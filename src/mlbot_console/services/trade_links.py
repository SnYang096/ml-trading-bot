"""Entry→exit trade links for Trade Map (chop_grid L-leg ↔ TP / market_exit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlbot_console.services.multileg_order_links import (
    _is_filled_row,
    _pick_filled_tp,
    _price,
    _protection_tp_rows,
    annotate_leg_group,
    build_leg_link_index,
    entry_link_id,
    is_entry_row,
    is_l_entry_row,
    leg_suffix,
)
from mlbot_console.services.db import query_rows
from mlbot_console.services.trade_markers import (
    STRATEGY_COLORS,
    _marker_id,
    _parse_ts,
)

PNL_LINK_COLOR_WIN = "#26a69a"
PNL_LINK_COLOR_LOSS = "#ef5350"
PNL_LINK_COLOR_FLAT = "#8b949e"


def _order_rows(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT local_order_id, strategy, symbol, side, purpose, status, order_type,
               filled_quantity, average_price, filled_at, created_at, price, quantity,
               stop_price, leg_id
        FROM multi_leg_orders
        WHERE symbol = ?
        ORDER BY COALESCE(filled_at, created_at) ASC
    """
    rows = query_rows(db_path, sql, (symbol.upper(),))
    for row in rows:
        row["order_id"] = row.get("local_order_id")
    annotate_leg_group(rows)
    return rows


def _ts_row(row: Dict[str, Any]) -> Optional[int]:
    ts = _parse_ts(row.get("filled_at")) or _parse_ts(row.get("created_at"))
    return int(ts) if ts is not None else None


def _leg_position_side(row: Dict[str, Any]) -> str:
    side = str(row.get("side") or "").upper()
    if side == "BUY":
        return "long"
    if side == "SELL":
        return "short"
    return "long"


def _closed_link_color(
    *,
    strategy: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_usdt: Optional[float] = None,
) -> str:
    if pnl_usdt is not None:
        try:
            pnl = float(pnl_usdt)
            if pnl > 0:
                return PNL_LINK_COLOR_WIN
            if pnl < 0:
                return PNL_LINK_COLOR_LOSS
            return PNL_LINK_COLOR_FLAT
        except (TypeError, ValueError):
            pass
    try:
        ep = float(entry_price)
        xp = float(exit_price)
    except (TypeError, ValueError):
        return STRATEGY_COLORS.get(str(strategy or "").lower(), "#aaaaaa")
    side_l = str(side or "long").lower()
    diff = (xp - ep) if side_l == "long" else (ep - xp)
    if diff > 0:
        return PNL_LINK_COLOR_WIN
    if diff < 0:
        return PNL_LINK_COLOR_LOSS
    return PNL_LINK_COLOR_FLAT


def _append_link(
    out: List[Dict[str, Any]],
    *,
    strategy: str,
    leg: str,
    entry_time: int,
    entry_price: float,
    exit_time: int,
    exit_price: float,
    entry_marker_id: str,
    exit_marker_id: Optional[str],
    status: str,
    exit_kind: str,
    side: str = "long",
    pnl_usdt: Optional[float] = None,
) -> None:
    if str(status).lower() != "closed":
        return
    if exit_marker_id is None or not str(exit_marker_id).strip():
        return
    if exit_time < entry_time:
        exit_time = entry_time
    strat = str(strategy or "multi_leg").lower()
    out.append(
        {
            "strategy": strat,
            "leg": leg,
            "status": "closed",
            "exit_kind": exit_kind,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "entry_marker_id": entry_marker_id,
            "exit_marker_id": exit_marker_id,
            "side": side,
            "pnl_usdt": pnl_usdt,
            "color": _closed_link_color(
                strategy=strat,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usdt=pnl_usdt,
            ),
        }
    )


def _link_overlaps_window(
    link: Dict[str, Any],
    *,
    start_ts: Optional[int],
    end_ts: Optional[int],
    since_ts: Optional[int],
) -> bool:
    entry_ts = int(link.get("entry_time") or 0)
    exit_ts = int(link.get("exit_time") or entry_ts)
    if since_ts is not None and max(entry_ts, exit_ts) <= since_ts:
        return False
    if start_ts is not None and exit_ts < start_ts:
        return False
    if end_ts is not None and entry_ts > end_ts:
        return False
    return True


def _append_entry_tp_links(
    links: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    row: Dict[str, Any],
    seen_entry: set[str],
) -> None:
    """Link filled L/S grid entry legs to their TP protection orders."""
    if not is_entry_row(row) or not _is_filled_row(row):
        return
    entry_ts = _ts_row(row)
    entry_px = _price(row)
    if entry_ts is None or entry_px is None:
        return
    oid = str(row.get("local_order_id") or "")
    ekey = entry_link_id(row)
    if ekey in seen_entry:
        return
    seen_entry.add(ekey)

    leg_label = leg_suffix(ekey) or leg_suffix(oid) or ""
    entry_mid = _marker_id("multi_leg", "multi_leg_orders", oid or ekey)
    tp_rows = _protection_tp_rows(group_rows, ekey)
    if not tp_rows and oid:
        tp_rows = _protection_tp_rows(group_rows, oid)
    filled_tp = _pick_filled_tp(tp_rows)
    if filled_tp is None:
        from mlbot_console.services.multileg_repair_tp import pick_repair_filled_tp

        filled_tp = pick_repair_filled_tp(group_rows, ekey)
    strat = str(row.get("strategy") or "chop_grid")
    if filled_tp is not None:
        exit_ts = _ts_row(filled_tp)
        exit_px = _price(filled_tp)
        if exit_ts is None or exit_px is None:
            return
        exit_mid = _marker_id(
            "multi_leg",
            "multi_leg_orders",
            str(filled_tp.get("local_order_id") or ""),
        )
        _append_link(
            links,
            strategy=strat,
            leg=leg_label,
            entry_time=entry_ts,
            entry_price=entry_px,
            exit_time=exit_ts,
            exit_price=exit_px,
            entry_marker_id=entry_mid,
            exit_marker_id=exit_mid,
            status="closed",
            exit_kind="take_profit",
            side=_leg_position_side(row),
        )
        return


def multi_leg_trade_links(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (trade_links, supplemental_markers). Markers come from trade_markers."""
    if not db_path.is_file():
        return [], []

    sym = symbol.upper()
    rows = _order_rows(db_path, sym)
    by_group = build_leg_link_index(rows)
    links: List[Dict[str, Any]] = []
    seen_entry: set[str] = set()

    for group_rows in by_group.values():
        for row in group_rows:
            _append_entry_tp_links(links, group_rows, row, seen_entry)

        # Grid flatten: market_exit rows close remaining longs at same timestamp.
        for row in group_rows:
            purpose = str(row.get("purpose") or "").lower()
            if "market_exit" not in purpose or not _is_filled_row(row):
                continue
            exit_ts = _ts_row(row)
            exit_px = _price(row)
            if exit_ts is None or exit_px is None:
                continue
            exit_mid = _marker_id(
                "multi_leg", "multi_leg_orders", str(row.get("local_order_id") or "")
            )
            for ent in group_rows:
                if not is_l_entry_row(ent) or not _is_filled_row(ent):
                    continue
                eoid = str(ent.get("local_order_id") or "")
                ekey = entry_link_id(ent)
                if "take_profit" in str(ent.get("purpose") or "").lower():
                    continue
                entry_ts = _ts_row(ent)
                entry_px = _price(ent)
                if entry_ts is None or entry_px is None or exit_ts < entry_ts:
                    continue
                entry_mid = _marker_id("multi_leg", "multi_leg_orders", eoid or ekey)
                if any(lk.get("entry_marker_id") == entry_mid for lk in links):
                    continue
                _append_link(
                    links,
                    strategy=str(ent.get("strategy") or "chop_grid"),
                    leg=leg_suffix(ekey) or leg_suffix(eoid) or "L",
                    entry_time=entry_ts,
                    entry_price=entry_px,
                    exit_time=exit_ts,
                    exit_price=exit_px,
                    entry_marker_id=_marker_id(
                        "multi_leg", "multi_leg_orders", eoid or ekey
                    ),
                    exit_marker_id=exit_mid,
                    status="closed",
                    exit_kind="market_exit",
                    side=_leg_position_side(ent),
                )

    links = [
        lk
        for lk in links
        if _link_overlaps_window(
            lk, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
        )
    ]
    return links, []


def trend_trade_links(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    current_time: Optional[int] = None,
    current_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    sym = symbol.upper()
    sql = """
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id
        FROM positions
        WHERE symbol = ?
        ORDER BY entry_time ASC
    """
    links: List[Dict[str, Any]] = []
    for row in query_rows(db_path, sql, (sym,)):
        entry_ts = _parse_ts(row.get("entry_time"))
        try:
            entry_px = float(row.get("entry_price"))
        except (TypeError, ValueError):
            entry_px = None
        if entry_ts is None or entry_px is None or entry_px != entry_px:
            continue
        exit_ts = _parse_ts(row.get("exit_time"))
        try:
            exit_px = float(row.get("exit_price"))
        except (TypeError, ValueError):
            exit_px = None
        strat = str(row.get("strategy_id") or "trend").lower()
        pid = str(row.get("position_id") or "")
        entry_mid = _marker_id("trend", "positions", f"{pid}:entry")
        if exit_ts is None or exit_px is None:
            continue
        exit_mid = _marker_id("trend", "positions", f"{pid}:exit")
        pnl = row.get("realized_pnl")
        side = str(row.get("side") or "long").lower()
        _append_link(
            links,
            strategy=strat,
            leg="",
            entry_time=int(entry_ts),
            entry_price=entry_px,
            exit_time=int(exit_ts),
            exit_price=exit_px,
            entry_marker_id=entry_mid,
            exit_marker_id=exit_mid,
            status="closed",
            exit_kind="exit",
            side=side,
            pnl_usdt=float(pnl) if pnl is not None else None,
        )
    op_sql = """
        SELECT po.operation_id, po.operation_type, po.operation_time, po.price,
               p.position_id, p.side, p.exit_time, p.exit_price, p.status, p.strategy_id
        FROM position_operations po
        JOIN positions p ON p.position_id = po.position_id
        WHERE p.symbol = ?
        ORDER BY po.operation_time ASC
    """
    for row in query_rows(db_path, op_sql, (sym,)):
        op_type = str(row.get("operation_type") or "").lower()
        if "add" not in op_type:
            continue
        add_ts = _parse_ts(row.get("operation_time"))
        add_px = _price(row)
        if add_ts is None or add_px is None:
            continue
        exit_ts = _parse_ts(row.get("exit_time"))
        try:
            exit_px = float(row.get("exit_price"))
        except (TypeError, ValueError):
            exit_px = None
        if exit_ts is None or exit_px is None:
            continue
        strat = str(row.get("strategy_id") or "trend").lower()
        op_id = str(row.get("operation_id") or "")
        pid = str(row.get("position_id") or "")
        side = str(row.get("position_side") or row.get("side") or "long").lower()
        exit_mid = _marker_id("trend", "positions", f"{pid}:exit")
        _append_link(
            links,
            strategy=strat,
            leg="add",
            entry_time=int(add_ts),
            entry_price=add_px,
            exit_time=int(exit_ts),
            exit_price=float(exit_px),
            entry_marker_id=_marker_id("trend", "position_operations", op_id),
            exit_marker_id=exit_mid,
            status="closed",
            exit_kind="exit",
            side=side,
        )
    return [
        lk
        for lk in links
        if _link_overlaps_window(
            lk, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
        )
    ]


def spot_trade_links(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    sym = symbol.upper()
    sql = """
        SELECT order_id, created_at, updated_at, symbol, side, order_type,
               quantity, price, status, filled_quantity, filled_quote_usdt
        FROM spot_orders
        WHERE symbol = ?
        ORDER BY COALESCE(updated_at, created_at) ASC
    """
    open_buys: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    for row in query_rows(db_path, sql, (sym,)):
        status = str(row.get("status") or "").lower()
        filled_qty = float(row.get("filled_quantity") or 0)
        if status not in {"filled", "closed", "partially_filled"} and filled_qty <= 0:
            continue
        ts = _parse_ts(row.get("updated_at")) or _parse_ts(row.get("created_at"))
        px = _price(row)
        if ts is None or px is None:
            continue
        side = str(row.get("side") or "").lower()
        if side == "buy":
            open_buys.append(row | {"_ts": ts, "_px": px})
            continue
        if side != "sell" or not open_buys:
            continue
        entry = open_buys.pop(0)
        entry_ts = int(entry["_ts"])
        entry_px = float(entry["_px"])
        _append_link(
            links,
            strategy="spot_accum_simple",
            leg="",
            entry_time=entry_ts,
            entry_price=entry_px,
            exit_time=int(ts),
            exit_price=float(px),
            entry_marker_id=_marker_id(
                "spot", "spot_orders", str(entry.get("order_id") or "")
            ),
            exit_marker_id=_marker_id(
                "spot", "spot_orders", str(row.get("order_id") or "")
            ),
            status="closed",
            exit_kind="sell",
            side="long",
        )
    return [
        lk
        for lk in links
        if _link_overlaps_window(
            lk, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
        )
    ]


def collect_trade_links(
    *,
    multi_leg_db: Path,
    symbol: str,
    scopes: List[str],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    trend_db: Optional[Path] = None,
    spot_db: Optional[Path] = None,
    current_time: Optional[int] = None,
    current_price: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scope_set = {s.strip().lower() for s in scopes if s.strip()}
    merged: List[Dict[str, Any]] = []
    extras: List[Dict[str, Any]] = []
    if "multi_leg" in scope_set:
        ml, ex = multi_leg_trade_links(
            multi_leg_db,
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            since_ts=since_ts,
        )
        merged.extend(ml)
        extras.extend(ex)
    if "trend" in scope_set and trend_db is not None and trend_db.is_file():
        merged.extend(
            trend_trade_links(
                trend_db,
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                since_ts=since_ts,
                current_time=current_time,
                current_price=current_price,
            )
        )
    if "spot" in scope_set and spot_db is not None and spot_db.is_file():
        merged.extend(
            spot_trade_links(
                spot_db,
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                since_ts=since_ts,
            )
        )
    return merged, extras
