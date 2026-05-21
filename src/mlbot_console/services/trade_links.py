"""Entry→exit trade links for Trade Map (chop_grid L-leg ↔ TP / market_exit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlbot_console.services.multileg_order_links import (
    _is_filled_row,
    _pick_filled_tp,
    _pick_planned_tp,
    _price,
    _protection_tp_rows,
    annotate_leg_group,
    build_leg_link_index,
    entry_link_id,
    is_l_entry_row,
    leg_suffix,
)
from mlbot_console.services.db import query_rows
from mlbot_console.services.trade_markers import (
    STRATEGY_COLORS,
    _marker_id,
    _parse_ts,
    _ts_in_chart_window,
)


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
) -> None:
    if exit_time < entry_time:
        exit_time = entry_time
    strat = str(strategy or "multi_leg").lower()
    out.append(
        {
            "strategy": strat,
            "leg": leg,
            "status": status,
            "exit_kind": exit_kind,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "entry_marker_id": entry_marker_id,
            "exit_marker_id": exit_marker_id,
            "color": STRATEGY_COLORS.get(strat, "#aaaaaa"),
        }
    )


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
            if not is_l_entry_row(row):
                continue
            if not _is_filled_row(row):
                continue
            entry_ts = _ts_row(row)
            entry_px = _price(row)
            if entry_ts is None or entry_px is None:
                continue
            oid = str(row.get("local_order_id") or "")
            ekey = entry_link_id(row)
            if ekey in seen_entry:
                continue
            seen_entry.add(ekey)

            entry_mid = _marker_id("multi_leg", "multi_leg_orders", oid or ekey)
            tp_rows = _protection_tp_rows(group_rows, ekey)
            if not tp_rows and oid:
                tp_rows = _protection_tp_rows(group_rows, oid)
            filled_tp = _pick_filled_tp(tp_rows)
            if filled_tp is not None:
                exit_ts = _ts_row(filled_tp)
                exit_px = _price(filled_tp)
                if exit_ts is None or exit_px is None:
                    continue
                exit_mid = _marker_id(
                    "multi_leg",
                    "multi_leg_orders",
                    str(filled_tp.get("local_order_id") or ""),
                )
                _append_link(
                    links,
                    strategy=str(row.get("strategy") or "chop_grid"),
                    leg=leg_suffix(ekey) or leg_suffix(oid) or "L",
                    entry_time=entry_ts,
                    entry_price=entry_px,
                    exit_time=exit_ts,
                    exit_price=exit_px,
                    entry_marker_id=entry_mid,
                    exit_marker_id=exit_mid,
                    status="closed",
                    exit_kind="take_profit",
                )
                continue

            planned = _pick_planned_tp(tp_rows)
            if planned is not None:
                exit_px = _price(planned)
                exit_ts = _ts_row(planned) or entry_ts
                if exit_px is None:
                    continue
                exit_mid = _marker_id(
                    "multi_leg",
                    "multi_leg_orders",
                    str(planned.get("local_order_id") or ""),
                )
                _append_link(
                    links,
                    strategy=str(row.get("strategy") or "chop_grid"),
                    leg=leg_suffix(ekey) or leg_suffix(oid) or "L",
                    entry_time=entry_ts,
                    entry_price=entry_px,
                    exit_time=exit_ts,
                    exit_price=exit_px,
                    entry_marker_id=entry_mid,
                    exit_marker_id=exit_mid,
                    status="open",
                    exit_kind="take_profit_planned",
                )

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
                )

    return links, []


def trend_trade_links(
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
        if not _ts_in_chart_window(
            entry_ts, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
        ):
            continue
        exit_ts = _parse_ts(row.get("exit_time"))
        try:
            exit_px = float(row.get("exit_price"))
        except (TypeError, ValueError):
            exit_px = None
        strat = str(row.get("strategy_id") or "trend").lower()
        pid = str(row.get("position_id") or "")
        entry_mid = _marker_id("trend", "positions", f"{pid}:entry")
        if exit_ts is not None and exit_px is not None:
            exit_mid = _marker_id("trend", "positions", f"{pid}:exit")
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
            )
    return links


def collect_trade_links(
    *,
    multi_leg_db: Path,
    symbol: str,
    scopes: List[str],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    trend_db: Optional[Path] = None,
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
            )
        )
    return merged, extras
