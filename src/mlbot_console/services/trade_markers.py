"""Extract live trade markers from read-only SQLite stores."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mlbot_console.services.db import query_rows

_LEG_SUFFIX_RE = re.compile(r"_(L|S)(\d+)$", re.I)
_TP_SUFFIX_RE = re.compile(r"_(L|S)(\d+)_tp$", re.I)


def _leg_label_from_order_id(order_id: str) -> str:
    oid = str(order_id or "")
    m = _LEG_SUFFIX_RE.search(oid)
    if not m:
        m = _TP_SUFFIX_RE.search(oid)
        if m:
            return f"{m.group(1)}{m.group(2)}_tp"
        return ""
    return f"{m.group(1)}{m.group(2)}"


# Visual tokens aligned with scripts/event_backtest/reporting/trading_map.py
# Exchange / OMS statuses that count as still-open working orders (shown when Pending is on).
_OPEN_ORDER_STATUSES = frozenset(
    {
        "open",
        "pending",
        "new",
        "submitted",
        "shadow",
        "partially_filled",
        "partial_filled",
    }
)

STRATEGY_COLORS: Dict[str, str] = {
    "tpc": "#3274D9",
    "bpc": "#3274D9",
    "fer": "#B877D9",
    "me": "#FF9830",
    "chop_grid": "#73BF69",
    "trend_scalp": "#ca8a04",
    "spot_accum_simple": "#2e7d32",
}

CHOP_GRID_REGIME_EXIT_COLOR = "#ff7043"


def _iso_from_unix(ts: Optional[int]) -> Optional[str]:
    """Unix seconds → ISO 8601 UTC string for SQLite TEXT column comparison."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )
    except (TypeError, ValueError, OSError):
        return None


def _sql_time_range(
    start_ts: Optional[int],
    end_ts: Optional[int],
    col: str = "created_at",
) -> tuple[str, List[str]]:
    """SQL clause + params for TEXT ISO timestamp column range filter."""
    clauses: List[str] = []
    params: List[str] = []
    if start_ts is not None:
        iso = _iso_from_unix(start_ts)
        if iso:
            clauses.append(f"{col} >= ?")
            params.append(iso)
    if end_ts is not None:
        iso = _iso_from_unix(end_ts)
        if iso:
            clauses.append(f"{col} <= ?")
            params.append(iso)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _sql_time_range_expr(
    start_ts: Optional[int],
    end_ts: Optional[int],
    expr: str,
) -> tuple[str, List[str]]:
    """Range filter on a SQL expression (e.g. COALESCE(filled_at, created_at))."""
    clauses: List[str] = []
    params: List[str] = []
    if start_ts is not None:
        iso = _iso_from_unix(start_ts)
        if iso:
            clauses.append(f"{expr} >= ?")
            params.append(iso)
    if end_ts is not None:
        iso = _iso_from_unix(end_ts)
        if iso:
            clauses.append(f"{expr} <= ?")
            params.append(iso)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _sql_any_col_in_window(
    start_ts: Optional[int],
    end_ts: Optional[int],
    cols: List[str],
) -> tuple[str, List[str]]:
    """Row matches if any column falls in [start_ts, end_ts] (positions entry/exit)."""
    if start_ts is None and end_ts is None:
        return "", []
    per_col: List[str] = []
    params: List[str] = []
    for col in cols:
        col_clauses: List[str] = []
        if start_ts is not None:
            iso = _iso_from_unix(start_ts)
            if iso:
                col_clauses.append(f"{col} >= ?")
                params.append(iso)
        if end_ts is not None:
            iso = _iso_from_unix(end_ts)
            if iso:
                col_clauses.append(f"{col} <= ?")
                params.append(iso)
        if col_clauses:
            per_col.append(f"({' AND '.join(col_clauses)})")
    if not per_col:
        return "", []
    return " AND (" + " OR ".join(per_col) + ")", params


_MARKER_QUERY_LIMIT = 5000


def _parse_ts(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            v = float(raw)
            if v > 1e12:
                return int(v / 1000)
            if v > 1e9:
                return int(v)
        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (TypeError, ValueError):
        return None


def _marker_id(scope: str, source: str, key: str) -> str:
    return f"{scope}:{source}:{key}"


def _action_reason_from_row(row: Dict[str, Any]) -> str:
    raw = row.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, dict):
        return str(raw.get("reason") or raw.get("exit_reason") or "").strip()
    return ""


def _merge_chop_regime_exit_markers(
    markers: List[Dict[str, Any]],
    regime_exits: List[Dict[str, Any]],
    *,
    time_tolerance_sec: int = 1,
) -> List[Dict[str, Any]]:
    """Add feature-bus regime exits unless a chop_grid exit marker already exists at that bar."""
    if not regime_exits:
        return markers
    chop_exit_times: Set[int] = set()
    for m in markers:
        if str(m.get("strategy") or "").lower() != "chop_grid":
            continue
        if str(m.get("event") or "").lower() != "exit":
            continue
        try:
            chop_exit_times.add(int(m["time"]))
        except (TypeError, ValueError):
            continue
    seen_ids: Set[str] = {str(m.get("id") or "") for m in markers}
    out = list(markers)
    for m in regime_exits:
        mid = str(m.get("id") or "")
        if mid in seen_ids:
            continue
        t = int(m["time"])
        if any(abs(t - et) <= time_tolerance_sec for et in chop_exit_times):
            continue
        seen_ids.add(mid)
        out.append(m)
    out.sort(key=lambda x: x["time"])
    return out


def _multi_leg_event(
    purpose: str,
    order_type: str = "",
    *,
    local_order_id: str = "",
    is_filled: bool = False,
) -> str:
    """Map order row to marker event: entry | grid | tp | exit.

    - grid: open resting grid limits (L2/S1/S2 …), not a close
    - tp: reduce-only take-profit protection (*_tp), not a grid S leg
    - exit: regime/market flatten only
    - entry: filled grid leg open
    """
    p = str(purpose or "").lower()
    ot = str(order_type or "").lower()
    oid = str(local_order_id or "")
    if "_tp" in oid or "take_profit" in p or "take_profit" in ot or p == "tp":
        return "tp"
    if "market_exit" in p:
        return "exit"
    if any(x in p for x in ("close", "reduce", "stop")) and "take_profit" not in p:
        return "exit"
    if p in {"exit"}:
        return "exit"
    if not is_filled and (p in {"place", "entry", ""} or "_L" in oid or "_S" in oid):
        return "grid"
    return "entry"


def _multi_leg_take_profit_price(row: Dict[str, Any]) -> Optional[float]:
    purpose = str(row.get("purpose") or "").lower()
    ot = str(row.get("order_type") or "").lower()
    sp = row.get("stop_price")
    pr = row.get("price")
    if "take_profit" in purpose or "take_profit" in ot:
        if sp is not None and sp == sp:
            return float(sp)
        if pr is not None and pr == pr:
            return float(pr)
    if sp is not None and sp == sp and float(sp) > 0:
        return float(sp)
    leg = str(row.get("leg_id") or row.get("local_order_id") or "")
    if "_S" in leg.upper() and str(row.get("side") or "").upper() in {"SELL", "SHORT"}:
        if pr is not None and pr == pr:
            return float(pr)
    return None


def _append(
    out: List[Dict[str, Any]],
    seen: Set[str],
    *,
    scope: str,
    source: str,
    key: str,
    symbol: str,
    event: str,
    side: str,
    price: Optional[float],
    qty: Optional[float] = None,
    strategy: str = "unknown",
    is_add: bool = False,
    status: str = "filled",
    pnl_usdt: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    color: Optional[str] = None,
) -> None:
    mid = _marker_id(scope, source, key)
    if mid in seen:
        return
    t = extra.get("time") if extra and "time" in extra else None
    if t is None:
        return
    seen.add(mid)
    strat = str(strategy or "unknown").lower()
    side_l = str(side or "").lower()
    payload: Dict[str, Any] = {
        "id": mid,
        "time": int(t),
        "symbol": str(symbol).upper(),
        "scope": scope,
        "strategy": strat,
        "event": event,
        "side": side_l,
        "price": price,
        "qty": qty,
        "pnl_usdt": pnl_usdt,
        "is_add": bool(is_add),
        "status": status,
        "color": color or STRATEGY_COLORS.get(strat, "#aaaaaa"),
    }
    if extra:
        payload["detail"] = {k: v for k, v in extra.items() if k != "time"}
    out.append(payload)


def _ts_in_chart_window(
    ts: int,
    *,
    start_ts: Optional[int],
    end_ts: Optional[int],
    since_ts: Optional[int],
) -> bool:
    if since_ts is not None and ts <= since_ts:
        return False
    if start_ts is not None and ts < start_ts:
        return False
    if end_ts is not None and ts > end_ts:
        return False
    return True


def trend_markers(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    include_open_orders: bool = False,
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    pos_time_clause, pos_time_params = _sql_any_col_in_window(
        start_ts, end_ts, ["entry_time", "exit_time"]
    )
    pos_sql = f"""
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id
        FROM positions
        WHERE symbol = ?{pos_time_clause}
        ORDER BY entry_time ASC
        LIMIT {_MARKER_QUERY_LIMIT}
    """
    for row in query_rows(db_path, pos_sql, (sym, *pos_time_params)):
        strat = str(row.get("strategy_id") or "unknown").lower()
        side = str(row.get("side") or "long").lower()
        et = _parse_ts(row.get("entry_time"))
        if et is not None:
            if since_ts is not None and et <= since_ts:
                pass
            elif start_ts is None or et >= start_ts:
                if end_ts is None or et <= end_ts:
                    _append(
                        out,
                        seen,
                        scope="trend",
                        source="positions",
                        key=f"{row['position_id']}:entry",
                        symbol=sym,
                        event="entry",
                        side=side,
                        price=_f(row.get("entry_price")),
                        strategy=strat,
                        extra={"time": et, "position_id": row["position_id"]},
                    )
        xt = _parse_ts(row.get("exit_time"))
        if xt is not None:
            if since_ts is not None and xt <= since_ts:
                continue
            if start_ts is not None and xt < start_ts:
                continue
            if end_ts is not None and xt > end_ts:
                continue
            _append(
                out,
                seen,
                scope="trend",
                source="positions",
                key=f"{row['position_id']}:exit",
                symbol=sym,
                event="exit",
                side=side,
                price=_f(row.get("exit_price")),
                strategy=strat,
                pnl_usdt=_f(row.get("realized_pnl")),
                extra={"time": xt, "position_id": row["position_id"]},
            )

    op_time_clause, op_time_params = _sql_time_range(
        start_ts, end_ts, col="po.operation_time"
    )
    op_sql = f"""
        SELECT po.operation_id, po.position_id, po.operation_type,
               po.operation_time, po.size, po.price, po.reason,
               p.side AS position_side, p.strategy_id
        FROM position_operations po
        JOIN positions p ON p.position_id = po.position_id
        WHERE p.symbol = ?{op_time_clause}
        ORDER BY po.operation_time ASC
        LIMIT {_MARKER_QUERY_LIMIT}
    """
    for row in query_rows(db_path, op_sql, (sym, *op_time_params)):
        ot = _parse_ts(row.get("operation_time"))
        if ot is None:
            continue
        if since_ts is not None and ot <= since_ts:
            continue
        if start_ts is not None and ot < start_ts:
            continue
        if end_ts is not None and ot > end_ts:
            continue
        op_type = str(row.get("operation_type") or "").lower()
        is_add = "add" in op_type
        event = "entry" if is_add or "open" in op_type or "entry" in op_type else "exit"
        position_side = str(row.get("position_side") or "long").lower()
        side = (
            position_side
            if event == "entry"
            else ("short" if position_side == "long" else "long")
        )
        _append(
            out,
            seen,
            scope="trend",
            source="position_operations",
            key=str(row.get("operation_id")),
            symbol=sym,
            event=event,
            side=side,
            price=_f(row.get("price")),
            qty=_f(row.get("size")),
            strategy=str(row.get("strategy_id") or "unknown").lower(),
            is_add=is_add,
            extra={
                "time": ot,
                "operation_type": op_type,
                "position_id": row.get("position_id"),
            },
        )

    # When include_open_orders=True, pending orders outside the chart window
    # must still be returned; skip SQL time pushdown so they aren't filtered out.
    if include_open_orders:
        ord_time_clause, ord_time_params = "", []
    else:
        ord_time_clause, ord_time_params = _sql_time_range_expr(
            start_ts, end_ts, "COALESCE(o.filled_at, o.created_at)"
        )
    ord_sql = f"""
        SELECT o.order_id, o.symbol AS symbol, o.side AS side, o.status,
               o.filled_at, o.created_at, o.average_price, o.filled_quantity, o.position_id,
               p.side AS position_side, p.strategy_id
        FROM orders o
        LEFT JOIN positions p ON p.position_id = o.position_id
        WHERE o.symbol = ?{ord_time_clause}
        ORDER BY COALESCE(o.filled_at, o.created_at) ASC
        LIMIT {_MARKER_QUERY_LIMIT}
    """
    for row in query_rows(db_path, ord_sql, (sym, *ord_time_params)):
        status = str(row.get("status") or "").lower()
        filled_qty = _f(row.get("filled_quantity")) or 0.0
        is_filled = status in {"filled", "partially_filled"} or filled_qty > 0
        if not is_filled and status not in _OPEN_ORDER_STATUSES:
            continue
        ft = _parse_ts(row.get("filled_at")) or _parse_ts(row.get("created_at"))
        if ft is None:
            continue
        if is_filled:
            if not _ts_in_chart_window(
                ft, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
            ):
                continue
        elif not include_open_orders:
            if not _ts_in_chart_window(
                ft, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
            ):
                continue
        elif since_ts is not None and ft <= since_ts:
            continue
        side_raw = str(row.get("side") or "").upper()
        position_side = str(row.get("position_side") or "").lower()
        if position_side in {"long", "short"}:
            event = (
                "entry"
                if (position_side == "long" and side_raw in {"BUY", "LONG"})
                or (position_side == "short" and side_raw in {"SELL", "SHORT"})
                else "exit"
            )
            side = (
                position_side
                if event == "entry"
                else ("short" if position_side == "long" else "long")
            )
        else:
            event = "entry"
            side = "long" if side_raw in {"BUY", "LONG"} else "short"
        _append(
            out,
            seen,
            scope="trend",
            source="orders",
            key=str(row.get("order_id")),
            symbol=sym,
            event=event,
            side=side,
            price=_f(row.get("average_price")),
            qty=filled_qty or None,
            strategy=str(row.get("strategy_id") or "unknown").lower(),
            status="filled" if is_filled else "pending",
            extra={"time": ft, "order_id": row.get("order_id")},
        )

    out.sort(key=lambda m: m["time"])
    return out


def spot_markers(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    include_open_orders: bool = False,
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    # Push time range into SQL — spot_orders.created_at is TEXT ISO 8601.
    # When include_open_orders=True, pending orders outside the chart window
    # must still be returned; skip SQL time pushdown so they aren't filtered out.
    if include_open_orders:
        time_clause, time_params = "", []
    else:
        time_clause, time_params = _sql_time_range(start_ts, end_ts, col="created_at")
    sql = f"""
        SELECT order_id, created_at, updated_at, symbol, side, order_type,
               quantity, price, status, filled_quantity, filled_quote_usdt
        FROM spot_orders
        WHERE symbol = ?{time_clause}
        ORDER BY created_at ASC
        LIMIT {_MARKER_QUERY_LIMIT}
    """
    for row in query_rows(db_path, sql, (sym, *time_params)):
        status = str(row.get("status") or "").lower()
        filled_qty = _f(row.get("filled_quantity")) or 0.0
        if status not in {"filled", "closed", "partially_filled"} and filled_qty <= 0:
            if status not in _OPEN_ORDER_STATUSES:
                continue
        ts = _parse_ts(row.get("updated_at")) or _parse_ts(row.get("created_at"))
        if ts is None:
            continue
        is_filled = status in {"filled", "closed"} or filled_qty > 0
        if is_filled:
            if not _ts_in_chart_window(
                ts, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
            ):
                continue
        elif not include_open_orders:
            if not _ts_in_chart_window(
                ts, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
            ):
                continue
        elif since_ts is not None and ts <= since_ts:
            continue
        side = str(row.get("side") or "buy").lower()
        event = "entry" if side == "buy" else "exit"
        _append(
            out,
            seen,
            scope="spot",
            source="spot_orders",
            key=str(row.get("order_id")),
            symbol=sym,
            event=event,
            side="long" if side == "buy" else "short",
            price=_f(row.get("price")),
            qty=filled_qty or _f(row.get("quantity")),
            strategy="spot_accum_simple",
            status="filled" if is_filled else "pending",
            extra={"time": ts, "order_type": row.get("order_type")},
        )
    out.sort(key=lambda m: m["time"])
    return out


def multi_leg_markers(
    db_path: Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    include_open_orders: bool = False,
    engine_data_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    if engine_data_root is not None:
        from mlbot_console.services.orders_list import fetch_multileg_raw_rows

        raw_rows = fetch_multileg_raw_rows(
            db_path,
            sym,
            engine_data_root=engine_data_root,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    else:
        ml_time_clause, ml_time_params = _sql_time_range_expr(
            start_ts, end_ts, "COALESCE(filled_at, created_at)"
        )
        ord_sql = f"""
            SELECT local_order_id, strategy, symbol, side, purpose, status, order_type,
                   filled_quantity, average_price, filled_at, created_at, price, quantity,
                   stop_price, leg_id
            FROM multi_leg_orders
            WHERE symbol = ?{ml_time_clause}
            ORDER BY COALESCE(filled_at, created_at) ASC
            LIMIT {_MARKER_QUERY_LIMIT}
        """
        raw_rows = query_rows(db_path, ord_sql, (sym, *ml_time_params))
        for row in raw_rows:
            row["order_id"] = row.get("local_order_id")
    for row in raw_rows:
        status = str(row.get("status") or "").lower()
        filled_qty = _f(row.get("filled_quantity")) or 0.0
        is_filled = status in {"filled", "closed"} or filled_qty > 0
        if not is_filled:
            if not include_open_orders:
                continue
            if status not in _OPEN_ORDER_STATUSES:
                continue
        ts = _parse_ts(row.get("filled_at")) or _parse_ts(row.get("created_at"))
        if ts is None:
            continue
        if not _ts_in_chart_window(
            ts, start_ts=start_ts, end_ts=end_ts, since_ts=since_ts
        ):
            continue
        if since_ts is not None and not is_filled and ts <= since_ts:
            continue
        purpose = str(row.get("purpose") or "")
        order_type = str(row.get("order_type") or "")
        local_oid = str(row.get("order_id") or row.get("local_order_id") or "")
        if purpose == "inventory":
            event = "entry"
        else:
            event = _multi_leg_event(
                purpose,
                order_type,
                local_order_id=local_oid,
                is_filled=is_filled,
            )
        side_raw = str(row.get("side") or "").upper()
        side = "long" if side_raw in {"BUY", "LONG"} else "short"
        strat = str(row.get("strategy") or "multi_leg").lower()
        tp_price = _multi_leg_take_profit_price(row)
        leg_label = _leg_label_from_order_id(local_oid) or _leg_label_from_order_id(
            str(row.get("leg_id") or "")
        )
        action_reason = _action_reason_from_row(row)
        extra: Dict[str, Any] = {
            "time": ts,
            "purpose": purpose,
            "order_type": order_type,
            "order_status": status,
            "local_order_id": row.get("local_order_id"),
            "leg_id": row.get("leg_id"),
            "leg_label": leg_label,
            "take_profit_price": tp_price,
            "stop_price": _f(row.get("stop_price")),
        }
        if action_reason:
            extra["exit_reason"] = action_reason
            if event == "exit" and "regime" in action_reason.lower():
                extra["exit_kind"] = "regime_or_risk_exit"
        marker_color = (
            CHOP_GRID_REGIME_EXIT_COLOR
            if event == "exit" and extra.get("exit_kind") == "regime_or_risk_exit"
            else None
        )
        _append(
            out,
            seen,
            scope="multi_leg",
            source="multi_leg_orders",
            key=local_oid or str(row.get("local_order_id") or ""),
            symbol=sym,
            event=event,
            side=side,
            price=_f(row.get("average_price")) or _f(row.get("price")),
            qty=filled_qty or _f(row.get("quantity")),
            strategy=strat,
            status="filled" if is_filled else "pending",
            extra=extra,
            color=marker_color,
        )

    rep_time_clause, rep_time_params = _sql_time_range(
        start_ts, end_ts, col="event_time"
    )
    rep_sql = f"""
        SELECT event_id, strategy, symbol, status, execution_type, event_time, order_id
        FROM multi_leg_execution_reports
        WHERE symbol = ?{rep_time_clause}
        ORDER BY event_time ASC
        LIMIT {_MARKER_QUERY_LIMIT}
    """
    for row in query_rows(db_path, rep_sql, (sym, *rep_time_params)):
        st = str(row.get("status") or "").upper()
        if st not in {"FILLED", "PARTIALLY_FILLED"}:
            continue
        ts = _parse_ts(row.get("event_time"))
        if ts is None:
            continue
        if since_ts is not None and ts <= since_ts:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        ex_type = str(row.get("execution_type") or "").upper()
        event = "exit" if "CLOSE" in ex_type or ex_type == "REDUCE" else "entry"
        strat = str(row.get("strategy") or "multi_leg").lower()
        side_raw = "SELL" if event == "exit" else "BUY"
        side = "long" if side_raw in {"BUY", "LONG"} else "short"
        _append(
            out,
            seen,
            scope="multi_leg",
            source="multi_leg_execution_reports",
            key=str(row.get("event_id")),
            symbol=sym,
            event=event,
            side=side,
            price=None,
            strategy=strat,
            status="filled",
            extra={
                "time": ts,
                "order_id": row.get("order_id"),
                "execution_type": ex_type,
            },
        )
    out.sort(key=lambda m: m["time"])
    return out


def _filter_pending(
    markers: List[Dict[str, Any]], include_pending: bool
) -> List[Dict[str, Any]]:
    if include_pending:
        return markers
    return [m for m in markers if str(m.get("status") or "filled").lower() != "pending"]


def _nearest_candle_time(candle_times: List[int], t: int) -> int:
    return min(candle_times, key=lambda x: abs(int(x) - t))


def align_markers_to_candles(
    markers: List[Dict[str, Any]],
    candle_times: List[int],
) -> List[Dict[str, Any]]:
    """Pin markers onto chart bar times (LWC only renders markers on series times)."""
    if not candle_times:
        return markers
    times = sorted(int(t) for t in candle_times)
    times_set = set(times)
    first_t, last_t = times[0], times[-1]
    out: List[Dict[str, Any]] = []
    for m in markers:
        item = dict(m)
        t = int(item["time"])
        pending = str(item.get("status") or "filled").lower() == "pending"
        if pending:
            if t < first_t:
                snapped = first_t
            elif t > last_t:
                snapped = last_t
            else:
                snapped = _nearest_candle_time(times, t)
            item["time"] = snapped
            out.append(item)
            continue
        if t < first_t:
            snapped = first_t
        elif t > last_t:
            snapped = last_t
        else:
            snapped = _nearest_candle_time(times, t)
        if snapped != t or t not in times_set:
            detail = dict(item.get("detail") or {})
            if "order_time" not in detail:
                detail["order_time"] = t
            item["detail"] = detail
        item["time"] = snapped
        out.append(item)
    return out


def align_pending_markers_to_candles(
    markers: List[Dict[str, Any]],
    candle_times: List[int],
) -> List[Dict[str, Any]]:
    """Backward-compatible alias."""
    return align_markers_to_candles(markers, candle_times)


def marker_scope_counts(markers: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"trend": 0, "spot": 0, "multi_leg": 0}
    for m in markers:
        scope = str(m.get("scope") or "").lower()
        if scope in counts:
            counts[scope] += 1
    counts["total"] = len(markers)
    return counts


def collect_markers(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    symbol: str,
    scopes: List[str],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    since_ts: Optional[int] = None,
    include_pending: bool = False,
    engine_data_root: Optional[Path] = None,
    strategies: Optional[List[str]] = None,
    feature_bus_root: Optional[Path] = None,
    strategies_root: Optional[Path] = None,
    map_timeframe: str = "2h",
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    scope_set = {s.strip().lower() for s in scopes if s.strip()}
    open_orders = bool(include_pending)
    if "trend" in scope_set and trend_db.is_file():
        merged.extend(
            trend_markers(
                trend_db,
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                since_ts=since_ts,
                include_open_orders=open_orders,
            )
        )
    if "spot" in scope_set and spot_db.is_file():
        merged.extend(
            spot_markers(
                spot_db,
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                since_ts=since_ts,
                include_open_orders=open_orders,
            )
        )
    if "multi_leg" in scope_set and multi_leg_db.is_file():
        merged.extend(
            multi_leg_markers(
                multi_leg_db,
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                since_ts=since_ts,
                include_open_orders=open_orders,
                engine_data_root=engine_data_root,
            )
        )
    if (
        "multi_leg" in scope_set
        and feature_bus_root is not None
        and strategies_root is not None
        and feature_bus_root.is_dir()
        and strategies_root.is_dir()
    ):
        try:
            from mlbot_console.services.strategy_stage_regions import (
                load_chop_grid_regime_exit_markers,
            )

            regime_exits = load_chop_grid_regime_exit_markers(
                feature_bus_root,
                symbol,
                map_timeframe,
                strategies_root,
                start=_pandas_ts_from_unix(start_ts),
                end=_pandas_ts_from_unix(end_ts),
            )
            if strategies:
                allowed = {str(s).strip().lower() for s in strategies if str(s).strip()}
                if allowed:
                    regime_exits = [
                        m
                        for m in regime_exits
                        if str(m.get("strategy") or "").lower() in allowed
                    ]
            merged = _merge_chop_regime_exit_markers(merged, regime_exits)
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "chop_grid regime exit markers skipped", exc_info=True
            )
    merged.sort(key=lambda m: m["time"])
    merged = _filter_pending(merged, include_pending)
    if strategies:
        allowed = {str(s).strip().lower() for s in strategies if str(s).strip()}
        if allowed:
            merged = [
                m
                for m in merged
                if str(m.get("strategy") or "").lower() in allowed
            ]
    return merged


def _pandas_ts_from_unix(ts: Optional[int]) -> Any:
    if ts is None:
        return None
    import pandas as pd

    return pd.Timestamp(int(ts), unit="s", tz="UTC")


def _f(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        v = float(raw)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None
