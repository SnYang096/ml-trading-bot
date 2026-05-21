"""Extract live trade markers from read-only SQLite stores."""

from __future__ import annotations

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
        "color": STRATEGY_COLORS.get(strat, "#aaaaaa"),
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

    pos_sql = """
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id
        FROM positions
        WHERE symbol = ?
        ORDER BY entry_time ASC
    """
    for row in query_rows(db_path, pos_sql, (sym,)):
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

    op_sql = """
        SELECT po.operation_id, po.position_id, po.operation_type,
               po.operation_time, po.size, po.price, po.reason
        FROM position_operations po
        JOIN positions p ON p.position_id = po.position_id
        WHERE p.symbol = ?
        ORDER BY po.operation_time ASC
    """
    for row in query_rows(db_path, op_sql, (sym,)):
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
        _append(
            out,
            seen,
            scope="trend",
            source="position_operations",
            key=str(row.get("operation_id")),
            symbol=sym,
            event=event,
            side="long",
            price=_f(row.get("price")),
            qty=_f(row.get("size")),
            strategy="unknown",
            is_add=is_add,
            extra={
                "time": ot,
                "operation_type": op_type,
                "position_id": row.get("position_id"),
            },
        )

    ord_sql = """
        SELECT order_id, symbol, side, status, filled_at, created_at,
               average_price, filled_quantity, position_id
        FROM orders
        WHERE symbol = ?
        ORDER BY COALESCE(filled_at, created_at) ASC
    """
    for row in query_rows(db_path, ord_sql, (sym,)):
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
        side = "long" if side_raw in {"BUY", "LONG"} else "short"
        _append(
            out,
            seen,
            scope="trend",
            source="orders",
            key=str(row.get("order_id")),
            symbol=sym,
            event="entry",
            side=side,
            price=_f(row.get("average_price")),
            qty=filled_qty or None,
            strategy="unknown",
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
    sql = """
        SELECT order_id, created_at, updated_at, symbol, side, order_type,
               quantity, price, status, filled_quantity, filled_quote_usdt
        FROM spot_orders
        WHERE symbol = ?
        ORDER BY created_at ASC
    """
    for row in query_rows(db_path, sql, (sym,)):
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
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    ord_sql = """
        SELECT local_order_id, strategy, symbol, side, purpose, status, order_type,
               filled_quantity, average_price, filled_at, created_at, price, quantity,
               stop_price, leg_id
        FROM multi_leg_orders
        WHERE symbol = ?
        ORDER BY COALESCE(filled_at, created_at) ASC
    """
    for row in query_rows(db_path, ord_sql, (sym,)):
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
        local_oid = str(row.get("local_order_id") or "")
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
        _append(
            out,
            seen,
            scope="multi_leg",
            source="multi_leg_orders",
            key=str(row.get("local_order_id")),
            symbol=sym,
            event=event,
            side=side,
            price=_f(row.get("average_price")) or _f(row.get("price")),
            qty=filled_qty or _f(row.get("quantity")),
            strategy=strat,
            status="filled" if is_filled else "pending",
            extra={
                "time": ts,
                "purpose": purpose,
                "order_type": order_type,
                "order_status": status,
                "local_order_id": row.get("local_order_id"),
                "leg_id": row.get("leg_id"),
                "leg_label": leg_label,
                "take_profit_price": tp_price,
                "stop_price": _f(row.get("stop_price")),
            },
        )

    rep_sql = """
        SELECT event_id, strategy, symbol, status, execution_type, event_time, order_id
        FROM multi_leg_execution_reports
        WHERE symbol = ?
        ORDER BY event_time ASC
    """
    for row in query_rows(db_path, rep_sql, (sym,)):
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


def _filter_pending(markers: List[Dict[str, Any]], include_pending: bool) -> List[Dict[str, Any]]:
    if include_pending:
        return markers
    return [m for m in markers if str(m.get("status") or "filled").lower() != "pending"]


def align_pending_markers_to_candles(
    markers: List[Dict[str, Any]],
    candle_times: List[int],
) -> List[Dict[str, Any]]:
    """Pin pending markers to visible bars (LWC ignores markers outside series range)."""
    if not candle_times:
        return markers
    first_t = int(candle_times[0])
    last_t = int(candle_times[-1])
    out: List[Dict[str, Any]] = []
    for m in markers:
        if str(m.get("status") or "filled").lower() != "pending":
            out.append(m)
            continue
        item = dict(m)
        t = int(item["time"])
        if t < first_t or t > last_t:
            detail = dict(item.get("detail") or {})
            detail["order_time"] = t
            item["detail"] = detail
            item["time"] = last_t
        out.append(item)
    return out


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
            )
        )
    merged.sort(key=lambda m: m["time"])
    return _filter_pending(merged, include_pending)


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
