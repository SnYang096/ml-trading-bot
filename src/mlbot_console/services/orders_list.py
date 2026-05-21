"""Read-only order list queries (trend / spot / multi-leg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services.db import query_rows
from mlbot_console.services.multileg_order_links import (
    enrich_multileg_rows_for_symbol,
    resolve_take_profit_display,
)
from mlbot_console.services.trade_markers import _marker_id, _parse_ts

_ALL_SYMBOLS = frozenset({"", "*", "ALL", "__ALL__"})


def _is_all_symbols(symbol: str) -> bool:
    return str(symbol or "").strip().upper() in _ALL_SYMBOLS


def _row_time(row: Dict[str, Any]) -> int:
    for key in (
        "filled_at",
        "updated_at",
        "created_at",
        "operation_time",
        "entry_time",
        "exit_time",
    ):
        ts = _parse_ts(row.get(key))
        if ts is not None:
            return ts
    return 0


def _exclude_statuses(
    rows: List[Dict[str, Any]], exclude: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if not exclude:
        return rows
    blocked = {str(s).strip().lower() for s in exclude if str(s).strip()}
    if not blocked:
        return rows
    return [r for r in rows if str(r.get("status") or "").lower() not in blocked]


def _first_positive_price(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num == num and num > 0:
            return num
    return None


def _stop_loss_hint(row: Dict[str, Any]) -> str:
    order_type = str(row.get("order_type") or "").lower()
    status = str(row.get("status") or "").lower()
    if "stop" not in order_type:
        return ""
    if status == "rejected":
        return "挂单失败"
    if status in {"pending", "open", "new", "submitted"}:
        return "挂单中"
    if status in {"filled", "closed"}:
        return "已成交"
    return ""


def _normalize(
    scope: str,
    row: Dict[str, Any],
    *,
    id_field: str = "order_id",
) -> Dict[str, Any]:
    oid = str(row.get(id_field) or "")
    sym = str(row.get("symbol") or "").upper()
    status = str(row.get("status") or "").lower()
    side = str(row.get("side") or "")
    filled_qty = float(row.get("filled_quantity") or 0)
    t = _row_time(row)
    source = str(row.get("_marker_source") or "").strip() or {
        "trend": "orders",
        "spot": "spot_orders",
        "multi_leg": "multi_leg_orders",
    }.get(scope, "orders")
    marker_key = oid
    item = {
        "scope": scope,
        "order_id": oid,
        "symbol": sym,
        "side": side,
        "status": status,
        "order_type": row.get("order_type") or row.get("purpose"),
        "purpose": row.get("purpose"),
        "quantity": row.get("quantity"),
        "price": row.get("price") or row.get("average_price"),
        "filled_quantity": filled_qty,
        "average_price": row.get("average_price"),
        "stop_price": row.get("stop_price"),
        "stop_loss_price": _first_positive_price(
            row.get("stop_price")
            if "stop" in str(row.get("order_type") or "").lower()
            else None,
            row.get("stop_loss_price"),
        ),
        "take_profit_price": _first_positive_price(row.get("take_profit_price")),
        "stop_loss_hint": _stop_loss_hint(row),
        "created_at": row.get("created_at"),
        "filled_at": row.get("filled_at"),
        "updated_at": row.get("updated_at"),
        "strategy": row.get("strategy") or row.get("strategy_id"),
        "leg_id": row.get("leg_id"),
        "time": t,
        "marker_id": _marker_id(scope, source, marker_key) if oid else None,
    }
    if scope == "multi_leg":
        tp_px, tp_hint = resolve_take_profit_display(row)
        item["take_profit_price"] = tp_px
        item["take_profit_hint"] = tp_hint
        if row.get("_link_exit_price") is not None:
            item["exit_price"] = row.get("_link_exit_price")
            item["exit_order_id"] = row.get("_link_exit_leg")
    return item


def _position_action_side(position_side: str, event: str) -> str:
    side = str(position_side or "long").lower()
    ev = str(event or "entry").lower()
    if ev == "exit":
        return "sell" if side == "long" else "buy"
    return "buy" if side == "long" else "sell"


def _trend_position_event_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        pid = str(row.get("position_id") or "")
        sym = str(row.get("symbol") or "").upper()
        pos_side = str(row.get("side") or "long").lower()
        strat = row.get("strategy_id")
        entry_ts = _parse_ts(row.get("entry_time"))
        if entry_ts is not None:
            out.append(
                _normalize(
                    "trend",
                    {
                        "order_id": f"{pid}:entry",
                        "symbol": sym,
                        "side": _position_action_side(pos_side, "entry"),
                        "status": "filled",
                        "order_type": "position_entry",
                        "quantity": None,
                        "price": row.get("entry_price"),
                        "average_price": row.get("entry_price"),
                        "filled_quantity": None,
                        "created_at": row.get("entry_time"),
                        "strategy_id": strat,
                        "stop_loss_price": row.get("stop_loss_price"),
                        "take_profit_price": row.get("take_profit_price"),
                        "_marker_source": "positions",
                    },
                )
            )
        exit_ts = _parse_ts(row.get("exit_time"))
        if exit_ts is not None:
            out.append(
                _normalize(
                    "trend",
                    {
                        "order_id": f"{pid}:exit",
                        "symbol": sym,
                        "side": _position_action_side(pos_side, "exit"),
                        "status": "closed",
                        "order_type": "position_exit",
                        "quantity": None,
                        "price": row.get("exit_price"),
                        "average_price": row.get("exit_price"),
                        "filled_quantity": None,
                        "created_at": row.get("exit_time"),
                        "strategy_id": strat,
                        "_marker_source": "positions",
                    },
                )
            )
    return out


def _trend_operation_rows(
    db_path: Path, symbol: str, limit: int
) -> List[Dict[str, Any]]:
    where = "" if _is_all_symbols(symbol) else "WHERE p.symbol = ?"
    params: tuple[Any, ...] = (
        (int(limit),) if _is_all_symbols(symbol) else (symbol.upper(), int(limit))
    )
    sql = f"""
        SELECT po.operation_id, po.position_id, po.operation_type,
               po.operation_time, po.size, po.price, po.reason,
               po.stop_loss_price, po.take_profit_price,
               p.symbol, p.side, p.strategy_id
        FROM position_operations po
        JOIN positions p ON p.position_id = po.position_id
        {where}
        ORDER BY po.operation_time DESC
        LIMIT ?
    """
    out: List[Dict[str, Any]] = []
    for row in query_rows(db_path, sql, params):
        op_type = str(row.get("operation_type") or "").lower()
        event = (
            "exit"
            if any(x in op_type for x in ("close", "reduce", "exit"))
            else "entry"
        )
        out.append(
            _normalize(
                "trend",
                {
                    "order_id": str(row.get("operation_id") or ""),
                    "symbol": row.get("symbol"),
                    "side": _position_action_side(
                        str(row.get("side") or "long"), event
                    ),
                    "status": "filled",
                    "order_type": f"position_{op_type or 'operation'}",
                    "quantity": row.get("size"),
                    "price": row.get("price"),
                    "average_price": row.get("price"),
                    "filled_quantity": row.get("size"),
                    "created_at": row.get("operation_time"),
                    "operation_time": row.get("operation_time"),
                    "strategy_id": row.get("strategy_id"),
                    "stop_loss_price": row.get("stop_loss_price"),
                    "take_profit_price": row.get("take_profit_price"),
                    "_marker_source": "position_operations",
                },
            )
        )
    return out


def _sql_excluded_status_clause(
    excluded: Optional[List[str]], *, alias: str = "o"
) -> tuple[str, tuple[Any, ...]]:
    """Build NOT IN filter so rejected/pending noise does not fill the row limit."""
    blocked = {str(s).strip().lower() for s in (excluded or []) if str(s).strip()}
    if not blocked:
        return "", ()
    placeholders = ",".join("?" for _ in blocked)
    return f" AND lower({alias}.status) NOT IN ({placeholders})", tuple(blocked)


def trend_orders(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    exclude_statuses: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    status_filter = str(status or "").strip().lower()
    excluded = list(exclude_statuses or [])
    if status_filter:
        excluded = [s for s in excluded if s.lower() != status_filter]
    status_clause, status_params = _sql_excluded_status_clause(excluded, alias="o")
    if _is_all_symbols(symbol):
        sql = f"""
            SELECT o.order_id, o.symbol AS symbol, o.side AS side, o.status, o.order_type,
                   o.quantity, o.price, o.stop_price, o.filled_quantity, o.average_price,
                   o.created_at, o.updated_at, o.filled_at,
                   o.position_id, p.strategy_id,
                   p.stop_loss_price, p.take_profit_price
            FROM orders o
            LEFT JOIN positions p ON p.position_id = o.position_id
            WHERE 1=1{status_clause}
            ORDER BY COALESCE(o.filled_at, o.created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (*status_params, int(limit)))
    else:
        sym = symbol.upper()
        sql = f"""
            SELECT o.order_id, o.symbol AS symbol, o.side AS side, o.status, o.order_type,
                   o.quantity, o.price, o.stop_price, o.filled_quantity, o.average_price,
                   o.created_at, o.updated_at, o.filled_at,
                   o.position_id, p.strategy_id,
                   p.stop_loss_price, p.take_profit_price
            FROM orders o
            LEFT JOIN positions p ON p.position_id = o.position_id
            WHERE o.symbol = ?{status_clause}
            ORDER BY COALESCE(o.filled_at, o.created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (sym, *status_params, int(limit)))
    pos_sql = """
        SELECT position_id, symbol, side, entry_time, exit_time,
               entry_price, exit_price, realized_pnl, status, strategy_id,
               stop_loss_price, take_profit_price
        FROM positions
    """
    pos_params: tuple[Any, ...] = ()
    if not _is_all_symbols(symbol):
        pos_sql += " WHERE symbol = ?"
        pos_params = (symbol.upper(),)
    pos_sql += " ORDER BY COALESCE(exit_time, entry_time) DESC LIMIT ?"
    pos_rows = query_rows(db_path, pos_sql, (*pos_params, int(limit)))
    out = [_normalize("trend", r) for r in rows]
    out.extend(_trend_position_event_rows(pos_rows))
    out.extend(_trend_operation_rows(db_path, symbol, int(limit)))
    if status:
        st = status.lower()
        out = [r for r in out if r["status"] == st]
    out.sort(key=lambda r: r.get("time") or 0, reverse=True)
    out = out[: int(limit)]
    return out


def spot_orders_list(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    if _is_all_symbols(symbol):
        sql = """
            SELECT order_id, symbol, side, status, order_type, quantity, price,
                   filled_quantity, filled_quote_usdt, created_at, updated_at
            FROM spot_orders
            ORDER BY COALESCE(updated_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (int(limit),))
    else:
        sym = symbol.upper()
        sql = """
            SELECT order_id, symbol, side, status, order_type, quantity, price,
                   filled_quantity, filled_quote_usdt, created_at, updated_at
            FROM spot_orders
            WHERE symbol = ?
            ORDER BY COALESCE(updated_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (sym, int(limit)))
    out = [_normalize("spot", r) for r in rows]
    if status:
        st = status.lower()
        out = [r for r in out if r["status"] == st]
    return out


def multi_leg_orders_list(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    if _is_all_symbols(symbol):
        sql = """
            SELECT local_order_id AS order_id, symbol, side, status, order_type, purpose,
                   quantity, price, stop_price, filled_quantity, average_price, created_at,
                   filled_at, strategy, leg_id
            FROM multi_leg_orders
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (int(limit),))
    else:
        sym = symbol.upper()
        sql = """
            SELECT local_order_id AS order_id, symbol, side, status, order_type, purpose,
                   quantity, price, stop_price, filled_quantity, average_price, created_at,
                   filled_at, strategy, leg_id
            FROM multi_leg_orders
            WHERE symbol = ?
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (sym, int(limit)))
    enrich_multileg_rows_for_symbol(db_path, symbol, rows)
    out = []
    for r in rows:
        item = _normalize("multi_leg", r)
        if r.get("purpose"):
            item["order_type"] = r.get("purpose")
        out.append(item)
    if status:
        st = status.lower()
        out = [r for r in out if r["status"] == st]
    return out


def collect_orders(
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    symbol: str,
    scopes: List[str],
    status: Optional[str] = None,
    exclude_statuses: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    scope_set = {s.strip().lower() for s in scopes if s.strip()}
    per_scope = max(int(limit), 1)
    if "trend" in scope_set and trend_db.is_file():
        merged.extend(
            trend_orders(
                trend_db,
                symbol,
                status=status,
                exclude_statuses=exclude_statuses,
                limit=per_scope,
            )
        )
    if "spot" in scope_set and spot_db.is_file():
        merged.extend(spot_orders_list(spot_db, symbol, status=status, limit=per_scope))
    if "multi_leg" in scope_set and multi_leg_db.is_file():
        merged.extend(
            multi_leg_orders_list(multi_leg_db, symbol, status=status, limit=per_scope)
        )
    merged.sort(key=lambda r: r.get("time") or 0, reverse=True)
    merged = _exclude_statuses(merged, exclude_statuses)
    return merged[: int(limit)]
