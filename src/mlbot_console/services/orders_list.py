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
    for key in ("filled_at", "updated_at", "created_at", "operation_time"):
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
    source = {
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


def trend_orders(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    if _is_all_symbols(symbol):
        sql = """
            SELECT order_id, symbol, side, status, order_type, quantity, price,
                   filled_quantity, average_price, created_at, updated_at, filled_at,
                   position_id
            FROM orders
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (int(limit),))
    else:
        sym = symbol.upper()
        sql = """
            SELECT order_id, symbol, side, status, order_type, quantity, price,
                   filled_quantity, average_price, created_at, updated_at, filled_at,
                   position_id
            FROM orders
            WHERE symbol = ?
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (sym, int(limit)))
    out = [_normalize("trend", r) for r in rows]
    if status:
        st = status.lower()
        out = [r for r in out if r["status"] == st]
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
            trend_orders(trend_db, symbol, status=status, limit=per_scope)
        )
    if "spot" in scope_set and spot_db.is_file():
        merged.extend(
            spot_orders_list(spot_db, symbol, status=status, limit=per_scope)
        )
    if "multi_leg" in scope_set and multi_leg_db.is_file():
        merged.extend(
            multi_leg_orders_list(multi_leg_db, symbol, status=status, limit=per_scope)
        )
    merged.sort(key=lambda r: r.get("time") or 0, reverse=True)
    merged = _exclude_statuses(merged, exclude_statuses)
    return merged[: int(limit)]
