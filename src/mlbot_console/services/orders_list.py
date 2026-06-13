"""Read-only order list queries (trend / spot / multi-leg)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mlbot_console.services.db import query_rows, table_columns
from mlbot_console.services.multileg_order_links import (
    enrich_multileg_rows_for_symbol,
    entry_link_id,
    hydrate_multileg_fill_fields,
    is_entry_row,
    leg_index,
    leg_suffix,
    resolve_take_profit_display,
    row_group_key,
)
from mlbot_console.services.multileg_repair_tp import (
    is_repair_tp_row,
    repair_display_leg_label,
)
from mlbot_console.services.trade_markers import _marker_id, _parse_ts

from mlbot_console.services.symbols import is_all_symbols


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


def _resolve_display_price(
    item: Dict[str, Any], row: Dict[str, Any]
) -> Optional[float]:
    """Fill/limit/trigger price for console tables (algo SL/TP store trigger in stop_price)."""
    status = str(row.get("status") or item.get("status") or "").lower()
    otype = str(
        row.get("order_type") or row.get("purpose") or item.get("order_type") or ""
    ).lower()
    is_stop_like = "stop" in otype or "take_profit" in otype
    unfilled = status in {
        "open",
        "new",
        "pending",
        "submitted",
        "partially_filled",
        "rejected",
        "canceled",
        "cancelled",
        "expired",
    }
    if is_stop_like and unfilled:
        return _first_positive_price(
            item.get("stop_price"),
            item.get("stop_loss_price"),
            item.get("take_profit_price"),
            item.get("average_price"),
            item.get("price"),
            item.get("exit_price"),
            row.get("exit_price"),
        )
    return _first_positive_price(
        item.get("average_price"),
        item.get("price"),
        item.get("stop_price"),
        item.get("stop_loss_price"),
        item.get("take_profit_price"),
        item.get("exit_price"),
        row.get("exit_price"),
    )


def _resolve_order_position_side(scope: str, row: Dict[str, Any]) -> Optional[str]:
    """Hedge position side (LONG/SHORT) for readable open/close labels."""
    from mlbot_console.services.multileg_order_links import (
        leg_side_kind,
        trend_entry_position_side,
    )

    raw = str(row.get("position_side") or "").upper()
    if raw in {"LONG", "SHORT"}:
        return raw
    pos_side = str(row.get("pos_side") or "").lower()
    if pos_side in {"long", "short"}:
        return pos_side.upper()
    if scope == "multi_leg":
        te = trend_entry_position_side(row)
        if te:
            return te
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        kind = leg_side_kind(oid)
        if kind == "L":
            return "LONG"
        if kind == "S":
            return "SHORT"
        side = str(row.get("side") or "").upper()
        purpose = str(row.get("purpose") or row.get("order_type") or "").lower()
        if side in {"LONG", "SHORT"} and (
            "entry" in purpose
            or "place" in purpose
            or purpose in {"", "limit", "market"}
        ):
            return side
    if scope == "trend":
        te = trend_entry_position_side(row)
        if te:
            return te
        ot = str(row.get("order_type") or "").lower()
        if ot in {"position_entry", "position_exit"}:
            side = str(row.get("side") or "").lower()
            if side in {"buy", "long"}:
                return "LONG"
            if side in {"sell", "short"}:
                return "SHORT"
    return None


def _resolve_order_is_closing(
    scope: str, row: Dict[str, Any], position_side: Optional[str]
) -> bool:
    """Whether the order closes/reduces an existing position."""
    if row.get("is_closing") is True:
        return True
    if row.get("is_closing") is False:
        return False
    purpose = str(row.get("purpose") or row.get("order_type") or "").lower()
    oid = str(row.get("order_id") or row.get("local_order_id") or "").lower()
    if (
        "take_profit" in purpose
        or "stop_loss" in purpose
        or "market_exit" in purpose
        or "position_exit" in purpose
    ):
        return True
    if "entry" in purpose or "position_entry" in purpose or purpose == "place":
        return False
    if oid.endswith(":exit") or oid.endswith("_tp") or "_sl" in oid:
        return True
    ps = str(position_side or "").upper()
    side = str(row.get("side") or "").upper()
    if ps == "LONG" and side == "SELL":
        return True
    if ps == "SHORT" and side == "BUY":
        return True
    return False


def _resolve_filled_quantity(row: Dict[str, Any], status: str) -> float:
    """Prefer filled_quantity; for closed rows fall back to order quantity."""
    filled = float(row.get("filled_quantity") or 0)
    if filled > 0:
        return filled
    qty_raw = row.get("quantity")
    if qty_raw is None:
        return filled
    try:
        qty = float(qty_raw)
    except (TypeError, ValueError):
        return filled
    if qty > 0 and status in {"filled", "closed"}:
        return qty
    return filled


def _trend_entry_qty_by_position(
    db_path: Path, symbol: Optional[str]
) -> Dict[str, float]:
    """Entry size per position_id from filled trend orders (for closed positions)."""
    if not db_path.is_file():
        return {}
    sym_clause = " AND o.symbol = ?" if symbol else ""
    params: tuple[Any, ...] = (symbol,) if symbol else ()
    out: Dict[str, float] = {}
    for row in query_rows(
        db_path,
        f"""
        SELECT o.position_id, o.side, o.filled_quantity, o.quantity, p.side AS pos_side
        FROM orders o
        INNER JOIN positions p ON p.position_id = o.position_id
        WHERE lower(o.status) = 'filled'{sym_clause}
        ORDER BY COALESCE(o.filled_at, o.created_at) ASC
        """,
        params,
    ):
        pid = str(row.get("position_id") or "")
        if not pid or pid in out:
            continue
        pos_side = str(row.get("pos_side") or "long").lower()
        o_side = str(row.get("side") or "").lower()
        is_entry = (
            o_side in {"buy", "long"}
            if pos_side == "long"
            else o_side in {"sell", "short"}
        )
        if not is_entry:
            continue
        qty = _resolve_filled_quantity(row, "filled")
        if qty > 0:
            out[pid] = qty
    return out


def _positions_select_from(db_path: Path) -> str:
    cols = table_columns(db_path, "positions")
    extra = [c for c in ("current_size", "initial_size") if c in cols]
    base = (
        "SELECT position_id, symbol, side, entry_time, exit_time, "
        "entry_price, exit_price, realized_pnl, status, strategy_id, "
        "stop_loss_price, take_profit_price"
    )
    if extra:
        base += ", " + ", ".join(extra)
    return base + "\n        FROM positions"


def _position_display_qty(
    row: Dict[str, Any], entry_qty_by_pid: Dict[str, float]
) -> Optional[float]:
    for key in ("current_size", "initial_size"):
        val = row.get(key)
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num > 0:
            return num
    pid = str(row.get("position_id") or "")
    if pid:
        cached = float(entry_qty_by_pid.get(pid) or 0)
        if cached > 0:
            return cached
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
    filled_qty = _resolve_filled_quantity(row, status)
    t = _row_time(row)
    source = str(row.get("_marker_source") or "").strip() or {
        "trend": "orders",
        "spot": "spot_orders",
        "multi_leg": "multi_leg_orders",
    }.get(scope, "orders")
    marker_key = oid
    position_side = _resolve_order_position_side(scope, row)
    is_closing = _resolve_order_is_closing(scope, row, position_side)
    item = {
        "scope": scope,
        "order_id": oid,
        "symbol": sym,
        "side": side,
        "position_side": position_side,
        "is_closing": is_closing,
        "status": status,
        "order_type": row.get("order_type") or row.get("purpose"),
        "purpose": row.get("purpose"),
        "quantity": row.get("quantity"),
        "price": row.get("price") or row.get("average_price"),
        "filled_quantity": filled_qty,
        "average_price": row.get("average_price"),
        "stop_price": row.get("stop_price"),
        "stop_loss_price": _first_positive_price(
            (
                row.get("stop_price")
                if "stop" in str(row.get("order_type") or "").lower()
                or "stop" in str(row.get("purpose") or "").lower()
                else None
            ),
            row.get("stop_loss_price"),
        ),
        "take_profit_price": _first_positive_price(row.get("take_profit_price")),
        "stop_loss_hint": _stop_loss_hint(row),
        "created_at": row.get("created_at"),
        "filled_at": row.get("filled_at"),
        "updated_at": row.get("updated_at"),
        "strategy": row.get("strategy") or row.get("strategy_id"),
        "strategy_id": row.get("strategy_id") or row.get("strategy"),
        "position_id": row.get("position_id"),
        "leg_id": row.get("leg_id"),
        "time": t,
        "marker_id": _marker_id(scope, source, marker_key) if oid else None,
        "pnl_usdt": row.get("pnl_usdt"),
        "realized_pnl": row.get("realized_pnl"),
        "unrealized_pnl": row.get("unrealized_pnl"),
        "pnl_hint": row.get("pnl_hint"),
    }
    if scope == "multi_leg":
        # 如果是 SL 行，只设置 stop_loss_price，不走 resolve_take_profit_display
        purpose = str(row.get("purpose") or "").lower()
        if "stop_loss" in purpose or oid.endswith("_sl") or "_sl_" in oid:
            item["stop_loss_price"] = _first_positive_price(
                row.get("stop_price"), row.get("price")
            )
            item["take_profit_price"] = None
            item["take_profit_hint"] = ""
        else:
            tp_px, tp_hint = resolve_take_profit_display(row)
            item["take_profit_price"] = tp_px
            item["take_profit_hint"] = tp_hint

        if row.get("_link_exit_price") is not None:
            item["exit_price"] = row.get("_link_exit_price")
            item["exit_order_id"] = row.get("_link_exit_leg")
        oid = str(row.get("order_id") or row.get("local_order_id") or "")
        lid = str(row.get("leg_id") or "")
        item["grid_batch"] = row_group_key(row)
        if is_repair_tp_row(row):
            item["leg_label"] = repair_display_leg_label(row)
        else:
            item["leg_label"] = leg_suffix(oid) or leg_suffix(lid)
        item["leg_index"] = leg_index(oid) or leg_index(lid)
        if is_repair_tp_row(row):
            item["is_repair_tp"] = True
        link_label = str(row.get("_link_tp_leg_label") or row.get("_link_tp_leg") or "")
        link_oid = str(row.get("_link_tp_order_id") or row.get("_link_exit_leg") or "")
        if link_label and not str(item.get("leg_label") or "").endswith("_tp"):
            item["linked_tp_leg_label"] = link_label
        if link_oid and not str(item.get("leg_label") or "").endswith("_tp"):
            item["linked_tp_order_id"] = link_oid
        if row.get("_link_tp_is_repair") or row.get("_link_exit_is_repair"):
            item["linked_tp_is_repair"] = True
    display = _resolve_display_price(item, row)
    if display is not None:
        item["display_price"] = display
    return item


def _position_action_side(position_side: str, event: str) -> str:
    side = str(position_side or "long").lower()
    ev = str(event or "entry").lower()
    if ev == "exit":
        return "sell" if side == "long" else "buy"
    return "buy" if side == "long" else "sell"


def _trend_position_event_rows(
    rows: List[Dict[str, Any]],
    *,
    entry_qty_by_pid: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    qty_map = entry_qty_by_pid or {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        pid = str(row.get("position_id") or "")
        sym = str(row.get("symbol") or "").upper()
        pos_side = str(row.get("side") or "long").lower()
        strat = row.get("strategy_id")
        pos_qty = _position_display_qty(row, qty_map)
        entry_ts = _parse_ts(row.get("entry_time"))
        if entry_ts is not None:
            out.append(
                _normalize(
                    "trend",
                    {
                        "order_id": f"{pid}:entry",
                        "symbol": sym,
                        "side": _position_action_side(pos_side, "entry"),
                        "pos_side": pos_side,
                        "is_closing": False,
                        "status": "filled",
                        "order_type": "position_entry",
                        "quantity": pos_qty,
                        "price": row.get("entry_price"),
                        "average_price": row.get("entry_price"),
                        "filled_quantity": pos_qty,
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
            from mlbot_console.services.account_summary import _trend_realized_pnl_usdt

            pnl = _trend_realized_pnl_usdt(row, entry_qty_by_pid=qty_map)
            out.append(
                _normalize(
                    "trend",
                    {
                        "order_id": f"{pid}:exit",
                        "symbol": sym,
                        "side": _position_action_side(pos_side, "exit"),
                        "pos_side": pos_side,
                        "is_closing": True,
                        "status": "closed",
                        "order_type": "position_exit",
                        "quantity": pos_qty,
                        "price": row.get("exit_price"),
                        "average_price": row.get("exit_price"),
                        "filled_quantity": pos_qty,
                        "created_at": row.get("exit_time"),
                        "strategy_id": strat,
                        "realized_pnl": pnl,
                        "pnl_usdt": pnl,
                        "pnl_hint": "已实现" if pnl is not None else None,
                        "_marker_source": "positions",
                    },
                )
            )
    return out


def _trend_operation_rows(
    db_path: Path, symbol: str, limit: int
) -> List[Dict[str, Any]]:
    where = "" if is_all_symbols(symbol) else "WHERE p.symbol = ?"
    params: tuple[Any, ...] = (
        (int(limit),) if is_all_symbols(symbol) else (symbol.upper(), int(limit))
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


def _enrich_trend_sl_tp(
    rows: List[Dict[str, Any]], pos_rows: List[Dict[str, Any]]
) -> None:
    """Fill stop/TP on exchange order rows when position join was missing."""
    by_pid = {
        str(p.get("position_id") or ""): p
        for p in pos_rows
        if str(p.get("position_id") or "").strip()
    }
    latest_by_sym_strat: Dict[tuple[str, str], Dict[str, Any]] = {}
    latest_by_sym: Dict[str, Dict[str, Any]] = {}
    for p in sorted(pos_rows, key=_row_time, reverse=True):
        sym_u = str(p.get("symbol") or "").upper()
        key = (sym_u, str(p.get("strategy_id") or ""))
        if key not in latest_by_sym_strat:
            latest_by_sym_strat[key] = p
        if sym_u not in latest_by_sym:
            latest_by_sym[sym_u] = p
    for item in rows:
        if str(item.get("scope") or "") != "trend":
            continue
        if item.get("stop_loss_price") is not None:
            continue
        pid = str(item.get("position_id") or "")
        pos = by_pid.get(pid) if pid else None
        if pos is None:
            sym_u = str(item.get("symbol") or "").upper()
            strat = str(item.get("strategy") or item.get("strategy_id") or "")
            pos = latest_by_sym_strat.get((sym_u, strat)) or latest_by_sym.get(sym_u)
        if not pos:
            continue
        sl = _first_positive_price(pos.get("stop_loss_price"))
        tp = _first_positive_price(pos.get("take_profit_price"))
        if sl is not None:
            item["stop_loss_price"] = sl
        if tp is not None and item.get("take_profit_price") is None:
            item["take_profit_price"] = tp


def _entry_leg_ids_in_rows(rows: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for row in rows:
        if not is_entry_row(row):
            continue
        eid = entry_link_id(row)
        if eid:
            out.add(eid)
    return out


def _query_open_multileg_positions(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    if is_all_symbols(symbol):
        sql = """
            SELECT leg_id, strategy, symbol, side, entry_price, quantity, status,
                   opened_at, updated_at
            FROM multi_leg_positions
            WHERE lower(trim(coalesce(status, ''))) = 'open'
            ORDER BY opened_at DESC
        """
        return query_rows(db_path, sql)
    sym = symbol.upper()
    sql = """
        SELECT leg_id, strategy, symbol, side, entry_price, quantity, status,
               opened_at, updated_at
        FROM multi_leg_positions
        WHERE symbol = ?
          AND lower(trim(coalesce(status, ''))) = 'open'
        ORDER BY opened_at DESC
    """
    return query_rows(db_path, sql, (sym,))


def _inventory_from_engine_state(
    engine_data_root: Path, symbol: str
) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    state_dir = engine_data_root / "multi_leg_live" / "state"
    if not state_dir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(state_dir.glob(f"chop_grid_{sym}.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        strategy = path.stem.rsplit("_", 1)[0] if "_" in path.stem else "chop_grid"
        for inv in data.get("inventory") or []:
            leg_id = str(inv.get("leg_id") or "").strip()
            if not leg_id:
                continue
            qty = float(inv.get("quantity") or 0)
            if qty <= 0:
                continue
            out.append(
                {
                    "leg_id": leg_id,
                    "strategy": strategy,
                    "symbol": sym,
                    "side": str(inv.get("side") or "").upper(),
                    "entry_price": float(inv.get("entry_price") or 0),
                    "quantity": qty,
                    "status": "open",
                    "opened_at": inv.get("entry_time"),
                }
            )
    return out


def _inventory_row_from_position(pos: Dict[str, Any]) -> Dict[str, Any]:
    leg_id = str(pos.get("leg_id") or "").strip()
    qty = float(pos.get("quantity") or 0)
    entry_px = float(pos.get("entry_price") or 0)
    side = str(pos.get("side") or "").upper()
    return {
        "order_id": leg_id,
        "local_order_id": leg_id,
        "leg_id": leg_id,
        "symbol": str(pos.get("symbol") or "").upper(),
        "side": side,
        "status": "filled",
        "order_type": "inventory_leg",
        "purpose": "inventory",
        "quantity": qty,
        "filled_quantity": qty,
        "price": entry_px,
        "average_price": entry_px,
        "strategy": str(pos.get("strategy") or "chop_grid"),
        "created_at": pos.get("opened_at") or pos.get("updated_at"),
        "filled_at": pos.get("opened_at") or pos.get("updated_at"),
        "_synthetic_inventory": True,
    }


def _query_repair_tp_orders(db_path: Path, symbol: str) -> List[Dict[str, Any]]:
    sym = symbol.upper()
    sql = """
        SELECT local_order_id AS order_id, symbol, side, status, order_type, purpose,
               quantity, price, stop_price, filled_quantity, average_price, created_at,
               filled_at, strategy, leg_id, client_order_id
        FROM multi_leg_orders
        WHERE symbol = ?
          AND (
            client_order_id LIKE 'cg_repair%'
            OR local_order_id LIKE 'cg_repair%'
          )
    """
    return query_rows(db_path, sql, (sym,))


def _supplement_multileg_repair_tp(
    db_path: Path,
    symbol: str,
    rows: List[Dict[str, Any]],
) -> None:
    if is_all_symbols(symbol) or not db_path.is_file():
        return
    known = {str(r.get("order_id") or "") for r in rows}
    for row in _query_repair_tp_orders(db_path, symbol):
        oid = str(row.get("order_id") or "")
        if oid and oid not in known:
            rows.append(row)
            known.add(oid)


def _supplement_multileg_inventory_entries(
    db_path: Path,
    symbol: str,
    rows: List[Dict[str, Any]],
    *,
    engine_data_root: Optional[Path] = None,
) -> None:
    """Add open inventory legs missing from multi_leg_orders (e.g. S1 when only S1_tp shows)."""
    if is_all_symbols(symbol):
        return
    covered = _entry_leg_ids_in_rows(rows)
    seen_legs: Set[str] = set(covered)
    candidates: List[Dict[str, Any]] = []
    if db_path.is_file():
        candidates.extend(_query_open_multileg_positions(db_path, symbol))
    if engine_data_root is not None:
        candidates.extend(_inventory_from_engine_state(engine_data_root, symbol))
    for pos in candidates:
        leg_id = str(pos.get("leg_id") or "").strip()
        if not leg_id or leg_id in seen_legs:
            continue
        seen_legs.add(leg_id)
        rows.append(_inventory_row_from_position(pos))


def _sort_orders_for_display(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep non-grid rows by time; cluster multi_leg chop_grid rows by grid_batch."""

    def _batch_sort_key(row: Dict[str, Any]) -> tuple:
        batch = str(row.get("grid_batch") or "")
        if batch:
            label = str(row.get("leg_label") or "")
            side_rank = 0 if label.upper().startswith("L") else 1
            tp_rank = 1 if label.endswith("_tp") else 0
            return (
                0,
                batch,
                side_rank,
                int(row.get("leg_index") or 0),
                tp_rank,
                label,
            )
        return (1, -(int(row.get("time") or 0)))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    standalone: List[Dict[str, Any]] = []
    for row in rows:
        batch = str(row.get("grid_batch") or "")
        if str(row.get("scope") or "") == "multi_leg" and batch:
            grouped.setdefault(batch, []).append(row)
        else:
            standalone.append(row)

    out: List[Dict[str, Any]] = []
    batch_keys = sorted(
        grouped.keys(),
        key=lambda b: max(int(r.get("time") or 0) for r in grouped[b]),
        reverse=True,
    )
    for batch in batch_keys:
        legs = sorted(grouped[batch], key=_batch_sort_key)
        out.extend(legs)
    standalone.sort(key=lambda r: int(r.get("time") or 0), reverse=True)
    out.extend(standalone)
    return out


def _effective_fetch_limit(limit: int, exclude_statuses: Optional[List[str]]) -> int:
    base = max(int(limit), 1)
    n_ex = len([s for s in (exclude_statuses or []) if str(s).strip()])
    if n_ex:
        return min(2000, base * max(4, n_ex + 2))
    return min(500, base)


def _sql_excluded_status_clause(
    excluded: Optional[List[str]], *, alias: str = "o"
) -> tuple[str, tuple[Any, ...]]:
    """Build NOT IN filter so rejected/pending noise does not fill the row limit."""
    blocked = {str(s).strip().lower() for s in (excluded or []) if str(s).strip()}
    if not blocked:
        return "", ()
    placeholders = ",".join("?" for _ in blocked)
    return f" AND lower({alias}.status) NOT IN ({placeholders})", tuple(blocked)


def _statuses_for_user_filter(status_filter: str) -> Optional[List[str]]:
    """Map UI status filter to DB status values (spot/ccxt uses ``closed`` for fills)."""
    sf = str(status_filter or "").strip().lower()
    if not sf or sf == "all":
        return None
    if sf == "filled":
        return ["filled", "closed"]
    if sf == "open":
        return ["open", "new", "submitted", "partially_filled"]
    if sf == "pending":
        return ["pending"]
    return [sf]


def _row_matches_status_filter(row_status: str, status_filter: str) -> bool:
    allowed = _statuses_for_user_filter(status_filter)
    if allowed is None:
        return True
    return str(row_status or "").lower() in allowed


def _row_matches_strategy(row: Dict[str, Any], strategy: Optional[str]) -> bool:
    want = str(strategy or "").strip().lower()
    if not want:
        return True
    got = str(row.get("strategy") or row.get("strategy_id") or "").strip().lower()
    return got == want


def _sql_included_status_clause(
    status_filter: str, *, alias: str
) -> tuple[str, tuple[Any, ...]]:
    allowed = _statuses_for_user_filter(status_filter)
    if not allowed:
        return "", ()
    if len(allowed) == 1:
        return f" AND lower({alias}.status) = ?", (allowed[0],)
    placeholders = ",".join("?" for _ in allowed)
    return f" AND lower({alias}.status) IN ({placeholders})", tuple(allowed)


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
    if is_all_symbols(symbol):
        sql = f"""
            SELECT o.order_id, o.symbol AS symbol, o.side AS side, o.status, o.order_type,
                   o.quantity, o.price, o.stop_price, o.filled_quantity, o.average_price,
                   o.created_at, o.updated_at, o.filled_at,
                   o.position_id, p.strategy_id, p.side AS pos_side,
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
                   o.position_id, p.strategy_id, p.side AS pos_side,
                   p.stop_loss_price, p.take_profit_price
            FROM orders o
            LEFT JOIN positions p ON p.position_id = o.position_id
            WHERE o.symbol = ?{status_clause}
            ORDER BY COALESCE(o.filled_at, o.created_at) DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (sym, *status_params, int(limit)))
    pos_sql = _positions_select_from(db_path)
    pos_params: tuple[Any, ...] = ()
    if not is_all_symbols(symbol):
        pos_sql += " WHERE symbol = ?"
        pos_params = (symbol.upper(),)
    pos_sql += " ORDER BY COALESCE(exit_time, entry_time) DESC LIMIT ?"
    pos_rows = query_rows(db_path, pos_sql, (*pos_params, int(limit)))
    entry_qty_by_pid = _trend_entry_qty_by_position(
        db_path, None if is_all_symbols(symbol) else symbol.upper()
    )
    out = [_normalize("trend", r) for r in rows]
    _enrich_trend_sl_tp(out, pos_rows)
    out.extend(_trend_position_event_rows(pos_rows, entry_qty_by_pid=entry_qty_by_pid))
    out.extend(_trend_operation_rows(db_path, symbol, int(limit)))
    if status:
        out = [r for r in out if _row_matches_status_filter(r["status"], status)]
    out.sort(key=lambda r: r.get("time") or 0, reverse=True)
    out = out[: int(limit)]
    return out


def _spot_orders_select_clause(db_path: Path) -> tuple[str, str]:
    """Build SELECT list and ORDER BY time expr for legacy or migrated spot_orders."""
    cols = table_columns(db_path, "spot_orders")
    if not cols:
        return "", ""
    filled_qty = (
        "filled_quantity" if "filled_quantity" in cols else "0 AS filled_quantity"
    )
    filled_quote = (
        "filled_quote_usdt"
        if "filled_quote_usdt" in cols
        else "NULL AS filled_quote_usdt"
    )
    updated_at = "updated_at" if "updated_at" in cols else "created_at AS updated_at"
    select = (
        "order_id, symbol, side, status, order_type, quantity, price, "
        f"{filled_qty}, {filled_quote}, created_at, {updated_at}"
    )
    order_ts = (
        "COALESCE(updated_at, created_at)" if "updated_at" in cols else "created_at"
    )
    return select, order_ts


def spot_orders_list(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    exclude_statuses: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    select, order_ts = _spot_orders_select_clause(db_path)
    if not select:
        return []
    status_filter = str(status or "").strip().lower()
    excluded = list(exclude_statuses or [])
    if status_filter:
        excluded = [s for s in excluded if s.lower() != status_filter]
    status_clause, status_params = _sql_excluded_status_clause(
        excluded, alias="spot_orders"
    )
    status_match, match_params = _sql_included_status_clause(
        status_filter, alias="spot_orders"
    )
    if is_all_symbols(symbol):
        sql = f"""
            SELECT {select}
            FROM spot_orders
            WHERE 1=1{status_clause}{status_match}
            ORDER BY {order_ts} DESC
            LIMIT ?
        """
        rows = query_rows(db_path, sql, (*status_params, *match_params, int(limit)))
    else:
        sym = symbol.upper()
        sql = f"""
            SELECT {select}
            FROM spot_orders
            WHERE symbol = ?{status_clause}{status_match}
            ORDER BY {order_ts} DESC
            LIMIT ?
        """
        rows = query_rows(
            db_path, sql, (sym, *status_params, *match_params, int(limit))
        )
    return [_normalize("spot", r) for r in rows]


def fetch_multileg_raw_rows(
    db_path: Path,
    symbol: str,
    *,
    engine_data_root: Optional[Path] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """All chop_grid-related rows for symbol, including synthetic inventory legs."""
    if not db_path.is_file() or is_all_symbols(symbol):
        return []
    sym = symbol.upper()
    from mlbot_console.services.trade_markers import _iso_from_unix

    # Build time clause handling mixed column types:
    #   filled_at  = INTEGER (Unix timestamp)
    #   created_at = TEXT    (ISO 8601 string like "2026-06-13 01:16:26")
    # The generic ``_sql_time_range_expr`` would compare an INTEGER against a
    # TEXT ISO param, which in SQLite always evaluates to false (INT < TEXT).
    time_parts: list = []
    time_params: list = []
    if start_ts is not None:
        iso = _iso_from_unix(start_ts)
        if iso:
            time_parts.append(
                "(filled_at >= ? OR (filled_at IS NULL AND created_at >= ?))"
            )
            time_params.extend([start_ts, iso])
    if end_ts is not None:
        iso = _iso_from_unix(end_ts)
        if iso:
            time_parts.append(
                "(filled_at <= ? OR (filled_at IS NULL AND created_at <= ?))"
            )
            time_params.extend([end_ts, iso])
    time_clause = " AND " + " AND ".join(time_parts) if time_parts else ""

    sql = f"""
        SELECT local_order_id AS order_id, symbol, side, position_side, status, order_type, purpose,
               quantity, price, stop_price, filled_quantity, average_price, created_at,
               filled_at, strategy, leg_id, client_order_id
        FROM multi_leg_orders
        WHERE symbol = ?{time_clause}
        ORDER BY COALESCE(filled_at, created_at) ASC
        LIMIT 5000
    """
    rows = query_rows(db_path, sql, (sym, *time_params))
    _supplement_multileg_inventory_entries(
        db_path, sym, rows, engine_data_root=engine_data_root
    )
    _supplement_multileg_repair_tp(db_path, sym, rows)
    return rows


def multi_leg_orders_list(
    db_path: Path,
    symbol: str,
    *,
    status: Optional[str] = None,
    exclude_statuses: Optional[List[str]] = None,
    limit: int = 100,
    engine_data_root: Optional[Path] = None,
    strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    status_filter = str(status or "").strip().lower()
    excluded = list(exclude_statuses or [])
    if status_filter:
        allowed = _statuses_for_user_filter(status_filter) or []
        excluded = [s for s in excluded if s.lower() not in allowed]
    status_clause, status_params = _sql_excluded_status_clause(
        excluded, alias="multi_leg_orders"
    )
    status_match, match_params = _sql_included_status_clause(
        status_filter, alias="multi_leg_orders"
    )
    strategy_clause = ""
    strategy_params: tuple[Any, ...] = ()
    strat = str(strategy or "").strip().lower()
    if strat:
        strategy_clause = " AND lower(multi_leg_orders.strategy) = ?"
        strategy_params = (strat,)
    if is_all_symbols(symbol):
        sql = f"""
            SELECT local_order_id AS order_id, symbol, side, position_side, status,
                   order_type, purpose, quantity, price, stop_price, filled_quantity,
                   average_price, created_at, filled_at, strategy, leg_id,
                   client_order_id, raw_json
            FROM multi_leg_orders
            WHERE 1=1{status_clause}{status_match}{strategy_clause}
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(
            db_path,
            sql,
            (*status_params, *match_params, *strategy_params, int(limit)),
        )
    else:
        sym = symbol.upper()
        sql = f"""
            SELECT local_order_id AS order_id, symbol, side, position_side, status,
                   order_type, purpose, quantity, price, stop_price, filled_quantity,
                   average_price, created_at, filled_at, strategy, leg_id,
                   client_order_id, raw_json
            FROM multi_leg_orders
            WHERE symbol = ?{status_clause}{status_match}{strategy_clause}
            ORDER BY COALESCE(filled_at, created_at) DESC
            LIMIT ?
        """
        rows = query_rows(
            db_path,
            sql,
            (sym, *status_params, *match_params, *strategy_params, int(limit)),
        )
    _supplement_multileg_inventory_entries(
        db_path, symbol, rows, engine_data_root=engine_data_root
    )
    _supplement_multileg_repair_tp(db_path, symbol, rows)
    enrich_multileg_rows_for_symbol(db_path, symbol, rows)
    out = []
    for r in rows:
        hydrate_multileg_fill_fields(r)
        qty = float(r.get("quantity") or 0)
        filled = float(r.get("filled_quantity") or 0)
        if qty <= 0 and filled > 0:
            r = dict(r)
            r["quantity"] = filled
        item = _normalize("multi_leg", r)
        if r.get("purpose"):
            item["order_type"] = r.get("purpose")
        out.append(item)
    if status:
        out = [r for r in out if _row_matches_status_filter(r["status"], status)]
    if strategy:
        out = [r for r in out if _row_matches_strategy(r, strategy)]
    return out


def _attach_pnl_fields(
    rows: List[Dict[str, Any]],
    *,
    trend_map: Dict[str, Dict[str, Any]],
    spot_map: Dict[str, Dict[str, Any]],
    multileg_map: Dict[str, Dict[str, Any]],
) -> None:
    for row in rows:
        if row.get("pnl_usdt") is not None:
            continue
        scope = str(row.get("scope") or "")
        oid = str(row.get("order_id") or "")
        rec = (
            trend_map.get(oid)
            if scope == "trend"
            else (
                spot_map.get(oid)
                if scope == "spot"
                else multileg_map.get(oid) if scope == "multi_leg" else None
            )
        )
        if not rec:
            continue
        for key in ("pnl_usdt", "realized_pnl", "unrealized_pnl", "pnl_hint"):
            if rec.get(key) is not None:
                row[key] = rec.get(key)


def enrich_orders_pnl(
    rows: List[Dict[str, Any]],
    *,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    feature_bus_root: Optional[Path],
    symbol: str,
    scopes: Optional[List[str]] = None,
) -> None:
    """Attach PnL from DB links and mark prices (multileg works without feature bus)."""
    from mlbot_console.services.account_summary import build_order_pnl_maps
    from mlbot_console.services.multileg_leg_pnl import attach_multileg_display_pnl

    scope_set = {str(s).strip().lower() for s in (scopes or []) if str(s).strip()}
    if not scope_set:
        scope_set = {"trend", "spot", "multi_leg"}

    bus = (
        feature_bus_root
        if feature_bus_root is not None and feature_bus_root.is_dir()
        else None
    )

    trend_map, spot_map, multileg_map = build_order_pnl_maps(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus,
        symbol=symbol,
        scopes=tuple(scope_set),
    )
    _attach_pnl_fields(
        rows,
        trend_map=trend_map,
        spot_map=spot_map,
        multileg_map=multileg_map,
    )

    if "multi_leg" in scope_set and multi_leg_db.is_file():
        marks: Dict[str, float] = {}
        sym_u = str(symbol or "").strip().upper()
        if bus is not None and sym_u and sym_u not in {"", "*", "ALL", "__ALL__"}:
            from mlbot_console.services.account_summary import latest_close_prices

            marks = latest_close_prices(bus, [sym_u])
        attach_multileg_display_pnl(
            rows,
            db_path=multi_leg_db,
            symbol=symbol,
            mark_prices=marks,
        )


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
    feature_bus_root: Optional[Path] = None,
    engine_data_root: Optional[Path] = None,
    strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    scope_set = {s.strip().lower() for s in scopes if s.strip()}
    per_scope = _effective_fetch_limit(int(limit), exclude_statuses)
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
        merged.extend(
            spot_orders_list(
                spot_db,
                symbol,
                status=status,
                exclude_statuses=exclude_statuses,
                limit=per_scope,
            )
        )
    if "multi_leg" in scope_set and multi_leg_db.is_file():
        merged.extend(
            multi_leg_orders_list(
                multi_leg_db,
                symbol,
                status=status,
                exclude_statuses=exclude_statuses,
                limit=per_scope,
                engine_data_root=engine_data_root,
                strategy=strategy,
            )
        )
    merged = _sort_orders_for_display(merged)
    merged = _exclude_statuses(merged, exclude_statuses)

    # 隐藏非法的 _supp 单
    merged = [r for r in merged if not str(r.get("order_id") or "").endswith("_supp")]

    if strategy:
        merged = [r for r in merged if _row_matches_strategy(r, strategy)]

    merged = merged[: int(limit)]
    enrich_orders_pnl(
        merged,
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=feature_bus_root,
        symbol=symbol,
        scopes=list(scope_set),
    )
    return merged
