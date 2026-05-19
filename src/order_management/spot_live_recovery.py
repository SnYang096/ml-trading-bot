"""Spot live ledger recovery: fill-driven deploy accounting and pending limit buys."""

from __future__ import annotations

import json
import ast
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

OPEN_BUY_STATUSES = frozenset(
    {"open", "new", "partially_filled", "submitted", "pending"}
)


def normalize_spot_symbol(symbol: str) -> str:
    """Normalize ccxt spot symbols like BTC/USDT into repo-style BTCUSDT."""
    base = str(symbol or "").upper().split(":", 1)[0]
    return base.replace("/", "")


def new_position_shell(
    symbol: str,
    *,
    profit_take_ladder_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    sym = str(symbol).upper()
    return {
        "symbol": sym,
        "_qty_base": 0.0,
        "_entry_notional_usdt": 0.0,
        "_spot_quote_deployed": 0.0,
        "structural_exit": "spot_simple_profit_ladder",
        "profit_take_ladder": dict(profit_take_ladder_cfg),
    }


def ensure_ladder_on_position(
    pos: Dict[str, Any],
    *,
    profit_take_ladder_cfg: Mapping[str, Any],
) -> None:
    ladder = pos.get("profit_take_ladder")
    if not isinstance(ladder, dict) or not ladder:
        pos["profit_take_ladder"] = dict(profit_take_ladder_cfg)


def effective_symbol_deployed(pos: Optional[Mapping[str, Any]]) -> float:
    """Filled deploy plus quote reserved for an open limit buy."""
    if not pos:
        return 0.0
    deployed = float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
    pending = pos.get("_pending_buy")
    if isinstance(pending, dict):
        reserved = float(pending.get("quote_reserved", 0.0) or 0.0)
        recorded = float(pending.get("filled_quote_recorded", 0.0) or 0.0)
        deployed += max(0.0, reserved - recorded)
    return deployed


def has_blocking_pending_buy(pos: Optional[Mapping[str, Any]]) -> bool:
    pending = (pos or {}).get("_pending_buy")
    return isinstance(pending, dict) and bool(pending.get("exchange_order_id") or pending.get("local_order_id"))


def pending_buy_quote_for_day(
    positions: Mapping[str, Mapping[str, Any]], *, day_key: str
) -> float:
    total = 0.0
    for pos in positions.values():
        pending = (pos or {}).get("_pending_buy")
        if not isinstance(pending, dict):
            continue
        placed = str(pending.get("placed_at") or "")
        if placed[:10] != str(day_key):
            continue
        reserved = float(pending.get("quote_reserved", 0.0) or 0.0)
        recorded = float(pending.get("filled_quote_recorded", 0.0) or 0.0)
        total += max(0.0, reserved - recorded)
    return total


def pending_buy_count_for_day(
    positions: Mapping[str, Mapping[str, Any]], *, day_key: str
) -> int:
    count = 0
    for pos in positions.values():
        pending = (pos or {}).get("_pending_buy")
        if isinstance(pending, dict) and str(pending.get("placed_at") or "")[:10] == str(day_key):
            count += 1
    return count


def parse_ccxt_fill(payload: Mapping[str, Any]) -> Tuple[str, float, float, float]:
    """Return (status, filled_qty, fill_quote_usdt, avg_price)."""
    status = str(payload.get("status") or "open").lower()
    filled_qty = float(payload.get("filled") or 0.0)
    cost = payload.get("cost")
    fill_quote = float(cost) if cost is not None else 0.0
    avg = payload.get("average")
    avg_px = float(avg) if avg is not None else 0.0
    if fill_quote <= 0.0 and filled_qty > 0.0:
        if avg_px > 0.0:
            fill_quote = filled_qty * avg_px
        else:
            px = float(payload.get("price") or 0.0)
            if px > 0.0:
                fill_quote = filled_qty * px
                avg_px = px
    return status, filled_qty, fill_quote, avg_px


def apply_buy_fill_to_position(
    pos: Dict[str, Any],
    *,
    fill_qty: float,
    fill_quote_usdt: float,
    profit_take_ladder_cfg: Mapping[str, Any],
    filled_at: Optional[str] = None,
) -> float:
    """Increase position and deploy counters from a buy fill. Returns quote applied."""
    qty = max(0.0, float(fill_qty))
    quote = max(0.0, float(fill_quote_usdt))
    if qty <= 0.0 or quote <= 0.0:
        return 0.0
    ensure_ladder_on_position(pos, profit_take_ladder_cfg=profit_take_ladder_cfg)
    pos["_qty_base"] = float(pos.get("_qty_base", 0.0) or 0.0) + qty
    pos["_entry_notional_usdt"] = float(pos.get("_entry_notional_usdt", 0.0) or 0.0) + quote
    pos["_spot_quote_deployed"] = float(pos.get("_spot_quote_deployed", 0.0) or 0.0) + quote
    if filled_at:
        pos["_last_buy_ts"] = filled_at
    return quote


def apply_sell_fill_to_position(
    pos: Dict[str, Any],
    *,
    fill_qty: float,
    exit_price: float,
) -> float:
    """Reduce position after a sell fill (proportional cost). Returns qty sold."""
    qty = max(0.0, float(fill_qty))
    if qty <= 0.0:
        return 0.0
    held = float(pos.get("_qty_base", 0.0) or 0.0)
    if held <= 0.0:
        return 0.0
    sell_qty = min(qty, held)
    cost = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
    deployed = float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
    ratio = sell_qty / held if held > 0 else 0.0
    pos["_qty_base"] = held - sell_qty
    pos["_entry_notional_usdt"] = max(0.0, cost * (1.0 - ratio))
    pos["_spot_quote_deployed"] = max(0.0, deployed * (1.0 - ratio))
    if float(pos.get("_qty_base", 0.0) or 0.0) <= 0.0:
        pos["_qty_base"] = 0.0
        pos["_entry_notional_usdt"] = 0.0
        pos["_spot_quote_deployed"] = 0.0
    return sell_qty


def clear_pending_buy(pos: Dict[str, Any]) -> None:
    pos.pop("_pending_buy", None)


def set_pending_buy(
    pos: Dict[str, Any],
    *,
    local_order_id: str,
    exchange_order_id: Optional[str],
    client_order_id: str,
    quantity: float,
    price: Optional[float],
    quote_reserved: float,
    placed_at: str,
    filled_quantity_recorded: float = 0.0,
    filled_quote_recorded: float = 0.0,
) -> None:
    pos["_pending_buy"] = {
        "local_order_id": local_order_id,
        "exchange_order_id": exchange_order_id or "",
        "client_order_id": client_order_id,
        "quantity": float(quantity),
        "price": float(price) if price is not None else None,
        "quote_reserved": float(quote_reserved),
        "placed_at": placed_at,
        "filled_quantity_recorded": float(filled_quantity_recorded),
        "filled_quote_recorded": float(filled_quote_recorded),
    }


def pending_fill_delta(
    pending: Mapping[str, Any],
    *,
    filled_qty: float,
    filled_quote: float,
) -> Tuple[float, float]:
    prev_qty = float(pending.get("filled_quantity_recorded", 0.0) or 0.0)
    prev_quote = float(pending.get("filled_quote_recorded", 0.0) or 0.0)
    return (
        max(0.0, float(filled_qty) - prev_qty),
        max(0.0, float(filled_quote) - prev_quote),
    )


def mark_pending_fill_recorded(
    pending: Dict[str, Any],
    *,
    filled_qty: float,
    filled_quote: float,
) -> None:
    pending["filled_quantity_recorded"] = max(
        float(pending.get("filled_quantity_recorded", 0.0) or 0.0),
        float(filled_qty),
    )
    pending["filled_quote_recorded"] = max(
        float(pending.get("filled_quote_recorded", 0.0) or 0.0),
        float(filled_quote),
    )


def pending_buy_age_hours(pending: Mapping[str, Any], *, now: datetime) -> float:
    raw = str(pending.get("placed_at") or "")
    if not raw:
        return 0.0
    try:
        placed = pd.Timestamp(raw)
        if placed.tzinfo is None:
            placed = placed.tz_localize("UTC")
        else:
            placed = placed.tz_convert("UTC")
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        return max(0.0, (now_ts - placed).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def rebuild_positions_from_filled_orders(
    orders: Iterable[Mapping[str, Any]],
    *,
    symbols: Iterable[str],
    profit_take_ladder_cfg: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Rebuild ledger positions from filled spot_orders rows (buys and sells)."""
    allowed = {str(s).upper() for s in symbols}
    positions: Dict[str, Dict[str, Any]] = {}
    rows = sorted(orders, key=lambda r: str(r.get("created_at") or ""))
    for row in rows:
        sym = normalize_spot_symbol(str(row.get("symbol") or ""))
        if sym not in allowed:
            continue
        side = str(row.get("side") or "").lower()
        status = str(row.get("status") or "").lower()
        if status not in {"filled", "closed", "partially_filled"} and side == "buy":
            continue
        payload: Dict[str, Any] = {}
        raw = row.get("raw_json")
        if raw:
            try:
                payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                try:
                    payload = ast.literal_eval(raw) if isinstance(raw, str) else {}
                except Exception:
                    payload = {}
        st, filled_qty, fill_quote, _avg = parse_ccxt_fill(payload)
        if filled_qty <= 0.0:
            qty = float(row.get("quantity") or 0.0)
            px = float(row.get("price") or 0.0)
            if status in {"filled", "closed"} and side == "buy" and qty > 0 and px > 0:
                filled_qty = qty
                fill_quote = qty * px
            elif status in {"filled", "closed"} and side == "sell" and qty > 0:
                filled_qty = qty
                fill_quote = qty * px if px > 0 else 0.0
            else:
                continue
        pos = positions.get(sym)
        if pos is None:
            pos = new_position_shell(sym, profit_take_ladder_cfg=profit_take_ladder_cfg)
            positions[sym] = pos
        if side == "buy":
            apply_buy_fill_to_position(
                pos,
                fill_qty=filled_qty,
                fill_quote_usdt=fill_quote,
                profit_take_ladder_cfg=profit_take_ladder_cfg,
                filled_at=str(row.get("created_at") or ""),
            )
        elif side == "sell" and filled_qty > 0:
            px = float(row.get("price") or 0.0) or (
                fill_quote / filled_qty if fill_quote > 0 and filled_qty > 0 else 0.0
            )
            apply_sell_fill_to_position(pos, fill_qty=filled_qty, exit_price=px)
    for sym in list(positions.keys()):
        if float(positions[sym].get("_qty_base", 0.0) or 0.0) <= 0.0:
            if not has_blocking_pending_buy(positions[sym]):
                del positions[sym]
    return positions


def merge_rebuilt_deploy_into_positions(
    positions: Dict[str, Dict[str, Any]],
    rebuilt: Mapping[str, Mapping[str, Any]],
    *,
    profit_take_ladder_cfg: Mapping[str, Any],
) -> None:
    """Prefer DB fill history for deploy/cost when local ledger looks empty or stale."""
    for sym, reb in rebuilt.items():
        local = positions.get(sym)
        reb_deploy = float(reb.get("_spot_quote_deployed", 0.0) or 0.0)
        reb_qty = float(reb.get("_qty_base", 0.0) or 0.0)
        if reb_qty <= 0.0 and reb_deploy <= 0.0:
            continue
        if local is None:
            positions[sym] = dict(reb)
            ensure_ladder_on_position(positions[sym], profit_take_ladder_cfg=profit_take_ladder_cfg)
            continue
        local_deploy = float(local.get("_spot_quote_deployed", 0.0) or 0.0)
        local_qty = float(local.get("_qty_base", 0.0) or 0.0)
        if local_deploy <= 0.0 and reb_deploy > 0.0:
            local["_spot_quote_deployed"] = reb_deploy
        if local_qty <= 0.0 and reb_qty > 0.0:
            local["_qty_base"] = reb_qty
        if float(local.get("_entry_notional_usdt", 0.0) or 0.0) <= 0.0 and float(
            reb.get("_entry_notional_usdt", 0.0) or 0.0
        ) > 0.0:
            local["_entry_notional_usdt"] = float(reb["_entry_notional_usdt"])
        ensure_ladder_on_position(local, profit_take_ladder_cfg=profit_take_ladder_cfg)


def sync_position_qty_from_balance(
    pos: Dict[str, Any],
    *,
    qty_live: float,
    mark_price: float,
) -> None:
    """Align base qty with exchange; scale cost/deploy proportionally on external sells."""
    qty_live = max(0.0, float(qty_live))
    old_qty = float(pos.get("_qty_base", 0.0) or 0.0)
    if qty_live <= 0.0:
        if not has_blocking_pending_buy(pos):
            pos["_qty_base"] = 0.0
            pos["_entry_notional_usdt"] = 0.0
        return
    if old_qty <= 0.0:
        cost = qty_live * max(mark_price, 0.0)
        pos["_qty_base"] = qty_live
        if float(pos.get("_entry_notional_usdt", 0.0) or 0.0) <= 0.0:
            pos["_entry_notional_usdt"] = cost
        if float(pos.get("_spot_quote_deployed", 0.0) or 0.0) <= 0.0:
            pos["_spot_quote_deployed"] = cost
        return
    if abs(qty_live - old_qty) < 1e-12:
        pos["_qty_base"] = qty_live
        return
    ratio = qty_live / old_qty
    pos["_qty_base"] = qty_live
    pos["_entry_notional_usdt"] = float(pos.get("_entry_notional_usdt", 0.0) or 0.0) * ratio
    pos["_spot_quote_deployed"] = float(pos.get("_spot_quote_deployed", 0.0) or 0.0) * ratio


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
