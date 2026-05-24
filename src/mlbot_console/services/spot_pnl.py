"""Spot order PnL — FIFO lots by buy time, realized on sell, unrealized on open buys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mlbot_console.services.db import query_rows
from mlbot_console.services.trade_markers import _parse_ts


@dataclass
class _BuyLot:
    order_id: str
    symbol: str
    qty: float
    cost_usdt: float
    unit_cost: float
    entry_ts: int
    strategy: str


def _spot_orders_columns(spot_db: Path) -> set[str]:
    if not spot_db.is_file():
        return set()
    try:
        rows = query_rows(spot_db, "PRAGMA table_info(spot_orders)")
    except Exception:
        return set()
    return {str(r.get("name") or "").lower() for r in rows}


def _default_spot_strategy() -> str:
    from mlbot_console.services.strategy_registry import default_spot_strategy_id

    return default_spot_strategy_id()


def _strategy_from_row(row: Dict[str, Any]) -> str:
    raw = row.get("strategy") or row.get("strategy_id")
    if raw:
        return str(raw).strip().lower()
    return _default_spot_strategy()


def _spot_select_sql(*, symbol_filter: str, columns: set[str]) -> tuple[str, tuple[Any, ...]]:
    extra = ", strategy" if "strategy" in columns else ""
    if "strategy_id" in columns and "strategy" not in columns:
        extra = ", strategy_id AS strategy"
    base = f"""
            SELECT order_id, created_at, updated_at, symbol, side, order_type,
                   quantity, price, status, filled_quantity, filled_quote_usdt{extra}
            FROM spot_orders
    """
    if symbol_filter in {"", "*", "ALL", "__ALL__"}:
        return base + " ORDER BY COALESCE(updated_at, created_at) ASC", ()
    return (
        base + " WHERE symbol = ? ORDER BY COALESCE(updated_at, created_at) ASC",
        (symbol_filter,),
    )


def _filled_qty_and_quote(row: Dict[str, Any]) -> tuple[float, float, float]:
    status = str(row.get("status") or "").lower()
    side = str(row.get("side") or "").lower()
    qty = float(row.get("filled_quantity") or 0.0)
    px = float(row.get("price") or 0.0)
    quote = float(row.get("filled_quote_usdt") or 0.0)
    if qty <= 0 and status in {"filled", "closed"}:
        qty = float(row.get("quantity") or 0.0)
    if quote <= 0 and qty > 0 and px > 0:
        quote = qty * px
    return qty, quote, px


def _is_filled_row(row: Dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    qty, _, _ = _filled_qty_and_quote(row)
    return status in {"filled", "closed", "partially_filled"} or qty > 0


def compute_spot_order_pnl(
    spot_db: Path,
    *,
    symbol: Optional[str] = None,
    mark_prices: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map spot order_id -> pnl fields for console display.

    Buys are FIFO lots keyed by fill time. Each sell realizes PnL against the
    oldest remaining lots. Open buys get unrealized PnL when ``mark_prices`` is set.
    """
    if not spot_db.is_file():
        return {}
    marks = {str(k).upper(): float(v) for k, v in (mark_prices or {}).items()}
    sym_filter = str(symbol or "").strip().upper()
    cols = _spot_orders_columns(spot_db)
    sql, params = _spot_select_sql(symbol_filter=sym_filter, columns=cols)
    rows = query_rows(spot_db, sql, params)
    lots_by_sym: Dict[str, List[_BuyLot]] = {}
    out: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not _is_filled_row(row):
            continue
        sym = str(row.get("symbol") or "").upper()
        oid = str(row.get("order_id") or "")
        side = str(row.get("side") or "").lower()
        qty, quote, px = _filled_qty_and_quote(row)
        if not oid or qty <= 0:
            continue
        ts = _parse_ts(row.get("updated_at")) or _parse_ts(row.get("created_at")) or 0
        lots = lots_by_sym.setdefault(sym, [])

        if side == "buy":
            unit = quote / qty if qty > 0 else px
            strat = _strategy_from_row(row)
            lots.append(
                _BuyLot(
                    order_id=oid,
                    symbol=sym,
                    qty=qty,
                    cost_usdt=quote,
                    unit_cost=unit,
                    entry_ts=int(ts),
                    strategy=strat,
                )
            )
            continue

        if side != "sell":
            continue

        sell_qty = qty
        sell_quote = quote if quote > 0 else sell_qty * px
        cost_basis = 0.0
        matched = 0.0
        entry_refs: List[str] = []
        matched_strategies: List[str] = []
        while sell_qty > 1e-12 and lots:
            lot = lots[0]
            take = min(sell_qty, lot.qty)
            frac = take / lot.qty if lot.qty > 0 else 0.0
            cost_basis += lot.cost_usdt * frac
            matched += take
            if not entry_refs or entry_refs[-1] != lot.order_id:
                entry_refs.append(lot.order_id)
                matched_strategies.append(lot.strategy)
            sell_qty -= take
            lot.qty -= take
            lot.cost_usdt *= 1.0 - frac
            if lot.qty <= 1e-12:
                lots.pop(0)
                out.pop(lot.order_id, None)
        if matched <= 0:
            continue
        proceeds = sell_quote * (matched / qty) if qty > 0 else sell_quote
        realized = proceeds - cost_basis
        matched_strat = (
            matched_strategies[0] if matched_strategies else _strategy_from_row(row)
        )
        out[oid] = {
            "realized_pnl": realized,
            "unrealized_pnl": None,
            "pnl_usdt": realized,
            "pnl_hint": "已实现",
            "matched_buy_orders": entry_refs,
            "matched_qty": matched,
            "exit_ts": int(ts),
            "strategy": matched_strat,
        }

    for sym, lots in lots_by_sym.items():
        mark = marks.get(sym)
        if mark is None or mark <= 0:
            continue
        for lot in lots:
            unreal = lot.qty * float(mark) - lot.cost_usdt
            out[lot.order_id] = {
                "realized_pnl": None,
                "unrealized_pnl": unreal,
                "pnl_usdt": unreal,
                "pnl_hint": "持仓浮盈",
                "symbol": sym,
                "entry_price": lot.unit_cost,
                "lot_qty": lot.qty,
                "lot_cost_usdt": lot.cost_usdt,
                "mark_price": float(mark),
                "strategy": lot.strategy,
            }
    return out


def spot_holdings_from_orders(
    spot_db: Path,
    *,
    mark_prices: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Open FIFO buy lots aggregated per symbol (avg entry for account page)."""
    per_order = compute_spot_order_pnl(spot_db, mark_prices=mark_prices)
    by_sym: Dict[str, Dict[str, float]] = {}
    for rec in per_order.values():
        if rec.get("unrealized_pnl") is None:
            continue
        sym = str(rec.get("symbol") or "").upper()
        lot_qty = float(rec.get("lot_qty") or 0.0)
        lot_cost = float(rec.get("lot_cost_usdt") or 0.0)
        if not sym or lot_qty <= 0:
            continue
        bucket = by_sym.setdefault(sym, {"qty": 0.0, "cost_usdt": 0.0, "unrealized": 0.0})
        bucket["qty"] += lot_qty
        bucket["cost_usdt"] += lot_cost
        bucket["unrealized"] += float(rec.get("unrealized_pnl") or 0.0)

    marks = {str(k).upper(): float(v) for k, v in (mark_prices or {}).items()}
    out: List[Dict[str, Any]] = []
    for sym, bucket in by_sym.items():
        qty = float(bucket["qty"] or 0.0)
        if qty <= 0:
            continue
        cost = float(bucket["cost_usdt"] or 0.0)
        avg_entry = cost / qty
        asset = sym[:-4] if sym.endswith("USDT") else sym
        mark = marks.get(sym) or marks.get(asset) or 0.0
        value = qty * mark if mark > 0 else cost
        out.append(
            {
                "asset": asset,
                "symbol": sym,
                "qty": qty,
                "avg_entry_usdt": avg_entry,
                "cost_notional_usdt": cost,
                "price_usdt": mark,
                "value_usdt": value,
                "unrealized_pnl_usdt": float(bucket["unrealized"] or 0.0),
                "price_source": "fifo_orders",
            }
        )
    return sorted(out, key=lambda x: x["value_usdt"], reverse=True)


def spot_holdings_from_orders(
    spot_db: Path,
    *,
    mark_prices: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate open FIFO buy lots into per-symbol holdings with avg entry."""
    if not spot_db.is_file():
        return []
    per_order = compute_spot_order_pnl(spot_db, mark_prices=mark_prices)
    by_sym: Dict[str, Dict[str, Any]] = {}
    for rec in per_order.values():
        if rec.get("unrealized_pnl") is None:
            continue
        sym = str(rec.get("symbol") or "").upper()
        if not sym:
            continue
        qty = float(rec.get("matched_qty") or rec.get("qty") or 0.0)
        entry_px = float(rec.get("entry_price") or 0.0)
        if qty <= 0 and entry_px <= 0:
            continue
        bucket = by_sym.setdefault(
            sym,
            {"symbol": sym, "qty": 0.0, "cost_usdt": 0.0, "unrealized_pnl_usdt": 0.0},
        )
        lot_qty = float(rec.get("lot_qty") or 0.0)
        lot_cost = float(rec.get("lot_cost_usdt") or 0.0)
        if lot_qty <= 0:
            # derive from unrealized when possible
            mark = float(rec.get("mark_price") or mark_prices.get(sym) or 0.0)  # type: ignore[union-attr]
            unreal = float(rec.get("unrealized_pnl") or 0.0)
            if mark > 0 and entry_px > 0:
                lot_qty = unreal / (mark - entry_px) if mark != entry_px else 0.0
                lot_cost = lot_qty * entry_px
        if lot_qty <= 0:
            continue
        bucket["qty"] += lot_qty
        bucket["cost_usdt"] += lot_cost
        bucket["unrealized_pnl_usdt"] += float(rec.get("unrealized_pnl") or 0.0)

    out: List[Dict[str, Any]] = []
    marks = {str(k).upper(): float(v) for k, v in (mark_prices or {}).items()}
    for sym, bucket in by_sym.items():
        qty = float(bucket["qty"] or 0.0)
        if qty <= 0:
            continue
        cost = float(bucket["cost_usdt"] or 0.0)
        avg_entry = cost / qty if qty > 0 else 0.0
        asset = sym[:-4] if sym.endswith("USDT") else sym
        mark = marks.get(sym) or marks.get(asset) or 0.0
        value = qty * mark if mark > 0 else cost
        out.append(
            {
                "asset": asset,
                "symbol": sym,
                "qty": qty,
                "avg_entry_usdt": avg_entry,
                "cost_notional_usdt": cost,
                "price_usdt": mark,
                "value_usdt": value,
                "unrealized_pnl_usdt": float(bucket["unrealized_pnl_usdt"] or 0.0),
                "price_source": "fifo_orders",
            }
        )
    return sorted(out, key=lambda x: x["value_usdt"], reverse=True)

