#!/usr/bin/env python3
"""Repair missing chop_grid reduce-only TP orders for open hedge positions.

Example (mainnet, dry-run):
  python scripts/repair_chop_grid_protection.py --symbol BNBUSDT --dry-run

Example (place orders + update DB):
  python scripts/repair_chop_grid_protection.py --symbol BNBUSDT --execute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.order_management.binance_api import BinanceAPI
from src.order_management.grid_execution_adapter import MultiLegExecutionAdapter
from src.order_management.multi_leg_storage import MultiLegStorage
from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine

logger = logging.getLogger(__name__)


def _api_from_env() -> BinanceAPI:
    key = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_KEY", "") or os.getenv(
        "MULTI_LEG_BINANCE_API_KEY", ""
    )
    secret = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_SECRET", "") or os.getenv(
        "MULTI_LEG_BINANCE_API_SECRET", ""
    )
    if not key or not secret:
        raise SystemExit("Set MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET")
    return BinanceAPI(key, secret, testnet=False)


def _round_price(px: float) -> float:
    return round(px, 2)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_grid_id(state_path: Path) -> str:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    return str(state.get("grid_id") or "").strip()


def _entry_position_side(row: Dict[str, Any]) -> Optional[str]:
    side = str(row.get("side") or "").upper()
    if side == "BUY":
        return "LONG"
    if side == "SELL":
        return "SHORT"
    return None


def _tp_price(entry_price: float, spacing: float, position_side: str) -> float:
    if position_side == "LONG":
        return _round_price(entry_price + spacing)
    return _round_price(entry_price - spacing)


def _sl_price(
    entry_price: float, spacing: float, position_side: str, max_levels_per_side: int = 2
) -> float:
    if position_side == "LONG":
        return _round_price(entry_price - spacing * (max_levels_per_side + 1))
    return _round_price(entry_price + spacing * (max_levels_per_side + 1))


def _live_order_keys(open_orders: Iterable[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for order in open_orders:
        info = order.get("info") or {}
        for key in (
            order.get("order_id"),
            order.get("client_order_id"),
            info.get("orderId"),
            info.get("clientOrderId"),
        ):
            if key:
                keys.add(str(key))
    return keys


def _live_tp_covers_entry(
    *,
    open_orders: Iterable[Dict[str, Any]],
    position_side: str,
    target_price: float,
    quantity: float,
    spacing: float,
) -> bool:
    want_side = "sell" if position_side == "LONG" else "buy"
    covered = 0.0
    tolerance = max(spacing * 0.05, target_price * 0.0005, 0.01)
    for order in open_orders:
        o_side = str(order.get("side") or "").lower()
        o_pos = str(
            order.get("position_side")
            or order.get("positionSide")
            or (order.get("info") or {}).get("positionSide")
            or ""
        ).upper()
        if o_side != want_side or o_pos != position_side:
            continue
        price = _as_float(order.get("price"))
        if price <= 0 or abs(price - target_price) > tolerance:
            continue
        qty = _as_float(order.get("remaining") or order.get("quantity"))
        if qty <= 0:
            qty = _as_float(order.get("filled"))
        covered += qty
    return covered >= quantity * 0.99


def _live_tp_qty_by_side(open_orders: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    qty_by_side = {"LONG": 0.0, "SHORT": 0.0}
    for order in open_orders:
        o_pos = str(
            order.get("position_side")
            or order.get("positionSide")
            or (order.get("info") or {}).get("positionSide")
            or ""
        ).upper()
        if o_pos not in qty_by_side:
            continue
        o_side = str(order.get("side") or "").lower()
        if (o_pos == "LONG" and o_side != "sell") or (
            o_pos == "SHORT" and o_side != "buy"
        ):
            continue
        if not bool(order.get("reduce_only", order.get("reduceOnly", True))):
            continue
        qty = _as_float(order.get("remaining") or order.get("quantity"))
        if qty <= 0:
            qty = _as_float(order.get("filled"))
        qty_by_side[o_pos] += qty
    return qty_by_side


def _open_position_qty_by_side(positions: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    qty_by_side = {"LONG": 0.0, "SHORT": 0.0}
    for pos in positions:
        side_raw = str(pos.get("side") or pos.get("positionSide") or "").lower()
        side = "LONG" if side_raw in {"long", "buy"} else "SHORT"
        qty = _as_float(
            pos.get("size") or pos.get("contracts") or pos.get("positionAmt")
        )
        if qty < 0:
            side = "SHORT"
            qty = abs(qty)
        qty_by_side[side] += qty
    return qty_by_side


def _db_entry_tp_actions(
    *,
    db_path: Path,
    symbol: str,
    grid_id: str,
    spacing: float,
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Plan missing per-entry TP orders, e.g. S2_tp when S2 filled but no TP exists."""
    if not grid_id or spacing <= 0 or not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    like = f"{grid_id}_%"
    entries = [
        dict(row)
        for row in con.execute(
            """
            SELECT local_order_id, symbol, side, quantity, filled_quantity,
                   price, average_price
            FROM multi_leg_orders
            WHERE symbol = ?
              AND local_order_id LIKE ?
              AND lower(COALESCE(purpose, '')) = 'entry'
              AND lower(COALESCE(status, '')) = 'filled'
            ORDER BY local_order_id
            """,
            (symbol, like),
        ).fetchall()
    ]
    tp_rows = [
        dict(row)
        for row in con.execute(
            """
            SELECT local_order_id, leg_id, exchange_order_id, client_order_id, status
            FROM multi_leg_orders
            WHERE symbol = ?
              AND local_order_id LIKE ?
              AND lower(COALESCE(purpose, '')) = 'take_profit'
              AND lower(COALESCE(status, '')) NOT IN ('filled', 'canceled', 'cancelled', 'expired', 'rejected')
            """,
            (symbol, like),
        ).fetchall()
    ]
    sl_rows = [
        dict(row)
        for row in con.execute(
            """
            SELECT local_order_id, leg_id, exchange_order_id, client_order_id, status
            FROM multi_leg_orders
            WHERE symbol = ?
              AND local_order_id LIKE ?
              AND lower(COALESCE(purpose, '')) = 'stop_loss'
              AND lower(COALESCE(status, '')) NOT IN ('filled', 'canceled', 'cancelled', 'expired', 'rejected')
            """,
            (symbol, like),
        ).fetchall()
    ]
    con.close()

    position_qty = _open_position_qty_by_side(positions)
    live_tp_qty = _live_tp_qty_by_side(open_orders)
    live_keys = _live_order_keys(open_orders)
    live_tp_by_leg: set[str] = set()
    for tp in tp_rows:
        if (
            str(tp.get("exchange_order_id") or "") in live_keys
            or str(tp.get("client_order_id") or "") in live_keys
        ):
            leg_id = str(tp.get("leg_id") or "").strip()
            if leg_id:
                live_tp_by_leg.add(leg_id)

    live_sl_by_leg: set[str] = set()
    for sl in sl_rows:
        if (
            str(sl.get("exchange_order_id") or "") in live_keys
            or str(sl.get("client_order_id") or "") in live_keys
        ):
            leg_id = str(sl.get("leg_id") or "").strip()
            if leg_id:
                live_sl_by_leg.add(leg_id)

    actions: List[Dict[str, Any]] = []
    for entry in entries:
        entry_id = str(entry.get("local_order_id") or "").strip()
        if not entry_id:
            continue
        position_side = _entry_position_side(entry)
        if not position_side:
            continue
        if position_qty.get(position_side, 0.0) <= 0:
            continue

        qty = _as_float(entry.get("filled_quantity")) or _as_float(
            entry.get("quantity")
        )
        entry_price = _as_float(entry.get("average_price")) or _as_float(
            entry.get("price")
        )
        if qty <= 0 or entry_price <= 0:
            continue

        # Check TP
        if (
            entry_id not in live_tp_by_leg
            and live_tp_qty.get(position_side, 0.0) < position_qty[position_side] * 0.99
        ):
            target_price = _tp_price(entry_price, spacing, position_side)
            if not _live_tp_covers_entry(
                open_orders=open_orders,
                position_side=position_side,
                target_price=target_price,
                quantity=qty,
                spacing=spacing,
            ):
                actions.append(
                    {
                        "action": "place_protection",
                        "order_id": f"{entry_id}_tp",
                        "leg_id": entry_id,
                        "symbol": symbol,
                        "side": position_side,
                        "quantity": qty,
                        "price": target_price,
                        "trigger_price": target_price,
                        "order_type": "limit",
                        "protection_type": "take_profit",
                        "reduce_only": True,
                        "post_only": False,
                        "time_in_force": "GTC",
                        "timestamp": "",
                        "repair_source": "db_filled_entry_missing_tp",
                    }
                )

        # Check SL
        if entry_id not in live_sl_by_leg:
            # We assume max_levels_per_side=2 for repair fallback if not passed explicitly
            sl_target = _sl_price(
                entry_price, spacing, position_side, max_levels_per_side=2
            )
            import uuid

            actions.append(
                {
                    "action": "place_protection",
                    "order_id": f"{entry_id}_sl_{uuid.uuid4().hex[:4]}",
                    "leg_id": entry_id,
                    "symbol": symbol,
                    "side": position_side,
                    "quantity": qty,
                    "trigger_price": sl_target,
                    "protection_type": "stop_loss",
                    "timestamp": "",
                    "repair_source": "db_filled_entry_missing_sl",
                }
            )
    return actions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BNBUSDT")
    p.add_argument("--state-dir", default="data/multi_leg_live/state")
    p.add_argument("--db", default="data/multi_leg_order_management.db")
    p.add_argument(
        "--chop-grid-config",
        default="live/highcap/config/strategies/chop_grid",
    )
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    dry_run = not args.execute

    symbol = str(args.symbol).upper()
    api = _api_from_env()
    positions = api.get_positions(symbol) or []
    open_orders = api.get_open_orders(symbol) or []
    logger.info("positions: %s", json.dumps(positions, default=str))
    logger.info("open_orders: %d", len(open_orders))

    state_path = Path(args.state_dir) / f"chop_grid_{symbol}.json"
    db_path = Path(args.db)
    engine = ChopGridLiveEngine(
        config_path=args.chop_grid_config,
        state_path=state_path,
        level_notional=200.0,
        bar_simulation=False,
    )
    grid_id = _load_grid_id(state_path)
    actions = _db_entry_tp_actions(
        db_path=db_path,
        symbol=symbol,
        grid_id=grid_id,
        spacing=float(engine.state.spacing or 0.0),
        positions=[dict(p) for p in positions],
        open_orders=[dict(o) for o in open_orders],
    )
    if actions:
        logger.info(
            "DB filled-entry repair planned %d missing TP actions for grid=%s",
            len(actions),
            grid_id,
        )
    else:
        actions = engine.actions_ensure_protection(
            exchange_positions=positions,
            exchange_orders=open_orders,
        )
    if not actions:
        logger.info("No protection actions needed (already covered or no inventory).")
        return

    for act in actions:
        side = act.get("side")
        px = act.get("price")
        logger.info(
            "planned %s %s qty=%s price=%s order_id=%s",
            act.get("protection_type"),
            side,
            act.get("quantity"),
            px,
            act.get("order_id"),
        )

    if dry_run:
        logger.info("Dry-run only; pass --execute to place orders.")
        return

    storage = MultiLegStorage(str(db_path))
    import uuid

    run_id = storage.create_run(
        mode="repair",
        strategies=["chop_grid"],
        symbols=[symbol],
        run_id=f"repair_protection_{uuid.uuid4().hex[:8]}",
    )
    adapter = MultiLegExecutionAdapter(
        api,
        shadow=False,
        storage=storage,
        run_id=run_id,
        strategy_name="chop_grid",
        default_symbol=symbol,
    )
    for act in actions:
        if str(act.get("protection_type") or "") == "take_profit":
            act["post_only"] = False
            act["time_in_force"] = "GTC"
    results = adapter.execute_actions(actions)
    for res in results:
        logger.info(
            "result action=%s status=%s order_id=%s client=%s",
            res.action,
            res.status,
            res.order_id,
            res.client_order_id,
        )
    storage.finish_run(run_id, status="done")
    logger.info("Done. Re-run backfill or wait for user stream to sync DB.")


if __name__ == "__main__":
    main()
