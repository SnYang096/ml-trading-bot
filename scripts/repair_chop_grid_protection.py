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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.order_management.binance_api import BinanceAPI
from src.order_management.grid_execution_adapter import MultiLegExecutionAdapter
from src.order_management.models import OrderSide, OrderType
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
    engine = ChopGridLiveEngine(
        config_path=args.chop_grid_config,
        state_path=state_path,
        level_notional=200.0,
        bar_simulation=False,
    )
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

    storage = MultiLegStorage(str(args.db))
    run_id = storage.create_run(
        mode="repair",
        strategies=["chop_grid"],
        symbols=[symbol],
        run_id="repair_protection",
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
