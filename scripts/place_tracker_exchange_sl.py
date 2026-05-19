#!/usr/bin/env python3
"""One-off: place exchange STOP_MARKET for open positions in PositionTracker JSON."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.order_management.binance_api import BinanceAPI
from src.order_management.models import OrderSide, OrderType
from src.order_management.position_tracker import PositionTracker
from src.order_management.order_manager import OrderManager
from src.order_management.storage import Storage

logger = logging.getLogger("place_tracker_exchange_sl")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dir",
        default=os.getenv(
            "MLBOT_POSITION_TRACKER_STATE_DIR",
            "live/highcap/data/position_tracker",
        ),
    )
    p.add_argument(
        "--db",
        default=os.getenv(
            "MLBOT_LIVE_BASE",
            "live/highcap/data",
        ),
    )
    args = p.parse_args()

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
        "BINANCE_FUTURES_API_SECRET", ""
    )
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")

    api = BinanceAPI(api_key, api_secret, testnet=False)
    db_path = Path(args.db) / "db" / "order_management.db"
    storage = Storage(str(db_path))
    om = OrderManager(storage=storage, binance_api=api, shadow=False)

    live_symbols = {
        str(p.get("symbol", "")).replace("/", "").split(":")[0].upper()
        for p in api.get_positions()
    }
    state_dir = Path(args.state_dir)
    placed_total = 0
    for path in sorted(state_dir.glob("*.json")):
        sym = path.stem.upper()
        if sym not in live_symbols:
            logger.info("skip %s: no exchange position", sym)
            continue
        tracker = PositionTracker(order_manager=om, symbol=sym, state_path=path)
        n = int(tracker.restore_from_disk(live_symbols=live_symbols) or 0)
        if n <= 0:
            continue
        placed = int(tracker.ensure_exchange_stop_losses() or 0)
        placed_total += placed
        logger.info("%s restored=%d placed_sl=%d", sym, n, placed)
    print(
        json.dumps(
            {"placed_total": placed_total, "state_dir": str(state_dir)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
