#!/usr/bin/env python3
"""Standalone heat update entry point for cron / systemd timer.

Fetches weekly OHLCV, computes heat, exports to Prometheus gauges.
Designed to run once per day (weekly EMA moves slowly).

Usage:
    python scripts/run_heat_update.py                  # one-shot update + prometheus push
    python scripts/run_heat_update.py --loop 3600      # run every hour
    python scripts/run_heat_update.py --print           # also print table to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.market_heat.data_fetcher import fetch_weekly_ohlcv
from src.market_heat.heat_calculator import compute_heat_batch
from src.market_heat.metrics import export_heat_to_prometheus
from src.market_heat.run import print_table
from src.market_heat.sector_aggregator import aggregate
from src.market_heat.sector_registry import load_sector_registry

logger = logging.getLogger(__name__)


def one_shot(*, show_table: bool = False):
    registry = load_sector_registry()
    ohlcv = fetch_weekly_ohlcv(registry.all_symbols, cache_max_age_hours=0.01)
    symbol_heats = compute_heat_batch(ohlcv)
    market = aggregate(symbol_heats, registry)
    export_heat_to_prometheus(market, registry)

    if show_table:
        print_table(market)

    logger.info(
        "Heat update complete: market=%s (%.3f), %d symbols scored",
        market.state,
        market.score,
        len(symbol_heats),
    )
    return market


def main():
    parser = argparse.ArgumentParser(description="Market Heat update daemon")
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="If > 0, loop every N seconds (default: one-shot)",
    )
    parser.add_argument("--print", dest="show_table", action="store_true")
    parser.add_argument("--metrics-port", type=int, default=9091)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    try:
        from prometheus_client import start_http_server

        start_http_server(args.metrics_port)
        logger.info("Heat metrics server on :%d/metrics", args.metrics_port)
    except ImportError:
        logger.warning("prometheus_client not installed, no metrics endpoint")
    except OSError as exc:
        logger.warning("Could not start metrics server: %s", exc)

    if args.loop > 0:
        logger.info("Running heat update loop every %ds", args.loop)
        while True:
            try:
                one_shot(show_table=args.show_table)
            except Exception:
                logger.exception("Heat update failed")
            time.sleep(args.loop)
    else:
        one_shot(show_table=args.show_table)


if __name__ == "__main__":
    main()
