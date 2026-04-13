"""CLI entry point for Market Heat Dashboard.

Usage:
    python -m src.market_heat.run                      # colored table (default)
    python -m src.market_heat.run --format json         # JSON output
    python -m src.market_heat.run --sector L1 --sector Meme  # filter sectors
    python -m src.market_heat.run --no-cache            # force fresh fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .data_fetcher import fetch_weekly_ohlcv
from .heat_calculator import HeatResult, compute_heat_batch
from .sector_aggregator import MarketHeat, SectorHeat, aggregate
from .sector_registry import SectorRegistry, load_sector_registry

logger = logging.getLogger(__name__)


def run_heat_update(
    registry: Optional[SectorRegistry] = None,
    force_refresh: bool = False,
) -> MarketHeat:
    """Fetch data, compute heat, and return aggregated result."""
    if registry is None:
        registry = load_sector_registry()

    all_symbols = registry.all_symbols
    logger.info("Fetching weekly OHLCV for %d symbols ...", len(all_symbols))

    cache_age = 0.01 if force_refresh else 12
    ohlcv = fetch_weekly_ohlcv(all_symbols, cache_max_age_hours=cache_age)
    logger.info("Got data for %d / %d symbols", len(ohlcv), len(all_symbols))

    symbol_heats = compute_heat_batch(ohlcv)
    logger.info("Computed heat for %d symbols", len(symbol_heats))

    return aggregate(symbol_heats, registry)


# ── Table rendering ──────────────────────────────────────────────


_STATE_COLORS = {
    "HOT": "\033[92m",  # bright green
    "WARM": "\033[93m",  # yellow
    "COLD": "\033[91m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _colorize(state: str) -> str:
    c = _STATE_COLORS.get(state, "")
    return f"{c}{state:4s}{_RESET}"


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    empty = width - filled
    if score >= 0.5:
        color = "\033[92m"
    elif score >= 0.2:
        color = "\033[93m"
    else:
        color = "\033[91m"
    return f"{color}{'█' * filled}{'░' * empty}{_RESET}"


def print_table(market: MarketHeat, sectors_filter: Optional[List[str]] = None) -> None:
    """Print a colored terminal table of heat scores."""
    print()
    print(f"{_BOLD}{'═' * 70}{_RESET}")
    print(f"{_BOLD}  MARKET HEAT DASHBOARD — Crypto{_RESET}")
    print(f"{_BOLD}{'═' * 70}{_RESET}")

    # Market overview
    print(
        f"\n  Market Score: {_bar(market.score)} {_colorize(market.state)} ({market.score:.2f})"
    )

    hot_sectors = sum(1 for s in market.sector_heats.values() if s.state == "HOT")
    cold_sectors = sum(1 for s in market.sector_heats.values() if s.state == "COLD")
    total_sectors = len(market.sector_heats)
    print(
        f"  Sectors: {hot_sectors} HOT / {total_sectors - hot_sectors - cold_sectors} WARM / {cold_sectors} COLD"
    )

    # Sector summary
    print(
        f"\n{_BOLD}  {'Sector':<10} {'Score':>6}  {'State':<6} {'Bar':<22} {'HOT':>4} {'WARM':>5} {'COLD':>5}{_RESET}"
    )
    print(f"  {'─' * 65}")

    sorted_sectors = sorted(
        market.sector_heats.values(),
        key=lambda s: s.score,
        reverse=True,
    )

    for sh in sorted_sectors:
        if sectors_filter and sh.name not in sectors_filter:
            continue
        print(
            f"  {sh.name:<10} {sh.score:>6.3f}  {_colorize(sh.state)} "
            f"{_bar(sh.score)}  {sh.hot_count:>3}  {sh.warm_count:>4}  {sh.cold_count:>4}"
        )

    # Symbol detail per sector
    for sh in sorted_sectors:
        if sectors_filter and sh.name not in sectors_filter:
            continue
        if not sh.member_heats:
            continue
        print(f"\n{_BOLD}  [{sh.name}] symbols:{_RESET}")
        print(
            f"  {'Symbol':<10} {'Score':>6}  {'State':<6} {'EMA Slope':>10} {'Distance':>10} {'Price':>12}"
        )
        print(f"  {'─' * 60}")
        for hr in sorted(sh.member_heats, key=lambda h: h.score, reverse=True):
            print(
                f"  {hr.symbol:<10} {hr.score:>6.3f}  {_colorize(hr.state)} "
                f"{hr.ema_slope:>+10.4f} {hr.ema_distance:>+10.4f} {hr.price:>12.2f}"
            )

    print(f"\n{_BOLD}{'═' * 70}{_RESET}\n")


def to_json(market: MarketHeat) -> dict:
    """Convert MarketHeat to a JSON-serializable dict."""
    return {
        "market": {
            "score": market.score,
            "state": market.state,
        },
        "sectors": {
            name: {
                "score": sh.score,
                "state": sh.state,
                "hot": sh.hot_count,
                "warm": sh.warm_count,
                "cold": sh.cold_count,
                "total": sh.total_count,
            }
            for name, sh in market.sector_heats.items()
        },
        "symbols": {
            sym: {
                "score": hr.score,
                "state": hr.state,
                "ema_slope": hr.ema_slope,
                "ema_distance": hr.ema_distance,
                "ema_value": hr.ema_value,
                "price": hr.price,
                "sector": None,
            }
            for sym, hr in market.symbol_heats.items()
        },
    }


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market Heat Dashboard — Crypto sector heat scanner",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--sector",
        action="append",
        dest="sectors",
        help="Filter output to specific sectors (can repeat)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data fetch, ignore cache",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to sector config YAML (default: config/market_heat/crypto_sectors.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    from pathlib import Path

    config_path = Path(args.config) if args.config else None
    registry = load_sector_registry(config_path)

    market = run_heat_update(registry=registry, force_refresh=args.no_cache)

    if args.format == "json":
        print(json.dumps(to_json(market), indent=2, ensure_ascii=False))
    else:
        print_table(market, sectors_filter=args.sectors)


if __name__ == "__main__":
    main()
