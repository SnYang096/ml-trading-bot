"""Sync crypto sector taxonomy from CoinGecko.

Fetches CoinGecko category data and compares with the local YAML config.
Outputs suggestions for new additions; does NOT auto-overwrite.

Usage:
    python -m src.market_heat.sync_sectors
    python -m src.market_heat.sync_sectors --update   # write suggestions into YAML
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import yaml

from .sector_registry import _DEFAULT_CONFIG, load_sector_registry

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

CATEGORY_MAPPING = {
    "layer-1": "L1",
    "layer-2": "L2",
    "decentralized-finance-defi": "DeFi",
    "meme-token": "Meme",
    "artificial-intelligence": "AI",
    "gaming": "GameFi",
}

MIN_MARKET_CAP_USD = 50_000_000


def _get_session():
    import requests

    session = requests.Session()
    if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
        host = os.getenv("SOCKS5_HOST", "127.0.0.1")
        port = os.getenv("SOCKS5_PORT", "7897")
        proxy = f"socks5h://{host}:{port}"
        session.proxies = {"http": proxy, "https": proxy}
    return session


def fetch_coingecko_categories() -> Dict[str, List[str]]:
    """Fetch symbols per sector from CoinGecko categories.

    Returns:
        Mapping from our sector name to list of uppercase ticker symbols.
    """
    session = _get_session()
    result: Dict[str, List[str]] = {}

    for cg_cat, our_sector in CATEGORY_MAPPING.items():
        url = f"{COINGECKO_BASE}/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": cg_cat,
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
        }
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited, sleeping 60s ...")
                time.sleep(60)
                resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            coins = resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch category %s: %s", cg_cat, exc)
            continue

        symbols = []
        for coin in coins:
            mc = coin.get("market_cap") or 0
            sym = (coin.get("symbol") or "").upper()
            if mc >= MIN_MARKET_CAP_USD and sym:
                symbols.append(sym)

        result[our_sector] = symbols
        logger.info("CoinGecko %s -> %s: %d symbols", cg_cat, our_sector, len(symbols))
        time.sleep(1.5)

    return result


def diff_sectors(
    cg_sectors: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Compare CoinGecko data with local config, return new suggestions."""
    registry = load_sector_registry()
    suggestions: Dict[str, List[str]] = {}

    for sector_name, cg_symbols in cg_sectors.items():
        existing: Set[str] = set()
        if sector_name in registry.sectors:
            existing = set(registry.sectors[sector_name].symbols)

        new_syms = [s for s in cg_symbols if s not in existing]
        if new_syms:
            suggestions[sector_name] = new_syms

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Sync crypto sectors from CoinGecko")
    parser.add_argument(
        "--update", action="store_true", help="Write suggestions into YAML"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    cg_data = fetch_coingecko_categories()
    suggestions = diff_sectors(cg_data)

    if not suggestions:
        print("No new symbols to suggest. Local config is up to date.")
        return

    print("\n=== Suggested additions ===\n")
    for sector, syms in suggestions.items():
        print(f"  {sector}: {', '.join(syms)}")

    if args.update:
        raw = yaml.safe_load(_DEFAULT_CONFIG.read_text(encoding="utf-8"))
        for sector, syms in suggestions.items():
            if sector in raw.get("sectors", {}):
                existing = raw["sectors"][sector].get("symbols", [])
                raw["sectors"][sector]["symbols"] = existing + syms
        _DEFAULT_CONFIG.write_text(
            yaml.dump(
                raw, allow_unicode=True, default_flow_style=False, sort_keys=False
            ),
            encoding="utf-8",
        )
        print(f"\nUpdated {_DEFAULT_CONFIG}")
    else:
        print("\nRun with --update to write changes.")


if __name__ == "__main__":
    main()
