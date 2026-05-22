"""Fetch mark prices for console (from feature bus, fallback to exchange ticker)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

from mlbot_console.services.account_summary import latest_close_prices

logger = logging.getLogger(__name__)


def fetch_mark_prices(
    feature_bus_root: Path,
    symbols: List[str],
) -> Dict[str, float]:
    """Fetch mark prices for symbols.
    
    1. Try feature bus bars_1min (latest_close_prices).
    2. For missing symbols, fallback to Binance API ticker.
    3. Stablecoins (USDT, USDC, BUSD) are 1.0.
    """
    marks = latest_close_prices(feature_bus_root, symbols)
    
    # Stablecoins
    for stable in ["USDT", "USDC", "BUSD"]:
        if stable in symbols or f"{stable}USDT" in symbols:
            marks[stable] = 1.0
            marks[f"{stable}USDT"] = 1.0
            
    missing = [s for s in symbols if s not in marks and f"{s}USDT" not in marks and s not in ["USDT", "USDC", "BUSD"]]
    
    if missing:
        try:
            from mlbot_console.services.spot_ccxt import spot_binance_exchange

            exchange = spot_binance_exchange(
                api_key=os.getenv("BINANCE_SPOT_API_KEY", ""),
                api_secret=os.getenv("BINANCE_SPOT_API_SECRET", ""),
            )
            exchange.load_markets()
            tickers = exchange.fetch_tickers()
            for sym in missing:
                # ccxt format is usually BASE/QUOTE
                ccxt_sym = f"{sym}/USDT" if not sym.endswith("USDT") else f"{sym[:-4]}/USDT"
                ticker = tickers.get(ccxt_sym)
                if ticker:
                    px = ticker.get("last") or ticker.get("close")
                    if px:
                        marks[sym] = float(px)
                        marks[f"{sym}USDT" if not sym.endswith("USDT") else sym] = float(px)
        except Exception as e:
            logger.warning("Failed to fetch fallback tickers for %s: %s", missing, e)
            
    return marks
