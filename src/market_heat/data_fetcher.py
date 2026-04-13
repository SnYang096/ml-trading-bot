"""Fetch weekly OHLCV data from Binance via ccxt.

Caches results to local parquet to avoid redundant API calls.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / "data" / "market_heat"


def _make_exchange():
    """Create a ccxt Binance instance with optional proxy."""
    import ccxt

    params: dict = {"enableRateLimit": True}
    if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
        host = os.getenv("SOCKS5_HOST", "127.0.0.1")
        port = os.getenv("SOCKS5_PORT", "7897")
        params["proxies"] = {
            "http": f"socks5h://{host}:{port}",
            "https": f"socks5h://{host}:{port}",
        }
    return ccxt.binance(params)


def fetch_weekly_ohlcv(
    symbols: List[str],
    limit: int = 60,
    cache_max_age_hours: float = 12,
    quote: str = "USDT",
) -> Dict[str, pd.DataFrame]:
    """Fetch weekly OHLCV for a list of base symbols.

    Returns:
        Mapping from base symbol (e.g. 'BTC') to DataFrame with columns
        [timestamp, open, high, low, close, volume] indexed by timestamp.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / "weekly_ohlcv.parquet"

    cached: Optional[pd.DataFrame] = None
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < cache_max_age_hours:
            try:
                cached = pd.read_parquet(cache_path)
                cached_symbols = (
                    set(cached["symbol"].unique())
                    if "symbol" in cached.columns
                    else set()
                )
                if set(symbols).issubset(cached_symbols):
                    logger.info("Using cached weekly OHLCV (age=%.1fh)", age_h)
                    return _to_dict(cached, symbols)
            except Exception as exc:
                logger.warning("Cache read failed, refetching: %s", exc)

    exchange = _make_exchange()
    frames = []

    for base in symbols:
        pair = f"{base}/{quote}"
        try:
            data = exchange.fetch_ohlcv(pair, "1w", limit=limit)
            if not data:
                logger.warning("No weekly data for %s", pair)
                continue
            df = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["symbol"] = base
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", pair, exc)
            continue

    if not frames:
        logger.error("No data fetched for any symbol")
        return {}

    combined = pd.concat(frames, ignore_index=True)

    try:
        combined.to_parquet(cache_path, index=False)
        logger.info("Cached weekly OHLCV -> %s (%d symbols)", cache_path, len(frames))
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)

    return _to_dict(combined, symbols)


def _to_dict(df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        mask = df["symbol"] == sym
        if mask.any():
            sdf = df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)
            out[sym] = sdf
    return out
