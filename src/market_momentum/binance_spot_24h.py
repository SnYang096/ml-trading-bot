"""Binance Spot 24h USDT gainers via public ticker/24hr (no API key)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

BINANCE_SPOT_BASE = "https://api.binance.com"
TICKER_24HR_PATH = "/api/v3/ticker/24hr"

# Quote assets we never treat as satellite buy targets.
_STABLE_BASES = frozenset(
    {
        "USDC",
        "BUSD",
        "FDUSD",
        "TUSD",
        "USDP",
        "DAI",
        "EUR",
        "GBP",
        "AEUR",
        "USD1",
    }
)

# Leveraged / structured tokens on Binance spot (suffix patterns).
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _get_session():
    import requests

    session = requests.Session()
    if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
        host = os.getenv("SOCKS5_HOST", "127.0.0.1")
        port = os.getenv("SOCKS5_PORT", "7897")
        proxy = f"socks5h://{host}:{port}"
        session.proxies = {"http": proxy, "https": proxy}
    return session


@dataclass(frozen=True)
class GainerRow:
    symbol: str
    base: str
    last_price: float
    price_change_pct: float
    quote_volume_usdt: float
    rank: int


def _base_from_usdt_symbol(symbol: str) -> Optional[str]:
    sym = str(symbol or "").upper()
    if not sym.endswith("USDT") or len(sym) <= 4:
        return None
    return sym[:-4]


def is_tradable_satellite_usdt_pair(
    symbol: str,
    *,
    stable_bases: Sequence[str] = (),
) -> bool:
    """Filter Binance *USDT pairs suitable for profit_satellite candidates."""
    base = _base_from_usdt_symbol(symbol)
    if not base:
        return False
    blocked = _STABLE_BASES.union({str(b).upper() for b in stable_bases})
    if base in blocked:
        return False
    for suf in _LEVERAGED_SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf) + 1:
            return False
    return True


def fetch_ticker_24hr(
    *,
    symbol_status: str = "TRADING",
    ticker_type: Optional[str] = None,
    base_url: str = BINANCE_SPOT_BASE,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Raw Binance GET /api/v3/ticker/24hr (all symbols when symbol omitted).

    Use ``ticker_type=None`` (FULL) when ``priceChangePercent`` is required;
    MINI omits change fields.
    """
    session = _get_session()
    url = f"{base_url.rstrip('/')}{TICKER_24HR_PATH}"
    params: Dict[str, str] = {}
    if symbol_status:
        params["symbolStatus"] = symbol_status
    if ticker_type:
        params["type"] = ticker_type
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError(f"expected list from ticker/24hr, got {type(payload)}")
    return payload


def fetch_usdt_24h_gainers(
    *,
    limit: int = 20,
    min_quote_volume_usdt: float = 1_000_000.0,
    min_price_change_pct: float = 0.0,
    symbol_status: str = "TRADING",
    base_url: str = BINANCE_SPOT_BASE,
) -> List[GainerRow]:
    """Return top USDT spot pairs by 24h priceChangePercent (descending)."""
    rows = fetch_ticker_24hr(
        symbol_status=symbol_status,
        ticker_type=None,
        base_url=base_url,
    )
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        sym = str(row.get("symbol") or "")
        if not is_tradable_satellite_usdt_pair(sym):
            continue
        try:
            pct = float(row.get("priceChangePercent", 0))
            qv = float(row.get("quoteVolume", 0))
            last_px = float(row.get("lastPrice", 0))
        except (TypeError, ValueError):
            continue
        if pct < min_price_change_pct or qv < min_quote_volume_usdt or last_px <= 0:
            continue
        candidates.append(
            {
                "symbol": sym,
                "base": sym[:-4],
                "last_price": last_px,
                "price_change_pct": pct,
                "quote_volume_usdt": qv,
            }
        )

    candidates.sort(key=lambda x: x["price_change_pct"], reverse=True)
    out: List[GainerRow] = []
    for i, c in enumerate(candidates[: max(1, int(limit))], start=1):
        out.append(
            GainerRow(
                symbol=c["symbol"],
                base=c["base"],
                last_price=c["last_price"],
                price_change_pct=c["price_change_pct"],
                quote_volume_usdt=c["quote_volume_usdt"],
                rank=i,
            )
        )
    return out


def weekly_deploy_usdt(
    profit_pool_usdt: float,
    *,
    deploy_frac: float = 0.01,
    tier_cap_remaining_usdt: Optional[float] = None,
    single_coin_cap_usdt: Optional[float] = None,
) -> float:
    """profit_satellite weekly buy notional (default profit_pool × 1%)."""
    pool = max(0.0, float(profit_pool_usdt))
    raw = pool * float(deploy_frac)
    caps = [raw]
    if tier_cap_remaining_usdt is not None:
        caps.append(max(0.0, float(tier_cap_remaining_usdt)))
    if single_coin_cap_usdt is not None:
        caps.append(max(0.0, float(single_coin_cap_usdt)))
    return max(0.0, min(caps))
