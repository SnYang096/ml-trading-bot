"""Fetch Binance futures income history for PnL reconciliation.

Calls ``/fapi/v1/income`` to retrieve REALIZED_PNL, COMMISSION, and
FUNDING_FEE records, then aggregates them for comparison against local
DB realized PnL.

Reference: https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Get-Income-History
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from mlbot_console.services.exchange_balances import (
    _env_first,
    _http_session,
    _SCOPE_META,
)

logger = logging.getLogger(__name__)

# Binance income types we care about
_INCOME_TYPES = ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")

# Max records per request (Binance limit)
_INCOME_LIMIT = 1000


def _fetch_income_raw(
    *,
    api_key: str,
    api_secret: str,
    symbol: Optional[str] = None,
    income_type: Optional[str] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    limit: int = _INCOME_LIMIT,
) -> List[Dict[str, Any]]:
    """Fetch one page of /fapi/v1/income."""
    session = _http_session()
    base = "https://fapi.binance.com"
    params: Dict[str, Any] = {
        "timestamp": int(time.time() * 1000),
        "limit": limit,
    }
    if symbol:
        params["symbol"] = symbol.upper()
    if income_type:
        params["incomeType"] = income_type
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    query = urlencode(sorted(params.items()))
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = session.get(
        f"{base}/fapi/v1/income?{query}&signature={sig}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected income response: {type(data)}")
    return data


def fetch_all_income(
    *,
    api_key: str,
    api_secret: str,
    symbol: Optional[str] = None,
    income_types: Optional[Tuple[str, ...]] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    max_pages: int = 20,
) -> List[Dict[str, Any]]:
    """Paginate through /fapi/v1/income to fetch all matching records."""
    all_records: List[Dict[str, Any]] = []
    types = income_types or _INCOME_TYPES
    for itype in types:
        cursor_ms = start_time_ms
        for _page in range(max_pages):
            try:
                batch = _fetch_income_raw(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbol=symbol,
                    income_type=itype,
                    start_time_ms=cursor_ms,
                    end_time_ms=end_time_ms,
                    limit=_INCOME_LIMIT,
                )
            except Exception:
                logger.warning(
                    "income fetch failed for type=%s page=%d",
                    itype,
                    _page,
                    exc_info=True,
                )
                break
            if not batch:
                break
            all_records.extend(batch)
            # Binance returns newest first; paginate by using the last record's time - 1
            last_ts = int(batch[-1].get("time", 0))
            if len(batch) < _INCOME_LIMIT:
                break
            if cursor_ms is not None and last_ts <= cursor_ms:
                break
            cursor_ms = last_ts - 1
    return all_records


def _scope_api_keys(scope: str) -> Tuple[str, str]:
    """Return (api_key, api_secret) for a given scope."""
    meta = _SCOPE_META.get(scope)
    if not meta:
        raise ValueError(f"unknown scope: {scope}")
    api_key = _env_first(*meta["key_envs"])
    api_secret = _env_first(*meta["secret_envs"])
    if not api_key or not api_secret:
        raise ValueError(f"missing API credentials for scope={scope}")
    return api_key, api_secret


def aggregate_income_by_symbol(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate income records by symbol and type.

    Returns::

        {
            "BTCUSDT": {
                "realized_pnl": 123.45,
                "commission": -6.78,
                "funding_fee": -2.34,
                "net_income": 114.33,
            },
            ...
        }
    """
    by_sym: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "realized_pnl": 0.0,
            "commission": 0.0,
            "funding_fee": 0.0,
            "net_income": 0.0,
        }
    )
    for rec in records:
        sym = str(rec.get("symbol") or "").upper()
        if not sym:
            continue
        amount = float(rec.get("income", 0.0))
        itype = str(rec.get("incomeType") or "").upper()
        bucket = by_sym[sym]
        bucket["net_income"] += amount
        if itype == "REALIZED_PNL":
            bucket["realized_pnl"] += amount
        elif itype == "COMMISSION":
            bucket["commission"] += amount
        elif itype == "FUNDING_FEE":
            bucket["funding_fee"] += amount
    return dict(by_sym)


def aggregate_income_total(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate all income records into totals by type."""
    total: Dict[str, float] = {
        "realized_pnl": 0.0,
        "commission": 0.0,
        "funding_fee": 0.0,
        "net_income": 0.0,
    }
    for rec in records:
        amount = float(rec.get("income", 0.0))
        itype = str(rec.get("incomeType") or "").upper()
        total["net_income"] += amount
        if itype == "REALIZED_PNL":
            total["realized_pnl"] += amount
        elif itype == "COMMISSION":
            total["commission"] += amount
        elif itype == "FUNDING_FEE":
            total["funding_fee"] += amount
    return total


# --- TTL cache for fetch_scope_income (avoids Binance rate-limit on dashboard refresh) ---
_income_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_income_cache_ttl = 60  # seconds
_income_cache_lock = threading.Lock()

_INCOME_CACHE_KEY_TPL = "{scope}|{symbol}|{start}|{end}"


def fetch_scope_income(
    scope: str,
    *,
    symbol: Optional[str] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch and aggregate income for a scope (trend / multi_leg).

    Results are cached for 60 seconds to avoid Binance rate-limit
    bursts when the dashboard auto-refreshes.

    Returns::

        {
            "scope": "multi_leg",
            "available": True,
            "total": {"realized_pnl": ..., "commission": ..., "funding_fee": ..., "net_income": ...},
            "by_symbol": { "BTCUSDT": {...}, ... },
            "record_count": N,
            "fetched_at": "ISO timestamp",
            "period": {"start_ms": ..., "end_ms": ...},
        }
    """
    cache_key = _INCOME_CACHE_KEY_TPL.format(
        scope=scope,
        symbol=symbol or "*",
        start=start_time_ms or "",
        end=end_time_ms or "",
    )
    now = time.time()

    # Check cache
    with _income_cache_lock:
        if cache_key in _income_cache:
            cached_ts, cached_val = _income_cache[cache_key]
            if now - cached_ts < _income_cache_ttl:
                return cached_val

    # Fetch fresh data
    try:
        api_key, api_secret = _scope_api_keys(scope)
    except ValueError as exc:
        result = {
            "scope": scope,
            "available": False,
            "error": str(exc),
            "total": aggregate_income_total([]),
            "by_symbol": {},
            "record_count": 0,
        }
        with _income_cache_lock:
            _income_cache[cache_key] = (now, result)
        return result

    try:
        records = fetch_all_income(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol if symbol and symbol != "ALL" else None,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
    except Exception as exc:
        logger.warning("fetch_scope_income failed for scope=%s: %s", scope, exc)
        result = {
            "scope": scope,
            "available": False,
            "error": str(exc),
            "total": aggregate_income_total([]),
            "by_symbol": {},
            "record_count": 0,
        }
        with _income_cache_lock:
            _income_cache[cache_key] = (now, result)
        return result

    result = {
        "scope": scope,
        "available": True,
        "total": aggregate_income_total(records),
        "by_symbol": aggregate_income_by_symbol(records),
        "record_count": len(records),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "period": {"start_ms": start_time_ms, "end_ms": end_time_ms},
    }
    with _income_cache_lock:
        _income_cache[cache_key] = (now, result)
    return result
