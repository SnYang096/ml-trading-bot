"""Live Binance balances for business-console account overview (read-only)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

_SCOPE_META = {
    "trend": {
        "label": "B·Trend",
        "account_type": "futures_usdtm",
        "binance_label": "U 本位合约（Trend）",
        "key_envs": ("BINANCE_API_KEY", "BINANCE_FUTURES_API_KEY"),
        "secret_envs": ("BINANCE_API_SECRET", "BINANCE_FUTURES_API_SECRET"),
    },
    "multi_leg": {
        "label": "C·Multi-leg",
        "account_type": "futures_usdtm",
        "binance_label": "U 本位合约（Hedge / Multi-leg）",
        "key_envs": ("MULTI_LEG_BINANCE_FUTURES_API_KEY", "MULTI_LEG_BINANCE_API_KEY"),
        "secret_envs": (
            "MULTI_LEG_BINANCE_FUTURES_API_SECRET",
            "MULTI_LEG_BINANCE_API_SECRET",
        ),
    },
    "spot": {
        "label": "A·Spot",
        "account_type": "spot",
        "binance_label": "现货（Spot）",
        "key_envs": ("BINANCE_SPOT_API_KEY",),
        "secret_envs": ("BINANCE_SPOT_API_SECRET",),
    },
}


def _env_first(*names: str) -> str:
    for name in names:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def _http_session():
    import requests

    session = requests.Session()
    if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
        host = os.getenv("SOCKS5_HOST", "127.0.0.1")
        port = os.getenv("SOCKS5_PORT", "7897")
        proxy = f"socks5h://{host}:{port}"
        session.proxies = {"http": proxy, "https": proxy}
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def _fetch_futures_account_raw(*, api_key: str, api_secret: str) -> Dict[str, Any]:
    session = _http_session()
    base = "https://fapi.binance.com"
    srv = session.get(f"{base}/fapi/v1/time", timeout=8)
    srv.raise_for_status()
    server_ts = int(srv.json().get("serverTime", int(time.time() * 1000)))
    query = f"timestamp={server_ts}"
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = session.get(
        f"{base}/fapi/v2/account?{query}&signature={sig}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected futures account response")
    return data


def parse_futures_account(data: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "wallet_balance_usdt": float(data.get("totalWalletBalance") or 0.0),
        "equity_usdt": float(data.get("totalMarginBalance") or 0.0),
        "available_usdt": float(data.get("availableBalance") or 0.0),
        "unrealized_pnl_usdt": float(data.get("totalUnrealizedProfit") or 0.0),
    }


def _fetch_spot_equity(
    *,
    api_key: str,
    api_secret: str,
    mark_prices: Mapping[str, float],
) -> Dict[str, Any]:
    from order_management.spot_binance_api import SpotBinanceAPI

    api = SpotBinanceAPI(api_key=api_key, api_secret=api_secret, testnet=False)
    bal = api.exchange.fetch_balance()
    usdt = bal.get("USDT") if isinstance(bal.get("USDT"), dict) else {}
    free_usdt = float(usdt.get("free") or 0.0)
    total_usdt = float(usdt.get("total") or free_usdt)
    
    holdings = []
    holdings_value_usdt = 0.0
    
    totals = bal.get("total") if isinstance(bal.get("total"), dict) else {}
    
    # Check if we need fallback tickers
    missing_assets = []
    for asset, qty in totals.items():
        sym_asset = str(asset or "").upper()
        if sym_asset in {"", "USDT", "USDC", "BUSD"}:
            continue
        try:
            q = float(qty or 0.0)
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue
        px = float(mark_prices.get(f"{sym_asset}USDT") or mark_prices.get(sym_asset) or 0.0)
        if px <= 0:
            missing_assets.append(sym_asset)
            
    fallback_marks = {}
    if missing_assets:
        try:
            tickers = api.exchange.fetch_tickers()
            for sym in missing_assets:
                ccxt_sym = f"{sym}/USDT"
                ticker = tickers.get(ccxt_sym)
                if ticker:
                    px = ticker.get("last") or ticker.get("close")
                    if px:
                        fallback_marks[sym] = float(px)
        except Exception as e:
            logger.warning("Failed to fetch fallback tickers for %s: %s", missing_assets, e)

    for asset, qty in totals.items():
        sym_asset = str(asset or "").upper()
        if sym_asset == "":
            continue
            
        try:
            q = float(qty or 0.0)
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue
            
        if sym_asset in {"USDT", "USDC", "BUSD"}:
            px = 1.0
            src = "stablecoin"
        else:
            px = float(mark_prices.get(f"{sym_asset}USDT") or mark_prices.get(sym_asset) or 0.0)
            src = "bars_1min"
            if px <= 0:
                px = fallback_marks.get(sym_asset, 0.0)
                src = "ticker" if px > 0 else "missing"
                
        val = q * px
        if sym_asset != "USDT":
            holdings_value_usdt += val
            holdings.append({
                "asset": sym_asset,
                "qty": q,
                "price_usdt": px,
                "value_usdt": val,
                "price_source": src
            })
        
    equity = total_usdt + holdings_value_usdt
    
    return {
        "wallet_balance_usdt": equity,  # Total equity in USDT
        "equity_usdt": equity,
        "available_usdt": free_usdt,
        "unrealized_pnl_usdt": 0.0,
        "usdt_cash": total_usdt,
        "holdings": sorted(holdings, key=lambda x: x["value_usdt"], reverse=True),
        "holdings_value_usdt": holdings_value_usdt,
    }


def _snapshot_shell(scope: str) -> Dict[str, Any]:
    meta = _SCOPE_META[scope]
    return {
        "scope": scope,
        "label": meta["label"],
        "account_type": meta["account_type"],
        "binance_label": meta["binance_label"],
        "configured": False,
        "ok": False,
        "error": None,
        "error_code": None,
        "wallet_balance_usdt": None,
        "equity_usdt": None,
        "available_usdt": None,
        "unrealized_pnl_usdt": None,
        "fetched_at": None,
    }


def fetch_scope_exchange_balance(
    scope: str,
    *,
    mark_prices: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Return one account row aligned with console scope (trend / spot / multi_leg)."""
    if scope not in _SCOPE_META:
        raise ValueError(f"unknown scope: {scope}")
    meta = _SCOPE_META[scope]
    out = _snapshot_shell(scope)
    api_key = _env_first(*meta["key_envs"])
    api_secret = _env_first(*meta["secret_envs"])
    out["configured"] = bool(api_key and api_secret)
    if not out["configured"]:
        out["error"] = "API 密钥未配置"
        out["error_code"] = "not_configured"
        return out
    try:
        if meta["account_type"] == "futures_usdtm":
            raw = _fetch_futures_account_raw(api_key=api_key, api_secret=api_secret)
            parsed = parse_futures_account(raw)
        else:
            parsed = _fetch_spot_equity(
                api_key=api_key,
                api_secret=api_secret,
                mark_prices=mark_prices or {},
            )
        out.update(parsed)
        out["ok"] = True
        out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        logger.debug("exchange balance fetch failed scope=%s", scope, exc_info=True)
        out["error"] = str(exc)[:200]
        out["error_code"] = "network_or_auth"
    return out


def build_exchange_ledger(
    *,
    mark_prices: Optional[Mapping[str, float]] = None,
    scopes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch Binance balances for each isolated account and sum into a ledger."""
    want = scopes or ["trend", "spot", "multi_leg"]
    accounts: List[Dict[str, Any]] = []
    for scope in want:
        if scope in _SCOPE_META:
            accounts.append(
                fetch_scope_exchange_balance(scope, mark_prices=mark_prices)
            )
    equity_sum = 0.0
    wallet_sum = 0.0
    available_sum = 0.0
    exchange_upnl = 0.0
    ok_count = 0
    for row in accounts:
        if not row.get("ok"):
            continue
        ok_count += 1
        equity_sum += float(row.get("equity_usdt") or 0.0)
        wallet_sum += float(row.get("wallet_balance_usdt") or 0.0)
        available_sum += float(row.get("available_usdt") or 0.0)
        exchange_upnl += float(row.get("unrealized_pnl_usdt") or 0.0)
    return {
        "accounts": accounts,
        "totals": {
            "equity_usdt": equity_sum,
            "wallet_balance_usdt": wallet_sum,
            "available_usdt": available_sum,
            "exchange_unrealized_pnl_usdt": exchange_upnl,
            "accounts_ok": ok_count,
            "accounts_total": len(accounts),
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
