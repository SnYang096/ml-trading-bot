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


def _fetch_open_orders_raw(*, api_key: str, api_secret: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    session = _http_session()
    base = "https://fapi.binance.com"
    srv = session.get(f"{base}/fapi/v1/time", timeout=8)
    srv.raise_for_status()
    server_ts = int(srv.json().get("serverTime", int(time.time() * 1000)))
    params = {"timestamp": server_ts}
    if symbol:
        params["symbol"] = symbol.upper()
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = session.get(
        f"{base}/fapi/v1/openOrders?{query}&signature={sig}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("unexpected open orders response")
    return data


from mlbot_console.services.symbols import is_all_symbols


def _symbol_base_asset(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if sym.endswith("USDT") and len(sym) > 4:
        return sym[:-4]
    return sym


def _compute_position_unrealized_pnl(pos: Mapping[str, Any]) -> float:
    """Per-position unrealized PnL from entry/mark (Binance per-leg field is unreliable).

    Prefer manual computation from entryPrice × markPrice when both are available,
    falling back to ``unRealizedProfit`` only when mark data is missing.
    """
    try:
        amt = float(pos.get("positionAmt") or 0.0)
        entry = float(pos.get("entryPrice") or 0.0)
        mark = float(pos.get("markPrice") or 0.0)
    except (TypeError, ValueError):
        return float(pos.get("unRealizedProfit") or 0.0)
    if amt == 0.0:
        return 0.0
    if entry > 0 and mark > 0:
        if amt > 0:
            return (mark - entry) * amt
        else:
            return (entry - mark) * abs(amt)
    return float(pos.get("unRealizedProfit") or 0.0)


def _parse_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def futures_open_positions(
    data: Mapping[str, Any],
    *,
    symbol: Optional[str] = None,
    mark_prices: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Non-flat futures legs from ``/fapi/v2/account`` ``positions`` array."""
    marks = mark_prices or {}
    sym_filter = (
        str(symbol).upper()
        if symbol and not is_all_symbols(symbol)
        else ""
    )
    out: List[Dict[str, Any]] = []
    for pos in data.get("positions") or []:
        sym = str(pos.get("symbol") or "").upper()
        if sym_filter and sym != sym_filter:
            continue
        try:
            amt = float(pos.get("positionAmt") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt == 0.0:
            continue
        mark = _parse_float(pos.get("markPrice"))
        if mark <= 0:
            mark = _parse_float(marks.get(sym))
        if mark <= 0:
            entry = _parse_float(pos.get("entryPrice"))
            if entry > 0:
                mark = entry
        entry = _parse_float(pos.get("entryPrice"))
        notional = abs(amt) * mark if mark > 0 else abs(_parse_float(pos.get("notional")))
        lev = int(_parse_float(pos.get("leverage")) or 0)
        init_margin = _parse_float(
            pos.get("positionInitialMargin") or pos.get("initialMargin")
        )
        upnl_pos = {**pos, "markPrice": mark, "entryPrice": entry}
        out.append(
            {
                "symbol": sym,
                "side": "long" if amt > 0 else "short",
                "quantity": abs(amt),
                "position_amt": amt,
                "entry_price": entry,
                "mark_price": mark,
                "notional_usdt": round(notional, 4),
                "leverage": lev if lev > 0 else None,
                "initial_margin_usdt": init_margin if init_margin > 0 else None,
                "maint_margin_usdt": _parse_float(pos.get("maintMargin")) or None,
                "margin_type": str(pos.get("marginType") or "").lower() or None,
                "unrealized_pnl_usdt": _compute_position_unrealized_pnl(upnl_pos),
                "liquidation_price": _parse_float(pos.get("liquidationPrice")) or None,
            }
        )
    return sorted(out, key=lambda x: (x["symbol"], x["side"]))


def futures_symbol_unrealized_pnl(
    data: Mapping[str, Any], symbol: str
) -> float:
    """Sum per-position unrealized PnL (computed from entry/mark, not Binance field)."""
    sym = str(symbol).upper()
    total = 0.0
    for pos in data.get("positions") or []:
        if str(pos.get("symbol") or "").upper() != sym:
            continue
        try:
            amt = float(pos.get("positionAmt") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt == 0.0:
            continue
        total += _compute_position_unrealized_pnl(pos)
    return total


def spot_symbol_holdings_value(
    holdings: List[Mapping[str, Any]], symbol: str
) -> float:
    asset = _symbol_base_asset(symbol)
    total = 0.0
    for row in holdings:
        if str(row.get("asset") or "").upper() != asset:
            continue
        total += float(row.get("value_usdt") or 0.0)
    return total


def futures_gross_notional(positions: List[Mapping[str, Any]]) -> float:
    return round(
        sum(abs(_parse_float(p.get("notional_usdt"))) for p in positions),
        4,
    )


def parse_futures_account(
    data: Mapping[str, Any],
    *,
    open_positions: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    margin_bal = _parse_float(data.get("totalMarginBalance"))
    maint_margin = _parse_float(data.get("totalMaintMargin"))
    available = _parse_float(data.get("availableBalance"))
    pos_init = _parse_float(data.get("totalPositionInitialMargin"))
    order_init = _parse_float(data.get("totalOpenOrderInitialMargin"))
    margin_locked = max(0.0, margin_bal - available)
    legs = list(open_positions or [])
    gross_notional = futures_gross_notional(legs)
    gross_leverage: Optional[float] = None
    if margin_bal > 0 and gross_notional > 0:
        gross_leverage = round(gross_notional / margin_bal, 4)
    margin_ratio: Optional[float] = None
    if margin_bal > 0:
        margin_ratio = round(maint_margin / margin_bal, 6)
    return {
        "wallet_balance_usdt": _parse_float(data.get("totalWalletBalance")),
        "equity_usdt": margin_bal,
        "available_usdt": available,
        "unrealized_pnl_usdt": _parse_float(data.get("totalUnrealizedProfit")),
        "maint_margin_usdt": maint_margin,
        "margin_ratio": margin_ratio,
        "position_initial_margin_usdt": pos_init,
        "open_order_initial_margin_usdt": order_init,
        "margin_locked_usdt": round(margin_locked, 4),
        "gross_notional_usdt": gross_notional,
        "gross_leverage": gross_leverage,
    }


def _fetch_position_risk_raw(*, api_key: str, api_secret: str) -> List[Dict[str, Any]]:
    session = _http_session()
    base = "https://fapi.binance.com"
    srv = session.get(f"{base}/fapi/v1/time", timeout=8)
    srv.raise_for_status()
    server_ts = int(srv.json().get("serverTime", int(time.time() * 1000)))
    query = f"timestamp={server_ts}"
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = session.get(
        f"{base}/fapi/v2/positionRisk?{query}&signature={sig}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("unexpected position risk response")
    return data


def futures_symbol_leverage_map(data: Mapping[str, Any]) -> Dict[str, int]:
    """Per-symbol leverage from ``/fapi/v2/account`` positions (includes flat symbols)."""
    out: Dict[str, int] = {}
    for pos in data.get("positions") or []:
        sym = str(pos.get("symbol") or "").upper()
        lev = int(_parse_float(pos.get("leverage")) or 0)
        if sym and lev > 0:
            out[sym] = lev
    return out


def futures_leverage_map_from_risk(risk_rows: List[Mapping[str, Any]]) -> Dict[str, int]:
    """Leverage per symbol from positionRisk (covers symbols with orders but flat position)."""
    out: Dict[str, int] = {}
    for row in risk_rows:
        sym = str(row.get("symbol") or "").upper()
        lev = int(_parse_float(row.get("leverage")) or 0)
        if sym and lev > 0:
            out[sym] = lev
    return out


def merge_leverage_maps(*maps: Mapping[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for m in maps:
        for sym, lev in m.items():
            if lev > 0:
                merged[str(sym).upper()] = int(lev)
    return merged


def merge_liquidation_from_position_risk(
    positions: List[Dict[str, Any]], risk_rows: List[Mapping[str, Any]]
) -> None:
    """Fill missing liquidation_price from /fapi/v2/positionRisk.

    Mutates ``positions`` list in place (each dict may get ``liquidation_price``).
    """
    liq_by_key: Dict[tuple, float] = {}
    for row in risk_rows:
        sym = str(row.get("symbol") or "").upper()
        try:
            amt = float(row.get("positionAmt") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt == 0.0:
            continue
        pos_side = str(row.get("positionSide") or "BOTH").upper()
        side = "long" if amt > 0 else "short"
        liq = _parse_float(row.get("liquidationPrice"))
        if liq > 0:
            liq_by_key[(sym, side)] = liq
            if pos_side in {"LONG", "SHORT"}:
                liq_by_key[(sym, pos_side.lower())] = liq
    for pos in positions:
        if _parse_float(pos.get("liquidation_price")) > 0:
            continue
        sym = str(pos.get("symbol") or "").upper()
        side = str(pos.get("side") or "long").lower()
        liq = liq_by_key.get((sym, side))
        if liq:
            pos["liquidation_price"] = liq


def _order_price_for_margin(o: Mapping[str, Any], marks: Mapping[str, float]) -> float:
    for key in ("price", "stopPrice", "activatePrice"):
        px = _parse_float(o.get(key))
        if px > 0:
            return px
    sym = str(o.get("symbol") or "").upper()
    return _parse_float(marks.get(sym))


def distribute_open_order_margin_fallback(
    orders: List[Dict[str, Any]], *, total_margin: float
) -> None:
    """Allocate account open-order margin across rows missing per-order values."""
    if total_margin <= 0:
        return
    need = [
        o
        for o in orders
        if o.get("initial_margin_usdt") is None and not o.get("reduce_only")
    ]
    if not need:
        return
    weights: List[float] = []
    for o in need:
        px = _parse_float(o.get("price"))
        qty = _parse_float(o.get("quantity"))
        weights.append(px * qty if px > 0 and qty > 0 else 1.0)
    wsum = sum(weights) or float(len(need))
    for o, w in zip(need, weights):
        o["initial_margin_usdt"] = round(total_margin * w / wsum, 4)
        o["margin_estimated"] = True
        o["margin_allocated"] = True


def parse_open_orders_margin(
    orders: List[Dict[str, Any]],
    *,
    leverage_by_symbol: Optional[Mapping[str, float]] = None,
    mark_prices: Optional[Mapping[str, float]] = None,
    total_open_order_margin: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Parse or estimate initial margin for open orders.

    ``/fapi/v1/openOrders`` usually omits ``initialMargin``; estimate as
    ``price * qty / leverage`` when leverage map is available.
    """
    lev_map = leverage_by_symbol or {}
    marks = mark_prices or {}
    out: List[Dict[str, Any]] = []
    for o in orders:
        sym = str(o.get("symbol") or "").upper()
        side = str(o.get("side") or "").lower()
        pos_side = str(o.get("positionSide") or "").upper()
        oid = str(o.get("orderId") or "")
        reduce_only = str(o.get("reduceOnly") or "").lower() in {"true", "1"}
        init_margin = _parse_float(o.get("initialMargin"))
        margin_estimated = False
        if reduce_only:
            init_margin = 0.0
        elif init_margin <= 0:
            lev = _parse_float(lev_map.get(sym))
            price = _order_price_for_margin(o, marks)
            qty = _parse_float(o.get("origQty"))
            if lev > 0 and price > 0 and qty > 0:
                init_margin = round(price * qty / lev, 4)
                margin_estimated = True
        px_out = _order_price_for_margin(o, marks)
        out.append(
            {
                "order_id": oid,
                "client_order_id": str(o.get("clientOrderId") or ""),
                "symbol": sym,
                "side": side,
                "position_side": pos_side if pos_side in {"LONG", "SHORT"} else None,
                "type": str(o.get("type") or "").lower(),
                "price": px_out if px_out > 0 else None,
                "quantity": _parse_float(o.get("origQty")),
                "initial_margin_usdt": init_margin if init_margin > 0 else (0.0 if reduce_only else None),
                "margin_estimated": margin_estimated,
                "margin_allocated": False,
                "reduce_only": reduce_only,
                "leverage": int(_parse_float(lev_map.get(sym))) or None,
                "status": str(o.get("status") or "").lower(),
            }
        )
    distribute_open_order_margin_fallback(
        out, total_margin=_parse_float(total_open_order_margin)
    )
    return out


def _fetch_spot_equity(
    *,
    api_key: str,
    api_secret: str,
    mark_prices: Mapping[str, float],
) -> Dict[str, Any]:
    from mlbot_console.services.spot_ccxt import spot_binance_exchange

    exchange = spot_binance_exchange(api_key=api_key, api_secret=api_secret)
    exchange.load_markets()
    bal = exchange.fetch_balance()
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
            tickers = exchange.fetch_tickers()
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
    usdt_locked = max(0.0, total_usdt - free_usdt)
    margin_locked = max(0.0, equity - free_usdt)
    cash_ratio: Optional[float] = None
    if equity > 0:
        cash_ratio = round(free_usdt / equity, 6)

    return {
        "wallet_balance_usdt": equity,  # Total equity in USDT
        "equity_usdt": equity,
        "available_usdt": free_usdt,
        "unrealized_pnl_usdt": 0.0,
        "cash_ratio": cash_ratio,
        "usdt_cash": total_usdt,
        "usdt_locked_usdt": round(usdt_locked, 4),
        "holdings_value_usdt": holdings_value_usdt,
        "margin_locked_usdt": round(margin_locked, 4),
        "holdings": sorted(holdings, key=lambda x: x["value_usdt"], reverse=True),
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
        "maint_margin_usdt": None,
        "margin_ratio": None,
        "cash_ratio": None,
        "position_initial_margin_usdt": None,
        "open_order_initial_margin_usdt": None,
        "margin_locked_usdt": None,
        "gross_notional_usdt": None,
        "gross_leverage": None,
        "usdt_locked_usdt": None,
        "holdings_value_usdt": None,
        "fetched_at": None,
    }


def fetch_scope_exchange_balance(
    scope: str,
    *,
    mark_prices: Optional[Mapping[str, float]] = None,
    symbol: Optional[str] = None,
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
    sym_filter = str(symbol or "").strip().upper()
    symbol_scoped = sym_filter and not is_all_symbols(sym_filter)
    try:
        if meta["account_type"] == "futures_usdtm":
            raw = _fetch_futures_account_raw(api_key=api_key, api_secret=api_secret)
            open_positions = futures_open_positions(
                raw,
                symbol=sym_filter if symbol_scoped else None,
                mark_prices=mark_prices or {},
            )
            try:
                risk_rows = _fetch_position_risk_raw(api_key=api_key, api_secret=api_secret)
                merge_liquidation_from_position_risk(open_positions, risk_rows)
            except Exception:
                logger.debug("positionRisk fetch failed scope=%s", scope, exc_info=True)
            parsed = parse_futures_account(raw, open_positions=open_positions)
            account_upnl = float(parsed.get("unrealized_pnl_usdt") or 0.0)
            parsed = dict(parsed)
            parsed["exchange_open_positions"] = open_positions
            parsed["exchange_open_position_count"] = len(open_positions)
            parsed["account_unrealized_pnl_usdt"] = account_upnl
            if symbol_scoped:
                sym_upnl = futures_symbol_unrealized_pnl(raw, sym_filter)
                parsed["symbol_unrealized_pnl_usdt"] = sym_upnl
                if sym_upnl == 0.0 and account_upnl != 0.0 and open_positions:
                    # Binance per-position unRealizedProfit can be 0 while
                    # account-level totalUnrealizedProfit is correct.
                    # Fall back to account-level to avoid showing 0 wrongly.
                    logger.debug(
                        "futures symbol_unrealized_pnl=0 for %s (account_upnl=%.2f, "
                        "%d open legs); keeping account-level unrealized",
                        sym_filter,
                        account_upnl,
                        len(open_positions),
                    )
                    parsed["unrealized_pnl_usdt"] = account_upnl
                    out["unrealized_pnl_basis"] = "account"
                else:
                    parsed["unrealized_pnl_usdt"] = sym_upnl
                    out["unrealized_pnl_basis"] = "symbol"
            else:
                parsed["symbol_unrealized_pnl_usdt"] = account_upnl
                out["unrealized_pnl_basis"] = "account"
        else:
            parsed = _fetch_spot_equity(
                api_key=api_key,
                api_secret=api_secret,
                mark_prices=mark_prices or {},
            )
            if symbol_scoped:
                parsed = dict(parsed)
                holdings = list(parsed.get("holdings") or [])
                parsed["holdings_value_usdt"] = spot_symbol_holdings_value(
                    holdings, sym_filter
                )
                out["unrealized_pnl_basis"] = "symbol"
            else:
                out["unrealized_pnl_basis"] = "account"
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
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch Binance balances for each isolated account and sum into a ledger."""
    want = scopes or ["trend", "spot", "multi_leg"]
    sym_meta = (
        str(symbol).upper()
        if symbol and not is_all_symbols(symbol)
        else "ALL"
    )
    accounts: List[Dict[str, Any]] = []
    for scope in want:
        if scope in _SCOPE_META:
            accounts.append(
                fetch_scope_exchange_balance(
                    scope, mark_prices=mark_prices, symbol=symbol
                )
            )
    equity_sum = 0.0
    wallet_sum = 0.0
    available_sum = 0.0
    exchange_upnl = 0.0
    margin_locked_sum = 0.0
    pos_init_sum = 0.0
    order_init_sum = 0.0
    gross_notional_sum = 0.0
    ok_count = 0
    for row in accounts:
        if not row.get("ok"):
            continue
        ok_count += 1
        equity_sum += float(row.get("equity_usdt") or 0.0)
        wallet_sum += float(row.get("wallet_balance_usdt") or 0.0)
        available_sum += float(row.get("available_usdt") or 0.0)
        margin_locked_sum += float(row.get("margin_locked_usdt") or 0.0)
        pos_init_sum += float(row.get("position_initial_margin_usdt") or 0.0)
        order_init_sum += float(row.get("open_order_initial_margin_usdt") or 0.0)
        gross_notional_sum += float(row.get("gross_notional_usdt") or 0.0)
        exchange_upnl += float(
            row.get("account_unrealized_pnl_usdt")
            if row.get("account_unrealized_pnl_usdt") is not None
            else row.get("unrealized_pnl_usdt")
            or 0.0
        )
    gross_leverage_sum: Optional[float] = None
    if equity_sum > 0 and gross_notional_sum > 0:
        gross_leverage_sum = round(gross_notional_sum / equity_sum, 4)
    return {
        "symbol": sym_meta,
        "accounts": accounts,
        "totals": {
            "equity_usdt": equity_sum,
            "wallet_balance_usdt": wallet_sum,
            "available_usdt": available_sum,
            "margin_locked_usdt": round(margin_locked_sum, 4),
            "position_initial_margin_usdt": round(pos_init_sum, 4),
            "open_order_initial_margin_usdt": round(order_init_sum, 4),
            "gross_notional_usdt": round(gross_notional_sum, 4),
            "gross_leverage": gross_leverage_sum,
            "exchange_unrealized_pnl_usdt": exchange_upnl,
            "accounts_ok": ok_count,
            "accounts_total": len(accounts),
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
