"""Account overview API — aggregate PnL and order stats."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.account_reconciliation import (
    reconcile_account,
    reconcile_all_accounts,
)
from mlbot_console.services.account_pnl_reconciliation import reconcile_pnl_vs_exchange
from mlbot_console.services.account_summary import build_account_summary
from mlbot_console.services.mark_prices import fetch_mark_prices
from mlbot_console.services.universe import load_universe_symbols

router = APIRouter(tags=["account"])


@router.get("/api/account/summary")
def account_summary(
    symbol: str = Query("*"),
    scopes: str = Query(
        "",
        description="Comma-separated: trend,spot,multi_leg (empty = all)",
    ),
    lookback_days: int = Query(
        0,
        ge=0,
        le=3650,
        description="0 = all history; otherwise days of realized PnL lookback",
    ),
) -> dict:
    scope_list = [s.strip() for s in scopes.split(",") if s.strip()] or None
    data = build_account_summary(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
        lookback_days=lookback_days,
        scopes=scope_list,
    )
    return ok(data, meta={"symbol": data.get("symbol"), "lookback_days": lookback_days})


@router.get("/api/account/reconciliation")
def account_reconciliation(
    scope: str = Query(..., description="trend, spot, or multi_leg"),
) -> dict:
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    marks = fetch_mark_prices(SETTINGS.feature_bus_root, symbols)
    
    data = reconcile_account(
        scope=scope,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        mark_prices=marks,
    )
    return ok(data)


@router.get("/api/account/reconciliation/pnl")
def account_pnl_reconciliation(
    symbol: str = Query("*"),
    lookback_days: int = Query(0, ge=0, le=3650),
) -> dict:
    data = reconcile_pnl_vs_exchange(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
        lookback_days=lookback_days,
    )
    return ok(data)


@router.get("/api/account/reconciliation/all")
def account_reconciliation_all(
    symbol: str = Query("*"),
    lookback_days: int = Query(0, ge=0, le=3650),
) -> dict:
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    marks = fetch_mark_prices(SETTINGS.feature_bus_root, symbols)
    data = reconcile_all_accounts(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        mark_prices=marks,
        symbol=symbol,
        lookback_days=lookback_days,
    )
    return ok(data)
