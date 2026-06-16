"""Account overview API — aggregate PnL and order stats."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.account_reconciliation import (
    reconcile_account,
    reconcile_all_accounts,
)
from mlbot_console.services.account_pnl_reconciliation import (
    reconcile_pnl_vs_exchange,
    reconcile_realized_pnl,
)
from mlbot_console.services.account_summary import build_account_summary
from mlbot_console.services.db import query_rows
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
        account_snapshot_db=SETTINGS.account_snapshot_db,
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


@router.get("/api/account/reconciliation/realized")
def account_realized_reconciliation(
    scope: str = Query("multi_leg", description="trend or multi_leg"),
    lookback_days: int = Query(
        90,
        ge=1,
        le=365,
        description="How many days of income to compare (default 90)",
    ),
) -> dict:
    """Compare local DB realized PnL vs Binance /fapi/v1/income for a scope.

    Returns the gap between local PnL calculation and exchange-reported
    realized PnL, commission, and funding fees.
    """
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_days * 86_400_000

    # Get local realized PnL from account summary
    summary = build_account_summary(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol="*",
        lookback_days=lookback_days,
    )

    local_realized = 0.0
    local_commission = 0.0
    for s in summary.get("scopes") or []:
        if str(s.get("scope")) == scope:
            local_realized = float(s.get("realized_pnl") or 0.0)
            break

    # Also sum commission from multi_leg_orders if scope is multi_leg
    if scope == "multi_leg":
        try:
            rows = query_rows(
                SETTINGS.multi_leg_db,
                "SELECT COALESCE(SUM(commission), 0) as total_commission "
                "FROM multi_leg_orders "
                "WHERE status IN ('FILLED', 'PARTIALLY_FILLED') "
                "AND error_message IS DISTINCT FROM 'bug' "
                "AND filled_at >= datetime('now', ?)",
                (f"-{lookback_days} days",),
            )
            if rows:
                local_commission = abs(float(rows[0].get("total_commission") or 0.0))
        except Exception:
            pass

    data = reconcile_realized_pnl(
        scope=scope,
        local_realized_pnl=local_realized,
        local_commission=local_commission,
        symbol="*",
        start_time_ms=start_ms,
        end_time_ms=now_ms,
    )
    data["lookback_days"] = lookback_days
    return ok(data)
