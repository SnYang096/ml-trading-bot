from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.orders_list import (
    collect_orders,
    enrich_orders_pnl,
    multi_leg_orders_list,
    spot_orders_list,
    trend_orders,
)
from mlbot_console.services.strategy_registry import strategy_account_layer
from mlbot_console.services.trend_funnel import fetch_funnel_snapshots

router = APIRouter(tags=["orders"])


def _scopes_list(scopes: str) -> List[str]:
    return [s.strip().lower() for s in scopes.split(",") if s.strip()]


@router.get("/api/orders/list")
def orders_list(
    symbol: str = Query("*"),
    scopes: str = Query("trend,spot"),
    status: Optional[str] = Query(None),
    exclude_status: str = Query(
        "",
        description="Comma-separated statuses to omit (e.g. expired,canceled)",
    ),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    exclude_statuses = [
        s.strip().lower() for s in exclude_status.split(",") if s.strip()
    ]
    rows = collect_orders(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        status=status,
        exclude_statuses=exclude_statuses or None,
        limit=limit,
        feature_bus_root=SETTINGS.feature_bus_root,
        engine_data_root=SETTINGS.engine_data_root,
    )
    sym_meta = "ALL" if str(symbol).strip().upper() in {"", "*", "ALL", "__ALL__"} else symbol.upper()
    return ok(rows, meta={"count": len(rows), "symbol": sym_meta})


@router.get("/api/trend/orders")
def trend_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    exclude_status: str = Query(""),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    exclude_statuses = [
        s.strip().lower() for s in exclude_status.split(",") if s.strip()
    ]
    fetch_limit = limit
    if exclude_statuses:
        from mlbot_console.services.orders_list import _effective_fetch_limit

        fetch_limit = _effective_fetch_limit(limit, exclude_statuses)
    rows = trend_orders(
        SETTINGS.trend_order_db,
        symbol,
        status=status,
        exclude_statuses=exclude_statuses or None,
        limit=fetch_limit,
    )
    rows = rows[:limit]
    enrich_orders_pnl(
        rows,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
    )
    return ok(rows, meta={"count": len(rows)})


@router.get("/api/trend/funnel")
def trend_funnel_api(
    symbol: str = Query("", description="Empty or * = all symbols in snapshot"),
    account_layer: str = Query("", description="trend | spot | multi_leg"),
    strategy: str = Query("", description="Specific strategy id, e.g. bpc or chop_grid"),
    limit: int = Query(96, ge=1, le=500, description="Recent 15min windows"),
) -> dict:
    rows = fetch_funnel_snapshots(
        SETTINGS.live_monitor_db,
        symbol=symbol,
        limit=limit,
    )
    if account_layer or strategy:
        filtered: list = []
        layer_filter = account_layer.strip().lower()
        strat_filter = strategy.strip().lower()
        for snap in rows:
            bys = snap.get("by_strategy") or {}
            if not isinstance(bys, dict):
                continue
            subset = {}
            for sid, st in bys.items():
                sid_l = str(sid).lower()
                if strat_filter and sid_l != strat_filter:
                    continue
                if layer_filter and strategy_account_layer(sid_l) != layer_filter:
                    continue
                subset[sid] = st
            if not subset:
                continue
            snap_copy = dict(snap)
            snap_copy["by_strategy"] = subset
            filtered.append(snap_copy)
        rows = filtered
    return ok(
        rows,
        meta={
            "count": len(rows),
            "db": str(SETTINGS.live_monitor_db),
            "symbol": symbol or "*",
            "account_layer": account_layer or "*",
            "strategy": strategy or "*",
        },
    )


@router.get("/api/spot/orders")
def spot_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    exclude_status: str = Query(""),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    exclude_statuses = [
        s.strip().lower() for s in exclude_status.split(",") if s.strip()
    ]
    from mlbot_console.services.orders_list import _effective_fetch_limit

    fetch_limit = (
        _effective_fetch_limit(limit, exclude_statuses)
        if exclude_statuses
        else limit
    )
    rows = spot_orders_list(
        SETTINGS.spot_order_db,
        symbol,
        status=status,
        exclude_statuses=exclude_statuses or None,
        limit=fetch_limit,
    )
    rows = rows[:limit]
    enrich_orders_pnl(
        rows,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
    )
    return ok(rows, meta={"count": len(rows)})


@router.get("/api/multileg/orders")
def multileg_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    exclude_status: str = Query(""),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    exclude_statuses = [
        s.strip().lower() for s in exclude_status.split(",") if s.strip()
    ]
    from mlbot_console.services.orders_list import _effective_fetch_limit

    fetch_limit = (
        _effective_fetch_limit(limit, exclude_statuses)
        if exclude_statuses
        else limit
    )
    rows = multi_leg_orders_list(
        SETTINGS.multi_leg_db,
        symbol,
        status=status,
        exclude_statuses=exclude_statuses or None,
        limit=fetch_limit,
        engine_data_root=SETTINGS.engine_data_root,
    )
    rows = rows[:limit]
    enrich_orders_pnl(
        rows,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
    )
    return ok(rows, meta={"count": len(rows)})
