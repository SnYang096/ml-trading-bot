from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from app.config import SETTINGS
from app.responses import ok
from app.services.orders_list import collect_orders, multi_leg_orders_list, spot_orders_list, trend_orders

router = APIRouter(tags=["orders"])


def _scopes_list(scopes: str) -> List[str]:
    return [s.strip().lower() for s in scopes.split(",") if s.strip()]


@router.get("/api/orders/list")
def orders_list(
    symbol: str = Query("*"),
    scopes: str = Query("trend,spot"),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = collect_orders(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        status=status,
        limit=limit,
    )
    sym_meta = "ALL" if str(symbol).strip().upper() in {"", "*", "ALL", "__ALL__"} else symbol.upper()
    return ok(rows, meta={"count": len(rows), "symbol": sym_meta})


@router.get("/api/trend/orders")
def trend_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = trend_orders(
        SETTINGS.trend_order_db,
        symbol,
        status=status,
        limit=limit,
    )
    return ok(rows, meta={"count": len(rows)})


@router.get("/api/spot/orders")
def spot_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = spot_orders_list(
        SETTINGS.spot_order_db,
        symbol,
        status=status,
        limit=limit,
    )
    return ok(rows, meta={"count": len(rows)})


@router.get("/api/multileg/orders")
def multileg_orders_api(
    symbol: str = Query(...),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = multi_leg_orders_list(
        SETTINGS.multi_leg_db,
        symbol,
        status=status,
        limit=limit,
    )
    return ok(rows, meta={"count": len(rows)})
