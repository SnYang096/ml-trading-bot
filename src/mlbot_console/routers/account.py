"""Account overview API — aggregate PnL and order stats."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.account_summary import build_account_summary

router = APIRouter(tags=["account"])


@router.get("/api/account/summary")
def account_summary(
    symbol: str = Query("*"),
    lookback_days: int = Query(
        0,
        ge=0,
        le=3650,
        description="0 = all history; otherwise days of realized PnL lookback",
    ),
) -> dict:
    data = build_account_summary(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
        lookback_days=lookback_days,
    )
    return ok(data, meta={"symbol": data.get("symbol"), "lookback_days": lookback_days})
