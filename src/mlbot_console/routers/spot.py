from __future__ import annotations

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.spot_eligibility import spot_eligibility_summary

router = APIRouter(tags=["spot"])


@router.get("/api/spot/eligibility")
def spot_eligibility(
    symbol: str = Query(...),
    timeframe: str = Query("2h"),
) -> dict:
    data = spot_eligibility_summary(
        feature_bus_root=SETTINGS.feature_bus_root,
        spot_db=SETTINGS.spot_order_db,
        symbol=symbol,
        timeframe=timeframe,
    )
    return ok(data)
