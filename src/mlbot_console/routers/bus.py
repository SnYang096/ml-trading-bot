from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services import ohlcv_reader
from mlbot_console.services.feature_overlay import list_feature_columns
from mlbot_console.services.feature_taxonomy import get_feature_taxonomy
from mlbot_console.services.ohlcv_reader import (
    OhlcvWindowError,
    assert_trade_map_timeframe,
)

router = APIRouter(tags=["bus"])


def _parse_range(
    from_: Optional[str],
    to: Optional[str],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    start = pd.Timestamp(from_, tz="UTC") if from_ else None
    end = pd.Timestamp(to, tz="UTC") if to else None
    return start, end


@router.get("/api/bus/ohlcv")
def bus_ohlcv(
    symbol: str = Query(...),
    timeframe: str = Query("2h"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    full_range: bool = Query(True),
) -> dict:
    start, end = _parse_range(from_, to)
    try:
        data = ohlcv_reader.fetch_ohlcv(
            SETTINGS.feature_bus_root,
            symbol,
            timeframe,
            start=start,
            end=end,
            max_days=SETTINGS.max_ohlcv_days,
            full_range=full_range and not from_ and not to,
            live_storage_bars_root=SETTINGS.live_storage_bars_root,
            stitch_live_storage=SETTINGS.stitch_live_storage,
            macro_kline_root=SETTINGS.macro_spot_kline_root,
            daily_ohlcv_start=SETTINGS.daily_ohlcv_start,
            max_daily_ohlcv_days=SETTINGS.max_daily_ohlcv_days,
        )
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(data, meta={"degraded_ohlc": data.get("degraded_ohlc", False)})


@router.get("/api/bus/features/columns")
def bus_feature_columns(
    symbol: str = Query(...),
    timeframe: str = Query("2h"),
) -> dict:
    try:
        tf = assert_trade_map_timeframe(timeframe)
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = list_feature_columns(SETTINGS.feature_bus_root, symbol, tf)
    return ok(data)


@router.get("/api/bus/features/taxonomy")
def bus_feature_taxonomy() -> dict:
    """Strategy × pipeline stage feature map from archetype YAML (read-only)."""
    return ok(get_feature_taxonomy())
