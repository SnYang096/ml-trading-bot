from __future__ import annotations

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import SETTINGS
from app.responses import ok
from app.services import ohlcv_reader
from app.services.feature_overlay import load_feature_overlay
from app.services.marker_detail import marker_detail
from app.services.ohlcv_reader import OhlcvWindowError
from app.services.trade_markers import collect_markers
from app.services.universe import load_universe_symbols

router = APIRouter(tags=["trade-map"])


def _parse_ts_param(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return int(ts.timestamp())
    except (TypeError, ValueError):
        return None


def _scopes_list(scopes: str) -> List[str]:
    return [s.strip().lower() for s in scopes.split(",") if s.strip()]


def _marker_kwargs(
    *,
    from_: Optional[str],
    to: Optional[str],
    since: Optional[str],
    include_pending: bool,
) -> dict:
    return {
        "start_ts": _parse_ts_param(from_),
        "end_ts": _parse_ts_param(to),
        "since_ts": _parse_ts_param(since),
        "include_pending": include_pending,
    }


@router.get("/api/trade-map/symbols")
def trade_map_symbols() -> dict:
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    items = []
    for sym in symbols:
        meta = ohlcv_reader.latest_bar_meta(SETTINGS.feature_bus_root, sym)
        items.append({"symbol": sym, "latest": meta})
    return ok(items)


@router.get("/api/trade-map/ohlcv")
def trade_map_ohlcv(
    symbol: str = Query(...),
    timeframe: str = Query("2h"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
) -> dict:
    start = pd.Timestamp(from_, tz="UTC") if from_ else None
    end = pd.Timestamp(to, tz="UTC") if to else None
    try:
        data = ohlcv_reader.fetch_ohlcv(
            SETTINGS.feature_bus_root,
            symbol,
            timeframe,
            start=start,
            end=end,
            max_days=SETTINGS.max_ohlcv_days,
        )
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(data)


@router.get("/api/trade-map/markers")
def trade_map_markers(
    symbol: str = Query(...),
    scopes: str = Query("trend,spot"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    include_pending: bool = Query(False),
) -> dict:
    markers = collect_markers(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        **_marker_kwargs(
            from_=from_, to=to, since=since, include_pending=include_pending
        ),
    )
    return ok(markers, meta={"count": len(markers)})


@router.get("/api/trade-map/marker-detail")
def trade_map_marker_detail(marker_id: str = Query(...)) -> dict:
    data = marker_detail(
        marker_id,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
    )
    return ok(data)


@router.get("/api/trade-map/bundle")
def trade_map_bundle(
    symbol: str = Query(...),
    timeframe: str = Query("2h"),
    scopes: str = Query("trend,spot"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    include_pending: bool = Query(False),
    overlay_weekly_ema: bool = Query(True),
) -> dict:
    start = pd.Timestamp(from_, tz="UTC") if from_ else None
    end = pd.Timestamp(to, tz="UTC") if to else None
    try:
        ohlcv = ohlcv_reader.fetch_ohlcv(
            SETTINGS.feature_bus_root,
            symbol,
            timeframe,
            start=start,
            end=end,
            max_days=SETTINGS.max_ohlcv_days,
        )
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mk = _marker_kwargs(
        from_=from_, to=to, since=since, include_pending=include_pending
    )
    markers = collect_markers(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        **mk,
    )
    overlays = {}
    if overlay_weekly_ema:
        overlays["weekly_ema_200_position"] = load_feature_overlay(
            SETTINGS.feature_bus_root,
            symbol,
            timeframe,
            start=start,
            end=end,
        )
    return ok(
        {
            "ohlcv": ohlcv,
            "markers": markers,
            "overlays": overlays,
        },
        meta={
            "poll_seconds": SETTINGS.map_poll_seconds,
            "degraded_ohlc": ohlcv.get("degraded_ohlc", False),
        },
    )
