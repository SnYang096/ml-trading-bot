from __future__ import annotations

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import SETTINGS
from app.responses import ok
from app.services import ohlcv_reader
from app.services.feature_overlay import (
    DEFAULT_SUBCHART_COLUMNS,
    load_feature_overlays,
)
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


def _feature_columns_list(
    feature_columns: Optional[str],
    *,
    overlay_weekly_ema: bool,
) -> List[str]:
    if feature_columns and feature_columns.strip():
        return [c.strip() for c in feature_columns.split(",") if c.strip()]
    if overlay_weekly_ema:
        return list(DEFAULT_SUBCHART_COLUMNS)
    return []


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
    full_range: bool = Query(True),
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
            full_range=full_range and not from_ and not to,
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
    feature_columns: Optional[str] = Query(None),
    overlay_weekly_ema: bool = Query(False),
    full_range: bool = Query(True),
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
            full_range=full_range and not from_ and not to,
        )
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mk = _marker_kwargs(
        from_=from_, to=to, since=since, include_pending=include_pending
    )
    if not from_ and ohlcv.get("range_start"):
        mk["start_ts"] = _parse_ts_param(str(ohlcv["range_start"]))
    if not to and ohlcv.get("range_end"):
        mk["end_ts"] = _parse_ts_param(str(ohlcv["range_end"]))
    markers = collect_markers(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        **mk,
    )
    cols = _feature_columns_list(feature_columns, overlay_weekly_ema=overlay_weekly_ema)
    overlays: dict = {}
    if cols:
        overlay_start = start
        overlay_end = end
        if overlay_start is None and ohlcv.get("range_start"):
            overlay_start = pd.Timestamp(str(ohlcv["range_start"]))
        if overlay_end is None and ohlcv.get("range_end"):
            overlay_end = pd.Timestamp(str(ohlcv["range_end"]))
        overlays = load_feature_overlays(
            SETTINGS.feature_bus_root,
            symbol,
            timeframe,
            cols,
            start=overlay_start,
            end=overlay_end,
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
            "bars_1min_rows": ohlcv.get("bars_1min_rows"),
            "range_start": ohlcv.get("range_start"),
            "range_end": ohlcv.get("range_end"),
            "range_clipped": ohlcv.get("range_clipped", False),
            "feature_columns": cols,
        },
    )
