from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services import ohlcv_reader
from mlbot_console.services.feature_overlay import (
    DEFAULT_SUBCHART_COLUMNS,
    load_feature_overlays,
)
from mlbot_console.services.marker_detail import marker_detail
from mlbot_console.services.main_chart_overlays import (
    load_main_chart_overlays,
    parse_main_overlay_keys,
)
from mlbot_console.services.ohlcv_reader import (
    OhlcvWindowError,
    assert_trade_map_timeframe,
    cap_window_to_max_days,
    resolve_trade_map_window,
)
from mlbot_console.services.signal_overview import build_signal_overview
from mlbot_console.services.trade_links import collect_trade_links
from mlbot_console.services.trade_markers import (
    align_markers_to_candles,
    collect_markers,
    marker_scope_counts,
)
from mlbot_console.services.universe import load_universe_symbols

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
    full_range: bool = Query(False),
) -> dict:
    try:
        tf = assert_trade_map_timeframe(timeframe)
        start, end, use_full = resolve_trade_map_window(
            tf,
            start=pd.Timestamp(from_, tz="UTC") if from_ else None,
            end=pd.Timestamp(to, tz="UTC") if to else None,
            full_range=full_range,
        )
        max_days = (
            SETTINGS.max_daily_ohlcv_days
            if tf in ("1d", "1w")
            else SETTINGS.max_ohlcv_days
        )
        start, end = cap_window_to_max_days(start, end, max_days)
        data = ohlcv_reader.fetch_ohlcv(
            SETTINGS.feature_bus_root,
            symbol,
            tf,
            start=start,
            end=end,
            max_days=SETTINGS.max_ohlcv_days,
            full_range=use_full,
            live_storage_bars_root=SETTINGS.live_storage_bars_root,
            stitch_live_storage=SETTINGS.stitch_live_storage,
            macro_kline_root=SETTINGS.macro_spot_kline_root,
            daily_ohlcv_start=SETTINGS.daily_ohlcv_start,
            max_daily_ohlcv_days=SETTINGS.max_daily_ohlcv_days,
            live_data_root=SETTINGS.live_data_root,
            live_root=SETTINGS.live_root,
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
    if include_pending and not from_ and not to:
        try:
            ohlcv = ohlcv_reader.fetch_ohlcv(
                SETTINGS.feature_bus_root,
                symbol,
                "2h",
                max_days=SETTINGS.max_ohlcv_days,
                full_range=True,
                live_storage_bars_root=SETTINGS.live_storage_bars_root,
                stitch_live_storage=SETTINGS.stitch_live_storage,
                macro_kline_root=SETTINGS.macro_spot_kline_root,
                daily_ohlcv_start=SETTINGS.daily_ohlcv_start,
                max_daily_ohlcv_days=SETTINGS.max_daily_ohlcv_days,
                live_data_root=SETTINGS.live_data_root,
                live_root=SETTINGS.live_root,
            )
            times = [
                int(c["time"])
                for c in ohlcv.get("candles") or []
                if c.get("time") is not None
            ]
            markers = align_markers_to_candles(markers, times)
        except OhlcvWindowError:
            pass
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
    main_overlays: Optional[str] = Query(
        None,
        description="Comma-separated: ema_1200, weekly_ema_200 (2h feature bus)",
    ),
    full_range: bool = Query(False),
    include_ohlcv: str = Query(
        "full",
        description="full | tail (small OHLCV slice) | none (markers/links only)",
    ),
    ohlcv_from: Optional[str] = Query(None, alias="ohlcv_from"),
    ohlcv_to: Optional[str] = Query(None, alias="ohlcv_to"),
    include_features: bool = Query(True),
) -> dict:
    ohlcv_mode = (include_ohlcv or "full").strip().lower()
    if ohlcv_mode not in ("full", "tail", "none"):
        raise HTTPException(
            status_code=400,
            detail=f"include_ohlcv must be full|tail|none (got {include_ohlcv!r})",
        )
    ohlcv: dict = {"candles": []}
    tf = ""
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    use_full = False
    try:
        tf = assert_trade_map_timeframe(timeframe)
        if ohlcv_mode != "none":
            ohlcv_start_raw = (
                ohlcv_from if ohlcv_mode == "tail" and ohlcv_from else from_
            )
            ohlcv_end_raw = ohlcv_to if ohlcv_mode == "tail" and ohlcv_to else to
            start, end, use_full = resolve_trade_map_window(
                tf,
                start=(
                    pd.Timestamp(ohlcv_start_raw, tz="UTC") if ohlcv_start_raw else None
                ),
                end=pd.Timestamp(ohlcv_end_raw, tz="UTC") if ohlcv_end_raw else None,
                full_range=full_range and ohlcv_mode == "full",
            )
            max_days = (
                SETTINGS.max_daily_ohlcv_days
                if tf in ("1d", "1w")
                else SETTINGS.max_ohlcv_days
            )
            start, end = cap_window_to_max_days(start, end, max_days)
            ohlcv = ohlcv_reader.fetch_ohlcv(
                SETTINGS.feature_bus_root,
                symbol,
                tf,
                start=start,
                end=end,
                max_days=SETTINGS.max_ohlcv_days,
                full_range=use_full,
                live_storage_bars_root=SETTINGS.live_storage_bars_root,
                stitch_live_storage=SETTINGS.stitch_live_storage,
                macro_kline_root=SETTINGS.macro_spot_kline_root,
                daily_ohlcv_start=SETTINGS.daily_ohlcv_start,
                max_daily_ohlcv_days=SETTINGS.max_daily_ohlcv_days,
                live_data_root=SETTINGS.live_data_root,
                live_root=SETTINGS.live_root,
            )
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "trade_map bundle ohlcv failed symbol=%s tf=%s", symbol, timeframe
        )
        raise HTTPException(status_code=500, detail=f"ohlcv: {exc}") from exc
    mk = _marker_kwargs(
        from_=from_, to=to, since=since, include_pending=include_pending
    )
    # Keep client from/to for marker DB query; do not narrow to sparse OHLCV span.
    markers = collect_markers(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        **mk,
    )
    marker_counts = marker_scope_counts(markers)
    # Tail poll returns only a few bars; aligning to that slice collapses all markers.
    if ohlcv_mode == "full" and ohlcv.get("candles"):
        candle_times = [
            int(c["time"]) for c in ohlcv["candles"] if c.get("time") is not None
        ]
        markers = align_markers_to_candles(markers, candle_times)
    current_time = None
    current_price = None
    candles_for_current = ohlcv.get("candles") or []
    if candles_for_current:
        last_candle = candles_for_current[-1]
        current_time = (
            int(last_candle["time"]) if last_candle.get("time") is not None else None
        )
        try:
            current_price = float(last_candle.get("close"))
        except (TypeError, ValueError):
            current_price = None
    trade_links, _ = collect_trade_links(
        multi_leg_db=SETTINGS.multi_leg_db,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        symbol=symbol,
        scopes=_scopes_list(scopes),
        start_ts=mk.get("start_ts"),
        end_ts=mk.get("end_ts"),
        since_ts=mk.get("since_ts"),
        current_time=current_time,
        current_price=current_price,
    )
    cols = _feature_columns_list(feature_columns, overlay_weekly_ema=overlay_weekly_ema)
    overlays: dict = {}
    if cols and include_features and ohlcv_mode != "none" and ohlcv.get("candles"):
        overlay_start = start
        overlay_end = end
        if overlay_start is None and ohlcv.get("range_start"):
            overlay_start = pd.Timestamp(str(ohlcv["range_start"]))
        if overlay_end is None and ohlcv.get("range_end"):
            overlay_end = pd.Timestamp(str(ohlcv["range_end"]))
        overlays = load_feature_overlays(
            SETTINGS.feature_bus_root,
            symbol,
            tf,
            cols,
            start=overlay_start,
            end=overlay_end,
        )
    main_keys = parse_main_overlay_keys(main_overlays)
    main_ol: dict = {}
    if main_keys and include_features and ohlcv.get("candles"):
        feat_start = start
        feat_end = end
        if feat_start is None and ohlcv.get("range_start"):
            feat_start = pd.Timestamp(str(ohlcv["range_start"]))
        if feat_end is None and ohlcv.get("range_end"):
            feat_end = pd.Timestamp(str(ohlcv["range_end"]))
        try:
            main_ol = load_main_chart_overlays(
                SETTINGS.feature_bus_root,
                symbol,
                ohlcv["candles"],
                main_keys,
                start=feat_start,
                end=feat_end,
            )
        except Exception as exc:
            logger.exception("main_chart_overlays failed symbol=%s", symbol)
            main_ol = {
                k: {
                    "available": False,
                    "key": k,
                    "error": str(exc),
                    "points": [],
                }
                for k in main_keys
            }
    return ok(
        {
            "ohlcv": ohlcv,
            "markers": markers,
            "trade_links": trade_links,
            "overlays": overlays,
            "main_overlays": main_ol,
        },
        meta={
            "poll_seconds": SETTINGS.map_poll_seconds,
            "server_timestamp": pd.Timestamp.utcnow().isoformat(),
            "degraded_ohlc": ohlcv.get("degraded_ohlc", False),
            "bars_1min_rows": ohlcv.get("bars_1min_rows"),
            "live_storage_1m_rows": ohlcv.get("live_storage_1m_rows"),
            "ohlcv_source": ohlcv.get("source"),
            "range_start": ohlcv.get("range_start"),
            "range_end": ohlcv.get("range_end"),
            "range_clipped": ohlcv.get("range_clipped", False),
            "feature_columns": cols,
            "main_overlays": main_keys,
            "full_range": use_full,
            "include_ohlcv": ohlcv_mode,
            "macro_kline_root": ohlcv.get("macro_kline_root"),
            "macro_available": ohlcv.get("macro_available"),
            "macro_rows": ohlcv.get("macro_rows"),
            "expected_bars": ohlcv.get("expected_bars"),
            "data_sparse": ohlcv.get("data_sparse"),
            "marker_counts": marker_counts,
            "trade_link_count": len(trade_links),
        },
    )


@router.get("/api/trade-map/signals")
def trade_map_signals(
    timeframe: str = Query("2h"),
    lookback_days: int = Query(7, ge=1, le=90),
) -> dict:
    try:
        tf = assert_trade_map_timeframe(timeframe)
    except OhlcvWindowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    rows = build_signal_overview(
        symbols,
        feature_bus_root=SETTINGS.feature_bus_root,
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        timeframe=tf,
        lookback_days=lookback_days,
    )
    return ok(rows, meta={"count": len(rows), "timeframe": tf})
