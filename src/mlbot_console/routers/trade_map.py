from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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
from mlbot_console.services.chop_grid_overlay import (
    load_chop_grid_map_overlay,
    load_chop_regime_regions,
)
from mlbot_console.services.strategy_stage_regions import load_bundle_stage_regions
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


def _bundle_ohlcv_query(
    tf: str,
    *,
    from_: Optional[str],
    to: Optional[str],
    ohlcv_from: Optional[str],
    ohlcv_to: Optional[str],
    ohlcv_mode: str,
    full_range: bool,
) -> tuple[Optional[str], Optional[str], bool]:
    """OHLCV bounds for bundle: 1d/1w full_range uses Vision macro, not client from/to."""
    use_full = full_range and ohlcv_mode == "full"
    if tf in ("1d", "1w") and use_full:
        return None, None, True
    if ohlcv_mode == "tail" and ohlcv_from:
        return ohlcv_from, ohlcv_to, False
    return from_, to, use_full


def _candles_for_chart_window(
    ohlcv: dict,
    *,
    ohlcv_mode: str,
    symbol: str,
    timeframe: str,
    chart_from: Optional[str],
    chart_to: Optional[str],
) -> List[Dict[str, Any]]:
    """Tail poll returns few OHLCV bars; overlays/features need the full loaded chart window."""
    tail = list(ohlcv.get("candles") or [])
    if ohlcv_mode != "tail" or not tail:
        return tail
    if not chart_from and not chart_to:
        return tail
    try:
        tf = assert_trade_map_timeframe(timeframe)
        o_start = pd.Timestamp(chart_from, tz="UTC") if chart_from else None
        o_end = pd.Timestamp(chart_to, tz="UTC") if chart_to else None
        if o_start is None:
            o_start = pd.Timestamp(int(tail[0]["time"]), unit="s", tz="UTC")
        if o_end is None:
            o_end = pd.Timestamp.now(tz="UTC")
        max_days = (
            SETTINGS.max_daily_ohlcv_days
            if tf in ("1d", "1w")
            else SETTINGS.max_ohlcv_days
        )
        o_start, o_end = cap_window_to_max_days(o_start, o_end, max_days)
        pack = ohlcv_reader.fetch_ohlcv(
            SETTINGS.feature_bus_root,
            symbol,
            tf,
            start=o_start,
            end=o_end,
            max_days=SETTINGS.max_ohlcv_days,
            full_range=False,
            live_storage_bars_root=SETTINGS.live_storage_bars_root,
            stitch_live_storage=SETTINGS.stitch_live_storage,
            macro_kline_root=SETTINGS.macro_spot_kline_root,
            daily_ohlcv_start=SETTINGS.daily_ohlcv_start,
            max_daily_ohlcv_days=SETTINGS.max_daily_ohlcv_days,
            live_data_root=SETTINGS.live_data_root,
            live_root=SETTINGS.live_root,
        )
        full = list(pack.get("candles") or [])
        return full if full else tail
    except Exception:
        logger.exception(
            "main overlay full-window fetch failed symbol=%s tf=%s",
            symbol,
            timeframe,
        )
        return tail


def _bundle_internal_ohlcv(
    symbol: str,
    tf: str,
    *,
    from_: Optional[str],
    to: Optional[str],
    full_range: bool,
) -> tuple[List[Dict[str, Any]], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Fetch OHLCV for overlay/chop computation when include_ohlcv=none."""
    try:
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
        pack = ohlcv_reader.fetch_ohlcv(
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
        candles = list(pack.get("candles") or [])
        return candles, start, end
    except Exception:
        logger.exception(
            "bundle internal ohlcv fetch failed symbol=%s tf=%s", symbol, tf
        )
        return [], None, None


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
    scope_list = _scopes_list(scopes)
    markers = collect_markers(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        symbol=symbol,
        scopes=scope_list,
        engine_data_root=SETTINGS.engine_data_root,
        feature_bus_root=SETTINGS.feature_bus_root,
        strategies_root=SETTINGS.strategies_root,
        map_timeframe="2h",
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
        description="Comma-separated: ema_1200, weekly_ema_200 (CMS-local OHLC/macro)",
    ),
    full_range: bool = Query(False),
    include_ohlcv: str = Query(
        "full",
        description="full | tail (small OHLCV slice) | none (markers/links only)",
    ),
    ohlcv_from: Optional[str] = Query(None, alias="ohlcv_from"),
    ohlcv_to: Optional[str] = Query(None, alias="ohlcv_to"),
    include_features: bool = Query(True),
    include_markers: bool = Query(True),
    include_trade_links: bool = Query(True),
    include_chop: bool = Query(True),
    stage_regions: Optional[str] = Query(
        None,
        description="Comma-separated stages to shade on main chart: prefilter, gate",
    ),
    strategy: str = Query(
        "",
        description="Optional strategy id filter (markers, stage bands, chop overlay)",
    ),
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
            ohlcv_start_raw, ohlcv_end_raw, use_full = _bundle_ohlcv_query(
                tf,
                from_=from_,
                to=to,
                ohlcv_from=ohlcv_from,
                ohlcv_to=ohlcv_to,
                ohlcv_mode=ohlcv_mode,
                full_range=full_range,
            )
            start, end, use_full = resolve_trade_map_window(
                tf,
                start=(
                    pd.Timestamp(ohlcv_start_raw, tz="UTC") if ohlcv_start_raw else None
                ),
                end=pd.Timestamp(ohlcv_end_raw, tz="UTC") if ohlcv_end_raw else None,
                full_range=use_full,
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
    scope_list = _scopes_list(scopes)
    strat_filter = str(strategy or "").strip().lower()
    strat_list = [strat_filter] if strat_filter else None
    stage_parts = {
        p.strip().lower()
        for p in (stage_regions or "").split(",")
        if p.strip()
    }
    include_prefilter_regions = "prefilter" in stage_parts
    include_gate_regions = "gate" in stage_parts
    cols = _feature_columns_list(feature_columns, overlay_weekly_ema=overlay_weekly_ema)
    need_stage_regions = include_prefilter_regions or include_gate_regions
    need_chop_regime = (
        include_chop
        and "multi_leg" in scope_list
        and (not strat_filter or strat_filter in ("chop_grid", "trend_scalp"))
    )
    need_internal_candles = ohlcv_mode == "none" and (
        (cols and include_features) or need_stage_regions or need_chop_regime
    )
    internal_candles: List[Dict[str, Any]] = []
    if need_internal_candles:
        internal_candles, int_start, int_end = _bundle_internal_ohlcv(
            symbol,
            tf,
            from_=from_,
            to=to,
            full_range=full_range,
        )
        if start is None:
            start = int_start
        if end is None:
            end = int_end
    chart_candles = list(ohlcv.get("candles") or []) or internal_candles

    markers: List[Dict[str, Any]] = []
    if include_markers:
        markers = collect_markers(
            trend_db=SETTINGS.trend_order_db,
            spot_db=SETTINGS.spot_order_db,
            multi_leg_db=SETTINGS.multi_leg_db,
            symbol=symbol,
            scopes=scope_list,
            engine_data_root=SETTINGS.engine_data_root,
            strategies=strat_list,
            feature_bus_root=SETTINGS.feature_bus_root,
            strategies_root=SETTINGS.strategies_root,
            map_timeframe=tf,
            **mk,
        )
        if ohlcv_mode == "full" and ohlcv.get("candles"):
            candle_times = [
                int(c["time"]) for c in ohlcv["candles"] if c.get("time") is not None
            ]
            markers = align_markers_to_candles(markers, candle_times)
    marker_counts = marker_scope_counts(markers)

    current_time = None
    current_price = None
    candles_for_current = chart_candles
    if candles_for_current:
        last_candle = candles_for_current[-1]
        current_time = (
            int(last_candle["time"]) if last_candle.get("time") is not None else None
        )
        try:
            current_price = float(last_candle.get("close"))
        except (TypeError, ValueError):
            current_price = None

    trade_links: List[Dict[str, Any]] = []
    if include_trade_links:
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

    overlays: dict = {}
    if cols and include_features and chart_candles:
        overlay_start = start
        overlay_end = end
        if overlay_start is None and ohlcv.get("range_start"):
            overlay_start = pd.Timestamp(str(ohlcv["range_start"]))
        if overlay_end is None and ohlcv.get("range_end"):
            overlay_end = pd.Timestamp(str(ohlcv["range_end"]))
        if ohlcv_mode == "tail" and ohlcv.get("candles"):
            feature_candles = _candles_for_chart_window(
                ohlcv,
                ohlcv_mode=ohlcv_mode,
                symbol=symbol,
                timeframe=tf,
                chart_from=from_,
                chart_to=to,
            )
        else:
            feature_candles = chart_candles
        overlays = load_feature_overlays(
            SETTINGS.feature_bus_root,
            symbol,
            tf,
            cols,
            start=overlay_start,
            end=overlay_end,
            candles=feature_candles,
        )
    main_keys = parse_main_overlay_keys(main_overlays)
    main_ol: dict = {}
    if main_keys and ohlcv.get("candles"):
        feat_start = start
        feat_end = end
        if feat_start is None and ohlcv.get("range_start"):
            feat_start = pd.Timestamp(str(ohlcv["range_start"]))
        if feat_end is None and ohlcv.get("range_end"):
            feat_end = pd.Timestamp(str(ohlcv["range_end"]))
        try:
            overlay_candles = _candles_for_chart_window(
                ohlcv,
                ohlcv_mode=ohlcv_mode,
                symbol=symbol,
                timeframe=tf,
                chart_from=from_,
                chart_to=to,
            )
            main_ol = load_main_chart_overlays(
                symbol,
                overlay_candles,
                main_keys,
                chart_timeframe=tf,
                macro_seed_root=SETTINGS.macro_weekly_ema_seed_root,
                macro_spot_kline_root=SETTINGS.macro_spot_kline_root,
                feature_bus_root=SETTINGS.feature_bus_root,
                live_storage_bars_root=SETTINGS.live_storage_bars_root,
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
    chop_grid_overlay: dict = {"batches": []}
    chop_regime_regions: list = []
    strategy_stage_regions: dict = {}
    if need_stage_regions and chart_candles:
        feat_start = start
        feat_end = end
        if feat_start is None and ohlcv.get("range_start"):
            feat_start = pd.Timestamp(str(ohlcv["range_start"]))
        if feat_end is None and ohlcv.get("range_end"):
            feat_end = pd.Timestamp(str(ohlcv["range_end"]))
        try:
            strategy_stage_regions = load_bundle_stage_regions(
                SETTINGS.feature_bus_root,
                SETTINGS.strategies_root,
                symbol,
                tf,
                scopes=scope_list,
                include_prefilter=include_prefilter_regions,
                include_gate=include_gate_regions,
                start=feat_start,
                end=feat_end,
            )
            if strat_filter and isinstance(strategy_stage_regions, dict):
                strategy_stage_regions = {
                    k: v
                    for k, v in strategy_stage_regions.items()
                    if str(k).lower() == strat_filter
                }
        except Exception as exc:
            logger.exception("strategy_stage_regions failed symbol=%s", symbol)
            strategy_stage_regions = {"error": str(exc)}
    if include_chop and "multi_leg" in scope_list and (
        not strat_filter or strat_filter in ("chop_grid", "trend_scalp")
    ):
        try:
            chop_grid_overlay = load_chop_grid_map_overlay(
                multi_leg_db=SETTINGS.multi_leg_db,
                engine_data_root=SETTINGS.engine_data_root,
                symbol=symbol,
            )
        except Exception as exc:
            logger.exception("chop_grid_map_overlay failed symbol=%s", symbol)
            chop_grid_overlay = {"batches": [], "error": str(exc)}
        if need_chop_regime and chart_candles:
            feat_start = start
            feat_end = end
            if feat_start is None and ohlcv.get("range_start"):
                feat_start = pd.Timestamp(str(ohlcv["range_start"]))
            if feat_end is None and ohlcv.get("range_end"):
                feat_end = pd.Timestamp(str(ohlcv["range_end"]))
            try:
                chop_regime_regions = load_chop_regime_regions(
                    SETTINGS.feature_bus_root,
                    symbol,
                    tf,
                    start=feat_start,
                    end=feat_end,
                )
            except Exception as exc:
                logger.exception("chop_regime_regions failed symbol=%s", symbol)
                chop_regime_regions = []
    return ok(
        {
            "ohlcv": ohlcv,
            "markers": markers,
            "trade_links": trade_links,
            "overlays": overlays,
            "main_overlays": main_ol,
            "chop_grid_overlay": chop_grid_overlay,
            "chop_regime_regions": chop_regime_regions,
            "strategy_stage_regions": strategy_stage_regions,
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
            "last_candle_time": ohlcv.get("last_candle_time"),
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
        live_monitor_db=SETTINGS.live_monitor_db,
        timeframe=tf,
        lookback_days=lookback_days,
    )
    return ok(rows, meta={"count": len(rows), "timeframe": tf})
