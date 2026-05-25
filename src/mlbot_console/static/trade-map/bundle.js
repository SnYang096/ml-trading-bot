/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function mergeChopMapPayload(prev, data) {
  const out = { ...(prev || {}) };
  if (data?.chop_grid_overlay != null) out.chop_grid_overlay = data.chop_grid_overlay;
  if (data?.chop_regime_regions?.length) {
    out.chop_regime_regions = mergeRegionList(
      out.chop_regime_regions || [],
      data.chop_regime_regions
    );
  }
  if (data?.strategy_stage_regions != null) {
    out.strategy_stage_regions = mergeStageRegions(
      out.strategy_stage_regions || {},
      data.strategy_stage_regions
    );
  }
  return out;
}

function mergeRegionList(existing, incoming) {
  const rows = [...(existing || []), ...(incoming || [])]
    .map((r) => ({
      ...r,
      start: Number(r.start),
      end: Number(r.end),
    }))
    .filter(
      (r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end >= r.start
    )
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  for (const r of rows) {
    const last = merged[merged.length - 1];
    if (last && r.start <= last.end + 1) {
      last.end = Math.max(last.end, r.end);
    } else {
      merged.push({ ...r });
    }
  }
  return merged;
}

function mergeStageRegions(existing, incoming) {
  const out = { ...(existing || {}) };
  for (const [strategy, stages] of Object.entries(incoming || {})) {
    if (!stages || typeof stages !== "object") continue;
    const nextStages = { ...(out[strategy] || {}) };
    for (const [stage, spans] of Object.entries(stages)) {
      if (!Array.isArray(spans) || !spans.length) continue;
      nextStages[stage] = mergeRegionList(nextStages[stage] || [], spans);
    }
    out[strategy] = nextStages;
  }
  return out;
}

async function refreshMarkersOnly() {
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  const q = new URLSearchParams({
    symbol,
    timeframe,
    scopes: scopesParam(),
    include_pending: String(layersState().pending),
    include_ohlcv: "none",
    include_features: "false",
    ...markerRangeParams(),
  });
  const { data, meta } = await Shell.api(`/api/trade-map/bundle?${q}`);
  S.lastMarkerCounts = meta.marker_counts || null;
  applyMarkers(data.markers || []);
  applyTradeLinks(data.trade_links || []);
}

async function refreshBundle(opts = {}) {
  const mode = opts.mode || "full";
  const symbol = document.getElementById("symbolSelect").value;
  Shell.setSymbol(symbol);
  const timeframe = document.getElementById("timeframeSelect").value;
  const scopes = scopesParam();
  const pending = layersState().pending;
  const featParam = Core.featureColumnsParam(S.selectedFeatureColumns);
  const mainOl = Core.mainOverlaysQueryParam(
    document.getElementById("mainEma1200")?.checked,
    document.getElementById("mainWeeklyEma200")?.checked
  );
  const stageRg = Core.stageRegionsQueryParam(
    document.getElementById("layerPrefilter")?.checked,
    document.getElementById("layerGate")?.checked
  );
  const stratFocus = String(S.featureStrategyFocus || "").trim();
  if (mode === "full") {
    setStatusLoading();
    if (opts.resetMarkerRange) {
      resetMarkerQueryRange();
    }
  }

  const q = new URLSearchParams({
    symbol,
    timeframe,
    scopes,
    include_pending: String(pending),
    ...markerRangeParams(),
  });

  if (mode === "poll") {
    q.set("include_ohlcv", "tail");
    if (stageRg) q.set("stage_regions", stageRg);
    if (stratFocus) q.set("strategy", stratFocus);
    if (mainOl) q.set("main_overlays", mainOl);
    const pollFeatures = S.selectedFeatureColumns.length > 0;
    q.set("include_features", pollFeatures ? "true" : "false");
    if (pollFeatures && featParam) q.set("feature_columns", featParam);
    if (S.lastMarkerPollSince) q.set("since", S.lastMarkerPollSince);
    if (S.ohlcvLoadedFrom) q.set("from", S.ohlcvLoadedFrom);
    q.set("to", S.ohlcvLoadedTo || new Date().toISOString());
    const lastT = S.lastCandles.length ? S.lastCandles[S.lastCandles.length - 1].time : null;
    if (lastT != null) {
      const barSec = Core.barDurationSec(timeframe);
      const tailFrom = Core.isoFromUnixSec(Number(lastT) - barSec * 5);
      q.set("ohlcv_from", tailFrom);
      q.set("ohlcv_to", new Date().toISOString());
    }
  } else {
    q.set("include_ohlcv", "full");
    q.set("include_features", "true");
    if (!S.ohlcvLoadedFrom || opts.resetOhlcvRange) {
      const init = initialOhlcvRangeIso();
      if (init.from) q.set("from", init.from);
      if (init.to) q.set("to", init.to);
      q.set("full_range", init.full_range || "false");
    } else {
      q.set("from", S.ohlcvLoadedFrom);
      q.set("to", new Date().toISOString());
      q.set("full_range", "false");
    }
    if (featParam) q.set("feature_columns", featParam);
    if (mainOl) q.set("main_overlays", mainOl);
    if (stageRg) q.set("stage_regions", stageRg);
    if (stratFocus) q.set("strategy", stratFocus);
  }

  const { data, meta } = await Shell.api(`/api/trade-map/bundle?${q}`);
  updateMarkerPollSince(meta?.server_timestamp);
  S.lastMarkerCounts = meta.marker_counts || null;
  const pageUrl = new URL(window.location.href);

  if (mode === "poll") {
    if (data.ohlcv?.candles?.length) {
      const newCandles = Core.sanitizeCandlesForLwc(data.ohlcv.candles);
      const prevLast =
        S.lastCandles.length > 0
          ? Number(S.lastCandles[S.lastCandles.length - 1].time)
          : null;
      const merged = Core.mergeCandlesByTime(S.lastCandles, newCandles);
      S.lastCandles = merged;
      const mergedByTime = new Map(merged.map((row) => [Number(row.time), row]));
      for (const c of newCandles) {
        const t = Number(c.time);
        if (!Number.isFinite(t)) continue;
        if (prevLast != null && t < prevLast) continue;
        const row = mergedByTime.get(t) || c;
        S.candleSeries.update(row);
      }
      S.ohlcvLoadedTo = meta.range_end || new Date().toISOString();
    }
    S.lastChopMapData = mergeChopMapPayload(S.lastChopMapData, data);
    if (S.lastCandles.length && S.lastChopMapData) {
      applyChopMapLayers(S.lastChopMapData, S.lastCandles);
      if (typeof isViewingHistoricalBars === "function" && !isViewingHistoricalBars()) {
        refreshMainPriceAutoscale();
      }
    }
    if (data.overlays && Object.keys(data.overlays).length) {
      const mergedOl = { ...(S.lastOverlays || {}) };
      for (const [col, spec] of Object.entries(data.overlays)) {
        if (!spec?.points?.length) continue;
        mergedOl[col] = {
          ...spec,
          points: Core.clipOverlayPointsToCandles(
            mergeOverlayPoints(mergedOl[col]?.points, spec.points),
            S.lastCandles
          ),
        };
      }
      S.lastOverlays = mergedOl;
    }
    if (data.main_overlays && Object.keys(data.main_overlays).length) {
      applyMainOverlays(data.main_overlays, { merge: false });
    }
    syncSubcharts(S.lastCandles, S.lastOverlays || {});
    if (
      typeof refreshFeatureMetricsPanel === "function" &&
      Core.chopMetricsTableActive(S.featureStrategyFocus, S.selectedFeatureColumns)
    ) {
      refreshFeatureMetricsPanel(S.highlightBarTime ?? null, {
        rebuild: true,
        scrollNow: false,
      });
    }
  } else if (mode !== "poll") {
    const candles = Core.sanitizeCandlesForLwc(data.ohlcv?.candles || []);
    S.lastCandles = candles;
    S.candleSeries.setData(candles);
    applyMainOverlays(data.main_overlays || {});
    applyChopMapLayers(data, candles);
    applyLoadedOhlcvRange(meta, candles);
    if (S.chartFitPending) {
      applyChartViewport(candles.length);
      S.chartFitPending = false;
    } else {
      refreshMainPriceAutoscale();
    }
    S.lastOverlays = data.overlays || {};
    S.lastChopMapData = {
      chop_grid_overlay: data.chop_grid_overlay,
      chop_regime_regions: data.chop_regime_regions,
      strategy_stage_regions: data.strategy_stage_regions,
    };
    syncSubcharts(candles, S.lastOverlays);
  }

  const markers = data.markers || [];
  if (mode === "poll") {
    applyMarkers(markers, { merge: true });
    S.lastTradeLinks = mergeTradeLinks(S.lastTradeLinks, data.trade_links || []);
    applyTradeLinks(S.lastTradeLinks);
  } else {
    applyMarkers(markers);
    S.lastTradeLinks = data.trade_links || [];
    applyTradeLinks(S.lastTradeLinks);
  }

  if (mode !== "poll") {
    setStatusFromBundle(
      symbol,
      timeframe,
      S.lastCandles,
      markers,
      meta,
      data.overlays || {}
    );
  }
  tickClock();

  if (S.ordersDockOpen && mode !== "poll") {
    await refreshOrdersList();
  }

  const markerId = pageUrl.searchParams.get("marker_id");
  if (markerId && S.markerById.has(markerId)) {
    selectMarker(markerId, { scrollChart: true, showDetail: true });
  } else if (S.selectedMarkerId && !S.markerById.has(S.selectedMarkerId)) {
    S.selectedMarkerId = null;
    highlightOrdersTableRow(null);
  }
}

function startPoll() {
  if (S.pollTimer) clearInterval(S.pollTimer);
  S.pollTimer = setInterval(() => {
    refreshBundle({ mode: "poll" }).catch((e) => setStatus(String(e)));
  }, S.POLL_MS);
}


