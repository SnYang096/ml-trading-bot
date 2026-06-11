/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

/** Overlay series must not participate in main price autoscale. */
function overlayAutoscaleInfoProvider() {
  return null;
}

function mainChartOverlaySeriesOptions(extra = {}) {
  return {
    priceLineVisible: false,
    crosshairMarkerVisible: false,
    autoscaleInfoProvider: overlayAutoscaleInfoProvider,
    ...extra,
  };
}

/** Lock Y-axis to visible OHLC only (EMA/grid lines must not stretch the scale). */
function refreshMainPriceAutoscale() {
  if (!S.chart || !S.lastCandles?.length) return;
  const logical = S.chart.timeScale().getVisibleLogicalRange();
  let pr = Core.priceRangeForChartAutoscale(S.lastCandles, logical);
  if (pr && S.mainOverlayData?.size) {
    pr = Core.expandPriceRangeForOverlays(
      pr,
      S.lastCandles,
      logical,
      S.mainOverlayData
    );
  }
  if (!pr) return;
  const ps = S.chart.priceScale("right");
  ps.applyOptions({ autoScale: false });
  if (typeof ps.setVisibleRange === "function") {
    ps.setVisibleRange({ from: pr.minValue, to: pr.maxValue });
  } else {
    ps.applyOptions({ autoScale: true });
  }
}
function bandHighlightSeriesOptions(rgba) {
  return mainChartOverlaySeriesOptions({
    upColor: rgba,
    downColor: rgba,
    borderVisible: false,
    wickVisible: false,
    priceLineVisible: false,
    lastValueVisible: false,
  });
}

function candleInAnySpan(c, spans) {
  const t = Number(c.time);
  return (spans || []).some(
    (r) => t >= Number(r.start) && t <= Number(r.end)
  );
}

/** Per-bar highlight candles (open=low, close=high) inside stage/chop spans. */
function spanHighlightCandles(candles, spans) {
  if (!candles?.length || !spans?.length) return [];
  return candles
    .filter((c) => candleInAnySpan(c, spans))
    .map((c) => {
      const lo = Number(c.low);
      const hi = Number(c.high);
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) return null;
      return { time: c.time, open: lo, high: hi, low: lo, close: hi };
    })
    .filter(Boolean);
}

const defaultLayout = () => ({
  volume: false,
  features: ["weekly_ema_200_position", "ema_1200_position"],
  mainEma1200: true,
  mainWeeklyEma200: true,
  chopGrid: true,
  layerPrefilter: true,
  layerGate: false,
  featureStrategyFocus: null,
  ordersDock: false,
});

function loadLayout() {
  const stored = Core.parseStoredLayout(localStorage.getItem(S.LAYOUT_KEY));
  const merged = { ...defaultLayout(), ...(stored || {}) };
  if (!Array.isArray(merged.features) || !merged.features.length) {
    merged.features = defaultLayout().features;
  }
  return merged;
}

function saveLayout() {
  const layout = {
    volume: document.getElementById("paneVolume").checked,
    features: [...S.selectedFeatureColumns],
    mainEma1200: !!document.getElementById("mainEma1200")?.checked,
    mainWeeklyEma200: !!document.getElementById("mainWeeklyEma200")?.checked,
    chopGrid: !!document.getElementById("layerChopGrid")?.checked,
    layerPrefilter: !!document.getElementById("layerPrefilter")?.checked,
    layerGate: !!document.getElementById("layerGate")?.checked,
    featureStrategyFocus: S.featureStrategyFocus,
    ordersDock: S.ordersDockOpen,
  };
  localStorage.setItem(S.LAYOUT_KEY, JSON.stringify(layout));
  Shell.setScopesState(layersState());
  Shell.saveOrdersFilter(Shell.ordersFilterFromControls());
}

function applyLayoutToControls(layout) {
  document.getElementById("paneVolume").checked = !!layout.volume;
  const ema1200 = document.getElementById("mainEma1200");
  const wkEma = document.getElementById("mainWeeklyEma200");
  if (ema1200) ema1200.checked = layout.mainEma1200 !== false;
  if (wkEma) wkEma.checked = layout.mainWeeklyEma200 !== false;
  const chopGrid = document.getElementById("layerChopGrid");
  if (chopGrid) chopGrid.checked = layout.chopGrid !== false;
  const layerPf = document.getElementById("layerPrefilter");
  const layerGt = document.getElementById("layerGate");
  if (layerPf) layerPf.checked = layout.layerPrefilter !== false;
  if (layerGt) layerGt.checked = !!layout.layerGate;
  let focus =
    layout.featureStrategyFocus != null && String(layout.featureStrategyFocus).trim()
      ? String(layout.featureStrategyFocus).trim()
      : null;
  if (!focus && layout.chopGrid !== false) {
    focus = "chop_grid";
  }
  S.featureStrategyFocus = focus;
  S.selectedFeatureColumns = Array.isArray(layout.features) ? [...layout.features] : [];
  S.ordersDockOpen = !!layout.ordersDock;
  Shell.applyOrdersFilterToControls(Shell.loadOrdersFilter());
  applyOrdersDockVisibility();
}

function ordersExcludeStatusParam() {
  return Shell.ordersExcludeStatusParamFromFilter(Shell.ordersFilterFromControls());
}

function applyScopesFromStorage() {
  const saved = Shell.getScopesDefault();
  if (!saved) return;
  if (saved.trend != null) document.getElementById("layerTrend").checked = !!saved.trend;
  if (saved.spot != null) document.getElementById("layerSpot").checked = !!saved.spot;
  if (saved.multiLeg != null) document.getElementById("layerMultiLeg").checked = !!saved.multiLeg;
  if (saved.pending != null) document.getElementById("layerPending").checked = !!saved.pending;
}

function layersState() {
  return {
    trend: document.getElementById("layerTrend").checked,
    spot: document.getElementById("layerSpot").checked,
    multiLeg: document.getElementById("layerMultiLeg").checked,
    pending: document.getElementById("layerPending").checked,
  };
}

function scopesParam() {
  return Core.scopesFromLayers(layersState());
}

function tickClock() {
  const el = document.getElementById("statusClock");
  if (el) el.textContent = new Date().toLocaleTimeString();
  const corner = document.getElementById("statusCorner");
  if (corner) {
    const timeStr = new Date().toLocaleTimeString();
    const current = (corner.textContent || "").trim();
    // Initial placeholder or empty → just show time.
    if (!current || current === "--:--:--") {
      corner.textContent = timeStr;
      return;
    }
    // Append clock to corner text, replacing any previous HH:MM:SS suffix.
    const base = current.replace(/ \d{2}:\d{2}:\d{2}$/, "");
    corner.textContent = (base + " " + timeStr).trim();
  }
}

function setStatusLoading() {
  const corner = document.getElementById("statusCorner");
  if (corner) {
    corner.textContent = "加载中…";
    corner.classList.add("status-corner--loading");
  }
  document.getElementById("statusPrimary").textContent = "加载中…";
  document.getElementById("statusMeta").textContent = "";
  document.getElementById("statusFeatures").textContent = "";
  document.getElementById("statusGrid").title = "加载中…";
}

function setStatusFromBundle(symbol, timeframe, candles, markers, meta, overlays) {
  const corner = document.getElementById("statusCorner");
  if (corner) {
    corner.textContent = `${symbol} ${timeframe}`;
    corner.classList.remove("status-corner--loading");
  }
  const deg = meta.degraded_ohlc;
  const parts = [
    `${symbol} ${timeframe}`,
    `${candles.length} bars`,
    `${markers.length} markers`,
  ];
  if (meta.trade_link_count != null && meta.trade_link_count > 0) {
    parts.push(`links=${meta.trade_link_count}`);
  }
  if (S.lastMarkerCounts?.total != null && S.lastMarkerCounts.total > markers.length) {
    parts[2] = `${markers.length}/${S.lastMarkerCounts.total} markers`;
    const scopes = [];
    if (S.lastMarkerCounts.trend) scopes.push(`B${S.lastMarkerCounts.trend}`);
    if (S.lastMarkerCounts.spot) scopes.push(`A${S.lastMarkerCounts.spot}`);
    if (S.lastMarkerCounts.multi_leg) scopes.push(`C${S.lastMarkerCounts.multi_leg}`);
    if (scopes.length) parts.push(`db:${scopes.join(",")}`);
  }
  if (meta.bars_1min_rows) parts.push(`bus1m=${meta.bars_1min_rows}`);
  if (meta.live_storage_1m_rows) parts.push(`hist1m=${meta.live_storage_1m_rows}`);
  if (meta.ohlcv_source) parts.push(meta.ohlcv_source);
  if (meta.macro_rows != null) parts.push(`macro=${meta.macro_rows}`);
  if (meta.macro_available === false) parts.push("macro_missing");
  if (meta.data_sparse || (meta.expected_bars && candles.length < meta.expected_bars * 0.25)) {
    parts.push(`数据不足(有${candles.length}根/约需${meta.expected_bars || "?"})`);
  }
  if (meta.range_start && meta.range_end) {
    parts.push(`${meta.range_start.slice(0, 10)}→${meta.range_end.slice(0, 10)}`);
  }
  if (candles.length && meta.last_candle_time != null) {
    const lastBar = Core.isoFromUnixSec(meta.last_candle_time);
    if (lastBar && (!meta.range_end || lastBar.slice(0, 10) !== meta.range_end.slice(0, 10))) {
      parts.push(`K线至${lastBar.slice(0, 10)}`);
    }
  }
  const featEnds = Object.values(overlays || {})
    .map((o) => o?.feature_range_end)
    .filter(Boolean);
  if (featEnds.length && candles.length) {
    const latestFeat = featEnds.sort().slice(-1)[0];
    const lastT = candles[candles.length - 1].time;
    const featTs = Math.floor(new Date(latestFeat).getTime() / 1000);
    if (Number.isFinite(featTs) && lastT - featTs > 7200 * 3) {
      parts.push(`特征滞后至${String(latestFeat).slice(0, 10)}`);
    }
  }
  if (meta.range_clipped) parts.push(`clipped ${meta.max_ohlcv_days || ""}d`);
  if (deg) parts.push("OHLC degraded");
  const feat = formatOverlayStatus(overlays);
  const featCap =
    S.selectedFeatureColumns.length > S.MAX_FEATURE_SUBCHARTS
      ? `附图限${S.MAX_FEATURE_SUBCHARTS}列`
      : "";

  document.getElementById("statusPrimary").textContent = parts.slice(0, 3).join(" · ");
  document.getElementById("statusMeta").textContent = parts.slice(3).join(" · ");
  document.getElementById("statusFeatures").textContent =
    (feat ? feat.replace(/^ · /, "") : "特征:未选") + (featCap || "");
  const full = [...parts, feat.replace(/^ · /, ""), featCap].filter(Boolean).join(" · ");
  document.getElementById("statusGrid").title = full;
}

function setStatus(msg) {
  const corner = document.getElementById("statusCorner");
  if (corner) {
    corner.textContent = msg;
    corner.classList.add("status-corner--loading");
  }
  document.getElementById("statusPrimary").textContent = msg;
  document.getElementById("statusMeta").textContent = "";
  document.getElementById("statusFeatures").textContent = "";
  document.getElementById("statusGrid").title = msg;
}

