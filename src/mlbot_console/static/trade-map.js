/**
 * Trade Map page — K线 + 账户层标记 + 动态附图 + Spot 资格侧栏。
 */

const Core = globalThis.MLBotTradeMapCore;
const Shell = globalThis.MLBotConsole;
const POLL_MS = 10000;
const LAYOUT_KEY = "mlbot_trade_map_layout_v2";

let chart;
let candleSeries;
/** @type {Map<string, import('lightweight-charts').ISeriesApi<'Line'>>} */
const mainOverlaySeries = new Map();
const mainOverlayData = new Map();
let pollTimer;
let markerById = new Map();
let lastRawMarkers = [];
let lastCandles = [];
let selectedMarkerId = null;
let ordersDockOpen = false;
let chartFitPending = true;
let timeSyncBound = false;
let clockTimer = null;

/** @type {Map<string, { chart, series, refSeries?, label, kind, host }>} */
const subcharts = new Map();

/** @type {import('lightweight-charts').ISeriesApi<'Line'>[]} */
let tradeLinkSeries = [];

/** @type {import('lightweight-charts').IPriceLine[]} */
let chopGridPriceLines = [];
/** @type {import('lightweight-charts').ISeriesApi<'Histogram'>|null} */
let chopRegimeSeries = null;

let availableFeatureColumns = [];
let selectedFeatureColumns = [];
let featureSearchQuery = "";
const MAX_FEATURE_SUBCHARTS = 8;

/** Loaded OHLCV window (ISO); null → use per-TF initial window. */
let ohlcvLoadedFrom = null;
let ohlcvLoadedTo = null;
/** Marker DB query window (wider than sparse candles). */
let markerQueryFromIso = null;
/** ISO timestamp for incremental marker poll (`since` query param). */
let lastMarkerPollSince = null;
let lastMarkerCounts = null;
/** @type {object[]} */
let lastTradeLinks = [];
let historyLoadInFlight = false;
let historyExhausted = false;
let historyLoadTimer = null;

const defaultLayout = () => ({
  volume: false,
  features: ["weekly_ema_200_position"],
  mainEma1200: false,
  mainWeeklyEma200: false,
  chopGrid: true,
  ordersDock: false,
});

function loadLayout() {
  const stored = Core.parseStoredLayout(localStorage.getItem(LAYOUT_KEY));
  const merged = { ...defaultLayout(), ...(stored || {}) };
  if (!Array.isArray(merged.features) || !merged.features.length) {
    merged.features = defaultLayout().features;
  }
  return merged;
}

function saveLayout() {
  const layout = {
    volume: document.getElementById("paneVolume").checked,
    features: [...selectedFeatureColumns],
    mainEma1200: !!document.getElementById("mainEma1200")?.checked,
    mainWeeklyEma200: !!document.getElementById("mainWeeklyEma200")?.checked,
    chopGrid: !!document.getElementById("layerChopGrid")?.checked,
    ordersDock: ordersDockOpen,
  };
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  Shell.setScopesState(layersState());
  Shell.saveOrdersFilter(Shell.ordersFilterFromControls());
}

function applyLayoutToControls(layout) {
  document.getElementById("paneVolume").checked = !!layout.volume;
  const ema1200 = document.getElementById("mainEma1200");
  const wkEma = document.getElementById("mainWeeklyEma200");
  if (ema1200) ema1200.checked = !!layout.mainEma1200;
  if (wkEma) wkEma.checked = !!layout.mainWeeklyEma200;
  const chopGrid = document.getElementById("layerChopGrid");
  if (chopGrid) chopGrid.checked = layout.chopGrid !== false;
  selectedFeatureColumns = Array.isArray(layout.features) ? [...layout.features] : [];
  ordersDockOpen = !!layout.ordersDock;
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
}

function setStatusLoading() {
  document.getElementById("statusPrimary").textContent = "加载中…";
  document.getElementById("statusMeta").textContent = "";
  document.getElementById("statusFeatures").textContent = "";
  document.getElementById("statusGrid").title = "加载中…";
}

function setStatusFromBundle(symbol, timeframe, candles, markers, meta, overlays) {
  const deg = meta.degraded_ohlc;
  const parts = [
    `${symbol} ${timeframe}`,
    `${candles.length} bars`,
    `${markers.length} markers`,
  ];
  if (meta.trade_link_count != null && meta.trade_link_count > 0) {
    parts.push(`links=${meta.trade_link_count}`);
  }
  if (lastMarkerCounts?.total != null && lastMarkerCounts.total > markers.length) {
    parts[2] = `${markers.length}/${lastMarkerCounts.total} markers`;
    const scopes = [];
    if (lastMarkerCounts.trend) scopes.push(`B${lastMarkerCounts.trend}`);
    if (lastMarkerCounts.spot) scopes.push(`A${lastMarkerCounts.spot}`);
    if (lastMarkerCounts.multi_leg) scopes.push(`C${lastMarkerCounts.multi_leg}`);
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
  if (meta.range_clipped) parts.push(`clipped ${meta.max_ohlcv_days || ""}d`);
  if (deg) parts.push("OHLC degraded");
  const feat = formatOverlayStatus(overlays);
  const featCap =
    selectedFeatureColumns.length > MAX_FEATURE_SUBCHARTS
      ? `附图限${MAX_FEATURE_SUBCHARTS}列`
      : "";

  document.getElementById("statusPrimary").textContent = parts.slice(0, 3).join(" · ");
  document.getElementById("statusMeta").textContent = parts.slice(3).join(" · ");
  document.getElementById("statusFeatures").textContent =
    (feat ? feat.replace(/^ · /, "") : "特征:未选") + (featCap || "");
  const full = [...parts, feat.replace(/^ · /, ""), featCap].filter(Boolean).join(" · ");
  document.getElementById("statusGrid").title = full;
}

function setStatus(msg) {
  document.getElementById("statusPrimary").textContent = msg;
  document.getElementById("statusMeta").textContent = "";
  document.getElementById("statusFeatures").textContent = "";
  document.getElementById("statusGrid").title = msg;
}

function chartBaseOptions() {
  return {
    layout: { background: { color: "#0f1419" }, textColor: "#8b949e" },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      barSpacing: 3,
      minBarSpacing: 0.5,
      rightOffset: 8,
    },
    rightPriceScale: {
      borderColor: "#30363d",
      scaleMargins: { top: 0.08, bottom: 0.12 },
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
  };
}

/** Feature/volume panes: no grid lines so stacked subcharts read as one block. */
function subchartBaseOptions() {
  return {
    ...chartBaseOptions(),
    grid: {
      vertLines: { visible: false },
      horzLines: { visible: false },
    },
    timeScale: { visible: false },
  };
}

function applyChartViewport(barCount) {
  const visible = Core.defaultVisibleBarCount(barCount);
  const spacing = Core.barSpacingForCount(visible);
  chart.timeScale().applyOptions({
    barSpacing: spacing,
    minBarSpacing: 0.5,
    rightOffset: 8,
  });
  const range = Core.visibleLogicalRange(barCount);
  if (range) {
    chart.timeScale().setVisibleLogicalRange(range);
    syncSubchartsToMainRange();
    chart.priceScale("right").applyOptions({ autoScale: true });
  }
}

function syncSubchartsToMainRange() {
  if (!chart) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  for (const pane of subcharts.values()) {
    pane.chart.timeScale().setVisibleLogicalRange(range);
  }
}

function initMainChart() {
  const el = document.getElementById("chart");
  chart = LightweightCharts.createChart(el, chartBaseOptions());
  candleSeries = chart.addCandlestickSeries({
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderVisible: false,
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
    autoscaleInfoProvider: (original) => {
      const range = chart.timeScale().getVisibleLogicalRange();
      const custom = Core.priceRangeForVisibleCandles(lastCandles, range);
      if (custom) {
        return {
          priceRange: custom,
          margins: { above: 10, below: 10 },
        };
      }
      return original();
    },
  });
  candleSeries.setMarkers([]);

  const resize = () => {
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    for (const pane of subcharts.values()) {
      pane.chart.applyOptions({
        width: pane.host.clientWidth,
        height: pane.host.clientHeight,
      });
    }
  };
  window.addEventListener("resize", () => {
    resize();
    resizeAllSubcharts();
  });
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => {
      resize();
      resizeAllSubcharts();
    });
    ro.observe(el);
  }
  resize();
  bindTimeScaleSync();
  chart.subscribeClick((param) => {
    if (!param || param.time === undefined) return;
    const tf = document.getElementById("timeframeSelect")?.value || "2h";
    const tol = Core.timeframeToleranceSec(tf);
    const hit = Core.findMarkerByTime(lastRawMarkers, param.time, tol);
    if (hit?.id) selectMarker(hit.id);
  });

  const legend = document.getElementById("chartLegend");
  chart.subscribeCrosshairMove((param) => {
    if (!param || !param.time || param.point.x < 0 || param.point.y < 0) {
      legend.classList.add("hidden");
      return;
    }
    const data = param.seriesData.get(candleSeries);
    if (!data) {
      legend.classList.add("hidden");
      return;
    }
    const o = data.open;
    const h = data.high;
    const l = data.low;
    const c = data.close;
    const timeStr = Shell.formatOrderTime(param.time);
    
    const pct = ((c - o) / o * 100).toFixed(2);
    const cls = c >= o ? "legend-pos" : "legend-neg";
    const sign = c >= o ? "+" : "";

    let overlayText = "";
    for (const [key, series] of mainOverlaySeries.entries()) {
      const val = param.seriesData.get(series);
      if (val && val.value !== undefined) {
        overlayText += `  ${key}: <span class="legend-price">${val.value.toFixed(4)}</span>`;
      }
    }

    legend.innerHTML = `${timeStr}  O <span class="legend-price">${o.toFixed(4)}</span>  H <span class="legend-price">${h.toFixed(4)}</span>  L <span class="legend-price">${l.toFixed(4)}</span>  C <span class="legend-price">${c.toFixed(4)}</span>  <span class="${cls}">${sign}${pct}%</span>${overlayText}`;
    legend.classList.remove("hidden");
  });
}

function resizeAllSubcharts() {
  for (const pane of subcharts.values()) {
    if (!pane.host) continue;
    const w = pane.host.clientWidth;
    const h = pane.host.clientHeight;
    if (w > 0 && h > 0) {
      pane.chart.applyOptions({ width: w, height: h });
    }
  }
}

function clearMainOverlaySeries() {
  for (const [, series] of mainOverlaySeries) {
    chart.removeSeries(series);
  }
  mainOverlaySeries.clear();
  mainOverlayData.clear();
}

function mergeOverlayPoints(existing, incoming) {
  const byTime = new Map();
  for (const p of existing || []) {
    if (p && p.time != null) byTime.set(Number(p.time), p);
  }
  for (const p of incoming || []) {
    if (p && p.time != null) byTime.set(Number(p.time), p);
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time);
}

function clearChopGridOverlay() {
  for (const pl of chopGridPriceLines) {
    try {
      candleSeries.removePriceLine(pl);
    } catch (_) {
      /* already removed */
    }
  }
  chopGridPriceLines = [];
  if (chopRegimeSeries) {
    try {
      chart.removeSeries(chopRegimeSeries);
    } catch (_) {
      /* */
    }
    chopRegimeSeries = null;
  }
}

function chopGridOverlayEnabled() {
  return (
    !!document.getElementById("layerChopGrid")?.checked &&
    layersState().multiLeg
  );
}

function addChopPriceLine(price, opts = {}) {
  if (!candleSeries || price == null || !Number.isFinite(Number(price))) return;
  const pl = candleSeries.createPriceLine({
    price: Number(price),
    color: opts.color || "#888",
    lineWidth: opts.lineWidth ?? 1,
    lineStyle: opts.lineStyle ?? 0,
    axisLabelVisible: true,
    title: opts.title || "",
  });
  chopGridPriceLines.push(pl);
}

function applyChopGridOverlay(overlay) {
  clearChopGridOverlay();
  if (!chopGridOverlayEnabled()) return;
  for (const batch of overlay?.batches || []) {
    const center = Number(batch.center);
    if (center > 0) {
      addChopPriceLine(center, {
        color: "#94a3b8",
        lineWidth: 2,
        lineStyle: 2,
        title: "中心",
      });
    }
    for (const lv of batch.levels || []) {
      const leg = lv.leg || "";
      const isLong = lv.side === "long";
      const gridColor = isLong
        ? "rgba(59, 130, 246, 0.45)"
        : "rgba(249, 115, 22, 0.45)";
      addChopPriceLine(lv.grid_price, {
        color: gridColor,
        lineStyle: 2,
        title: `${leg} 格`,
      });
      const entryPx = lv.entry_price != null ? Number(lv.entry_price) : null;
      if (entryPx != null && entryPx > 0) {
        const st = String(lv.entry_status || "");
        addChopPriceLine(entryPx, {
          color: isLong ? "#3b82f6" : "#f97316",
          lineWidth: st === "filled" ? 2 : 1,
          title: `${leg} ${st === "open" ? "挂单" : st === "filled" ? "成交" : "入场"}`,
        });
      }
      const tpPx = lv.tp_price != null ? Number(lv.tp_price) : null;
      if (tpPx != null && tpPx > 0) {
        const tpSt = String(lv.tp_status || "").toLowerCase();
        const tpOpen = ["open", "pending", "new", "submitted", "shadow"].includes(
          tpSt
        );
        addChopPriceLine(tpPx, {
          color: tpOpen ? "#a855f7" : "#6b7280",
          lineStyle: 1,
          title: `${leg}_tp`,
        });
      }
    }
  }
}

function applyChopRegimeBands(regions, candles) {
  if (!chopGridOverlayEnabled() || !candles?.length) return;
  if (!chopRegimeSeries) {
    chopRegimeSeries = chart.addHistogramSeries({
      priceScaleId: "chop-band",
      lastValueVisible: false,
      priceLineVisible: false,
    });
    chart.priceScale("chop-band").applyOptions({
      scaleMargins: { top: 0.78, bottom: 0 },
      visible: false,
    });
  }
  const spans = regions || [];
  const data = candles.map((c) => {
    const t = c.time;
    const inChop = spans.some(
      (r) => t >= Number(r.start) && t <= Number(r.end)
    );
    return {
      time: t,
      value: inChop ? 1 : 0,
      color: inChop ? "rgba(115, 191, 105, 0.22)" : "rgba(0,0,0,0)",
    };
  });
  chopRegimeSeries.setData(data);
}

function applyChopMapLayers(data, candles, opts = {}) {
  if (!opts.merge) {
    applyChopGridOverlay(data?.chop_grid_overlay || {});
    applyChopRegimeBands(data?.chop_regime_regions || [], candles);
  } else if (chopGridOverlayEnabled()) {
    applyChopGridOverlay(data?.chop_grid_overlay || {});
    applyChopRegimeBands(data?.chop_regime_regions || [], candles);
  }
}

function applyMainOverlays(mainOverlays, opts = {}) {
  const merge = !!opts.merge;
  if (!merge) clearMainOverlaySeries();
  const emaOn = document.getElementById("mainEma1200")?.checked;
  const wkOn = document.getElementById("mainWeeklyEma200")?.checked;
  const want = [];
  if (emaOn) want.push("ema_1200");
  if (wkOn) want.push("weekly_ema_200");
  for (const key of want) {
    const spec = (mainOverlays || {})[key];
    if (!spec?.available || !spec.points?.length) continue;
    let line = mainOverlaySeries.get(key);
    if (!line) {
      line = chart.addLineSeries({
        color: spec.color || "#888",
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: spec.label || key,
      });
      mainOverlaySeries.set(key, line);
    }
    const next = spec.points.map((p) => ({
        time: p.time,
        value: p.value,
      }));
    const points = merge ? mergeOverlayPoints(mainOverlayData.get(key) || [], next) : next;
    mainOverlayData.set(key, points);
    line.setData(points);
  }
}

function resetOhlcvLoadedRange() {
  ohlcvLoadedFrom = null;
  ohlcvLoadedTo = null;
  markerQueryFromIso = null;
  lastMarkerPollSince = null;
  lastTradeLinks = [];
  lastMarkerCounts = null;
  historyExhausted = false;
}

function resetMarkerQueryRange() {
  markerQueryFromIso = initialOhlcvRangeIso().from;
}

function initialOhlcvRangeIso() {
  const pageUrl = new URL(window.location.href);
  const fromUrl = pageUrl.searchParams.get("from");
  const toUrl = pageUrl.searchParams.get("to");
  if (fromUrl || toUrl) {
    const out = { full_range: "false" };
    if (fromUrl) out.from = fromUrl;
    if (toUrl) out.to = toUrl;
    else out.to = new Date().toISOString();
    return out;
  }
  const tf = document.getElementById("timeframeSelect")?.value || "2h";
  return Core.ohlcvInitialQueryRange(tf);
}

function mergeMarkersById(existing, incoming) {
  const byId = new Map((existing || []).map((m) => [m.id, m]));
  for (const m of incoming || []) {
    if (m?.id) byId.set(m.id, m);
  }
  return [...byId.values()].sort((a, b) => Number(a.time) - Number(b.time));
}

function tradeLinkKey(lk) {
  return [
    lk?.scope,
    lk?.strategy,
    lk?.entry_time,
    lk?.exit_time,
    lk?.entry_price,
    lk?.exit_price,
  ].join("|");
}

function mergeTradeLinks(existing, incoming) {
  const byKey = new Map();
  for (const lk of [...(existing || []), ...(incoming || [])]) {
    byKey.set(tradeLinkKey(lk), lk);
  }
  return [...byKey.values()];
}

function updateMarkerPollSince(serverTimestamp) {
  if (serverTimestamp) {
    lastMarkerPollSince = serverTimestamp;
  } else {
    // Fallback: Using client time minus 2 seconds to account for clock skew/latency.
    lastMarkerPollSince = new Date(Date.now() - 2000).toISOString();
  }
}

function markerRangeParams() {
  const to = new Date().toISOString();
  const from = markerQueryFromIso || initialOhlcvRangeIso().from;
  return { from, to, full_range: "false" };
}

function applyLoadedOhlcvRange(meta, candles) {
  if (candles?.length) {
    ohlcvLoadedFrom = Core.isoFromUnixSec(candles[0].time);
    ohlcvLoadedTo = Core.isoFromUnixSec(candles[candles.length - 1].time);
  } else {
    if (meta?.range_start) ohlcvLoadedFrom = String(meta.range_start);
    if (meta?.range_end) ohlcvLoadedTo = String(meta.range_end);
  }
}

function scheduleHistoryPrefetch(range) {
  if (!range || historyLoadInFlight || historyExhausted || !lastCandles.length) return;
  if (range.from > 25) return;
  if (historyLoadTimer) clearTimeout(historyLoadTimer);
  historyLoadTimer = setTimeout(() => {
    loadMoreHistory().catch((e) => setStatus(String(e)));
  }, 350);
}

async function loadMoreHistory() {
  if (historyLoadInFlight || historyExhausted || !lastCandles.length) return;
  const timeframe = document.getElementById("timeframeSelect").value;
  const symbol = document.getElementById("symbolSelect").value;
  const oldest = lastCandles[0].time;
  const chunkDays = Core.tradeMapHistoryChunkDays(timeframe);
  const newFromMs =
    Number(oldest) * 1000 - chunkDays * 86400000;
  const newFromIso = new Date(newFromMs).toISOString();
  historyLoadInFlight = true;
  try {
    const q = new URLSearchParams({
      symbol,
      timeframe,
      scopes: scopesParam(),
      include_pending: String(layersState().pending),
      from: newFromIso,
      to: Core.isoFromUnixSec(oldest),
      include_ohlcv: "full",
      include_features: "false",
      full_range: "false",
    });
    const { data, meta } = await Shell.api(`/api/trade-map/bundle?${q}`);
    const more = Core.sanitizeCandlesForLwc(data.ohlcv?.candles || []);
    if (!more.length) {
      historyExhausted = true;
      return;
    }
    const merged = Core.mergeCandlesByTime(more, lastCandles);
    if (merged.length === lastCandles.length) {
      historyExhausted = true;
      return;
    }
    lastCandles = merged;
    candleSeries.setData(merged);
    applyLoadedOhlcvRange(meta, merged);
    if (
      markerQueryFromIso == null ||
      new Date(newFromIso).getTime() < new Date(markerQueryFromIso).getTime()
    ) {
      markerQueryFromIso = newFromIso;
    }
    await refreshMarkersOnly();
    applyTradeLinks(data.trade_links || []);
  } finally {
    historyLoadInFlight = false;
  }
}

function bindTimeScaleSync() {
  if (timeSyncBound) return;
  timeSyncBound = true;
  chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range) return;
    for (const pane of subcharts.values()) {
      pane.chart.timeScale().setVisibleLogicalRange(range);
    }
    chart.priceScale("right").applyOptions({ autoScale: true });
    scheduleHistoryPrefetch(range);
  });
}

function destroySubchart(id) {
  const pane = subcharts.get(id);
  if (!pane) return;
  if (pane.chart) pane.chart.remove();
  const hostEl = document.getElementById(subchartDomId(id));
  if (hostEl) hostEl.remove();
  subcharts.delete(id);
}

function subchartDomId(id) {
  return `subchart-${String(id).replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

function clearStrategyChrome() {
  document
    .querySelectorAll(".subchart-strategy-header, .subchart-stage-header, .subchart-strategy-gap")
    .forEach((el) => {
      el.remove();
    });
}

function headerDomKey(item) {
  if (item.headerKind === "layer") return `hdr-layer-${item.strategy}`;
  if (item.headerKind === "stage") {
    return `hdr-stage-${item.accountLayer}-${item.strategy}-${item.stage}`;
  }
  return `hdr-strat-${item.accountLayer}-${item.strategy}`;
}

function ensureSubchartHeader(item) {
  const domId = subchartDomId(headerDomKey(item));
  let el = document.getElementById(domId);
  if (!el) {
    el = document.createElement("div");
    el.id = domId;
    el.className =
      item.headerKind === "stage" ? "subchart-stage-header" : "subchart-strategy-header";
    if (item.accountLayer) el.dataset.accountLayer = item.accountLayer;
    if (item.strategy) el.dataset.strategy = item.strategy;
    if (item.stage) el.dataset.stage = item.stage;
    if (item.headerKind) el.dataset.headerKind = item.headerKind;
    el.textContent = item.title;
    document.getElementById("subchartStack").appendChild(el);
  }
  return el;
}

function ensureStrategyGap(gapId) {
  const domId = subchartDomId(gapId);
  let el = document.getElementById(domId);
  if (!el) {
    el = document.createElement("div");
    el.id = domId;
    el.className = "subchart-strategy-gap";
    el.setAttribute("aria-hidden", "true");
    document.getElementById("subchartStack").appendChild(el);
  }
  return el;
}

function reorderSubchartStackDom(orderedDomIds) {
  const stack = document.getElementById("subchartStack");
  if (!stack) return;
  for (const domId of orderedDomIds) {
    const el = document.getElementById(domId);
    if (el) stack.appendChild(el);
  }
}

function ensureSubchartHost(id, label, strategyId) {
  const domId = subchartDomId(id);
  let host = document.getElementById(domId);
  if (!host) {
    host = document.createElement("div");
    host.id = domId;
    host.className = "subchart-pane";
    if (strategyId) host.dataset.strategy = strategyId;
    const caption = document.createElement("span");
    caption.className = "subchart-label";
    caption.textContent = label;
    host.appendChild(caption);
    document.getElementById("subchartStack").appendChild(host);
  }
  return host;
}

function ensureVolumePane(show, candles) {
  const id = "volume";
  if (!show) {
    destroySubchart(id);
    return;
  }
  let pane = subcharts.get(id);
  if (!pane) {
    const host = ensureSubchartHost(id, "成交量", "shared");
    const inner = document.createElement("div");
    inner.className = "subchart-pane-inner";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, subchartBaseOptions());
    const series = c.addHistogramSeries({ color: "#546e7a" });
    pane = { chart: c, series, host: inner, label: "成交量", kind: "volume" };
    subcharts.set(id, pane);
    bindTimeScaleSync();
  }
  const data = (candles || [])
    .filter((x) => x.volume != null)
    .map((x) => ({ time: x.time, value: x.volume, color: "#546e7a" }));
  pane.series.setData(data);
  requestAnimationFrame(() => resizeAllSubcharts());
}

function ensureFeaturePane(column, overlay, colorIndex) {
  const id = `feat:${column}`;
  if (!overlay?.available) {
    destroySubchart(id);
    return;
  }
  let pane = subcharts.get(id);
  if (!pane) {
    const meta = Core.lookupFeatureMeta(column);
    const label =
      meta.strategy_title && meta.stage_title
        ? `${meta.strategy_title}·${meta.stage_title}`
        : column;
    const host = ensureSubchartHost(id, label, meta.account_layer || meta.strategy);
    host.title = column;
    const inner = document.createElement("div");
    inner.className = "subchart-pane-inner";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, subchartBaseOptions());
    const color = Core.subchartColor(colorIndex);
    const series = c.addLineSeries({ color, lineWidth: 2 });
    let refSeries = null;
    if (overlay.reference_y != null && overlay.reference_y === overlay.reference_y) {
      refSeries = c.addLineSeries({
        color: "#8b949e",
        lineWidth: 1,
        lineStyle: 2,
      });
    }
    pane = {
      chart: c,
      series,
      refSeries,
      host: inner,
      label: column,
      kind: "feature",
    };
    subcharts.set(id, pane);
    bindTimeScaleSync();
  }
  const pts = overlay.points || [];
  pane.series.setData(pts.map((p) => ({ time: p.time, value: p.value })));
  if (pane.refSeries) {
    const y = overlay.reference_y ?? 0;
    pane.refSeries.setData(pts.map((p) => ({ time: p.time, value: y })));
  }
  requestAnimationFrame(() => {
    resizeAllSubcharts();
    syncSubchartsToMainRange();
  });
}

function syncSubcharts(candles, overlays) {
  const showVol = document.getElementById("paneVolume").checked;
  ensureVolumePane(showVol, candles);
  const wantFeatures = new Set(selectedFeatureColumns);
  for (const id of [...subcharts.keys()]) {
    if (id.startsWith("feat:") && !wantFeatures.has(id.slice(5))) destroySubchart(id);
  }
  clearStrategyChrome();

  const colsForPanes = selectedFeatureColumns.slice(0, MAX_FEATURE_SUBCHARTS);
  const panePlan = Core.orderFeaturePaneItems(colsForPanes, layersState());
  const domOrder = [];
  if (showVol) domOrder.push(subchartDomId("volume"));

  let colorIdx = 0;
  for (const item of panePlan) {
    if (item.type === "gap") {
      ensureStrategyGap(item.id);
      domOrder.push(subchartDomId(item.id));
    } else if (item.type === "header") {
      ensureSubchartHeader(item);
      domOrder.push(subchartDomId(headerDomKey(item)));
    } else if (item.type === "feature") {
      const fid = `feat:${item.column}`;
      ensureFeaturePane(item.column, overlays?.[item.column], colorIdx);
      colorIdx += 1;
      domOrder.push(subchartDomId(fid));
    }
  }
  reorderSubchartStackDom(domOrder);

  requestAnimationFrame(() => {
    resizeAllSubcharts();
    syncSubchartsToMainRange();
  });
}

function formatOverlayStatus(overlays) {
  if (!selectedFeatureColumns.length) return " · 特征:未选";
  const parts = selectedFeatureColumns.map((col) => {
    const o = overlays?.[col];
    if (!o) return `${col}:?`;
    if (!o.available) return `${col}:无数据`;
    return `${col}:${o.point_count ?? o.points?.length ?? 0}pts`;
  });
  return ` · 特征:${parts.join(",")}`;
}

function applyMarkers(rawMarkers, opts = {}) {
  const aligned = alignMarkersToLoadedCandles(rawMarkers || []);
  lastRawMarkers = opts.merge ? mergeMarkersById(lastRawMarkers, aligned) : aligned;
  markerById = new Map(lastRawMarkers.map((m) => [m.id, m]));
  candleSeries.setMarkers(Core.markersToLwc(lastRawMarkers, selectedMarkerId));
}

function nearestLoadedCandleTime(rawTime) {
  const t = Number(rawTime);
  if (!Number.isFinite(t) || !lastCandles.length) return t;
  let best = Number(lastCandles[0].time);
  let bestDist = Math.abs(best - t);
  for (const c of lastCandles) {
    const ct = Number(c.time);
    if (!Number.isFinite(ct)) continue;
    const dist = Math.abs(ct - t);
    if (dist < bestDist) {
      best = ct;
      bestDist = dist;
    }
  }
  return best;
}

function alignMarkersToLoadedCandles(markers) {
  if (!lastCandles.length) return markers || [];
  const times = lastCandles.map((c) => Number(c.time)).filter(Number.isFinite);
  if (!times.length) return markers || [];
  const first = times[0];
  const last = times[times.length - 1];
  const timeSet = new Set(times);
  return (markers || []).map((m) => {
    const t = Number(m.time);
    if (!Number.isFinite(t) || timeSet.has(t)) return m;
    const out = { ...m };
    const detail = { ...(out.detail || {}) };
    if (detail.order_time == null) detail.order_time = t;
    out.detail = detail;
    if (t < first) out.time = first;
    else if (t > last) out.time = last;
    else out.time = nearestLoadedCandleTime(t);
    return out;
  });
}

function clearTradeLinks() {
  if (!chart) return;
  for (const s of tradeLinkSeries) {
    try {
      chart.removeSeries(s);
    } catch (_) {
      /* already removed */
    }
  }
  tradeLinkSeries = [];
}

function clipLinkToCandles(link, candles) {
  if (!candles?.length) return link;
  const times = candles.map((c) => Number(c.time)).filter(Number.isFinite);
  const first = times[0];
  const last = times[times.length - 1];
  const out = { ...link };
  let t0 = Number(link.entry_time);
  let t1 = Number(link.exit_time);
  if (t0 < first) t0 = first;
  if (t0 > last) t0 = last;
  if (t1 < first) t1 = first;
  if (t1 > last) t1 = last;
  t0 = nearestLoadedCandleTime(t0);
  t1 = nearestLoadedCandleTime(t1);
  if (t1 <= t0) t1 = Math.min(last, t0 + Core.barDurationSec(document.getElementById("timeframeSelect")?.value || "2h"));
  out.entry_time = t0;
  out.exit_time = t1;
  return out;
}

function applyTradeLinks(links) {
  clearTradeLinks();
  if (!chart || !Array.isArray(links) || !links.length) return;
  const clipped = lastCandles.length
    ? links.map((lk) => clipLinkToCandles(lk, lastCandles))
    : links;
  for (const lk of clipped) {
    const t0 = Number(lk.entry_time);
    let t1 = Number(lk.exit_time);
    const p0 = Number(lk.entry_price);
    const p1 = Number(lk.exit_price);
    if (![t0, t1, p0, p1].every(Number.isFinite)) continue;
    if (t1 <= t0) {
      t1 = t0 + Core.barDurationSec(document.getElementById("timeframeSelect")?.value || "2h");
    }
    const color = lk.color || "#73BF69";
    const open = String(lk.status || "").toLowerCase() === "open";
    const series = chart.addLineSeries({
      color,
      lineWidth: 1,
      lineStyle: open ? 2 : 0,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    series.setData([
      { time: t0, value: p0 },
      { time: t1, value: p1 },
    ]);
    tradeLinkSeries.push(series);
  }
}

function scrollChartToMarker(markerTime) {
  if (!chart || !lastCandles.length) return;
  const idx = Core.scrollIndexForTime(lastCandles, markerTime);
  if (idx < 0) return;
  const pad = 15;
  const from = Math.max(0, idx - pad);
  const to = Math.min(lastCandles.length - 1, idx + pad);
  chart.timeScale().setVisibleLogicalRange({ from, to });
}

function highlightOrdersTableRow(markerId) {
  const tbody = document.getElementById("ordersDockBody");
  if (!tbody) return;
  tbody.querySelectorAll("tr[data-marker-id]").forEach((tr) => {
    const mid = tr.getAttribute("data-marker-id") || "";
    tr.classList.toggle("selected", !!markerId && mid === markerId);
    if (markerId && mid === markerId) {
      tr.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });
}

function selectMarker(markerId, { scrollChart = true, showDetail = true } = {}) {
  selectedMarkerId = markerId || null;
  applyMarkers(lastRawMarkers);
  highlightOrdersTableRow(selectedMarkerId);
  if (selectedMarkerId && scrollChart) {
    const raw = markerById.get(selectedMarkerId);
    if (raw?.time != null) scrollChartToMarker(raw.time);
  }
  if (selectedMarkerId && showDetail) {
    showMarkerDetail(selectedMarkerId);
  }
}

function applyOrdersDockVisibility() {
  const dock = document.getElementById("ordersDock");
  const btn = document.getElementById("ordersDockToggle");
  if (!dock || !btn) return;
  dock.classList.toggle("hidden", !ordersDockOpen);
  btn.classList.toggle("active", ordersDockOpen);
  btn.setAttribute("aria-pressed", ordersDockOpen ? "true" : "false");
}

function toggleOrdersDock(forceOpen) {
  ordersDockOpen = forceOpen ?? !ordersDockOpen;
  applyOrdersDockVisibility();
  saveLayout();
  if (ordersDockOpen) {
    refreshOrdersList().catch((e) => setStatus(String(e)));
  }
  requestAnimationFrame(() => {
    const el = document.getElementById("chart");
    if (chart && el) {
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
      resizeAllSubcharts();
    }
  });
}

async function refreshOrdersList() {
  if (!ordersDockOpen) return;
  const symbol = document.getElementById("symbolSelect").value;
  const tbody = document.getElementById("ordersDockBody");
  const countEl = document.getElementById("ordersDockCount");
  const showSym = Shell.isAllSymbols(symbol);
  document.querySelectorAll(".orders-th-symbol").forEach((th) => {
    th.classList.toggle("hidden", !showSym);
  });
  const colspan = Shell.ordersTableColspan(showSym);
  const q = new URLSearchParams({
    symbol,
    scopes: scopesParam(),
    limit: "500",
  });
  const exclude = ordersExcludeStatusParam();
  if (exclude) q.set("exclude_status", exclude);
  try {
    const { data, meta } = await Shell.api(`/api/orders/list?${q}`);
    const rows = data || [];
    countEl.textContent = `(${meta.count ?? rows.length})`;
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">无订单</td></tr>`;
      return;
    }
    tbody.innerHTML = Shell.buildOrdersTableRows(rows, {
      showSymbol: showSym,
      escHtml,
    });
    Shell.bindOrdersTableResize(document.getElementById("ordersDockTable"));
    tbody.querySelectorAll("tr[data-idx]").forEach((tr) => {
      tr.addEventListener("click", () => {
        tbody.querySelectorAll("tr").forEach((x) => x.classList.remove("selected"));
        tr.classList.add("selected");
        const mid = tr.getAttribute("data-marker-id");
        if (mid) selectMarker(mid);
        else {
          selectedMarkerId = null;
          highlightOrdersTableRow(null);
          applyMarkers(lastRawMarkers);
        }
      });
    });
    highlightOrdersTableRow(selectedMarkerId);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">${escHtml(String(e))}</td></tr>`;
    countEl.textContent = "";
  }
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/"/g, "&quot;");
}

function toggleFeatureDrawer(forceOpen) {
  const drawer = document.getElementById("featureDrawer");
  const backdrop = document.getElementById("featureDrawerBackdrop");
  const btn = document.getElementById("featurePanelBtn");
  if (!drawer || !btn) return;
  const open = forceOpen ?? drawer.classList.contains("hidden");
  drawer.classList.toggle("hidden", !open);
  if (backdrop) backdrop.classList.toggle("hidden", !open);
  document.body.classList.toggle("feature-drawer-open", open);
  btn.setAttribute("aria-expanded", open ? "true" : "false");
}

function setSelectedFeatures(cols, { refresh = true } = {}) {
  selectedFeatureColumns = [...new Set(cols.filter(Boolean))];
  renderFeaturePicker();
  saveLayout();
  if (refresh) refreshBundle().catch((e) => setStatus(String(e)));
}

function renderSelectedChips() {
  const el = document.getElementById("featureSelectedChips");
  if (!selectedFeatureColumns.length) {
    el.innerHTML = '<span class="muted">点击下方列名添加；或点「推荐」</span>';
    return;
  }
  el.innerHTML = selectedFeatureColumns
    .map((col) => {
      const m = Core.lookupFeatureMeta(col);
      const tag = `${m.account_layer_title || ""} › ${m.strategy_title || ""} › ${m.stage_title || ""}`;
      return `<span class="feature-chip"><span class="feature-chip-strategy" data-strategy="${escHtml(m.account_layer || "")}">${escHtml(tag)}</span>${escHtml(col)}<button type="button" data-remove-col="${escHtml(col)}" aria-label="移除">×</button></span>`;
    })
    .join("");
  el.querySelectorAll("[data-remove-col]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const col = btn.getAttribute("data-remove-col");
      setSelectedFeatures(selectedFeatureColumns.filter((c) => c !== col));
    });
  });
}

function renderFeaturePicker() {
  const list = document.getElementById("featureColumnList");
  const hint = document.getElementById("featurePickerHint");
  renderSelectedChips();
  if (!availableFeatureColumns.length) {
    list.innerHTML = '<p class="muted">当前周期无 features Parquet</p>';
    hint.textContent = "0";
    return;
  }
  hint.textContent = `${selectedFeatureColumns.length}/${availableFeatureColumns.length}`;
  const filtered = Core.filterFeatureColumns(availableFeatureColumns, featureSearchQuery);
  if (!filtered.length) {
    list.innerHTML = '<p class="muted">无匹配列</p>';
    return;
  }
  const groups = Core.groupFeatureColumnsByStrategy(filtered, layersState());
  list.innerHTML = groups
    .map(([title, cols, meta]) => {
      const items = cols
        .map((col) => {
          const on = selectedFeatureColumns.includes(col);
          const m = Core.lookupFeatureMeta(col);
          return `<label class="feature-item${on ? " selected" : ""}" data-account-layer="${escHtml(m.account_layer || "")}" data-stage="${escHtml(m.stage || "")}">
            <input type="checkbox" data-feature-col="${escHtml(col)}" ${on ? "checked" : ""} />
            <span>${escHtml(col)}</span>
          </label>`;
        })
        .join("");
      const dataAttrs = meta
        ? ` data-account-layer="${escHtml(meta.layer || "")}" data-strategy="${escHtml(meta.strategy || "")}" data-stage="${escHtml(meta.stage || "")}"`
        : "";
      return `<section class="feature-group"${dataAttrs}><h4 class="feature-group-title">${escHtml(title)} <span class="strategy-hint">(${cols.length})</span></h4><div class="feature-grid">${items}</div></section>`;
    })
    .join("");
  list.querySelectorAll("input[data-feature-col]").forEach((inp) => {
    inp.addEventListener("change", () => {
      const col = inp.getAttribute("data-feature-col");
      let next = [...selectedFeatureColumns];
      if (inp.checked) {
        if (!next.includes(col)) next.push(col);
      } else {
        next = next.filter((c) => c !== col);
      }
      setSelectedFeatures(next);
    });
  });
}

function bindFeaturePanel() {
  const btn = document.getElementById("featurePanelBtn");
  const drawer = document.getElementById("featureDrawer");
  const backdrop = document.getElementById("featureDrawerBackdrop");
  const closeBtn = document.getElementById("featureDrawerClose");
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    toggleFeatureDrawer(drawer.classList.contains("hidden"));
  });
  if (closeBtn) {
    closeBtn.addEventListener("click", () => toggleFeatureDrawer(false));
  }
  if (backdrop) {
    backdrop.addEventListener("click", () => toggleFeatureDrawer(false));
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && drawer && !drawer.classList.contains("hidden")) {
      toggleFeatureDrawer(false);
    }
  });
  drawer.addEventListener("click", (ev) => ev.stopPropagation());
  document.getElementById("featureSearch").addEventListener("input", (ev) => {
    featureSearchQuery = ev.target.value;
    renderFeaturePicker();
  });
  drawer.querySelectorAll("[data-feature-action]").forEach((el) => {
    el.addEventListener("click", () => {
      const action = el.getAttribute("data-feature-action");
      if (action === "clear") {
        setSelectedFeatures([]);
        return;
      }
      if (action === "preset-default" || action.startsWith("preset-")) {
        const key =
          action === "preset-default" ? "default" : action.replace("preset-", "");
        const strategyPresets = new Set([
          "tpc",
          "bpc",
          "me",
          "srb",
          "chop_grid",
          "trend_scalp",
          "spot_accum_simple",
        ]);
        let picks = [];
        if (strategyPresets.has(key)) {
          picks = Core.presetColumnsForStrategy(
            key,
            availableFeatureColumns,
            MAX_FEATURE_SUBCHARTS
          );
        } else {
          picks = Core.presetColumnsForAccountLayer(
            key,
            availableFeatureColumns,
            MAX_FEATURE_SUBCHARTS
          );
        }
        if (!picks.length) {
          const preset = Core.FEATURE_PRESETS[key] || Core.FEATURE_PRESETS.default;
          for (const name of preset) {
            if (availableFeatureColumns.includes(name)) picks.push(name);
          }
        }
        if (key === "default" || key === "spot") {
          for (const c of availableFeatureColumns) {
            if (String(c).toLowerCase().includes("weekly_ema") && !picks.includes(c)) {
              picks.push(c);
            }
          }
        }
        if (!picks.length && availableFeatureColumns.length) {
          picks.push(availableFeatureColumns[0]);
        }
        if (action.startsWith("preset-") && action !== "preset-default") {
          const merged = [...selectedFeatureColumns];
          for (const c of picks) {
            if (!merged.includes(c)) merged.push(c);
          }
          setSelectedFeatures(merged.slice(0, MAX_FEATURE_SUBCHARTS));
        } else {
          setSelectedFeatures(picks.slice(0, MAX_FEATURE_SUBCHARTS));
        }
      }
    });
  });
}

async function loadFeatureColumns() {
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  try {
    const { data } = await Shell.api(
      `/api/bus/features/columns?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`
    );
    availableFeatureColumns = data.columns || [];
    Core.setFeatureTaxonomy(data.taxonomy || null);
    const defaults = data.defaults || [];
    selectedFeatureColumns = selectedFeatureColumns.filter((c) =>
      availableFeatureColumns.includes(c)
    );
    if (!selectedFeatureColumns.length && defaults.length) {
      selectedFeatureColumns = [...defaults];
    }
    if (!selectedFeatureColumns.length && availableFeatureColumns.length) {
      selectedFeatureColumns = [availableFeatureColumns[0]];
    }
  } catch (_) {
    availableFeatureColumns = [];
  }
  renderFeaturePicker();
  saveLayout();
}

async function showMarkerDetail(markerId) {
  const panel = document.getElementById("detailPanel");
  const body = document.getElementById("detailBody");
  panel.classList.remove("hidden");
  const raw = markerById.get(markerId);
  body.textContent = JSON.stringify(raw || { id: markerId }, null, 2);
  try {
    const { data } = await Shell.api(
      `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(markerId)}`
    );
    body.textContent = JSON.stringify({ marker: raw, db: data }, null, 2);
  } catch (e) {
    body.textContent += `\n\n(DB lookup failed: ${e})`;
  }
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
  lastMarkerCounts = meta.marker_counts || null;
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
  const featParam = Core.featureColumnsParam(selectedFeatureColumns);
  const mainOl = Core.mainOverlaysQueryParam(
    document.getElementById("mainEma1200")?.checked,
    document.getElementById("mainWeeklyEma200")?.checked
  );
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
    q.set("include_features", mainOl ? "true" : "false");
    if (mainOl) q.set("main_overlays", mainOl);
    if (lastMarkerPollSince) q.set("since", lastMarkerPollSince);
    const lastT = lastCandles.length ? lastCandles[lastCandles.length - 1].time : null;
    if (lastT != null) {
      const barSec = Core.barDurationSec(timeframe);
      const tailFrom = Core.isoFromUnixSec(Number(lastT) - barSec * 5);
      q.set("ohlcv_from", tailFrom);
      q.set("ohlcv_to", new Date().toISOString());
    }
  } else {
    q.set("include_ohlcv", "full");
    q.set("include_features", "true");
    if (!ohlcvLoadedFrom || opts.resetOhlcvRange) {
      const init = initialOhlcvRangeIso();
      if (init.from) q.set("from", init.from);
      if (init.to) q.set("to", init.to);
      q.set("full_range", init.full_range || "false");
    } else {
      q.set("from", ohlcvLoadedFrom);
      q.set("to", new Date().toISOString());
      q.set("full_range", "false");
    }
    if (featParam) q.set("feature_columns", featParam);
    if (mainOl) q.set("main_overlays", mainOl);
  }

  const { data, meta } = await Shell.api(`/api/trade-map/bundle?${q}`);
  updateMarkerPollSince(meta?.server_timestamp);
  lastMarkerCounts = meta.marker_counts || null;
  const pageUrl = new URL(window.location.href);

  if (mode === "poll" && data.ohlcv?.candles?.length) {
    const newCandles = Core.sanitizeCandlesForLwc(data.ohlcv.candles);
    const prevLast =
      lastCandles.length > 0
        ? Number(lastCandles[lastCandles.length - 1].time)
        : null;
    const merged = Core.mergeCandlesByTime(lastCandles, newCandles);
    lastCandles = merged;
    for (const c of newCandles) {
      const t = Number(c.time);
      if (!Number.isFinite(t)) continue;
      if (prevLast != null && t < prevLast) continue;
      candleSeries.update(c);
    }
    applyMainOverlays(data.main_overlays || {}, { merge: true });
    applyChopMapLayers(data, lastCandles, { merge: true });
    ohlcvLoadedTo = meta.range_end || new Date().toISOString();
  } else if (mode !== "poll") {
    const candles = Core.sanitizeCandlesForLwc(data.ohlcv?.candles || []);
    lastCandles = candles;
    candleSeries.setData(candles);
    applyMainOverlays(data.main_overlays || {});
    applyChopMapLayers(data, candles);
    applyLoadedOhlcvRange(meta, candles);
    if (chartFitPending) {
      applyChartViewport(candles.length);
      chartFitPending = false;
    }
    syncSubcharts(candles, data.overlays || {});
  }

  const markers = data.markers || [];
  if (mode === "poll") {
    applyMarkers(markers, { merge: true });
    lastTradeLinks = mergeTradeLinks(lastTradeLinks, data.trade_links || []);
    applyTradeLinks(lastTradeLinks);
  } else {
    applyMarkers(markers);
    lastTradeLinks = data.trade_links || [];
    applyTradeLinks(lastTradeLinks);
  }

  if (mode !== "poll") {
    setStatusFromBundle(
      symbol,
      timeframe,
      lastCandles,
      markers,
      meta,
      data.overlays || {}
    );
  }
  tickClock();

  if (ordersDockOpen && mode !== "poll") {
    await refreshOrdersList();
  }

  const markerId = pageUrl.searchParams.get("marker_id");
  if (markerId && markerById.has(markerId)) {
    selectMarker(markerId, { scrollChart: true, showDetail: true });
  } else if (selectedMarkerId && !markerById.has(selectedMarkerId)) {
    selectedMarkerId = null;
    highlightOrdersTableRow(null);
  }
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    refreshBundle({ mode: "poll" }).catch((e) => setStatus(String(e)));
  }, POLL_MS);
}

function bindControls() {
  const rerun = (opts = {}) =>
    refreshBundle({ mode: "full", ...opts }).catch((e) => setStatus(String(e)));
  const rerunAll = async () => {
    chartFitPending = true;
    resetOhlcvLoadedRange();
    resetMarkerQueryRange();
    saveLayout();
    await loadFeatureColumns();
    await rerun();
  };
  document.getElementById("refreshBtn").addEventListener("click", () => {
    chartFitPending = true;
    rerunAll();
  });
  const resetChartRangeIds = new Set(["symbolSelect", "timeframeSelect"]);
  [
    "symbolSelect",
    "timeframeSelect",
    "mainEma1200",
    "mainWeeklyEma200",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "layerPending",
    "layerChopGrid",
    "paneVolume",
  ].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => {
      if (id === "paneVolume") {
        saveLayout();
        rerun();
        return;
      }
      if (resetChartRangeIds.has(id)) {
        if (id === "symbolSelect") {
          Shell.setSymbol(document.getElementById("symbolSelect").value);
        }
        resetOhlcvLoadedRange();
        resetMarkerQueryRange();
        chartFitPending = true;
        if (id.startsWith("layer")) renderFeaturePicker();
        if (ordersDockOpen) refreshOrdersList().catch(() => {});
        rerunAll();
        return;
      }
      if (id.startsWith("layer")) renderFeaturePicker();
      if (ordersDockOpen) refreshOrdersList().catch(() => {});
      rerun();
    })
  );
  document.getElementById("detailCloseBtn").addEventListener("click", () => {
    document.getElementById("detailPanel").classList.add("hidden");
  });
  document.getElementById("ordersDockToggle").addEventListener("click", () => {
    toggleOrdersDock();
  });
  ["hideExpired", "hideCanceled"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      Shell.saveOrdersFilter(Shell.ordersFilterFromControls());
      saveLayout();
      if (ordersDockOpen) refreshOrdersList().catch((e) => setStatus(String(e)));
    });
  });
  Shell.bindSymbolPersist("symbolSelect");
}

(async () => {
  try {
    Shell.initAppNav("trade-map");
    applyScopesFromStorage();
    applyLayoutToControls(loadLayout());
    initMainChart();
    bindFeaturePanel();
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect");
    Shell.bindOrdersFilterSync(() => {
      if (ordersDockOpen) refreshOrdersList().catch(() => {});
    });
    const pageUrl = new URL(window.location.href);
    const symParam = pageUrl.searchParams.get("symbol");
    if (symParam) {
      const sel = document.getElementById("symbolSelect");
      if ([...sel.options].some((o) => o.value === symParam)) {
        sel.value = symParam;
        Shell.setSymbol(symParam);
      }
    }
    await loadFeatureColumns();
    tickClock();
    if (clockTimer) clearInterval(clockTimer);
    clockTimer = setInterval(tickClock, 1000);
    await refreshBundle();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
