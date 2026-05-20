/**
 * Trade Map page — K线 + 账户层标记 + 动态附图 + Spot 资格侧栏。
 */

const Core = globalThis.MLBotTradeMapCore;
const Shell = globalThis.MLBotConsole;
const POLL_MS = 10000;
const LAYOUT_KEY = "mlbot_trade_map_layout_v1";

let chart;
let candleSeries;
let pollTimer;
let markerById = new Map();
let chartFitPending = true;
let timeSyncBound = false;

/** @type {Map<string, { chart, series, refSeries?, label, kind, host }>} */
const subcharts = new Map();

let availableFeatureColumns = [];
let selectedFeatureColumns = [];

const defaultLayout = () => ({
  volume: false,
  features: ["weekly_ema_200_position"],
  paneEligibility: true,
});

function loadLayout() {
  const stored = Core.parseStoredLayout(localStorage.getItem(LAYOUT_KEY));
  return { ...defaultLayout(), ...(stored || {}) };
}

function saveLayout() {
  const layout = {
    volume: document.getElementById("paneVolume").checked,
    features: [...selectedFeatureColumns],
    paneEligibility: document.getElementById("paneEligibility").checked,
  };
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  Shell.setScopesState(layersState());
}

function applyLayoutToControls(layout) {
  document.getElementById("paneVolume").checked = !!layout.volume;
  document.getElementById("paneEligibility").checked = layout.paneEligibility !== false;
  selectedFeatureColumns = Array.isArray(layout.features) ? [...layout.features] : [];
  applySidePanels();
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

function setStatus(msg) {
  document.getElementById("statusLine").textContent = msg;
}

function chartBaseOptions() {
  return {
    layout: { background: { color: "#0f1419" }, textColor: "#8b949e" },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "#30363d" },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
  };
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
  window.addEventListener("resize", resize);
  resize();
  bindTimeScaleSync();
}

function bindTimeScaleSync() {
  if (timeSyncBound) return;
  timeSyncBound = true;
  chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range) return;
    for (const pane of subcharts.values()) {
      pane.chart.timeScale().setVisibleLogicalRange(range);
    }
  });
}

function destroySubchart(id) {
  const pane = subcharts.get(id);
  if (!pane) return;
  pane.chart.remove();
  const hostEl = document.getElementById(`subchart-${id}`);
  if (hostEl) hostEl.remove();
  subcharts.delete(id);
}

function ensureSubchartHost(id, label) {
  let host = document.getElementById(`subchart-${id}`);
  if (!host) {
    host = document.createElement("div");
    host.id = `subchart-${id}`;
    host.className = "subchart-pane";
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
    const host = ensureSubchartHost(id, "成交量");
    const inner = document.createElement("div");
    inner.style.cssText = "position:absolute;inset:0;top:18px;";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, {
      ...chartBaseOptions(),
      timeScale: { visible: false },
    });
    const series = c.addHistogramSeries({ color: "#546e7a" });
    pane = { chart: c, series, host: inner, label: "成交量", kind: "volume" };
    subcharts.set(id, pane);
    bindTimeScaleSync();
  }
  const data = (candles || [])
    .filter((x) => x.volume != null)
    .map((x) => ({ time: x.time, value: x.volume, color: "#546e7a" }));
  pane.series.setData(data);
  pane.chart.applyOptions({
    width: pane.host.clientWidth,
    height: pane.host.clientHeight,
  });
}

function ensureFeaturePane(column, overlay, colorIndex) {
  const id = `feat:${column}`;
  if (!overlay?.available) {
    destroySubchart(id);
    return;
  }
  let pane = subcharts.get(id);
  if (!pane) {
    const host = ensureSubchartHost(id, column);
    const inner = document.createElement("div");
    inner.style.cssText = "position:absolute;inset:0;top:18px;";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, {
      ...chartBaseOptions(),
      timeScale: { visible: false },
    });
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
  pane.chart.applyOptions({
    width: pane.host.clientWidth,
    height: pane.host.clientHeight,
  });
}

function syncSubcharts(candles, overlays) {
  ensureVolumePane(document.getElementById("paneVolume").checked, candles);
  const wantFeatures = new Set(selectedFeatureColumns);
  for (const id of [...subcharts.keys()]) {
    if (id.startsWith("feat:") && !wantFeatures.has(id.slice(5))) destroySubchart(id);
  }
  let idx = 0;
  for (const col of selectedFeatureColumns) {
    ensureFeaturePane(col, overlays?.[col], idx);
    idx += 1;
  }
}

function applyMarkers(lwcMarkers) {
  markerById = new Map(lwcMarkers.map((m) => [m.id, m._raw]));
  candleSeries.setMarkers(lwcMarkers);
}

function applySidePanels() {
  const showElig = document.getElementById("paneEligibility").checked;
  document.getElementById("eligibilityPanel").classList.toggle("hidden", !showElig);
  document.getElementById("sidePanels").classList.toggle("collapsed", !showElig);
}

function renderFeaturePicker() {
  const list = document.getElementById("featureColumnList");
  const hint = document.getElementById("featurePickerHint");
  if (!availableFeatureColumns.length) {
    list.innerHTML = '<span class="muted">当前周期无 features Parquet</span>';
    hint.textContent = "";
    return;
  }
  hint.textContent = `(${selectedFeatureColumns.length}/${availableFeatureColumns.length})`;
  list.innerHTML = availableFeatureColumns
    .map((col) => {
      const checked = selectedFeatureColumns.includes(col) ? "checked" : "";
      return `<label><input type="checkbox" data-feature-col="${col}" ${checked} /> ${col}</label>`;
    })
    .join("");
  list.querySelectorAll("input[data-feature-col]").forEach((inp) => {
    inp.addEventListener("change", () => {
      const col = inp.getAttribute("data-feature-col");
      if (inp.checked) {
        if (!selectedFeatureColumns.includes(col)) selectedFeatureColumns.push(col);
      } else {
        selectedFeatureColumns = selectedFeatureColumns.filter((c) => c !== col);
      }
      saveLayout();
      refreshBundle().catch((e) => setStatus(String(e)));
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
    const defaults = data.defaults || [];
    selectedFeatureColumns = selectedFeatureColumns.filter((c) =>
      availableFeatureColumns.includes(c)
    );
    if (!selectedFeatureColumns.length && defaults.length) {
      selectedFeatureColumns = [...defaults];
    }
  } catch (_) {
    availableFeatureColumns = [];
  }
  renderFeaturePicker();
}

async function loadEligibility() {
  if (!document.getElementById("paneEligibility").checked) return;
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  try {
    const { data } = await Shell.api(
      `/api/spot/eligibility?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`
    );
    document.getElementById("eligibilityBody").textContent = Core.formatEligibility(data);
  } catch (e) {
    document.getElementById("eligibilityBody").textContent = String(e);
  }
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

async function refreshBundle() {
  const symbol = document.getElementById("symbolSelect").value;
  Shell.setSymbol(symbol);
  const timeframe = document.getElementById("timeframeSelect").value;
  const scopes = scopesParam();
  const pending = layersState().pending;
  const featParam = Core.featureColumnsParam(selectedFeatureColumns);
  setStatus("加载中…");
  const q = new URLSearchParams({
    symbol,
    timeframe,
    scopes,
    include_pending: String(pending),
    full_range: "true",
  });
  if (featParam) q.set("feature_columns", featParam);
  const pageUrl = new URL(window.location.href);
  if (pageUrl.searchParams.get("from")) {
    q.set("from", pageUrl.searchParams.get("from"));
    q.set("full_range", "false");
  }
  if (pageUrl.searchParams.get("to")) {
    q.set("to", pageUrl.searchParams.get("to"));
    q.set("full_range", "false");
  }
  const { data, meta } = await Shell.api(`/api/trade-map/bundle?${q}`);
  const candles = data.ohlcv?.candles || [];
  candleSeries.setData(candles);
  applyMarkers(Core.markersToLwc(data.markers || []));
  if (chartFitPending) {
    chart.timeScale().fitContent();
    chartFitPending = false;
  }
  syncSubcharts(candles, data.overlays || {});

  const deg = meta.degraded_ohlc || data.ohlcv?.degraded_ohlc;
  const rangeHint =
    meta.range_start && meta.range_end
      ? ` · ${meta.range_start.slice(0, 10)}→${meta.range_end.slice(0, 10)}`
      : "";
  const clipHint = meta.range_clipped ? ` · clipped ${meta.max_ohlcv_days || ""}d` : "";
  const busRows = meta.bars_1min_rows ? ` · 1m=${meta.bars_1min_rows}` : "";
  const featHint = meta.feature_columns?.length
    ? ` · 附图:${meta.feature_columns.length}`
    : "";
  setStatus(
    `${symbol} ${timeframe} · ${candles.length} bars · ${(data.markers || []).length} markers` +
      busRows +
      rangeHint +
      clipHint +
      featHint +
      (deg ? " · OHLC degraded" : "") +
      ` · ${new Date().toLocaleTimeString()}`
  );
  await loadEligibility();

  const markerId = pageUrl.searchParams.get("marker_id");
  if (markerId && markerById.has(markerId)) {
    showMarkerDetail(markerId);
  }
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    refreshBundle().catch((e) => setStatus(String(e)));
  }, POLL_MS);
}

function bindControls() {
  const rerun = () => refreshBundle().catch((e) => setStatus(String(e)));
  const rerunAll = async () => {
    chartFitPending = true;
    saveLayout();
    await loadFeatureColumns();
    rerun();
  };
  [
    "symbolSelect",
    "timeframeSelect",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "layerPending",
    "paneVolume",
    "paneEligibility",
  ].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => {
      if (id === "paneEligibility") {
        applySidePanels();
        saveLayout();
        loadEligibility().catch(() => {});
        return;
      }
      if (id === "paneVolume") {
        saveLayout();
        rerun();
        return;
      }
      if (id === "symbolSelect") Shell.setSymbol(document.getElementById("symbolSelect").value);
      rerunAll();
    })
  );
  document.getElementById("refreshBtn").addEventListener("click", rerunAll);
  Shell.bindSymbolPersist("symbolSelect");

  chart.subscribeClick((param) => {
    if (!param || param.time === undefined) return;
    const markers = candleSeries.markers?.() || [];
    const hit = markers.find((m) => m.time === param.time);
    if (hit?.id) showMarkerDetail(hit.id);
  });
}

(async () => {
  try {
    Shell.initAppNav("trade-map");
    applyScopesFromStorage();
    applyLayoutToControls(loadLayout());
    initMainChart();
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect");
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
    await refreshBundle();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
