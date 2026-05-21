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
let featureSearchQuery = "";
const MAX_FEATURE_SUBCHARTS = 8;

const defaultLayout = () => ({
  volume: false,
  features: ["weekly_ema_200_position"],
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
  };
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  Shell.setScopesState(layersState());
}

function applyLayoutToControls(layout) {
  document.getElementById("paneVolume").checked = !!layout.volume;
  selectedFeatureColumns = Array.isArray(layout.features) ? [...layout.features] : [];
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
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      barSpacing: 3,
      minBarSpacing: 0.5,
      rightOffset: 8,
    },
    rightPriceScale: { borderColor: "#30363d" },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
  };
}

function applyChartViewport(barCount) {
  const spacing = Core.barSpacingForCount(barCount);
  chart.timeScale().applyOptions({
    barSpacing: spacing,
    minBarSpacing: 0.5,
    rightOffset: 8,
  });
  chart.timeScale().fitContent();
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
  window.addEventListener("resize", () => {
    resize();
    resizeAllSubcharts();
  });
  resize();
  bindTimeScaleSync();
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

function fitFeatureSubcharts() {
  for (const pane of subcharts.values()) {
    if (pane.kind === "feature") {
      pane.chart.timeScale().fitContent();
    }
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
  requestAnimationFrame(() => {
    resizeAllSubcharts();
    pane.chart.timeScale().fitContent();
  });
}

function syncSubcharts(candles, overlays) {
  ensureVolumePane(document.getElementById("paneVolume").checked, candles);
  const wantFeatures = new Set(selectedFeatureColumns);
  for (const id of [...subcharts.keys()]) {
    if (id.startsWith("feat:") && !wantFeatures.has(id.slice(5))) destroySubchart(id);
  }
  const colsForPanes = selectedFeatureColumns.slice(0, MAX_FEATURE_SUBCHARTS);
  let idx = 0;
  for (const col of colsForPanes) {
    ensureFeaturePane(col, overlays?.[col], idx);
    idx += 1;
  }
  requestAnimationFrame(() => {
    resizeAllSubcharts();
    if (subcharts.size) fitFeatureSubcharts();
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

function applyMarkers(lwcMarkers) {
  markerById = new Map(lwcMarkers.map((m) => [m.id, m._raw]));
  candleSeries.setMarkers(lwcMarkers);
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/"/g, "&quot;");
}

function toggleFeaturePanel(forceOpen) {
  const panel = document.getElementById("featurePanel");
  const btn = document.getElementById("featurePanelBtn");
  const open = forceOpen ?? panel.classList.contains("hidden");
  panel.classList.toggle("hidden", !open);
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
    .map(
      (col) =>
        `<span class="feature-chip">${escHtml(col)}<button type="button" data-remove-col="${escHtml(col)}" aria-label="移除">×</button></span>`
    )
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
  const groups = Core.groupFeatureColumns(filtered);
  list.innerHTML = groups
    .map(([title, cols]) => {
      const items = cols
        .map((col) => {
          const on = selectedFeatureColumns.includes(col);
          return `<label class="feature-item${on ? " selected" : ""}">
            <input type="checkbox" data-feature-col="${escHtml(col)}" ${on ? "checked" : ""} />
            <span>${escHtml(col)}</span>
          </label>`;
        })
        .join("");
      return `<section class="feature-group"><h4 class="feature-group-title">${escHtml(title)} (${cols.length})</h4><div class="feature-grid">${items}</div></section>`;
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
  const panel = document.getElementById("featurePanel");
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    toggleFeaturePanel(panel.classList.contains("hidden"));
  });
  document.addEventListener("click", (ev) => {
    if (!panel.classList.contains("hidden") && !panel.contains(ev.target) && ev.target !== btn) {
      toggleFeaturePanel(false);
    }
  });
  panel.addEventListener("click", (ev) => ev.stopPropagation());
  document.getElementById("featureSearch").addEventListener("input", (ev) => {
    featureSearchQuery = ev.target.value;
    renderFeaturePicker();
  });
  panel.querySelectorAll("[data-feature-action]").forEach((el) => {
    el.addEventListener("click", () => {
      const action = el.getAttribute("data-feature-action");
      if (action === "clear") {
        setSelectedFeatures([]);
        return;
      }
      if (action === "preset-default") {
        const picks = [];
        for (const name of Core.FEATURE_PRESETS.default) {
          if (availableFeatureColumns.includes(name)) picks.push(name);
        }
        const fromApi = availableFeatureColumns.filter((c) =>
          String(c).toLowerCase().includes("weekly_ema")
        );
        for (const c of fromApi) {
          if (!picks.includes(c)) picks.push(c);
        }
        if (!picks.length && availableFeatureColumns.length) {
          picks.push(availableFeatureColumns[0]);
        }
        setSelectedFeatures(picks.slice(0, MAX_FEATURE_SUBCHARTS));
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
    applyChartViewport(candles.length);
    chartFitPending = false;
  }
  syncSubcharts(candles, data.overlays || {});

  const deg = meta.degraded_ohlc || data.ohlcv?.degraded_ohlc;
  const rangeHint =
    meta.range_start && meta.range_end
      ? ` · ${meta.range_start.slice(0, 10)}→${meta.range_end.slice(0, 10)}`
      : "";
  const clipHint = meta.range_clipped ? ` · clipped ${meta.max_ohlcv_days || ""}d` : "";
  const busRows = meta.bars_1min_rows ? ` · bus1m=${meta.bars_1min_rows}` : "";
  const histRows = meta.live_storage_1m_rows
    ? ` · hist1m=${meta.live_storage_1m_rows}`
    : "";
  const srcHint = meta.ohlcv_source ? ` · ${meta.ohlcv_source}` : "";
  const featCap =
    selectedFeatureColumns.length > MAX_FEATURE_SUBCHARTS
      ? ` · 附图限${MAX_FEATURE_SUBCHARTS}列`
      : "";
  setStatus(
    `${symbol} ${timeframe} · ${candles.length} bars · ${(data.markers || []).length} markers` +
      busRows +
      histRows +
      srcHint +
      rangeHint +
      clipHint +
      formatOverlayStatus(data.overlays || {}) +
      (featCap || "") +
      (deg ? " · OHLC degraded" : "") +
      ` · ${new Date().toLocaleTimeString()}`
  );
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
  document.getElementById("refreshBtn").addEventListener("click", () => {
    chartFitPending = true;
    rerunAll();
  });
  [
    "symbolSelect",
    "timeframeSelect",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "layerPending",
    "paneVolume",
  ].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => {
      if (id === "paneVolume") {
        saveLayout();
        rerun();
        return;
      }
      if (id === "symbolSelect") Shell.setSymbol(document.getElementById("symbolSelect").value);
      rerunAll();
    })
  );
  document.getElementById("detailCloseBtn").addEventListener("click", () => {
    document.getElementById("detailPanel").classList.add("hidden");
  });
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
    bindFeaturePanel();
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
