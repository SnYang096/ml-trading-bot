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
let lastRawMarkers = [];
let lastCandles = [];
let selectedMarkerId = null;
let ordersDockOpen = false;
let chartFitPending = true;
let timeSyncBound = false;
let clockTimer = null;

/** @type {Map<string, { chart, series, refSeries?, label, kind, host }>} */
const subcharts = new Map();

let availableFeatureColumns = [];
let selectedFeatureColumns = [];
let featureSearchQuery = "";
const MAX_FEATURE_SUBCHARTS = 8;

const defaultLayout = () => ({
  volume: false,
  features: ["weekly_ema_200_position"],
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
    ordersDock: ordersDockOpen,
  };
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  Shell.setScopesState(layersState());
}

function applyLayoutToControls(layout) {
  document.getElementById("paneVolume").checked = !!layout.volume;
  selectedFeatureColumns = Array.isArray(layout.features) ? [...layout.features] : [];
  ordersDockOpen = !!layout.ordersDock;
  applyOrdersDockVisibility();
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
  if (meta.bars_1min_rows) parts.push(`bus1m=${meta.bars_1min_rows}`);
  if (meta.live_storage_1m_rows) parts.push(`hist1m=${meta.live_storage_1m_rows}`);
  if (meta.ohlcv_source) parts.push(meta.ohlcv_source);
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
    rightPriceScale: { borderColor: "#30363d" },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
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

function applyMarkers(rawMarkers) {
  lastRawMarkers = rawMarkers || [];
  markerById = new Map(lastRawMarkers.map((m) => [m.id, m]));
  candleSeries.setMarkers(Core.markersToLwc(lastRawMarkers, selectedMarkerId));
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
      if (action === "preset-default" || action.startsWith("preset-")) {
        const key =
          action === "preset-default" ? "default" : action.replace("preset-", "");
        let picks = Core.presetColumnsForAccountLayer(
          key,
          availableFeatureColumns,
          MAX_FEATURE_SUBCHARTS
        );
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

async function refreshBundle() {
  const symbol = document.getElementById("symbolSelect").value;
  Shell.setSymbol(symbol);
  const timeframe = document.getElementById("timeframeSelect").value;
  const scopes = scopesParam();
  const pending = layersState().pending;
  const featParam = Core.featureColumnsParam(selectedFeatureColumns);
  setStatusLoading();
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
  const candles = Core.sanitizeCandlesForLwc(data.ohlcv?.candles || []);
  lastCandles = candles;
  candleSeries.setData(candles);
  const markers = data.markers || [];
  applyMarkers(markers);
  if (chartFitPending) {
    applyChartViewport(candles.length);
    chartFitPending = false;
  }
  syncSubcharts(candles, data.overlays || {});

  setStatusFromBundle(symbol, timeframe, candles, markers, meta, data.overlays || {});
  tickClock();

  if (ordersDockOpen) {
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
      if (id.startsWith("layer")) renderFeaturePicker();
      if (ordersDockOpen) refreshOrdersList().catch(() => {});
      rerunAll();
    })
  );
  document.getElementById("detailCloseBtn").addEventListener("click", () => {
    document.getElementById("detailPanel").classList.add("hidden");
  });
  document.getElementById("ordersDockToggle").addEventListener("click", () => {
    toggleOrdersDock();
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
