/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function destroySubchart(id) {
  const pane = S.subcharts.get(id);
  if (!pane) return;
  if (pane.chart) pane.chart.remove();
  const hostEl = document.getElementById(subchartDomId(id));
  if (hostEl) hostEl.remove();
  S.subcharts.delete(id);
}

function visibleCandleIndexRange(candles) {
  const list = candles || [];
  if (!list.length) return { from: 0, to: 0 };
  if (!S.chart?.timeScale) {
    return { from: 0, to: list.length - 1 };
  }
  const logical = S.chart.timeScale().getVisibleLogicalRange();
  if (!logical) {
    return { from: Math.max(0, list.length - 80), to: list.length - 1 };
  }
  const from = Math.max(
    0,
    Math.min(list.length - 1, Math.floor(Number(logical.from)))
  );
  const to = Math.max(
    0,
    Math.min(list.length - 1, Math.ceil(Number(logical.to)))
  );
  return { from: Math.min(from, to), to: Math.max(from, to) };
}

function candleIndexAtTime(candles, timeSec) {
  const list = candles || [];
  const t = Number(timeSec);
  if (!list.length || !Number.isFinite(t)) return -1;
  for (let i = 0; i < list.length; i++) {
    if (Number(list[i].time) === t) return i;
  }
  let best = -1;
  for (let i = 0; i < list.length; i++) {
    if (Number(list[i].time) <= t) best = i;
  }
  return best;
}

/** Visible window plus crosshair bar (so highlight/scroll work when bar is off-screen). */
function indicesForMetricsTable(candles, highlightTimeSec) {
  let { from, to } = visibleCandleIndexRange(candles);
  const hi = candleIndexAtTime(candles, highlightTimeSec);
  if (hi >= 0) {
    const pad = 12;
    from = Math.min(from, Math.max(0, hi - pad));
    to = Math.max(to, Math.min((candles || []).length - 1, hi + pad));
  }
  return { from, to };
}

function metricsTableDomId(item) {
  return subchartDomId(item.id || "metrics-table");
}

function subchartDomId(id) {
  return `subchart-${String(id).replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function clearStrategyChrome({ keepMetricsTable = false } = {}) {
  const selectors = [
    ".subchart-strategy-header",
    ".subchart-stage-header",
    ".subchart-strategy-gap",
    ".subchart-threshold-status",
  ];
  if (!keepMetricsTable) selectors.push(".subchart-feature-metrics");
  document.querySelectorAll(selectors.join(", ")).forEach((el) => {
    el.remove();
  });
  const insp = document.getElementById("featureBarInspector");
  if (insp) insp.classList.add("hidden");
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

function scheduleSubchartLayout() {
  resizeAllSubcharts();
  requestAnimationFrame(() => {
    resizeAllSubcharts();
    syncSubchartsToMainRange();
  });
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
  let pane = S.subcharts.get(id);
  if (!pane) {
    const host = ensureSubchartHost(id, "成交量", "shared");
    const inner = document.createElement("div");
    inner.className = "subchart-pane-inner";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, subchartBaseOptions());
    const series = c.addHistogramSeries({ color: "#546e7a" });
    pane = { chart: c, series, host: inner, label: "成交量", kind: "volume" };
    S.subcharts.set(id, pane);
    bindTimeScaleSync();
  }
  const cap = pane.host?.parentElement?.querySelector(".subchart-label");
  if (cap) {
    cap.textContent = "成交量";
    cap.title = "每根K线周期内1分钟成交量求和（与0-1特征尺度不同）";
  }
  const data = (candles || [])
    .filter((x) => x.volume != null && Number.isFinite(Number(x.volume)))
    .map((x) => ({ time: x.time, value: Number(x.volume), color: "#546e7a" }));
  pane.series.setData(data);
  if (cap) {
    if (!data.length) {
      cap.title =
        "K 线无 volume 字段（检查 feature bus bars_1min 是否含成交量列）";
    } else {
      cap.title = "每根K线周期内1分钟成交量求和（与0-1特征尺度不同）";
    }
  }
  scheduleSubchartLayout();
}

function passBadge(passed) {
  if (passed === true) return " ✓";
  if (passed === false) return " ✗";
  return "";
}

function featurePaneCaption(column, overlay) {
  const meta = Core.lookupFeatureMeta(column);
  const base =
    meta.strategy_title && meta.stage_title
      ? `${meta.strategy_title} · ${meta.stage_title}`
      : column;
  const latest = overlay?.latest;
  const hint = overlay?.semantic_hint || "";
  const refLines = overlay?.reference_lines || [];
  const refHint =
    refLines.length > 0
      ? refLines.map((r) => r.label || `阈${r.y}`).join(" · ")
      : overlay?.reference_y != null && overlay.reference_y === overlay.reference_y
        ? `阈=${Number(overlay.reference_y)}`
        : "";
  if (latest != null && latest === latest && Number.isFinite(Number(latest))) {
    const v = Number(latest);
    const decimals =
      column.includes("chop") || column.includes("pct") || column.includes("pos")
        ? 3
        : 2;
    const valStr = v.toFixed(decimals);
    let pass = null;
    if (refLines.length === 2 && String(column).includes("box_pos")) {
      const lo = refLines.find((r) => String(r.operator).includes(">="));
      const hi = refLines.find((r) => String(r.operator).includes("<="));
      const okLo = lo ? Core.valuePassesRefLine(v, lo) : true;
      const okHi = hi ? Core.valuePassesRefLine(v, hi) : true;
      pass = okLo === true && okHi === true;
    } else if (refLines.length === 1) {
      pass = Core.valuePassesRefLine(v, refLines[0]);
    }
    const parts = [base, valStr + passBadge(pass)];
    if (hint) parts.push(`(${hint})`);
    else if (refHint) parts.push(`(${refHint})`);
    return parts.join(" ");
  }
  if (refHint) return `${base} · ${refHint}`;
  if (overlay?.available === false) return `${base} · 无数据`;
  return base;
}

function thresholdStatusDomId(item) {
  return subchartDomId(`status-${item.strategy}-${item.stage}`);
}

function renderThresholdMetricTable(rows, { caption } = {}) {
  if (!rows.length) {
    return '<p class="muted threshold-table-empty">box_prefilter: 无数据</p>';
  }
  const head = caption
    ? `<div class="threshold-table-caption">${escHtml(caption)}</div>`
    : "";
  const body = rows
    .map((r) => {
      const passCls =
        r.pass === true ? "pass-ok" : r.pass === false ? "pass-fail" : "pass-na";
      const val = r.value != null ? escHtml(String(r.value)) : "—";
      return `<tr class="${passCls}">
        <td class="yaml-key">${escHtml(r.yaml)}</td>
        <td class="yaml-val">${val}</td>
        <td class="yaml-th">${escHtml(r.threshold)}</td>
        <td class="yaml-pf">${passBadge(r.pass)}</td>
      </tr>`;
    })
    .join("");
  return `${head}<table class="threshold-metric-table"><thead><tr>
    <th>YAML</th><th>值</th><th>阈</th><th></th>
  </tr></thead><tbody>${body}</tbody></table>`;
}

function ensureThresholdStatusPane(item, overlays, timeSec) {
  const domId = thresholdStatusDomId(item);
  let el = document.getElementById(domId);
  if (!el) {
    el = document.createElement("div");
    el.id = domId;
    el.className = "subchart-threshold-status";
    el.dataset.strategy = item.strategy || "";
    el.dataset.stage = item.stage || "";
    el.dataset.columns = (item.columns || []).join(",");
    document.getElementById("subchartStack").appendChild(el);
  }
  const rows = Core.buildThresholdMetricRows(item.columns || [], overlays, timeSec);
  const cap =
    timeSec != null
      ? `regime.box_prefilter · ${Shell.formatOrderTime(timeSec)}`
      : "regime.box_prefilter · 最新 bar";
  el.innerHTML = renderThresholdMetricTable(rows, { caption: cap });
  return domId;
}

function ensureFeatureBarInspector() {
  let el = document.getElementById("featureBarInspector");
  if (!el) {
    el = document.createElement("div");
    el.id = "featureBarInspector";
    el.className = "feature-bar-inspector hidden";
    const stack = document.getElementById("subchartStack");
    if (stack?.parentElement) {
      stack.parentElement.insertBefore(el, stack);
    }
  }
  return el;
}

function updateFeatureBarInspector(timeSec, overlays) {
  const el = ensureFeatureBarInspector();
  const focus = String(S.featureStrategyFocus || "").trim();
  if (!focus || timeSec == null) {
    el.classList.add("hidden");
    return;
  }
  const layers = layersState();
  const cols = Core.resolveSubchartColumns(
    S.selectedFeatureColumns,
    S.availableFeatureColumns,
    layers,
    focus,
    S.MAX_FEATURE_SUBCHARTS
  );
  const regimeCols = (cols || []).filter((c) =>
    ["box_stability_60", "box_width_pct_60", "box_touches_hi_60", "box_touches_lo_60"].includes(c)
  );
  const chopRows =
    focus === "chop_grid" && regimeCols.length
      ? Core.buildThresholdMetricRows(regimeCols, overlays, timeSec)
      : [];
  const prefilterCol = cols.find((c) => c === "box_pos_60");
  const preRows = [];
  if (prefilterCol) {
    const o = overlays?.[prefilterCol] || S.lastOverlays?.[prefilterCol];
    const v = Core.overlayValueAtTime(o, timeSec);
    const refs = o?.reference_lines || [];
    const lo = refs.find((r) => String(r.operator).includes(">="));
    const hi = refs.find((r) => String(r.operator).includes("<="));
    let pass = null;
    if (v != null && lo && hi) {
      pass =
        Core.valuePassesRefLine(v, lo) === true && Core.valuePassesRefLine(v, hi) === true;
    }
    preRows.push({
      yaml: "rules.box_pos_60",
      label: "box_pos_60",
      value: v != null ? v.toFixed(3) : null,
      threshold: "0.35 – 0.65",
      pass,
    });
  }
  const chopLine = cols.find((c) => c === "bpc_semantic_chop");
  const chopRows2 = [];
  if (chopLine) {
    const o = overlays?.[chopLine] || S.lastOverlays?.[chopLine];
    const v = Core.overlayValueAtTime(o, timeSec);
    const refs = o?.reference_lines || [];
    const enter = refs.find((r) => String(r.operator).includes(">="));
    const exitR = refs.find((r) => String(r.operator).includes("<"));
    chopRows2.push({
      yaml: "regime.entry_chop_min",
      label: "enter",
      value: v != null ? v.toFixed(3) : null,
      threshold: enter ? enter.label : "≥0.50",
      pass: enter && v != null ? Core.valuePassesRefLine(v, enter) : null,
    });
    chopRows2.push({
      yaml: "regime.exit_chop_below",
      label: "exit",
      value: v != null ? v.toFixed(3) : null,
      threshold: exitR ? exitR.label : "<0.32",
      pass: exitR && v != null ? Core.valuePassesRefLine(v, exitR) : null,
    });
  }
  let html = `<div class="inspector-head">${escHtml(focus)} · ${escHtml(Shell.formatOrderTime(timeSec))}</div>`;
  if (chopRows2.length) {
    html += renderThresholdMetricTable(chopRows2, { caption: "regime · bpc_semantic_chop" });
  }
  if (chopRows.length) {
    html += renderThresholdMetricTable(chopRows, { caption: "regime.box_prefilter" });
  }
  if (preRows.length) {
    html += renderThresholdMetricTable(preRows, { caption: "prefilter · rules" });
  }
  if (!chopRows2.length && !chopRows.length && !preRows.length) {
    el.classList.add("hidden");
    return;
  }
  el.innerHTML = html;
  el.classList.remove("hidden");
  document.querySelectorAll(".subchart-threshold-status").forEach((pane) => {
    const cols = String(pane.dataset.columns || "")
      .split(",")
      .filter(Boolean);
    if (!cols.length) return;
    pane.innerHTML = renderThresholdMetricTable(
      Core.buildThresholdMetricRows(cols, overlays, timeSec),
      {
        caption: `regime.box_prefilter · ${Shell.formatOrderTime(timeSec)}`,
      }
    );
  });
}

function formatMetricsBarHeader(timeSec) {
  const s = Shell.formatOrderTime(timeSec);
  if (!s) return String(timeSec);
  const m = s.match(/(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})/);
  if (m) return `${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
  return s.length > 14 ? s.slice(5, 16) : s;
}

function renderChopMetricsTableHtml(candles, overlays, cols, highlightTimeSec) {
  const rowSpecs = Core.chopGridMetricsRowSpecs(cols, overlays);
  if (!rowSpecs.length) {
    return '<p class="muted">无指标列（请选 Chop 预设或勾选 bpc / box_pos / box 结构列）</p>';
  }
  const { from, to } = indicesForMetricsTable(candles, highlightTimeSec);
  const bars = [];
  for (let i = from; i <= to; i++) {
    const t = candles[i]?.time;
    if (t == null) continue;
    bars.push({ time: t, label: formatMetricsBarHeader(t) });
  }
  if (!bars.length) {
    return '<p class="muted">主图可见区间无 K 线</p>';
  }
  let head = '<thead><tr><th class="row-label-h">可入场</th>';
  for (const b of bars) {
    const active =
      highlightTimeSec != null && Number(highlightTimeSec) === Number(b.time);
    const canEnter = Core.chopGridBarCanEnter(cols, overlays, b.time);
    const gate = canEnter ? "✓" : "×";
    const gateCls = canEnter ? "bar-enter-ok" : "bar-enter-fail";
    head += `<th class="bar-col-h bar-enter-h ${gateCls}${active ? " bar-col-active" : ""}" data-time="${b.time}"><span class="bar-enter-mark">${gate}</span></th>`;
  }
  head += '</tr><tr><th class="row-label-h">时间</th>';
  for (const b of bars) {
    const active =
      highlightTimeSec != null && Number(highlightTimeSec) === Number(b.time);
    head += `<th class="bar-col-h${active ? " bar-col-active" : ""}" data-time="${b.time}">${escHtml(b.label)}</th>`;
  }
  head += "</tr></thead>";
  let body = "<tbody>";
  for (const row of rowSpecs) {
    body += "<tr>";
    body += `<th class="row-label"><div class="col-name">${escHtml(row.label)}</div><div class="col-thresh">${escHtml(row.threshold || "")}</div></th>`;
    for (const b of bars) {
      const cell = Core.chopGridMetricsRowCell(row, overlays, b.time);
      const active =
        highlightTimeSec != null && Number(highlightTimeSec) === Number(b.time);
      const cls =
        (cell.pass === true ? "pass-ok" : cell.pass === false ? "pass-fail" : "") +
        (active ? " bar-col-active" : "");
      body += `<td class="yaml-val ${cls}" data-time="${b.time}">${escHtml(cell.value)}</td>`;
    }
    body += "</tr>";
  }
  body += "</tbody>";
  return `<table class="feature-metrics-table layout-pivot">${head}${body}</table>`;
}

const METRICS_TABLE_SCROLL_DELAY_MS = 3000;
let metricsTableScrollTimer = null;
let metricsTableScrollPendingTime = null;
let lastMetricsTableRangeKey = "";

function visibleMetricsRangeKey(candles, highlightTimeSec) {
  const { from, to } = indicesForMetricsTable(candles, highlightTimeSec);
  const hi =
    highlightTimeSec != null
      ? Number(highlightTimeSec)
      : S.highlightBarTime != null
        ? Number(S.highlightBarTime)
        : null;
  return `${from}:${to}:${hi ?? ""}`;
}

function applyMetricsColumnHighlight(scroll, timeSec) {
  const t = timeSec != null ? Number(timeSec) : null;
  scroll.querySelectorAll(".bar-col-active").forEach((el) => {
    el.classList.remove("bar-col-active");
  });
  if (t == null || !Number.isFinite(t)) return;
  scroll.querySelectorAll("[data-time]").forEach((el) => {
    if (Number(el.getAttribute("data-time")) === t) {
      el.classList.add("bar-col-active");
    }
  });
}

function scrollMetricsTableToActiveColumn(scroll, timeSec) {
  const t = timeSec != null ? Number(timeSec) : null;
  if (t == null || !Number.isFinite(t)) return;
  let anchor = null;
  scroll.querySelectorAll("th.bar-col-h[data-time]").forEach((el) => {
    if (Number(el.getAttribute("data-time")) === t) anchor = el;
  });
  if (!anchor) {
    anchor = scroll.querySelector("th.bar-col-h.bar-col-active");
  }
  if (!anchor) return;
  const targetLeft =
    anchor.offsetLeft - scroll.clientWidth / 2 + anchor.offsetWidth / 2;
  scroll.scrollTo({
    left: Math.max(0, targetLeft),
    behavior: "auto",
  });
}

function cancelMetricsTableScrollSchedule() {
  if (metricsTableScrollTimer != null) {
    clearTimeout(metricsTableScrollTimer);
    metricsTableScrollTimer = null;
  }
  metricsTableScrollPendingTime = null;
}

/** Highlight only unless allowScroll (crosshair idle); poll/resize never scroll. */
function highlightMetricsTableColumn(timeSec, { allowScroll = false } = {}) {
  const scroll = document.getElementById("featureMetricsScroll");
  if (!scroll) return;
  const t = timeSec != null ? Number(timeSec) : null;
  applyMetricsColumnHighlight(scroll, t);
  if (!allowScroll || t == null || !Number.isFinite(t)) {
    cancelMetricsTableScrollSchedule();
    return;
  }
  metricsTableScrollPendingTime = t;
  cancelMetricsTableScrollSchedule();
  metricsTableScrollTimer = setTimeout(() => {
    metricsTableScrollTimer = null;
    const pending = metricsTableScrollPendingTime;
    metricsTableScrollPendingTime = null;
    if (pending == null) return;
    const el = document.getElementById("featureMetricsScroll");
    if (!el) return;
    scrollMetricsTableToActiveColumn(el, pending);
  }, METRICS_TABLE_SCROLL_DELAY_MS);
}

function ensureFeatureMetricsTablePane(item, candles, overlays, highlightTimeSec) {
  const domId = metricsTableDomId(item);
  let host = document.getElementById(domId);
  if (!host) {
    host = document.createElement("div");
    host.id = domId;
    host.className = "subchart-feature-metrics";
    host.dataset.strategy = item.strategy || "";
    const cap = document.createElement("div");
    cap.className = "metrics-table-caption";
    cap.textContent =
      "Chop Grid 指标表 · 顶行✓/×=可新开网格 · 滚轮自管横向位置 · 仅十字线停住约3s才自动滚到列";
    const scroll = document.createElement("div");
    scroll.id = "featureMetricsScroll";
    scroll.className = "metrics-table-scroll layout-pivot-host";
    host.appendChild(cap);
    host.appendChild(scroll);
    document.getElementById("subchartStack").appendChild(host);
    const subId = item.id || "metrics-chop_grid";
    S.subcharts.set(subId, { kind: "metrics_table", scrollEl: scroll, host });
  }
  const scroll = document.getElementById("featureMetricsScroll");
  if (!scroll) return domId;
  if (scroll.querySelector(".feature-metrics-table")) {
    refreshFeatureMetricsPanel(highlightTimeSec, {
      rebuild: true,
      preserveScrollLeft: true,
    });
    return domId;
  }
  const cols = item.columns || [];
  scroll.innerHTML = renderChopMetricsTableHtml(
    candles,
    overlays,
    cols,
    highlightTimeSec
  );
  lastMetricsTableRangeKey = visibleMetricsRangeKey(candles, highlightTimeSec);
  requestAnimationFrame(() => highlightMetricsTableColumn(highlightTimeSec));
  return domId;
}

function metricsTableHasBarColumn(scroll, timeSec) {
  const t = timeSec != null ? Number(timeSec) : null;
  if (!scroll || t == null || !Number.isFinite(t)) return false;
  let found = false;
  scroll.querySelectorAll("[data-time]").forEach((el) => {
    if (Number(el.getAttribute("data-time")) === t) found = true;
  });
  return found;
}

function refreshFeatureMetricsPanel(
  highlightTimeSec,
  { rebuild = true, preserveScrollLeft = true } = {}
) {
  if (!document.getElementById("subchartStack")) return;
  const overlays = S.lastOverlays || {};
  const candles = S.lastCandles || [];
  const cols = Core.resolveSubchartColumns(
    S.selectedFeatureColumns,
    S.availableFeatureColumns,
    layersState(),
    S.featureStrategyFocus,
    S.MAX_FEATURE_SUBCHARTS
  );
  if (!Core.chopMetricsTableActive(S.featureStrategyFocus, cols) || !candles.length) return;
  const host = document.querySelector(".subchart-feature-metrics");
  if (!host) return;
  const scroll = document.getElementById("featureMetricsScroll");
  if (!scroll) return;
  const timeSec =
    highlightTimeSec != null
      ? highlightTimeSec
      : S.highlightBarTime != null
        ? S.highlightBarTime
        : candles[candles.length - 1]?.time;
  const rangeKey = visibleMetricsRangeKey(candles, timeSec);
  const hasTable = !!scroll.querySelector(".feature-metrics-table");
  const rangeChanged = rangeKey !== lastMetricsTableRangeKey;
  const needsBar =
    rebuild && (!hasTable || !metricsTableHasBarColumn(scroll, timeSec));
  const mustRebuild = rebuild && (needsBar || rangeChanged);
  if (!mustRebuild) {
    highlightMetricsTableColumn(timeSec);
    return;
  }
  const prevScrollLeft =
    preserveScrollLeft && hasTable ? scroll.scrollLeft : null;
  scroll.innerHTML = renderChopMetricsTableHtml(candles, overlays, cols, timeSec);
  lastMetricsTableRangeKey = rangeKey;
  if (prevScrollLeft != null) scroll.scrollLeft = prevScrollLeft;
  requestAnimationFrame(() => highlightMetricsTableColumn(timeSec));
}

function refreshThresholdTablesAtTime(timeSec) {
  if (!document.getElementById("subchartStack")) return;
  if (Core.chopMetricsTableActive(S.featureStrategyFocus, S.selectedFeatureColumns)) {
    if (timeSec != null) S.highlightBarTime = timeSec;
    refreshFeatureMetricsPanel(timeSec, { rebuild: true, preserveScrollLeft: true });
    const insp = document.getElementById("featureBarInspector");
    if (insp) insp.classList.add("hidden");
    return;
  }
  const overlays = S.lastOverlays || {};
  document.querySelectorAll(".subchart-threshold-status").forEach((pane) => {
    const cols = String(pane.dataset.columns || "")
      .split(",")
      .filter(Boolean);
    if (!cols.length) return;
    pane.innerHTML = renderThresholdMetricTable(
      Core.buildThresholdMetricRows(cols, overlays, timeSec),
      {
        caption:
          timeSec != null
            ? `regime.box_prefilter · ${Shell.formatOrderTime(timeSec)}`
            : "regime.box_prefilter · 最新 bar",
      }
    );
  });
  if (!Core.chopGridUsesMetricsTable("chop_grid", S.featureStrategyFocus)) {
    updateFeatureBarInspector(timeSec, overlays);
  }
}

function refLineTimeline(pts, candles) {
  if (pts?.length) return pts.map((p) => ({ time: p.time, value: p.value }));
  return (candles || [])
    .filter((c) => c && c.time != null)
    .map((c) => ({ time: c.time, value: 0 }));
}

function syncFeatureRefLines(pane, overlay, pts, candles) {
  if (pane.refSeriesList) {
    for (const s of pane.refSeriesList) {
      try {
        pane.chart.removeSeries(s);
      } catch (_) {
        /* */
      }
    }
  }
  pane.refSeriesList = [];
  const refLines =
    overlay.reference_lines?.length > 0
      ? overlay.reference_lines
      : overlay.reference_y != null && overlay.reference_y === overlay.reference_y
        ? [{ y: overlay.reference_y, label: "" }]
        : [];
  const timeline = refLineTimeline(pts, candles);
  if (!timeline.length || !refLines.length) return;
  for (const rl of refLines) {
    const y = Number(rl.y);
    if (!Number.isFinite(y)) continue;
    const rs = pane.chart.addLineSeries({
      color: "#8b949e",
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: rl.label || "",
    });
    rs.setData(timeline.map((p) => ({ time: p.time, value: y })));
    pane.refSeriesList.push(rs);
  }
}

function ensureFeaturePane(column, overlay, colorIndex, candles) {
  const id = `feat:${column}`;
  if (!overlay) {
    destroySubchart(id);
    return;
  }
  let pane = S.subcharts.get(id);
  const caption = featurePaneCaption(column, overlay);
  if (!pane) {
    const meta = Core.lookupFeatureMeta(column);
    const host = ensureSubchartHost(
      id,
      caption,
      meta.account_layer || meta.strategy
    );
    host.title = column;
    const inner = document.createElement("div");
    inner.className = "subchart-pane-inner";
    host.appendChild(inner);
    const c = LightweightCharts.createChart(inner, subchartBaseOptions());
    const color = Core.subchartColor(colorIndex);
    const series = c.addLineSeries({ color, lineWidth: 2 });
    pane = {
      chart: c,
      series,
      refSeriesList: [],
      host: inner,
      label: column,
      kind: "feature",
    };
    S.subcharts.set(id, pane);
    bindTimeScaleSync();
  } else {
    const capEl = pane.host?.parentElement?.querySelector(".subchart-label");
    if (capEl) capEl.textContent = caption;
  }
  const pts = Core.alignSeriesToCandleTimes(
    Core.clipOverlayPointsToCandles(overlay.points || [], candles),
    candles
  );
  pane.series.setData(pts);
  syncFeatureRefLines(pane, overlay, pts, candles);
  scheduleSubchartLayout();
}

function syncSubcharts(candles, overlays) {
  const showVol = document.getElementById("paneVolume").checked;
  ensureVolumePane(showVol, candles);
  const colsForPanesEarly = Core.resolveSubchartColumns(
    S.selectedFeatureColumns,
    S.availableFeatureColumns,
    layersState(),
    S.featureStrategyFocus,
    S.MAX_FEATURE_SUBCHARTS
  );
  const tableFirst = Core.chopMetricsTableActive(
    S.featureStrategyFocus,
    colsForPanesEarly
  );
  if (tableFirst) {
    document
      .querySelectorAll(
        ".subchart-strategy-header, .subchart-stage-header, .subchart-strategy-gap, .subchart-threshold-status, .subchart-pane"
      )
      .forEach((el) => el.remove());
    for (const id of [...S.subcharts.keys()]) {
      if (id.startsWith("feat:")) destroySubchart(id);
    }
  }
  const wantFeatures = new Set(S.selectedFeatureColumns);
  for (const id of [...S.subcharts.keys()]) {
    if (id.startsWith("metrics-") && !tableFirst) destroySubchart(id);
    if (id.startsWith("feat:") && (tableFirst || !wantFeatures.has(id.slice(5)))) {
      destroySubchart(id);
    }
  }
  clearStrategyChrome({ keepMetricsTable: tableFirst });

  const layers = layersState();
  const colsForPanes = colsForPanesEarly;
  const panePlan = Core.orderFeaturePaneItems(colsForPanes, layers, S.featureStrategyFocus);
  const domOrder = [];
  if (showVol) domOrder.push(subchartDomId("volume"));

  let colorIdx = 0;
  let metricsTableDone = false;
  for (const item of panePlan) {
    if (item.type === "gap") {
      ensureStrategyGap(item.id);
      domOrder.push(subchartDomId(item.id));
    } else if (item.type === "header") {
      if (tableFirst) continue;
      ensureSubchartHeader(item);
      domOrder.push(subchartDomId(headerDomKey(item)));
    } else if (item.type === "metrics_table") {
      if (!metricsTableDone) {
        domOrder.push(
          ensureFeatureMetricsTablePane(
            item,
            candles,
            overlays,
            S.highlightBarTime ?? candles[candles.length - 1]?.time
          )
        );
        metricsTableDone = true;
      }
    } else if (item.type === "threshold_status") {
      if (!tableFirst) {
        domOrder.push(ensureThresholdStatusPane(item, overlays, null));
      }
    } else if (item.type === "feature") {
      if (tableFirst) continue;
      const fid = `feat:${item.column}`;
      const overlaySpec =
        overlays?.[item.column] ||
        S.lastOverlays?.[item.column] ||
        {
          available: false,
          column: item.column,
          points: [],
          reference_lines: [],
          reference_y: null,
        };
      ensureFeaturePane(item.column, overlaySpec, colorIdx, candles);
      colorIdx += 1;
      domOrder.push(subchartDomId(fid));
    }
  }
  reorderSubchartStackDom(domOrder);

  if (!tableFirst) {
    const stack = document.getElementById("subchartStack");
    if (stack && S.lastCandles?.length) {
      const tail = S.lastCandles[S.lastCandles.length - 1].time;
      refreshThresholdTablesAtTime(
        S.highlightBarTime != null ? S.highlightBarTime : tail
      );
    }
  }

  if (tableFirst) {
    resizeAllSubcharts();
  } else {
    scheduleSubchartLayout();
  }
}

function formatOverlayStatus(overlays) {
  if (!S.selectedFeatureColumns.length) return " · 特征:未选";
  const parts = S.selectedFeatureColumns.map((col) => {
    const o = overlays?.[col];
    if (!o) return `${col}:?`;
    if (!o.available) return `${col}:无数据`;
    const latest =
      o.latest != null && o.latest === o.latest ? Number(o.latest).toFixed(3) : "?";
    const hint = o.semantic_hint ? ` ${o.semantic_hint}` : "";
    const lag =
      o.feature_range_end && o.aligned
        ? ` · bus至${String(o.feature_range_end).slice(0, 10)}`
        : "";
    const aligned = o.aligned ? "" : " (未对齐K线)";
    return `${col}=${latest}${hint}${lag}${aligned}`;
  });
  return ` · 特征:${parts.join("; ")}`;
}
