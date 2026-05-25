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

function featurePaneCaption(column, overlay) {
  const meta = Core.lookupFeatureMeta(column);
  const base =
    meta.strategy_title && meta.stage_title
      ? `${meta.strategy_title}·${meta.stage_title}`
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
    const parts = [base, valStr];
    if (hint) parts.push(`(${hint})`);
    else if (refHint) parts.push(`(${refHint})`);
    return parts.join(" ");
  }
  if (refHint) return `${base} · ${refHint}`;
  if (overlay?.available === false) return `${base} · 无数据`;
  return base;
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
  const wantFeatures = new Set(S.selectedFeatureColumns);
  for (const id of [...S.subcharts.keys()]) {
    if (id.startsWith("feat:") && !wantFeatures.has(id.slice(5))) destroySubchart(id);
  }
  clearStrategyChrome();

  const layers = layersState();
  const colsForPanes = Core.resolveSubchartColumns(
    S.selectedFeatureColumns,
    S.availableFeatureColumns,
    layers,
    S.featureStrategyFocus,
    S.MAX_FEATURE_SUBCHARTS
  );
  const panePlan = Core.orderFeaturePaneItems(colsForPanes, layers, S.featureStrategyFocus);
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

  scheduleSubchartLayout();
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
