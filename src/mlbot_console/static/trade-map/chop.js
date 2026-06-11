/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function clearChopGridOverlay() {
  for (const pl of S.chopGridPriceLines) {
    try {
      S.candleSeries.removePriceLine(pl);
    } catch (_) {
      /* already removed */
    }
  }
  S.chopGridPriceLines = [];
  for (const s of S.chopSegmentSeries) {
    try {
      S.chart.removeSeries(s);
    } catch (_) {
      /* */
    }
  }
  S.chopSegmentSeries = [];
  S.chopGridLabelSpecs = [];
  const labelLayer = document.getElementById("chopGridLabelLayer");
  if (labelLayer) labelLayer.innerHTML = "";
  if (S.chopBandAreaSeries) {
    try {
      S.chopBandAreaSeries.setData([]);
    } catch (_) {
      /* */
    }
  }
  for (const s of [S.prefilterBandAreaSeries, S.gateBandAreaSeries]) {
    if (s) {
      try {
        s.setData([]);
      } catch (_) {
        /* */
      }
    }
  }
}

function filterStageRegionsForFocus(byStrategy) {
  if (!byStrategy || typeof byStrategy !== "object" || byStrategy.error) {
    return byStrategy;
  }
  const focus = String(S.featureStrategyFocus || "").trim().toLowerCase();
  if (!focus) return byStrategy;
  const out = {};
  for (const [k, v] of Object.entries(byStrategy)) {
    if (String(k).toLowerCase() === focus) out[k] = v;
  }
  return Object.keys(out).length ? out : byStrategy;
}

function chopMapDataForStrategyFocus(data) {
  const raw = data || S.lastChopMapData || {};
  const stages = filterStageRegionsForFocus(raw.strategy_stage_regions);
  if (chopGridOverlayEnabled()) {
    return { ...raw, strategy_stage_regions: stages };
  }
  return {
    chop_grid_overlay: { batches: [] },
    chop_regime_regions: [],
    strategy_stage_regions: stages,
  };
}

/** Re-apply main-chart layers after strategy/layer switch (before bundle returns). */
function refreshMainChartForStrategyFocus() {
  if (!S.chart || !S.lastCandles?.length) return;
  applyChopMapLayers(chopMapDataForStrategyFocus(S.lastChopMapData), S.lastCandles);
  if (typeof applyMarkers === "function") {
    applyMarkers(S.allRawMarkers || []);
  }
  if (typeof applyTradeLinks === "function") {
    applyTradeLinks(S.lastTradeLinks || []);
  }
}

function flattenStageRegions(byStrategy, stage) {
  const spans = [];
  if (!byStrategy || typeof byStrategy !== "object") return spans;
  for (const strat of Object.keys(byStrategy)) {
    const block = byStrategy[strat];
    if (!block || typeof block !== "object") continue;
    for (const r of block[stage] || []) {
      if (r && r.start != null && r.end != null) spans.push(r);
    }
  }
  return spans;
}

function ensureStageBandSeries(kind) {
  if (kind === "prefilter") {
    if (S.prefilterBandAreaSeries) return S.prefilterBandAreaSeries;
    S.prefilterBandAreaSeries = S.chart.addCandlestickSeries(
      bandHighlightSeriesOptions("rgba(239, 68, 68, 0.35)")
    );
    return S.prefilterBandAreaSeries;
  }
  if (S.gateBandAreaSeries) return S.gateBandAreaSeries;
  S.gateBandAreaSeries = S.chart.addCandlestickSeries(
    bandHighlightSeriesOptions("rgba(124, 58, 237, 0.35)")
  );
  return S.gateBandAreaSeries;
}

function applyStagePriceBand(stage, spans, candles) {
  if (!candles?.length || !spans?.length) return;
  const series =
    stage === "gate" ? ensureStageBandSeries("gate") : ensureStageBandSeries("prefilter");
  if (!series) return;
  series.setData(spanHighlightCandles(candles, spans));
}

function applyStrategyStageRegions(data, candles) {
  const by = data?.strategy_stage_regions;
  if (!by || by.error) {
    S.lastPrefilterSpans = [];
    return;
  }
  S.lastPrefilterSpans = flattenStageRegions(by, "prefilter");
  if (document.getElementById("layerPrefilter")?.checked) {
    applyStagePriceBand("prefilter", S.lastPrefilterSpans, candles);
  }
  if (document.getElementById("layerGate")?.checked) {
    applyStagePriceBand("gate", flattenStageRegions(by, "gate"), candles);
  }
}

function chopGridBandExtents(batch) {
  const center = Number(batch.center);
  const spacing = Number(batch.spacing) || 0;
  if (center <= 0 || spacing <= 0) return null;
  let maxLv = 2;
  for (const lv of batch.levels || []) {
    const m = String(lv.leg || "").match(/(\d+)/);
    if (m) maxLv = Math.max(maxLv, parseInt(m[1], 10));
  }
  return {
    top: center + spacing * maxLv,
    bottom: center - spacing * maxLv,
  };
}

/** Regime chop tint envelope from OHLC inside chop_regime_regions (not grid spacing height). */
function chopRegimePriceEnvelope(candles, spans) {
  let top = -Infinity;
  let bottom = Infinity;
  for (const c of candles) {
    const t = Number(c.time);
    if (!spans.some((r) => t >= Number(r.start) && t <= Number(r.end))) continue;
    const lo = Number(c.low);
    const hi = Number(c.high);
    if (Number.isFinite(lo)) bottom = Math.min(bottom, lo);
    if (Number.isFinite(hi)) top = Math.max(top, hi);
  }
  if (!Number.isFinite(top) || !Number.isFinite(bottom) || top <= bottom) return null;
  const span = Math.max(top - bottom, top * 0.0005);
  const pad = Math.max(span * 0.04, top * 0.002);
  return { top: top + pad, bottom: bottom - pad };
}

function chopGridOverlayEnabled() {
  if (!document.getElementById("layerChopGrid")?.checked) return false;
  if (!layersState().multiLeg) return false;
  const focus = String(S.featureStrategyFocus || "").trim().toLowerCase();
  if (focus && focus !== "chop_grid") return false;
  return true;
}

function fullWidthPriceLine(candles, price) {
  if (!candles?.length || price == null || !Number.isFinite(Number(price))) {
    return [];
  }
  const px = Number(price);
  const first = candles[0];
  const last = candles[candles.length - 1];
  return [
    { time: first.time, value: px },
    { time: last.time, value: px },
  ];
}

/** Grid/chop lines inside prefilter (or chop regime) time spans only.
 * Merges overlapping spans, clips to candle range, and inserts NaN breakpoints
 * (matching `Core.chopSegmentedLinePoints`) so the series doesn't visually
 * connect across gaps. */
function priceLineInSpans(candles, spans, price) {
  const px = Number(price);
  if (!candles?.length || !spans?.length || !Number.isFinite(px)) return [];
  const firstCandle = Number(candles[0].time);
  const lastCandle = Number(candles[candles.length - 1].time);
  const sortedRaw = [...spans]
    .map((s) => ({ start: Number(s.start), end: Number(s.end) }))
    .filter(
      (s) =>
        Number.isFinite(s.start) &&
        Number.isFinite(s.end) &&
        s.end >= s.start &&
        s.end >= firstCandle &&
        s.start <= lastCandle
    )
    .map((s) => ({
      start: Math.max(s.start, firstCandle),
      end: Math.min(s.end, lastCandle),
    }))
    .sort((a, b) => a.start - b.start);
  if (!sortedRaw.length) return [];
  const merged = [sortedRaw[0]];
  for (let i = 1; i < sortedRaw.length; i++) {
    const cur = sortedRaw[i];
    const last = merged[merged.length - 1];
    if (cur.start <= last.end) {
      last.end = Math.max(last.end, cur.end);
    } else {
      merged.push({ ...cur });
    }
  }
  const pts = [];
  for (let i = 0; i < merged.length; i++) {
    const span = merged[i];
    if (span.start === span.end) {
      pts.push({ time: span.start, value: px });
    } else {
      pts.push({ time: span.start, value: px });
      pts.push({ time: span.end, value: px });
    }
    if (i < merged.length - 1) {
      const next = merged[i + 1];
      const gapT = span.end + 1;
      if (gapT < next.start) {
        pts.push({ time: gapT, value: NaN });
      }
    }
  }
  return pts;
}

function chopOverlaySpans(data) {
  // Grid + chop band share prefilter windows when available (actual trading spans).
  if (S.lastPrefilterSpans?.length) return S.lastPrefilterSpans;
  return data?.chop_regime_regions || [];
}

function chopGridLineSpans(candles, data) {
  return chopOverlaySpans(data);
}

function ensureChopGridLabelLayer() {
  const chartEl = document.getElementById("chart");
  if (!chartEl) return null;
  let layer = document.getElementById("chopGridLabelLayer");
  if (!layer) {
    layer = document.createElement("div");
    layer.id = "chopGridLabelLayer";
    layer.className = "chop-grid-label-layer";
    chartEl.appendChild(layer);
  }
  return layer;
}

function labelTimeForSpans(spans, candles) {
  const rows = candles?.length ? candles : S.lastCandles;
  if (!rows?.length) return null;
  if (!spans?.length) return Number(rows[rows.length - 1].time);
  let best = null;
  for (const span of spans) {
    const start = Number(span.start);
    const end = Number(span.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    for (const c of rows) {
      const t = Number(c.time);
      if (t >= start && t <= end && (best == null || t > best)) best = t;
    }
  }
  return best ?? Number(rows[rows.length - 1].time);
}

function layoutChopGridLabels(candles) {
  const layer = ensureChopGridLabelLayer();
  if (!layer || !S.chart || !S.candleSeries?.priceToCoordinate) return;
  layer.innerHTML = "";
  if (!S.chopGridLabelSpecs.length || !chopGridOverlayEnabled()) return;
  const ts = S.chart.timeScale();
  for (const spec of S.chopGridLabelSpecs) {
    const anchor = Core.chopGridLabelAnchor(spec.side, spec.kind);
    const labelTime = labelTimeForSpans(spec.spans, candles);
    if (labelTime == null || !Number.isFinite(labelTime)) continue;
    const x = ts.timeToCoordinate(labelTime);
    const y = S.candleSeries.priceToCoordinate(Number(spec.price));
    if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    const el = document.createElement("span");
    el.className = `chop-grid-label chop-grid-label--${anchor}`;
    el.style.left = `${Math.round(x)}px`;
    el.style.top = `${Math.round(y)}px`;
    if (spec.color) el.style.borderColor = spec.color;
    el.textContent = spec.text;
    layer.appendChild(el);
  }
}

function bindChopGridLabelSync() {
  if (S.chopGridLabelsBound || !S.chart) return;
  S.chopGridLabelsBound = true;
  S.chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    layoutChopGridLabels(S.lastCandles);
  });
}

function addChopFullWidthLine(candles, price, opts = {}) {
  if (!S.chart || price == null || !Number.isFinite(Number(price))) return;
  const spans = opts.spans;
  // If spans were explicitly provided but are empty, don't draw (avoid full-width fallback).
  if (spans !== undefined && spans !== null && !spans.length) return;
  const pts = spans?.length
    ? priceLineInSpans(candles, spans, price)
    : fullWidthPriceLine(candles, price);
  if (!pts.length) return;
  const series = S.chart.addLineSeries(
    mainChartOverlaySeriesOptions({
      color: opts.color || "#888",
      lineWidth: opts.lineWidth ?? 1,
      lineStyle: opts.lineStyle ?? 2,
      lastValueVisible: false,
    })
  );
  series.setData(pts);
  S.chopSegmentSeries.push(series);
  if (opts.title) {
    S.chopGridLabelSpecs.push({
      price: Number(price),
      text: opts.title,
      side: opts.labelSide ?? null,
      kind: opts.labelKind || "grid",
      color: opts.color,
      spans: spans ? spans.map((s) => ({ start: s.start, end: s.end })) : null,
    });
  }
}

function applyChopPriceBand(regions, candles, _overlay) {
  if (!chopGridOverlayEnabled() || !candles?.length) return;
  const spans = regions || [];
  if (!spans.length) return;
  ensureChopBandAreaSeries();
  if (!S.chopBandAreaSeries) return;
  S.chopBandAreaSeries.setData(spanHighlightCandles(candles, spans));
}

function applyChopGridOverlay(overlay, candles, lineSpans) {
  if (!chopGridOverlayEnabled() || !candles?.length) return;
  const spans = lineSpans?.length ? lineSpans : null;
  for (const batch of overlay?.batches || []) {
    const center = Number(batch.center);
    if (center > 0) {
      addChopFullWidthLine(candles, center, {
        color: "#94a3b8",
        lineWidth: 2,
        lineStyle: 2,
        title: "中心",
        labelSide: "long",
        labelKind: "center",
        spans,
      });
    }
    for (const lv of batch.levels || []) {
      const leg = String(lv.leg || "").toUpperCase();
      const isLong = lv.side === "long";
      const gridColor = isLong
        ? "rgba(59, 130, 246, 0.55)"
        : "rgba(249, 115, 22, 0.55)";
      const gridPx = Number(lv.grid_price);
      if (Number.isFinite(gridPx) && gridPx > 0) {
        addChopFullWidthLine(candles, gridPx, {
          color: gridColor,
          lineStyle: 2,
          title: `${leg} 格`,
          labelSide: isLong ? "long" : "short",
          labelKind: "grid",
          spans,
        });
      }
      const tpPx = lv.tp_price != null ? Number(lv.tp_price) : null;
      if (tpPx != null && tpPx > 0) {
        const tpSt = String(lv.tp_status || "").toLowerCase();
        const tpOpen = ["open", "pending", "new", "submitted", "shadow"].includes(
          tpSt
        );
        // Filled TP is historical; overlay grid already explains the ladder.
        if (!tpOpen) continue;
        addChopFullWidthLine(candles, tpPx, {
          color: tpOpen ? "#a855f7" : "#6b7280",
          lineStyle: 1,
          title: `${leg}_TP`,
          labelSide: isLong ? "long" : "short",
          labelKind: "tp",
          spans,
        });
      }
    }
  }
}

function applyChopMapLayers(data, candles) {
  clearChopGridOverlay();
  if (!data) return;
  const payload = chopMapDataForStrategyFocus(data);
  applyStrategyStageRegions(payload, candles);
  const gridSpans = chopGridLineSpans(candles, payload);
  if (chopGridOverlayEnabled()) {
    applyChopPriceBand(gridSpans, candles, payload.chop_grid_overlay || {});
    applyChopGridOverlay(payload.chop_grid_overlay || {}, candles, gridSpans);
    bindChopGridLabelSync();
    layoutChopGridLabels(candles);
  }
  refreshMainPriceAutoscale();
}


