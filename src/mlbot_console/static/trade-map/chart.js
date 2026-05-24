/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function chartBaseOptions() {
  return {
    layout: {
      background: { color: "#0f1419" },
      textColor: "#8b949e",
      attributionLogo: false,
    },
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

/** Feature/volume panes: no grid lines so stacked S.subcharts read as one block. */
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
  S.chart.timeScale().applyOptions({
    barSpacing: spacing,
    minBarSpacing: 0.5,
    rightOffset: 8,
  });
  const range = Core.visibleLogicalRange(barCount);
  if (range) {
    S.chart.timeScale().setVisibleLogicalRange(range);
    syncSubchartsToMainRange();
    refreshMainPriceAutoscale();
  }
}

function mainVisibleTimeRange() {
  if (!S.chart || !S.lastCandles?.length) return null;
  const logical = S.chart.timeScale().getVisibleLogicalRange();
  if (!logical) return null;
  const fromIdx = Math.max(
    0,
    Math.min(S.lastCandles.length - 1, Math.floor(Number(logical.from)))
  );
  const toIdx = Math.max(
    0,
    Math.min(S.lastCandles.length - 1, Math.ceil(Number(logical.to)))
  );
  const fromTime = S.lastCandles[fromIdx]?.time;
  const toTime = S.lastCandles[toIdx]?.time;
  if (fromTime == null || toTime == null) return null;
  return { from: fromTime, to: toTime };
}

function syncSubchartScales() {
  if (!S.chart) return;
  const mainTs = S.chart.options()?.timeScale || {};
  const barSpacing = mainTs.barSpacing ?? 3;
  const rightOffset = mainTs.rightOffset ?? 8;
  for (const pane of S.subcharts.values()) {
    pane.chart.timeScale().applyOptions({
      barSpacing,
      rightOffset,
      minBarSpacing: 0.5,
    });
  }
}

function syncSubchartsToMainRange() {
  if (!S.chart) return;
  syncSubchartScales();
  const timeRange = mainVisibleTimeRange();
  if (timeRange) {
    for (const pane of S.subcharts.values()) {
      try {
        pane.chart.timeScale().setVisibleRange(timeRange);
      } catch (_) {
        const range = S.chart.timeScale().getVisibleLogicalRange();
        if (range) pane.chart.timeScale().setVisibleLogicalRange(range);
      }
    }
    return;
  }
  const range = S.chart.timeScale().getVisibleLogicalRange();
  if (!range) return;
  for (const pane of S.subcharts.values()) {
    pane.chart.timeScale().setVisibleLogicalRange(range);
  }
}

function ensureChopBandAreaSeries() {
  if (S.chopBandAreaSeries || !S.chart) return;
  S.chopBandAreaSeries = S.chart.addCandlestickSeries(
    bandHighlightSeriesOptions(S.CHOP_REGIME_FILL)
  );
  S.chopBandAreaSeries.setData([]);
}

function initMainChart() {
  const el = document.getElementById("chart");
  S.chart = LightweightCharts.createChart(el, chartBaseOptions());
  ensureChopBandAreaSeries();
  S.candleSeries = S.chart.addCandlestickSeries({
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderVisible: false,
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
    autoscaleInfoProvider: () => {
      const range = S.chart.timeScale().getVisibleLogicalRange();
      const custom = Core.priceRangeForChartAutoscale(S.lastCandles, range);
      if (!custom) return null;
      return {
        priceRange: custom,
        margins: { above: 10, below: 10 },
      };
    },
  });
  S.candleSeries.setMarkers([]);

  const resize = () => {
    S.chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    for (const pane of S.subcharts.values()) {
      pane.chart.applyOptions({
        width: pane.host.clientWidth,
        height: pane.host.clientHeight,
      });
    }
    if (typeof layoutChopGridLabels === "function") {
      layoutChopGridLabels(S.lastCandles);
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
  S.chart.subscribeClick((param) => {
    if (!param || param.time === undefined) return;
    const tf = document.getElementById("timeframeSelect")?.value || "2h";
    const tol = Core.timeframeToleranceSec(tf);
    const hit = Core.findMarkerByTime(S.lastRawMarkers, param.time, tol);
    if (hit?.id) selectMarker(hit.id);
  });

  const legend = document.getElementById("chartLegend");
  S.chart.subscribeCrosshairMove((param) => {
    if (!param || !param.time || param.point.x < 0 || param.point.y < 0) {
      legend.classList.add("hidden");
      return;
    }
    const data = param.seriesData.get(S.candleSeries);
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
    for (const [key, series] of S.mainOverlaySeries.entries()) {
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
  for (const pane of S.subcharts.values()) {
    if (!pane.host) continue;
    const w = pane.host.clientWidth;
    const h = pane.host.clientHeight;
    if (w > 0 && h > 0) {
      pane.chart.applyOptions({ width: w, height: h });
    }
  }
}

function clearMainOverlaySeries() {
  for (const [, series] of S.mainOverlaySeries) {
    S.chart.removeSeries(series);
  }
  S.mainOverlaySeries.clear();
  S.mainOverlayData.clear();
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
    let line = S.mainOverlaySeries.get(key);
    if (!line) {
      line = S.chart.addLineSeries(
        mainChartOverlaySeriesOptions({
          color: spec.color || "#888",
          lineWidth: 2,
          lastValueVisible: true,
          title: spec.label || key,
        })
      );
      S.mainOverlaySeries.set(key, line);
    } else {
      line.applyOptions(mainChartOverlaySeriesOptions());
    }
    const next = spec.points.map((p) => ({
      time: p.time,
      value: p.value,
    }));
    const points = merge ? mergeOverlayPoints(S.mainOverlayData.get(key) || [], next) : next;
    S.mainOverlayData.set(key, points);
    line.setData(points);
  }
  refreshMainPriceAutoscale();
}


