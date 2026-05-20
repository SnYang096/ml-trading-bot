/**
 * Trade Map Live — Lightweight Charts + REST bundle polling (P1–P3).
 */

const Core = globalThis.MLBotTradeMapCore;
const POLL_MS = 10000;

let chart;
let candleSeries;
let volumeChart;
let volumeSeries;
let emaChart;
let emaSeries;
let emaRefSeries;
let pollTimer;
let markerById = new Map();

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

async function api(path) {
  const r = await fetch(path);
  const j = await r.json();
  if (!j.ok) throw new Error(j.error?.message || r.statusText || "API error");
  return j;
}

function initCharts() {
  const el = document.getElementById("chart");
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#0f1419" }, textColor: "#8b949e" },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "#30363d" },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderVisible: false,
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
  });
  candleSeries.setMarkers([]);

  const volEl = document.getElementById("volumeChart");
  volumeChart = LightweightCharts.createChart(volEl, {
    layout: { background: { color: "#0f1419" }, textColor: "#8b949e" },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    timeScale: { visible: false },
  });
  volumeSeries = volumeChart.addHistogramSeries({ color: "#546e7a" });

  const emaEl = document.getElementById("emaChart");
  emaChart = LightweightCharts.createChart(emaEl, {
    layout: { background: { color: "#0f1419" }, textColor: "#8b949e" },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    timeScale: { visible: false },
  });
  emaSeries = emaChart.addLineSeries({ color: "#ffeb3b", lineWidth: 2 });
  emaRefSeries = emaChart.addLineSeries({
    color: "#8b949e",
    lineWidth: 1,
    lineStyle: 2,
  });

  const resize = () => {
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    if (!document.getElementById("volumeChart").classList.contains("hidden")) {
      volumeChart.applyOptions({
        width: volEl.clientWidth,
        height: volEl.clientHeight,
      });
    }
    if (!document.getElementById("emaChart").classList.contains("hidden")) {
      emaChart.applyOptions({ width: emaEl.clientWidth, height: emaEl.clientHeight });
    }
  };
  window.addEventListener("resize", resize);
  resize();
}

function applyMarkers(lwcMarkers) {
  markerById = new Map(lwcMarkers.map((m) => [m.id, m._raw]));
  candleSeries.setMarkers(lwcMarkers);
}

function applyVolume(candles) {
  const data = (candles || [])
    .filter((c) => c.volume != null)
    .map((c) => ({ time: c.time, value: c.volume, color: "#546e7a" }));
  volumeSeries.setData(data);
}

function applyWeeklyEma(overlay) {
  const pts = overlay?.points || [];
  emaSeries.setData(pts.map((p) => ({ time: p.time, value: p.value })));
  if (pts.length) {
    const ref = pts.map((p) => ({ time: p.time, value: overlay.reference_y ?? 0 }));
    emaRefSeries.setData(ref);
  } else {
    emaRefSeries.setData([]);
  }
}

function togglePane(id, show) {
  document.getElementById(id).classList.toggle("hidden", !show);
}

async function loadLinks() {
  try {
    const { data } = await api("/api/links");
    const nav = document.getElementById("extLinks");
    nav.innerHTML = "";
    for (const link of data.links || []) {
      const a = document.createElement("a");
      a.href = link.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = link.label;
      nav.appendChild(a);
    }
  } catch (_) {
    /* optional */
  }
}

async function loadSymbols() {
  const { data } = await api("/api/trade-map/symbols");
  const sel = document.getElementById("symbolSelect");
  sel.innerHTML = "";
  const list = data.length ? data : [{ symbol: "ETHUSDT" }];
  for (const row of list) {
    const sym = row.symbol || row;
    const opt = document.createElement("option");
    opt.value = sym;
    opt.textContent = sym;
    sel.appendChild(opt);
  }
  if (!sel.value && list[0]) sel.value = list[0].symbol || "ETHUSDT";
}

async function loadEligibility() {
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  try {
    const { data } = await api(
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
    const { data } = await api(
      `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(markerId)}`
    );
    body.textContent = JSON.stringify({ marker: raw, db: data }, null, 2);
  } catch (e) {
    body.textContent += `\n\n(DB lookup failed: ${e})`;
  }
}

async function refreshBundle() {
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  const layers = layersState();
  const scopes = Core.scopesFromLayers(layers);
  const pending = layers.pending;
  const overlayEma = document.getElementById("overlayWeeklyEma").checked;
  setStatus("加载中…");
  const q = new URLSearchParams({
    symbol,
    timeframe,
    scopes,
    include_pending: String(pending),
    overlay_weekly_ema: String(overlayEma),
  });
  const pageUrl = new URL(window.location.href);
  if (pageUrl.searchParams.get("from")) {
    q.set("from", pageUrl.searchParams.get("from"));
  }
  if (pageUrl.searchParams.get("to")) {
    q.set("to", pageUrl.searchParams.get("to"));
  }
  const { data, meta } = await api(`/api/trade-map/bundle?${q}`);
  const candles = data.ohlcv?.candles || [];
  candleSeries.setData(candles);
  const lwc = Core.markersToLwc(data.markers || []);
  applyMarkers(lwc);
  chart.timeScale().fitContent();

  togglePane("volumeChart", document.getElementById("overlayVolume").checked);
  if (document.getElementById("overlayVolume").checked) {
    applyVolume(candles);
    volumeChart.timeScale().fitContent();
  }

  togglePane("emaChart", overlayEma && data.overlays?.weekly_ema_200_position?.available);
  if (overlayEma) {
    applyWeeklyEma(data.overlays?.weekly_ema_200_position);
    emaChart.timeScale().fitContent();
  }

  const deg = meta.degraded_ohlc || data.ohlcv?.degraded_ohlc;
  setStatus(
    `${symbol} ${timeframe} · ${candles.length} bars · ${(data.markers || []).length} markers` +
      (deg ? " · OHLC degraded" : "") +
      ` · ${new Date().toLocaleTimeString()}`
  );
  await loadEligibility();
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => refreshBundle().catch((e) => setStatus(String(e))), POLL_MS);
}

function bindControls() {
  const rerun = () => refreshBundle().catch((e) => setStatus(String(e)));
  document.getElementById("refreshBtn").addEventListener("click", rerun);
  [
    "symbolSelect",
    "timeframeSelect",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "layerPending",
    "overlayVolume",
    "overlayWeeklyEma",
  ].forEach((id) => document.getElementById(id).addEventListener("change", rerun));

  chart.subscribeClick((param) => {
    if (!param || param.time === undefined) return;
    const markers = candleSeries.markers?.() || [];
    const hit = markers.find((m) => m.time === param.time);
    if (hit?.id) showMarkerDetail(hit.id);
  });
}

(async () => {
  try {
    initCharts();
    bindControls();
    await loadLinks();
    await loadSymbols();
    await refreshBundle();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
