/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function applyMarkers(rawMarkers, opts = {}) {
  const aligned = alignMarkersToLoadedCandles(rawMarkers || []);
  S.lastRawMarkers = opts.merge ? mergeMarkersById(S.lastRawMarkers, aligned) : aligned;
  S.markerById = new Map(S.lastRawMarkers.map((m) => [m.id, m]));
  S.candleSeries.setMarkers(Core.markersToLwc(S.lastRawMarkers, S.selectedMarkerId));
}

function nearestLoadedCandleTime(rawTime) {
  const t = Number(rawTime);
  if (!Number.isFinite(t) || !S.lastCandles.length) return t;
  let best = Number(S.lastCandles[0].time);
  let bestDist = Math.abs(best - t);
  for (const c of S.lastCandles) {
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
  if (!S.lastCandles.length) return markers || [];
  const times = S.lastCandles.map((c) => Number(c.time)).filter(Number.isFinite);
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
  if (!S.chart) return;
  for (const s of S.tradeLinkSeries) {
    try {
      S.chart.removeSeries(s);
    } catch (_) {
      /* already removed */
    }
  }
  S.tradeLinkSeries = [];
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
  if (!S.chart || !Array.isArray(links) || !links.length) return;
  const clipped = S.lastCandles.length
    ? links.map((lk) => clipLinkToCandles(lk, S.lastCandles))
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
    const series = S.chart.addLineSeries(
      mainChartOverlaySeriesOptions({
        color,
        lineWidth: 1,
        lineStyle: open ? 2 : 0,
        lastValueVisible: false,
      })
    );
    series.setData([
      { time: t0, value: p0 },
      { time: t1, value: p1 },
    ]);
    S.tradeLinkSeries.push(series);
  }
  refreshMainPriceAutoscale();
}

function scrollChartToMarker(markerTime) {
  if (!S.chart || !S.lastCandles.length) return;
  const idx = Core.scrollIndexForTime(S.lastCandles, markerTime);
  if (idx < 0) return;
  const pad = 15;
  const from = Math.max(0, idx - pad);
  const to = Math.min(S.lastCandles.length - 1, idx + pad);
  S.chart.timeScale().setVisibleLogicalRange({ from, to });
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
  S.selectedMarkerId = markerId || null;
  applyMarkers(S.lastRawMarkers);
  highlightOrdersTableRow(S.selectedMarkerId);
  if (S.selectedMarkerId && scrollChart) {
    const raw = S.markerById.get(S.selectedMarkerId);
    if (raw?.time != null) scrollChartToMarker(raw.time);
  }
  if (S.selectedMarkerId && showDetail) {
    showMarkerDetail(S.selectedMarkerId);
  }
}

function applyOrdersDockVisibility() {
  const dock = document.getElementById("ordersDock");
  const btn = document.getElementById("ordersDockToggle");
  if (!dock || !btn) return;
  dock.classList.toggle("hidden", !S.ordersDockOpen);
  btn.classList.toggle("active", S.ordersDockOpen);
  btn.setAttribute("aria-pressed", S.ordersDockOpen ? "true" : "false");
}

function toggleOrdersDock(forceOpen) {
  S.ordersDockOpen = forceOpen ?? !S.ordersDockOpen;
  applyOrdersDockVisibility();
  saveLayout();
  if (S.ordersDockOpen) {
    refreshOrdersList().catch((e) => setStatus(String(e)));
  }
  requestAnimationFrame(() => {
    const el = document.getElementById("chart");
    if (S.chart && el) {
      S.chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
      resizeAllSubcharts();
    }
  });
}

async function refreshOrdersList() {
  if (!S.ordersDockOpen) return;
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
          S.selectedMarkerId = null;
          highlightOrdersTableRow(null);
          applyMarkers(S.lastRawMarkers);
        }
      });
    });
    highlightOrdersTableRow(S.selectedMarkerId);
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


