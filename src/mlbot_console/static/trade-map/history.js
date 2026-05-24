/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function resetOhlcvLoadedRange() {
  S.ohlcvLoadedFrom = null;
  S.ohlcvLoadedTo = null;
  S.markerQueryFromIso = null;
  S.lastMarkerPollSince = null;
  S.lastTradeLinks = [];
  S.lastMarkerCounts = null;
  S.historyExhausted = false;
}

function resetMarkerQueryRange() {
  S.markerQueryFromIso = initialOhlcvRangeIso().from;
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
    S.lastMarkerPollSince = serverTimestamp;
  } else {
    // Fallback: Using client time minus 2 seconds to account for clock skew/latency.
    S.lastMarkerPollSince = new Date(Date.now() - 2000).toISOString();
  }
}

function markerRangeParams() {
  const to = new Date().toISOString();
  const from = S.markerQueryFromIso || initialOhlcvRangeIso().from;
  return { from, to, full_range: "false" };
}

function applyLoadedOhlcvRange(meta, candles) {
  if (candles?.length) {
    S.ohlcvLoadedFrom = Core.isoFromUnixSec(candles[0].time);
    S.ohlcvLoadedTo = Core.isoFromUnixSec(candles[candles.length - 1].time);
  } else {
    if (meta?.range_start) S.ohlcvLoadedFrom = String(meta.range_start);
    if (meta?.range_end) S.ohlcvLoadedTo = String(meta.range_end);
  }
}

function scheduleHistoryPrefetch(range) {
  if (!range || S.historyLoadInFlight || S.historyExhausted || !S.lastCandles.length) return;
  if (range.from > 25) return;
  if (S.historyLoadTimer) clearTimeout(S.historyLoadTimer);
  S.historyLoadTimer = setTimeout(() => {
    loadMoreHistory().catch((e) => setStatus(String(e)));
  }, 350);
}

async function loadMoreHistory() {
  if (S.historyLoadInFlight || S.historyExhausted || !S.lastCandles.length) return;
  const timeframe = document.getElementById("timeframeSelect").value;
  const symbol = document.getElementById("symbolSelect").value;
  const oldest = S.lastCandles[0].time;
  const chunkDays = Core.tradeMapHistoryChunkDays(timeframe);
  const newFromMs =
    Number(oldest) * 1000 - chunkDays * 86400000;
  const newFromIso = new Date(newFromMs).toISOString();
  S.historyLoadInFlight = true;
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
      S.historyExhausted = true;
      return;
    }
    const merged = Core.mergeCandlesByTime(more, S.lastCandles);
    if (merged.length === S.lastCandles.length) {
      S.historyExhausted = true;
      return;
    }
    S.lastCandles = merged;
    S.candleSeries.setData(merged);
    applyLoadedOhlcvRange(meta, merged);
    if (S.lastChopMapData) applyChopMapLayers(S.lastChopMapData, merged);
    syncSubcharts(merged, S.lastOverlays);
    if (
      S.markerQueryFromIso == null ||
      new Date(newFromIso).getTime() < new Date(S.markerQueryFromIso).getTime()
    ) {
      S.markerQueryFromIso = newFromIso;
    }
    await refreshMarkersOnly();
    applyTradeLinks(data.trade_links || []);
  } finally {
    S.historyLoadInFlight = false;
  }
}

function bindTimeScaleSync() {
  if (S.timeSyncBound) return;
  S.timeSyncBound = true;
  S.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range) return;
    const timeRange = mainVisibleTimeRange();
    for (const pane of S.subcharts.values()) {
      if (timeRange) {
        try {
          pane.chart.timeScale().setVisibleRange(timeRange);
          continue;
        } catch (_) {
          /* fallback */
        }
      }
      pane.chart.timeScale().setVisibleLogicalRange(range);
    }
    refreshMainPriceAutoscale();
    if (typeof layoutChopGridLabels === "function") {
      layoutChopGridLabels(S.lastCandles);
    }
    scheduleHistoryPrefetch(range);
  });
}


