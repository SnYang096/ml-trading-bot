/**
 * Trade Map pure helpers — 10-ohlcv. Attaches to MLBotTradeMapCore (load in numeric order).
 */
(function (root) {
  const Core = (root.MLBotTradeMapCore = root.MLBotTradeMapCore || {});
    function scopesFromLayers(layers) {
      const parts = [];
      if (layers.trend) parts.push("trend");
      if (layers.spot) parts.push("spot");
      if (layers.multiLeg) parts.push("multi_leg");
      return parts.join(",") || "trend,spot";
    }
    function timeframeToleranceSec(timeframe) {
      const tf = String(timeframe || "2h").toLowerCase();
      if (tf === "1min") return 90;
      if (tf === "15min") return 900;
      if (tf === "1d") return 86400;
      if (tf === "1w") return 604800;
      return 7200;
    }

    /**
     * Initial bundle OHLCV query for a timeframe.
     * 1d/1w: Vision macro full history (no from/to; backend full_range).
     */
    function ohlcvInitialQueryRange(timeframe) {
      const tf = String(timeframe || "2h");
      if (tf === "1d" || tf === "1w") {
        return { full_range: "true" };
      }
      const days = tradeMapInitialDays(tf);
      const end = new Date();
      const start = new Date(end.getTime() - days * 86400000);
      return {
        from: start.toISOString(),
        to: end.toISOString(),
        full_range: "true",
      };
    }

    /** Default OHLCV window (days) — keep in sync with TRADE_MAP_INITIAL_DAYS. */
    function tradeMapInitialDays(timeframe) {
      const tf = String(timeframe || "2h");
      const map = {
        "15min": 14,
        "2h": 60,
        "120T": 60,
        "1d": 120,
        "1w": 365,
      };
      return map[tf] ?? 60;
    }

    /** One pan-left prefetch chunk (days). */
    function tradeMapHistoryChunkDays(timeframe) {
      const tf = String(timeframe || "2h");
      const map = {
        "15min": 7,
        "2h": 30,
        "120T": 30,
        "1d": 90,
        "1w": 180,
      };
      return map[tf] ?? 30;
    }

    function barDurationSec(timeframe) {
      const tf = String(timeframe || "2h").toLowerCase();
      if (tf === "15min") return 900;
      if (tf === "1d") return 86400;
      if (tf === "1w") return 604800;
      return 7200;
    }

    function mergeCandlesByTime(existing, incoming) {
      const byTime = new Map();
      for (const c of existing || []) {
        if (c && c.time != null) byTime.set(Number(c.time), c);
      }
      for (const c of incoming || []) {
        if (c && c.time != null) byTime.set(Number(c.time), c);
      }
      return [...byTime.values()].sort((a, b) => a.time - b.time);
    }

    /** Drop feature points outside the loaded OHLCV window (poll/history merge safety). */
    function clipOverlayPointsToCandles(points, candles) {
      if (!points?.length || !candles?.length) return points || [];
      const tMin = Number(candles[0].time);
      const tMax = Number(candles[candles.length - 1].time);
      if (!Number.isFinite(tMin) || !Number.isFinite(tMax)) return points;
      return points.filter((p) => {
        const t = Number(p?.time);
        return Number.isFinite(t) && t >= tMin && t <= tMax;
      });
    }

    /** Main-chart MA overlays: one point per candle, backward as-of + forward-fill. */
    function forwardFillOverlayToCandles(points, candles) {
      if (!candles?.length) return [];
      const sorted = [...(points || [])]
        .filter(
          (p) =>
            p &&
            Number.isFinite(Number(p.time)) &&
            Number.isFinite(Number(p.value))
        )
        .sort((a, b) => Number(a.time) - Number(b.time));
      if (!sorted.length) return [];
      let j = 0;
      let last = null;
      const out = [];
      for (const c of candles) {
        const t = Number(c?.time);
        if (!Number.isFinite(t)) continue;
        while (j + 1 < sorted.length && Number(sorted[j + 1].time) <= t) {
          j += 1;
        }
        if (Number(sorted[j].time) <= t) {
          last = Number(sorted[j].value);
        }
        if (last != null && Number.isFinite(last)) {
          out.push({ time: t, value: last });
        }
      }
      return out;
    }

    /** One timeline entry per OHLCV bar (whitespace where feature is missing). */
    function alignSeriesToCandleTimes(points, candles) {
      if (!candles?.length) return points || [];
      const byTime = new Map();
      for (const p of points || []) {
        const t = Number(p?.time);
        if (!Number.isFinite(t)) continue;
        byTime.set(t, p?.value);
      }
      const out = [];
      for (const c of candles) {
        const t = Number(c?.time);
        if (!Number.isFinite(t)) continue;
        if (!byTime.has(t)) {
          out.push({ time: t });
          continue;
        }
        const v = byTime.get(t);
        if (v == null || (typeof v === "number" && v !== v)) {
          out.push({ time: t });
        } else {
          out.push({ time: t, value: Number(v) });
        }
      }
      return out;
    }
    function isoFromUnixSec(sec) {
      return new Date(Number(sec) * 1000).toISOString();
    }

  Object.assign(Core, {
    scopesFromLayers,
    timeframeToleranceSec,
    ohlcvInitialQueryRange,
    tradeMapInitialDays,
    tradeMapHistoryChunkDays,
    barDurationSec,
    mergeCandlesByTime,
    clipOverlayPointsToCandles,
    forwardFillOverlayToCandles,
    alignSeriesToCandleTimes,
    isoFromUnixSec,
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
