import type { Candle, LayerVisibility, OverlayPoint } from './types.ts';

export function scopesFromLayers(layers: LayerVisibility | null | undefined): string {
  const parts: string[] = [];
  if (layers?.trend) parts.push('trend');
  if (layers?.spot) parts.push('spot');
  if (layers?.multiLeg) parts.push('multi_leg');
  return parts.join(',') || 'trend,spot';
}

export function timeframeToleranceSec(timeframe: string | null | undefined): number {
  const tf = String(timeframe || '2h').toLowerCase();
  if (tf === '1min') return 90;
  if (tf === '15min') return 900;
  if (tf === '1d') return 86400;
  if (tf === '1w') return 604800;
  return 7200;
}

export interface OhlcvInitialQueryRange {
  from?: string;
  to?: string;
  full_range: string;
}

/**
 * Initial bundle OHLCV query for a timeframe.
 * 1d/1w: Vision macro full history (no from/to; backend full_range).
 */
export function ohlcvInitialQueryRange(timeframe: string | null | undefined): OhlcvInitialQueryRange {
  const tf = String(timeframe || '2h');
  if (tf === '1d' || tf === '1w') {
    return { full_range: 'true' };
  }
  const days = tradeMapInitialDays(tf);
  const end = new Date();
  const start = new Date(end.getTime() - days * 86400000);
  return {
    from: start.toISOString(),
    to: end.toISOString(),
    full_range: 'true',
  };
}

/** Default OHLCV window (days) — keep in sync with TRADE_MAP_INITIAL_DAYS. */
export function tradeMapInitialDays(timeframe: string | null | undefined): number {
  const tf = String(timeframe || '2h');
  const map: Record<string, number> = {
    '15min': 14,
    '2h': 60,
    '120T': 60,
    '1d': 120,
    '1w': 365,
  };
  return map[tf] ?? 60;
}

/** One pan-left prefetch chunk (days). */
export function tradeMapHistoryChunkDays(timeframe: string | null | undefined): number {
  const tf = String(timeframe || '2h');
  const map: Record<string, number> = {
    '15min': 7,
    '2h': 30,
    '120T': 30,
    '1d': 90,
    '1w': 180,
  };
  return map[tf] ?? 30;
}

export function barDurationSec(timeframe: string | null | undefined): number {
  const tf = String(timeframe || '2h').toLowerCase();
  if (tf === '15min') return 900;
  if (tf === '1d') return 86400;
  if (tf === '1w') return 604800;
  return 7200;
}

export function mergeCandlesByTime(
  existing: Candle[] | null | undefined,
  incoming: Candle[] | null | undefined,
): Candle[] {
  const byTime = new Map<number, Candle>();
  for (const c of existing || []) {
    if (c && c.time != null) byTime.set(Number(c.time), c);
  }
  for (const c of incoming || []) {
    if (!c || c.time == null) continue;
    const t = Number(c.time);
    const prev = byTime.get(t);
    if (prev) {
      const next: Candle = { ...prev, ...c };
      if (c.volume == null && prev.volume != null) next.volume = prev.volume;
      byTime.set(t, next);
    } else {
      byTime.set(t, c);
    }
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time);
}

/** Drop feature points outside the loaded OHLCV window (poll/history merge safety). */
export function clipOverlayPointsToCandles(
  points: OverlayPoint[] | null | undefined,
  candles: Candle[],
): OverlayPoint[] {
  if (!points?.length || !candles?.length) return points || [];
  const tMin = Number(candles[0].time);
  const tMax = Number(candles[candles.length - 1].time);
  if (!Number.isFinite(tMin) || !Number.isFinite(tMax)) return points;
  return points.filter((p) => {
    const t = Number(p?.time);
    return Number.isFinite(t) && t >= tMin && t <= tMax;
  });
}

/**
 * Per-candle backward as-of only (no forward-fill past last feature row).
 * Use for chop regime hysteresis so stale 1.0 does not block exit on later bars.
 */
export function overlayAsOfAtCandleTimes(
  points: OverlayPoint[] | null | undefined,
  candles: Candle[],
): OverlayPoint[] {
  if (!candles?.length) return [];
  const sorted = [...(points || [])]
    .filter(
      (p) =>
        p &&
        Number.isFinite(Number(p.time)) &&
        Number.isFinite(Number(p.value)),
    )
    .sort((a, b) => Number(a.time) - Number(b.time));
  const out: OverlayPoint[] = [];
  for (const c of candles) {
    const t = Number(c?.time);
    if (!Number.isFinite(t)) continue;
    if (!sorted.length) {
      out.push({ time: t, value: null });
      continue;
    }
    const lastPtTime = Number(sorted[sorted.length - 1].time);
    let idx = -1;
    for (let k = 0; k < sorted.length; k++) {
      if (Number(sorted[k].time) <= t) idx = k;
      else break;
    }
    let v = idx >= 0 ? Number(sorted[idx].value) : null;
    if (v != null && Number.isFinite(v) && t > lastPtTime) {
      v = null;
    }
    out.push({
      time: t,
      value: v != null && Number.isFinite(v) ? v : null,
    });
  }
  return out;
}

/** Main-chart MA overlays: one point per candle, backward as-of + forward-fill. */
export function forwardFillOverlayToCandles(
  points: OverlayPoint[] | null | undefined,
  candles: Candle[],
): OverlayPoint[] {
  if (!candles?.length) return [];
  const sorted = [...(points || [])]
    .filter(
      (p) =>
        p &&
        Number.isFinite(Number(p.time)) &&
        Number.isFinite(Number(p.value)),
    )
    .sort((a, b) => Number(a.time) - Number(b.time));
  if (!sorted.length) return [];
  let j = 0;
  let last: number | null = null;
  const out: OverlayPoint[] = [];
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

/** Scalar feature at a chart bar (forward-fill aligned, same as subchart panes). */
export function overlayValueAtCandle(
  points: OverlayPoint[] | null | undefined,
  candles: Candle[],
  timeSec: number | null | undefined,
): number | null {
  if (timeSec == null || !Number.isFinite(Number(timeSec))) return null;
  if (!candles?.length) return null;
  const t = Number(timeSec);
  const filled = forwardFillOverlayToCandles(points, candles);
  const hit = filled.find((p) => Number(p.time) === t);
  if (hit?.value == null || !Number.isFinite(Number(hit.value))) return null;
  return Number(hit.value);
}

/** One timeline entry per OHLCV bar (whitespace where feature is missing). */
export function alignSeriesToCandleTimes(
  points: OverlayPoint[] | null | undefined,
  candles: Candle[],
): OverlayPoint[] {
  if (!candles?.length) return points || [];
  const byTime = new Map<number, number | null | undefined>();
  for (const p of points || []) {
    const t = Number(p?.time);
    if (!Number.isFinite(t)) continue;
    byTime.set(t, p?.value);
  }
  const out: OverlayPoint[] = [];
  for (const c of candles) {
    const t = Number(c?.time);
    if (!Number.isFinite(t)) continue;
    if (!byTime.has(t)) {
      out.push({ time: t });
      continue;
    }
    const v = byTime.get(t);
    if (v == null || (typeof v === 'number' && v !== v)) {
      out.push({ time: t });
    } else {
      out.push({ time: t, value: Number(v) });
    }
  }
  return out;
}

export function isoFromUnixSec(sec: number): string {
  return new Date(Number(sec) * 1000).toISOString();
}
