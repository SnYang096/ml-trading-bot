import { DEFAULT_VISIBLE_BARS } from './constants.ts';
import type { Candle, LogicalRange, OverlayPoint, PriceRange } from './types.ts';

/** How many bars to show by default (tail window); avoids fitContent squashing 2k+ bars to 0px. */
export function defaultVisibleBarCount(barCount: number, cap?: number): number {
  const n = Math.max(0, Number(barCount) || 0);
  if (n <= 0) return 0;
  const limit = Number(cap) > 0 ? Number(cap) : DEFAULT_VISIBLE_BARS;
  return Math.min(n, Math.max(30, limit));
}

export function visibleLogicalRange(
  barCount: number,
  visibleBars?: number,
): LogicalRange | null {
  const n = Math.max(0, Number(barCount) || 0);
  if (n <= 0) return null;
  const vis = defaultVisibleBarCount(n, visibleBars);
  return { from: Math.max(0, n - vis), to: n - 1 };
}

/** Shift visible logical range after prepending `added` bars (apply only after setData). */
export function logicalRangeAfterHistoryPrepend(
  snap: LogicalRange | null | undefined,
  added: number,
  barCount: number,
): LogicalRange | null {
  if (!snap || added <= 0 || barCount <= 0) return null;
  const from = Number(snap.from) + added;
  const to = Number(snap.to) + added;
  if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return null;
  if (to >= barCount) return null;
  return { from: Math.max(0, from), to };
}

export function isValidLogicalRange(
  range: LogicalRange | null | undefined,
  barCount: number,
): range is LogicalRange {
  if (!range || barCount <= 0) return false;
  const from = Number(range.from);
  const to = Number(range.to);
  return Number.isFinite(from) && Number.isFinite(to) && from >= 0 && to > from && to < barCount;
}

export function clampCandleOhlc(
  open: number,
  high: number,
  low: number,
  close: number,
): { open: number; high: number; low: number; close: number } {
  let o = open;
  let h = high;
  let l = low;
  let c = close;
  if (!Number.isFinite(o)) o = c;
  if (!Number.isFinite(h)) h = Math.max(o, c);
  if (!Number.isFinite(l)) l = Math.min(o, c);
  if (l < 0) l = Math.min(o, c);
  if (h < l) {
    const t = h;
    h = l;
    l = t;
  }
  const ref = Math.max(Math.abs(c), Math.abs(o), 1);
  const wickCap = Math.max(ref * 0.35, 5);
  if (h > c + wickCap * 8) h = Math.max(o, c);
  if (l < c - wickCap * 8) l = Math.min(o, c);
  if (h < Math.max(o, c)) h = Math.max(o, c);
  if (l > Math.min(o, c)) l = Math.min(o, c);
  return { open: o, high: h, low: l, close: c };
}

export function sanitizeCandlesForLwc(candles: Candle[] | null | undefined): Candle[] {
  if (!Array.isArray(candles) || !candles.length) return [];
  const out: Candle[] = [];
  let lastT: number | null = null;
  for (const raw of candles) {
    const time = Number(raw?.time);
    const close = Number(raw?.close);
    if (!Number.isFinite(time) || !Number.isFinite(close) || close <= 0) continue;
    if (lastT != null && time <= lastT) continue;
    lastT = time;
    const ohlc = clampCandleOhlc(
      Number(raw?.open),
      Number(raw?.high),
      Number(raw?.low),
      close,
    );
    const c: Candle = { time, ...ohlc };
    if (raw?.volume != null && Number.isFinite(Number(raw.volume))) {
      c.volume = Number(raw.volume);
    }
    out.push(c);
  }
  return out;
}

/** Min/max price for bars in the visible logical index window (for autoscale). */
export function priceRangeForVisibleCandles(
  candles: Candle[],
  logicalRange: LogicalRange | null | undefined,
): PriceRange | null {
  if (!Array.isArray(candles) || !candles.length || !logicalRange) return null;
  const from = Math.max(0, Math.floor(Number(logicalRange.from)));
  const to = Math.min(candles.length - 1, Math.ceil(Number(logicalRange.to)));
  if (to < from) return null;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = from; i <= to; i++) {
    const c = candles[i];
    if (!c) continue;
    const lo = Number(c.low);
    const hi = Number(c.high);
    if (Number.isFinite(lo)) minV = Math.min(minV, lo);
    if (Number.isFinite(hi)) maxV = Math.max(maxV, hi);
  }
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return null;
  const span = Math.max(maxV - minV, maxV * 0.0005);
  const pad = Math.max(span * 0.06, maxV * 0.001);
  return { minValue: minV - pad, maxValue: maxV + pad };
}

/** Visible-window range, else full series (never return null when candles exist). */
export function priceRangeForChartAutoscale(
  candles: Candle[],
  logicalRange: LogicalRange | null | undefined,
): PriceRange | null {
  if (!Array.isArray(candles) || !candles.length) return null;
  const vis = priceRangeForVisibleCandles(candles, logicalRange);
  if (vis) return vis;
  return priceRangeForVisibleCandles(candles, {
    from: 0,
    to: candles.length - 1,
  });
}

/** Expand OHLC autoscale to include main-chart overlay values in the visible window. */
export function expandPriceRangeForOverlays(
  baseRange: PriceRange | null | undefined,
  candles: Candle[],
  logicalRange: LogicalRange | null | undefined,
  overlayDataByKey: Map<string, OverlayPoint[]> | null | undefined,
): PriceRange | null | undefined {
  if (!baseRange || !overlayDataByKey || typeof overlayDataByKey.forEach !== 'function') {
    return baseRange;
  }
  let minV = Number(baseRange.minValue);
  let maxV = Number(baseRange.maxValue);
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return baseRange;
  const fromIdx =
    logicalRange && Number.isFinite(Number(logicalRange.from))
      ? Math.max(0, Math.floor(Number(logicalRange.from)))
      : 0;
  const toIdx =
    logicalRange && Number.isFinite(Number(logicalRange.to))
      ? Math.min(candles.length - 1, Math.ceil(Number(logicalRange.to)))
      : candles.length - 1;
  const tMin = Number(candles[fromIdx]?.time);
  const tMax = Number(candles[toIdx]?.time);
  overlayDataByKey.forEach((pts) => {
    for (const p of pts || []) {
      const t = Number(p.time);
      const v = Number(p.value);
      if (!Number.isFinite(v)) continue;
      if (Number.isFinite(tMin) && Number.isFinite(tMax) && (t < tMin || t > tMax)) continue;
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  });
  const pad = Math.max((maxV - minV) * 0.02, 1e-6);
  return { minValue: minV - pad, maxValue: maxV + pad };
}

/** Bar spacing in px for the *visible* window (not full history length). */
export function barSpacingForCount(barCount: number): number {
  const n = Math.max(0, Number(barCount) || 0);
  if (n > 600) return 3;
  if (n > 300) return 4;
  if (n > 120) return 5;
  if (n > 50) return 6;
  return 8;
}
