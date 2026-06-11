import type { Candle, TradeLink } from '@/api/types.ts';
import { CHART_THEME } from './constants.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';
import type { TradeMarker } from './types.ts';

export type { TradeLink };

export function tradeLinkAccountLayer(link: TradeLink): string {
  const strat = String(link?.strategy || '').toLowerCase();
  if (strat === 'spot_accum_simple' || strat.startsWith('spot')) return 'spot';
  if (strat === 'chop_grid' || strat === 'trend_scalp') return 'multi_leg';
  return 'trend';
}

export function tradeLinksForDisplay(
  links: TradeLink[] | null | undefined,
  layers: LayerState,
  strategyFocus: string,
): TradeLink[] {
  const allowed: string[] = [];
  if (layers.trend) allowed.push('trend');
  if (layers.spot) allowed.push('spot');
  if (layers.multiLeg) allowed.push('multi_leg');
  const focus = String(strategyFocus || '')
    .trim()
    .toLowerCase();
  return (links || []).filter((lk) => {
    const layer = tradeLinkAccountLayer(lk);
    if (!allowed.includes(layer)) return false;
    if (focus && String(lk.strategy || '').toLowerCase() !== focus) return false;
    if (String(lk.status || '').toLowerCase() !== 'closed') return false;
    if (!lk.exit_marker_id) return false;
    return true;
  });
}

export function tradeLinkKey(link: TradeLink): string {
  return [
    link?.scope,
    link?.strategy,
    link?.entry_time,
    link?.exit_time,
    link?.entry_price,
    link?.exit_price,
  ].join('|');
}

export function mergeTradeLinks(
  existing: TradeLink[] | null | undefined,
  incoming: TradeLink[] | null | undefined,
): TradeLink[] {
  const byKey = new Map<string, TradeLink>();
  for (const lk of [...(existing || []), ...(incoming || [])]) {
    byKey.set(tradeLinkKey(lk), lk);
  }
  return [...byKey.values()];
}

export function nearestLoadedCandleTime(candles: Candle[], rawTime: number): number {
  const t = Number(rawTime);
  if (!Number.isFinite(t) || !candles.length) return t;
  let best = Number(candles[0].time);
  let bestDist = Math.abs(best - t);
  for (const c of candles) {
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

export function clipLinkToCandles(
  link: TradeLink,
  candles: Candle[],
  _timeframe?: string,
): TradeLink {
  if (!candles.length) return link;
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
  t0 = nearestLoadedCandleTime(candles, t0);
  t1 = nearestLoadedCandleTime(candles, t1);
  if (t1 < t0) t1 = t0;
  out.entry_time = t0;
  out.exit_time = t1;
  return out;
}

function markerById(
  markers: TradeMarker[] | null | undefined,
  markerId: string | null | undefined,
): TradeMarker | null {
  const id = String(markerId || '').trim();
  if (!id) return null;
  return (markers || []).find((m) => m.id === id) || null;
}

function linkEndpoint(
  marker: TradeMarker | null,
  fallbackTime: number,
  fallbackPrice: number,
): { time: number; value: number } | null {
  const tRaw = marker ? Number(marker.time) : fallbackTime;
  const t = Number.isFinite(tRaw) ? tRaw : fallbackTime;
  let value = fallbackPrice;
  if (marker?.price != null && Number.isFinite(Number(marker.price))) {
    value = Number(marker.price);
  }
  if (!Number.isFinite(t) || !Number.isFinite(value)) return null;
  return { time: t, value };
}

/** Resolve link endpoints from chart markers when ids match (same times as arrow markers). */
export function resolveTradeLinkEndpoints(
  link: TradeLink,
  markers: TradeMarker[] | null | undefined,
  candles: Candle[],
  timeframe: string,
): { entry: { time: number; value: number }; exit: { time: number; value: number } } | null {
  const clipped = candles.length ? clipLinkToCandles(link, candles, timeframe) : link;
  const entryMarker = markerById(markers, link.entry_marker_id);
  const exitMarker = markerById(markers, link.exit_marker_id);
  const entry = linkEndpoint(
    entryMarker,
    Number(clipped.entry_time),
    Number(clipped.entry_price),
  );
  const exit = linkEndpoint(
    exitMarker,
    Number(clipped.exit_time),
    Number(clipped.exit_price),
  );
  if (!entry || !exit) return null;
  let t0 = entry.time;
  let t1 = exit.time;
  if (t1 < t0) t1 = t0;
  return {
    entry: { time: t0, value: entry.value },
    exit: { time: t1, value: exit.value },
  };
}

export interface TradeLinkLine {
  color: string;
  points: Array<{ time: number; value: number }>;
}

export function buildTradeLinkLines(
  links: TradeLink[] | null | undefined,
  candles: Candle[],
  layers: LayerState,
  strategyFocus: string,
  timeframe: string,
  markers?: TradeMarker[] | null,
): TradeLinkLine[] {
  const scoped = tradeLinksForDisplay(links, layers, strategyFocus);
  const out: TradeLinkLine[] = [];
  for (const raw of scoped) {
    const resolved = resolveTradeLinkEndpoints(raw, markers, candles, timeframe);
    if (!resolved) continue;
    const { entry, exit } = resolved;
    out.push({
      color: raw.color || CHART_THEME.linkFallback,
      points: [
        { time: entry.time, value: entry.value },
        { time: exit.time, value: exit.value },
      ],
    });
  }
  return out;
}
