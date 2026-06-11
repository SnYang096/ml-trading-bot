import type { Candle, TradeLink } from '@/api/types.ts';
import { CHART_THEME } from './constants.ts';
import { barDurationSec } from './ohlcv.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';

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
  timeframe: string,
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
  if (t1 <= t0) t1 = Math.min(last, t0 + barDurationSec(timeframe));
  out.entry_time = t0;
  out.exit_time = t1;
  return out;
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
): TradeLinkLine[] {
  const scoped = tradeLinksForDisplay(links, layers, strategyFocus);
  const out: TradeLinkLine[] = [];
  for (const raw of scoped) {
    const lk = candles.length ? clipLinkToCandles(raw, candles, timeframe) : raw;
    const t0 = Number(lk.entry_time);
    let t1 = Number(lk.exit_time);
    const p0 = Number(lk.entry_price);
    const p1 = Number(lk.exit_price);
    if (![t0, t1, p0, p1].every(Number.isFinite)) continue;
    if (t1 <= t0) t1 = t0 + barDurationSec(timeframe);
    out.push({
      color: lk.color || CHART_THEME.linkFallback,
      points: [
        { time: t0, value: p0 },
        { time: t1, value: p1 },
      ],
    });
  }
  return out;
}
