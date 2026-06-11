import type { Candle, TradeMarker } from '@/api/types.ts';
import type { FeatureOverlays } from './types.ts';
import { clipOverlayPointsToCandles } from './ohlcv.ts';
import type { FeatureOverlay } from './types.ts';

export function mergeOverlayPoints<T extends { time: number }>(
  existing: T[] | null | undefined,
  incoming: T[] | null | undefined,
): T[] {
  const byTime = new Map<number, T>();
  for (const p of existing || []) {
    if (p && p.time != null) byTime.set(Number(p.time), p);
  }
  for (const p of incoming || []) {
    if (p && p.time != null) byTime.set(Number(p.time), p);
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time);
}

export function mergeMarkersById(
  existing: TradeMarker[] | null | undefined,
  incoming: TradeMarker[] | null | undefined,
): TradeMarker[] {
  const byId = new Map<string, TradeMarker>();
  for (const m of existing || []) {
    if (m?.id) byId.set(m.id, m);
  }
  for (const m of incoming || []) {
    if (m?.id) byId.set(m.id, m);
  }
  return [...byId.values()].sort((a, b) => Number(a.time) - Number(b.time));
}

export function mergeFeatureOverlays(
  existing: FeatureOverlays,
  incoming: FeatureOverlays,
  candles: Candle[],
): FeatureOverlays {
  const merged: FeatureOverlays = { ...existing };
  for (const [col, spec] of Object.entries(incoming)) {
    if (!spec) continue;
    const prevPts = merged[col]?.points || [];
    const nextPts = spec.points || [];
    merged[col] = {
      ...spec,
      points: clipOverlayPointsToCandles(
        mergeOverlayPoints(prevPts, nextPts) as FeatureOverlay['points'],
        candles,
      ),
    } as FeatureOverlay;
  }
  return merged;
}
