import type { OrderRow } from '@/api/types.ts';

/** Bar duration in seconds for Trade Map timeframes. */
export function barSecForTimeframe(timeframe: string): number {
  const map: Record<string, number> = {
    '15min': 900,
    '1h': 3600,
    '2h': 7200,
    '1d': 86400,
    '1w': 604800,
  };
  return map[String(timeframe || '').trim()] || 3600;
}

/** Parse order timestamp to unix seconds (chart bar time). */
export function orderRowUnixSec(row: OrderRow | null | undefined): number | null {
  if (!row) return null;
  const raw = row.time ?? row.filled_at ?? row.created_at;
  if (raw == null || raw === '') return null;
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return raw > 1e12 ? Math.floor(raw / 1000) : Math.floor(raw);
  }
  const ms = Date.parse(String(raw));
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : null;
}

export function orderOnBar(
  row: OrderRow,
  barTimeSec: number,
  barSec: number,
): boolean {
  const t = orderRowUnixSec(row);
  if (t == null || !Number.isFinite(barTimeSec)) return false;
  const half = Math.max(1, Math.floor(barSec / 2));
  return Math.abs(t - barTimeSec) <= half || (t >= barTimeSec && t < barTimeSec + barSec);
}
