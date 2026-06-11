import { SUBCHART_COLORS } from './constants.ts';
import type { EligibilityInfo, NavLink } from './types.ts';

export function stageRegionsQueryParam(prefilter: boolean, gate: boolean): string {
  const parts: string[] = [];
  if (prefilter) parts.push('prefilter');
  if (gate) parts.push('gate');
  return parts.length ? parts.join(',') : '';
}

export function mainOverlaysQueryParam(ema1200: boolean, weeklyEma200: boolean): string {
  const parts: string[] = [];
  if (ema1200) parts.push('ema_1200');
  if (weeklyEma200) parts.push('weekly_ema_200');
  return parts.length ? parts.join(',') : '';
}

export function formatEligibility(elig: EligibilityInfo | null | undefined): string {
  if (!elig) return '—';
  const lines = [
    `can_buy: ${elig.can_buy}`,
    `weekly_ema_200_position: ${elig.weekly_ema_200_position ?? 'n/a'}`,
    `blockers: ${(elig.blockers || []).join(', ') || 'none'}`,
  ];
  return lines.join('\n');
}

export function browserLocalUrl(port: number, path?: string): string {
  const host =
    (typeof globalThis !== 'undefined' &&
      globalThis.location &&
      globalThis.location.hostname) ||
    '127.0.0.1';
  return `http://${host}:${port}${path || ''}`;
}

export function resolveLinkUrl(link: NavLink | null | undefined): string {
  if (link && link.id === 'grafana') {
    return browserLocalUrl(3000);
  }
  return (link && link.url) || '';
}

export function subchartColor(index: number): string {
  const colors = SUBCHART_COLORS || [];
  return colors[Math.abs(index) % colors.length];
}

export function filterFeatureColumns(
  columns: string[] | null | undefined,
  query: string | null | undefined,
): string[] {
  const q = String(query || '')
    .trim()
    .toLowerCase();
  if (!q) return columns || [];
  return (columns || []).filter((c) => String(c).toLowerCase().includes(q));
}

export function featureColumnsParam(selected: string[] | null | undefined): string {
  return (selected || []).filter(Boolean).join(',');
}

export function parseStoredLayout(raw: string | null | undefined): unknown {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
