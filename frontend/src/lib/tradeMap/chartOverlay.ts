import type { Candle } from '@/api/types.ts';
import { CHART_THEME, METRICS_TABLE_MAX_COLS } from './constants.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';

export interface TimeSpan {
  start: number;
  end: number;
}

export interface ChopGridLabelSpec {
  price: number;
  text: string;
  side: string | null;
  kind: string;
  color?: string;
  spans: TimeSpan[] | null;
}

export interface ChopGridBatch {
  center?: number;
  spacing?: number;
  levels?: Array<{
    leg?: string;
    side?: string;
    grid_price?: number;
    tp_price?: number | null;
    tp_status?: string;
  }>;
}

export interface ChopMapPayload {
  chop_grid_overlay?: { batches?: ChopGridBatch[]; error?: string };
  chop_regime_regions?: TimeSpan[];
  strategy_stage_regions?: Record<string, Record<string, TimeSpan[]>>;
}

export function overlayAutoscaleInfoProvider(): null {
  return null;
}

export function mainChartOverlaySeriesOptions(
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    priceLineVisible: false,
    crosshairMarkerVisible: false,
    autoscaleInfoProvider: overlayAutoscaleInfoProvider,
    ...extra,
  };
}

export function bandHighlightSeriesOptions(rgba: string): Record<string, unknown> {
  return mainChartOverlaySeriesOptions({
    upColor: rgba,
    downColor: rgba,
    borderVisible: false,
    wickVisible: false,
    priceLineVisible: false,
    lastValueVisible: false,
  });
}

export function candleInAnySpan(c: Candle, spans: TimeSpan[]): boolean {
  const t = Number(c.time);
  return spans.some((r) => t >= Number(r.start) && t <= Number(r.end));
}

export function spanHighlightCandles(candles: Candle[], spans: TimeSpan[]): Candle[] {
  if (!candles.length || !spans.length) return [];
  return candles
    .filter((c) => candleInAnySpan(c, spans))
    .map((c) => {
      const lo = Number(c.low);
      const hi = Number(c.high);
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) return null;
      return { time: c.time, open: lo, high: hi, low: lo, close: hi };
    })
    .filter((c): c is Candle => c != null);
}

export function flattenStageRegions(
  byStrategy: Record<string, Record<string, TimeSpan[]>> | null | undefined,
  stage: string,
): TimeSpan[] {
  const spans: TimeSpan[] = [];
  if (!byStrategy || typeof byStrategy !== 'object') return spans;
  for (const strat of Object.keys(byStrategy)) {
    const block = byStrategy[strat];
    if (!block || typeof block !== 'object') continue;
    for (const r of block[stage] || []) {
      if (r && r.start != null && r.end != null) spans.push(r);
    }
  }
  return spans;
}

export function filterStageRegionsForFocus(
  byStrategy: Record<string, unknown> | null | undefined,
  strategyFocus: string,
): Record<string, unknown> | null | undefined {
  if (!byStrategy || typeof byStrategy !== 'object' || 'error' in byStrategy) {
    return byStrategy;
  }
  const focus = String(strategyFocus || '')
    .trim()
    .toLowerCase();
  if (!focus) return byStrategy;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(byStrategy)) {
    if (String(k).toLowerCase() === focus) out[k] = v;
  }
  return Object.keys(out).length ? out : byStrategy;
}

export function chopGridOverlayEnabled(
  layers: LayerState,
  strategyFocus: string,
): boolean {
  if (!layers.chopGrid || !layers.multiLeg) return false;
  const focus = String(strategyFocus || '')
    .trim()
    .toLowerCase();
  return !focus || focus === 'chop_grid';
}

export function fullWidthPriceLine(
  candles: Candle[],
  price: number,
): Array<{ time: number; value: number }> {
  if (!candles.length || !Number.isFinite(price)) return [];
  const first = candles[0];
  const last = candles[candles.length - 1];
  return [
    { time: first.time, value: price },
    { time: last.time, value: price },
  ];
}

export function priceLineInSpans(
  candles: Candle[],
  spans: TimeSpan[],
  price: number,
): Array<{ time: number; value: number }> {
  if (!candles.length || !spans.length || !Number.isFinite(price)) return [];
  const firstCandle = Number(candles[0].time);
  const lastCandle = Number(candles[candles.length - 1].time);
  const sortedRaw = spans
    .map((s) => ({ start: Number(s.start), end: Number(s.end) }))
    .filter(
      (s) =>
        Number.isFinite(s.start) &&
        Number.isFinite(s.end) &&
        s.end >= s.start &&
        s.end >= firstCandle &&
        s.start <= lastCandle,
    )
    .map((s) => ({
      start: Math.max(s.start, firstCandle),
      end: Math.min(s.end, lastCandle),
    }))
    .sort((a, b) => a.start - b.start);
  if (!sortedRaw.length) return [];
  const merged: TimeSpan[] = [sortedRaw[0]];
  for (let i = 1; i < sortedRaw.length; i++) {
    const cur = sortedRaw[i];
    const last = merged[merged.length - 1];
    if (cur.start <= last.end) last.end = Math.max(last.end, cur.end);
    else merged.push({ ...cur });
  }
  const pts: Array<{ time: number; value: number }> = [];
  for (let i = 0; i < merged.length; i++) {
    const span = merged[i];
    if (span.start === span.end) pts.push({ time: span.start, value: price });
    else {
      pts.push({ time: span.start, value: price });
      pts.push({ time: span.end, value: price });
    }
    if (i < merged.length - 1) {
      const next = merged[i + 1];
      const gapT = span.end + 1;
      if (gapT < next.start) pts.push({ time: gapT, value: Number.NaN });
    }
  }
  return pts;
}

export interface ChopLineSpec {
  points: Array<{ time: number; value: number }>;
  color: string;
  lineWidth?: number;
  lineStyle?: number;
  label?: ChopGridLabelSpec;
}

export function buildChopGridLineSpecs(
  overlay: { batches?: ChopGridBatch[] },
  candles: Candle[],
  lineSpans: TimeSpan[] | null,
): ChopLineSpec[] {
  const specs: ChopLineSpec[] = [];
  const spans = lineSpans?.length ? lineSpans : null;
  for (const batch of overlay?.batches || []) {
    const center = Number(batch.center);
    if (center > 0) {
      specs.push({
        points: spans?.length
          ? priceLineInSpans(candles, spans, center)
          : fullWidthPriceLine(candles, center),
        color: CHART_THEME.text,
        lineWidth: 2,
        lineStyle: 2,
        label: {
          price: center,
          text: '中心',
          side: 'long',
          kind: 'center',
          color: CHART_THEME.text,
          spans: spans ? spans.map((s) => ({ start: s.start, end: s.end })) : null,
        },
      });
    }
    for (const lv of batch.levels || []) {
      const leg = String(lv.leg || '').toUpperCase();
      const isLong = lv.side === 'long';
      const gridColor = isLong ? 'rgba(59, 130, 246, 0.55)' : 'rgba(249, 115, 22, 0.55)';
      const gridPx = Number(lv.grid_price);
      if (Number.isFinite(gridPx) && gridPx > 0) {
        specs.push({
          points: spans?.length
            ? priceLineInSpans(candles, spans, gridPx)
            : fullWidthPriceLine(candles, gridPx),
          color: gridColor,
          lineStyle: 2,
          label: {
            price: gridPx,
            text: `${leg} 格`,
            side: isLong ? 'long' : 'short',
            kind: 'grid',
            color: gridColor,
            spans: spans ? spans.map((s) => ({ start: s.start, end: s.end })) : null,
          },
        });
      }
      const tpPx = lv.tp_price != null ? Number(lv.tp_price) : null;
      if (tpPx != null && tpPx > 0) {
        const tpSt = String(lv.tp_status || '').toLowerCase();
        const tpOpen = ['open', 'pending', 'new', 'submitted', 'shadow'].includes(tpSt);
        if (!tpOpen) continue;
        specs.push({
          points: spans?.length
            ? priceLineInSpans(candles, spans, tpPx)
            : fullWidthPriceLine(candles, tpPx),
          color: CHART_THEME.accentPurple,
          lineStyle: 1,
          label: {
            price: tpPx,
            text: `${leg}_TP`,
            side: isLong ? 'long' : 'short',
            kind: 'tp',
            color: CHART_THEME.accentPurple,
            spans: spans ? spans.map((s) => ({ start: s.start, end: s.end })) : null,
          },
        });
      }
    }
  }
  return specs.filter((s) => s.points.length);
}

export function chopOverlaySpans(
  prefilterSpans: TimeSpan[],
  data: ChopMapPayload,
): TimeSpan[] {
  if (prefilterSpans.length) return prefilterSpans;
  return (data.chop_regime_regions || []) as TimeSpan[];
}

export function labelTimeForSpans(spans: TimeSpan[] | null, candles: Candle[]): number | null {
  if (!candles.length) return null;
  if (!spans?.length) return Number(candles[candles.length - 1].time);
  let best: number | null = null;
  for (const span of spans) {
    const start = Number(span.start);
    const end = Number(span.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    for (const c of candles) {
      const t = Number(c.time);
      if (t >= start && t <= end && (best == null || t > best)) best = t;
    }
  }
  return best ?? Number(candles[candles.length - 1].time);
}

export function chopMapDataForStrategyFocus(
  data: ChopMapPayload | null | undefined,
  strategyFocus: string,
  chopEnabled: boolean,
): ChopMapPayload {
  const raw = data || {};
  const stages = filterStageRegionsForFocus(
    raw.strategy_stage_regions as Record<string, unknown> | undefined,
    strategyFocus,
  ) as ChopMapPayload['strategy_stage_regions'];
  if (chopEnabled) return { ...raw, strategy_stage_regions: stages };
  return {
    chop_grid_overlay: { batches: [] },
    chop_regime_regions: [],
    strategy_stage_regions: stages,
  };
}

export function stageBandFill(stage: 'prefilter' | 'gate'): string {
  return stage === 'gate' ? 'rgba(124, 58, 237, 0.11)' : 'rgba(239, 68, 68, 0.14)';
}

export function visibleCandleIndexRange(
  candles: Candle[],
  logical: { from: number; to: number } | null,
  maxCols: number = METRICS_TABLE_MAX_COLS,
): { from: number; to: number } {
  const list = candles || [];
  if (!list.length) return { from: 0, to: 0 };
  let from: number;
  let to: number;
  if (!logical) {
    to = list.length - 1;
    from = Math.max(0, to - Math.max(1, maxCols) + 1);
  } else {
    from = Math.max(0, Math.min(list.length - 1, Math.floor(Number(logical.from))));
    to = Math.max(0, Math.min(list.length - 1, Math.ceil(Number(logical.to))));
    from = Math.min(from, to);
    to = Math.max(from, to);
  }
  const span = to - from + 1;
  const cap = Math.max(1, Number(maxCols) || METRICS_TABLE_MAX_COLS);
  if (span > cap) {
    from = to - cap + 1;
  }
  return { from, to };
}

export function formatMetricsBarHeader(timeSec: number): string {
  const d = new Date(Number(timeSec) * 1000);
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${mm}-${dd} ${hh}h`;
}

export const subchartBaseOptions = {
  layout: {
    background: { color: CHART_THEME.bg },
    textColor: CHART_THEME.text,
    attributionLogo: false,
  },
  grid: { vertLines: { visible: false }, horzLines: { visible: false } },
  timeScale: { visible: false },
  rightPriceScale: {
    borderColor: CHART_THEME.border,
    scaleMargins: { top: 0.05, bottom: 0.05 },
    minimumWidth: 72,
  },
  handleScroll: false,
  handleScale: false,
};
