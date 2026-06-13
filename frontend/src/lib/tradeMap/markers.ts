import {
  ENTRY_SHAPES,
  EXIT_SHAPE,
  TP_MARKER_COLOR,
  TP_MARKER_SHAPE,
} from './constants.ts';
import { overlayAsOfAtCandleTimes } from './ohlcv.ts';
import type {
  Candle,
  ChopRegimeRegion,
  FeatureOverlay,
  FeatureOverlays,
  LwcSeriesMarker,
  OverlayPoint,
  TradeMarker,
} from './types.ts';

export type MarkerRole = 'tp' | 'grid' | 'exit' | 'entry';

export function markerRole(marker: TradeMarker): MarkerRole {
  const ev = String(marker.event || '').toLowerCase();
  if (ev === 'tp') return 'tp';
  if (ev === 'grid') return 'grid';
  if (ev === 'exit') return 'exit';
  return 'entry';
}

export function markerShape(marker: TradeMarker): string {
  const role = markerRole(marker);
  const pending = (marker.status || 'filled').toLowerCase() === 'pending';
  const regimeExit =
    role === 'exit' &&
    marker.detail &&
    String(marker.detail.exit_kind || '').toLowerCase() === 'regime_or_risk_exit';
  if (regimeExit) return 'circle';
  if (role === 'tp') return pending ? 'circle' : TP_MARKER_SHAPE;
  if (role === 'exit') return EXIT_SHAPE;
  if (role === 'grid') return 'square';
  const side = (marker.side || 'long').toLowerCase();
  if (marker.is_add && side === 'long') return 'diamond';
  if (marker.is_add && side === 'short') return 'diamond';
  return ENTRY_SHAPES[side] || 'arrowUp';
}

export function markerColor(marker: TradeMarker): string {
  const pending = (marker.status || 'filled').toLowerCase() === 'pending';
  if (pending) return '#1faa1f';
  const role = markerRole(marker);
  const side = (marker.side || 'long').toLowerCase();
  const pnl = marker.pnl_usdt;
  if (role === 'exit' && pnl != null) {
    return pnl >= 0 ? '#00ff88' : '#ff0040';
  }
  if (role === 'tp') return TP_MARKER_COLOR;
  if (
    role === 'exit' &&
    marker.detail &&
    String(marker.detail.exit_kind || '').toLowerCase() === 'regime_or_risk_exit'
  ) {
    return marker.color || '#ffb000';
  }
  if (role === 'grid') return marker.color || '#00ff41';
  if (role === 'entry') {
    return side === 'long' ? '#00ff41' : '#ff0040';
  }
  return marker.color || '#00ffff';
}

export function filterMarkersByStrategy(
  markers: TradeMarker[] | null | undefined,
  strategyFocus: string | null | undefined,
): TradeMarker[] {
  const focus = strategyFocus ? String(strategyFocus).trim().toLowerCase() : '';
  if (!focus) return markers || [];
  return (markers || []).filter(
    (m) => String(m.strategy || '').toLowerCase() === focus,
  );
}

/** Chart markers: strategy filter + keep selected id visible for highlight/scroll. */
export function markersForChartDisplay(
  allMarkers: TradeMarker[] | null | undefined,
  strategyFocus: string | null | undefined,
  selectedMarkerId: string | null | undefined,
): TradeMarker[] {
  const scoped = filterMarkersByStrategy(allMarkers || [], strategyFocus);
  const sel = selectedMarkerId ? String(selectedMarkerId).trim() : '';
  if (!sel) return scoped;
  if (scoped.some((m) => m.id === sel)) return scoped;
  const hit = (allMarkers || []).find((m) => m.id === sel);
  if (!hit) return scoped;
  return [...scoped, hit].sort((a, b) => Number(a.time) - Number(b.time));
}

export function findMarkerByTime(
  markers: TradeMarker[] | null | undefined,
  clickTime: number,
  toleranceSec: number,
): TradeMarker | null {
  const t = Number(clickTime);
  if (!Number.isFinite(t)) return null;
  const tol = Number(toleranceSec) || 3600;
  let best: TradeMarker | null = null;
  let bestDist = Infinity;
  for (const m of markers || []) {
    const mt = Number(m.time);
    if (!Number.isFinite(mt)) continue;
    const dist = Math.abs(mt - t);
    if (dist <= tol && dist < bestDist) {
      bestDist = dist;
      best = m;
    }
  }
  return best;
}

/** Prefer markers pinned to the clicked bar (avoids jumping to another bar's marker). */
export function findMarkerOnBar(
  markers: TradeMarker[] | null | undefined,
  clickTime: number,
  toleranceSec: number,
): TradeMarker | null {
  const t = Number(clickTime);
  if (!Number.isFinite(t)) return null;
  const onBar = (markers || []).filter((m) => Number(m.time) === t);
  if (!onBar.length) return findMarkerByTime(markers, clickTime, toleranceSec);
  const regime = onBar.filter(
    (m) =>
      String(m.event || '').toLowerCase() === 'exit' &&
      m.detail &&
      String(m.detail.exit_kind || '').toLowerCase() === 'regime_or_risk_exit',
  );
  if (regime.length) return regime[regime.length - 1];
  return onBar[onBar.length - 1];
}

export function isFeatureBusRegimeExitMarker(m: TradeMarker | null | undefined): boolean {
  if (!m) return false;
  const id = String(m.id || '');
  if (!id.startsWith('multi_leg:regime_exit:')) return false;
  return String(m.detail?.source || '') === 'feature_bus_hysteresis';
}

export function chopRegimeThresholdsFromOverlay(overlay: FeatureOverlay | null | undefined): {
  entryMin: number;
  exitBelow: number;
} {
  let entryMin = 0.5;
  let exitBelow = 0.32;
  const refs = overlay?.reference_lines || [];
  for (const r of refs) {
    const y = Number(r.y != null ? r.y : r.value);
    if (!Number.isFinite(y)) continue;
    const op = String(r.operator || '');
    if (op.includes('>=')) entryMin = y;
    else if (op === '<' || (op.includes('<') && !op.includes('='))) exitBelow = y;
  }
  return { entryMin, exitBelow };
}

/** Match live: missing chop reads as 0.0 → not in regime (do not hold active on gaps). */
export function chopGridHysteresisActive(
  values: Array<number | null | undefined>,
  entryMin: number,
  exitBelow: number,
): boolean[] {
  let active = false;
  const out: boolean[] = [];
  for (const val of values) {
    if (val == null || !Number.isFinite(val)) {
      active = false;
      out.push(false);
      continue;
    }
    if (!active) active = val >= entryMin;
    else if (val < exitBelow) active = false;
    out.push(active);
  }
  return out;
}

export interface ChopRegimeSeriesResult {
  series: OverlayPoint[];
  vals: Array<number | null>;
  chopOn: boolean[];
  entryMin: number;
  exitBelow: number;
}

export function chopRegimeSeriesFromOverlay(
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
): ChopRegimeSeriesResult {
  const ol = overlays?.bpc_semantic_chop;
  if (!ol || !Array.isArray(candles) || !candles.length) {
    return { series: [], vals: [], chopOn: [], entryMin: 0.5, exitBelow: 0.32 };
  }
  const series = overlayAsOfAtCandleTimes(ol.points || [], candles);
  const { entryMin, exitBelow } = chopRegimeThresholdsFromOverlay(ol);
  const vals = series.map((p) =>
    p.value == null || !Number.isFinite(Number(p.value)) ? null : Number(p.value),
  );
  const chopOn = chopGridHysteresisActive(vals, entryMin, exitBelow);
  return { series, vals, chopOn, entryMin, exitBelow };
}

export function chopRegimeHysteresisOnAtTime(
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
  timeSec: number,
): boolean {
  const { series, chopOn } = chopRegimeSeriesFromOverlay(candles, overlays);
  const t = Number(timeSec);
  if (!Number.isFinite(t)) return false;
  for (let i = 0; i < series.length; i++) {
    if (Number(series[i].time) === t) return !!chopOn[i];
  }
  return false;
}

/** Synthetic regime exits from bpc_semantic_chop overlay (matches live hysteresis flatten). */
export function synthesizeChopRegimeExitMarkers(
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
): TradeMarker[] {
  const ol = overlays?.bpc_semantic_chop;
  if (!ol || !Array.isArray(candles) || !candles.length) return [];
  const { vals, chopOn } = chopRegimeSeriesFromOverlay(candles, overlays);
  if (!chopOn.length) return [];
  const { entryMin, exitBelow } = chopRegimeThresholdsFromOverlay(ol);
  const sym = String((candles[0] && candles[0].symbol) || 'BNBUSDT').toUpperCase();
  const markers: TradeMarker[] = [];
  for (let i = 1; i < chopOn.length; i++) {
    if (!(chopOn[i - 1] && !chopOn[i])) continue;
    const val = vals[i];
    if (val != null && Number.isFinite(val) && val >= exitBelow) continue;
    const t = Number(candles[i].time);
    if (!Number.isFinite(t)) continue;
    markers.push({
      id: `multi_leg:regime_exit:${sym}:${t}`,
      time: t,
      symbol: sym,
      scope: 'multi_leg',
      strategy: 'chop_grid',
      event: 'exit',
      side: 'long',
      status: 'filled',
      color: '#ffb000',
      detail: {
        exit_kind: 'regime_or_risk_exit',
        exit_reason: 'regime_or_risk_exit',
        chop: val,
        entry_min: entryMin,
        exit_below: exitBelow,
        source: 'overlay_hysteresis',
      },
    });
  }
  return markers;
}

/** Bar times (unix sec) where chop regime hysteresis turns off — for metrics table headers. */
export function chopRegimeExitBarTimes(
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
): Set<number> {
  const times = new Set<number>();
  for (const m of synthesizeChopRegimeExitMarkers(candles, overlays)) {
    const t = Number(m.time);
    if (Number.isFinite(t)) times.add(t);
  }
  return times;
}

export function chopRegimeHysteresisOnBarTimes(
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
): Set<number> {
  const { series, chopOn } = chopRegimeSeriesFromOverlay(candles, overlays);
  const on = new Set<number>();
  for (let i = 0; i < series.length; i++) {
    if (chopOn[i]) on.add(Number(series[i].time));
  }
  return on;
}

export function mergeRegimeExitMarkers(
  markers: TradeMarker[] | null | undefined,
  regimeExits: TradeMarker[] | null | undefined,
): TradeMarker[] {
  const base = markers || [];
  const adds = regimeExits || [];
  if (!adds.length) return base;
  const chopExitTimes = new Set<number>();
  for (const m of base) {
    if (String(m.strategy || '').toLowerCase() !== 'chop_grid') continue;
    if (String(m.event || '').toLowerCase() !== 'exit') continue;
    const t = Number(m.time);
    if (Number.isFinite(t)) chopExitTimes.add(t);
  }
  const seen = new Set(base.map((m) => String(m.id || '')));
  const out = base.slice();
  for (const m of adds) {
    const id = String(m.id || '');
    if (seen.has(id)) continue;
    const t = Number(m.time);
    if (!Number.isFinite(t)) continue;
    let dup = false;
    for (const et of chopExitTimes) {
      if (Math.abs(t - et) <= 1) {
        dup = true;
        break;
      }
    }
    if (dup) continue;
    seen.add(id);
    chopExitTimes.add(t);
    out.push(m);
  }
  return out.sort((a, b) => Number(a.time) - Number(b.time));
}

export function chopGridMarkerDisplayText(m: TradeMarker, pending: boolean): string {
  const strat = (m.strategy || '').toLowerCase();
  if (strat !== 'chop_grid') return '';
  const ev = String(m.event || '').toLowerCase();
  if (
    ev === 'exit' &&
    m.detail &&
    String(m.detail.exit_kind || '').toLowerCase() === 'regime_or_risk_exit'
  ) {
    return 'regime退出';
  }
  const leg = String((m.detail && m.detail.leg_label) || '').trim().toUpperCase();
  if (!leg) return '';
  if (ev === 'tp') return leg.endsWith('_TP') ? leg : `${leg}_TP`;
  if (ev === 'entry' && !pending) return `${leg} 成交`;
  if (pending || ev === 'grid') return `${leg} 挂单`;
  return leg;
}

export function chopGridLegSide(legLabel: string | null | undefined): 'long' | 'short' | null {
  const leg = String(legLabel || '').toUpperCase();
  const m = leg.match(/(?:^|_)([LS])(\d+)/);
  if (!m) return null;
  return m[1] === 'L' ? 'long' : 'short';
}

/** Long grid labels below price line; short above; long TP above; short TP below. */
export function chopGridLabelAnchor(side: string, kind: string): 'above' | 'below' {
  if (kind === 'center') return 'below';
  const isLong = String(side || '').toLowerCase() === 'long';
  const isTp = kind === 'tp';
  if (isTp) return isLong ? 'above' : 'below';
  return isLong ? 'below' : 'above';
}

export function chopSegmentedLinePoints(
  regions: ChopRegimeRegion[] | null | undefined,
  price: number,
  barSec: number,
): OverlayPoint[] {
  const px = Number(price);
  if (!Number.isFinite(px) || !regions?.length) return [];
  const gap = Math.max(1, Number(barSec) || 7200);
  const pts: OverlayPoint[] = [];
  const sorted = [...regions].sort((a, b) => Number(a.start) - Number(b.start));
  for (const r of sorted) {
    const start = Number(r.start);
    const end = Number(r.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    pts.push({ time: start, value: px });
    pts.push({ time: end, value: px });
    pts.push({ time: end + gap, value: NaN });
  }
  return pts;
}

export interface MarkersToLwcOptions {
  /** false = no text; true = full label; 'compact' = strategy slug only (mini grid). */
  showText?: boolean | 'compact';
}

/** Short strategy id for mini charts (tpc, chop, spot, …). */
export function compactMarkerLabel(marker: TradeMarker): string {
  const strat = String(marker.strategy || '').trim().toLowerCase();
  const aliases: Record<string, string> = {
    spot_accum_simple: 'spot',
    chop_grid: 'chop',
    trend_scalp: 'scalp',
  };
  if (aliases[strat]) return aliases[strat];
  if (strat) return strat.split('_')[0].slice(0, 8);
  const scope = String(marker.scope || '').toLowerCase();
  if (scope === 'multi_leg') return 'ml';
  if (scope === 'trend') return 'trend';
  if (scope === 'spot') return 'spot';
  return '';
}

export function markersToLwc(
  markers: TradeMarker[] | null | undefined,
  selectedId: string | null | undefined,
  options?: MarkersToLwcOptions,
): LwcSeriesMarker[] {
  const textMode = options?.showText ?? true;
  return (markers || []).map((m) => {
    const role = markerRole(m);
    const pending = (m.status || 'filled').toLowerCase() === 'pending';
    const selected = selectedId && m.id === selectedId;
    const strat = (m.strategy || m.scope || '').toLowerCase();
    const chopText = chopGridMarkerDisplayText(m, pending);
    const leg = (m.detail && (m.detail.leg_label || m.detail.leg_id)) || '';
    let legToken = '';
    if (leg) {
      const parts = String(leg).split('_').filter(Boolean);
      legToken = parts[parts.length - 1] || '';
      if (legToken.toLowerCase() === String(m.event || '').toLowerCase()) {
        legToken = parts[parts.length - 2] || '';
      }
    }
    const legTag = legToken ? `:${legToken}` : '';
    const baseText = chopText
      ? chopText
      : `${strat}:${m.event}${legTag}${pending ? ':pending' : ''}`;
    let aboveBar = role === 'exit' || role === 'tp';
    let position = aboveBar ? 'aboveBar' : 'belowBar';
    if (strat === 'chop_grid') {
      const regimeExit =
        role === 'exit' &&
        m.detail &&
        String(m.detail.exit_kind || '').toLowerCase() === 'regime_or_risk_exit';
      if (regimeExit) {
        position = 'inBar';
      } else {
        const legSide = chopGridLegSide(
          (m.detail && m.detail.leg_label) || (m.detail && m.detail.leg_id) || legToken,
        );
        // chop_grid stacks: S entry + L TP collide aboveBar; L entry + S TP
        // collide belowBar. Anchor filled entries on the leg's home side and
        // route TPs through inBar so they sit on the candle body instead of
        // piling onto the opposite-side entry.
        if (role === 'tp') {
          position = 'inBar';
        } else if (role === 'entry' && !pending) {
          if (legSide === 'short') position = 'aboveBar';
          else if (legSide === 'long') position = 'belowBar';
        } else {
          position = aboveBar ? 'aboveBar' : 'belowBar';
        }
      }
    }
    const isTp = role === 'tp';
    const highlightSelected = selected && !isTp;
    const label = highlightSelected ? `★ ${baseText}` : baseText;
    let text = '';
    if (textMode === true) text = label;
    else if (textMode === 'compact') text = compactMarkerLabel(m);
    return {
      time: m.time,
      position,
      color: highlightSelected ? '#ffff00' : markerColor(m),
      shape: markerShape(m),
      text,
      id: m.id,
    };
  });
}

function nearestCandleTime(times: number[], t: number): number {
  let best = times[0];
  let bestDist = Math.abs(best - t);
  for (let i = 1; i < times.length; i++) {
    const dist = Math.abs(times[i] - t);
    if (dist < bestDist) {
      best = times[i];
      bestDist = dist;
    }
  }
  return best;
}

function markerSourcePriority(markerId: string | null | undefined): number {
  const id = String(markerId || '');
  if (id.includes(':positions:')) return 3;
  if (id.includes(':position_operations:')) return 2;
  if (id.includes(':orders:')) return 1;
  return 0;
}

function chopGridLegDedupeToken(marker: TradeMarker): string {
  const strat = String(marker.strategy || '').toLowerCase();
  if (strat !== 'chop_grid') return '';
  const leg = String(marker.detail?.leg_label || marker.detail?.leg_id || '').trim();
  if (!leg) return '';
  const parts = leg.split('_').filter(Boolean);
  for (let i = parts.length - 1; i >= 0; i--) {
    const m = parts[i].toUpperCase().match(/^([LS]\d+)/);
    if (m) return m[1];
  }
  return '';
}

/** Dedupe key; chop_grid legs on the same 2h bar must not collapse (L2 vs L3). */
export function markerDedupeKey(marker: TradeMarker): string {
  const legToken = chopGridLegDedupeToken(marker);
  return [
    String(marker.strategy || '').toLowerCase(),
    String(marker.event || '').toLowerCase(),
    Number(marker.time),
    String(marker.side || '').toLowerCase(),
    String(marker.status || 'filled').toLowerCase(),
    legToken,
  ].join('|');
}

/** One marker per strategy/event/bar/leg — prefer positions over orders duplicates. */
export function dedupeMarkersForChart(markers: TradeMarker[]): TradeMarker[] {
  const byKey = new Map<string, TradeMarker>();
  for (const m of markers) {
    const key = markerDedupeKey(m);
    const prev = byKey.get(key);
    if (!prev || markerSourcePriority(m.id) > markerSourcePriority(prev.id)) {
      byKey.set(key, m);
    }
  }
  return [...byKey.values()].sort((a, b) => Number(a.time) - Number(b.time));
}

/** LWC requires marker time to match a candle time exactly (unix sec). */
export function snapMarkersToCandleTimes(
  markers: TradeMarker[],
  candles: Candle[],
): TradeMarker[] {
  const times = [...new Set(candles.map((c) => Number(c.time)).filter(Number.isFinite))].sort(
    (a, b) => a - b,
  );
  if (!times.length) return markers;
  const timeSet = new Set(times);
  const first = times[0];
  const last = times[times.length - 1];
  return markers.map((m) => {
    const t = Number(m.time);
    if (!Number.isFinite(t)) return m;
    const pending = (m.status || 'filled').toLowerCase() === 'pending';
    let snapped: number;
    if (pending) {
      if (t < first) snapped = first;
      else if (t > last) snapped = last;
      else snapped = nearestCandleTime(times, t);
    } else if (t < first) snapped = first;
    else if (t > last) snapped = last;
    else snapped = nearestCandleTime(times, t);
    if (snapped === t && timeSet.has(t)) return m;
    const detail = { ...(m.detail || {}) };
    if (snapped !== t && detail.order_time == null) {
      detail.order_time = t;
    }
    return { ...m, time: snapped, detail };
  });
}

export function prepareChartMarkers(
  raw: TradeMarker[] | null | undefined,
  candles: Candle[],
  overlays: FeatureOverlays | null | undefined,
  layers: { trend: boolean; spot: boolean; multiLeg: boolean; pending?: boolean },
  strategyFocus: string,
): TradeMarker[] {
  let incoming = (raw || []).filter((m) => !isFeatureBusRegimeExitMarker(m));
  if (layers.pending === false) {
    incoming = incoming.filter(
      (m) => String(m.status || 'filled').toLowerCase() !== 'pending',
    );
  }
  const focus = String(strategyFocus || '')
    .trim()
    .toLowerCase();
  const chopFocus = !focus || focus === 'chop_grid';
  if (chopFocus && candles.length && overlays) {
    incoming = mergeRegimeExitMarkers(
      incoming,
      synthesizeChopRegimeExitMarkers(candles, overlays),
    );
  }
  const scopes = new Set<string>();
  if (layers.trend) scopes.add('trend');
  if (layers.spot) scopes.add('spot');
  if (layers.multiLeg) scopes.add('multi_leg');
  incoming = incoming.filter((m) => scopes.has(String(m.scope || '').toLowerCase()));
  if (focus) {
    incoming = incoming.filter((m) => String(m.strategy || '').toLowerCase() === focus);
  }
  if (candles.length) {
    incoming = snapMarkersToCandleTimes(incoming, candles);
    incoming = dedupeMarkersForChart(incoming);
  }
  return incoming;
}

export function scrollIndexForTime(candles: Candle[], targetTime: number): number {
  const t = Number(targetTime);
  if (!Array.isArray(candles) || !candles.length || !Number.isFinite(t)) {
    return -1;
  }
  let idx = candles.findIndex((c) => Number(c.time) >= t);
  if (idx < 0) idx = candles.length - 1;
  return idx;
}
