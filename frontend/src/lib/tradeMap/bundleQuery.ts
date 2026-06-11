import { apiQuery } from '@/api/client.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';
import { isoFromUnixSec, ohlcvInitialQueryRange } from './ohlcv.ts';
import {
  featureColumnsParam,
  mainOverlaysQueryParam,
  stageRegionsQueryParam,
} from './misc.ts';

export interface BundleQueryState {
  symbol: string;
  timeframe: string;
  layers: LayerState;
  ohlcvLoadedFrom: string | null;
  ohlcvLoadedTo: string | null;
  markerQueryFromIso: string | null;
  lastMarkerPollSince: string | null;
  selectedFeatureColumns: string[];
  mainEma1200: boolean;
  mainWeeklyEma200: boolean;
  featureStrategyFocus: string;
  lastCandles: Array<{ time: number }>;
}

export function markerRangeFrom(state: BundleQueryState): {
  from?: string;
  to?: string;
} {
  const out: Record<string, string> = {};
  if (state.ohlcvLoadedFrom) out.from = state.ohlcvLoadedFrom;
  if (state.ohlcvLoadedTo) out.to = state.ohlcvLoadedTo;
  return out;
}

function scopesFromLayers(layers: LayerState): string {
  const parts: string[] = [];
  if (layers.trend) parts.push('trend');
  if (layers.spot) parts.push('spot');
  if (layers.multiLeg) parts.push('multi_leg');
  return parts.join(',') || 'trend,spot';
}

export function buildMarkersOnlyQuery(state: BundleQueryState): string {
  const init = ohlcvInitialQueryRange(state.timeframe);
  const from = state.markerQueryFromIso || state.ohlcvLoadedFrom || init.from;
  return apiQuery({
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    include_ohlcv: 'none',
    include_features: 'false',
    include_markers: 'true',
    include_trade_links: 'true',
    include_chop: 'false',
    from,
    to: state.ohlcvLoadedTo || new Date().toISOString(),
    full_range: 'false',
  });
}

export function buildPollQuery(state: BundleQueryState): string {
  const init = ohlcvInitialQueryRange(state.timeframe);
  const range = markerRangeFrom(state);
  const from = range.from || state.markerQueryFromIso || init.from;
  const candles = state.lastCandles;
  const tailAnchor = candles.length
    ? candles[Math.max(0, candles.length - 5)]
    : null;
  return apiQuery({
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    from,
    to: range.to || new Date().toISOString(),
    since: state.lastMarkerPollSince || undefined,
    include_ohlcv: 'tail',
    ohlcv_from: tailAnchor ? isoFromUnixSec(Number(tailAnchor.time)) : undefined,
    include_features: 'false',
    include_markers: 'true',
    include_trade_links: 'true',
    include_chop: 'false',
    full_range: 'false',
  });
}

export function buildFullShellQuery(state: BundleQueryState): string {
  const init = ohlcvInitialQueryRange(state.timeframe);
  const mainOl = mainOverlaysQueryParam(state.mainEma1200, state.mainWeeklyEma200);
  return apiQuery({
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    include_ohlcv: 'full',
    include_features: 'false',
    include_markers: 'false',
    include_trade_links: 'false',
    include_chop: 'false',
    from: state.ohlcvLoadedFrom || init.from,
    to: state.ohlcvLoadedTo || init.to,
    full_range: state.ohlcvLoadedFrom ? 'false' : init.full_range || 'false',
    main_overlays: mainOl || undefined,
  });
}

export function buildFullMarkersQuery(
  state: BundleQueryState,
  markerFrom: string | undefined,
): string {
  const range = markerRangeFrom(state);
  const init = ohlcvInitialQueryRange(state.timeframe);
  return apiQuery({
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    from: range.from || markerFrom || state.ohlcvLoadedFrom || init.from,
    to: range.to || state.ohlcvLoadedTo || new Date().toISOString(),
    include_ohlcv: 'none',
    include_features: 'false',
    include_markers: 'true',
    include_trade_links: 'true',
    include_chop: 'false',
  });
}

export function buildFullFeaturesQuery(
  state: BundleQueryState,
  markerFrom: string | undefined,
): string {
  const range = markerRangeFrom(state);
  const init = ohlcvInitialQueryRange(state.timeframe);
  const featParam = featureColumnsParam(state.selectedFeatureColumns);
  const stageRg = stageRegionsQueryParam(state.layers.prefilter, state.layers.gate);
  const stratFocus = state.featureStrategyFocus.trim();
  return apiQuery({
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    from: range.from || markerFrom || state.ohlcvLoadedFrom || init.from,
    to: range.to || state.ohlcvLoadedTo || new Date().toISOString(),
    include_ohlcv: 'none',
    include_features: 'true',
    include_markers: 'false',
    include_trade_links: 'false',
    include_chop: 'true',
    feature_columns: featParam || undefined,
    stage_regions: stageRg || undefined,
    strategy: stratFocus || undefined,
  });
}

export function buildMiniGridQuery(
  symbol: string,
  timeframe: string,
  layers: LayerState,
  range: { from: string; to: string; full_range: string },
): string {
  return apiQuery({
    symbol,
    timeframe,
    scopes: scopesFromLayers(layers),
    include_pending: String(layers.pending),
    from: range.from,
    to: range.to,
    full_range: range.full_range,
    include_ohlcv: 'full',
    include_features: 'false',
    include_markers: 'true',
    include_trade_links: 'true',
    include_chop: 'false',
  });
}

export function buildGridPollQuery(
  symbol: string,
  timeframe: string,
  layers: LayerState,
  range: { from: string; to: string },
  candles: Array<{ time: number }>,
  lastMarkerPollSince: string | null,
): string {
  const tailAnchor = candles.length ? candles[Math.max(0, candles.length - 5)] : null;
  return apiQuery({
    symbol,
    timeframe,
    scopes: scopesFromLayers(layers),
    include_pending: String(layers.pending),
    from: range.from,
    to: range.to,
    since: lastMarkerPollSince || undefined,
    include_ohlcv: 'tail',
    ohlcv_from: tailAnchor ? isoFromUnixSec(Number(tailAnchor.time)) : undefined,
    include_features: 'false',
    include_markers: 'true',
    include_trade_links: 'true',
    include_chop: 'false',
    full_range: 'false',
  });
}
