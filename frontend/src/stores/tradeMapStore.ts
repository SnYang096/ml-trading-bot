import { create } from 'zustand';
import type { BundleData, Candle, MainOverlaySpec, TradeLink, TradeMarker } from '@/api/types.ts';
import type { FeatureOverlays, LogicalRange } from '@/lib/tradeMap/types.ts';

export const POLL_MS = 10_000;
export const LAYOUT_KEY = 'mlbot_trade_map_layout_v2';

export interface LayerState {
  trend: boolean;
  spot: boolean;
  multiLeg: boolean;
  pending: boolean;
  chopGrid: boolean;
  prefilter: boolean;
  gate: boolean;
}

interface TradeMapState {
  symbol: string;
  timeframe: string;
  layers: LayerState;
  selectedFeatureColumns: string[];
  availableFeatureColumns: string[];
  featureStrategyFocus: string;
  featureSearchQuery: string;
  featureDrawerOpen: boolean;
  paneVolume: boolean;
  ordersDockOpen: boolean;
  lastCandles: Candle[];
  lastOverlays: FeatureOverlays;
  lastMainOverlays: Record<string, MainOverlaySpec>;
  lastChopMapData: BundleData['chop_grid_overlay'] | null;
  chopRegimeRegions: unknown[];
  strategyStageRegions: Record<string, unknown>;
  markers: TradeMarker[];
  lastTradeLinks: TradeLink[];
  markerQueryFromIso: string | null;
  historyExhausted: boolean;
  selectedMarkerId: string | null;
  highlightBarTime: number | null;
  ohlcvLoadedFrom: string | null;
  ohlcvLoadedTo: string | null;
  lastMarkerPollSince: string | null;
  statusText: string;
  loading: boolean;
  historyLoading: boolean;
  chartFitPending: boolean;
  /** Set by history prepend; applied in main chart after setData, then cleared. */
  historyScrollAdjust: LogicalRange | null;
  mainEma1200: boolean;
  mainWeeklyEma200: boolean;
  setSymbol: (s: string) => void;
  setTimeframe: (tf: string) => void;
  setLayers: (l: Partial<LayerState>) => void;
  setSelectedFeatureColumns: (cols: string[]) => void;
  setAvailableFeatureColumns: (cols: string[]) => void;
  setFeatureStrategyFocus: (s: string) => void;
  setFeatureSearchQuery: (s: string) => void;
  setFeatureDrawerOpen: (v: boolean) => void;
  setPaneVolume: (v: boolean) => void;
  setOrdersDockOpen: (v: boolean) => void;
  setLastCandles: (c: Candle[]) => void;
  setBundlePhase: (payload: Partial<TradeMapState>) => void;
  setSelectedMarkerId: (id: string | null) => void;
  setHighlightBarTime: (t: number | null) => void;
  setStatusText: (s: string) => void;
  setLoading: (v: boolean) => void;
}

const defaultLayers: LayerState = {
  trend: true,
  spot: true,
  multiLeg: true,
  pending: false,
  chopGrid: true,
  prefilter: true,
  gate: false,
};

export const useTradeMapStore = create<TradeMapState>((set) => ({
  symbol: 'ETHUSDT',
  timeframe: '2h',
  layers: defaultLayers,
  selectedFeatureColumns: [],
  availableFeatureColumns: [],
  featureStrategyFocus: '',
  featureSearchQuery: '',
  featureDrawerOpen: false,
  paneVolume: true,
  ordersDockOpen: true,
  lastCandles: [],
  lastOverlays: {},
  lastMainOverlays: {},
  lastChopMapData: null,
  chopRegimeRegions: [],
  strategyStageRegions: {},
  markers: [],
  lastTradeLinks: [],
  markerQueryFromIso: null,
  historyExhausted: false,
  selectedMarkerId: null,
  highlightBarTime: null,
  ohlcvLoadedFrom: null,
  ohlcvLoadedTo: null,
  lastMarkerPollSince: null,
  statusText: '',
  loading: false,
  historyLoading: false,
  chartFitPending: true,
  historyScrollAdjust: null,
  mainEma1200: true,
  mainWeeklyEma200: true,
  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setLayers: (l) => set((s) => ({ layers: { ...s.layers, ...l } })),
  setSelectedFeatureColumns: (selectedFeatureColumns) => set({ selectedFeatureColumns }),
  setAvailableFeatureColumns: (availableFeatureColumns) => set({ availableFeatureColumns }),
  setFeatureStrategyFocus: (featureStrategyFocus) => set({ featureStrategyFocus }),
  setFeatureSearchQuery: (featureSearchQuery) => set({ featureSearchQuery }),
  setFeatureDrawerOpen: (featureDrawerOpen) => set({ featureDrawerOpen }),
  setPaneVolume: (paneVolume) => set({ paneVolume }),
  setOrdersDockOpen: (ordersDockOpen) => set({ ordersDockOpen }),
  setLastCandles: (lastCandles) => set({ lastCandles }),
  setBundlePhase: (payload) => set(payload),
  setSelectedMarkerId: (selectedMarkerId) => set({ selectedMarkerId }),
  setHighlightBarTime: (highlightBarTime) => set({ highlightBarTime }),
  setStatusText: (statusText) => set({ statusText }),
  setLoading: (loading) => set({ loading }),
}));

export function scopesFromLayers(layers: LayerState): string {
  const parts: string[] = [];
  if (layers.trend) parts.push('trend');
  if (layers.spot) parts.push('spot');
  if (layers.multiLeg) parts.push('multi_leg');
  return parts.join(',') || 'trend,spot';
}

export function loadLayout(): Record<string, unknown> | null {
  try {
    return JSON.parse(localStorage.getItem(LAYOUT_KEY) || 'null') as Record<string, unknown> | null;
  } catch {
    return null;
  }
}

export function resetHistoryState(): void {
  useTradeMapStore.setState({
    lastCandles: [],
    markers: [],
    lastOverlays: {},
    lastMainOverlays: {},
    lastChopMapData: null,
    chopRegimeRegions: [],
    strategyStageRegions: {},
    lastTradeLinks: [],
    ohlcvLoadedFrom: null,
    ohlcvLoadedTo: null,
    markerQueryFromIso: null,
    lastMarkerPollSince: null,
    selectedMarkerId: null,
    highlightBarTime: null,
    statusText: '',
    loading: false,
    historyLoading: false,
    historyExhausted: false,
    chartFitPending: true,
    historyScrollAdjust: null,
  });
}

export function saveLayout(state: {
  layers: LayerState;
  selectedFeatureColumns: string[];
  featureStrategyFocus: string;
  mainEma1200: boolean;
  mainWeeklyEma200: boolean;
  paneVolume?: boolean;
  ordersDockOpen?: boolean;
}): void {
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(state));
}
