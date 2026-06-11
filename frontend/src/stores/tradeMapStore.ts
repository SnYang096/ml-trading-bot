import { create } from 'zustand';
import type { BundleData, Candle, TradeMarker } from '@/api/types.ts';

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
  featureStrategyFocus: string;
  lastCandles: Candle[];
  lastOverlays: Record<string, unknown>;
  lastChopMapData: BundleData['chop_grid_overlay'] | null;
  chopRegimeRegions: unknown[];
  strategyStageRegions: Record<string, unknown>;
  markers: TradeMarker[];
  selectedMarkerId: string | null;
  ohlcvLoadedFrom: string | null;
  ohlcvLoadedTo: string | null;
  lastMarkerPollSince: string | null;
  statusText: string;
  loading: boolean;
  chartFitPending: boolean;
  mainEma1200: boolean;
  mainWeeklyEma200: boolean;
  setSymbol: (s: string) => void;
  setTimeframe: (tf: string) => void;
  setLayers: (l: Partial<LayerState>) => void;
  setSelectedFeatureColumns: (cols: string[]) => void;
  setFeatureStrategyFocus: (s: string) => void;
  setLastCandles: (c: Candle[]) => void;
  setBundlePhase: (payload: Partial<TradeMapState>) => void;
  setSelectedMarkerId: (id: string | null) => void;
  setStatusText: (s: string) => void;
  setLoading: (v: boolean) => void;
}

const defaultLayers: LayerState = {
  trend: true,
  spot: true,
  multiLeg: true,
  pending: false,
  chopGrid: true,
  prefilter: false,
  gate: false,
};

export const useTradeMapStore = create<TradeMapState>((set) => ({
  symbol: 'ETHUSDT',
  timeframe: '2h',
  layers: defaultLayers,
  selectedFeatureColumns: [],
  featureStrategyFocus: '',
  lastCandles: [],
  lastOverlays: {},
  lastChopMapData: null,
  chopRegimeRegions: [],
  strategyStageRegions: {},
  markers: [],
  selectedMarkerId: null,
  ohlcvLoadedFrom: null,
  ohlcvLoadedTo: null,
  lastMarkerPollSince: null,
  statusText: '',
  loading: false,
  chartFitPending: true,
  mainEma1200: false,
  mainWeeklyEma200: false,
  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setLayers: (l) => set((s) => ({ layers: { ...s.layers, ...l } })),
  setSelectedFeatureColumns: (selectedFeatureColumns) => set({ selectedFeatureColumns }),
  setFeatureStrategyFocus: (featureStrategyFocus) => set({ featureStrategyFocus }),
  setLastCandles: (lastCandles) => set({ lastCandles }),
  setBundlePhase: (payload) => set(payload),
  setSelectedMarkerId: (selectedMarkerId) => set({ selectedMarkerId }),
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

export function saveLayout(state: {
  layers: LayerState;
  selectedFeatureColumns: string[];
  featureStrategyFocus: string;
  mainEma1200: boolean;
  mainWeeklyEma200: boolean;
}): void {
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(state));
}
