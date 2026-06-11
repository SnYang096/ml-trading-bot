import type { AccountLayerId, AccountLayerMeta, StrategyRecord } from './types.ts';

export const ENTRY_SHAPES: Record<string, string> = {
  long: 'arrowUp',
  short: 'arrowDown',
};
export const EXIT_SHAPE = 'circle';
/** Filled take-profit markers (chop_grid legs); pending TP stays gray circle. */
export const TP_MARKER_COLOR = '#ffb000';
export const TP_MARKER_SHAPE = 'square';
export const SUBCHART_COLORS = [
  '#00ff41',
  '#00ffff',
  '#ffb000',
  '#ff0040',
  '#bf00ff',
  '#ffff00',
];

/** Lightweight Charts theme (hacker / terminal). */
export const CHART_THEME = {
  bg: '#020402',
  text: '#7ae87a',
  grid: '#143814',
  border: '#2a7a2a',
  candleUp: '#00ff41',
  candleDown: '#ff3366',
  emaPrimary: '#5cffff',
  emaSecondary: '#ffcc44',
  volume: '#2a7a2a',
  linkFallback: '#7ae87a',
  accentPurple: '#d966ff',
} as const;

export const DEFAULT_VISIBLE_BARS = 320;
/** Max metric-matrix columns; wider logical windows keep the tail (latest bars). */
export const METRICS_TABLE_MAX_COLS = 80;
export const MAX_FEATURE_SUBCHARTS = 8;
export const CHOP_REGIME_FILL = 'rgba(0, 255, 65, 0.12)';
export const PREFILTER_STAGE_FILL = 'rgba(255, 0, 64, 0.12)';
export const GATE_STAGE_FILL = 'rgba(191, 0, 255, 0.1)';
export const FEATURE_PRESETS: Record<string, string[]> = {
  default: ['weekly_ema_200_position', 'ema_1200_position'],
  trend: ['ema_1200_position', 'tpc_pullback_depth', 'tpc_semantic_chop', 'bpc_pullback_depth'],
  spot: ['weekly_ema_200_position'],
  multi_leg: ['bpc_semantic_chop', 'box_pos_60', 'box_stability_60'],
};

/** Metrics-table row columns when taxonomy / bus catalog is not yet loaded. */
export const STRATEGY_METRICS_FALLBACK: Record<string, string[]> = {
  tpc: ['ema_1200_position', 'tpc_pullback_depth', 'tpc_semantic_chop'],
  bpc: ['ema_1200_position', 'bpc_pullback_depth', 'bpc_semantic_chop'],
  me: ['ema_1200_position'],
  srb: ['ema_1200_position'],
  spot_accum_simple: ['weekly_ema_200_position'],
  chop_grid: ['bpc_semantic_chop', 'box_pos_60'],
  trend_scalp: ['trend_confidence', 'bpc_semantic_chop'],
};

export const ACCOUNT_LAYER_ORDER: AccountLayerId[] = [
  'trend',
  'spot',
  'multi_leg',
  'shared',
];

export const STAGE_ORDER: string[] = [
  'regime',
  'prefilter',
  'direction',
  'gate',
  'entry',
  'evidence',
  'execution',
];

export const ACCOUNT_LAYER_META: Record<string, AccountLayerMeta> = {
  trend: { id: 'trend', title: 'B·Trend', layerKey: 'trend' },
  spot: { id: 'spot', title: 'A·Spot', layerKey: 'spot' },
  multi_leg: { id: 'multi_leg', title: 'C·Multi-leg', layerKey: 'multiLeg' },
  shared: { id: 'shared', title: '未归类', layerKey: null },
};

/** Fallback when taxonomy YAML is missing on disk (matches CONSOLE_STRATEGIES). */
export const KNOWN_STRATEGIES: StrategyRecord[] = [
  { id: 'tpc', account_layer: 'trend', title: 'TPC' },
  { id: 'bpc', account_layer: 'trend', title: 'BPC' },
  { id: 'me', account_layer: 'trend', title: 'ME' },
  { id: 'srb', account_layer: 'trend', title: 'SRB' },
  { id: 'spot_accum_simple', account_layer: 'spot', title: 'spot_accum_simple' },
  { id: 'chop_grid', account_layer: 'multi_leg', title: 'Chop Grid' },
  { id: 'trend_scalp', account_layer: 'multi_leg', title: 'Trend Scalp' },
];
