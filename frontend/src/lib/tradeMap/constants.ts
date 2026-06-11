import type { AccountLayerId, AccountLayerMeta, StrategyRecord } from './types.ts';

export const ENTRY_SHAPES: Record<string, string> = {
  long: 'arrowUp',
  short: 'arrowDown',
};
export const EXIT_SHAPE = 'circle';
/** Filled take-profit markers (chop_grid legs); pending TP stays gray circle. */
export const TP_MARKER_COLOR = '#E8B923';
export const TP_MARKER_SHAPE = 'square';
export const SUBCHART_COLORS = [
  '#ffeb3b',
  '#58a6ff',
  '#f78166',
  '#7ee787',
  '#d2a8ff',
  '#ffa657',
];
export const DEFAULT_VISIBLE_BARS = 320;
export const FEATURE_PRESETS: Record<string, string[]> = {
  default: ['weekly_ema_200_position', 'ema_1200_position'],
  trend: ['ema_1200_position', 'tpc_pullback_depth', 'tpc_semantic_chop', 'bpc_pullback_depth'],
  spot: ['weekly_ema_200_position'],
  multi_leg: ['bpc_semantic_chop', 'box_pos_60', 'box_stability_60'],
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
