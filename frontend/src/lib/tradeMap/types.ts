/** OHLCV bar (unix seconds). */
export interface Candle {
  time: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number | null;
  symbol?: string;
}

export interface TradeMarkerDetail {
  exit_kind?: string;
  exit_reason?: string;
  chop?: number | null;
  entry_min?: number;
  exit_below?: number;
  source?: string;
  leg_label?: string;
  leg_id?: string;
  [key: string]: unknown;
}

export interface TradeMarker {
  id?: string;
  time: number;
  symbol?: string;
  scope?: string;
  strategy?: string;
  event?: string;
  side?: string;
  status?: string;
  color?: string;
  pnl_usdt?: number | null;
  is_add?: boolean;
  detail?: TradeMarkerDetail;
}

export type AccountLayerId = 'trend' | 'spot' | 'multi_leg' | 'shared' | string;

export type LayerKey = 'trend' | 'spot' | 'multiLeg';

export interface AccountLayerMeta {
  id: AccountLayerId;
  title: string;
  layerKey: LayerKey | null;
}

/** UI layer toggles; `false` disables the layer. */
export interface LayerVisibility {
  trend?: boolean;
  spot?: boolean;
  multiLeg?: boolean;
}

export interface FeatureIndexEntry {
  column: string;
  strategy: string;
  strategy_title?: string;
  account_layer: AccountLayerId;
  account_layer_title?: string;
  stage: string;
  stage_title?: string;
}

export interface StrategyRecord {
  id: string;
  account_layer: AccountLayerId;
  account_layer_title?: string;
  title?: string;
  stages?: Record<string, string[]>;
}

export interface FeatureTaxonomy {
  strategies?: StrategyRecord[];
  live_strategies?: StrategyRecord[];
  live_strategy_ids?: string[];
  index?: Record<string, FeatureIndexEntry[]>;
  stage_order?: string[];
  stage_labels?: Record<string, string>;
  account_layer_labels?: Record<string, string>;
  constitution_source?: string;
}

export interface FeatureMeta {
  column: string;
  account_layer: AccountLayerId;
  account_layer_title?: string;
  strategy: string;
  strategy_title?: string;
  stage: string;
  stage_title?: string;
}

export interface InferredStrategy {
  strategy: string;
  account_layer: AccountLayerId;
}

export interface StrategyMeta {
  id: string;
  title: string;
  layerKey: LayerKey | null;
  account_layer: AccountLayerId;
}

export interface OverlayPoint {
  time: number;
  value?: number | null;
}

export interface ReferenceLine {
  y?: number;
  value?: number;
  operator?: string;
  label?: string;
}

export interface FeatureOverlay {
  available?: boolean;
  column?: string;
  points?: OverlayPoint[];
  latest?: number | null;
  reference_y?: number | null;
  reference_lines?: ReferenceLine[];
  semantic_hint?: string;
  path?: string;
}

export type FeatureOverlays = Record<string, FeatureOverlay | undefined>;

export interface MetricsCell {
  value: string;
  pass: boolean | null;
}

export interface MetricsColumnSpec {
  kind: 'scalar' | 'regime_box';
  column?: string;
  columns?: string[];
  header: string;
  threshold: string;
}

export interface MetricsRowSpec {
  kind: 'scalar' | 'threshold_row';
  column?: string;
  regimeCols?: string[];
  yaml?: string;
  label: string;
  threshold: string;
}

export interface ThresholdMetricRow {
  yaml: string;
  label: string;
  value: string | null;
  threshold: string;
  pass: boolean | null;
}

export interface StagePanePlan {
  chartCols: string[];
  statusCols: string[];
  skipStage: boolean;
}

export interface FeatureGroupMeta {
  layer: AccountLayerId;
  strategy: string;
  stage: string;
}

export type FeatureGroupTuple = [string, string[], FeatureGroupMeta];

export interface StrategyListEntry {
  id: string;
  account_layer: AccountLayerId;
  account_layer_title: string;
  title: string;
  stages: Record<string, string[]>;
}

export type FeaturePaneHeaderKind = 'layer' | 'strategy' | 'stage';

export type FeaturePaneItem =
  | { type: 'gap'; id: string }
  | {
      type: 'header';
      strategy: string;
      title: string;
      headerKind: FeaturePaneHeaderKind;
      accountLayer?: AccountLayerId;
      stage?: string;
    }
  | {
      type: 'metrics_table';
      id: string;
      strategy: string;
      accountLayer: AccountLayerId;
      columns: string[];
    }
  | {
      type: 'threshold_status';
      id: string;
      strategy: string;
      accountLayer: AccountLayerId;
      stage: string;
      columns: string[];
    }
  | {
      type: 'feature';
      column: string;
      strategy: string;
      accountLayer: AccountLayerId;
      stage: string;
    };

/** layer → strategy → stage → columns */
export type FeatureTaxonomyTree = Record<
  string,
  Record<string, Record<string, string[]>>
>;

export interface LogicalRange {
  from: number;
  to: number;
}

export interface PriceRange {
  minValue: number;
  maxValue: number;
}

export interface LwcSeriesMarker {
  time: number;
  position: string;
  color: string;
  shape: string;
  text: string;
  id?: string;
}

export interface ChopRegimeRegion {
  start: number;
  end: number;
}

export interface EligibilityInfo {
  can_buy?: boolean;
  weekly_ema_200_position?: number | string | null;
  blockers?: string[];
}

export interface NavLink {
  id?: string;
  url?: string;
}
