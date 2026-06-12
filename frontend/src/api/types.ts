/** API response envelope from mlbot_console.responses.ok() */
export interface ApiEnvelope<T> {
  ok: boolean;
  data: T;
  meta?: Record<string, unknown>;
  error?: { message?: string };
}

export interface NavLink {
  id: string;
  label: string;
  url: string;
}

export interface SymbolRow {
  symbol: string;
  latest?: Record<string, unknown>;
}

export interface TradeMarker {
  id: string;
  time: number;
  symbol: string;
  scope: string;
  strategy: string;
  event: string;
  side: string;
  price?: number | null;
  qty?: number | null;
  pnl_usdt?: number | null;
  is_add?: boolean;
  status?: string;
  color?: string;
  detail?: Record<string, unknown>;
}

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TradeLink {
  strategy?: string;
  scope?: string;
  symbol?: string;
  leg?: string;
  status?: string;
  exit_kind?: string;
  entry_time: number;
  entry_price: number;
  exit_time: number;
  exit_price: number;
  entry_marker_id?: string;
  exit_marker_id?: string;
  side?: string;
  pnl_usdt?: number | null;
  color?: string;
}

export interface BundleData {
  ohlcv: {
    candles: Candle[];
    source?: string;
    range_start?: string;
    range_end?: string;
    last_candle_time?: number;
    degraded_ohlc?: boolean;
  };
  markers: TradeMarker[];
  trade_links: TradeLink[];
  overlays: Record<string, FeatureOverlaySpec>;
  main_overlays: Record<string, MainOverlaySpec>;
  chop_grid_overlay: { batches?: unknown[]; error?: string };
  chop_regime_regions: unknown[];
  strategy_stage_regions: Record<string, unknown>;
}

export interface FeatureOverlaySpec {
  available?: boolean;
  points?: { time: number; value: number | null }[];
  reference_lines?: { y?: number; value?: number; operator?: string }[];
  error?: string;
}

export interface MainOverlaySpec {
  available?: boolean;
  key?: string;
  source?: string;
  points?: { time: number; value: number }[];
  error?: string;
}

export interface OrderRow {
  order_id: string;
  symbol: string;
  scope: string;
  strategy?: string;
  side?: string;
  status?: string;
  order_type?: string;
  quantity?: number;
  filled_quantity?: number;
  price?: number;
  average_price?: number;
  time?: number | string;
  created_at?: string;
  filled_at?: string;
  purpose?: string;
  leg_label?: string;
  grid_batch?: string;
  marker_id?: string;
  pnl_usdt?: number;
  take_profit_price?: number;
  stop_loss_price?: number;
  [key: string]: unknown;
}

export interface FunnelStrategyStats {
  regime_passed?: number;
  regime_denied?: number;
  prefilter_passed?: number;
  prefilter_denied?: number;
  direction?: number;
  gate_passed?: number;
}

export interface FunnelSnapshot {
  timestamp?: string;
  symbol?: string;
  by_strategy?: Record<string, FunnelStrategyStats>;
}

export interface SignalRow {
  symbol: string;
  map_href?: string;
  bars_1min_rows?: number;
  latest_bar?: { timestamp?: string };
  strategies?: Record<
    string,
    {
      summary?: string;
      last_summary?: string;
      by_strategy?: Record<
        string,
        { summary?: string; last_summary?: string; funnel_summary?: string; blockers?: string[] }
      >;
    }
  >;
}

export interface MonitoringDashboard {
  summary?: {
    any_alert?: boolean;
    any_missed?: boolean;
    any_uncalibrated?: boolean;
  };
  index_updated_at?: string;
  cards?: MonitoringCard[];
  strategy_alerts?: Record<string, MonitoringIssueRow[]>;
  strategy_uncalibrated?: Record<string, MonitoringIssueRow[]>;
}

export interface MonitoringIssueRow {
  source?: string;
  strategy?: string;
  messages?: string[];
}

export interface MonitoringCard {
  cadence: string;
  display_status?: string;
  run_ts?: string;
  last_run_at?: string | null;
  valid_until_at?: string | null;
  next_run_at?: string | null;
  timer_calendar?: string;
  age_hours?: number | null;
  max_age_hours?: number;
  output_dir?: string;
  watchdog_any_alert?: boolean | null;
  drift_any_alert?: boolean | null;
  drift_no_plateaus?: boolean;
  alert_details?: string[];
  uncalibrated_details?: string[];
}

export interface RegimeOpsRow {
  account_layer?: string;
  account_layer_title?: string;
  strategy: string;
  present?: boolean;
  regime_source?: string;
  regime_path?: string;
  n_rules?: number;
  allowed_sides?: string[];
  last_calibration?: Record<string, unknown>;
  drift_status?: string;
  drift_detail?: string;
  drift_checked_at?: string;
  config_reference_at?: string;
}

export interface DailyPnlPoint {
  date?: string;
  week_start?: string;
  label?: string;
  pnl?: number | null;
  cumulative?: number | null;
}

export interface AccountScopeBlock {
  scope?: string;
  label?: string;
  realized_pnl?: number | null;
  unrealized_pnl?: number | null;
  closed_trades?: number | null;
  open_positions?: number | null;
  exchange?: Record<string, unknown>;
  daily_realized?: DailyPnlPoint[];
}

export interface AccountStrategyRow {
  scope?: string;
  scope_label?: string;
  strategy?: string;
  strategy_title?: string;
  realized_pnl?: number | null;
  unrealized_pnl?: number | null;
  closed_trades?: number | null;
  open_positions?: number | null;
}

export interface AccountSummary {
  symbol?: string;
  totals?: Record<string, number | null>;
  recent_realized?: Record<string, unknown>;
  exchange_ledger?: {
    totals?: Record<string, number | null>;
    accounts?: Array<Record<string, unknown>>;
  };
  ledger?: { totals?: Record<string, number | null> };
  scopes?: AccountScopeBlock[];
  strategies?: AccountStrategyRow[];
  daily_realized?: DailyPnlPoint[];
  weekly_realized?: DailyPnlPoint[];
  cumulative_realized?: DailyPnlPoint[];
  notes?: string[];
  by_scope?: unknown[];
  reconciliation?: unknown;
}

export interface AccountReconIssue {
  kind?: string;
  scope?: string;
  layer?: string;
  message?: string;
  [key: string]: unknown;
}

export interface AccountReconScopeBlock {
  scope?: string;
  ok?: boolean;
  issues?: AccountReconIssue[];
  local?: Record<string, number | null>;
  exchange?: Record<string, number | null>;
  local_snapshot?: Record<string, unknown>;
  exchange_snapshot?: Record<string, unknown>;
  error?: string;
}

export interface AccountReconciliationAll {
  ok?: boolean;
  symbol?: string;
  lookback_days?: number;
  issues?: AccountReconIssue[];
  engine?: Record<string, AccountReconScopeBlock>;
  pnl?: {
    ok?: boolean;
    scopes?: Record<string, AccountReconScopeBlock>;
    totals?: Record<string, number | null>;
    fetched_at?: string;
  };
}
