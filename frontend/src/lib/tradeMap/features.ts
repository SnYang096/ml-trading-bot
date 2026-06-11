import {
  ACCOUNT_LAYER_META,
  ACCOUNT_LAYER_ORDER,
  KNOWN_STRATEGIES,
  STAGE_ORDER,
} from './constants.ts';
import { overlayAsOfAtCandleTimes } from './ohlcv.ts';
import type {
  Candle,
  FeatureGroupTuple,
  FeatureMeta,
  FeatureOverlays,
  FeaturePaneItem,
  FeatureTaxonomy,
  FeatureTaxonomyTree,
  InferredStrategy,
  LayerVisibility,
  MetricsCell,
  MetricsColumnSpec,
  MetricsRowSpec,
  ReferenceLine,
  StagePanePlan,
  StrategyListEntry,
  StrategyMeta,
  StrategyRecord,
  ThresholdMetricRow,
  FeatureOverlay,
} from './types.ts';

function isTrendScalpOtherPaneColumn(column: string, meta: FeatureMeta): boolean {
  const stage = meta.stage || 'other';
  if (stage !== 'other') return false;
  const strat = meta.strategy || inferStrategyIdFromColumn(column).strategy;
  if (strat !== 'trend_scalp') return false;
  const lc = String(column || '').toLowerCase();
  if (
    lc.includes('semantic_chop') ||
    lc.startsWith('box_') ||
    lc.includes('grid') ||
    lc.includes('scalp')
  ) {
    return false;
  }
  return true;
}

export function filterSubchartColumns(
  columns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
  strategyFocus: string | null | undefined,
): string[] {
  const focus = strategyFocus ? String(strategyFocus).trim() : '';
  return (columns || []).filter((col) => {
    const meta = lookupFeatureMeta(col);
    if (!isLayerEnabled(meta.account_layer, layers)) return false;
    if (focus && meta.strategy !== focus) return false;
    if (isTrendScalpOtherPaneColumn(col, meta) && focus !== 'trend_scalp') {
      return false;
    }
    return true;
  });
}

/**
 * Columns to render in feature subcharts: honor strategy focus, fall back to preset
 * when the current selection belongs to another strategy.
 */
export function resolveSubchartColumns(
  selectedColumns: string[] | null | undefined,
  availableColumns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
  strategyFocus: string | null | undefined,
  maxCols: number,
): string[] {
  const max = Math.max(1, Number(maxCols) || 6);
  let selected = (selectedColumns || []).slice(0, max);
  const focus = strategyFocus ? String(strategyFocus).trim() : '';
  if (focus) {
    selected = subchartColumnsForStrategy(focus, selected, focus);
  }
  const filtered = filterSubchartColumns(selected, layers, focus);
  if (!focus) {
    return filtered.length
      ? filtered
      : filterSubchartColumns(selected, layers, null).slice(0, max);
  }
  const hasForeign = selected.some(
    (col) => lookupFeatureMeta(col).strategy !== focus,
  );
  if (!hasForeign && filtered.length) return filtered;
  const pool = filterColumnsForFeaturePicker(availableColumns, layers, null);
  const preset = presetColumnsForStrategy(focus, pool, max);
  return preset.length ? preset : filtered;
}

function knownStrategyRecord(strategyId: string): StrategyRecord | null {
  const sid = String(strategyId || '').trim().toLowerCase();
  if (!sid) return null;
  if (featureTaxonomy && featureTaxonomy.live_strategies) {
    const liveHit = featureTaxonomy.live_strategies.find(
      (s) => String(s.id).toLowerCase() === sid,
    );
    if (liveHit) return liveHit;
  }
  if (featureTaxonomy && featureTaxonomy.strategies) {
    const hit = featureTaxonomy.strategies.find(
      (s) => String(s.id).toLowerCase() === sid,
    );
    if (hit) return hit;
  }
  return (
    KNOWN_STRATEGIES.find((s) => String(s.id).toLowerCase() === sid) || null
  );
}

function liveStrategyRecords(): StrategyRecord[] {
  const tax = featureTaxonomy;
  if (tax && Array.isArray(tax.live_strategies) && tax.live_strategies.length) {
    return tax.live_strategies;
  }
  const ids = tax && tax.live_strategy_ids;
  if (!Array.isArray(ids) || !ids.length) return [];
  const out: StrategyRecord[] = [];
  const seen = new Set<string>();
  for (const rawId of ids) {
    const sid = String(rawId || '').trim().toLowerCase();
    if (!sid || seen.has(sid)) continue;
    const meta = knownStrategyRecord(sid);
    if (meta) {
      seen.add(sid);
      out.push(meta);
    }
  }
  return out;
}

export function listStrategiesForLayers(
  layers: LayerVisibility | null | undefined,
): StrategyListEntry[] {
  const out: StrategyListEntry[] = [];
  const seen = new Set<string>();
  for (const meta of liveStrategyRecords()) {
    const sid = String(meta.id || '').trim().toLowerCase();
    if (!sid || seen.has(sid)) continue;
    if (!isLayerEnabled(meta.account_layer, layers)) continue;
    seen.add(sid);
    const layer = meta.account_layer;
    out.push({
      id: meta.id || sid,
      account_layer: layer,
      account_layer_title:
        meta.account_layer_title ||
        (ACCOUNT_LAYER_META[layer] || {}).title ||
        layer,
      title: meta.title || sid,
      stages: meta.stages || {},
    });
  }
  return out.sort((a, b) => String(a.id).localeCompare(String(b.id)));
}

/**
 * Picker columns: optional single-strategy focus; otherwise enabled layers only (no shared flood).
 */
export function filterColumnsForFeaturePicker(
  columns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
  strategyFocus: string | null | undefined,
): string[] {
  const focus = strategyFocus ? String(strategyFocus).trim() : '';
  return (columns || []).filter((col) => {
    const m = lookupFeatureMeta(col);
    if (focus) {
      return m.strategy === focus;
    }
    if (m.account_layer === 'shared' || m.strategy === 'shared') {
      return false;
    }
    return isLayerEnabled(m.account_layer, layers);
  });
}

/** Drop selected columns whose owning account_layer is now disabled (keeps shared cols). */
export function filterSelectedFeaturesByLayers(
  columns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
): string[] {
  return (columns || []).filter((col) => {
    const m = lookupFeatureMeta(col);
    const layer = m && m.account_layer ? m.account_layer : 'shared';
    if (layer === 'shared') return true;
    return isLayerEnabled(layer, layers);
  });
}

export function inferStrategyFocusFromLayers(
  layers: LayerVisibility | null | undefined,
): string | null {
  const enabled = ACCOUNT_LAYER_ORDER.filter((id) => isLayerEnabled(id, layers));
  if (enabled.length !== 1) return null;
  if (enabled[0] === 'multi_leg') return 'chop_grid';
  if (enabled[0] === 'spot') return 'spot_accum_simple';
  return null;
}

export function strategyFocusLabel(strategyId: string | null | undefined): string {
  if (!strategyId) return '全部（当前账户层）';
  return strategyMeta(strategyId).title || strategyId;
}

let featureTaxonomy: FeatureTaxonomy | null = null;

export function setFeatureTaxonomy(taxonomy: FeatureTaxonomy | null | undefined): void {
  featureTaxonomy = taxonomy && typeof taxonomy === 'object' ? taxonomy : null;
}

export function getFeatureTaxonomy(): FeatureTaxonomy | null {
  return featureTaxonomy;
}

/** API stage_order may omit live-only stages (e.g. regime); still show buckets present in data. */
function orderedStagesForNode(stageNode: Record<string, string[]> | null | undefined): string[] {
  const present = Object.keys(stageNode || {});
  if (!present.length) return [];
  const tax = featureTaxonomy?.stage_order;
  const skeleton = tax?.length ? [...tax, ...STAGE_ORDER] : STAGE_ORDER;
  const out: string[] = [];
  for (const s of skeleton) {
    if (present.includes(s) && !out.includes(s)) out.push(s);
  }
  for (const s of present) {
    if (s !== 'other' && !out.includes(s)) out.push(s);
  }
  if (present.includes('other') && !out.includes('other')) out.push('other');
  return out;
}

function taxonomyIndex(): Record<string, FeatureMeta[]> {
  return (featureTaxonomy && featureTaxonomy.index) || {};
}

function layerKeyForAccount(accountLayer: string): keyof LayerVisibility | null {
  const m = ACCOUNT_LAYER_META[accountLayer];
  return m ? m.layerKey : null;
}

function isLayerEnabled(
  accountLayer: string,
  layers: LayerVisibility | null | undefined,
): boolean {
  const key = layerKeyForAccount(accountLayer);
  if (!key || !layers) return true;
  return layers[key] !== false;
}

/** Account layer when strategy slug is unknown. */
export function classifyFeatureColumn(column: string): string {
  return inferStrategyIdFromColumn(column).account_layer;
}

/** Map feature column name -> { strategy, account_layer } when not in YAML index. */
export function inferStrategyIdFromColumn(column: string): InferredStrategy {
  const lc = String(column || '').toLowerCase();
  if (lc === 'chop_grid' || lc === 'trend_scalp') {
    return {
      strategy: lc,
      account_layer: 'multi_leg',
    };
  }
  if (
    lc.includes('weekly_ema') ||
    lc.startsWith('spot_') ||
    lc.includes('can_buy') ||
    lc.includes('spot_accum')
  ) {
    return { strategy: 'spot_accum_simple', account_layer: 'spot' };
  }
  if (lc.startsWith('tpc_')) return { strategy: 'tpc', account_layer: 'trend' };
  if (lc.startsWith('fer_')) return { strategy: 'fer', account_layer: 'trend' };
  if (lc.startsWith('me_')) return { strategy: 'me', account_layer: 'trend' };
  if (lc.startsWith('srb_')) return { strategy: 'srb', account_layer: 'trend' };
  if (lc.startsWith('bpc_') && !lc.includes('chop')) {
    return { strategy: 'bpc', account_layer: 'trend' };
  }
  if (
    lc.startsWith('chop_') ||
    (lc.includes('semantic_chop') && !lc.startsWith('tpc_')) ||
    lc.includes('grid') ||
    lc.includes('vol_clustering') ||
    lc.startsWith('box_pos_60') ||
    lc.startsWith('box_stability_60') ||
    lc.startsWith('box_width_pct_60') ||
    lc.startsWith('box_touches_')
  ) {
    return { strategy: 'chop_grid', account_layer: 'multi_leg' };
  }
  if (
    lc.startsWith('vpin') ||
    lc === 'trend_confidence' ||
    lc.startsWith('trend_confidence')
  ) {
    return { strategy: 'trend_scalp', account_layer: 'multi_leg' };
  }
  if (
    lc.includes('trend_div') ||
    lc.startsWith('ema_1200') ||
    lc.startsWith('macd_atr') ||
    lc.startsWith('box_pos_120') ||
    lc.startsWith('box_breakout')
  ) {
    return { strategy: 'tpc', account_layer: 'trend' };
  }
  return { strategy: 'shared', account_layer: 'shared' };
}

function inferStageForColumn(column: string, strategyId: string): string {
  const sid = String(strategyId || '').toLowerCase();
  const lc = String(column || '').toLowerCase();
  if (sid === 'chop_grid') {
    if (lc === 'box_pos_60') return 'prefilter';
    if (
      lc === 'box_prefilter' ||
      lc === 'bpc_semantic_chop' ||
      lc === 'semantic_chop' ||
      lc === 'tpc_semantic_chop' ||
      lc.startsWith('box_stability_') ||
      lc.startsWith('box_width_pct_') ||
      lc.startsWith('box_touches_')
    ) {
      return 'regime';
    }
  }
  return 'other';
}

export function lookupFeatureMeta(column: string): FeatureMeta {
  const col = String(column || '');
  const idx = taxonomyIndex();
  const hits = idx[col] || (col.endsWith('_f') ? idx[col.slice(0, -2)] : null);
  if (hits && hits.length) return hits[0];
  const inferred = inferStrategyIdFromColumn(col);
  const layer = inferred.account_layer;
  const layerMeta = ACCOUNT_LAYER_META[layer] || ACCOUNT_LAYER_META.shared;
  const sm = strategyMeta(inferred.strategy);
  const stage = inferStageForColumn(col, inferred.strategy);
  const stageTitle =
    (featureTaxonomy &&
      featureTaxonomy.stage_labels &&
      featureTaxonomy.stage_labels[stage]) ||
    (stage === 'regime' ? 'Regime' : stage === 'prefilter' ? 'Prefilter' : '其他');
  return {
    column: col,
    account_layer: layer,
    account_layer_title: layerMeta.title,
    strategy: inferred.strategy,
    strategy_title: sm.title || inferred.strategy,
    stage,
    stage_title: stageTitle,
  };
}

export function strategyMeta(strategyId: string): StrategyMeta {
  const hit = knownStrategyRecord(strategyId);
  if (hit) {
    return {
      id: hit.id,
      title: hit.title || hit.id,
      layerKey: layerKeyForAccount(hit.account_layer),
      account_layer: hit.account_layer,
    };
  }
  const layer = strategyId === 'shared' ? 'shared' : classifyFeatureColumn(strategyId);
  const lm = ACCOUNT_LAYER_META[layer] || ACCOUNT_LAYER_META.shared;
  return { id: strategyId, title: strategyId, layerKey: lm.layerKey, account_layer: layer };
}

function _bucketColumnsByTaxonomy(columns: string[] | null | undefined): FeatureTaxonomyTree {
  const tree: FeatureTaxonomyTree = {};
  for (const col of columns || []) {
    const meta = lookupFeatureMeta(col);
    const layer = meta.account_layer || 'shared';
    const strat = meta.strategy || 'unknown';
    const stage = meta.stage || 'other';
    if (!tree[layer]) tree[layer] = {};
    if (!tree[layer][strat]) tree[layer][strat] = {};
    if (!tree[layer][strat][stage]) tree[layer][strat][stage] = [];
    tree[layer][strat][stage].push(String(col));
  }
  for (const layer of Object.keys(tree)) {
    for (const strat of Object.keys(tree[layer])) {
      for (const stage of Object.keys(tree[layer][strat])) {
        tree[layer][strat][stage].sort();
      }
    }
  }
  return tree;
}

function _strategyOrderForLayer(layerId: string): string[] {
  if (featureTaxonomy && featureTaxonomy.strategies) {
    return featureTaxonomy.strategies
      .filter((s) => s.account_layer === layerId)
      .map((s) => s.id);
  }
  return [];
}

export function groupFeatureColumnsByStrategy(
  columns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
): FeatureGroupTuple[] {
  const tree = _bucketColumnsByTaxonomy(columns);
  const out: FeatureGroupTuple[] = [];
  for (const layerId of ACCOUNT_LAYER_ORDER) {
    if (!isLayerEnabled(layerId, layers)) continue;
    const layerNode = tree[layerId];
    if (!layerNode) continue;
    const layerTitle = (ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
    const stratOrder = _strategyOrderForLayer(layerId);
    const stratIds = [
      ...stratOrder.filter((id) => layerNode[id]),
      ...Object.keys(layerNode).filter((id) => !stratOrder.includes(id)),
    ];
    for (const stratId of stratIds) {
      const stageNode = layerNode[stratId];
      if (!stageNode) continue;
      const stratTitle =
        (lookupFeatureMeta(stageNode[Object.keys(stageNode)[0]][0]) || {}).strategy_title ||
        stratId;
      for (const stage of orderedStagesForNode(stageNode)) {
        const cols = stageNode[stage];
        if (!cols || !cols.length) continue;
        const stageTitle =
          (featureTaxonomy &&
            featureTaxonomy.stage_labels &&
            featureTaxonomy.stage_labels[stage]) ||
          stage;
        out.push([
          `${layerTitle} › ${stratTitle} › ${stageTitle}`,
          cols,
          { layer: layerId, strategy: stratId, stage },
        ]);
      }
    }
  }
  const sharedNode = tree.shared;
  if (sharedNode && isLayerEnabled('shared', layers)) {
    for (const stratId of Object.keys(sharedNode)) {
      for (const stage of Object.keys(sharedNode[stratId])) {
        const cols = sharedNode[stratId][stage];
        if (!cols.length) continue;
        out.push([`未归类 › ${stage}`, cols, { layer: 'shared', strategy: stratId, stage }]);
      }
    }
  }
  return out;
}

export function groupFeatureColumns(columns: string[] | null | undefined): FeatureGroupTuple[] {
  return groupFeatureColumnsByStrategy(columns, {
    trend: true,
    spot: true,
    multiLeg: true,
  });
}

const CHOP_GRID_REGIME_CHART = new Set(['bpc_semantic_chop']);
const CHOP_GRID_REGIME_CHART_FALLBACK = new Set([
  'semantic_chop',
  'tpc_semantic_chop',
]);
const CHOP_GRID_REGIME_STATUS = new Set([
  'box_stability_60',
  'box_width_pct_60',
  'box_touches_hi_60',
  'box_touches_lo_60',
]);
const CHOP_GRID_SKIP_SUBCHART = new Set([
  'box_prefilter',
  'semantic_chop',
  'tpc_semantic_chop',
]);
const CHOP_GRID_PREFILTER_CHART = new Set(['box_pos_60']);

export function valuePassesRefLine(
  value: number | null | undefined,
  refLine: ReferenceLine | null | undefined,
): boolean | null {
  const v = Number(value);
  const y = Number(refLine?.y);
  if (!Number.isFinite(v) || !Number.isFinite(y)) return null;
  const op = String(refLine?.operator || '').trim();
  if (op === '>=') return v >= y;
  if (op === '<=') return v <= y;
  if (op === '<') return v < y;
  if (op === '>') return v > y;
  return null;
}

export function overlayValueAtTime(
  overlay: FeatureOverlay | null | undefined,
  timeSec: number | null | undefined,
): number | null {
  const pts = overlay?.points;
  if (!pts?.length || timeSec == null) return null;
  const t = Number(timeSec);
  if (!Number.isFinite(t)) return null;
  let best: { time: number; value?: number | null } | null = null;
  for (const p of pts) {
    const pt = Number(p.time);
    if (!Number.isFinite(pt)) continue;
    if (pt > t) break;
    best = p;
  }
  if (!best || best.value == null || best.value !== best.value) return null;
  return Number(best.value);
}

/** Regime chop: same as-of + no stale extension as regime滞回 / regime退出. */
function overlayChopValueAtBar(
  candles: Candle[] | null | undefined,
  overlay: FeatureOverlay | null | undefined,
  timeSec: number | null | undefined,
): number | null {
  if (!candles?.length || !overlay?.points?.length || timeSec == null) return null;
  const series = overlayAsOfAtCandleTimes(overlay.points, candles);
  const hit = series.find((p) => Number(p.time) === Number(timeSec));
  if (!hit || hit.value == null || !Number.isFinite(Number(hit.value))) return null;
  return Number(hit.value);
}

/** YAML-shaped rows for regime.box_prefilter + rules (table, not multi-scale charts). */
export function buildThresholdMetricRows(
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
): ThresholdMetricRow[] {
  const colSet = new Set(columns || []);
  const rows: ThresholdMetricRow[] = [];
  const pickVal = (col: string) => {
    const o = overlays?.[col];
    if (!o?.available) return { value: null as number | null, overlay: o };
    const at =
      timeSec != null
        ? overlayValueAtTime(o, timeSec)
        : o.latest != null
          ? Number(o.latest)
          : null;
    return { value: at, overlay: o };
  };

  if (colSet.has('box_touches_hi_60') || colSet.has('box_touches_lo_60')) {
    const hi = pickVal('box_touches_hi_60');
    const lo = pickVal('box_touches_lo_60');
    const ref =
      (hi.overlay?.reference_lines || [])[0] ||
      (lo.overlay?.reference_lines || [])[0];
    const minTouch = ref ? Number(ref.y) : null;
    let pass: boolean | null = null;
    if (
      minTouch != null &&
      hi.value != null &&
      lo.value != null &&
      Number.isFinite(hi.value) &&
      Number.isFinite(lo.value)
    ) {
      pass = hi.value >= minTouch && lo.value >= minTouch;
    }
    rows.push({
      yaml: 'regime.box_prefilter.touches_min',
      label: 'touches_min',
      value:
        hi.value != null && lo.value != null
          ? `hi=${hi.value.toFixed(0)} lo=${lo.value.toFixed(0)}`
          : null,
      threshold: minTouch != null ? `≥${minTouch}` : '—',
      pass,
    });
  }

  if (colSet.has('box_width_pct_60')) {
    const { value, overlay: o } = pickVal('box_width_pct_60');
    const refs = o?.reference_lines || [];
    const lo = refs.find((r) => String(r.operator).includes('>='));
    const hi = refs.find((r) => String(r.operator).includes('<='));
    let pass: boolean | null = null;
    if (value != null && lo && hi) {
      const okLo = valuePassesRefLine(value, lo);
      const okHi = valuePassesRefLine(value, hi);
      pass = okLo === true && okHi === true;
    }
    const thresh =
      lo && hi
        ? `${lo.label || '≥' + lo.y} · ${hi.label || '≤' + hi.y}`
        : '—';
    rows.push({
      yaml: 'regime.box_prefilter.width',
      label: 'box_width_pct',
      value: value != null && Number.isFinite(value) ? value.toFixed(3) : null,
      threshold: thresh,
      pass,
    });
  }

  for (const col of [
    'box_stability_60',
    'box_width_pct_60',
    'box_touches_hi_60',
    'box_touches_lo_60',
  ]) {
    if (!colSet.has(col)) continue;
    if (col === 'box_width_pct_60' || col.startsWith('box_touches')) continue;
    const { value, overlay: o } = pickVal(col);
    const ref = (o?.reference_lines || [])[0];
    const pass = ref && value != null ? valuePassesRefLine(value, ref) : null;
    rows.push({
      yaml: `regime.box_prefilter.${col.replace(/_60$/, '')}`,
      label: col.replace(/_60$/, ''),
      value: value != null && Number.isFinite(value) ? value.toFixed(3) : null,
      threshold: ref ? ref.label || `${ref.operator}${ref.y}` : '—',
      pass,
    });
  }
  return rows;
}

const CHOP_GRID_REGIME_TABLE_COLS = new Set([
  'box_stability_60',
  'box_width_pct_60',
  'box_touches_hi_60',
  'box_touches_lo_60',
]);
const REGIME_BOX_TABLE_COLS = CHOP_GRID_REGIME_TABLE_COLS;
const METRICS_TABLE_STAGES = [
  'regime',
  'prefilter',
  'gate',
  'direction',
  'entry',
  'evidence',
];
const METRICS_TABLE_DEFAULT_STRATEGIES = new Set([
  'chop_grid',
  'trend_scalp',
  'tpc',
  'bpc',
  'spot_accum_simple',
]);

function isRegimeBoxMetricColumn(col: string): boolean {
  return (
    REGIME_BOX_TABLE_COLS.has(col) ||
    /^box_(stability|width_pct|touches)/.test(String(col || ''))
  );
}

function scalarThresholdHint(_column: string, overlay: FeatureOverlay | null | undefined): string {
  const refs = overlay?.reference_lines || [];
  if (!refs.length) return '';
  return refs
    .map((r) => r.label || `${r.operator || ''}${r.y != null ? r.y : ''}`)
    .filter(Boolean)
    .join(' · ');
}

export function columnsForStrategy(
  strategyId: string,
  columns: string[] | null | undefined,
): string[] {
  const sid = String(strategyId || '').toLowerCase();
  return (columns || []).filter(
    (c) => inferStrategyIdFromColumn(c).strategy === sid,
  );
}

/** Pivot metrics table when a specific strategy is focused and it has table columns. */
export function strategyMetricsTableActive(
  strategyFocus: string | null | undefined,
  columns: string[] | null | undefined,
): boolean {
  const focus = String(strategyFocus || '').trim().toLowerCase();
  if (!focus) return false;
  const all = columns || [];
  const scoped = columnsForStrategy(focus, all);
  if (all.length && !scoped.length) return false;
  if (!scoped.length) {
    return METRICS_TABLE_DEFAULT_STRATEGIES.has(focus);
  }
  return strategyMetricsColumnSpecs(focus, scoped, {}).length > 0;
}

export function chopMetricsTableActive(
  strategyFocus: string | null | undefined,
  columns: string[] | null | undefined,
): boolean {
  return strategyMetricsTableActive(strategyFocus, columns);
}

export function strategyUsesMetricsTable(
  strategyId: string,
  strategyFocus: string | null | undefined,
): boolean {
  const sid = String(strategyId || '').toLowerCase();
  const focus = String(strategyFocus || '').trim().toLowerCase();
  return !!focus && sid === focus;
}

export function chopGridUsesMetricsTable(
  strategyId: string,
  strategyFocus: string | null | undefined,
): boolean {
  return strategyUsesMetricsTable(strategyId, strategyFocus);
}

/** Column specs for per-bar metrics table (headers carry YAML thresholds). */
export function strategyMetricsColumnSpecs(
  strategyId: string,
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
): MetricsColumnSpec[] {
  const sid = String(strategyId || '').toLowerCase();
  if (sid === 'chop_grid') {
    return chopGridMetricsColumnSpecs(columns);
  }
  const colSet = new Set(columnsForStrategy(sid, columns));
  const specs: MetricsColumnSpec[] = [];
  const seenScalar = new Set<string>();
  const strat = knownStrategyRecord(sid);
  const ordered: string[] = [];
  for (const stage of METRICS_TABLE_STAGES) {
    for (const c of (strat?.stages && strat.stages[stage]) || []) {
      if (colSet.has(c) && !isRegimeBoxMetricColumn(c)) ordered.push(c);
    }
  }
  for (const c of colSet) {
    if (!ordered.includes(c) && !isRegimeBoxMetricColumn(c)) ordered.push(c);
  }
  for (const c of ordered) {
    if (seenScalar.has(c)) continue;
    seenScalar.add(c);
    const hint = scalarThresholdHint(c, overlays?.[c]);
    specs.push({
      kind: 'scalar',
      column: c,
      header: c,
      threshold: hint || '—',
    });
  }
  const regimeBoxCols = [...colSet].filter((c) => isRegimeBoxMetricColumn(c));
  if (regimeBoxCols.length) {
    specs.push({
      kind: 'regime_box',
      columns: regimeBoxCols,
      header: 'regime.box_prefilter',
      threshold: 'stability/width/touches',
    });
  }
  return specs;
}

export function chopGridMetricsColumnSpecs(
  columns: string[] | null | undefined,
): MetricsColumnSpec[] {
  const colSet = new Set(columns || []);
  const specs: MetricsColumnSpec[] = [];
  if (colSet.has('bpc_semantic_chop')) {
    specs.push({
      kind: 'scalar',
      column: 'bpc_semantic_chop',
      header: 'bpc_semantic_chop',
      threshold: 'enter≥0.50 · exit<0.32',
    });
  }
  if (colSet.has('box_pos_60')) {
    specs.push({
      kind: 'scalar',
      column: 'box_pos_60',
      header: 'box_pos_60',
      threshold: 'rules 0.35–0.65',
    });
  }
  const regimeCols = [...CHOP_GRID_REGIME_TABLE_COLS].filter((c) => colSet.has(c));
  if (regimeCols.length) {
    specs.push({
      kind: 'regime_box',
      columns: regimeCols,
      header: 'regime.box_prefilter',
      threshold: 'stability/width/touches',
    });
  }
  return specs;
}

export function chopGridMetricsCell(
  spec: MetricsRowSpec,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
  candles: Candle[] | null | undefined,
): MetricsCell {
  if (spec.kind === 'scalar') {
    const o = overlays?.[spec.column || ''];
    const v =
      spec.column === 'bpc_semantic_chop' && candles?.length
        ? overlayChopValueAtBar(candles, o, timeSec)
        : overlayValueAtTime(o, timeSec);
    if (v == null || !Number.isFinite(v)) return { value: '—', pass: null };
    const refs = o?.reference_lines || [];
    let pass: boolean | null = null;
    if (spec.column === 'box_pos_60' && refs.length >= 2) {
      const lo = refs.find((r) => String(r.operator).includes('>='));
      const hi = refs.find((r) => String(r.operator).includes('<='));
      pass =
        valuePassesRefLine(v, lo) === true && valuePassesRefLine(v, hi) === true;
    } else if (refs.length === 1) {
      pass = valuePassesRefLine(v, refs[0]);
    } else if (refs.length >= 2 && (spec.column || '').includes('chop')) {
      const enter = refs.find((r) => String(r.operator).includes('>='));
      pass = enter ? valuePassesRefLine(v, enter) : null;
    }
    const decimals =
      (spec.column || '').includes('chop') || (spec.column || '').includes('pos') ? 3 : 2;
    return { value: v.toFixed(decimals), pass };
  }
  const rows = buildThresholdMetricRows(spec.regimeCols, overlays, timeSec);
  if (!rows.length) return { value: '—', pass: null };
  const parts = rows.map((r) => {
    const badge = r.pass === true ? '✓' : r.pass === false ? '✗' : '·';
    return `${r.label}:${r.value ?? '—'}${badge}`;
  });
  const pass = rows.every((r) => r.pass === true)
    ? true
    : rows.some((r) => r.pass === false)
      ? false
      : null;
  return { value: parts.join(' '), pass };
}

export function strategyMetricsRowCell(
  strategyId: string,
  row: MetricsRowSpec,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
  candles?: Candle[] | null | undefined,
): MetricsCell {
  const sid = String(strategyId || '').toLowerCase();
  if (sid === 'chop_grid') {
    return chopGridMetricsCell(row, overlays, timeSec, candles);
  }
  if (row.kind === 'scalar') {
    return strategyMetricsCell(
      {
        kind: 'scalar',
        column: row.column,
        header: row.label,
        threshold: row.threshold,
      },
      overlays,
      timeSec,
    );
  }
  const built = buildThresholdMetricRows(row.regimeCols, overlays, timeSec);
  const hit =
    built.find((r) => r.yaml === row.yaml) ||
    built.find((r) => r.label === row.label);
  if (!hit) return { value: '—', pass: null };
  return { value: hit.value != null ? String(hit.value) : '—', pass: hit.pass };
}

function strategyMetricsCell(
  spec: MetricsColumnSpec,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
): MetricsCell {
  if (spec.kind === 'scalar') {
    const o = overlays?.[spec.column || ''];
    const v = overlayValueAtTime(o, timeSec);
    if (v == null || !Number.isFinite(v)) return { value: '—', pass: null };
    const refs = o?.reference_lines || [];
    let pass: boolean | null = null;
    if (spec.column === 'box_pos_60' && refs.length >= 2) {
      const lo = refs.find((r) => String(r.operator).includes('>='));
      const hi = refs.find((r) => String(r.operator).includes('<='));
      pass =
        valuePassesRefLine(v, lo) === true && valuePassesRefLine(v, hi) === true;
    } else if (refs.length === 1) {
      pass = valuePassesRefLine(v, refs[0]);
    } else if (refs.length >= 2 && String(spec.column).includes('chop')) {
      const enter = refs.find((r) => String(r.operator).includes('>='));
      pass = enter ? valuePassesRefLine(v, enter) : null;
    } else if (refs.length >= 1) {
      pass = refs.every((r) => valuePassesRefLine(v, r) === true)
        ? true
        : refs.some((r) => valuePassesRefLine(v, r) === false)
          ? false
          : null;
    }
    const decimals =
      String(spec.column).includes('chop') ||
      String(spec.column).includes('pos') ||
      String(spec.column).includes('confidence')
        ? 3
        : 2;
    return { value: v.toFixed(decimals), pass };
  }
  const rows = buildThresholdMetricRows(spec.columns, overlays, timeSec);
  if (!rows.length) return { value: '—', pass: null };
  const parts = rows.map((r) => {
    const badge = r.pass === true ? '✓' : r.pass === false ? '✗' : '·';
    return `${r.label}:${r.value ?? '—'}${badge}`;
  });
  const pass = rows.every((r) => r.pass === true)
    ? true
    : rows.some((r) => r.pass === false)
      ? false
      : null;
  return { value: parts.join(' '), pass };
}

/** Pivot table: one row per metric; columns = visible bars (no vertical scroll on crosshair). */
export function strategyMetricsRowSpecs(
  strategyId: string,
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
): MetricsRowSpec[] {
  const sid = String(strategyId || '').toLowerCase();
  const out: MetricsRowSpec[] = [];
  for (const spec of strategyMetricsColumnSpecs(sid, columns, overlays)) {
    if (spec.kind === 'scalar') {
      out.push({
        kind: 'scalar',
        column: spec.column,
        label: spec.header,
        threshold: spec.threshold,
      });
      continue;
    }
    const regimeCols = spec.columns || [];
    const template = buildThresholdMetricRows(regimeCols, overlays, null);
    for (const r of template) {
      out.push({
        kind: 'threshold_row',
        regimeCols,
        yaml: r.yaml,
        label: r.label,
        threshold: r.threshold,
      });
    }
  }
  return out;
}

export function chopGridMetricsRowSpecs(
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
): MetricsRowSpec[] {
  return strategyMetricsRowSpecs('chop_grid', columns, overlays);
}

export function chopGridMetricsRowCell(
  row: MetricsRowSpec,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
  candles: Candle[] | null | undefined,
): MetricsCell {
  return chopGridMetricsCell(row, overlays, timeSec, candles);
}

/** Top-row gate for pivot table: strategy-specific or generic all-pass. */
export function strategyBarGateEvaluator(
  strategyId: string,
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
  candles?: Candle[] | null | undefined,
): boolean | null {
  const sid = String(strategyId || '').toLowerCase();
  if (sid === 'chop_grid') {
    return chopGridBarCanEnter(columns, overlays, timeSec, candles);
  }
  const rows = strategyMetricsRowSpecs(sid, columns, overlays);
  if (!rows.length) return null;
  let anyFail = false;
  let anyPass = false;
  for (const row of rows) {
    const cell = strategyMetricsRowCell(sid, row, overlays, timeSec);
    if (cell.pass === false) anyFail = true;
    if (cell.pass === true) anyPass = true;
  }
  if (anyFail) return false;
  if (anyPass) return true;
  return null;
}

/**
 * Match chop_grid live: enter when chop≥entry, box_pos in band, and not a stable box
 * (all regime.box_prefilter rows pass).
 */
export function chopGridBarCanEnter(
  columns: string[] | null | undefined,
  overlays: FeatureOverlays | null | undefined,
  timeSec: number | null | undefined,
  candles: Candle[] | null | undefined,
): boolean {
  const rows = chopGridMetricsRowSpecs(columns, overlays);
  let chopPass: boolean | null = null;
  let boxPosPass: boolean | null = null;
  const regimePasses: Array<boolean | null> = [];
  for (const row of rows) {
    const cell = chopGridMetricsCell(row, overlays, timeSec, candles);
    if (row.kind === 'scalar' && row.column === 'bpc_semantic_chop') {
      chopPass = cell.pass;
    } else if (row.kind === 'scalar' && row.column === 'box_pos_60') {
      boxPosPass = cell.pass;
    } else if (row.kind === 'threshold_row') {
      regimePasses.push(cell.pass);
    }
  }
  if (chopPass !== true) return false;
  if (boxPosPass === false) return false;
  const stableBox =
    regimePasses.length > 0 && regimePasses.every((p) => p === true);
  if (stableBox) return false;
  return true;
}

export function strategyStagePanePlan(
  stratId: string,
  stage: string,
  cols: string[] | null | undefined,
  strategyFocus: string | null | undefined,
): StagePanePlan {
  const list = cols || [];
  const sid = String(stratId).toLowerCase();
  const focus = strategyFocus ? String(strategyFocus).trim().toLowerCase() : '';
  if (strategyMetricsTableActive(focus, list) && focus === sid) {
    if (stage === 'other') {
      return { chartCols: [], statusCols: [], skipStage: true };
    }
    return { chartCols: [], statusCols: [], skipStage: false };
  }
  if (sid !== 'chop_grid') {
    return { chartCols: list, statusCols: [], skipStage: false };
  }
  return chopGridStagePanePlan(stratId, stage, cols, strategyFocus);
}

/** One chop line chart + YAML table for box_prefilter; prefilter = box_pos chart only. */
function chopGridStagePanePlan(
  stratId: string,
  stage: string,
  cols: string[] | null | undefined,
  strategyFocus: string | null | undefined,
): StagePanePlan {
  const list = cols || [];
  if (String(stratId).toLowerCase() !== 'chop_grid') {
    return { chartCols: list, statusCols: [], skipStage: false };
  }
  if (strategyMetricsTableActive(strategyFocus, list)) {
    if (stage === 'other') {
      return { chartCols: [], statusCols: [], skipStage: true };
    }
    return { chartCols: [], statusCols: [], skipStage: false };
  }
  if (stage === 'other') {
    return { chartCols: [], statusCols: [], skipStage: true };
  }
  if (stage === 'regime') {
    let chartPick = list.filter((c) => CHOP_GRID_REGIME_CHART.has(c));
    if (!chartPick.length) {
      chartPick = list.filter((c) => CHOP_GRID_REGIME_CHART_FALLBACK.has(c));
    }
    const status = list.filter((c) => CHOP_GRID_REGIME_STATUS.has(c));
    return { chartCols: chartPick.slice(0, 1), statusCols: status, skipStage: false };
  }
  if (stage === 'prefilter') {
    const chart = list.filter((c) => CHOP_GRID_PREFILTER_CHART.has(c));
    return {
      chartCols: chart.length ? chart : [],
      statusCols: [],
      skipStage: false,
    };
  }
  return { chartCols: [], statusCols: [], skipStage: true };
}

export function subchartColumnsForStrategy(
  strategyId: string,
  columns: string[] | null | undefined,
  strategyFocus: string | null | undefined,
): string[] {
  const sid = String(strategyId || '').toLowerCase();
  const focus = strategyFocus ? String(strategyFocus).trim().toLowerCase() : '';
  if (
    focus &&
    sid === focus &&
    strategyMetricsTableActive(focus, columns)
  ) {
    return [];
  }
  if (sid !== 'chop_grid') return columns || [];
  return (columns || []).filter((c) => !CHOP_GRID_SKIP_SUBCHART.has(c));
}

export function orderFeaturePaneItems(
  columns: string[] | null | undefined,
  layers: LayerVisibility | null | undefined,
  strategyFocus: string | null | undefined,
): FeaturePaneItem[] {
  const focus = strategyFocus ? String(strategyFocus).trim() : '';
  const tableOnly = strategyMetricsTableActive(focus, columns);
  const tree = _bucketColumnsByTaxonomy(columns);
  const items: FeaturePaneItem[] = [];
  let firstLayer = true;
  for (const layerId of ACCOUNT_LAYER_ORDER) {
    if (!isLayerEnabled(layerId, layers)) continue;
    const layerNode = tree[layerId];
    if (!layerNode) continue;
    if (!tableOnly) {
      if (!firstLayer) items.push({ type: 'gap', id: `gap-layer-${layerId}` });
      firstLayer = false;
      const layerTitle = (ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
      items.push({
        type: 'header',
        strategy: layerId,
        title: layerTitle,
        headerKind: 'layer',
      });
    }
    const stratOrder = _strategyOrderForLayer(layerId);
    const stratIds = [
      ...stratOrder.filter((id) => layerNode[id]),
      ...Object.keys(layerNode).filter((id) => !stratOrder.includes(id)),
    ];
    let firstStrat = true;
    for (const stratId of stratIds) {
      if (tableOnly && stratId !== focus) continue;
      if (focus && stratId !== focus) continue;
      const stageNode = layerNode[stratId];
      if (!stageNode) continue;
      if (!tableOnly && !firstStrat) {
        items.push({ type: 'gap', id: `gap-strat-${layerId}-${stratId}` });
      }
      firstStrat = false;
      const sample = stageNode[Object.keys(stageNode)[0]][0];
      const sm = lookupFeatureMeta(sample);
      const tableMode =
        focus && stratId === focus && strategyMetricsTableActive(focus, columns);
      const tableCols = tableMode
        ? columnsForStrategy(stratId, columns)
        : [];
      if (!tableMode) {
        items.push({
          type: 'header',
          strategy: stratId,
          title: sm.strategy_title || stratId,
          headerKind: 'strategy',
          accountLayer: layerId,
        });
      }
      if (tableMode) {
        items.push({
          type: 'metrics_table',
          id: `metrics-${stratId}`,
          strategy: stratId,
          accountLayer: layerId,
          columns: tableCols.length ? tableCols : (columns || []).slice(),
        });
        continue;
      }
      for (const stage of orderedStagesForNode(stageNode)) {
        const cols = stageNode[stage];
        if (!cols || !cols.length) continue;
        const stageTitle =
          (featureTaxonomy &&
            featureTaxonomy.stage_labels &&
            featureTaxonomy.stage_labels[stage]) ||
          stage;
        items.push({
          type: 'header',
          strategy: stratId,
          title: stageTitle,
          headerKind: 'stage',
          accountLayer: layerId,
          stage,
        });
        const plan = strategyStagePanePlan(stratId, stage, cols, focus);
        if (plan.skipStage) continue;
        if (plan.statusCols.length) {
          items.push({
            type: 'threshold_status',
            id: `status-${stratId}-${stage}`,
            strategy: stratId,
            accountLayer: layerId,
            stage,
            columns: plan.statusCols,
          });
        }
        for (const col of plan.chartCols) {
          items.push({
            type: 'feature',
            column: col,
            strategy: stratId,
            accountLayer: layerId,
            stage,
          });
        }
      }
    }
  }
  return items;
}

export function presetColumnsForStrategy(
  strategyId: string,
  available: string[] | null | undefined,
  maxCols: number,
): string[] {
  const sid = String(strategyId || '').toLowerCase();
  const avail = new Set(available || []);
  const picks: string[] = [];
  if (sid === 'chop_grid') {
    // Match prefilter.yaml: regime.entry_feature + rules (box_pos band).
    // regime.box_prefilter (stability/width/touches) is optional reference only.
    for (const c of ['bpc_semantic_chop', 'box_pos_60']) {
      if (avail.has(c) && !picks.includes(c)) picks.push(c);
      if (picks.length >= maxCols) return picks;
    }
  }
  if (sid === 'trend_scalp') {
    for (const c of ['trend_confidence', 'bpc_semantic_chop']) {
      if (avail.has(c) && !picks.includes(c)) picks.push(c);
      if (picks.length >= maxCols) return picks;
    }
  }
  if (featureTaxonomy && featureTaxonomy.strategies) {
    const strat = featureTaxonomy.strategies.find((s) => s.id === sid);
    if (strat) {
      for (const stage of ['regime', 'prefilter', 'direction', 'gate', 'entry', 'evidence']) {
        for (const c of (strat.stages && strat.stages[stage]) || []) {
          if (avail.has(c) && !picks.includes(c)) picks.push(c);
          if (picks.length >= maxCols) return picks;
        }
      }
    }
  }
  return picks;
}

export function presetColumnsForAccountLayer(
  layerId: string,
  available: string[] | null | undefined,
  maxCols: number,
): string[] {
  const avail = new Set(available || []);
  const picks: string[] = [];
  if (layerId === 'multi_leg') {
    return presetColumnsForStrategy('chop_grid', available, maxCols);
  }
  if (featureTaxonomy && featureTaxonomy.strategies) {
    for (const s of featureTaxonomy.strategies) {
      if (s.account_layer !== layerId) continue;
      for (const stage of ['regime', 'prefilter', 'gate', 'entry']) {
        for (const c of (s.stages && s.stages[stage]) || []) {
          if (avail.has(c) && !picks.includes(c)) picks.push(c);
          if (picks.length >= maxCols) return picks;
        }
      }
    }
  }
  return picks;
}
