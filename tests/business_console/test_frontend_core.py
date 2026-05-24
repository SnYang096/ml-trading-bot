"""Node-based tests for trade-map-core.js (web logic)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

CORE_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mlbot_console"
    / "static"
    / "trade-map-core.js"
)
STATIC_ROOT = CORE_JS.parent
NODE_SCRIPT = """
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync(process.argv[1], 'utf8');
const ctx = { globalThis: {} };
vm.runInContext(code, vm.createContext(ctx));
const Core = ctx.globalThis.MLBotTradeMapCore;
const markers = [
  { id: 'trend:orders:1', time: 1, scope: 'trend', event: 'entry', side: 'long', status: 'filled' },
  { id: 'spot:spot_orders:2', time: 2, scope: 'spot', event: 'exit', side: 'long', status: 'pending', pnl_usdt: -1 },
  { id: 'multi_leg:orders:3', time: 3, scope: 'multi_leg', strategy: 'chop_grid', event: 'tp', side: 'short', status: 'pending', detail: { leg_label: 'L1_tp' } },
  { id: 'multi_leg:orders:4', time: 4, scope: 'multi_leg', strategy: 'chop_grid', event: 'grid', side: 'long', status: 'pending', detail: { leg_label: 'L2' } },
  { id: 'multi_leg:orders:5', time: 5, scope: 'multi_leg', strategy: 'chop_grid', event: 'entry', side: 'short', status: 'filled', detail: { leg_label: 'S1' } },
  { id: 'multi_leg:orders:6', time: 6, scope: 'multi_leg', strategy: 'chop_grid', event: 'tp', side: 'long', status: 'filled', detail: { leg_label: 'S2_tp' } },
];
const lwc = Core.markersToLwc(markers);
const scopes = Core.scopesFromLayers({ trend: true, spot: false, multiLeg: true, pending: false });
const grafana = Core.resolveLinkUrl({ id: "grafana", url: "http://host.docker.internal:3000/" });
const spacing = Core.barSpacingForCount(500);
const vis = Core.defaultVisibleBarCount(2161);
const range = Core.visibleLogicalRange(2161);
const clean = Core.sanitizeCandlesForLwc([
  { time: 100, open: 1, high: 2, low: 0.5, close: 1.5 },
  { time: 100, open: 9, high: 9, low: 9, close: 9 },
  { time: 200, open: 2, high: 3, low: 1, close: 2.5, volume: 10 },
  { time: 300, open: 2100, high: 9000, low: -3200, close: 2100 },
]);
const pr = Core.priceRangeForVisibleCandles(clean, { from: 2, to: 3 });
const prAuto = Core.priceRangeForChartAutoscale(clean, null);
Core.setFeatureTaxonomy({
  strategies: [
    { id: "tpc", account_layer: "trend", title: "TPC", stages: { prefilter: ["tpc_pullback_depth"], gate: ["tpc_semantic_chop"] } },
    { id: "spot_accum_simple", account_layer: "spot", title: "Spot", stages: { prefilter: ["weekly_ema_200_position"] } },
  ],
  index: {
    "tpc_pullback_depth": [{ column: "tpc_pullback_depth", account_layer: "trend", account_layer_title: "B·Trend", strategy: "tpc", strategy_title: "TPC", stage: "prefilter", stage_title: "Prefilter" }],
    "weekly_ema_200_position": [{ column: "weekly_ema_200_position", account_layer: "spot", account_layer_title: "A·Spot", strategy: "spot_accum_simple", strategy_title: "Spot", stage: "prefilter", stage_title: "Prefilter" }],
  },
  stage_order: ["prefilter", "gate"],
  stage_labels: { prefilter: "Prefilter", gate: "Gate" },
});
const grouped = Core.groupFeatureColumns(["weekly_ema_200_position", "tpc_pullback_depth"]);
const plan = Core.orderFeaturePaneItems(
  ["tpc_pullback_depth", "weekly_ema_200_position"],
  { trend: true, spot: true, multiLeg: true }
);
const meta = Core.lookupFeatureMeta("tpc_pullback_depth");
const hit = Core.findMarkerByTime(markers, 1, 7200);
const sel = Core.markersToLwc(markers, "trend:orders:1");
const merged = Core.mergeCandlesByTime(
  [{ time: 100, open: 1, high: 2, low: 0.5, close: 1 }],
  [{ time: 200, open: 2, high: 3, low: 1, close: 2 }, { time: 100, open: 9, high: 9, low: 9, close: 9 }]
);
console.log(JSON.stringify({
  scopes, lwcCount: lwc.length, pendingShape: lwc[1].shape, grafana, spacing,
  vis, range, cleanLen: clean.length, cleanTime: clean[1].time,
  prMin: pr && pr.minValue, prMax: pr && pr.maxValue,
  prAutoMin: prAuto && prAuto.minValue, prAutoMax: prAuto && prAuto.maxValue,
  groupTitles: grouped.map((g) => g[0]),
  planTypes: plan.map((p) => p.type),
  metaStage: meta.stage,
  hitId: hit && hit.id, selColor: sel[0].color,
  init2h: Core.tradeMapInitialDays("2h"),
  init1d: Core.ohlcvInitialQueryRange("1d"),
  init2hRange: Core.ohlcvInitialQueryRange("2h"),
  chunk1d: Core.tradeMapHistoryChunkDays("1d"),
  mergedLen: merged.length,
  mergedFirst: merged[0].time,
  mergedLast: merged[1].time,
  tpText: Core.markersToLwc(markers)[2].text,
  l2Pending: Core.markersToLwc(markers)[3].text,
  s1Filled: Core.markersToLwc(markers)[4].text,
  s1Above: Core.markersToLwc(markers)[4].position,
  s2TpBelow: Core.markersToLwc(markers)[5].position,
  l1TpAbove: Core.markersToLwc(markers)[2].position,
  segPts: Core.chopSegmentedLinePoints(
    [{ start: 100, end: 200 }, { start: 300, end: 400 }],
    640.5,
    7200
  ),
  gridLabelLong: Core.chopGridLabelAnchor("long", "grid"),
  gridLabelShortTp: Core.chopGridLabelAnchor("short", "tp"),
  prAuto2: Core.priceRangeForChartAutoscale(
    [{ time: 100, low: 600, high: 620, close: 610 }, { time: 200, low: 610, high: 630, close: 625 }],
    { from: 0, to: 1 }
  ),
}));
"""


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_trade_map_core_node():
    proc = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(CORE_JS)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout.strip())
    assert out["scopes"] == "trend,multi_leg"
    assert out["lwcCount"] == 6
    assert out["pendingShape"] == "circle"
    assert out["spacing"] == 4
    assert out["vis"] == 320
    assert out["range"] == {"from": 1841, "to": 2160}
    assert out["cleanLen"] == 3
    assert out["cleanTime"] == 200
    assert out["prMin"] is not None and out["prMin"] > 2000
    assert out["prMax"] is not None and out["prMax"] < 2500
    assert out["prAutoMin"] is not None and out["prAutoMax"] is not None
    assert "B·Trend › TPC › Prefilter" in out["groupTitles"][0]
    assert "Prefilter" in out["groupTitles"][1]
    assert out["metaStage"] == "prefilter"
    assert "header" in out["planTypes"] and "feature" in out["planTypes"]
    assert out["hitId"] == "trend:orders:1"
    assert out["selColor"] == "#ffeb3b"
    assert out["init2h"] == 60
    assert out["init1d"]["full_range"] == "true"
    assert "from" not in out["init1d"]
    assert "from" in out["init2hRange"]
    assert out["chunk1d"] == 90
    assert out["mergedLen"] == 2
    assert out["mergedFirst"] == 100
    assert out["mergedLast"] == 200
    assert out["tpText"] == "L1_TP"
    assert out["l2Pending"] == "L2 挂单"
    assert out["s1Filled"] == "S1 成交"
    assert out["s1Above"] == "aboveBar"
    assert out["s2TpBelow"] == "belowBar"
    assert out["l1TpAbove"] == "aboveBar"
    assert out["gridLabelLong"] == "below"
    assert out["gridLabelShortTp"] == "below"
    assert len(out["segPts"]) >= 4
    assert out["segPts"][2]["value"] is None  # NaN segment gap
    assert out["prAuto2"]["minValue"] < 600
    assert out["prAuto2"]["maxValue"] > 625


MODULE_LOAD_SCRIPT = """
const fs = require('fs');
const vm = require('vm');
const root = process.argv[1];
const modules = [
  'trade-map-core.js',
  'trade-map/state.js',
  'trade-map/layout.js',
  'trade-map/chart.js',
  'trade-map/chop.js',
  'trade-map/history.js',
  'trade-map/subcharts.js',
  'trade-map/markers.js',
  'trade-map/features.js',
  'trade-map/bundle.js',
];
const noop = () => {};
const fakeEl = () => ({
  checked: false,
  value: '2h',
  style: {},
  classList: { add: noop, remove: noop, toggle: noop },
  addEventListener: noop,
  setAttribute: noop,
  appendChild: noop,
  querySelector: () => null,
  querySelectorAll: () => [],
  innerHTML: '',
  textContent: '',
});
const ctx = {
  console,
  globalThis: null,
  window: null,
  setTimeout: () => 1,
  clearTimeout: noop,
  setInterval: () => 1,
  clearInterval: noop,
  requestAnimationFrame: (fn) => fn(),
  localStorage: { getItem: () => null, setItem: noop },
  document: {
    getElementById: fakeEl,
    createElement: fakeEl,
    querySelectorAll: () => [],
    addEventListener: noop,
    body: { classList: { toggle: noop } },
  },
  LightweightCharts: {
    createChart: () => ({
      addAreaSeries: () => ({ setData: noop, applyOptions: noop }),
      addCandlestickSeries: () => ({
        setData: noop,
        setMarkers: noop,
        update: noop,
        applyOptions: noop,
        priceToCoordinate: (v) => Number(v) + 10,
      }),
      addLineSeries: () => ({ setData: noop, applyOptions: noop }),
      addHistogramSeries: () => ({ setData: noop }),
      removeSeries: noop,
      timeScale: () => ({
        getVisibleLogicalRange: () => null,
        setVisibleLogicalRange: noop,
        applyOptions: noop,
        subscribeVisibleLogicalRangeChange: noop,
        timeToCoordinate: (t) => Number(t) * 10,
      }),
      priceScale: () => ({ applyOptions: noop, setVisibleRange: noop }),
      applyOptions: noop,
      subscribeClick: noop,
      subscribeCrosshairMove: noop,
    }),
  },
  MLBotConsole: {
    api: async () => ({ data: {}, meta: {} }),
    setSymbol: noop,
    formatOrderTime: String,
    getScopesDefault: () => null,
    setScopesState: noop,
    saveOrdersFilter: noop,
    ordersFilterFromControls: () => ({}),
    ordersExcludeStatusParamFromFilter: () => '',
    applyOrdersFilterToControls: noop,
    loadOrdersFilter: () => ({}),
    isAllSymbols: () => false,
    ordersTableColspan: () => 8,
    buildOrdersTableRows: () => '',
    bindOrdersTableResize: noop,
    loadExtLinks: async () => {},
    loadSymbols: async () => {},
    bindOrdersFilterSync: noop,
    bindSymbolPersist: noop,
    escHtml: String,
  },
};
ctx.globalThis = ctx;
ctx.window = ctx;
const context = vm.createContext(ctx);
for (const mod of modules) {
  vm.runInContext(fs.readFileSync(`${root}/${mod}`, 'utf8'), context, { filename: mod });
}
const pts = context.spanHighlightCandles(
  [{ time: 1, low: 10, high: 20 }, { time: 2, low: 12, high: 22 }],
  [{ start: 1, end: 1 }]
);
const mergedPayload = context.mergeChopMapPayload(
  {
    chop_regime_regions: [{ start: 10, end: 20 }],
    strategy_stage_regions: { chop_grid: { prefilter: [{ start: 10, end: 20 }] } },
  },
  {
    chop_regime_regions: [{ start: 19, end: 30 }],
    strategy_stage_regions: { chop_grid: { prefilter: [{ start: 21, end: 30 }] } },
  }
);
const pollPayload = context.mergeChopMapPayload(
  {
    chop_grid_overlay: { center: 100, lines: [{ price: 99 }] },
    chop_regime_regions: [{ start: 1, end: 5 }],
  },
  { chop_regime_regions: [{ start: 4, end: 8 }] }
);
context.MLBotTradeMapPage.chart = context.LightweightCharts.createChart();
context.MLBotTradeMapPage.candleSeries = context.MLBotTradeMapPage.chart.addCandlestickSeries();
context.MLBotTradeMapPage.chopGridLabelSpecs = [
  { price: 100, text: 'L1 格', side: 'long', kind: 'grid', spans: [{ start: 1, end: 2 }] },
];
const labelLayer = { innerHTML: '', appendChild: noop, style: {} };
const checkedLayerIds = new Set(['layerChopGrid', 'layerMultiLeg']);
context.document.getElementById = (id) => {
  if (id === 'chopGridLabelLayer') return labelLayer;
  if (id === 'chart') return { clientWidth: 800, clientHeight: 400, appendChild: noop };
  if (checkedLayerIds.has(id)) return { ...fakeEl(), checked: true };
  return fakeEl();
};
let gridLabelsOk = true;
try {
  context.layoutChopGridLabels([{ time: 1, open: 90, high: 110, low: 85, close: 105 }]);
} catch (e) {
  gridLabelsOk = false;
}
console.log(JSON.stringify({
  loaded: typeof context.refreshBundle === 'function',
  initMainChart: typeof context.initMainChart === 'function',
  syncSubcharts: typeof context.syncSubcharts === 'function',
  highlight: pts,
  stateArrays: Array.isArray(context.MLBotTradeMapPage.lastCandles),
  mergedChopEnd: mergedPayload.chop_regime_regions[0].end,
  mergedPrefilterEnd: mergedPayload.strategy_stage_regions.chop_grid.prefilter[0].end,
  pollKeepsOverlay: pollPayload.chop_grid_overlay?.center === 100,
  pollMergedChopEnd: pollPayload.chop_regime_regions[0].end,
  gridLabelsOk,
}));
"""


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_trade_map_modules_load_in_one_browser_context():
    """Regression: split classic scripts must not collide or hide functions in IIFEs."""
    proc = subprocess.run(
        ["node", "-e", MODULE_LOAD_SCRIPT, str(STATIC_ROOT)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout.strip())
    assert out["loaded"] is True
    assert out["initMainChart"] is True
    assert out["syncSubcharts"] is True
    assert out["stateArrays"] is True
    assert out["highlight"] == [
        {"time": 1, "open": 10, "high": 20, "low": 10, "close": 20}
    ]
    assert out["mergedChopEnd"] == 30
    assert out["mergedPrefilterEnd"] == 30
    assert out["pollKeepsOverlay"] is True
    assert out["pollMergedChopEnd"] == 8
    assert out["gridLabelsOk"] is True
