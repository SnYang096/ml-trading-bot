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
  segPts: Core.chopSegmentedLinePoints(
    [{ start: 100, end: 200 }, { start: 300, end: 400 }],
    640.5,
    7200
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
    assert out["lwcCount"] == 5
    assert out["pendingShape"] == "circle"
    assert out["spacing"] == 4
    assert out["vis"] == 320
    assert out["range"] == {"from": 1841, "to": 2160}
    assert out["cleanLen"] == 3
    assert out["cleanTime"] == 200
    assert out["prMin"] is not None and out["prMin"] > 2000
    assert out["prMax"] is not None and out["prMax"] < 2500
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
    assert len(out["segPts"]) >= 4
    assert out["segPts"][2]["value"] is None  # NaN segment gap
