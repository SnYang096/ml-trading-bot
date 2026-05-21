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
]);
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
console.log(JSON.stringify({
  scopes, lwcCount: lwc.length, pendingShape: lwc[1].shape, grafana, spacing,
  vis, range, cleanLen: clean.length, cleanTime: clean[1].time,
  groupTitles: grouped.map((g) => g[0]),
  planTypes: plan.map((p) => p.type),
  metaStage: meta.stage,
  hitId: hit && hit.id, selColor: sel[0].color,
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
    assert out["lwcCount"] == 2
    assert out["pendingShape"] == "circle"
    assert out["spacing"] == 4
    assert out["vis"] == 320
    assert out["range"] == {"from": 1841, "to": 2160}
    assert out["cleanLen"] == 2
    assert out["cleanTime"] == 200
    assert "B·Trend › TPC › Prefilter" in out["groupTitles"][0]
    assert "Prefilter" in out["groupTitles"][1]
    assert out["metaStage"] == "prefilter"
    assert "header" in out["planTypes"] and "feature" in out["planTypes"]
    assert out["hitId"] == "trend:orders:1"
    assert out["selColor"] == "#ffeb3b"
