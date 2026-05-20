"""Node-based tests for trade-map-core.js (web logic)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

CORE_JS = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "business-console"
    / "frontend"
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
console.log(JSON.stringify({ scopes, lwcCount: lwc.length, pendingShape: lwc[1].shape, grafana }));
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
