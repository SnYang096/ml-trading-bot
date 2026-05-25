"""Node tests for orders table HTML (marker id + scroll time attributes)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SHELL_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mlbot_console"
    / "static"
    / "console-shell.js"
)

NODE_SCRIPT = """
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync(process.argv[1], 'utf8');
const ctx = { globalThis: {}, fetch: async () => ({ ok: true, json: async () => ({}) }) };
ctx.globalThis = ctx;
vm.runInContext(code, vm.createContext(ctx));
const Shell = ctx.globalThis.MLBotConsole;
const esc = (s) => String(s ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/"/g, '&quot;');
const html = Shell.buildOrdersTableRows(
  [
    {
      scope: 'multi_leg',
      strategy: 'chop_grid',
      marker_id: 'multi_leg:multi_leg_orders:cg_BNB_S1_sl',
      time: 1716379584,
      leg_label: 'S1_sl',
      purpose: 'stop_loss',
      side: 'short',
      status: 'new',
      order_id: 'cg_BNB_S1_sl',
      quantity: 1,
      price: 672.51,
    },
    {
      scope: 'multi_leg',
      strategy: 'chop_grid',
      marker_id: '',
      time: 0,
      leg_label: 'L2',
      side: 'long',
      status: 'open',
      order_id: '',
    },
  ],
  { showSymbol: false, escHtml: esc }
);
console.log(
  JSON.stringify({
    hasMarkerId: html.includes('data-marker-id="multi_leg:multi_leg_orders:cg_BNB_S1_sl"'),
    hasMarkerTime: html.includes('data-marker-time="1716379584"'),
    emptyTimeAttr: html.includes('data-marker-time=""'),
    slRowClass: html.includes('orders-leg-sl-row'),
  })
);
"""


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_orders_table_row_marker_attributes():
    proc = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(SHELL_JS)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout.strip())
    assert out["hasMarkerId"] is True
    assert out["hasMarkerTime"] is True
    assert out["emptyTimeAttr"] is True
    assert out["slRowClass"] is True
