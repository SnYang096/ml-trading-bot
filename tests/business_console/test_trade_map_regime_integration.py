"""Integration: chop_grid regime hysteresis — Python core vs trade-map JS."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mlbot_console.services.strategy_stage_regions import _hysteresis_active

STATIC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlbot_console" / "static"
CORE_MODULES = [
    "trade-map/core/00-constants.js",
    "trade-map/core/10-ohlcv.js",
    "trade-map/core/20-markers.js",
    "trade-map/core/30-candles.js",
    "trade-map/core/40-features.js",
    "trade-map/core/50-misc.js",
]

NODE_REGIME_SCRIPT = """
const fs = require('fs');
const vm = require('vm');
const root = process.argv[1];
const ctx = { globalThis: {} };
const vctx = vm.createContext(ctx);
for (const rel of process.argv[2].split(',')) {
  vm.runInContext(fs.readFileSync(`${root}/${rel}`, 'utf8'), vctx);
}
const Core = ctx.globalThis.MLBotTradeMapCore;

function hysteresisJs(vals, entryMin, exitBelow) {
  return Core.chopGridHysteresisActive(vals, entryMin, exitBelow);
}

const candles = [
  { time: 1000 },
  { time: 2000 },
  { time: 3000 },
  { time: 4000 },
];
const overlaysStale = {
  bpc_semantic_chop: {
    points: [
      { time: 1000, value: 0.55 },
      { time: 2000, value: 0.55 },
    ],
    reference_lines: [{ y: 0.5, operator: ">=" }, { y: 0.32, operator: "<" }],
  },
};
const { vals, chopOn } = Core.chopRegimeSeriesFromOverlay(candles, overlaysStale);
const exits = Core.chopRegimeExitBarTimes(candles, overlaysStale);
const onAt = {
  2000: Core.chopRegimeHysteresisOnAtTime(candles, overlaysStale, 2000),
  3000: Core.chopRegimeHysteresisOnAtTime(candles, overlaysStale, 3000),
  4000: Core.chopRegimeHysteresisOnAtTime(candles, overlaysStale, 4000),
};
const rows = Core.chopGridMetricsRowSpecs(["bpc_semantic_chop"], overlaysStale);
const chopRow = rows.find((r) => r.column === "bpc_semantic_chop");
const metrics = {
  2000: Core.chopGridMetricsRowCell(chopRow, overlaysStale, 2000, candles),
  4000: Core.chopGridMetricsRowCell(chopRow, overlaysStale, 4000, candles),
};

const ff = Core.forwardFillOverlayToCandles(
  overlaysStale.bpc_semantic_chop.points,
  candles
);
const asof = Core.overlayAsOfAtCandleTimes(
  overlaysStale.bpc_semantic_chop.points,
  candles
);

console.log(JSON.stringify({
  finitePyCase: hysteresisJs([0.2, 0.55, 0.45, 0.25, 0.35], 0.5, 0.32),
  vals,
  chopOn,
  exits: [...exits],
  onAt,
  metrics,
  ffLast: ff[ff.length - 1].value,
  asofLast: asof[asof.length - 1].value,
}));
"""


def _run_node_regime() -> dict:
    proc = subprocess.run(
        ["node", "-e", NODE_REGIME_SCRIPT, str(STATIC_ROOT), ",".join(CORE_MODULES)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_python_and_node_hysteresis_agree_on_finite_series():
    vals = [0.2, 0.55, 0.45, 0.25, 0.35]
    py = _hysteresis_active(vals, entry_min=0.50, exit_below=0.32)
    out = _run_node_regime()
    assert out["finitePyCase"] == py


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_stale_chop_overlay_exit_metrics_and_asof_aligned():
    """After last feature point, as-of is null (no trailing ffill) → exit + metrics OFF."""
    out = _run_node_regime()
    assert out["vals"] == [0.55, 0.55, None, None]
    assert out["chopOn"] == [True, True, False, False]
    assert out["exits"] == [3000]
    assert out["onAt"]["2000"] is True
    assert out["onAt"]["3000"] is False
    assert out["onAt"]["4000"] is False
    assert out["metrics"]["2000"]["value"] == "0.550"
    assert out["metrics"]["2000"]["pass"] is True
    assert out["metrics"]["4000"]["value"] == "—"
    assert out["metrics"]["4000"]["pass"] is None
    assert out["ffLast"] == 0.55
    assert out["asofLast"] is None


def test_subcharts_regime_hysteresis_rows(client):
    subcharts = client.get("/static/trade-map/subcharts.js").text
    assert "regime滞回" in subcharts
    assert "regime退出" in subcharts
    assert "row-regime-on-h" in subcharts
    assert "chopRegimeHysteresisOnBarTimes" in subcharts
    assert "chopRegimeExitBarTimes" in subcharts


def test_features_use_overlay_chop_at_bar(client):
    body = client.get("/static/trade-map/core/40-features.js").text
    assert "overlayChopValueAtBar" in body
    assert "overlayAsOfAtCandleTimes" in body


def test_markers_strip_feature_bus_regime_duplicates(client):
    body = client.get("/static/trade-map/markers.js").text
    assert "isFeatureBusRegimeExitMarker" in body


def test_chart_click_does_not_scroll_to_bar(client):
    chart = client.get("/static/trade-map/chart.js").text
    idx = chart.index("S.chart.subscribeClick")
    click = chart[idx : idx + 900]
    assert "scrollChartToBarTime" not in click
    assert "scrollChart: false" in click
    assert "rebuild: false" in click
