"""Integration: chop_grid regime hysteresis — Python core vs trade-map TS (Vitest)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mlbot_console.services.strategy_stage_regions import _hysteresis_active

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = REPO_ROOT / "frontend"
FRONTEND_SRC = FRONTEND_ROOT / "src"


def _read(rel: str) -> str:
    return (FRONTEND_SRC / rel).read_text(encoding="utf-8")


def test_python_hysteresis_finite_series():
    vals = [0.2, 0.55, 0.45, 0.25, 0.35]
    py = _hysteresis_active(vals, entry_min=0.50, exit_below=0.32)
    assert py == [False, True, True, False, False]


@pytest.mark.skipif(
    not (FRONTEND_ROOT / "package.json").is_file(),
    reason="frontend/ not present",
)
def test_vitest_covers_chop_regime_hysteresis():
    proc = subprocess.run(
        ["npm", "test"],
        cwd=str(FRONTEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tradeMap.test.ts" in proc.stdout
    assert "10 passed" in proc.stdout or "10 tests" in proc.stdout


def test_regime_helpers_in_trade_map_ts():
    markers = _read("lib/tradeMap/markers.ts")
    features = _read("lib/tradeMap/features.ts")
    assert "chopRegimeHysteresisOnBarTimes" in markers
    assert "chopRegimeExitBarTimes" in markers
    assert "overlayAsOfAtCandleTimes" in features or "overlayChopValueAtBar" in features
    assert "isFeatureBusRegimeExitMarker" in markers


def test_lwc_hook_does_not_scroll_on_click():
    hook = _read("hooks/useLightweightChart.ts")
    assert "scrollChartToBarTime" not in hook
    assert "subscribeClick" not in hook
