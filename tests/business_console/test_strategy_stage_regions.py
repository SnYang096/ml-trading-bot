"""Strategy prefilter / gate regions for Trade Map."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlbot_console.services.strategy_stage_regions import (
    _evaluate_prefilter_rules,
    _hysteresis_active,
    load_bundle_stage_regions,
    load_chop_grid_prefilter_regions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "config" / "strategies"


def test_main_imports_with_docker_like_layout(tmp_path: Path) -> None:
    """CMS Docker image: PYTHONPATH=/app/src, cwd=/app, needs src/config on disk."""
    app_root = tmp_path / "app"
    src_root = app_root / "src"
    for name in ("mlbot_console", "time_series_model", "config"):
        shutil.copytree(PROJECT_ROOT / "src" / name, src_root / name)
    prev_path = list(sys.path)
    prev_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(src_root))
        import importlib

        for mod in (
            "mlbot_console.main",
            "mlbot_console.services.strategy_stage_regions",
        ):
            importlib.invalidate_caches()
            if mod in sys.modules:
                del sys.modules[mod]
        os.chdir(app_root)
        importlib.import_module("mlbot_console.main")
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def test_hysteresis_active_enter_exit():
    vals = [0.2, 0.55, 0.45, 0.25, 0.35]
    out = _hysteresis_active(vals, entry_min=0.50, exit_below=0.32)
    assert out == [False, True, True, False, False]


def test_chop_grid_regime_exit_markers(bus_root) -> None:
    from mlbot_console.services.strategy_stage_regions import (
        load_chop_grid_regime_exit_markers,
    )

    sym = "ETHUSDT"
    feat_dir = bus_root / "features" / "120T"
    path = feat_dir / f"{sym}.parquet"
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i in range(6):
        chop = 0.55 if i < 3 else 0.25
        rows.append(
            {
                "timestamp": start + pd.Timedelta(hours=2 * i),
                "bpc_semantic_chop": chop,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    markers = load_chop_grid_regime_exit_markers(
        bus_root,
        sym,
        "2h",
        STRATEGIES_ROOT,
        start=start,
        end=start + pd.Timedelta(days=2),
    )
    assert len(markers) == 1
    assert markers[0]["event"] == "exit"
    assert markers[0]["detail"]["exit_kind"] == "regime_or_risk_exit"


def test_evaluate_prefilter_all_of():
    rules = [
        {
            "all_of": [
                {"feature": "box_pos_60", "operator": ">=", "value": 0.35},
                {"feature": "box_pos_60", "operator": "<=", "value": 0.65},
            ]
        }
    ]
    assert _evaluate_prefilter_rules(rules, {"box_pos_60": 0.5})
    assert not _evaluate_prefilter_rules(rules, {"box_pos_60": 0.2})


def test_chop_grid_prefilter_regions(bus_root) -> None:
    sym = "ETHUSDT"
    feat_dir = bus_root / "features" / "120T"
    path = feat_dir / f"{sym}.parquet"
    df = pd.read_parquet(path)
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(0, len(df), max(1, len(df) // 12)):
        ts = df.iloc[i]["timestamp"]
        rows.append(
            {
                "timestamp": ts,
                "close": 100.0,
                "bpc_semantic_chop": 0.55 if i % 2 == 0 else 0.40,
                "box_pos_60": 0.50,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)

    regions = load_chop_grid_prefilter_regions(
        bus_root,
        sym,
        "2h",
        STRATEGIES_ROOT,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert isinstance(regions, list)


def test_bundle_stage_regions_multi_leg_scope(bus_root) -> None:
    out = load_bundle_stage_regions(
        bus_root,
        STRATEGIES_ROOT,
        "ETHUSDT",
        "2h",
        scopes=["multi_leg"],
        include_prefilter=True,
        include_gate=False,
    )
    assert "chop_grid" in out or out == {}


def test_trend_prefilter_skips_timestamp_only_parquet(bus_root, caplog) -> None:
    """Timestamp-only bus parquet must not evaluate prefilter per row (log spam)."""
    sym = "XRPUSDT"
    path = bus_root / "features" / "120T" / f"{sym}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]}).to_parquet(
        path, index=False
    )

    with caplog.at_level("INFO"):
        regions = load_bundle_stage_regions(
            bus_root,
            STRATEGIES_ROOT,
            sym,
            "2h",
            scopes=["trend"],
            include_prefilter=True,
            include_gate=False,
        )

    assert regions.get("tpc", {}).get("prefilter") is None
    assert "Prefilter feature" not in caplog.text
    assert "skip stage evaluation" in caplog.text
