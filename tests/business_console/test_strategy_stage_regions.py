"""Strategy prefilter / gate regions for Trade Map."""

from __future__ import annotations

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


def test_hysteresis_active_enter_exit():
    vals = [0.2, 0.55, 0.45, 0.25, 0.35]
    out = _hysteresis_active(vals, entry_min=0.50, exit_below=0.32)
    assert out == [False, True, True, False, False]


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
