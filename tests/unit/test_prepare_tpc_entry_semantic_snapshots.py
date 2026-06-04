"""Unit tests for prepare_tpc_entry_semantic_snapshots materialization."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from research.prepare_tpc_entry_semantic_snapshots import (  # noqa: E402
    SNAPSHOTS,
    _patch_direction_inner_abs,
    _patch_entry_filters,
    _patch_prefilter,
    _patch_regime_any_of,
)


def _minimal_tpc_tree(root: Path) -> Path:
    arch = root / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "prefilter.yaml").write_text(
        yaml.safe_dump(
            {"rules": [{"feature": "a", "operator": ">", "value": 0}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (arch / "entry_filters.yaml").write_text(
        yaml.safe_dump(
            {"filters": [{"id": "prod", "enabled": True, "conditions": []}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (arch / "regime.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "any_of": [
                            {
                                "feature": "ema_1200_position",
                                "operator": ">=",
                                "value": 0.05,
                            }
                        ]
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (arch / "direction.yaml").write_text(
        yaml.safe_dump(
            {
                "direction_rules": [
                    {
                        "id": "tpc_macd_ema1200_align",
                        "position_band": {"inner_abs": 0.0, "outer_abs": 0.85},
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root


def test_patch_prefilter_s50_depth_rule(tmp_path: Path):
    tree = _minimal_tpc_tree(tmp_path / "snap")
    rule = SNAPSHOTS["tpc_semantic_depth_gt50_strategies"]["prefilter_rules_append"][0]
    _patch_prefilter(tree, [rule])
    data = yaml.safe_load(
        (tree / "tpc/archetypes/prefilter.yaml").read_text(encoding="utf-8")
    )
    rules = data["rules"]
    assert rules[-1]["feature"] == "tpc_pullback_depth"
    assert rules[-1]["operator"] == ">"
    assert rules[-1]["value"] == 0.5


def test_patch_entry_filters_e2_anti_chase_bilateral(tmp_path: Path):
    tree = _minimal_tpc_tree(tmp_path / "snap")
    patch = SNAPSHOTS["tpc_entry_anti_chase_strategies"]["entry_filters_patch"]
    _patch_entry_filters(tree, patch)
    data = yaml.safe_load(
        (tree / "tpc/archetypes/entry_filters.yaml").read_text(encoding="utf-8")
    )
    assert data["combination_mode"] == "and"
    ids = [f["id"] for f in data["filters"]]
    assert "tpc_anti_chase_not_at_high" in ids
    assert "tpc_anti_chase_not_at_low" in ids
    long_f = next(f for f in data["filters"] if f["id"] == "tpc_anti_chase_not_at_high")
    short_f = next(f for f in data["filters"] if f["id"] == "tpc_anti_chase_not_at_low")
    assert long_f.get("direction") == "long"
    assert short_f.get("direction") == "short"
    assert long_f["conditions"][0]["feature"] == "bars_since_local_high"
    assert short_f["conditions"][0]["feature"] == "bars_since_local_low"


def test_patch_regime_and_direction_s51(tmp_path: Path):
    tree = _minimal_tpc_tree(tmp_path / "snap")
    spec = SNAPSHOTS["tpc_semantic_depth_gt50_ema_near_strategies"]
    _patch_regime_any_of(tree, spec["regime_any_of_replace"])
    _patch_direction_inner_abs(tree, float(spec["direction_inner_abs"]))
    regime = yaml.safe_load(
        (tree / "tpc/archetypes/regime.yaml").read_text(encoding="utf-8")
    )
    any_of = regime["rules"][0]["any_of"]
    assert any_of[0]["value"] == 0.03
    assert any_of[1]["value"] == -0.03
    direction = yaml.safe_load(
        (tree / "tpc/archetypes/direction.yaml").read_text(encoding="utf-8")
    )
    pb = direction["direction_rules"][0]["position_band"]
    assert pb["inner_abs"] == -0.1


def test_snapshot_names_cover_experiment_variants():
    assert "tpc_semantic_depth_gt50_strategies" in SNAPSHOTS
    assert "tpc_semantic_depth_gt50_ema_near_strategies" in SNAPSHOTS
    assert "tpc_entry_anti_chase_strategies" in SNAPSHOTS
    assert "tpc_exec_turbo_20260424_strategies" in SNAPSHOTS
