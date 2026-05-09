from pathlib import Path

import pytest
import yaml

from src.config.multileg_config import (
    load_multileg_effective_config,
    update_multileg_calibration_candidate,
)


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def test_load_multileg_effective_config_merges_archetype_layers(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "chop_grid"
    _write_yaml(
        cfg_dir / "research/turbo.yaml",
        {
            "strategy_type": "grid",
            "status": "research",
            "live": {"mode": "dry_run"},
        },
    )
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml",
        {"regime": {"entry_chop_min": 0.4, "exit_chop_below": 0.25}},
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {"grid": {"spacing": {"atr_mult": 0.5, "min_pct": 0.004}}},
    )

    got = load_multileg_effective_config(config_dir=cfg_dir, strategy_type="grid")
    assert got["strategy_type"] == "grid"
    assert got["regime"]["entry_chop_min"] == 0.4
    assert got["grid"]["spacing"]["min_pct"] == 0.004
    assert got["live"]["mode"] == "dry_run"


def test_load_multileg_effective_config_respects_custom_engine_path(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "custom_pack"
    _write_yaml(cfg_dir / "research/turbo.yaml", {"strategy_type": "grid"})
    _write_yaml(
        cfg_dir / "custom_grid.yaml",
        {
            "strategy_type": "grid",
            "archetypes": {"prefilter": "layers/prefilter.yaml"},
        },
    )
    _write_yaml(
        cfg_dir / "layers/prefilter.yaml",
        {"regime": {"entry_chop_min": 0.47}},
    )

    got = load_multileg_effective_config(
        config_dir=cfg_dir,
        strategy_type="grid",
        engine_path=cfg_dir / "custom_grid.yaml",
    )
    assert got["regime"]["entry_chop_min"] == 0.47


def test_update_multileg_candidate_writes_archetype_layers(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "dual_add_trend"
    _write_yaml(
        cfg_dir / "research/turbo.yaml",
        {
            "strategy_type": "dual_add_trend",
        },
    )
    _write_yaml(cfg_dir / "archetypes/prefilter.yaml", {"regime": {}})
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {"inventory": {}, "add_spacing": {}, "take_profit": {}},
    )

    update_multileg_calibration_candidate(
        config_dir=cfg_dir,
        strategy_type="dual_add_trend",
        candidate={
            "entry_min": 0.82,
            "step_atr_mult": 0.65,
            "tp_pct": 0.001,
            "flip_action": "close_offside_all",
        },
    )
    pre = yaml.safe_load((cfg_dir / "archetypes/prefilter.yaml").read_text()) or {}
    exe = yaml.safe_load((cfg_dir / "archetypes/execution.yaml").read_text()) or {}
    assert pre["regime"]["entry_min"] == 0.82
    assert exe["add_spacing"]["atr_mult"] == 0.65
    assert exe["take_profit"]["min_pct"] == 0.001
    assert exe["inventory"]["flip_action"] == "close_offside_all"


def test_load_multileg_effective_config_missing_profile_raises(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "missing_profile_pack"
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml", {"regime": {"entry_chop_min": 0.4}}
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml", {"grid": {"spacing": {"atr_mult": 0.5}}}
    )
    with pytest.raises(ValueError, match="missing multileg profile yaml"):
        load_multileg_effective_config(config_dir=cfg_dir, strategy_type="grid")
