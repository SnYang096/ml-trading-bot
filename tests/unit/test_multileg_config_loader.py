from pathlib import Path

import pytest
import yaml

from scripts.diagnose_chop_grid import merge_chop_grid_yaml
from scripts.diagnose_dual_add_trend import _load_dual_add_defaults
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
        cfg_dir / "research/calibrate_roll.default.yaml",
        {
            "strategy_type": "grid",
            "status": "research",
            "live": {"mode": "dry_run"},
        },
    )
    _write_yaml(
        cfg_dir / "archetypes/regime.yaml",
        {"regime": {"entry_chop_min": 0.4, "exit_chop_below": 0.25}},
    )
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml",
        {"rules": [{"feature": "box_pos_60", "operator": ">=", "value": 0.35}]},
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {"inventory": {"spacing": {"atr_mult": 0.5, "min_pct": 0.004}}},
    )

    got = load_multileg_effective_config(config_dir=cfg_dir, strategy_type="grid")
    assert got["strategy_type"] == "grid"
    assert got["regime"]["entry_chop_min"] == 0.4
    assert got["rules"][0]["feature"] == "box_pos_60"
    assert got["inventory"]["spacing"]["min_pct"] == 0.004
    assert got["live"]["mode"] == "dry_run"


def test_load_multileg_effective_config_respects_custom_engine_path(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "custom_pack"
    _write_yaml(
        cfg_dir / "research/calibrate_roll.default.yaml", {"strategy_type": "grid"}
    )
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
        cfg_dir / "research/calibrate_roll.default.yaml",
        {
            "strategy_type": "dual_add_trend",
        },
    )
    _write_yaml(
        cfg_dir / "archetypes/regime.yaml",
        {
            "allowed_regimes": ["bull", "bear", "neutral"],
            "allowed_sides": ["long", "short"],
            "rules": [],
            "extensions": {"multileg": {}},
        },
    )
    _write_yaml(cfg_dir / "archetypes/prefilter.yaml", {"rules": []})
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
    reg = yaml.safe_load((cfg_dir / "archetypes/regime.yaml").read_text()) or {}
    exe = yaml.safe_load((cfg_dir / "archetypes/execution.yaml").read_text()) or {}
    assert reg["extensions"]["multileg"]["entry_min"] == 0.82
    assert exe["add_spacing"]["atr_mult"] == 0.65
    assert exe["take_profit"]["min_pct"] == 0.001
    assert exe["inventory"]["flip_action"] == "close_offside_all"


def test_load_multileg_effective_config_supports_archetype_only_package(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "archetype_only_pack"
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml", {"regime": {"entry_chop_min": 0.4}}
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {"inventory": {"spacing": {"atr_mult": 0.5}}},
    )
    got = load_multileg_effective_config(config_dir=cfg_dir, strategy_type="grid")
    assert got["strategy_type"] == "grid"
    assert got["status"] == "research"
    assert got["regime"]["entry_chop_min"] == 0.4
    assert got["inventory"]["spacing"]["atr_mult"] == 0.5


def test_load_multileg_effective_config_explicit_missing_profile_raises(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "missing_profile_pack"
    missing = cfg_dir / "research/calibrate_roll.default.yaml"
    with pytest.raises(ValueError, match="missing multileg profile yaml"):
        load_multileg_effective_config(
            config_dir=cfg_dir,
            strategy_type="grid",
            profile_path=missing,
        )


def test_chop_grid_cost_defaults_come_from_backtest_costs(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "chop_grid"
    _write_yaml(
        cfg_dir / "research/calibrate_roll.default.yaml",
        {
            "strategy_type": "grid",
            "grid_backtest": {
                "same_bar_entry_exit": True,
                "costs": {
                    "fee_bps": 7.0,
                    "maker_fee_bps": 2.0,
                    "taker_fee_bps": 9.0,
                    "forced_exit_slippage_bps": 11.0,
                    "funding_cost_bps_per_8h": 1.5,
                },
            },
        },
    )
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml",
        {"regime": {"entry_chop_min": 0.4, "exit_chop_below": 0.25}},
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {
            "inventory": {"spacing": {"atr_mult": 0.5, "min_pct": 0.004}},
            "risk": {"fee_bps": 99.0},
        },
    )

    got = merge_chop_grid_yaml(cfg_dir / "research/calibrate_roll.default.yaml")
    assert got["same_bar_entry_exit"] is True
    assert got["fee_bps"] == 7.0
    assert got["maker_fee_bps"] == 2.0
    assert got["taker_fee_bps"] == 9.0
    assert got["forced_exit_slippage_bps"] == 11.0
    assert got["funding_cost_bps_per_8h"] == 1.5


def test_dual_add_cost_defaults_come_from_backtest_costs(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "dual_add_trend"
    _write_yaml(
        cfg_dir / "research/calibrate_roll.default.yaml",
        {
            "strategy_type": "dual_add_trend",
            "dual_add_backtest": {
                "costs": {
                    "fee_bps": 12.0,
                    "market_exit_slippage_bps": 6.0,
                    "intrabar_touch_buffer_bps": 4.0,
                },
                "execution_timeframe": "1min",
                "scale_max_loser_hold_to_signal": True,
            },
        },
    )
    _write_yaml(
        cfg_dir / "archetypes/prefilter.yaml",
        {"regime": {"entry_min": 0.8, "exit_below": 0.5}},
    )
    _write_yaml(
        cfg_dir / "archetypes/execution.yaml",
        {
            "inventory": {"initial_legs": ["TREND"]},
            "add_spacing": {"atr_mult": 1.0},
            "take_profit": {"mode": "basket"},
            "risk": {"diagnostic_fee_bps": 99.0},
        },
    )

    got = _load_dual_add_defaults(cfg_dir / "research/calibrate_roll.default.yaml")
    assert got["fee_bps"] == 12.0
    assert got["market_exit_slippage_bps"] == 6.0
    assert got["intrabar_touch_buffer_bps"] == 4.0
    assert got["execution_timeframe"] == "1min"
    assert got["scale_max_loser_hold_to_signal"] is True
