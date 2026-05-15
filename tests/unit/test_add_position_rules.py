from __future__ import annotations

import pytest

from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_max_times,
    resolve_add_position_size_multiplier,
    resolve_float_r_ladder_only,
    validate_add_position_trigger,
)


def test_resolve_add_position_max_times_infers_longest_non_empty_vector():
    assert (
        resolve_add_position_max_times(
            {
                "max_add_times": 7,
                "add_size_multipliers": [1, 2],
                "min_current_r_by_add": [2, 4],
            }
        )
        == 2
    )
    assert resolve_add_position_max_times({"max_add_times": 3}) == 3


def test_resolve_add_position_max_times_fallback_without_vectors():
    assert resolve_add_position_max_times(None) == 1
    assert resolve_add_position_max_times({}) == 1
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [3.0],
    }
    signal = {
        "current_leverage": 1.8,
        "base_leverage_unit": 1.2,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # gap = 1.2, base_unit=1.2 -> 1.0x
    assert mult == pytest.approx(1.0)


def test_target_leverage_gap_respects_max_total_leverage():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [5.0],
        "max_total_leverage": 3.0,
    }
    signal = {
        "current_leverage": 2.8,
        "base_leverage_unit": 1.0,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # target gap=2.2, but max_total room=0.2
    assert mult == pytest.approx(0.2)


def test_target_leverage_gap_falls_back_when_gap_non_positive():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [2.0],
        "add_size_multipliers": [0.35],
    }
    signal = {
        "current_leverage": 2.2,
        "base_leverage_unit": 1.0,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    assert mult == pytest.approx(0.35)


def test_target_leverage_gap_applies_notional_caps():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [5.0],
        "max_add_notional_frac": 0.30,
    }
    signal = {
        "current_leverage": 1.0,
        "base_leverage_unit": 1.0,
        "base_notional_frac": 0.10,
        "current_notional_frac": 0.25,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # notional room = 0.05, base_notional_frac=0.10 -> 0.5x
    assert mult == pytest.approx(0.5)


def test_resolve_float_r_ladder_only_from_trigger_type_only():
    assert (
        resolve_float_r_ladder_only({"trigger": {"type": "float_r_ladder_only"}})
        is True
    )
    assert (
        resolve_float_r_ladder_only({"trigger": {"type": "bpc_follow_signal"}}) is False
    )
    assert resolve_float_r_ladder_only({}) is False
    assert resolve_float_r_ladder_only({"trigger": {}}) is False
    assert resolve_float_r_ladder_only({"float_r_ladder_only": True}) is False


def test_validate_add_trigger_float_r_ladder_only_only_checks_min_r():
    cfg = {
        "min_current_r_by_add": [0.5],
        "trigger": {"type": "float_r_ladder_only"},
    }
    signal = {"add_position_seq": 1}
    assert (
        validate_add_position_trigger(
            archetype="bpc-long-120T",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is True
    )


def test_validate_add_trigger_without_trigger_still_checks_min_current_r():
    cfg = {
        "min_current_r_by_add": [0.5, 1.0, 1.5],
    }
    signal = {"add_position_seq": 2}
    ok = validate_add_position_trigger(
        archetype="bpc-long-120T",
        direction=1,
        signal=signal,
        add_position_cfg=cfg,
        current_r=0.8,  # below add #2 threshold 1.0
    )
    assert ok is False


def test_validate_add_trigger_without_trigger_passes_when_min_current_r_met():
    cfg = {
        "min_current_r_by_add": [0.5, 1.0, 1.5],
    }
    signal = {"add_position_seq": 2}
    ok = validate_add_position_trigger(
        archetype="bpc-long-120T",
        direction=1,
        signal=signal,
        add_position_cfg=cfg,
        current_r=1.05,  # above add #2 threshold 1.0
    )
    assert ok is True
