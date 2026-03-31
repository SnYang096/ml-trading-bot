from __future__ import annotations

import pytest

from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_min_current_r,
    resolve_add_position_size_multiplier,
    validate_add_position_trigger,
)


def test_target_leverage_gap_uses_gap_to_compute_multiplier():
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


def test_resolve_min_current_r_converts_atr_unit_to_current_r():
    cfg = {
        "min_current_r_unit": "atr",
        "min_current_r_by_add": [0.5],
    }
    # parent_initial_r=4 => 0.5 ATR threshold equals 0.125 current_r.
    out = resolve_add_position_min_current_r(
        add_position_cfg=cfg,
        add_number=1,
        signal={"parent_initial_r": 4.0},
    )
    assert out == pytest.approx(0.125)


def test_validate_add_trigger_passes_with_atr_unit_when_current_r_meets_converted_threshold():
    cfg = {
        "min_current_r_unit": "atr",
        "min_current_r_by_add": [0.5],
    }
    ok = validate_add_position_trigger(
        archetype="bpc-long-120T",
        direction=1,
        signal={"add_position_seq": 1, "parent_initial_r": 4.0},
        add_position_cfg=cfg,
        current_r=0.13,  # above converted threshold 0.125
    )
    assert ok is True
