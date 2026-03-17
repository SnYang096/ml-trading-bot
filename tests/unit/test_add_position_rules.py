from __future__ import annotations

import pytest

from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_size_multiplier,
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
