"""Unit tests for SRB regime / SR level helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.live.srb_regime import (
    maybe_inject_srb_experiment_features,
    path_efficiency_last,
    pick_srb_true_sr_level,
    resolve_regime_bucket,
    should_reject_srb_wide_entry,
    srb_add_position_allowed,
)
from src.time_series_model.core.trade_intent import TradeIntent


def test_path_efficiency_last_trivial():
    c = np.array([1.0, 1.0, 1.0, 2.0], dtype=float)
    er = path_efficiency_last(c, window=3)
    assert abs(er - 1.0) < 1e-9


def test_srb_add_position_allowed_regime_and_compression():
    pol = {
        "enabled": True,
        "allow_regime_buckets": ["high_adx_low_er"],
        "max_volume_compression_pct": 0.5,
        "volume_compression_feature": "bpc_volume_compression_pct",
    }
    ok, why = srb_add_position_allowed(
        {"srb_regime_bucket": "low_adx_low_er", "bpc_volume_compression_pct": 0.1},
        pol,
    )
    assert not ok and why == "srb_policy_regime_bucket"
    ok2, why2 = srb_add_position_allowed(
        {"srb_regime_bucket": "high_adx_low_er", "bpc_volume_compression_pct": 0.6},
        pol,
    )
    assert not ok2 and why2 == "srb_policy_volume_compression"
    ok3, _ = srb_add_position_allowed(
        {"srb_regime_bucket": "high_adx_low_er", "bpc_volume_compression_pct": 0.4},
        pol,
    )
    assert ok3


def test_resolve_regime_bucket():
    assert (
        resolve_regime_bucket(50.0, 0.2, {"adx_high": 40.0, "er_high": 0.36})
        == "high_adx_low_er"
    )
    assert (
        resolve_regime_bucket(50.0, 0.5, {"adx_high": 40.0, "er_high": 0.36})
        == "high_adx_high_er"
    )


def test_maybe_inject_disabled():
    idx = pd.date_range("2024-01-01", periods=30, freq="2h", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": np.linspace(1, 1.5, len(idx))},
        index=idx,
    )
    out = maybe_inject_srb_experiment_features(
        df=df, ts=idx[25], exec_raw={}, out={"close": 1.2}
    )
    assert "srb_regime_adx14" not in out


def test_maybe_inject_wide_sr_levels():
    idx = pd.date_range("2024-01-01", periods=120, freq="2h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": np.linspace(1.05, 2.0, len(idx)),
            "low": np.linspace(0.95, 1.9, len(idx)),
            "close": np.linspace(1.0, 1.95, len(idx)),
        },
        index=idx,
    )
    out = maybe_inject_srb_experiment_features(
        df=df,
        ts=idx[100],
        exec_raw={
            "fake_break_reverse": {"enabled": True},
            "sr_feature_injection": {"swing_lookback_wide_bars": 96},
        },
        out={},
    )
    assert "srb_sr_support" in out and "srb_sr_support_wide" in out
    assert np.isfinite(float(out["srb_sr_support_wide"]))
    assert np.isfinite(float(out["srb_sr_resistance_wide"]))


def test_execution_param_generator_regime_and_sr():
    raw = {
        "stop_loss": {
            "initial_r": 6.0,
            "trailing": {"enabled": True, "activation_r": 6.0, "trail_r": 5.0},
        },
        "take_profit": {"enabled": False},
        "holding": {"time_stop_bars": 0},
        "execution_constraints": {"allow_add_on": False},
        "regime_execution": {
            "enabled": True,
            "buckets": {
                "high_adx_high_er": {
                    "initial_r": 4.5,
                    "activation_r": 5.0,
                    "trail_r": 4.5,
                },
            },
        },
        "sr_structural_exit": {"enabled": True, "buffer_atr": 0.2, "lookback_bars": 5},
    }
    gen = ExecutionParamGenerator(raw)
    feats = {
        "srb_regime_bucket": "high_adx_high_er",
        "srb_sr_support": 0.95,
        "srb_sr_resistance": 1.2,
    }
    p = gen.generate_params(0.5, features=feats, direction=1)
    assert p["initial_r"] == 4.5
    assert p["structural_exit"] == "sr_break_level"
    assert abs(p["sr_exit_price"] - 0.95) < 1e-9


def _make_reverse_candidate(
    *,
    original_side="LONG",
    true_sr_level=100.0,
    stop_hunt_extreme=94.4,
    entry_price=100.0,
    sl_bar=10,
    sl_price=95.0,
    atr_at_entry=2.0,
):
    return {
        "true_sr_level": true_sr_level,
        "stop_hunt_extreme": stop_hunt_extreme,
        "entry_price": entry_price,
        "original_side": original_side,
        "sl_bar": sl_bar,
        "sl_price": sl_price,
        "reclaim_count": 0,
        "confirm_count": 0,
        "recover_stage": False,
        "used": False,
        "symbol": "BTCUSDT",
        "atr_at_entry": atr_at_entry,
        "tier_name": "global",
        "evidence_score": 0.5,
    }


def test_check_srb_reverse_two_stage_long():
    """Two-stage confirmation: reclaim from extreme, then confirm at SR."""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "reclaim_k": 1,
        "confirm_k": 2,
        "fake_lookahead": 8,
        "cooldown_bars": 3,
        "stop_hunt_buffer_atr": 0.3,
    }
    sim._primary_bar_count = 10
    # LONG stopped out at 95, extreme = 94.4, true_sr = 100
    sim._reverse_candidate = _make_reverse_candidate()

    # bar 11: price still below extreme → reclaim_count stays 0
    assert sim.check_srb_reverse(93.0, 11) is None
    assert not sim._reverse_candidate.get("recover_stage")
    assert sim._reverse_candidate["reclaim_count"] == 0

    # bar 12: price above extreme (94.4) but below true_sr → reclaim_k=1 met
    assert sim.check_srb_reverse(96.0, 12) is None
    assert sim._reverse_candidate.get("recover_stage")
    # stage 2 starts: 96 < 100 → confirm_count = 0
    assert sim._reverse_candidate["confirm_count"] == 0

    # bar 13: price above true_sr (100) → confirm 1
    assert sim.check_srb_reverse(101.0, 13) is None
    assert sim._reverse_candidate["confirm_count"] == 1

    # bar 14: price above true_sr → confirm 2 → triggers
    rev = sim.check_srb_reverse(102.0, 14)
    assert rev is not None
    assert rev["side"] == "LONG"
    assert rev["symbol"] == "BTCUSDT"
    assert rev["true_sr_level"] == 100.0
    assert sim._reverse_candidate["used"]
    assert sim._reverse_cooldown_until_bar == 14 + 3


def test_check_srb_reverse_two_stage_short():
    """Two-stage confirmation for SHORT: reclaim down from extreme, confirm below SR."""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "reclaim_k": 1,
        "confirm_k": 1,
        "fake_lookahead": 5,
        "cooldown_bars": 5,
        "stop_hunt_buffer_atr": 0.3,
    }
    sim._primary_bar_count = 20
    # SHORT stopped out at 105, extreme = 105.9, true_sr = 100
    sim._reverse_candidate = _make_reverse_candidate(
        original_side="SHORT",
        true_sr_level=100.0,
        stop_hunt_extreme=105.9,
        sl_bar=20,
        sl_price=105.0,
    )

    # bar 21: price below extreme → reclaim
    assert sim.check_srb_reverse(104.0, 21) is None
    assert sim._reverse_candidate.get("recover_stage")

    # bar 22: price below true_sr → confirm 1 → triggers
    rev = sim.check_srb_reverse(99.0, 22)
    assert rev is not None
    assert rev["side"] == "SHORT"
    assert sim._reverse_candidate["used"]


def test_check_srb_reverse_expires():
    """反手候选超出 fake_lookahead 后过期。"""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "reclaim_k": 1,
        "confirm_k": 2,
        "fake_lookahead": 2,
        "cooldown_bars": 5,
        "stop_hunt_buffer_atr": 0.3,
    }
    sim._primary_bar_count = 10
    sim._reverse_candidate = _make_reverse_candidate(
        original_side="SHORT",
        true_sr_level=100.0,
        stop_hunt_extreme=105.9,
        sl_bar=10,
        sl_price=105.0,
    )
    # bar 13: 3 bars since sl_bar=10, lookahead=2 → expired
    assert sim.check_srb_reverse(98.0, 13) is None
    assert sim._reverse_candidate is None
    assert sim._last_reverse_status == "expired"


def test_check_srb_reverse_reclaim_resets():
    """Reclaim count resets when price dips back below extreme."""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "reclaim_k": 2,
        "confirm_k": 1,
        "fake_lookahead": 10,
        "cooldown_bars": 3,
        "stop_hunt_buffer_atr": 0.3,
    }
    sim._primary_bar_count = 10
    sim._reverse_candidate = _make_reverse_candidate()

    # bar 11: above extreme → reclaim 1
    assert sim.check_srb_reverse(95.0, 11) is None
    assert sim._reverse_candidate["reclaim_count"] == 1
    assert not sim._reverse_candidate.get("recover_stage")

    # bar 12: drops below extreme → reclaim resets
    assert sim.check_srb_reverse(93.0, 12) is None
    assert sim._reverse_candidate["reclaim_count"] == 0

    # bar 13-14: above extreme for 2 consecutive → reclaim_k met
    assert sim.check_srb_reverse(95.0, 13) is None
    assert sim._reverse_candidate["reclaim_count"] == 1
    assert sim.check_srb_reverse(96.0, 14) is None
    assert sim._reverse_candidate.get("recover_stage")


def test_structural_sr_break_long():
    intent = TradeIntent(
        action="LONG",
        symbol="X",
        archetype="srb",
        execution_profile={
            "rr_constraints": {
                "stop_loss_r": 2.0,
                "take_profit_r": 0.0,
                "allow_trailing": False,
                "structural_exit": "sr_break_level",
                "sr_exit_price": 100.0,
                "sr_exit_buffer_atr": 0.25,
                "min_stop_pct": None,
                "max_stop_pct": None,
            }
        },
    )
    pos = build_position_dict(intent, entry_price=110.0, atr=1.0, bar_minutes=120)
    reason, px = enforce_position(
        pos,
        price_high=105.0,
        price_low=98.0,
        price_close=98.0,
        now=pos["entry_time"],
        default_bar_minutes=120,
    )
    assert reason == "structural_exit_sr_break"


def test_should_reject_srb_wide_entry_long_and_short():
    assert not should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 103.0, 2.0)
    assert should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 101.9, 2.0)
    assert not should_reject_srb_wide_entry("SHORT", 100.0, 1.0, 97.0, None, 2.0)
    assert should_reject_srb_wide_entry("SHORT", 100.0, 1.0, 98.1, None, 2.0)
    assert not should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 104.0, 2.0)


def test_pick_srb_true_sr_level_fallback_and_no_fallback():
    ts = pick_srb_true_sr_level(
        "LONG",
        100.5,
        1.0,
        narrow_support=100.0,
        narrow_resistance=105.0,
        wide_support=95.0,
        wide_resistance=110.0,
        fallback_atr=2.0,
    )
    assert abs(ts - 95.0) < 1e-9

    ts2 = pick_srb_true_sr_level(
        "SHORT",
        99.5,
        1.0,
        narrow_support=92.0,
        narrow_resistance=100.0,
        wide_support=90.0,
        wide_resistance=105.0,
        fallback_atr=2.0,
    )
    assert abs(ts2 - 105.0) < 1e-9

    ts3 = pick_srb_true_sr_level(
        "LONG",
        105.0,
        1.0,
        narrow_support=100.0,
        narrow_resistance=110.0,
        wide_support=95.0,
        wide_resistance=115.0,
        fallback_atr=2.0,
    )
    assert abs(ts3 - 100.0) < 1e-9


def test_build_position_dict_propagates_srb_true_sr_level():
    intent = TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype="srb",
        confidence=0.5,
        execution_profile={
            "rr_constraints": {"stop_loss_r": 6.0},
            "strategy_specific": {"srb_true_sr_level": 98765.0},
        },
    )
    pos = build_position_dict(intent, entry_price=99000.0, atr=500.0)
    assert pos.get("_srb_true_sr_level") == 98765.0
