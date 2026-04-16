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
    resolve_regime_bucket,
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


def test_check_srb_reverse_basic():
    """PositionSimulator.check_srb_reverse: 基本流程测试。"""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "confirm_k": 2,
        "fake_lookahead": 5,
        "cooldown_bars": 3,
    }
    sim._primary_bar_count = 10
    sim._reverse_candidate = {
        "sr_level": 100.0,
        "original_side": "LONG",
        "sl_bar": 10,
        "sl_price": 95.0,
        "confirm_count": 0,
        "used": False,
        "symbol": "BTCUSDT",
        "atr_at_entry": 2.0,
        "tier_name": "global",
        "evidence_score": 0.5,
    }
    # bar 11: price below sr_level → confirm resets
    assert sim.check_srb_reverse(99.0, 11) is None
    assert sim._reverse_candidate["confirm_count"] == 0
    # bar 12: price above → confirm 1
    assert sim.check_srb_reverse(101.0, 12) is None
    assert sim._reverse_candidate["confirm_count"] == 1
    # bar 13: price above → confirm 2 → triggers
    rev = sim.check_srb_reverse(102.0, 13)
    assert rev is not None
    assert rev["side"] == "LONG"
    assert rev["symbol"] == "BTCUSDT"
    assert sim._reverse_candidate["used"]
    assert sim._reverse_cooldown_until_bar == 13 + 3


def test_check_srb_reverse_expires():
    """反手候选超出 fake_lookahead 后过期。"""
    from scripts.event_backtest import PositionSimulator

    sim = PositionSimulator()
    sim._srb_reverse_policy = {
        "enabled": True,
        "confirm_k": 3,
        "fake_lookahead": 2,
        "cooldown_bars": 5,
    }
    sim._primary_bar_count = 10
    sim._reverse_candidate = {
        "sr_level": 100.0,
        "original_side": "SHORT",
        "sl_bar": 10,
        "sl_price": 105.0,
        "confirm_count": 0,
        "used": False,
        "symbol": "ETHUSDT",
        "atr_at_entry": 3.0,
        "tier_name": "global",
        "evidence_score": 0.5,
    }
    # bar 13: 3 bars since sl_bar=10, lookahead=2 → expired
    assert sim.check_srb_reverse(98.0, 13) is None
    assert sim._reverse_candidate is None
    assert sim._last_reverse_status == "expired"


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
