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
    should_reject_srb_add_by_shape,
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


def test_maybe_inject_narrow_sr_only():
    """
    L1 窄窗 SR 依然由 srb_regime 注入；L3 大级别 SR 已统一由 wide_sr_swing_f 特征管线提供，
    不再在此函数里重复计算（不应看到 srb_sr_*_wide）。
    """
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
            "sr_wide_entry_guard": {"enabled": True},
            "sr_structural_exit": {"enabled": True, "lookback_bars": 20},
        },
        out={},
    )
    assert "srb_sr_support" in out and "srb_sr_resistance" in out
    assert np.isfinite(float(out["srb_sr_support"]))
    assert np.isfinite(float(out["srb_sr_resistance"]))
    # L3 大级别 SR 不再由 inject 写入
    assert "srb_sr_support_wide" not in out
    assert "srb_sr_resistance_wide" not in out


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
    # 参数顺序: (side, close, atr, wide_lower_px, wide_upper_px, min_distance_atr)
    # LONG：看上方 wide_upper_px 是否过近
    assert not should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 103.0, 2.0)
    assert should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 101.9, 2.0)
    # SHORT：看下方 wide_lower_px 是否过近
    assert not should_reject_srb_wide_entry("SHORT", 100.0, 1.0, 97.0, None, 2.0)
    assert should_reject_srb_wide_entry("SHORT", 100.0, 1.0, 98.1, None, 2.0)
    # 边界：距离正好 = 阈值 → 不拦截
    assert not should_reject_srb_wide_entry("LONG", 100.0, 1.0, None, 104.0, 2.0)


def test_pick_srb_true_sr_level_fallback_and_no_fallback():
    # LONG: 窄窗 support=100.0 距 entry 100.5 仅 0.5 < 2×ATR → 回退到 L3 下沿
    ts = pick_srb_true_sr_level(
        "LONG",
        100.5,
        1.0,
        narrow_support=100.0,
        narrow_resistance=105.0,
        wide_lower_px=95.0,
        wide_upper_px=110.0,
        fallback_atr=2.0,
    )
    assert abs(ts - 95.0) < 1e-9

    # SHORT: 窄窗 resistance=100.0 距 entry 99.5 仅 0.5 < 2×ATR → 回退到 L3 上沿
    ts2 = pick_srb_true_sr_level(
        "SHORT",
        99.5,
        1.0,
        narrow_support=92.0,
        narrow_resistance=100.0,
        wide_lower_px=90.0,
        wide_upper_px=105.0,
        fallback_atr=2.0,
    )
    assert abs(ts2 - 105.0) < 1e-9

    # LONG: 窄窗 support=100.0 距 entry 105.0 是 5×ATR ≥ 2×ATR → 不回退，用窄窗
    ts3 = pick_srb_true_sr_level(
        "LONG",
        105.0,
        1.0,
        narrow_support=100.0,
        narrow_resistance=110.0,
        wide_lower_px=95.0,
        wide_upper_px=115.0,
        fallback_atr=2.0,
    )
    assert abs(ts3 - 100.0) < 1e-9


def _shape_gate_cfg(**overrides):
    cfg = {
        "retrace_guard": {"enabled": False, "min_captured_pct": 0.7},
        "recent_momentum": {
            "enabled": False,
            "lookback_bars": 6,
            "min_net_move_atr": 1.5,
        },
        "trend_r2_gate": {"enabled": False, "min_r2": 0.4},
        "wide_sr_expansion": {"enabled": False, "min_expansion_atr": 1.0},
        "trend_health_gate": {
            "enabled": False,
            "min_mother_mfe_r": 1.0,
            "max_bars_since_mother_entry": 360,
        },
    }
    for k, v in overrides.items():
        cfg[k] = {**cfg[k], **v}
    return cfg


def test_shape_gate_all_disabled_passes():
    rej, why = should_reject_srb_add_by_shape({}, {}, _shape_gate_cfg())
    assert rej is False
    assert why == ""


def test_shape_gate_retrace_guard_rejects_when_current_r_deep_below_mfe():
    cfg = _shape_gate_cfg(retrace_guard={"enabled": True, "min_captured_pct": 0.7})
    # mfe=5, current=2 → captured=0.4 < 0.7 → reject
    rej, why = should_reject_srb_add_by_shape(
        {"mfe_r": 5.0, "current_r": 2.0}, {"side": "LONG"}, cfg
    )
    assert rej is True
    assert why == "shape_gate_retrace"
    # mfe=5, current=4 → captured=0.8 ≥ 0.7 → pass
    rej, _ = should_reject_srb_add_by_shape(
        {"mfe_r": 5.0, "current_r": 4.0}, {"side": "LONG"}, cfg
    )
    assert rej is False


def test_shape_gate_recent_momentum_long_and_short():
    cfg = _shape_gate_cfg(
        recent_momentum={"enabled": True, "lookback_bars": 6, "min_net_move_atr": 1.5}
    )
    # LONG: move 1.0 < 1.5 → reject
    rej, why = should_reject_srb_add_by_shape(
        {"recent_net_move_atr": 1.0}, {"side": "LONG"}, cfg
    )
    assert rej is True and why == "shape_gate_momentum"
    # LONG: move 2.0 ≥ 1.5 → pass
    rej, _ = should_reject_srb_add_by_shape(
        {"recent_net_move_atr": 2.0}, {"side": "LONG"}, cfg
    )
    assert rej is False
    # SHORT: move -1.0 > -1.5 → reject
    rej, why = should_reject_srb_add_by_shape(
        {"recent_net_move_atr": -1.0}, {"side": "SHORT"}, cfg
    )
    assert rej is True and why == "shape_gate_momentum"
    # SHORT: move -2.0 ≤ -1.5 → pass
    rej, _ = should_reject_srb_add_by_shape(
        {"recent_net_move_atr": -2.0}, {"side": "SHORT"}, cfg
    )
    assert rej is False


def test_shape_gate_trend_r2_gate():
    cfg = _shape_gate_cfg(trend_r2_gate={"enabled": True, "min_r2": 0.4})
    rej, why = should_reject_srb_add_by_shape(
        {"trend_r2_20": 0.2}, {"side": "LONG"}, cfg
    )
    assert rej is True and why == "shape_gate_r2"
    rej, _ = should_reject_srb_add_by_shape({"trend_r2_20": 0.5}, {"side": "LONG"}, cfg)
    assert rej is False


def test_shape_gate_wide_sr_expansion_requires_expansion():
    cfg = _shape_gate_cfg(wide_sr_expansion={"enabled": True, "min_expansion_atr": 1.0})
    # 入场时 dist=3，加仓时 dist=3.5 → 扩张 0.5 < 1.0 → reject
    rej, why = should_reject_srb_add_by_shape(
        {"wide_sr_dist_atr": 3.5},
        {"side": "LONG", "entry_wide_sr_dist_atr": 3.0},
        cfg,
    )
    assert rej is True and why == "shape_gate_wide_expansion"
    # 扩张 2.0 ≥ 1.0 → pass
    rej, _ = should_reject_srb_add_by_shape(
        {"wide_sr_dist_atr": 5.0},
        {"side": "LONG", "entry_wide_sr_dist_atr": 3.0},
        cfg,
    )
    assert rej is False


def test_shape_gate_trend_health_gate_mfe():
    """E4: 母仓 MFE < min_mother_mfe_r 时拒绝加仓（新 gate）"""
    cfg = _shape_gate_cfg(
        trend_health_gate={
            "enabled": True,
            "min_mother_mfe_r": 1.0,
            "max_bars_since_mother_entry": 360,
        }
    )
    # MFE = 0.4 < 1.0 → reject (mfe bucket)
    rej, why = should_reject_srb_add_by_shape(
        {"mfe_r": 0.4, "bars_since_mother_entry": 10},
        {"side": "LONG"},
        cfg,
    )
    assert rej is True and why == "shape_gate_trend_health_mfe"
    # MFE = 1.5 ≥ 1.0 且 bars 充足 → pass
    rej, _ = should_reject_srb_add_by_shape(
        {"mfe_r": 1.5, "bars_since_mother_entry": 50},
        {"side": "LONG"},
        cfg,
    )
    assert rej is False


def test_shape_gate_trend_health_gate_stale():
    """E4: 母仓入场 > max_bars_since_mother_entry 时拒绝（趋势未爆发却拖太久）"""
    cfg = _shape_gate_cfg(
        trend_health_gate={
            "enabled": True,
            "min_mother_mfe_r": 1.0,
            "max_bars_since_mother_entry": 360,
        }
    )
    # MFE 通过 gate 第一项，但 bars > 360 → stale 触发
    rej, why = should_reject_srb_add_by_shape(
        {"mfe_r": 1.5, "bars_since_mother_entry": 500},
        {"side": "LONG"},
        cfg,
    )
    assert rej is True and why == "shape_gate_trend_health_stale"


def test_shape_gate_trend_health_gate_disabled_is_noop():
    cfg = _shape_gate_cfg()  # 全部默认 off
    rej, _ = should_reject_srb_add_by_shape(
        {"mfe_r": 0.0, "bars_since_mother_entry": 1000}, {"side": "LONG"}, cfg
    )
    assert rej is False


def test_shape_gate_first_match_wins():
    # 两门同时 enabled 且都会 reject，retrace 先于 momentum 判断
    cfg = _shape_gate_cfg(
        retrace_guard={"enabled": True, "min_captured_pct": 0.7},
        recent_momentum={"enabled": True, "min_net_move_atr": 1.5},
    )
    rej, why = should_reject_srb_add_by_shape(
        {"mfe_r": 5.0, "current_r": 1.0, "recent_net_move_atr": 0.5},
        {"side": "LONG"},
        cfg,
    )
    assert rej is True and why == "shape_gate_retrace"


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
