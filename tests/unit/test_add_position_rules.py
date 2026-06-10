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


# ── signal_add 路径回归测试（2026-06-10 backtester.py 修复后）──


def test_no_trigger_type_means_signal_add_path():
    """无 trigger.type 时 resolve_float_r_ladder_only 返回 False → signal_add 路径。

    事件回测中 _strats_float_ladder_meta 仅包含 float_r_ladder_only archetype，
    无 trigger 的 archetype 走 signal_add（PCM 再信号时加仓）。
    此前 backtester.py 的 _dup_open elif 在 _add_pos_enabled 之前拦截导致 signal_add 永为 0。
    """
    cfg_no_trigger = {
        "add_size_multipliers": [0.5, 0.25],
        "min_current_r_by_add": [0.5, 1],
    }
    assert resolve_float_r_ladder_only(cfg_no_trigger) is False

    cfg_empty_trigger = {
        "add_size_multipliers": [0.5],
        "trigger": {},
    }
    assert resolve_float_r_ladder_only(cfg_empty_trigger) is False


def test_signal_add_path_trigger_validates_min_r_only():
    """signal_add 路径（无 trigger.type）仅检查 min_current_r_by_add，不检查特征。

    这是设计意图：signal_add 的信号已经通过了 PCM 的 entry pipeline
    （prefilter→gate→direction→entry_filter→PCM仲裁），加仓时只需确认浮盈门槛。
    """
    cfg = {
        "min_current_r_by_add": [0.5, 1.0],
        # 无 trigger — signal_add 路径
    }
    signal = {"add_position_seq": 1, "position_action": "LONG"}
    # current_r 低于门槛 → 拒绝
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.3,
        )
        is False
    )
    # current_r 高于门槛 → 通过（不检查任何特征）
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is True
    )


def test_signal_add_vs_float_ladder_trigger_distinction():
    """确保 signal_add（无 trigger）和 float_r_ladder_only 的区分不会退化。

    两者都只检查 min_current_r_by_add，但路径不同：
    - signal_add: PCM 每次给新信号时触发一次（sparse）
    - float_r_ladder_only: 每 bar 检查（dense）
    """
    signal_cfg = {"min_current_r_by_add": [0.5]}
    float_cfg = {
        "min_current_r_by_add": [0.5],
        "trigger": {"type": "float_r_ladder_only"},
    }

    assert resolve_float_r_ladder_only(signal_cfg) is False
    assert resolve_float_r_ladder_only(float_cfg) is True

    signal = {"add_position_seq": 1}
    # 相同 current_r 下两者 trigger 验证结果一致
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=signal_cfg,
            current_r=0.6,
        )
        is True
    )
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=float_cfg,
            current_r=0.6,
        )
        is True
    )
