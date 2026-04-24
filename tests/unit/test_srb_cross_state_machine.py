"""Unit tests for the SRB SR-cross state machine (纯函数)。

覆盖场景：
  1. detect_cross up/down/无穿越
  2. advance → 连续 K 根同侧 → confirmed
  3. advance → 连续 K 根反侧 → fake
  4. wick-prior + 低量 → 打 fake_stage，K 根 fail 后落 fake
  5. bar0 + fake_lookahead 内未决 → expired
  6. 有持仓时不起新候选 / cooldown 生效
"""

from __future__ import annotations

import pytest

from src.time_series_model.live.srb_cross_state_machine import (
    CrossCandidate,
    CrossConfig,
    detect_cross,
    update_cross_state,
)


# ---------------------------------------------------------------------------
# detect_cross
# ---------------------------------------------------------------------------


def test_detect_cross_up_break():
    assert detect_cross(100.0, 101.5, support=95.0, resistance=101.0) == ("up", 101.0)


def test_detect_cross_down_break():
    assert detect_cross(100.0, 94.5, support=95.0, resistance=105.0) == ("down", 95.0)


def test_detect_cross_none_when_inside_range():
    assert detect_cross(100.0, 100.5, support=95.0, resistance=101.0) is None


def test_detect_cross_prefers_resistance_when_both():
    # 不应同时触发（价格从下往上同时穿越支撑+阻力是不合理的），
    # 但按实现优先级 resistance 应优先。
    got = detect_cross(95.0, 105.5, support=95.0, resistance=105.0)
    assert got == ("up", 105.0)


def test_detect_cross_invalid_inputs():
    assert detect_cross(None, 101.0, 95.0, 101.0) is None
    assert detect_cross(100.0, None, 95.0, 101.0) is None
    assert detect_cross(100.0, 101.0, None, None) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**over):
    return CrossConfig(
        enabled=True,
        confirm_k=over.get("confirm_k", 3),
        fake_lookahead=over.get("fake_lookahead", 10),
        wick_ratio_threshold=over.get("wick_ratio_threshold", 2.0),
        low_vol_ratio=over.get("low_vol_ratio", 0.8),
        cooldown_bars=over.get("cooldown_bars", 10),
        max_reverse_per_level=over.get("max_reverse_per_level", 1),
    )


def _step(cand, idx, close_prev, close_curr, *, has_position=False, cooldown=0, **kw):
    return update_cross_state(
        candidate=cand,
        bar_index=idx,
        close_prev=close_prev,
        close_curr=close_curr,
        support=95.0,
        resistance=101.0,
        has_position=has_position,
        cfg=kw.pop("cfg", _cfg()),
        cooldown_until_bar=cooldown,
        open_px=kw.get("open_px"),
        high_px=kw.get("high_px"),
        low_px=kw.get("low_px"),
        volume=kw.get("volume"),
        volume_ma=kw.get("volume_ma"),
    )


# ---------------------------------------------------------------------------
# 2. confirmed
# ---------------------------------------------------------------------------


def test_advance_confirmed_after_three_consecutive():
    cfg = _cfg(confirm_k=3)
    # bar1: 起候选（close 101.5 已越过 resistance=101）→ confirm_count=1
    cand, dec = _step(None, 1, 100.0, 101.5, cfg=cfg)
    assert dec.status == "pending"
    assert cand is not None and cand.direction == "up"
    # bar2: close 102 同侧 → confirm_count=2
    cand, dec = _step(cand, 2, 101.5, 102.0, cfg=cfg)
    assert dec.status == "pending" and cand.confirm_count == 2
    # bar3: 再同侧 → confirm_count=3 ≥ K → confirmed
    cand, dec = _step(cand, 3, 102.0, 102.5, cfg=cfg)
    assert dec.status == "confirmed"
    assert dec.side == "LONG"
    assert dec.level == pytest.approx(101.0)
    assert cand is None


def test_advance_confirmed_short_on_down_break():
    cfg = _cfg(confirm_k=2)
    cand, dec = _step(None, 1, 96.0, 94.0, cfg=cfg)
    assert dec.status == "pending" and cand.direction == "down"
    cand, dec = _step(cand, 2, 94.0, 93.5, cfg=cfg)
    assert dec.status == "confirmed" and dec.side == "SHORT"


# ---------------------------------------------------------------------------
# 3. fake via fail_count
# ---------------------------------------------------------------------------


def test_advance_fake_after_three_wrong_side():
    cfg = _cfg(confirm_k=3)
    # bar1：up-break 起候选（confirm=1）
    cand, dec = _step(None, 1, 100.0, 101.5, cfg=cfg)
    assert dec.status == "pending"
    # bar2–4：close 连续回到 level 下方（< 101）→ fail_count 累积
    cand, dec = _step(cand, 2, 101.5, 100.5, cfg=cfg)
    assert dec.status == "pending" and cand.fail_count == 1
    cand, dec = _step(cand, 3, 100.5, 100.2, cfg=cfg)
    assert dec.status == "pending" and cand.fail_count == 2
    cand, dec = _step(cand, 4, 100.2, 99.8, cfg=cfg)
    assert dec.status == "fake"
    # up-break 假突破 → 反向 SHORT
    assert dec.side == "SHORT"
    assert cand is None


# ---------------------------------------------------------------------------
# 4. wick prior + low volume
# ---------------------------------------------------------------------------


def test_advance_wick_prior_low_volume_triggers_fake():
    cfg = _cfg(confirm_k=2, wick_ratio_threshold=2.0, low_vol_ratio=0.8)
    # bar1: 起候选；body 小(开高 101.3/收 101.4 → body=0.1)，上影很大(high=105 → upper=3.6)
    # volume 低(1.0)，volume_ma 高(10) → wick_prior 应命中 → fake_stage=True (count=1)
    cand, dec = _step(
        None,
        1,
        100.0,
        101.4,
        cfg=cfg,
        open_px=101.3,
        high_px=105.0,
        low_px=101.0,
        volume=1.0,
        volume_ma=10.0,
    )
    assert dec.status == "pending" and cand.fake_stage is True
    # bar2: close 回到 level 下方 → fail_count=1, fake_stage_count=2 ≥ K → fake
    cand, dec = _step(cand, 2, 101.4, 100.5, cfg=cfg)
    assert dec.status == "fake" and dec.side == "SHORT"


# ---------------------------------------------------------------------------
# 5. expired
# ---------------------------------------------------------------------------


def test_advance_expired_when_lookahead_exceeded():
    # confirm_k=10 足够高 → bar5 既未达 confirm 也未达 fail 计数，
    # 但 bar_index - bar0 = 4 > fake_lookahead=3 → expired。
    cfg = _cfg(confirm_k=10, fake_lookahead=3)
    cand, dec = _step(None, 1, 100.0, 101.5, cfg=cfg)  # bar0=1, confirm=1
    assert dec.status == "pending"
    # bar2–4：close 交替 —— 保证 confirm_count / fail_count 都不会冲到 10
    cand, dec = _step(cand, 2, 101.5, 100.5, cfg=cfg)  # fail=1
    cand, dec = _step(cand, 3, 100.5, 101.5, cfg=cfg)  # confirm=2
    cand, dec = _step(cand, 4, 101.5, 100.5, cfg=cfg)  # fail=2
    assert dec.status == "pending"
    # bar5：bar_index - bar0 = 4 > lookahead=3 → expired
    cand, dec = _step(cand, 5, 100.5, 101.5, cfg=cfg)
    assert dec.status == "expired"
    assert cand is None


# ---------------------------------------------------------------------------
# 6. has_position / cooldown
# ---------------------------------------------------------------------------


def test_has_position_prevents_new_candidate():
    cfg = _cfg()
    cand, dec = _step(None, 1, 100.0, 101.5, has_position=True, cfg=cfg)
    assert cand is None and dec.status == "idle"


def test_cooldown_blocks_new_candidate():
    cfg = _cfg(cooldown_bars=5)
    # bar 5，但 cooldown 到 bar 10 → 即便有 cross 也不起候选
    cand, dec = _step(None, 5, 100.0, 101.5, cooldown=10, cfg=cfg)
    assert cand is None and dec.status == "idle"
    # bar 11 cooldown 过期 → 可以起候选
    cand, dec = _step(None, 11, 100.0, 101.5, cooldown=10, cfg=cfg)
    assert cand is not None and dec.status == "pending"


# ---------------------------------------------------------------------------
# CrossConfig.from_mapping
# ---------------------------------------------------------------------------


def test_cross_config_from_mapping_defaults():
    cfg = CrossConfig.from_mapping(None)
    assert cfg.enabled is True and cfg.confirm_k == 3


def test_cross_config_from_mapping_overrides():
    cfg = CrossConfig.from_mapping(
        {
            "enabled": False,
            "confirm_k": 2,
            "fake_lookahead": 7,
            "cooldown_bars": 3,
        }
    )
    assert cfg.enabled is False
    assert cfg.confirm_k == 2
    assert cfg.fake_lookahead == 7
    assert cfg.cooldown_bars == 3
