from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.order_management.multi_leg_kill_switch import (
    MultiLegKillSwitchConfig,
    MultiLegKillSwitchTracker,
)


def _tracker(tmp_path: Path, **cfg_overrides) -> MultiLegKillSwitchTracker:
    cfg = MultiLegKillSwitchConfig(
        enabled=True,
        daily_loss_limit=0.06,
        weekly_loss_limit=0.08,
        monthly_loss_limit=0.12,
        max_dd=0.20,
        cooldown_minutes=0,
        **cfg_overrides,
    )
    return MultiLegKillSwitchTracker(
        config=cfg,
        state_path=tmp_path / "kill_switch_state.json",
    )


def test_daily_loss_blocks_place_allows_market_exit(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=now)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=now)  # -7% daily

    assert tracker.is_halted()
    assert "daily_loss_limit" in tracker.halt_reasons()
    assert tracker.blocks_action("place") == "kill_switch:daily_loss_limit"
    assert tracker.blocks_action("place_protection") == "kill_switch:daily_loss_limit"
    assert tracker.blocks_action("market_exit") is None
    assert tracker.blocks_action("cancel") is None


def test_max_drawdown_halt(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=now)
    tracker.begin_batch()
    tracker.update_from_equity(7_900.0, now=now)  # -21% from peak

    assert tracker.is_halted()
    assert "max_dd" in tracker.halt_reasons()


def test_persistence_reloads_halt(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=now)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=now)
    assert tracker.is_halted()

    reloaded = _tracker(tmp_path)
    reloaded.load()
    assert reloaded.is_halted()
    assert reloaded.peak_equity == pytest.approx(10_000.0)


def test_utc_day_reset_clears_daily_loss_metric(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    day1 = datetime(2026, 6, 16, 23, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=day1)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=day1)
    assert tracker.is_halted()

    day2 = datetime(2026, 6, 17, 0, 30, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=day2)
    # Still halted until cooldown/daily gate clears in evaluate_safety_state;
    # daily_loss fraction on new day should be 0 at day open anchor.
    metrics = tracker.safety.last_metrics
    assert float(metrics.get("daily_loss") or 0.0) == pytest.approx(0.0)
