"""Unit tests for multileg engine audit → funnel → stats digest."""

from __future__ import annotations

import logging
import time

import pytest

from src.time_series_model.live.multileg_funnel import (
    chop_grid_bar_outcome,
    funnel_for_chop_grid_bar,
    funnel_for_multileg_bar,
    funnel_for_trend_scalp_bar,
    trend_scalp_bar_outcome,
)
from src.time_series_model.live.metrics_exporter import METRICS
from src.time_series_model.live.stats_collector import StatsCollector


@pytest.fixture(autouse=True)
def _stub_metrics_disk_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(METRICS, "update_system_health", lambda: None, raising=True)
    monkeypatch.setattr(METRICS, "update_disk_health", lambda: None, raising=True)


def test_chop_outcome_flat_blocked_chop_low() -> None:
    assert (
        chop_grid_bar_outcome(
            active_at_open=False,
            wanted_enter=False,
            is_box=False,
            chop=0.2,
            entry_chop_min=0.5,
            actions=[],
        )
        == "flat_blocked_chop_low"
    )


def test_chop_outcome_open_grid_placed() -> None:
    assert (
        chop_grid_bar_outcome(
            active_at_open=False,
            wanted_enter=True,
            is_box=False,
            chop=0.8,
            entry_chop_min=0.5,
            actions=[{"action": "place"}],
        )
        == "open_grid_placed"
    )


def test_chop_funnel_maps_wanted_enter_not_direction() -> None:
    audit = {
        "engine": "chop_grid",
        "is_box": False,
        "wanted_enter": True,
        "active_at_open": False,
        "outcome": "flat_other",
    }
    f = funnel_for_chop_grid_bar(
        audit=audit,
        actions=[],
        approved_actions=[],
        rejected=[],
    )
    assert f["multileg"] is True
    assert f["wanted_enter"] is True
    assert "direction" not in f


def test_trend_funnel_has_direction_and_regime() -> None:
    audit = {
        "engine": "trend_scalp",
        "trend_conf": 0.9,
        "entry_trend_min": 0.6,
        "is_box": False,
        "exclude_box_prefilter": True,
        "wanted_enter": True,
        "active_at_open": False,
        "trend_side": "LONG",
        "outcome": "flat_blocked_trend_low",
    }
    f = funnel_for_trend_scalp_bar(
        audit=audit,
        actions=[],
        approved_actions=[],
        rejected=[],
    )
    assert f["direction"] is True
    assert f["direction_value"] == 1
    assert f["regime"] is True


def test_trend_outcome_exit_close_and_chop_high_block() -> None:
    assert (
        trend_scalp_bar_outcome(
            active_at_open=True,
            wanted_enter=False,
            trend_conf=0.8,
            chop=0.5,
            entry_trend_min=0.6,
            max_entry_chop=0.4,
            exclude_box=True,
            is_box=False,
            actions=[{"action": "market_exit"}],
        )
        == "exit_close"
    )
    assert (
        trend_scalp_bar_outcome(
            active_at_open=False,
            wanted_enter=False,
            trend_conf=0.9,
            chop=0.9,
            entry_trend_min=0.6,
            max_entry_chop=0.4,
            exclude_box=True,
            is_box=False,
            actions=[],
        )
        == "flat_blocked_chop_high"
    )


def test_chop_engine_on_bar_sets_last_bar_audit(tmp_path) -> None:
    from pathlib import Path

    from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine

    cfg = tmp_path / "grid.yaml"
    cfg.write_text(
        """
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 1
risk:
  fee_bps: 4.0
  max_loss_per_grid: 0.03
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=cfg,
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.2, "box_prefilter": False},
    )
    audit = engine._last_bar_audit
    assert audit["engine"] == "chop_grid"
    assert audit["wanted_enter"] is False
    assert audit["outcome"] == "flat_blocked_chop_low"


def test_multileg_dispatch_by_audit_engine() -> None:
    f = funnel_for_multileg_bar(
        strategy="chop_grid",
        engine_audit={
            "engine": "chop_grid",
            "is_box": True,
            "wanted_enter": False,
            "active_at_open": False,
            "outcome": "flat_blocked_box",
        },
        actions=[],
        approved_actions=[],
        rejected=[],
    )
    assert f["prefilter"] is False
    assert f["outcome"] == "flat_blocked_box"


def test_stats_collector_multileg_digest_log(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    sc = StatsCollector(
        db_path=str(tmp_path / "live_monitor.db"),
        auto_cleanup=False,
    )
    sc.digest_interval_s = 1.0
    sc._digest_last_mono = 0.0
    audit = {
        "engine": "chop_grid",
        "is_box": False,
        "wanted_enter": True,
        "active_at_open": False,
        "outcome": "flat_blocked_chop_low",
    }
    sc.record_strategy_eval(
        "BTCUSDT",
        "chop_grid",
        funnel_for_multileg_bar(
            strategy="chop_grid",
            engine_audit=audit,
            actions=[],
            approved_actions=[],
            rejected=[],
        ),
    )
    sc.flush(symbol="ALL")
    sc._digest_last_mono = time.monotonic() - 5.0
    sc.record_strategy_eval(
        "BTCUSDT",
        "chop_grid",
        funnel_for_multileg_bar(
            strategy="chop_grid",
            engine_audit=audit,
            actions=[],
            approved_actions=[],
            rejected=[],
        ),
    )
    sc.flush(symbol="ALL")
    digest_msgs = [
        r.message for r in caplog.records if "multileg funnel digest" in r.message
    ]
    assert len(digest_msgs) == 1
    msg = digest_msgs[0]
    assert "wanted_enter" in msg
    assert "no_dir≈" not in msg
    assert "exit_grid" in msg
