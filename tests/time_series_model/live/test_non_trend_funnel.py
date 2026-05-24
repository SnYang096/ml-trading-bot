"""Unit tests for non-trend (spot / multi_leg) funnel helpers."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.time_series_model.live.metrics_exporter import METRICS
from src.time_series_model.live.non_trend_funnel import (
    FifteenMinFlusher,
    default_live_monitor_db_path,
    funnel_for_multileg_bar,
    funnel_for_spot_decision,
)
from src.time_series_model.live.stats_collector import StatsCollector


@pytest.fixture(autouse=True)
def _stub_metrics_disk_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """update_system_health() shells out to ``du -sb`` over project paths
    (120s default timeout per volume). Stub it for unit tests."""
    monkeypatch.setattr(METRICS, "update_system_health", lambda: None, raising=True)
    monkeypatch.setattr(METRICS, "update_disk_health", lambda: None, raising=True)


# ── path resolution ────────────────────────────────────────────


def test_default_live_monitor_db_path_uses_env_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLBOT_LIVE_BASE", "live/foo")
    monkeypatch.delenv("MLBOT_STATS_DB_PATH", raising=False)
    assert default_live_monitor_db_path() == Path("live/foo/data/db/live_monitor.db")


def test_default_live_monitor_db_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBOT_STATS_DB_PATH", "/tmp/custom.db")
    assert default_live_monitor_db_path() == Path("/tmp/custom.db")


# ── spot funnel mapping ────────────────────────────────────────


def test_spot_funnel_no_intent_means_no_direction() -> None:
    f = funnel_for_spot_decision(has_intent=False, can_submit=False)
    assert f["direction"] is False
    assert f["direction_value"] == 0
    assert f["gate"] is False
    assert f["gate_reasons"] == []


def test_spot_funnel_intent_blocked_records_reason() -> None:
    f = funnel_for_spot_decision(
        has_intent=True, can_submit=False, blocker="schedule_window_closed"
    )
    assert f["direction"] is True
    assert f["direction_value"] == 1
    assert f["gate"] is False
    assert f["gate_reasons"] == ["schedule_window_closed"]
    assert f["entry_filter"] is False
    assert f["evidence"] is False


def test_spot_funnel_intent_passes_full_chain() -> None:
    f = funnel_for_spot_decision(has_intent=True, can_submit=True)
    assert f["regime"] is True
    assert f["prefilter"] is True
    assert f["direction"] is True
    assert f["gate"] is True
    assert f["gate_reasons"] == []
    assert f["entry_filter"] is True
    assert f["evidence"] is True


def test_spot_funnel_truncates_long_blocker() -> None:
    long = "x" * 200
    f = funnel_for_spot_decision(has_intent=True, can_submit=False, blocker=long)
    assert len(f["gate_reasons"]) == 1
    assert len(f["gate_reasons"][0]) == 60


# ── multi-leg funnel mapping ───────────────────────────────────


def test_multileg_funnel_no_actions() -> None:
    f = funnel_for_multileg_bar(actions=[], approved_actions=[], rejected=[])
    assert f["direction"] is False
    assert f["direction_value"] == 0
    assert f["gate"] is False
    assert f["gate_reasons"] == []


def test_multileg_funnel_all_approved() -> None:
    actions = [{"action": "place"}]
    f = funnel_for_multileg_bar(actions=actions, approved_actions=actions, rejected=[])
    assert f["direction"] is True
    assert f["gate"] is True
    assert f["gate_reasons"] == []


def test_multileg_funnel_rejected_collects_reason() -> None:
    rejected = [
        {"action": {"action": "place"}, "reason": "max_gross_notional exceeded"}
    ]
    f = funnel_for_multileg_bar(
        actions=[{"action": "place"}], approved_actions=[], rejected=rejected
    )
    assert f["direction"] is True
    assert f["gate"] is False
    assert f["gate_reasons"] == ["max_gross_notional exceeded"]


def test_multileg_funnel_rejected_object_with_reason_attr() -> None:
    class Rej:
        reason = "max_drawdown_pct breach"

    f = funnel_for_multileg_bar(
        actions=[{"action": "place"}], approved_actions=[], rejected=[Rej()]
    )
    assert f["gate_reasons"] == ["max_drawdown_pct breach"]


# ── flusher ────────────────────────────────────────────────────


class FakeCollector:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def flush(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return kwargs


def test_flusher_skips_until_interval_elapsed() -> None:
    fc = FakeCollector()
    flusher = FifteenMinFlusher(fc, interval_s=900.0)
    assert flusher.maybe_flush(symbol="BTCUSDT") is False
    assert fc.calls == []


def test_flusher_fires_after_interval() -> None:
    fc = FakeCollector()
    flusher = FifteenMinFlusher(fc, interval_s=1.0)
    flusher._last_flush_mono = time.monotonic() - 5.0
    assert flusher.maybe_flush(symbol="ALL") is True
    assert len(fc.calls) == 1
    assert fc.calls[0]["symbol"] == "ALL"


def test_flusher_force_flush_always_writes() -> None:
    fc = FakeCollector()
    flusher = FifteenMinFlusher(fc, interval_s=900.0)
    assert flusher.force_flush(symbol="ETHUSDT") is True
    assert fc.calls[0]["symbol"] == "ETHUSDT"


def test_flusher_resets_timer_after_flush() -> None:
    fc = FakeCollector()
    flusher = FifteenMinFlusher(fc, interval_s=1.0)
    flusher._last_flush_mono = time.monotonic() - 5.0
    assert flusher.maybe_flush() is True
    assert flusher.maybe_flush() is False  # too soon
    assert len(fc.calls) == 1


def test_flusher_handles_none_collector() -> None:
    flusher = FifteenMinFlusher(None, interval_s=1.0)
    flusher._last_flush_mono = time.monotonic() - 5.0
    assert flusher.maybe_flush() is False
    assert flusher.force_flush() is False


def test_flusher_passes_positions_and_health() -> None:
    fc = FakeCollector()
    flusher = FifteenMinFlusher(fc, interval_s=1.0, regime="HIGH_VOL")
    flusher._last_flush_mono = time.monotonic() - 5.0
    flusher.maybe_flush(
        positions={"BTCUSDT": {"qty": 1.0}},
        system_health={"tick_count": 5},
    )
    assert fc.calls[0]["positions"] == {"BTCUSDT": {"qty": 1.0}}
    assert fc.calls[0]["system_health"] == {"tick_count": 5}
    assert fc.calls[0]["regime"] == "HIGH_VOL"


# ── integration: stats_collector + funnel mappings ─────────────


def test_stats_collector_round_trip_multileg(tmp_path: Path) -> None:
    db = tmp_path / "live_monitor.db"
    sc = StatsCollector(db_path=str(db), auto_cleanup=False)
    sc.record_bar_processed(1)
    sc.record_strategy_eval(
        "BTCUSDT",
        "chop_grid",
        funnel_for_multileg_bar(
            actions=[{"action": "place"}],
            approved_actions=[{"action": "place"}],
            rejected=[],
        ),
    )
    sc.record_order_placed("BTCUSDT", "chop_grid")
    sc.flush(symbol="BTCUSDT")
    with sqlite3.connect(str(db)) as conn:
        rows = list(
            conn.execute(
                "SELECT bars_processed, orders_placed, by_strategy FROM stats_15min"
            )
        )
    assert len(rows) == 1
    bars, orders, _bys = rows[0]
    assert bars == 1
    assert orders == 1


def test_stats_collector_round_trip_spot(tmp_path: Path) -> None:
    db = tmp_path / "live_monitor.db"
    sc = StatsCollector(db_path=str(db), auto_cleanup=False)
    sc.record_bar_processed(1)
    sc.record_strategy_eval(
        "ETHUSDT",
        "spot_accum_simple",
        funnel_for_spot_decision(
            has_intent=True, can_submit=False, blocker="daily_cap_reached"
        ),
    )
    sc.flush(symbol="ETHUSDT")
    with sqlite3.connect(str(db)) as conn:
        bys = conn.execute("SELECT by_strategy FROM stats_15min").fetchone()[0]
    assert "spot_accum_simple" in bys
    assert "daily_cap_reached" in bys
