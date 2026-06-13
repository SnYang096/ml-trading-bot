from __future__ import annotations

import argparse
import asyncio
import os

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from scripts import run_market_feature_publisher as publisher
from src.live_data_stream.auto_gap_fill import GapFillRunResult
from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.tick_load_policy import load_ticks_for_feature_compute


def _ticks(start: str, n_minutes: int, rows_per_minute: int = 2) -> pd.DataFrame:
    base = pd.Timestamp(start, tz="UTC")
    rows = []
    for i in range(n_minutes):
        ts = base + pd.Timedelta(minutes=i)
        for j in range(rows_per_minute):
            rows.append(
                {
                    "timestamp": ts + pd.Timedelta(milliseconds=j),
                    "price": 100.0 + j,
                    "volume": 1.0,
                    "side": "buy" if j == 0 else "sell",
                }
            )
    return pd.DataFrame(rows)


def test_load_ticks_recent_window_sufficient(tmp_path, monkeypatch: pytest.MonkeyPatch):
    storage = StorageManager(tmp_path)
    monkeypatch.setenv("MLBOT_MIN_TICKS_REQUIRED", "100")
    monkeypatch.setenv("MLBOT_TICK_LOOKBACK_DAYS", "2")
    for day in ("2026-06-10", "2026-06-11", "2026-06-12"):
        storage.ticks.append("BTCUSDT", day, _ticks(f"{day}T00:00:00Z", 1440))

    ticks, recent = load_ticks_for_feature_compute(
        storage,
        "BTCUSDT",
        now=pd.Timestamp("2026-06-12T12:00:00Z"),
        bar_end="2026-06-12",
    )
    assert recent >= 100
    assert len(ticks) >= 100


def test_load_ticks_extends_in_chunks_not_whole_span(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    storage = StorageManager(tmp_path)
    monkeypatch.setenv("MLBOT_MIN_TICKS_REQUIRED", "5000")
    monkeypatch.setenv("MLBOT_TICK_LOOKBACK_DAYS", "2")
    monkeypatch.setenv("MLBOT_TICK_EXTENDED_MAX_DAYS", "30")
    monkeypatch.setenv("MLBOT_TICK_LOAD_CHUNK_DAYS", "7")

    storage.ticks.append("HYPEUSDT", "2026-06-12", _ticks("2026-06-12T00:00:00Z", 60))
    for day in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"):
        storage.ticks.append("HYPEUSDT", day, _ticks(f"{day}T00:00:00Z", 1440, 2))

    ticks, recent = load_ticks_for_feature_compute(
        storage,
        "HYPEUSDT",
        now=pd.Timestamp("2026-06-12T12:00:00Z"),
        bar_end="2026-06-12",
    )
    assert recent < 5000
    assert len(ticks) >= 5000


def test_resolve_post_warmup_audit_symbols_gap_fill_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    bus_root = tmp_path / "bus"
    (bus_root / "latest" / "features" / "120T").mkdir(parents=True)
    (bus_root / "latest" / "features" / "120T" / "BTCUSDT.json").write_text("{}")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "gap-fill")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_ROOT", str(bus_root))
    monkeypatch.setenv("MLBOT_PIPELINE_FEATURE_TFS", "120T")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS", "ETHUSDT")

    scope = publisher._resolve_post_warmup_audit_symbols(
        ["BTCUSDT", "ETHUSDT", "HYPEUSDT"]
    )
    assert scope == {"ETHUSDT", "HYPEUSDT"}


def test_resolve_post_warmup_audit_symbols_all_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "all")
    scope = publisher._resolve_post_warmup_audit_symbols(["BTCUSDT", "ETHUSDT"])
    assert scope == {"BTCUSDT", "ETHUSDT"}


def test_resolve_post_warmup_audit_symbols_off_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "off")
    scope = publisher._resolve_post_warmup_audit_symbols(["BTCUSDT"])
    assert scope == set()


def test_resolve_post_warmup_audit_symbols_gap_fill_skips_when_all_present(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    bus_root = tmp_path / "bus"
    latest = bus_root / "latest" / "features" / "120T"
    latest.mkdir(parents=True)
    for sym in ("BTCUSDT", "ETHUSDT", "HYPEUSDT"):
        (latest / f"{sym}.json").write_text("{}")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "gap-fill")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_ROOT", str(bus_root))
    monkeypatch.setenv("MLBOT_PIPELINE_FEATURE_TFS", "120T")
    monkeypatch.delenv("MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS", raising=False)

    scope = publisher._resolve_post_warmup_audit_symbols(
        ["BTCUSDT", "ETHUSDT", "HYPEUSDT"]
    )
    assert scope == set()


@pytest.mark.asyncio
async def test_startup_gap_repair_sets_audit_env_then_scope_is_gap_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS", raising=False)
    bus_root = tmp_path / "bus"
    latest = bus_root / "latest" / "features" / "120T"
    latest.mkdir(parents=True)
    for sym in ("BTCUSDT", "ETHUSDT", "HYPEUSDT"):
        (latest / f"{sym}.json").write_text("{}")
    monkeypatch.setenv("MLBOT_FEATURE_BUS_ROOT", str(bus_root))
    monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "gap-fill")
    monkeypatch.setenv("MLBOT_PIPELINE_FEATURE_TFS", "120T")

    args = argparse.Namespace(
        auto_gap_fill_interval_minutes=15,
        auto_gap_fill_startup_lookback_hours=48,
        auto_gap_fill_lookback_hours=24,
        auto_gap_fill_min_gap_minutes=60,
        auto_gap_fill_max_gaps_per_run=24,
        auto_gap_fill_sparse_lookback_hours=72,
        auto_gap_fill_sparse_min_rows_per_day=1435,
        warmup_days=7,
    )
    manager = MagicMock()
    manager.gap_filler = MagicMock()
    manager.storage_manager = MagicMock()
    writer = MagicMock()
    repair = GapFillRunResult(
        written_bars=3,
        symbols_repaired=frozenset({"ETHUSDT", "HYPEUSDT"}),
    )

    with patch(
        "scripts.run_market_feature_publisher.run_auto_gap_fill_once",
        return_value=repair,
    ):
        result = await publisher._startup_gap_repair(
            args,
            manager,
            ["BTCUSDT", "ETHUSDT", "HYPEUSDT"],
            writer,
        )

    assert result.symbols_repaired == frozenset({"ETHUSDT", "HYPEUSDT"})
    assert os.environ["MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS"] == "ETHUSDT,HYPEUSDT"
    scope = publisher._resolve_post_warmup_audit_symbols(
        ["BTCUSDT", "ETHUSDT", "HYPEUSDT"]
    )
    assert scope == {"ETHUSDT", "HYPEUSDT"}


@pytest.mark.asyncio
async def test_startup_gap_repair_empty_repair_does_not_set_audit_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS", raising=False)
    args = argparse.Namespace(
        auto_gap_fill_interval_minutes=15,
        auto_gap_fill_startup_lookback_hours=48,
        auto_gap_fill_lookback_hours=24,
        auto_gap_fill_min_gap_minutes=60,
        auto_gap_fill_max_gaps_per_run=24,
        auto_gap_fill_sparse_lookback_hours=72,
        auto_gap_fill_sparse_min_rows_per_day=1435,
        warmup_days=7,
    )
    manager = MagicMock()
    manager.gap_filler = MagicMock()
    manager.storage_manager = MagicMock()
    empty = GapFillRunResult(written_bars=0, symbols_repaired=frozenset())

    with patch(
        "scripts.run_market_feature_publisher.run_auto_gap_fill_once",
        return_value=empty,
    ):
        await publisher._startup_gap_repair(args, manager, ["BTCUSDT"], MagicMock())

    assert "MLBOT_FEATURE_BUS_AUDIT_GAP_FILL_SYMBOLS" not in os.environ
