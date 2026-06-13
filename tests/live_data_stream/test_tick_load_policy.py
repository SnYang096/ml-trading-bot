from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_market_feature_publisher as publisher
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
