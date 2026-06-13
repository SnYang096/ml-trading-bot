"""Tests for atomic parquet I/O and corrupt file quarantine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.live_data_stream.feature_storage import (
    TickStorage,
    sanitize_dated_parquet_for_symbols,
)
from src.live_data_stream.parquet_io import (
    MIN_PARQUET_BYTES,
    atomic_write_parquet,
    is_unreadable_parquet,
    read_parquet_safe,
)
from src.live_data_stream.auto_gap_fill import (
    detect_large_tick_gaps,
    run_auto_gap_fill_once,
)
from src.live_data_stream.feature_storage import StorageManager


def test_atomic_write_parquet_readable(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    df = pd.DataFrame({"timestamp": [1], "value": [2.0]})
    atomic_write_parquet(df, path)
    assert path.stat().st_size >= MIN_PARQUET_BYTES
    back = pd.read_parquet(path)
    assert len(back) == 1


def test_read_parquet_safe_quarantines_4byte_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"xxxx")
    assert is_unreadable_parquet(path)
    empty = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
    out = read_parquet_safe(path, empty=empty)
    assert out.empty
    assert not path.exists()


def test_tick_storage_load_range_skips_bad_day(tmp_path: Path) -> None:
    storage = TickStorage(tmp_path / "ticks")
    sym = "ETHUSDT"
    good_day = "2026-05-21"
    bad_day = "2026-05-22"
    ok = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-21 10:00:00"], utc=True),
            "price": [100.0],
            "volume": [1.0],
            "side": [1],
        }
    )
    storage.append(sym, good_day, ok)
    (tmp_path / "ticks" / sym / f"{bad_day}.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "ticks" / sym / f"{bad_day}.parquet").write_bytes(b"bad")

    combined = storage.load_range(sym, good_day, bad_day)
    assert len(combined) == 1
    assert not (tmp_path / "ticks" / sym / f"{bad_day}.parquet").exists()


def test_detect_tick_gaps_isolated_per_symbol(tmp_path: Path) -> None:
    mgr = StorageManager(base_path=tmp_path / "live")
    sym_ok = "BTCUSDT"
    sym_bad = "ETHUSDT"
    day = "2026-05-20"
    ts = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-05-20 10:00", periods=3, freq="1h", tz="UTC"
            ),
            "price": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
            "side": [1, -1, 1],
        }
    )
    mgr.ticks.append(sym_ok, day, ts)
    bad_path = mgr.ticks.root / sym_bad / f"{day}.parquet"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"x")

    gaps = detect_large_tick_gaps(
        mgr,
        [sym_ok, sym_bad],
        lookback_hours=48.0,
        min_gap_minutes=60.0,
        now=pd.Timestamp("2026-05-20 14:00:00", tz="UTC"),
    )
    assert isinstance(gaps, list)


def test_run_auto_gap_fill_once_swallows_top_level_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = StorageManager(base_path=tmp_path / "live")

    class _BrokenGapFiller:
        _pending_vision_gaps = []

        def retry_pending_gaps(self) -> bool:
            raise RuntimeError("boom")

    assert (
        run_auto_gap_fill_once(
            mgr,
            _BrokenGapFiller(),  # type: ignore[arg-type]
            ["BTCUSDT"],
        ).written_bars
        == 0
    )


def test_sanitize_dated_parquet_for_symbols(tmp_path: Path) -> None:
    mgr = StorageManager(base_path=tmp_path / "live")
    sym = "ETHUSDT"
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    bad = mgr.ticks.root / sym / f"{day}.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"zz")
    n = sanitize_dated_parquet_for_symbols(mgr, [sym], lookback_days=3)
    assert n == 1
    assert not bad.exists()
