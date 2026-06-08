"""Tests for Binance Vision spot weekly EMA200 macro seed."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_weekly_ema_position_from_ohlc,
)
from src.live_data_stream.spot_weekly_ema_seed import (
    _parse_kline_zip_bytes,
    compute_weekly_ema_table,
    prepare_spot_weekly_ema_seed,
    seed_ema_plausible_vs_close,
    weekly_ema_position_from_seed,
)


def _make_kline_zip(rows: list[list]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv = "\n".join(",".join(str(x) for x in r) for r in rows)
        zf.writestr("TEST-1d.csv", csv)
    return buf.getvalue()


def test_parse_kline_zip_bytes() -> None:
    # open_time ms, OHLCV + padding columns
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    raw = _make_kline_zip(
        [
            [t0, 100, 110, 90, 105, 1000, 0, 0, 0, 0, 0, 0],
            [t0 + 86_400_000, 105, 115, 95, 110, 1100, 0, 0, 0, 0, 0, 0],
        ]
    )
    df = _parse_kline_zip_bytes(raw)
    assert len(df) == 2
    assert float(df["close"].iloc[-1]) == 110.0


def test_parse_kline_zip_bytes_microsecond_open_time() -> None:
    """Vision spot 1d switched to 16-digit microsecond open_time from 2025."""
    t0 = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1_000_000)
    raw = _make_kline_zip(
        [
            [t0, 100, 110, 90, 105, 1000, 0, 0, 0, 0, 0, 0],
            [t0 + 86_400_000_000, 105, 115, 95, 110, 1100, 0, 0, 0, 0, 0, 0],
        ]
    )
    df = _parse_kline_zip_bytes(raw)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert float(df["close"].iloc[-1]) == 110.0


def test_insufficient_live_history_returns_nan_not_zero() -> None:
    # ~180 calendar days of 2h bars — far below 40 weekly bars for span=200.
    idx = pd.date_range("2025-01-01", periods=180 * 12, freq="2h", tz="UTC")
    close = pd.Series(3000.0, index=idx)
    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=close * 1.001,
        low=close * 0.999,
        ema_span_weeks=200,
    )["weekly_ema_200_position"]
    last = out.iloc[-1]
    assert pd.isna(last), f"expected NaN for insufficient history, got {last}"


def test_context_seed_produces_negative_position(tmp_path: Path) -> None:
    idx = pd.date_range("2026-03-01", periods=10, freq="2h", tz="UTC")
    close = pd.Series(2100.0, index=idx)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    weekly = pd.DataFrame(
        {
            "week_ts": pd.date_range("2024-01-07", periods=60, freq="W-SUN", tz="UTC"),
            "weekly_close": 3000.0,
            "weekly_ema_200": 2540.0,
        }
    )
    weekly.to_parquet(seed_dir / "ETHUSDT.parquet", index=False)

    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=close * 1.01,
        low=close * 0.99,
        weekly_ema_context_dir=str(seed_dir),
        symbol="ETHUSDT",
    )["weekly_ema_200_position"]

    assert float(out.iloc[-1]) < 0.0


def test_seed_ema_plausible_rejects_bnb_like_stale_flat_line() -> None:
    assert not seed_ema_plausible_vs_close(412.0, 658.0)
    assert seed_ema_plausible_vs_close(570.0, 658.0)


def test_weekly_ema_position_from_seed_rejects_implausible_ema(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    weekly = pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2026-05-18", tz="UTC")],
            "weekly_close": [400.0],
            "weekly_ema_200": [412.0],
        }
    )
    weekly.to_parquet(seed_dir / "BNBUSDT.parquet", index=False)
    pos = weekly_ema_position_from_seed(
        close=658.0,
        bar_ts=pd.Timestamp("2026-05-23", tz="UTC"),
        seed_root=seed_dir,
        symbol="BNBUSDT",
    )
    assert pos is None


def test_weekly_ema_position_from_seed_single_bar(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    weekly = pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2026-03-02", tz="UTC")],
            "weekly_close": [3000.0],
            "weekly_ema_200": [2540.0],
        }
    )
    weekly.to_parquet(seed_dir / "BTCUSDT.parquet", index=False)
    pos = weekly_ema_position_from_seed(
        close=2100.0,
        bar_ts=pd.Timestamp("2026-03-10", tz="UTC"),
        seed_root=seed_dir,
        symbol="BTCUSDT",
    )
    assert pos is not None and pos < 0.0


def test_compute_weekly_ema_table_has_valid_tail() -> None:
    idx = pd.date_range("2018-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    weekly = compute_weekly_ema_table(close, ema_span_weeks=20)
    assert "weekly_ema_200" in weekly.columns
    assert int(weekly["weekly_ema_200"].notna().sum()) >= 1


def test_prepare_spot_weekly_ema_seed_uses_cached_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sym = "TESTUSDT"
    kroot = tmp_path / "klines"
    sroot = tmp_path / "seed"
    month_dir = kroot / sym / "monthly" / "1d"
    month_dir.mkdir(parents=True)
    t0 = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    zbytes = _make_kline_zip(
        [
            [t0 + i * 86_400_000, 100, 110, 90, 100 + i, 1, 0, 0, 0, 0, 0, 0]
            for i in range(5)
        ]
    )
    (month_dir / f"{sym}-1d-2024-06.zip").write_bytes(zbytes)

    def _no_download(self, url, dest):  # noqa: ANN001
        return dest.exists()

    monkeypatch.setattr(
        "src.live_data_stream.spot_weekly_ema_seed.SpotDailyKlineDownloader._download_url",
        _no_download,
    )
    written = prepare_spot_weekly_ema_seed(
        [sym],
        kline_root=kroot,
        seed_root=sroot,
        start_date=date(2024, 6, 1),
        end_date=date(2024, 6, 10),
        ema_span_weeks=4,
        refresh_recent_days=0,
    )
    assert sym in written
    assert written[sym].exists()


def test_incremental_feature_computer_seed_override(tmp_path: Path) -> None:
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )

    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    weekly = pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2026-03-02", tz="UTC")],
            "weekly_close": [3000.0],
            "weekly_ema_200": [2540.0],
        }
    )
    weekly.to_parquet(seed_dir / "ETHUSDT.parquet", index=False)

    fc = IncrementalFeatureComputer(weekly_ema_seed_root=str(seed_dir))
    fc.live_feature_set = {"weekly_ema_200_position"}
    fc._current_symbol = "ETHUSDT"
    idx = pd.date_range("2026-03-10", periods=3, freq="2h", tz="UTC")
    bars = pd.DataFrame({"close": [2100.0, 2100.0, 2100.0]}, index=idx)
    out = fc._apply_weekly_ema_seed_override(bars, {"weekly_ema_200_position": 0.0})
    assert float(out["weekly_ema_200_position"]) < 0.0
