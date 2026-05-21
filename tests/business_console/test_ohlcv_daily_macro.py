"""1d OHLCV from macro spot_klines (not limited to feature-bus 180d window)."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.services.macro_spot_daily import MacroSpotDailyLoader
from app.services.ohlcv_reader import fetch_ohlcv


def _write_monthly_zip(dest: Path, rows: list[list]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    csv_body = "\n".join(",".join(str(v) for v in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("klines.csv", csv_body)
    dest.write_bytes(buf.getvalue())


@pytest.fixture
def macro_kline_root(tmp_path: Path) -> Path:
    sym = "ETHUSDT"
    zp = tmp_path / sym / "monthly" / "1d" / f"{sym}-1d-2024-01.zip"
    base_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rows = []
    for day in range(31):
        t = base_ms + day * 86_400_000
        price = 100.0 + day
        rows.append(
            [
                t,
                price,
                price + 1,
                price - 1,
                price + 0.5,
                1000.0,
                t + 86_399_999,
                0,
                0,
                0,
                0,
                0,
            ]
        )
    _write_monthly_zip(zp, rows)
    return tmp_path


def test_macro_loader_reads_monthly(macro_kline_root: Path) -> None:
    loader = MacroSpotDailyLoader(macro_kline_root)
    daily = loader.load_symbol_daily(
        "ETHUSDT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert len(daily) == 31


def test_fetch_ohlcv_1d_uses_macro_not_180d_cap(
    bus_root: Path, macro_kline_root: Path
) -> None:
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "1d",
        macro_kline_root=macro_kline_root,
        daily_ohlcv_start=date(2024, 1, 1),
        max_daily_ohlcv_days=3650,
        full_range=True,
    )
    assert data["source"] == "macro_spot_klines"
    assert data["row_count"] == 31
    assert len(data["candles"]) == 31


def test_fetch_ohlcv_2h_still_bars_1min(bus_root: Path, macro_kline_root: Path) -> None:
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        macro_kline_root=macro_kline_root,
        full_range=True,
        max_days=7,
    )
    assert data["source"] == "bars_1min"
