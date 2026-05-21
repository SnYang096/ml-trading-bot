"""Macro kline root resolution and glob monthly load."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mlbot_console.services.macro_spot_daily import MacroSpotDailyLoader
from mlbot_console.services.ohlcv_reader import resolve_macro_kline_root


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


def test_resolve_macro_kline_root_falls_back_to_live_data(tmp_path: Path) -> None:
    macro = tmp_path / "live_data" / "macro" / "spot_klines"
    macro.mkdir(parents=True)
    root, ok = resolve_macro_kline_root(
        tmp_path / "missing",
        live_data_root=tmp_path / "live_data",
    )
    assert ok is True
    assert root == macro


def test_macro_loader_globs_all_monthly_zips(macro_kline_root: Path) -> None:
    loader = MacroSpotDailyLoader(macro_kline_root)
    daily = loader.load_symbol_daily(
        "ETHUSDT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert len(daily) == 31
