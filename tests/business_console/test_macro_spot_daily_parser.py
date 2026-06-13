"""Vision spot 1d ZIP open_time unit (ms vs us) for CMS macro loader."""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from mlbot_console.services.macro_spot_daily import parse_kline_zip_bytes


def _zip_with_open_times(values: list[int]) -> bytes:
    buf = io.BytesIO()
    rows = []
    for i, ot in enumerate(values):
        rows.append(f"{ot},100,101,99,100.5,1,0,0,0,0,0,0")
    csv = "\n".join(rows)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("klines.csv", csv)
    return buf.getvalue()


def test_parse_kline_zip_bytes_us_timestamps_2026() -> None:
    us_open = int(pd.Timestamp("2026-03-31", tz="UTC").value // 1000)
    raw = _zip_with_open_times([us_open])
    df = parse_kline_zip_bytes(raw)
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2026-03-31", tz="UTC")


def test_parse_kline_zip_bytes_ms_timestamps_legacy() -> None:
    ms_open = 1_704_067_200_000  # 2024-01-01 UTC
    raw = _zip_with_open_times([ms_open])
    df = parse_kline_zip_bytes(raw)
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
