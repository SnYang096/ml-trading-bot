"""Tests for pipeline freshness metrics collection."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.live_data_stream.pipeline_freshness import collect_pipeline_ages


def test_collect_pipeline_ages_ticks_bars_and_bus(tmp_path: Path) -> None:
    sym = "ETHUSDT"
    storage = tmp_path / "storage"
    bus = tmp_path / "bus"
    (storage / "ticks" / sym).mkdir(parents=True)
    (storage / "bars" / sym).mkdir(parents=True)
    tick_path = storage / "ticks" / sym / "2026-05-20.parquet"
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-20", tz="UTC")]}).to_parquet(
        tick_path
    )
    bar_path = storage / "bars" / sym / "2026-05-20.parquet"
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-20", tz="UTC")]}).to_parquet(
        bar_path
    )
    (bus / "bars_1min").mkdir(parents=True)
    pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-05-20 10:00:00", tz="UTC")], "close": [1.0]}
    ).to_parquet(bus / "bars_1min" / f"{sym}.parquet")
    (bus / "latest" / "features" / "120T").mkdir(parents=True)
    meta = bus / "latest" / "features" / "120T" / f"{sym}.json"
    meta.write_text(
        '{"timestamp": "2026-05-20T10:00:00+00:00"}',
        encoding="utf-8",
    )

    ages = collect_pipeline_ages(
        [sym],
        storage_base=storage,
        bus_root=bus,
        feature_timeframes=["120T"],
    )
    assert ("ticks_1m", sym) in ages
    assert ("bars_1m", sym) in ages
    assert ("bus_bars_1min", sym) in ages
    assert ("features_120T", sym) in ages
    assert ages[("ticks_1m", sym)] < 60
