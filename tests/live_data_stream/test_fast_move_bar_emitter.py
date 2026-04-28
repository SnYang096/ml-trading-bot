from __future__ import annotations

import pandas as pd

from scripts.run_market_feature_publisher import FastMoveBarEmitter
from src.live_data_stream.feature_bus import FeatureBusReader, FeatureBusWriter
from src.live_data_stream.websocket_client import BinanceTick


def _tick(ts: str, price: float) -> BinanceTick:
    return BinanceTick(
        symbol="BTCUSDT",
        timestamp_ms=int(pd.Timestamp(ts).timestamp() * 1000),
        price=price,
        volume=1.0,
        turnover=price,
        side=1,
        trade_id=1,
    )


def test_fast_move_bar_emitter_writes_supplemental_execution_bar(tmp_path):
    writer = FeatureBusWriter(tmp_path)
    emitter = FastMoveBarEmitter(writer, threshold_pct=0.03, bucket_seconds=10)

    emitter.on_tick(_tick("2026-01-01T00:00:01Z", 100.0))
    emitter.on_tick(_tick("2026-01-01T00:00:05Z", 103.2))
    emitter.on_tick(_tick("2026-01-01T00:00:07Z", 104.0))

    bars = FeatureBusReader(tmp_path).latest_bars_1m(symbol="BTCUSDT")

    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["_bar_kind"] == "fast_intraminute"
    assert row["_source_timeframe_seconds"] == 10
    assert float(row["_trigger_move_pct"]) >= 0.03
    assert pd.Timestamp(row["timestamp"]) == pd.Timestamp("2026-01-01T00:00:05Z")
