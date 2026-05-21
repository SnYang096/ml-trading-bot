from __future__ import annotations

import pandas as pd

from src.live_data_stream.auto_gap_fill import (
    BarGap,
    detect_large_bar_gaps,
    fill_large_bar_gaps,
)
from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.gap_filler import GapFiller


class FakeGapFiller:
    def __init__(self) -> None:
        self.data_gap_filler = None

    def fill_from_binance_api(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        freq = "1min" if timeframe == "1m" else timeframe
        timestamps = pd.date_range(start_time, end_time, freq=freq)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
            }
        )

    def fill_missing_ticks(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        return pd.DataFrame()


class FakeAggTradeBackend:
    def _aggregate_trades_to_1min(self, trades: pd.DataFrame) -> pd.DataFrame:
        out = trades.copy()
        out["timestamp"] = out["timestamp"].dt.floor("1min")
        return pd.DataFrame(
            {
                "timestamp": sorted(out["timestamp"].unique()),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "buy_volume": 1.0,
                "sell_volume": 0.0,
                "trade_count": 1,
                "delta": 1.0,
            }
        )


class FakeAggTradeGapFiller:
    def __init__(self) -> None:
        self.data_gap_filler = FakeAggTradeBackend()

    def fill_missing_ticks(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        timestamps = pd.date_range(start_time, end_time, freq="1min")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": 100.0,
                "volume": 1.0,
                "side": 1,
            }
        )

    def fill_from_binance_api(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("kline fallback should not be used when aggTrades works")


def _bars(times: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
        }
    )


def test_detect_large_bar_gaps_finds_internal_and_tail(tmp_path):
    storage = StorageManager(tmp_path)
    storage.bar_1min.append(
        "ETHUSDT",
        "2026-05-21",
        _bars(
            [
                "2026-05-21T00:00:00Z",
                "2026-05-21T00:01:00Z",
                "2026-05-21T02:10:00Z",
                "2026-05-21T02:11:00Z",
            ]
        ),
    )

    gaps = detect_large_bar_gaps(
        storage,
        ["ETHUSDT"],
        lookback_hours=12,
        min_gap_minutes=60,
        ignore_recent_minutes=0,
        now=pd.Timestamp("2026-05-21T04:00:00Z"),
    )

    assert [g.kind for g in gaps] == ["internal", "tail"]
    assert gaps[0].start == pd.Timestamp("2026-05-21T00:02:00Z")
    assert gaps[0].end == pd.Timestamp("2026-05-21T02:09:00Z")
    assert gaps[1].start == pd.Timestamp("2026-05-21T02:12:00Z")
    assert gaps[1].end == pd.Timestamp("2026-05-21T04:00:00Z")


def test_fill_large_bar_gaps_saves_kline_bars(tmp_path):
    storage = StorageManager(tmp_path)
    gap = BarGap(
        symbol="ETHUSDT",
        start=pd.Timestamp("2026-05-21T01:00:00Z"),
        end=pd.Timestamp("2026-05-21T01:02:00Z"),
        minutes=3.0,
    )

    written = fill_large_bar_gaps(
        storage,
        FakeGapFiller(),  # type: ignore[arg-type]
        [gap],
        now=pd.Timestamp("2026-05-21T04:00:00Z"),
    )

    assert written == 3
    saved = storage.bar_1min.load("ETHUSDT", "2026-05-21")
    assert list(saved["timestamp"]) == list(
        pd.date_range(
            "2026-05-21T01:00:00Z",
            "2026-05-21T01:02:00Z",
            freq="1min",
        )
    )


def test_fill_large_bar_gaps_prefers_aggtrades_and_saves_ticks(tmp_path):
    storage = StorageManager(tmp_path)
    gap = BarGap(
        symbol="ETHUSDT",
        start=pd.Timestamp("2026-05-21T01:00:00Z"),
        end=pd.Timestamp("2026-05-21T01:02:00Z"),
        minutes=3.0,
    )

    written = fill_large_bar_gaps(
        storage,
        FakeAggTradeGapFiller(),  # type: ignore[arg-type]
        [gap],
        now=pd.Timestamp("2026-05-21T04:00:00Z"),
    )

    assert written == 3
    bars = storage.bar_1min.load("ETHUSDT", "2026-05-21")
    ticks = storage.ticks.load("ETHUSDT", "2026-05-21")
    assert len(bars) == 3
    assert len(ticks) == 3
    assert {"buy_volume", "sell_volume", "delta"}.issubset(bars.columns)


def test_gap_filler_uses_minute_frequency_for_ccxt_1m(tmp_path):
    class FakeDownloader:
        def __init__(self) -> None:
            self.expected_count = 0

        def download_missing_bars(self, symbol, missing_timestamps, timeframe):
            self.expected_count = len(missing_timestamps)
            return pd.DataFrame(
                {
                    "timestamp": missing_timestamps,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                }
            )

    filler = GapFiller(StorageManager(tmp_path))
    fake = FakeDownloader()
    filler.data_gap_filler = fake

    out = filler.fill_from_binance_api(
        "ETHUSDT",
        pd.Timestamp("2026-05-21T01:00:00Z"),
        pd.Timestamp("2026-05-21T01:02:00Z"),
        timeframe="1m",
    )

    assert fake.expected_count == 3
    assert len(out) == 3
