"""Feature Bus provider for classic live trading.

This adapter lets ``scripts/run_live.py`` keep its existing strategy/execution
stack while replacing the market WebSocket feature source with rows produced by
``scripts/run_market_feature_publisher.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassicFeatureBusEvent:
    """One decision event reconstructed from feature-bus snapshots."""

    symbol: str
    timestamp: pd.Timestamp
    features: Dict[str, Any]
    features_by_timeframe: Dict[str, Dict[str, Any]]
    bars: List[Dict[str, Any]]


class ClassicFeatureBusProvider:
    """Poll Feature Bus snapshots and build classic live decision events."""

    def __init__(
        self,
        *,
        feature_bus_root: str | Path,
        symbols: Iterable[str],
        primary_timeframe: str,
        timeframes: Iterable[str],
        max_staleness_seconds: float = 1800.0,
        bars_lookback: int = 240,
        initial_bars_lookback: int = 1,
    ) -> None:
        self.reader = FeatureBusReader(feature_bus_root)
        self.symbols = [str(s).upper() for s in symbols]
        self.primary_timeframe = str(primary_timeframe)
        self.timeframes = self._ordered_timeframes(
            [self.primary_timeframe, *[str(tf) for tf in timeframes]]
        )
        self.max_staleness_seconds = float(max_staleness_seconds)
        self.bars_lookback = int(bars_lookback)
        self.initial_bars_lookback = max(0, int(initial_bars_lookback))
        self._last_seen_features: Dict[str, pd.Timestamp] = {}
        self._last_seen_bars: Dict[str, pd.Timestamp] = {}

    @staticmethod
    def _ordered_timeframes(raw: Iterable[str]) -> List[str]:
        out: List[str] = []
        for tf in raw:
            key = str(tf or "").strip()
            if key and key not in out:
                out.append(key)
        return out

    def poll(self) -> List[ClassicFeatureBusEvent]:
        """Return new primary-timeframe decision events, at most one per symbol."""
        events: List[ClassicFeatureBusEvent] = []
        for symbol in self.symbols:
            row = self.reader.latest_features(
                symbol=symbol,
                timeframe=self.primary_timeframe,
                after=self._last_seen_features.get(symbol),
            )
            if row is None:
                continue

            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            if self._is_stale(ts):
                logger.warning(
                    "[%s] feature-bus row stale: tf=%s ts=%s max_staleness=%.0fs",
                    symbol,
                    self.primary_timeframe,
                    ts,
                    self.max_staleness_seconds,
                )
                self._last_seen_features[symbol] = ts
                continue

            by_tf = self._load_timeframe_bundle(symbol)
            if self.primary_timeframe not in by_tf:
                logger.warning(
                    "[%s] feature-bus missing primary timeframe %s",
                    symbol,
                    self.primary_timeframe,
                )
                self._last_seen_features[symbol] = ts
                continue

            bars = self._load_new_bars(symbol)
            events.append(
                ClassicFeatureBusEvent(
                    symbol=symbol,
                    timestamp=ts,
                    features=dict(by_tf[self.primary_timeframe]),
                    features_by_timeframe=by_tf,
                    bars=bars,
                )
            )
            self._last_seen_features[symbol] = ts
        return events

    def poll_bars(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return new 1m/fast execution bars without requiring a new feature row."""
        return {
            symbol: bars
            for symbol in self.symbols
            if (bars := self._load_new_bars(symbol))
        }

    def latest_feature_bundle(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Return the latest non-stale feature rows for all configured timeframes."""
        return self._load_timeframe_bundle(str(symbol).upper())

    def _load_timeframe_bundle(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        bundle: Dict[str, Dict[str, Any]] = {}
        for tf in self.timeframes:
            row = self.reader.latest_features(symbol=symbol, timeframe=tf)
            if row is None:
                logger.debug("[%s] feature-bus timeframe unavailable: %s", symbol, tf)
                continue
            feat = self._series_to_dict(row)
            if self._is_stale(pd.Timestamp(feat["timestamp"])):
                logger.warning(
                    "[%s] feature-bus stale timeframe skipped: %s", symbol, tf
                )
                continue
            bundle[tf] = feat
        return bundle

    def _load_new_bars(self, symbol: str) -> List[Dict[str, Any]]:
        bars = self.reader.latest_bars_1m(
            symbol=symbol,
            after=self._last_seen_bars.get(symbol),
        )
        if bars.empty:
            return []
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            latest_ts = bars["timestamp"].max()
            limit = (
                self.initial_bars_lookback
                if symbol not in self._last_seen_bars
                else self.bars_lookback
            )
            self._last_seen_bars[symbol] = latest_ts
        else:
            limit = self.bars_lookback
        return bars.tail(limit).to_dict("records")

    def _is_stale(self, timestamp: pd.Timestamp) -> bool:
        if self.max_staleness_seconds <= 0:
            return False
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
        return age > self.max_staleness_seconds

    @staticmethod
    def _series_to_dict(row: pd.Series) -> Dict[str, Any]:
        out = row.to_dict()
        if "timestamp" in out:
            out["timestamp"] = pd.Timestamp(out["timestamp"])
            if out["timestamp"].tzinfo is None:
                out["timestamp"] = out["timestamp"].tz_localize("UTC")
            else:
                out["timestamp"] = out["timestamp"].tz_convert("UTC")
        return out
