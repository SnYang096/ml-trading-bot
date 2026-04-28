"""Feature-store backed provider for multi-leg live strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusReader
from src.order_management.multi_leg_daemon import MultiLegBarEvent


def _as_float(row: pd.Series, keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return float(default)


def _features_from_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "semantic_chop": _as_float(row, ["semantic_chop"], 0.0),
        "bpc_semantic_chop": _as_float(
            row, ["bpc_semantic_chop", "semantic_chop"], 0.0
        ),
        "box_prefilter": bool(row.get("box_prefilter", False)),
        "trend_confidence": _as_float(row, ["trend_confidence"], 0.0),
        "trend_direction": str(row.get("trend_direction", "UP")),
        **{k: v for k, v in row.to_dict().items() if k != "timestamp"},
    }


class FeatureStoreBarProvider:
    """Poll latest multi-leg signal features from ``live/shared_feature_bus``."""

    def __init__(
        self,
        *,
        feature_bus_root: str | Path,
        timeframe: str = "2h",
        execution_timeframe: str = "1min",
    ) -> None:
        self.reader = FeatureBusReader(feature_bus_root)
        self.timeframe = str(timeframe)
        self.execution_timeframe = str(execution_timeframe)
        self._last_seen_features: Dict[str, pd.Timestamp] = {}
        self._last_seen_bars: Dict[str, pd.Timestamp] = {}

    def latest_closed_bars(self, symbols: Iterable[str]) -> List[MultiLegBarEvent]:
        out: List[MultiLegBarEvent] = []
        for raw_symbol in symbols:
            symbol = str(raw_symbol).upper()
            signal = self.reader.latest_features(
                symbol=symbol, timeframe=self.timeframe
            )
            if signal is None:
                continue
            signal_ts = pd.Timestamp(signal["timestamp"])
            bars = self.reader.latest_bars_1m(
                symbol=symbol,
                after=self._last_seen_bars.get(symbol),
            )
            if not bars.empty:
                bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
                self._last_seen_bars[symbol] = bars["timestamp"].max()
                for _, bar in bars.iterrows():
                    out.append(self._event_from_signal_and_bar(symbol, signal, bar))
                self._last_seen_features[symbol] = signal_ts
                continue

            if self._last_seen_features.get(symbol) == signal_ts:
                continue
            self._last_seen_features[symbol] = signal_ts
            close = _as_float(signal, ["close", "Close"], 0.0)
            out.append(
                MultiLegBarEvent(
                    symbol=symbol,
                    timestamp=str(signal_ts),
                    high=_as_float(signal, ["high", "High"], close),
                    low=_as_float(signal, ["low", "Low"], close),
                    close=close,
                    atr=_as_float(
                        signal, ["atr14", "atr", "ATR", "volatility_atr"], 0.0
                    ),
                    features=_features_from_row(signal),
                )
            )
        return out

    def _event_from_signal_and_bar(
        self, symbol: str, signal: pd.Series, bar: pd.Series
    ) -> MultiLegBarEvent:
        close = _as_float(bar, ["close", "Close"], _as_float(signal, ["close"], 0.0))
        features = _features_from_row(signal)
        features.update(
            {
                "_signal_timestamp": str(pd.Timestamp(signal["timestamp"])),
                "_execution_timeframe": self.execution_timeframe,
                "_execution_bar_kind": str(bar.get("_bar_kind", "1min")),
            }
        )
        return MultiLegBarEvent(
            symbol=symbol,
            timestamp=str(pd.Timestamp(bar["timestamp"])),
            high=_as_float(bar, ["high", "High"], close),
            low=_as_float(bar, ["low", "Low"], close),
            close=close,
            atr=_as_float(signal, ["atr14", "atr", "ATR", "volatility_atr"], 0.0),
            features=features,
        )
