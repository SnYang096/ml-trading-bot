"""Feature-store backed provider for multi-leg live strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.features.semantic_chop import (
    normalize_semantic_chop_aliases,
    resolve_feature_float,
    resolve_semantic_chop,
)
from src.time_series_model.live.multileg_runtime_features import trend_direction_label
from src.live_data_stream.feature_bus import FeatureBusReader
from src.order_management.multi_leg_daemon import MultiLegBarEvent


def _as_float(row: pd.Series, keys: List[str], default: float = 0.0) -> float:
    return resolve_feature_float(row, keys, default=default)


def _features_from_row(row: pd.Series) -> Dict[str, Any]:
    raw = {k: v for k, v in row.to_dict().items() if k != "timestamp"}
    normalize_semantic_chop_aliases(raw)
    chop = resolve_semantic_chop(raw, default=0.0)
    return {
        **raw,
        "semantic_chop": float(chop if chop is not None else 0.0),
        "bpc_semantic_chop": float(chop if chop is not None else 0.0),
        "box_prefilter": bool(row.get("box_prefilter", False)),
        "trend_confidence": _as_float(
            row, ["trend_confidence", "trend_confidence_f"], 0.0
        ),
        "trend_direction": trend_direction_label(raw),
    }


class FeatureStoreBarProvider:
    """Poll latest multi-leg signal features from ``live/shared_feature_bus``."""

    def __init__(
        self,
        *,
        feature_bus_root: str | Path,
        timeframe: str = "2h",
        execution_timeframe: str = "1min",
        initial_backfill_bars: int = 1,
    ) -> None:
        self.reader = FeatureBusReader(feature_bus_root)
        self.timeframe = str(timeframe)
        self.execution_timeframe = str(execution_timeframe)
        self.initial_backfill_bars = max(0, int(initial_backfill_bars))
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
                latest_ts = bars["timestamp"].max()
                if symbol not in self._last_seen_bars:
                    bars = bars.tail(self.initial_backfill_bars)
                self._last_seen_bars[symbol] = latest_ts
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
                    features={
                        **_features_from_row(signal),
                        "_signal_timeframe": self.timeframe,
                    },
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
                "_signal_timeframe": self.timeframe,
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
