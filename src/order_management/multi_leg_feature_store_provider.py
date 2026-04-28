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
    ) -> None:
        self.reader = FeatureBusReader(feature_bus_root)
        self.timeframe = str(timeframe)
        self._last_seen: Dict[str, pd.Timestamp] = {}

    def latest_closed_bars(self, symbols: Iterable[str]) -> List[MultiLegBarEvent]:
        out: List[MultiLegBarEvent] = []
        for raw_symbol in symbols:
            symbol = str(raw_symbol).upper()
            row = self.reader.latest_features(
                symbol=symbol,
                timeframe=self.timeframe,
                after=self._last_seen.get(symbol),
            )
            if row is None:
                continue
            ts = pd.Timestamp(row["timestamp"])
            self._last_seen[symbol] = ts
            close = _as_float(row, ["close", "Close"], 0.0)
            out.append(
                MultiLegBarEvent(
                    symbol=symbol,
                    timestamp=str(ts),
                    high=_as_float(row, ["high", "High"], close),
                    low=_as_float(row, ["low", "Low"], close),
                    close=close,
                    atr=_as_float(row, ["atr14", "atr", "ATR", "volatility_atr"], 0.0),
                    features=_features_from_row(row),
                )
            )
        return out
