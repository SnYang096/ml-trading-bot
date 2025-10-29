"""Utilities for detecting feature importance drift across rolling windows."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


class DriftDetector:
    """Feature-importance based drift detector."""

    def __init__(
        self,
        *,
        js_threshold: float = 0.3,
        overlap_threshold: int = 3,
        window_size: int = 5,
        min_common_features: int = 10,
    ) -> None:
        self.js_threshold = js_threshold
        self.overlap_threshold = overlap_threshold
        self.window_size = window_size
        self.min_common_features = min_common_features
        self.importance_history: list[Dict[str, float]] = []

    def add_importance(self, importance: Dict[str, float]) -> None:
        """Append a new importance snapshot to the history."""

        self.importance_history.append(importance.copy())

        max_history = self.window_size * 2
        if len(self.importance_history) > max_history:
            self.importance_history = self.importance_history[-max_history:]

    def should_trigger(self) -> Tuple[bool, Dict[str, Any]]:
        """Return whether drift should trigger and supporting diagnostics."""

        if len(self.importance_history) < self.window_size + 1:
            return False, {"reason": "insufficient_history"}

        latest = pd.Series(self.importance_history[-1])
        historical = pd.DataFrame(self.importance_history[-self.window_size -
                                                          1:-1]).mean()

        common_features = set(latest.index) & set(historical.index)
        if len(common_features) < self.min_common_features:
            return False, {"reason": "insufficient_common_features"}

        latest_aligned = latest[list(common_features)].fillna(0.0)
        historical_aligned = historical[list(common_features)].fillna(0.0)

        latest_values = latest_aligned.values
        historical_values = historical_aligned.values

        if not np.any(latest_values) or not np.any(historical_values):
            return False, {"reason": "degenerate_importances"}

        js_div = jensenshannon(latest_values, historical_values)

        top5_current = set(latest_aligned.nlargest(5).index)
        top5_historical = set(historical_aligned.nlargest(5).index)
        overlap = len(top5_current & top5_historical)

        importance_change = float(
            np.mean(np.abs(latest_values - historical_values)))

        diagnostics: Dict[str, Any] = {
            "js_divergence": float(js_div),
            "top5_overlap": overlap,
            "importance_change": importance_change,
            "latest_top5": list(top5_current),
            "historical_top5": list(top5_historical),
        }

        triggered = (js_div > self.js_threshold
                     or overlap < self.overlap_threshold
                     or importance_change > 0.1)

        diagnostics["triggered"] = triggered
        return triggered, diagnostics


__all__ = ["DriftDetector"]
