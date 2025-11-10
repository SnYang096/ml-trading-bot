"""
Configuration objects for the regime detection module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping


@dataclass(slots=True)
class RegimeDetectorConfig:
    """
    Configuration for the rule-based regime detector.

    The defaults follow the guidance from ``docs/行情：Regime Detection（行情状态识别）.md``
    and are tuned for high-frequency crypto time series (5m/15m/1h).
    """

    hurst_window: int = 100
    regression_window: int = 120
    atr_window: int = 14
    atr_percentile_window: int = 1000
    compression_window: int = 50
    volume_window: int = 120
    smoothing_alpha: float = 0.2
    hmm_enabled: bool = True
    hmm_n_states: int = 4
    hmm_covariance_type: str = "full"
    hmm_random_state: int = 42
    hmm_n_iter: int = 200

    trend_score_weights: Dict[str, float] = field(
        default_factory=lambda: {"hurst": 0.5, "r2": 0.3, "slope": 0.2}
    )

    volatility_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "low": 0.3,
            "mid": 0.45,
            "high": 0.7,
            "extreme": 0.9,
        }
    )

    trend_threshold: float = 0.6
    compression_threshold: float = 0.8
    collapse_return_threshold: float = -0.02
    collapse_vol_threshold: float = 0.85

    multi_timeframe_weights: Mapping[str, float] = field(
        default_factory=lambda: {"5m": 0.4, "15m": 0.35, "1h": 0.25}
    )

    valid_regimes: Iterable[str] = ("range", "pre_breakout", "trending", "collapse")

    def normalized_trend_weights(self) -> Dict[str, float]:
        total = sum(self.trend_score_weights.values())
        if total == 0:
            raise ValueError("trend_score_weights cannot sum to zero.")
        return {key: value / total for key, value in self.trend_score_weights.items()}


