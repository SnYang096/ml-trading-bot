"""
Rule-based regime detection orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import RegimeDetectorConfig
from .features import (
    TrendFeatureSet,
    VolatilityFeatureSet,
    StructureFeatureSet,
    average_true_range,
    atr_percentile,
    bollinger_band_width,
    compression_score,
    rolling_hurst_exponent,
    rolling_linear_regression,
    volume_health,
)

if TYPE_CHECKING:  # pragma: no cover
    from .hmm_smoother import RegimeHMMSmoother


class RegimeLabel(str, Enum):
    RANGE = "range"
    PRE_BREAKOUT = "pre_breakout"
    TRENDING = "trending"
    COLLAPSE = "collapse"
    TRANSITION = "transition"


_REGIME_TO_SCORE = {
    RegimeLabel.RANGE: 0,
    RegimeLabel.PRE_BREAKOUT: 1,
    RegimeLabel.TRENDING: 2,
    RegimeLabel.COLLAPSE: -1,
    RegimeLabel.TRANSITION: 0,
}


@dataclass(frozen=True)
class RegimeDetectionResult:
    labels: pd.Series
    trend_features: TrendFeatureSet
    volatility_features: VolatilityFeatureSet
    structure_features: StructureFeatureSet
    decision_factors: pd.DataFrame
    smoothed_labels: Optional[pd.Series] = None
    label_probabilities: Optional[pd.DataFrame] = None


class RuleBasedRegimeDetector:
    """
    Implements the multi-layer regime detection flow described in the documentation.
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self.trend_weights = self.config.normalized_trend_weights()

    # --------------------------------------------------------------------- #
    # Feature extraction
    def _extract_trend_features(self, close: pd.Series) -> TrendFeatureSet:
        hurst = rolling_hurst_exponent(close, self.config.hurst_window)
        slope, r2 = rolling_linear_regression(close, self.config.regression_window)

        slope_norm = (slope - slope.rolling(self.config.regression_window).mean()) / (
            slope.rolling(self.config.regression_window).std()
        )
        slope_norm = slope_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        trend_score = (
            self.trend_weights["hurst"] * hurst
            + self.trend_weights["r2"] * r2
            + self.trend_weights["slope"] * (1 / (1 + np.exp(-slope_norm)))
        )
        trend_score = trend_score.clip(0.0, 1.0)

        return TrendFeatureSet(
            hurst=hurst,
            slope=slope,
            r2=r2,
            trend_score=trend_score,
        )

    def _extract_volatility_features(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> VolatilityFeatureSet:
        atr = average_true_range(high, low, close, self.config.atr_window)
        atr_rank = atr_percentile(atr, self.config.atr_percentile_window).clip(0, 1)
        band_width = (
            bollinger_band_width(close, self.config.regression_window)
            .ffill()
        )
        return VolatilityFeatureSet(
            atr=atr,
            atr_percentile=atr_rank,
            bollinger_width=band_width,
        )

    def _extract_structure_features(
        self,
        close: pd.Series,
        vol_features: VolatilityFeatureSet,
        volume: Optional[pd.Series],
    ) -> StructureFeatureSet:
        compression = compression_score(
            close=close,
            band_width=vol_features.bollinger_width,
            atr_rank=vol_features.atr_percentile,
            window=self.config.compression_window,
        )
        vol_health = (
            volume_health(volume, self.config.volume_window)
            if volume is not None
            else None
        )
        return StructureFeatureSet(
            compression=compression,
            volume_health=vol_health,
        )

    # ------------------------------------------------------------------ #
    def _apply_rules(
        self,
        trend: TrendFeatureSet,
        vol: VolatilityFeatureSet,
        structure: StructureFeatureSet,
        returns: pd.Series,
    ) -> pd.Series:
        cfg = self.config
        labels = pd.Series(index=trend.trend_score.index, dtype=object)

        vol_regime = vol.atr_percentile
        compression = structure.compression
        trend_score = trend.trend_score
        recent_return = returns.fillna(0.0)

        pre_breakout = (vol_regime < cfg.volatility_thresholds["low"]) & (
            compression > cfg.compression_threshold
        )
        trending = (trend_score > cfg.trend_threshold) & (
            vol_regime > cfg.volatility_thresholds["high"]
        )
        range_bound = (vol_regime < cfg.volatility_thresholds["mid"]) & (
            trend_score < cfg.trend_threshold * 0.8
        )
        collapse = (
            (vol_regime > cfg.collapse_vol_threshold)
            & (recent_return < cfg.collapse_return_threshold)
        )

        labels.loc[:] = RegimeLabel.TRANSITION
        labels.loc[range_bound] = RegimeLabel.RANGE
        labels.loc[pre_breakout] = RegimeLabel.PRE_BREAKOUT
        labels.loc[trending] = RegimeLabel.TRENDING
        labels.loc[collapse] = RegimeLabel.COLLAPSE

        return labels.astype(object)

    def detect(
        self,
        data: pd.DataFrame,
        volume_column: str = "volume",
        hmm_smoother: Optional["RegimeHMMSmoother"] = None,
    ) -> RegimeDetectionResult:
        """
        Execute regime detection on a single timeframe.

        Parameters
        ----------
        data:
            DataFrame with columns ``high``, ``low``, ``close``. ``volume`` is optional.
        volume_column:
            Column name to use for volume health heuristics.
        """
        required_columns = {"high", "low", "close"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data[volume_column].astype(float) if volume_column in data else None

        trend_features = self._extract_trend_features(close)
        vol_features = self._extract_volatility_features(high, low, close)
        structure_features = self._extract_structure_features(
            close, vol_features, volume
        )

        returns = close.pct_change().fillna(0.0)
        labels = self._apply_rules(
            trend_features, vol_features, structure_features, returns
        )

        decision_factors = pd.DataFrame(
            {
                "trend_score": trend_features.trend_score,
                "vol_regime": vol_features.atr_percentile,
                "compression": structure_features.compression,
                "return": returns,
            }
        )

        smoothed_labels: Optional[pd.Series] = None
        label_probabilities: Optional[pd.DataFrame] = None
        if hmm_smoother is not None or self.config.hmm_enabled:
            smoother = hmm_smoother
            if smoother is None:
                try:
                    from .hmm_smoother import RegimeHMMSmoother
                except ImportError:
                    smoother = None
                else:
                    smoother = RegimeHMMSmoother(
                        n_states=self.config.hmm_n_states,
                        covariance_type=self.config.hmm_covariance_type,
                        random_state=self.config.hmm_random_state,
                        n_iter=self.config.hmm_n_iter,
                    )
            if smoother is not None:
                try:
                    smoothed_labels, label_probabilities = smoother.smooth(
                        decision_factors, labels
                    )
                except (ValueError, ImportError):
                    smoothed_labels = None
                    label_probabilities = None

        return RegimeDetectionResult(
            labels=labels,
            trend_features=trend_features,
            volatility_features=vol_features,
            structure_features=structure_features,
            decision_factors=decision_factors,
            smoothed_labels=smoothed_labels,
            label_probabilities=label_probabilities,
        )

    # ------------------------------------------------------------------ #
    def detect_multi_timeframe(
        self,
        data_by_timeframe: Mapping[str, pd.DataFrame],
        volume_column: str = "volume",
    ) -> pd.DataFrame:
        """
        Detect regimes for multiple timeframes and aggregate with weighted voting.
        """
        results: MutableMapping[str, RegimeDetectionResult] = {}
        for timeframe, df in data_by_timeframe.items():
            df = df.copy()
            df = df.sort_index()
            results[timeframe] = self.detect(df, volume_column=volume_column)

        scores: Dict[str, pd.Series] = {}
        for timeframe, result in results.items():
            aggregated = result.labels.map(_REGIME_TO_SCORE).astype(float)
            scores[timeframe] = aggregated

        weights = self._resolve_timeframe_weights(data_by_timeframe.keys())

        union_index = pd.Index([])
        for series in scores.values():
            union_index = union_index.union(series.index)

        combined_score = pd.Series(0.0, index=union_index)
        for tf, weight in weights.items():
            combined_score = combined_score.add(
                scores[tf].reindex(union_index).ffill().fillna(0.0)
                * weight,
                fill_value=0.0,
            )

        final_label = combined_score.apply(self._score_to_label)

        assembled = pd.DataFrame({"regime": final_label})
        for timeframe, result in results.items():
            assembled[f"regime_{timeframe}"] = result.labels.reindex(union_index).fillna(
                method="ffill"
            )

        return assembled

    def _resolve_timeframe_weights(
        self, timeframes: Iterable[str]
    ) -> Dict[str, float]:
        cfg_weights = dict(self.config.multi_timeframe_weights)
        missing = [tf for tf in timeframes if tf not in cfg_weights]
        if missing:
            remainder = 1.0 - sum(cfg_weights.values())
            if remainder <= 0:
                equal_weight = 1.0 / (len(cfg_weights) + len(missing))
                for key in cfg_weights:
                    cfg_weights[key] = equal_weight
                for tf in missing:
                    cfg_weights[tf] = equal_weight
            else:
                equal = remainder / len(missing)
                for tf in missing:
                    cfg_weights[tf] = equal
        total = sum(cfg_weights[tf] for tf in timeframes)
        return {tf: cfg_weights[tf] / total for tf in timeframes}

    def _score_to_label(self, score: float) -> RegimeLabel:
        if score >= 1.25:
            return RegimeLabel.TRENDING
        if 0.4 <= score < 1.25:
            return RegimeLabel.PRE_BREAKOUT
        if -0.8 < score < 0.4:
            return RegimeLabel.RANGE
        if score <= -0.8:
            return RegimeLabel.COLLAPSE
        return RegimeLabel.TRANSITION


