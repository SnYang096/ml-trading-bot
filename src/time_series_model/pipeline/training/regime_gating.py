from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from regime_detection.detector import RegimeLabel, RuleBasedRegimeDetector
from time_series_model.models.lightgbm_model import LightGBMModel


def _default_feature_filter(
        keywords: Sequence[str]) -> Callable[[pd.DataFrame], List[str]]:
    """
	Create a feature selector that prefers columns containing any of the given keywords.
	Falls back to "all non-price columns" if no keyword matches are found.
	"""
    lower_keywords = [k.lower() for k in keywords]

    def selector(df: pd.DataFrame) -> List[str]:
        exclude = {"open", "high", "low", "close", "volume"}
        candidates = [c for c in df.columns if c not in exclude]
        if not candidates:
            return []
        matched = [
            c for c in candidates
            if any(k in c.lower() for k in lower_keywords)
        ]
        return matched if matched else candidates

    return selector


def _compute_future_returns(close: pd.Series, forward_bars: int) -> pd.Series:
    """
	Compute future returns over the next 'forward_bars' periods.
	"""
    close = close.astype(float)
    ret = (close.shift(-forward_bars) / close) - 1.0
    return ret


@dataclass
class ExpertConfig:
    name: str
    handled_regimes: Sequence[RegimeLabel]
    model_type: str = "regression"  # "regression" | "quantile" | "classification"
    quantile_alpha: Optional[float] = None
    feature_selector: Optional[Callable[[pd.DataFrame], List[str]]] = None
    forward_bars: Optional[int] = None
    allowed_timeframes: Optional[Sequence[str]] = None


def default_expert_configs() -> List[ExpertConfig]:
    """
	Provide three default experts (aligned with documentation):
	- Momentum@1h (TRENDING)
	- MeanReversion@15m (RANGE)
	- Breakout@1h/4h (PRE_BREAKOUT)
	"""
    return [
        ExpertConfig(
            name="momentum",
            handled_regimes=[RegimeLabel.TRENDING],
            model_type="regression",
            feature_selector=_default_feature_filter(
                ["trend", "slope", "adx", "ma", "hurst", "r2", "momentum"]),
            forward_bars=6,
            allowed_timeframes=("1H", "60T"),
        ),
        ExpertConfig(
            name="mean_reversion",
            handled_regimes=[RegimeLabel.RANGE],
            model_type="regression",
            feature_selector=_default_feature_filter([
                "bb", "bollinger", "rsi", "stoch", "poc", "reversal", "zscore"
            ]),
            forward_bars=2,
            allowed_timeframes=("15T", "15m"),
        ),
        ExpertConfig(
            name="breakout",
            handled_regimes=[RegimeLabel.PRE_BREAKOUT],
            model_type="regression",
            feature_selector=_default_feature_filter([
                "compression", "atr_percentile", "bandwidth", "volume",
                "entropy"
            ]),
            forward_bars=6,
            allowed_timeframes=("1H", "60T", "4H", "240T"),
        ),
    ]


class RegimeGatedTimeSeriesModel:
    """
	Train regime-specialized experts per timeframe and combine with regime probabilities.
	- Inputs: engineered features per timeframe (DataFrame with price + features)
	- Labels: future returns over forward_bars
	- Gating: regime labels/probabilities from RuleBasedRegimeDetector (or provided externally)
	"""

    def __init__(self,
                 forward_bars: int,
                 include_regime_features: bool = True) -> None:
        self.forward_bars = int(forward_bars)
        self.include_regime_features = bool(include_regime_features)
        # stores: experts[expert_name][timeframe] = LightGBMModel
        self.experts: Dict[str, Dict[str, LightGBMModel]] = {}
        self.trained_: bool = False

    def _select_features(
            self, df: pd.DataFrame,
            selector: Optional[Callable[[pd.DataFrame],
                                        List[str]]]) -> List[str]:
        if selector is None:
            # fallback: all non-price columns
            exclude = {"open", "high", "low", "close", "volume"}
            return [c for c in df.columns if c not in exclude]
        cols = selector(df)
        return [c for c in cols if c in df.columns]

    def _detect_regimes(
        self,
        data_by_timeframe: Mapping[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
		Run rule-based regime detection for each timeframe.
		Returns a mapping timeframe -> pd.Series of RegimeLabel aligned to that timeframe's index.
		"""
        detector = RuleBasedRegimeDetector()
        labels_by_tf: Dict[str, pd.Series] = {}
        for tf, df in data_by_timeframe.items():
            df = df.copy().sort_index()
            res = detector.detect(df)
            labels_by_tf[tf] = res.labels.reindex(df.index).ffill()
        return labels_by_tf

    def train(
        self,
        engineered_data: Mapping[str, pd.DataFrame],
        expert_configs: Optional[Sequence[ExpertConfig]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
		Train experts per timeframe. Each expert uses rows belonging to its handled regimes.
		Returns training metrics per expert and timeframe.
		"""
        if expert_configs is None:
            expert_configs = default_expert_configs()

        regime_labels = self._detect_regimes(engineered_data)
        self.experts = {cfg.name: {} for cfg in expert_configs}

        all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
            cfg.name: {}
            for cfg in expert_configs
        }

        for timeframe, df in engineered_data.items():
            df = df.copy().sort_index()
            labels = regime_labels[timeframe].reindex(df.index).astype(object)

            for cfg in expert_configs:
                # Enforce per-expert allowed timeframes if specified
                if cfg.allowed_timeframes is not None and timeframe not in cfg.allowed_timeframes:
                    continue
                X_cols = self._select_features(df, cfg.feature_selector)
                if not X_cols:
                    continue
                X = df[X_cols].copy()
                # Optionally inject regime features (one-hot from labels)
                if self.include_regime_features:
                    reg_onehot = pd.DataFrame(
                        0.0,
                        index=X.index,
                        columns=[e.value for e in RegimeLabel])
                    for lab in RegimeLabel:
                        reg_onehot.loc[labels == lab, lab.value] = 1.0
                    # rename columns to avoid collision
                    reg_onehot = reg_onehot.add_prefix("regime_")
                    X = pd.concat([X, reg_onehot], axis=1)
                mask = labels.isin(cfg.handled_regimes)
                fwd = int(
                    cfg.forward_bars
                ) if cfg.forward_bars is not None else self.forward_bars
                future_ret = _compute_future_returns(df["close"], fwd)
                y = future_ret[mask].dropna()
                if y.empty:
                    continue

                Xy_idx = y.index.intersection(X.index)
                X_fit = X.loc[Xy_idx]
                y_fit = y.loc[Xy_idx]
                if len(y_fit) < 100:  # simple guard
                    continue

                if cfg.model_type == "quantile":
                    alpha = cfg.quantile_alpha if cfg.quantile_alpha is not None else 0.5
                    model = LightGBMModel(model_type="quantile",
                                          quantile_alpha=float(alpha))
                elif cfg.model_type == "classification":
                    # fall back to regression target sign if classification requested
                    model = LightGBMModel(model_type="classification")
                    y_fit_cls = (y_fit > 0).astype(int)
                    metrics = model.train(X_fit, y_fit_cls)
                    self.experts[cfg.name][timeframe] = model
                    all_metrics[cfg.name][timeframe] = metrics
                    continue
                else:
                    model = LightGBMModel(model_type="regression")

                metrics = model.train(X_fit, y_fit)
                self.experts[cfg.name][timeframe] = model
                all_metrics[cfg.name][timeframe] = metrics

        self.trained_ = True
        return all_metrics

    def predict(
        self,
        engineered_data: Mapping[str, pd.DataFrame],
        regime_probs: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.Series]:
        """
		Make regime-weighted predictions per timeframe.
		- regime_probs: optional mapping timeframe -> DataFrame with columns named by RegimeLabel values;
		                if None, hard-assign one-hot using rule-based detector.
		Returns: mapping timeframe -> pd.Series of weighted predictions aligned to input index.
		"""
        if not self.trained_:
            raise RuntimeError("Model must be trained before calling predict.")

        # Build regime probabilities if not provided (one-hot from detector labels)
        if regime_probs is None:
            labels = self._detect_regimes(engineered_data)
            prob_maps: Dict[str, pd.DataFrame] = {}
            for tf, series in labels.items():
                index = series.index
                cols = [e.value for e in RegimeLabel]
                prob_df = pd.DataFrame(0.0, index=index, columns=cols)
                for lab in RegimeLabel:
                    prob_df.loc[series == lab, lab.value] = 1.0
                prob_maps[tf] = prob_df
            regime_probs = prob_maps

        # Weighted sum of expert outputs by sum of handled-regime probabilities
        results: Dict[str, pd.Series] = {}
        for timeframe, df in engineered_data.items():
            df = df.copy().sort_index()
            weights_by_expert: Dict[str, pd.Series] = {}
            for expert_name, models_by_tf in self.experts.items():
                if timeframe not in models_by_tf:
                    continue
                model = models_by_tf[timeframe]
                # choose available features
                X_cols = [
                    c for c in df.columns
                    if c not in {"open", "high", "low", "close", "volume"}
                ]
                if not X_cols:
                    continue
                X = df[X_cols].copy()
                # If the model was trained with regime features, add current regime probs as features
                if self.include_regime_features:
                    # Build regime probabilities if not provided
                    if not (isinstance(regime_probs, Mapping)
                            and timeframe in regime_probs):
                        # one-hot from detector
                        labels_map = self._detect_regimes({timeframe: df})
                        series = labels_map[timeframe]
                        prob_df = pd.DataFrame(
                            0.0,
                            index=df.index,
                            columns=[e.value for e in RegimeLabel])
                        for lab in RegimeLabel:
                            prob_df.loc[series == lab, lab.value] = 1.0
                    else:
                        prob_df = regime_probs[timeframe].reindex(
                            df.index).ffill().fillna(0.0)
                        # ensure all columns present
                        for lab in RegimeLabel:
                            if lab.value not in prob_df.columns:
                                prob_df[lab.value] = 0.0
                        prob_df = prob_df[[lab.value for lab in RegimeLabel]]
                    reg_feats = prob_df.add_prefix("regime_")
                    X = pd.concat([X, reg_feats], axis=1)
                preds = pd.Series(model.predict(X), index=X.index, dtype=float)

                # compute gating weight = sum of probs of this expert's regimes
                if isinstance(regime_probs,
                              Mapping) and timeframe in regime_probs:
                    prob_df = regime_probs[timeframe]
                    # sum probabilities across handled regimes
                    handled_cols = [
                        lab.value for lab in self._handled_regimes(expert_name)
                    ]
                    available = [
                        c for c in handled_cols if c in prob_df.columns
                    ]
                    if available:
                        weight = prob_df[available].sum(axis=1).reindex(
                            preds.index).ffill().fillna(0.0)
                    else:
                        weight = pd.Series(0.0, index=preds.index)
                else:
                    weight = pd.Series(1.0, index=preds.index)

                weights_by_expert[expert_name] = (preds * weight).astype(float)

            if not weights_by_expert:
                results[timeframe] = pd.Series(0.0, index=df.index)
                continue
            # Sum weighted expert predictions; normalize by total weight where weight>0
            total = None
            total_weight = None
            for expert_name, series in weights_by_expert.items():
                if total is None:
                    total = series.copy()
                    total_weight = series.abs(
                    ) * 0  # init zeros with same index
                else:
                    total = total.add(series, fill_value=0.0)
                # approximate weight accumulation as the gating weight magnitude; safer: infer from regime_probs
            # If regime_probs provided, we can compute explicit total gating weight:
            if isinstance(regime_probs, Mapping) and timeframe in regime_probs:
                prob_df = regime_probs[timeframe]
                # cap row-wise prob sum to [0, 1]
                row_weight = prob_df.sum(axis=1).clip(0.0, 1.0).reindex(
                    total.index).ffill().fillna(1.0)
                denom = row_weight.replace(0.0, 1.0)
                results[timeframe] = (total / denom).astype(float)
            else:
                results[timeframe] = total.astype(float)

        return results

    def _handled_regimes(self, expert_name: str) -> List[RegimeLabel]:
        # Helper to reconstruct handled regimes from expert naming conventions used in defaults
        if expert_name == "momentum":
            return [RegimeLabel.TRENDING]
        if expert_name == "mean_reversion":
            return [RegimeLabel.RANGE]
        if expert_name == "breakout":
            return [RegimeLabel.PRE_BREAKOUT]
        # Fallback: consider all regimes
        return [e for e in RegimeLabel]
