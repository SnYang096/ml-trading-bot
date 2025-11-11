"""
Hidden Markov Model based smoothing for regime detection.

The implementation relies on ``hmmlearn`` to estimate a Gaussian HMM over the
rule-based decision factors, and then remaps the inferred hidden states back to
the original discrete regime labels using majority voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .detector import RegimeLabel

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - handled lazily
    GaussianHMM = None


@dataclass
class RegimeHMMSmoother:
    """
    HMM-based smoother that enhances the rule-based detector output.
    """

    n_states: int = 4
    covariance_type: str = "full"
    random_state: int = 42
    n_iter: int = 200
    min_sequence_length: int = 200

    def __post_init__(self) -> None:
        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.state_label_map: Dict[int, RegimeLabel] = {}

    def fit(
        self, features: pd.DataFrame, base_labels: pd.Series
    ) -> "RegimeHMMSmoother":
        if GaussianHMM is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "hmmlearn is required for RegimeHMMSmoother. Install it via "
                "`pip install hmmlearn`."
            )

        features = self._prepare_features(features)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )
        self.model.fit(X)
        hidden_states = self.model.predict(X)
        self.state_label_map = self._derive_state_map(hidden_states, base_labels)
        return self

    def smooth(
        self, features: pd.DataFrame, base_labels: pd.Series
    ) -> Tuple[pd.Series, pd.DataFrame]:
        if len(features) < self.min_sequence_length:
            raise ValueError(
                f"Sequence too short for HMM smoothing ("
                f"{len(features)} < {self.min_sequence_length})."
            )

        if self.model is None or self.scaler is None:
            self.fit(features, base_labels)

        assert self.model is not None
        assert self.scaler is not None

        features_clean = self._prepare_features(features)
        X = self.scaler.transform(features_clean.values)

        hidden_states = self.model.predict(X)
        state_probs = self.model.predict_proba(X)

        smoothed_labels = pd.Series(
            [self.state_label_map[state] for state in hidden_states],
            index=features.index,
            dtype=RegimeLabel,
        )

        label_probs = pd.DataFrame(
            0.0,
            index=features.index,
            columns=[label.value for label in RegimeLabel],
        )
        for state, label in self.state_label_map.items():
            label_probs[label.value] += state_probs[:, state]

        row_sum = label_probs.sum(axis=1)
        label_probs = label_probs.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

        return smoothed_labels, label_probs

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")
        if features.isna().all(axis=None):
            raise ValueError("features must contain at least one non-NaN value.")
        filled = features.copy()
        filled = filled.ffill().bfill()
        return filled

    @staticmethod
    def _derive_state_map(
        hidden_states: np.ndarray, base_labels: pd.Series
    ) -> Dict[int, RegimeLabel]:
        label_series = base_labels.astype(RegimeLabel)
        mapping: Dict[int, RegimeLabel] = {}
        for state in np.unique(hidden_states):
            mask = hidden_states == state
            if mask.sum() == 0:
                continue
            majority_label = label_series[mask].mode()
            if majority_label.empty:
                majority_label = pd.Series([RegimeLabel.TRANSITION])
            mapping[int(state)] = RegimeLabel(majority_label.iloc[0])
        return mapping


