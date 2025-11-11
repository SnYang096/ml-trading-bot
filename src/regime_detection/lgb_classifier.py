from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .detector import RegimeLabel


@dataclass
class RegimeClassifierConfig:
    learning_rate: float = 0.05
    n_estimators: int = 200
    num_leaves: int = 31
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    class_weight: Optional[Dict[str, float]] = None
    min_data_in_leaf: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: Optional[int] = None
    eval_fraction: float = 0.2
    enable_proba_calibration: bool = False
    calibration_fraction: float = 0.2
    verbose: int = -1


class LGBRegimeClassifier:
    """
    Gradient boosted multi-class classifier for regime detection.

    This augments the rule-based detector with a learnable model.
    """

    def __init__(self, config: Optional[RegimeClassifierConfig] = None) -> None:
        self.config = config or RegimeClassifierConfig()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.label_to_idx = {label: idx for idx, label in enumerate(RegimeLabel)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.feature_names: List[str] = []
        self.eval_results_: Dict[str, float] = {}

    def _prepare_features(
        self, X: pd.DataFrame, required_columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        if required_columns:
            missing = set(required_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
        numeric = X.select_dtypes(include=[np.number]).copy()
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric.dropna(axis=0, inplace=True)
        if numeric.empty:
            raise ValueError("No valid numeric features available for training.")
        return numeric

    def _prepare_labels(self, y: Iterable[RegimeLabel]) -> np.ndarray:
        if isinstance(y, pd.Series):
            series = y.map(self.label_to_idx)
        else:
            series = pd.Series(list(y)).map(self.label_to_idx)
        if series.isna().any():
            raise ValueError("Encountered labels not present in RegimeLabel enum.")
        return series.to_numpy(dtype=int)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[RegimeLabel],
        eval_set: Optional[Sequence[pd.DataFrame]] = None,
        eval_labels: Optional[Sequence[Sequence[RegimeLabel]]] = None,
    ) -> Dict[str, float]:
        X_prepared = self._prepare_features(X)
        y_prepared = self._prepare_labels(y)
        self.feature_names = list(X_prepared.columns)

        params = {
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "random_state": self.config.random_state,
            "class_weight": self.config.class_weight,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "verbosity": self.config.verbose,
        }

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(RegimeLabel),
            **params,
        )

        fit_kwargs = {}
        if eval_set and eval_labels:
            eval_x = [self._prepare_features(df) for df in eval_set]
            eval_y = [self._prepare_labels(lbls) for lbls in eval_labels]
            eval_list = [(ex, ey) for ex, ey in zip(eval_x, eval_y)]
            fit_kwargs["eval_set"] = eval_list
            if self.config.early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.model.fit(X_prepared, y_prepared, **fit_kwargs)

        preds = self.model.predict(X_prepared)
        self.eval_results_ = {
            "train_accuracy": accuracy_score(y_prepared, preds),
            "train_f1_macro": f1_score(y_prepared, preds, average="macro"),
        }

        return self.eval_results_

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X_prepared = self._prepare_features(X, required_columns=self.feature_names)
        preds = self.model.predict(X_prepared)
        labels = [self.idx_to_label[int(idx)] for idx in preds]
        return pd.Series(labels, index=X_prepared.index, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X_prepared = self._prepare_features(X, required_columns=self.feature_names)
        proba = self.model.predict_proba(X_prepared)
        trained_classes = [int(cls) for cls in self.model.classes_]
        trained_labels = [self.idx_to_label[idx].value for idx in trained_classes]
        proba_df = pd.DataFrame(proba, index=X_prepared.index, columns=trained_labels)
        for label in RegimeLabel:
            if label.value not in proba_df.columns:
                proba_df[label.value] = 0.0
        return proba_df[[label.value for label in RegimeLabel]]

    def feature_importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(
            ascending=False
        )

    def classification_report(self, X: pd.DataFrame, y: Sequence[RegimeLabel]) -> str:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X_prepared = self._prepare_features(X, required_columns=self.feature_names)
        y_prepared = self._prepare_labels(y)
        preds = self.model.predict(X_prepared)
        target_names = [label.value for label in RegimeLabel]
        return classification_report(
            y_prepared,
            preds,
            target_names=target_names,
            zero_division=0,
        )

    def confusion_matrix(self, X: pd.DataFrame, y: Sequence[RegimeLabel]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X_prepared = self._prepare_features(X, required_columns=self.feature_names)
        y_prepared = self._prepare_labels(y)
        preds = self.model.predict(X_prepared)
        matrix = confusion_matrix(y_prepared, preds, labels=list(self.idx_to_label.keys()))
        index = [label.value for label in RegimeLabel]
        return pd.DataFrame(matrix, index=index, columns=index)


