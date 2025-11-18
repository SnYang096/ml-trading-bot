from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class BoostingEvalResult:
    """Diagnostics for cross-sectional boosting models."""

    predictions: pd.Series
    information_coefficients: pd.Series
    rank_ic: pd.Series
    mse_by_timestamp: pd.Series

    def ic_summary(self) -> Dict[str, float]:
        ic = self.information_coefficients.dropna()
        rank_ic = self.rank_ic.dropna()
        return {
            "ic_mean": float(ic.mean()) if not ic.empty else np.nan,
            "ic_std": float(ic.std(ddof=0)) if not ic.empty else np.nan,
            "ic_t": (
                float(ic.mean() / (ic.std(ddof=0) / np.sqrt(len(ic))))
                if ic.std(ddof=0) > 0
                else np.nan
            ),
            "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
            "rank_ic_std": float(rank_ic.std(ddof=0)) if not rank_ic.empty else np.nan,
        }


class CrossSectionalBoostingModel:
    """
    Gradient boosting wrapper for cross-sectional panel data.

    Unlike the linear regressor, this treats each (timestamp, symbol) observation
    as an independent sample while preserving timestamp metadata for evaluation.
    """

    def __init__(
        self,
        estimator: Optional[HistGradientBoostingRegressor] = None,
        timestamp_level: int = 0,
        dropna: bool = True,
    ):
        self.timestamp_level = timestamp_level
        self.dropna = dropna
        self.estimator = estimator or HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            l2_regularization=1.0,
            random_state=42,
        )
        self.feature_cols_: List[str] = []
        self.target_col_: Optional[str] = None
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    def fit(
        self,
        panel: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> "CrossSectionalBoostingModel":
        panel = self._ensure_panel(panel)
        features = [c for c in feature_cols if c in panel.columns]
        if not features:
            raise ValueError(
                "CrossSectionalBoostingModel.fit: no valid feature columns found."
            )
        if target_col not in panel.columns:
            raise ValueError(f"Target column '{target_col}' is missing from panel.")

        df = self._flatten_panel(panel, columns=features + [target_col])

        if self.dropna:
            df = df.dropna(subset=features + [target_col])
        else:
            df = df.fillna(0.0)

        X = df[features].values.astype(float)
        y = df[target_col].values.astype(float)

        self.estimator.fit(X, y)
        self.feature_cols_ = list(features)
        self.target_col_ = target_col
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Model must be fit before calling predict.")
        panel = self._ensure_panel(panel)
        missing = [col for col in self.feature_cols_ if col not in panel.columns]
        if missing:
            raise ValueError(f"Panel is missing required feature columns: {missing}")

        df = self._flatten_panel(panel, columns=self.feature_cols_)
        df = df.fillna(0.0)
        X = df[self.feature_cols_].values.astype(float)
        preds = self.estimator.predict(X)

        predictions = pd.Series(preds, index=df.index, name="predicted_return")
        return predictions

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        panel: pd.DataFrame,
        predictions: Optional[pd.Series] = None,
        target_col: Optional[str] = None,
    ) -> BoostingEvalResult:
        if not self.fitted_:
            raise RuntimeError("Model must be fit before evaluation.")
        panel = self._ensure_panel(panel)
        target_col = target_col or self.target_col_
        if not target_col or target_col not in panel.columns:
            raise ValueError("Target column must be provided for evaluation.")

        if predictions is None:
            predictions = self.predict(panel)

        df = self._flatten_panel(panel, columns=[target_col])
        aligned = df.join(predictions.rename("prediction"), how="inner")

        grouped = aligned.groupby(level=self.timestamp_level)
        ic = grouped.apply(lambda x: x[target_col].corr(x["prediction"]))
        rank_ic = grouped.apply(
            lambda x: x[target_col].corr(x["prediction"], method="spearman")
        )
        mse = grouped.apply(lambda x: np.mean((x[target_col] - x["prediction"]) ** 2))

        return BoostingEvalResult(
            predictions=predictions,
            information_coefficients=ic,
            rank_ic=rank_ic,
            mse_by_timestamp=mse,
        )

    # ------------------------------------------------------------------ #
    def _flatten_panel(
        self,
        panel: pd.DataFrame,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        df = panel[columns].copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.sort_index()
        return df

    @staticmethod
    def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
            raise ValueError(
                "Cross-sectional panel expected MultiIndex (timestamp, symbol). "
                "Construct it with FactorPanelBuilder first."
            )
        return panel
