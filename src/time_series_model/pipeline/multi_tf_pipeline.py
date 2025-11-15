"""Multi-timeframe pipeline with quantile regression models (q10, q50, q90), classification models, and volatility model."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.config.settings import TIMEFRAMES
from time_series_model.pipeline.training.label_utils import (
    log_return_magnitude,
    rolling_rms_volatility,
    rolling_quantile_classification_labels,
)


class MultiTimeframePipeline:
    """Multi-timeframe pipeline with quantile regression (q10, q50, q90), classification, and volatility models."""

    def __init__(self, forward_bars: int = 1, model_type: str = "quantile"):
        """Initialize the multi-timeframe pipeline.
        
        Args:
            forward_bars: Number of bars ahead for label prediction (default: 1)
            model_type: Model type - "quantile" (default), "classification", or "regression"
        """
        # Four models per timeframe: q10, q50, q90, volatility (for quantile mode)
        # Or: classification, return regression, volatility (for classification mode)
        self.q10_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model
        self.q50_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model
        self.q90_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model
        self.classification_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model (for classification mode)
        self.return_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model (for return regression in classification mode)
        self.volatility_models: Dict[str, LightGBMTrainer] = {}  # timeframe -> model
        self.is_trained = False
        self.forward_bars = forward_bars
        self.model_type = model_type

    def prepare_targets(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Prepare targets for quantile regression (returns), classification, and volatility models.

        Args:
            data: Market data with features

        Returns:
            Tuple of (returns_target, volatility_target, classification_target)
            classification_target is None if model_type != "classification"
        """
        # Calculate future returns
        future_returns = data["close"].shift(-self.forward_bars) / data["close"] - 1

        # Volatility proxy: rolling RMS of future returns (trailing window to avoid leakage)
        window = max(self.forward_bars, 5)
        future_volatility = rolling_rms_volatility(
            future_returns,
            window=window,
            min_periods=min(3, window),
        )

        classification_target = None
        if self.model_type == "classification":
            cls_series, _, _, _ = rolling_quantile_classification_labels(
                future_returns,
                window=5000,
                lower_quantile=0.4,
                upper_quantile=0.6,
                min_periods=200,
            )
            if cls_series.nunique() < 2:
                classification_target = (future_returns > 0).astype(int)
            else:
                classification_target = cls_series.reindex_like(future_returns)

        return future_returns, future_volatility, classification_target

    def train_quantile_models(
        self,
        engineered_data: Dict[str, pd.DataFrame],
        quantile_alpha: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train quantile regression models (q10, q50, or q90) for each timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data
            quantile_alpha: Quantile alpha (0.1, 0.5, or 0.9)

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}
        model_dict = {
            0.1: self.q10_models,
            0.5: self.q50_models,
            0.9: self.q90_models,
        }

        target_models = model_dict[quantile_alpha]

        for timeframe, data in engineered_data.items():
            # Prepare features
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets (returns)
            returns_target, _, _ = self.prepare_targets(data)

            # Create and train quantile regression model
            model = LightGBMTrainer(model_type="quantile", quantile_alpha=quantile_alpha)
            model_metrics = model.train(X, returns_target)

            # Store model and metrics
            target_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_volatility_models(
        self,
        engineered_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, float]]:
        """
        Train volatility regression models for each timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}

        for timeframe, data in engineered_data.items():
            # Prepare features
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets (volatility)
            _, volatility_target, _ = self.prepare_targets(data)

            # Create and train regression model for volatility
            model = LightGBMTrainer(model_type="regression")
            model_metrics = model.train(X, volatility_target)

            # Store model and metrics
            self.volatility_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_classification_models(
        self,
        engineered_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, float]]:
        """
        Train classification models for each timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}

        for timeframe, data in engineered_data.items():
            # Prepare features
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets (classification with neutral filtering)
            _, _, classification_target = self.prepare_targets(data)
            if classification_target is None:
                continue
            classification_target = classification_target.dropna().astype(int)
            if classification_target.nunique() < 2:
                continue
            X_cls = X.loc[classification_target.index]

            # Create and train classification model
            model = LightGBMTrainer(model_type="classification")
            model_metrics = model.train(X_cls, classification_target)

            # Store model and metrics
            self.classification_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_return_models(
        self,
        engineered_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, float]]:
        """
        Train return regression models for each timeframe (for magnitude prediction).

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}

        for timeframe, data in engineered_data.items():
            # Prepare features
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets (future returns)
            future_returns, _, _ = self.prepare_targets(data)
            log_mag = log_return_magnitude(future_returns)

            # Create and train return regression model
            model = LightGBMTrainer(model_type="regression")
            model_metrics = model.train(X, log_mag)

            # Store model and metrics
            self.return_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_pipeline(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Train the complete pipeline with models per timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each model type
        """
        if self.model_type == "classification":
            print("Training classification models...")
            classification_metrics = self.train_classification_models(engineered_data)

            print("Training return regression models...")
            return_metrics = self.train_return_models(engineered_data)

            print("Training volatility models...")
            volatility_metrics = self.train_volatility_models(engineered_data)

            self.is_trained = True

            return {
                "classification": classification_metrics,
                "return": return_metrics,
                "volatility": volatility_metrics,
            }
        else:
            # Quantile mode (default)
            print("Training q10 models (10% quantile)...")
            q10_metrics = self.train_quantile_models(engineered_data, 0.1)

            print("Training q50 models (50% quantile / median)...")
            q50_metrics = self.train_quantile_models(engineered_data, 0.5)

            print("Training q90 models (90% quantile)...")
            q90_metrics = self.train_quantile_models(engineered_data, 0.9)

            print("Training volatility models...")
            volatility_metrics = self.train_volatility_models(engineered_data)

            self.is_trained = True

            return {
                "q10": q10_metrics,
                "q50": q50_metrics,
                "q90": q90_metrics,
                "volatility": volatility_metrics,
            }

    def predict_q10(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Make predictions using q10 models."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.q10_models:
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.q10_models[timeframe].predict(X)
                predictions[timeframe] = pred
        return predictions

    def predict_q50(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Make predictions using q50 models."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.q50_models:
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.q50_models[timeframe].predict(X)
                predictions[timeframe] = pred
        return predictions

    def predict_q90(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Make predictions using q90 models."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.q90_models:
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.q90_models[timeframe].predict(X)
                predictions[timeframe] = pred
        return predictions

    def predict_volatility(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Make predictions using volatility models."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.volatility_models:
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.volatility_models[timeframe].predict(X)
                predictions[timeframe] = pred
        return predictions

    def predict_classification(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Make predictions using classification models (returns probabilities)."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.classification_models:
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.classification_models[timeframe].predict(X)  # Returns probabilities [0, 1]
                predictions[timeframe] = pred
        return predictions

    def predict_all_models(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Make predictions using all four model types.

        Returns:
            Dictionary with keys 'q10', 'q50', 'q90', 'volatility',
            each containing a dict mapping timeframe to predictions
        """
        return {
            "q10": self.predict_q10(engineered_data),
            "q50": self.predict_q50(engineered_data),
            "q90": self.predict_q90(engineered_data),
            "volatility": self.predict_volatility(engineered_data),
        }
