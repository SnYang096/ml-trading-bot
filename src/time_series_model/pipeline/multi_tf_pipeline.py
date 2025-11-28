"""Multi-timeframe pipeline with quantile regression models (q10, q50, q90), classification models, and volatility model."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.pipeline.training.volatility_model_config import (
    prepare_volatility_model_data,
    get_volatility_model_params,
    load_volatility_model_config,
)
from time_series_model.config.settings import TIMEFRAMES
from time_series_model.pipeline.training.label_utils import (
    future_volatility_label,
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
        self.classification_models: Dict[str, LightGBMTrainer] = (
            {}
        )  # timeframe -> model (for classification mode)
        self.return_models: Dict[str, LightGBMTrainer] = (
            {}
        )  # timeframe -> model (for return regression in classification mode)
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
        # ⚠️  FIXED: Use close[t+1] as entry price to avoid current bar's close
        close_next = data["close"].shift(-1)  # Use next bar's close as entry
        future_returns = close_next.shift(-self.forward_bars) / close_next - 1

        # ✅ Compute future volatility label: RMS of future single-period returns
        future_volatility = future_volatility_label(
            data["close"],
            horizon=self.forward_bars,
            min_periods=max(3, self.forward_bars // 2),
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
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets (returns)
            returns_target, _, _ = self.prepare_targets(data)

            # Create and train quantile regression model
            model = LightGBMTrainer(
                model_type="quantile", quantile_alpha=quantile_alpha
            )
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
        # 加载配置
        try:
            config = load_volatility_model_config()
            trainer_config = config.get("trainer", {})
            use_gpu = trainer_config.get("use_gpu", True)
            n_splits = trainer_config.get("n_splits", 5)
            model_params = get_volatility_model_params(config)
        except Exception as e:
            print(f"   ⚠️ Failed to load volatility model config: {e}, using defaults")
            config = None
            use_gpu = True
            n_splits = 5
            model_params = None

        metrics = {}

        for timeframe, data in engineered_data.items():
            print(f"   Training volatility model for {timeframe}...")

            # Prepare targets (volatility)
            _, volatility_target, _ = self.prepare_targets(data)

            # 使用配置文件进行特征选择
            if config is not None:
                try:
                    X_vol, vol_features, categorical_features = (
                        prepare_volatility_model_data(data, config, feature_loader=None)
                    )
                    print(
                        f"   ✅ Selected {len(vol_features)} volatility features from config"
                    )
                except Exception as e:
                    print(f"   ⚠️ Feature selection failed: {e}, using all features")
                    # Fallback: 排除基础价格/成交量列
                    feature_columns = [
                        col
                        for col in data.columns
                        if col
                        not in [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "volatility_target",
                        ]
                    ]
                    X_vol = data[feature_columns]
                    vol_features = list(X_vol.columns)
                    categorical_features = None
            else:
                # Fallback: 排除基础价格/成交量列
                feature_columns = [
                    col
                    for col in data.columns
                    if col
                    not in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "volatility_target",
                    ]
                ]
                X_vol = data[feature_columns]
                vol_features = list(X_vol.columns)
                categorical_features = None

            # Create and train regression model for volatility
            model = LightGBMTrainer(model_type="regression", use_gpu=use_gpu)

            # 如果配置了模型参数，设置它们
            if model_params:
                model.params = model_params

            model_metrics, _ = model.train(
                X_vol,
                volatility_target,
                n_splits=n_splits,
                use_time_series_cv=True,
                groups=None,
                categorical_features=categorical_features,
            )

            # 存储使用的特征列表
            model._volatility_features = vol_features
            if categorical_features:
                model._categorical_features = categorical_features

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
                col
                for col in data.columns
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
                col
                for col in data.columns
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
                    col
                    for col in data.columns
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
                    col
                    for col in data.columns
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
                    col
                    for col in data.columns
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
                    col
                    for col in data.columns
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
                    col
                    for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                pred = self.classification_models[timeframe].predict(
                    X
                )  # Returns probabilities [0, 1]
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
