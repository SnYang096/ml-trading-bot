"""Strategy handler for classification, return regression, and volatility models."""

import pandas as pd
import numpy as np
from typing import Dict
from time_series_model.strategies.base_strategy_handler import BaseStrategyHandler
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.pipeline.training.label_utils import invert_log_return_magnitude


class ClassificationStrategyHandler(BaseStrategyHandler):
    """Strategy handler for classification, return regression, and volatility models."""

    def __init__(
        self,
        pipeline,
        signal_strength_threshold: float = 1.0,
        confidence_threshold: float = 0.3,
        base_position_size: float = 0.1,
        classification_threshold: float = 0.5,
    ):
        """
        Initialize the classification strategy handler.

        Args:
            pipeline: MultiTimeframePipeline instance
            signal_strength_threshold: Minimum signal strength to enter trade
            confidence_threshold: Minimum confidence to enter trade
            base_position_size: Base position size multiplier
            classification_threshold: Probability threshold for classification model
        """
        super().__init__(pipeline, signal_strength_threshold,
                         confidence_threshold, base_position_size)
        self.classification_threshold = classification_threshold

    def generate_signals(self, X: pd.DataFrame, data: pd.DataFrame,
                         timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals using classification, return regression, and volatility models.

        Signal generation logic:
        - Use classification probability to determine direction
        - Use return regression to predict magnitude
        - Use volatility model to assess risk
        - Enter long if prob > threshold, positive return, high combined confidence
        - Enter short if prob < (1 - threshold), negative return, high combined confidence
        - Position size = base_size * |return_pred| * combined_confidence / vol

        Args:
            X: Feature matrix
            data: Original data DataFrame (for index)
            timeframe: Timeframe string

        Returns:
            DataFrame with trading signals and positions
        """
        # Validate models exist
        if timeframe not in self.pipeline.classification_models:
            raise ValueError(
                f"Classification model for timeframe {timeframe} not found. "
                f"Available: {list(self.pipeline.classification_models.keys())}"
            )
        if timeframe not in self.pipeline.return_models:
            raise ValueError(
                f"Return regression model for timeframe {timeframe} not found. "
                f"Available: {list(self.pipeline.return_models.keys())}")
        if timeframe not in self.pipeline.volatility_models:
            raise ValueError(
                f"Volatility model for timeframe {timeframe} not found. "
                f"Available: {list(self.pipeline.volatility_models.keys())}")

        # Get predictions from all three models
        # 1. Classification: probability [0, 1] where 1 = up, 0 = down
        class_proba = self.pipeline.classification_models[timeframe].predict(X)
        # 2. Return regression: predicted log-magnitude → convert back to absolute return
        return_log_pred = self.pipeline.return_models[timeframe].predict(X)
        return_pred = invert_log_return_magnitude(return_log_pred)
        # 3. Volatility: predicted volatility
        vol_pred = self.pipeline.volatility_models[timeframe].predict(X)

        # Create DataFrame with predictions
        signals_df = pd.DataFrame(
            {
                "class_proba":
                class_proba,  # Probability of up (1) vs down (0)
                "return_pred": return_pred,  # Predicted return magnitude (>=0)
                "vol": vol_pred,  # Predicted volatility
            },
            index=data.index if hasattr(data, 'index') else range(
                len(class_proba)),
        )

        # Calculate derived metrics using all three models
        # Confidence: distance from classification threshold (0.5)
        signals_df["confidence"] = np.abs(signals_df["class_proba"] -
                                          0.5) * 2  # Scale to [0, 1]

        # Signal strength: (return_pred * direction) / vol (risk-adjusted return)
        # Use classification probability to determine direction, return_pred for magnitude
        # If class_proba > 0.5, use positive return_pred; else use negative return_pred
        direction = np.where(signals_df["class_proba"] > 0.5, 1, -1)
        signals_df["signal_strength"] = (
            signals_df["return_pred"] * direction) / (
                signals_df["vol"] + 1e-8)  # Avoid division by zero

        # Combined confidence: classification confidence * return magnitude confidence
        # Return magnitude confidence: |return_pred| / (vol + small_value)
        return_confidence = signals_df["return_pred"] / (signals_df["vol"] +
                                                         1e-8)
        signals_df["combined_confidence"] = signals_df["confidence"] * np.clip(
            return_confidence, 0, 1)

        # Generate trading signals based on thresholds
        signals_df["signal"] = 0  # 0 = Hold, 1 = Long, -1 = Short

        # Long signal: prob > threshold, positive return, high combined confidence
        long_mask = (
            (signals_df["class_proba"] > self.classification_threshold)
            & (signals_df["return_pred"] > 0)
            & (signals_df["combined_confidence"] > self.confidence_threshold))
        signals_df.loc[long_mask, "signal"] = 1

        # Short signal: prob < (1 - threshold), negative return, high combined confidence
        short_mask = (
            (signals_df["class_proba"] < (1 - self.classification_threshold))
            & (signals_df["return_pred"] < 0)
            & (signals_df["combined_confidence"] > self.confidence_threshold))
        signals_df.loc[short_mask, "signal"] = -1

        # Position sizing: base_size * |return_pred| * combined_confidence / vol
        # Use return magnitude and combined confidence for better position sizing
        signals_df["position_size"] = 0.0
        trade_mask = signals_df["signal"] != 0
        signals_df.loc[trade_mask, "position_size"] = (
            self.base_position_size *
            np.abs(signals_df.loc[trade_mask, "return_pred"]) *
            signals_df.loc[trade_mask, "combined_confidence"] /
            (signals_df.loc[trade_mask, "vol"] + 1e-8))

        return signals_df

    def optimize_models(self, engineered_data: Dict[str, pd.DataFrame],
                        n_trials: int) -> Dict[str, Dict[str, float]]:
        """
        Optimize classification models.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data
            n_trials: Number of optimization trials

        Returns:
            Dictionary of best parameters for each classification model
        """
        best_params = {}

        # Optimize classification models
        print("Optimizing classification models...")
        for timeframe, data in engineered_data.items():
            # Prepare features and targets
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]
            _, _, classification_target = self.pipeline.prepare_targets(data)

            # Create and optimize model
            model = LightGBMTrainer(model_type="classification")
            best_params[
                f"classification_{timeframe}"] = model.optimize_hyperparameters(
                    X, classification_target, n_trials=n_trials // 2)

        return best_params
