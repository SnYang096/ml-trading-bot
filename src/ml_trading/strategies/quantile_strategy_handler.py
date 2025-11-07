"""Strategy handler for quantile regression models (q10, q50, q90)."""

import pandas as pd
import numpy as np
from typing import Dict
from ml_trading.strategies.base_strategy_handler import BaseStrategyHandler
from ml_trading.models.lightgbm_model import LightGBMModel


class QuantileStrategyHandler(BaseStrategyHandler):
    """Strategy handler for quantile regression models (q10, q50, q90)."""

    def generate_signals(
        self, X: pd.DataFrame, data: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Generate trading signals using quantile regression models (q10, q50, q90).

        Signal generation logic:
        - signal_strength = q50_pred / pred_vol (收益风险比)
        - confidence = |q50_pred| / (q90 - q10) (模型信心)
        - Enter trade if signal_strength > k1 AND confidence > k2
        - Position size = base_size * signal_strength * confidence

        Args:
            X: Feature matrix
            data: Original data DataFrame (for index)
            timeframe: Timeframe string

        Returns:
            DataFrame with trading signals and positions
        """
        # Validate models exist
        if timeframe not in self.pipeline.q50_models:
            raise ValueError(
                f"Model for timeframe {timeframe} not found. "
                f"Available: {list(self.pipeline.q50_models.keys())}"
            )

        # Get predictions from quantile models
        q10_pred = self.pipeline.q10_models[timeframe].predict(X)
        q50_pred = self.pipeline.q50_models[timeframe].predict(X)
        q90_pred = self.pipeline.q90_models[timeframe].predict(X)
        vol_pred = self.pipeline.volatility_models[timeframe].predict(X)

        # Create DataFrame with predictions
        signals_df = pd.DataFrame(
            {
                "q10": q10_pred,
                "q50": q50_pred,
                "q90": q90_pred,
                "vol": vol_pred,
            },
            index=data.index
            if hasattr(data, 'index') else range(len(q50_pred)),
        )

        # Calculate derived metrics
        # Interval width: uncertainty measure
        signals_df["interval_width"] = signals_df["q90"] - signals_df["q10"]

        # Confidence: |q50| / interval_width (how certain is the prediction)
        signals_df["confidence"] = np.abs(signals_df["q50"]) / (
            signals_df["interval_width"] + 1e-8  # Avoid division by zero
        )

        # Signal strength: q50 / vol (risk-adjusted return)
        signals_df["signal_strength"] = signals_df["q50"] / (
            signals_df["vol"] + 1e-8  # Avoid division by zero
        )

        # Generate trading signals based on thresholds
        signals_df["signal"] = 0  # 0 = Hold, 1 = Long, -1 = Short

        # Long signal: positive q50, high signal strength, high confidence
        long_mask = (
            (signals_df["q50"] > 0)
            & (signals_df["signal_strength"] > self.signal_strength_threshold)
            & (signals_df["confidence"] > self.confidence_threshold)
        )
        signals_df.loc[long_mask, "signal"] = 1

        # Short signal: negative q50, high signal strength (absolute), high confidence
        short_mask = (
            (signals_df["q50"] < 0)
            & (np.abs(signals_df["signal_strength"]) > self.signal_strength_threshold)
            & (signals_df["confidence"] > self.confidence_threshold)
        )
        signals_df.loc[short_mask, "signal"] = -1

        # Position sizing: base_size * signal_strength * confidence
        signals_df["position_size"] = 0.0
        trade_mask = signals_df["signal"] != 0
        signals_df.loc[trade_mask, "position_size"] = (
            self.base_position_size
            * np.abs(signals_df.loc[trade_mask, "signal_strength"])
            * signals_df.loc[trade_mask, "confidence"]
        )

        return signals_df

    def optimize_models(
        self, engineered_data: Dict[str, pd.DataFrame], n_trials: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize quantile regression models (q10, q50, q90).

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data
            n_trials: Number of optimization trials

        Returns:
            Dictionary of best parameters for each quantile model
        """
        best_params = {}

        # Optimize quantile models (q10, q50, q90)
        for quantile_alpha, model_name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
            print(f"Optimizing {model_name} models...")
            for timeframe, data in engineered_data.items():
                # Prepare features and targets
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]
                returns_target, _, _ = self.pipeline.prepare_targets(data)

                # Create and optimize model
                model = LightGBMModel(model_type="quantile",
                                      quantile_alpha=quantile_alpha)
                best_params[f"{model_name}_{timeframe}"] = model.optimize_hyperparameters(
                    X, returns_target, n_trials=n_trials // 4
                )

        return best_params

