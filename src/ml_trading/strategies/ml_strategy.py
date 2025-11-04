"""Main ML trading strategy integrating all components."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.feature_engineering import FeatureEngineer
from ml_trading.pipeline.multi_tf_pipeline import MultiTimeframePipeline
from ml_trading.pipeline.risk_management import RiskManager
from ml_trading.config.settings import TIMEFRAMES


class MLTradingStrategy:
    """Main ML trading strategy using quantile regression (q10, q50, q90) and volatility models."""

    def __init__(
        self,
        forward_bars: int = 1,
        signal_strength_threshold: float = 1.0,
        confidence_threshold: float = 0.3,
        base_position_size: float = 0.1,
    ):
        """
        Initialize the ML trading strategy.

        Args:
            forward_bars: Number of bars ahead for label prediction (default: 1)
            signal_strength_threshold: Minimum signal_strength (q50/vol) to enter trade (default: 1.0)
            confidence_threshold: Minimum confidence (|q50|/(q90-q10)) to enter trade (default: 0.3)
            base_position_size: Base position size multiplier (default: 0.1)
        """
        self.data_loader = MarketDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.pipeline = MultiTimeframePipeline(forward_bars=forward_bars)
        self.risk_manager = RiskManager()
        self.forward_bars = forward_bars
        self.signal_strength_threshold = signal_strength_threshold
        self.confidence_threshold = confidence_threshold
        self.base_position_size = base_position_size
        self.is_trained = False

    def prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data for all timeframes.

        Returns:
            Dictionary mapping timeframe to engineered data
        """
        # Load raw data only if not already provided
        print("Loading market data...")
        if self.data_loader.raw_data is None:
            raw_data = self.data_loader.load_data()
        else:
            raw_data = self.data_loader.raw_data

        # Get multi-timeframe data
        print("Resampling data to multiple timeframes...")
        multi_tf_data = self.data_loader.get_multi_timeframe_data()

        # Engineer features
        print("Engineering features...")
        engineered_data = self.feature_engineer.engineer_features(
            multi_tf_data)

        return engineered_data

    def train_strategy(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Train the complete ML trading strategy.

        Returns:
            Training metrics for all stages
        """
        # Prepare training data
        engineered_data = self.prepare_training_data()

        # Train pipeline
        print("Training multi-timeframe pipeline...")
        metrics = self.pipeline.train_pipeline(engineered_data)

        self.is_trained = True
        return metrics

    def generate_signals(
        self, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals using the four-model architecture.

        Decision Logic:
        - signal_strength = q50_pred / pred_vol (收益风险比)
        - confidence = |q50_pred| / (q90 - q10) (模型信心)
        - Enter trade if signal_strength > k1 AND confidence > k2
        - Position size = base_size * signal_strength * confidence

        Args:
            data: Optional new data for signal generation

        Returns:
            DataFrame with trading signals and positions
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")

        # If no data provided, use data loader to get latest data
        if data is None:
            multi_tf_data = self.data_loader.get_multi_timeframe_data()
            engineered_data = self.feature_engineer.engineer_features(multi_tf_data)
        else:
            # Assume data is already engineered for the base timeframe
            engineered_data = {"5T": data}  # Simplified for demo

        # Generate predictions from all four models
        print("Generating predictions from all models...")
        all_predictions = self.pipeline.predict_all_models(engineered_data)

        # Get reference timeframe for alignment
        reference_tf = list(all_predictions["q50"].keys())[0]
        n_samples = len(all_predictions["q50"][reference_tf])

        # Create ensemble DataFrame
        ensemble_df = pd.DataFrame(index=range(n_samples))

        # Ensemble predictions across timeframes (weighted average)
        for tf in all_predictions["q50"].keys():
            ensemble_df[f"q10_{tf}"] = all_predictions["q10"][tf]
            ensemble_df[f"q50_{tf}"] = all_predictions["q50"][tf]
            ensemble_df[f"q90_{tf}"] = all_predictions["q90"][tf]
            ensemble_df[f"vol_{tf}"] = all_predictions["volatility"][tf]

        # Weighted ensemble: larger timeframes get more weight
        weights = self._get_timeframe_weights(list(all_predictions["q50"].keys()))
        
        # Ensemble q10, q50, q90, volatility predictions
        ensemble_df["q10_ensemble"] = sum(
            ensemble_df[f"q10_{tf}"] * weights[i]
            for i, tf in enumerate(all_predictions["q50"].keys())
        )
        ensemble_df["q50_ensemble"] = sum(
            ensemble_df[f"q50_{tf}"] * weights[i]
            for i, tf in enumerate(all_predictions["q50"].keys())
        )
        ensemble_df["q90_ensemble"] = sum(
            ensemble_df[f"q90_{tf}"] * weights[i]
            for i, tf in enumerate(all_predictions["q50"].keys())
        )
        ensemble_df["vol_ensemble"] = sum(
            ensemble_df[f"vol_{tf}"] * weights[i]
            for i, tf in enumerate(all_predictions["q50"].keys())
        )

        # Calculate derived metrics
        # Interval width: uncertainty measure
        ensemble_df["interval_width"] = ensemble_df["q90_ensemble"] - ensemble_df["q10_ensemble"]
        
        # Confidence: |q50| / interval_width (how certain is the prediction)
        ensemble_df["confidence"] = np.abs(ensemble_df["q50_ensemble"]) / (
            ensemble_df["interval_width"] + 1e-8  # Avoid division by zero
        )

        # Signal strength: q50 / vol (risk-adjusted return)
        ensemble_df["signal_strength"] = ensemble_df["q50_ensemble"] / (
            ensemble_df["vol_ensemble"] + 1e-8  # Avoid division by zero
        )

        # Generate trading signals based on thresholds
        ensemble_df["signal"] = 0  # 0 = Hold, 1 = Long, -1 = Short
        
        # Long signal: positive q50, high signal strength, high confidence
        long_mask = (
            (ensemble_df["q50_ensemble"] > 0) &
            (ensemble_df["signal_strength"] > self.signal_strength_threshold) &
            (ensemble_df["confidence"] > self.confidence_threshold)
        )
        ensemble_df.loc[long_mask, "signal"] = 1

        # Short signal: negative q50, high signal strength (absolute), high confidence
        short_mask = (
            (ensemble_df["q50_ensemble"] < 0) &
            (np.abs(ensemble_df["signal_strength"]) > self.signal_strength_threshold) &
            (ensemble_df["confidence"] > self.confidence_threshold)
        )
        ensemble_df.loc[short_mask, "signal"] = -1

        # Position sizing: base_size * signal_strength * confidence
        ensemble_df["position_size"] = 0.0
        trade_mask = ensemble_df["signal"] != 0
        ensemble_df.loc[trade_mask, "position_size"] = (
            self.base_position_size *
            np.abs(ensemble_df.loc[trade_mask, "signal_strength"]) *
            ensemble_df.loc[trade_mask, "confidence"]
        )
        
        # Cap position size at reasonable maximum (e.g., 1.0 = 100% of portfolio)
        ensemble_df["position_size"] = np.clip(ensemble_df["position_size"], 0.0, 1.0)

        # Apply risk management
        if data is None:
            price_data = self.data_loader.raw_data
        else:
            price_data = data

        # Ensure we have valid price data
        if price_data is None:
            # Create dummy price data if needed
            dates = pd.date_range("2020-01-01", periods=len(ensemble_df), freq="5T")
            price_data = pd.DataFrame(
                {
                    "close": np.ones(len(ensemble_df)) * 100,
                    "open": np.ones(len(ensemble_df)) * 100,
                    "high": np.ones(len(ensemble_df)) * 100,
                    "low": np.ones(len(ensemble_df)) * 100,
                    "volume": np.ones(len(ensemble_df)) * 1000,
                },
                index=dates,
            )

        print("Applying risk management rules...")
        final_df = self.risk_manager.apply_risk_management(ensemble_df, price_data)

        return final_df

    def _get_timeframe_weights(self, timeframes: List[str]) -> List[float]:
        """
        Get weights for each timeframe based on their size.
        Larger timeframes get more weight.

        Args:
            timeframes: List of timeframe strings (e.g., ['5T', '15T', '60T'])

        Returns:
            List of weights that sum to 1.0
        """
        # Extract timeframe minutes and sort
        tf_minutes = [int(tf.rstrip("T")) for tf in timeframes]

        # Create weights proportional to timeframe size
        # Using square root to avoid too much dominance of large TF
        raw_weights = [np.sqrt(minutes) for minutes in tf_minutes]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        return weights

    def optimize_strategy(
        self, n_trials: int = 50
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Optimize the strategy using Optuna.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Best parameters for each model
        """
        # Prepare training data
        engineered_data = self.prepare_training_data()

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
                returns_target, _ = self.pipeline.prepare_targets(data)

                # Create and optimize model
                model = LightGBMModel(model_type="quantile", quantile_alpha=quantile_alpha)
                best_params[f"{model_name}_{timeframe}"] = model.optimize_hyperparameters(
                    X, returns_target, n_trials=n_trials // 4
                )

        # Optimize volatility models
        print("Optimizing volatility models...")
        for timeframe, data in engineered_data.items():
            # Prepare features and targets
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]
            _, volatility_target = self.pipeline.prepare_targets(data)

            # Create and optimize model
            model = LightGBMModel(model_type="regression")
            best_params[f"volatility_{timeframe}"] = model.optimize_hyperparameters(
                X, volatility_target, n_trials=n_trials // 4
            )

        return best_params


# Need to import LightGBMModel to fix the reference in optimize_strategy
from ml_trading.models.lightgbm_model import LightGBMModel
