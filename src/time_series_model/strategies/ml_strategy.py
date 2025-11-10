"""Main ML trading strategy integrating all components."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_tools.data_loader import MarketDataLoader
from data_tools.feature_engineering import FeatureEngineer
from time_series_model.pipeline.multi_tf_pipeline import MultiTimeframePipeline
from time_series_model.pipeline.risk_management import RiskManager
from time_series_model.config.settings import TIMEFRAMES
from time_series_model.strategies.quantile_strategy_handler import QuantileStrategyHandler
from time_series_model.strategies.classification_strategy_handler import ClassificationStrategyHandler
from time_series_model.models.lightgbm_model import LightGBMModel


class MLTradingStrategy:
    """Main ML trading strategy using quantile regression (q10, q50, q90), classification, or regression models with volatility models."""

    def __init__(
        self,
        forward_bars: int = 1,
        signal_strength_threshold: float = 1.0,
        confidence_threshold: float = 0.3,
        base_position_size: float = 0.1,
        model_type: str = "quantile",
        classification_threshold: float = 0.5,
    ):
        """
        Initialize the ML trading strategy.

        Args:
            forward_bars: Number of bars ahead for label prediction (default: 1)
            signal_strength_threshold: Minimum signal_strength (q50/vol) to enter trade (default: 1.0)
            confidence_threshold: Minimum confidence (|q50|/(q90-q10)) to enter trade (default: 0.3)
            base_position_size: Base position size multiplier (default: 0.1)
            model_type: Model type - "quantile" (default), "classification", or "regression"
            classification_threshold: Probability threshold for classification model (default: 0.5)
        """
        self.data_loader = MarketDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.pipeline = MultiTimeframePipeline(forward_bars=forward_bars,
                                               model_type=model_type)
        self.risk_manager = RiskManager()
        self.forward_bars = forward_bars
        self.signal_strength_threshold = signal_strength_threshold
        self.confidence_threshold = confidence_threshold
        self.base_position_size = base_position_size
        self.model_type = model_type
        self.classification_threshold = classification_threshold
        self.is_trained = False

        # Initialize strategy handler based on model type
        self._init_strategy_handler()

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

    def generate_signals(self,
                         data: Optional[pd.DataFrame] = None,
                         timeframe: str = "5T") -> pd.DataFrame:
        """
        Generate trading signals using the model architecture (single timeframe).

        Args:
            data: Optional new data for signal generation (should be single timeframe)
            timeframe: Timeframe to use for predictions (default: "5T")

        Returns:
            DataFrame with trading signals and positions
        """
        if not self.is_trained:
            raise ValueError(
                "Strategy must be trained before generating signals")

        # If no data provided, use data loader to get latest data
        if data is None:
            multi_tf_data = self.data_loader.get_multi_timeframe_data()
            engineered_data = self.feature_engineer.engineer_features(
                multi_tf_data)
            if timeframe not in engineered_data:
                raise ValueError(
                    f"Timeframe {timeframe} not found in engineered data. Available: {list(engineered_data.keys())}"
                )
            data = engineered_data[timeframe]

        # Generate predictions from models for single timeframe
        print(
            f"Generating predictions from models for timeframe {timeframe}...")

        # Prepare features
        feature_columns = [
            col for col in data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X = data[feature_columns]

        # Use strategy handler to generate signals
        signals_df = self.strategy_handler.generate_signals(X, data, timeframe)

        # Cap position size at reasonable maximum (e.g., 1.0 = 100% of portfolio)
        signals_df["position_size"] = np.clip(signals_df["position_size"], 0.0,
                                              1.0)

        # Apply risk management
        price_data = self._prepare_price_data(data, signals_df, timeframe)
        print("Applying risk management rules...")
        final_df = self.risk_manager.apply_risk_management(
            signals_df, price_data)

        return final_df

    def _init_strategy_handler(self) -> None:
        """Initialize the appropriate strategy handler based on model type."""
        if self.model_type == "classification":
            self.strategy_handler = ClassificationStrategyHandler(
                pipeline=self.pipeline,
                signal_strength_threshold=self.signal_strength_threshold,
                confidence_threshold=self.confidence_threshold,
                base_position_size=self.base_position_size,
                classification_threshold=self.classification_threshold,
            )
        else:
            self.strategy_handler = QuantileStrategyHandler(
                pipeline=self.pipeline,
                signal_strength_threshold=self.signal_strength_threshold,
                confidence_threshold=self.confidence_threshold,
                base_position_size=self.base_position_size,
            )

    def _prepare_price_data(self, data: Optional[pd.DataFrame],
                            signals_df: pd.DataFrame,
                            timeframe: str) -> pd.DataFrame:
        """
        Prepare price data for risk management.

        Args:
            data: Original data DataFrame
            signals_df: Signals DataFrame (for length/index)
            timeframe: Timeframe string

        Returns:
            DataFrame with price data
        """
        if data is not None and "close" in data.columns:
            price_data = data[["close", "open", "high", "low",
                               "volume"]].copy()
        elif self.data_loader.raw_data is not None:
            price_data = self.data_loader.raw_data.copy()
        else:
            # Create dummy price data if needed
            dates = pd.date_range("2020-01-01",
                                  periods=len(signals_df),
                                  freq=timeframe)
            price_data = pd.DataFrame(
                {
                    "close": np.ones(len(signals_df)) * 100,
                    "open": np.ones(len(signals_df)) * 100,
                    "high": np.ones(len(signals_df)) * 100,
                    "low": np.ones(len(signals_df)) * 100,
                    "volume": np.ones(len(signals_df)) * 1000,
                },
                index=dates,
            )
        return price_data

    def optimize_strategy(
            self,
            n_trials: int = 50) -> Dict[str, Dict[str, Dict[str, float]]]:
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

        # Use strategy handler to optimize models
        model_params = self.strategy_handler.optimize_models(engineered_data, n_trials)
        best_params.update(model_params)

        # Optimize volatility models (common for both model types)
        volatility_params = self._optimize_volatility_models(engineered_data, n_trials)
        best_params.update(volatility_params)

        return best_params

    def _optimize_volatility_models(
        self, engineered_data: Dict[str, pd.DataFrame], n_trials: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize volatility models (common for both quantile and classification modes).

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data
            n_trials: Number of optimization trials

        Returns:
            Dictionary of best parameters for each volatility model
        """
        best_params = {}

        # Optimize volatility models
        print("Optimizing volatility models...")
        for timeframe, data in engineered_data.items():
            # Prepare features and targets
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]
            _, volatility_target, _ = self.pipeline.prepare_targets(data)

            # Create and optimize model
            model = LightGBMModel(model_type="regression")
            best_params[f"volatility_{timeframe}"] = model.optimize_hyperparameters(
                X, volatility_target, n_trials=n_trials // 4
            )

        return best_params
