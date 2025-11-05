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
        self, data: Optional[pd.DataFrame] = None, timeframe: str = "5T"
    ) -> pd.DataFrame:
        """
        Generate trading signals using the four-model architecture (single timeframe).

        Decision Logic:
        - signal_strength = q50_pred / pred_vol (收益风险比)
        - confidence = |q50_pred| / (q90 - q10) (模型信心)
        - Enter trade if signal_strength > k1 AND confidence > k2
        - Position size = base_size * signal_strength * confidence

        Args:
            data: Optional new data for signal generation (should be single timeframe)
            timeframe: Timeframe to use for predictions (default: "5T")

        Returns:
            DataFrame with trading signals and positions
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")

        # If no data provided, use data loader to get latest data
        if data is None:
            multi_tf_data = self.data_loader.get_multi_timeframe_data()
            engineered_data = self.feature_engineer.engineer_features(multi_tf_data)
            if timeframe not in engineered_data:
                raise ValueError(f"Timeframe {timeframe} not found in engineered data. Available: {list(engineered_data.keys())}")
            data = engineered_data[timeframe]

        # Generate predictions from all four models for single timeframe
        print(f"Generating predictions from all models for timeframe {timeframe}...")
        
        # Prepare features
        feature_columns = [
            col for col in data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X = data[feature_columns]

        # Get predictions from all four models
        if timeframe not in self.pipeline.q50_models:
            raise ValueError(f"Model for timeframe {timeframe} not found. Available: {list(self.pipeline.q50_models.keys())}")
        
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
            index=data.index if hasattr(data, 'index') else range(len(q50_pred)),
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
            (signals_df["q50"] > 0) &
            (signals_df["signal_strength"] > self.signal_strength_threshold) &
            (signals_df["confidence"] > self.confidence_threshold)
        )
        signals_df.loc[long_mask, "signal"] = 1

        # Short signal: negative q50, high signal strength (absolute), high confidence
        short_mask = (
            (signals_df["q50"] < 0) &
            (np.abs(signals_df["signal_strength"]) > self.signal_strength_threshold) &
            (signals_df["confidence"] > self.confidence_threshold)
        )
        signals_df.loc[short_mask, "signal"] = -1

        # Position sizing: base_size * signal_strength * confidence
        signals_df["position_size"] = 0.0
        trade_mask = signals_df["signal"] != 0
        signals_df.loc[trade_mask, "position_size"] = (
            self.base_position_size *
            np.abs(signals_df.loc[trade_mask, "signal_strength"]) *
            signals_df.loc[trade_mask, "confidence"]
        )
        
        # Cap position size at reasonable maximum (e.g., 1.0 = 100% of portfolio)
        signals_df["position_size"] = np.clip(signals_df["position_size"], 0.0, 1.0)

        # Apply risk management
        if data is not None and "close" in data.columns:
            price_data = data[["close", "open", "high", "low", "volume"]].copy()
        elif self.data_loader.raw_data is not None:
            price_data = self.data_loader.raw_data.copy()
        else:
            # Create dummy price data if needed
            dates = pd.date_range("2020-01-01", periods=len(signals_df), freq=timeframe)
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

        print("Applying risk management rules...")
        final_df = self.risk_manager.apply_risk_management(signals_df, price_data)

        return final_df


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
