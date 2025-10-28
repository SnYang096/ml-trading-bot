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
    """Main ML trading strategy integrating all components."""

    def __init__(self, ensemble_method: str = "weighted"):
        """
        Initialize the ML trading strategy.

        Args:
            ensemble_method: Method to ensemble multi-timeframe predictions
                - 'weighted': Weighted voting (larger TF has more weight) - BALANCED
                - 'hierarchical': Large TF for direction, small TF for timing - CONSERVATIVE
                - 'independent': Any TF can trigger signal - AGGRESSIVE
                - 'majority': Majority voting - MODERATE
                - 'average': Original average method - MOST CONSERVATIVE
        """
        self.data_loader = MarketDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.pipeline = MultiTimeframePipeline()
        self.risk_manager = RiskManager()
        self.ensemble_method = ensemble_method
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
        engineered_data = self.feature_engineer.engineer_features(multi_tf_data)

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

    def generate_signals(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate trading signals using the trained strategy.

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

        # Generate predictions
        print("Generating stage 1 predictions (signals)...")
        stage1_preds = self.pipeline.predict_stage1(engineered_data)

        print("Generating stage 2 predictions (returns)...")
        stage2_preds = self.pipeline.predict_stage2(engineered_data)

        # Ensemble predictions
        print(
            f"Ensembling multi-timeframe predictions using '{self.ensemble_method}' method..."
        )
        ensemble_df = self.pipeline.ensemble_predictions(
            stage1_preds, stage2_preds, ensemble_method=self.ensemble_method
        )

        # Get price data for risk management
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

        # Apply risk management
        print("Applying risk management rules...")
        final_df = self.risk_manager.apply_risk_management(ensemble_df, price_data)

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

        # Optimize stage 1 models
        print("Optimizing stage 1 models...")
        for timeframe, data in engineered_data.items():
            # Prepare features and targets
            feature_columns = [
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]
            y_stage1, _ = self.pipeline.prepare_targets(data)

            # Create and optimize model
            model = self.pipeline.stage1_models.get(
                timeframe,
                MultiTimeframePipeline().stage1_models.get(
                    timeframe, LightGBMModel(model_type="classification")
                ),
            )
            best_params[f"stage1_{timeframe}"] = model.optimize_hyperparameters(
                X, y_stage1, n_trials=n_trials // 2
            )

        # Optimize stage 2 models
        print("Optimizing stage 2 models...")
        for timeframe, data in engineered_data.items():
            # Prepare features and targets
            feature_columns = [
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]
            _, y_stage2 = self.pipeline.prepare_targets(data)

            # Create and optimize model
            model = self.pipeline.stage2_models.get(
                timeframe,
                MultiTimeframePipeline().stage2_models.get(
                    timeframe, LightGBMModel(model_type="regression")
                ),
            )
            best_params[f"stage2_{timeframe}"] = model.optimize_hyperparameters(
                X, y_stage2, n_trials=n_trials // 2
            )

        return best_params


# Need to import LightGBMModel to fix the reference in optimize_strategy
from ml_trading.models.lightgbm_model import LightGBMModel
