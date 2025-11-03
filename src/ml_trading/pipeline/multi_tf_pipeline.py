"""Multi-timeframe three-stage pipeline for trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ml_trading.models.lightgbm_model import LightGBMModel
from ml_trading.config.settings import TIMEFRAMES, STAGE_1_TARGET, STAGE_2_TARGET


class MultiTimeframePipeline:
    """Multi-timeframe three-stage pipeline for trading strategy."""

    def __init__(self, forward_bars: int = 1):
        """Initialize the multi-timeframe pipeline.
        
        Args:
            forward_bars: Number of bars ahead for label prediction (default: 1)
        """
        self.stage1_models: Dict[str, LightGBMModel] = {}  # timeframe -> model
        self.stage2_models: Dict[str, LightGBMModel] = {}  # timeframe -> model
        self.is_trained = False
        self.forward_bars = forward_bars

    def prepare_targets(self,
                        data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare targets for both stages.

        Args:
            data: Market data with features

        Returns:
            Tuple of (stage1_target, stage2_target)
        """
        # Stage 1: 3-class classification target (0=Hold, 1=Long, 2=Short)
        # Unified with create_labels: if next period return > threshold, long; if < -threshold, short
        future_returns = data["close"].shift(
            -self.forward_bars) / data["close"] - 1
        threshold = 0.001  # 0.1% threshold

        stage1_target = pd.Series(0, index=data.index)  # Hold by default
        stage1_target[future_returns > threshold] = 1  # Long signal
        stage1_target[future_returns < -threshold] = 2  # Short signal

        # Stage 2: Regression target (expected return)
        stage2_target = future_returns

        return stage1_target, stage2_target

    def train_stage1(
        self,
        engineered_data: Dict[str,
                              pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Train stage 1 models (signal classification) for each timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}

        for timeframe, data in engineered_data.items():
            # Prepare features (exclude OHLCV and target columns)
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets
            y_stage1, _ = self.prepare_targets(data)

            # Create and train model
            model = LightGBMModel(model_type="classification")
            model_metrics = model.train(X, y_stage1)

            # Store model and metrics
            self.stage1_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_stage2(
        self,
        engineered_data: Dict[str,
                              pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Train stage 2 models (return regression) for each timeframe.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each timeframe
        """
        metrics = {}

        for timeframe, data in engineered_data.items():
            # Prepare features (exclude OHLCV and target columns)
            feature_columns = [
                col for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = data[feature_columns]

            # Prepare targets
            _, y_stage2 = self.prepare_targets(data)

            # Create and train model
            model = LightGBMModel(model_type="regression")
            model_metrics = model.train(X, y_stage2)

            # Store model and metrics
            self.stage2_models[timeframe] = model
            metrics[timeframe] = model_metrics

        return metrics

    def train_pipeline(
        self, engineered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Train the complete three-stage pipeline.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary of training metrics for each stage
        """
        print("Training Stage 1 models (signal classification)...")
        stage1_metrics = self.train_stage1(engineered_data)

        print("Training Stage 2 models (return regression)...")
        stage2_metrics = self.train_stage2(engineered_data)

        self.is_trained = True

        return {"stage1": stage1_metrics, "stage2": stage2_metrics}

    def predict_stage1(
            self,
            engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Make predictions using stage 1 models.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary mapping timeframe to predictions
        """
        if not self.is_trained:
            raise ValueError(
                "Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.stage1_models:
                # Prepare features
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]

                # Make predictions
                pred = self.stage1_models[timeframe].predict(X)
                predictions[timeframe] = pred

        return predictions

    def predict_stage2(
            self,
            engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Make predictions using stage 2 models.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data

        Returns:
            Dictionary mapping timeframe to predictions
        """
        if not self.is_trained:
            raise ValueError(
                "Pipeline must be trained before making predictions")

        predictions = {}
        for timeframe, data in engineered_data.items():
            if timeframe in self.stage2_models:
                # Prepare features
                feature_columns = [
                    col for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                ]
                X = data[feature_columns]

                # Make predictions
                pred = self.stage2_models[timeframe].predict(X)
                predictions[timeframe] = pred

        return predictions

    def ensemble_predictions(
        self,
        stage1_preds: Dict[str, np.ndarray],
        stage2_preds: Dict[str, np.ndarray],
        ensemble_method: str = "weighted",
    ) -> pd.DataFrame:
        """
        Ensemble predictions from multiple timeframes.

        Args:
            stage1_preds: Signal predictions from stage 1
            stage2_preds: Return predictions from stage 2
            ensemble_method: Method to ensemble predictions
                - 'weighted': Weighted voting (larger TF has more weight)
                - 'hierarchical': Large TF for direction, small TF for timing
                - 'independent': Any TF can trigger signal
                - 'majority': Majority voting without requiring all same direction
                - 'average': Original average method (most conservative)

        Returns:
            DataFrame with ensemble predictions
        """
        # Convert predictions to DataFrames for easier handling
        reference_tf = list(stage1_preds.keys())[0]
        n_samples = len(stage1_preds[reference_tf])

        # Create ensemble DataFrame
        ensemble_df = pd.DataFrame(index=range(n_samples))

        # Add predictions from each timeframe
        for tf in stage1_preds.keys():
            ensemble_df[f"signal_{tf}"] = stage1_preds[tf]
            ensemble_df[f"return_{tf}"] = stage2_preds[tf]

        signal_cols = [
            col for col in ensemble_df.columns if col.startswith("signal")
        ]
        return_cols = [
            col for col in ensemble_df.columns if col.startswith("return")
        ]

        # Apply different ensemble methods
        if ensemble_method == "weighted":
            # Weighted voting: larger timeframes have more weight
            # e.g., 15m: 0.5, 5m: 0.3, 1m: 0.2
            weights = self._get_timeframe_weights(list(stage1_preds.keys()))
            ensemble_df["ensemble_signal"] = sum(
                ensemble_df[col] * weights[i]
                for i, col in enumerate(signal_cols))
            ensemble_df["ensemble_return"] = sum(
                ensemble_df[col] * weights[i]
                for i, col in enumerate(return_cols))
            # More lenient threshold for weighted method
            ensemble_df["discrete_signal"] = 0
            ensemble_df.loc[ensemble_df["ensemble_signal"] > 0.3,
                            "discrete_signal"] = 1
            ensemble_df.loc[ensemble_df["ensemble_signal"] < -0.3,
                            "discrete_signal"] = -1

        elif ensemble_method == "hierarchical":
            # Large TF defines direction, small TF for timing
            # Sort timeframes by size (assuming format like '5T', '15T')
            sorted_tfs = sorted(stage1_preds.keys(),
                                key=lambda x: int(x.rstrip("T")),
                                reverse=True)

            # Use largest TF for direction
            largest_tf = sorted_tfs[0]
            ensemble_df["trend_direction"] = ensemble_df[
                f"signal_{largest_tf}"]

            # Use smallest TF for entry timing (only when aligned with trend)
            smallest_tf = sorted_tfs[-1]
            ensemble_df["entry_signal"] = ensemble_df[f"signal_{smallest_tf}"]

            # Only take trades when smallest TF agrees with largest TF direction
            ensemble_df["discrete_signal"] = 0
            # Long: both positive
            mask_long = (ensemble_df["trend_direction"]
                         > 0.5) & (ensemble_df["entry_signal"] > 0.5)
            ensemble_df.loc[mask_long, "discrete_signal"] = 1
            # Short: both negative
            mask_short = (ensemble_df["trend_direction"]
                          < -0.5) & (ensemble_df["entry_signal"] < -0.5)
            ensemble_df.loc[mask_short, "discrete_signal"] = -1

            # Ensemble return is weighted toward trend direction
            ensemble_df["ensemble_return"] = (
                0.7 * ensemble_df[f"return_{largest_tf}"] +
                0.3 * ensemble_df[f"return_{smallest_tf}"])

        elif ensemble_method == "independent":
            # Any timeframe can independently trigger a signal
            # Take the strongest signal among all timeframes
            ensemble_df["ensemble_signal"] = ensemble_df[signal_cols].max(
                axis=1)
            ensemble_df["min_signal"] = ensemble_df[signal_cols].min(axis=1)

            # Use the return prediction from the timeframe with strongest signal
            ensemble_df["ensemble_return"] = ensemble_df[return_cols].mean(
                axis=1)

            # Convert to discrete signals
            ensemble_df["discrete_signal"] = 0
            # Long if any timeframe strongly suggests long
            ensemble_df.loc[ensemble_df["ensemble_signal"] > 0.5,
                            "discrete_signal"] = 1
            # Short if any timeframe strongly suggests short
            ensemble_df.loc[ensemble_df["min_signal"] < -0.5,
                            "discrete_signal"] = -1

        elif ensemble_method == "majority":
            # Majority voting: take signal if more than half agree
            # Count positive, negative, and neutral signals
            ensemble_df["count_long"] = (ensemble_df[signal_cols]
                                         > 0.5).sum(axis=1)
            ensemble_df["count_short"] = (ensemble_df[signal_cols]
                                          < -0.5).sum(axis=1)
            n_timeframes = len(signal_cols)

            ensemble_df["discrete_signal"] = 0
            # Long if majority is long
            ensemble_df.loc[ensemble_df["count_long"] > n_timeframes / 2,
                            "discrete_signal"] = 1
            # Short if majority is short
            ensemble_df.loc[ensemble_df["count_short"] > n_timeframes / 2,
                            "discrete_signal"] = -1

            # Average returns
            ensemble_df["ensemble_return"] = ensemble_df[return_cols].mean(
                axis=1)
            ensemble_df["ensemble_signal"] = ensemble_df[signal_cols].mean(
                axis=1)

        else:  # 'average' - original conservative method
            ensemble_df["ensemble_signal"] = ensemble_df[signal_cols].mean(
                axis=1)
            ensemble_df["ensemble_return"] = ensemble_df[return_cols].mean(
                axis=1)

            ensemble_df["discrete_signal"] = 0
            ensemble_df.loc[ensemble_df["ensemble_signal"] > 0.1,
                            "discrete_signal"] = 1
            ensemble_df.loc[ensemble_df["ensemble_signal"] < -0.1,
                            "discrete_signal"] = -1

        return ensemble_df

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
