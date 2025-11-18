"""Base strategy handler for different model types."""

import pandas as pd
from typing import Dict
from abc import ABC, abstractmethod


class BaseStrategyHandler(ABC):
    """Base class for strategy handlers."""

    def __init__(
        self,
        pipeline,
        signal_strength_threshold: float = 1.0,
        confidence_threshold: float = 0.3,
        base_position_size: float = 0.1,
    ):
        """
        Initialize the strategy handler.

        Args:
            pipeline: MultiTimeframePipeline instance
            signal_strength_threshold: Minimum signal strength to enter trade
            confidence_threshold: Minimum confidence to enter trade
            base_position_size: Base position size multiplier
        """
        self.pipeline = pipeline
        self.signal_strength_threshold = signal_strength_threshold
        self.confidence_threshold = confidence_threshold
        self.base_position_size = base_position_size

    @abstractmethod
    def generate_signals(
        self, X: pd.DataFrame, data: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Generate trading signals using the model architecture.

        Args:
            X: Feature matrix
            data: Original data DataFrame (for index)
            timeframe: Timeframe string

        Returns:
            DataFrame with trading signals and positions
        """
        pass

    @abstractmethod
    def optimize_models(
        self, engineered_data: Dict[str, pd.DataFrame], n_trials: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize models for this strategy type.

        Args:
            engineered_data: Dictionary mapping timeframe to engineered data
            n_trials: Number of optimization trials

        Returns:
            Dictionary of best parameters for each model
        """
        pass
