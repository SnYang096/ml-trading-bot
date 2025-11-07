"""Base class for model trainers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd


class BaseModelTrainer(ABC):
    """Base class for model trainers."""

    def __init__(
        self,
        model_type: str,
        use_gpu: bool = True,
        auto_tune_params: bool = False,
        tune_trials: int = 20,
    ):
        """
        Initialize the base model trainer.

        Args:
            model_type: Model type ("quantile", "classification", "regression")
            use_gpu: Enable GPU acceleration
            auto_tune_params: Auto-tune hyperparameters
            tune_trials: Number of tuning trials
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.auto_tune_params = auto_tune_params
        self.tune_trials = tune_trials

    @abstractmethod
    def train_models(
        self,
        X_df: pd.DataFrame,
        y_return: pd.Series,
        y_vol: pd.Series,
        train_df: pd.DataFrame,
        n_splits: int,
        groups: Optional[np.ndarray],
        preprocess_fn: Optional[Callable] = None,
        preprocess_kwargs: Optional[Dict] = None,
        q50_params: Optional[Dict] = None,
        feature_winsorize_k: float = 4.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Train models for the given model type.

        Args:
            X_df: Feature matrix
            y_return: Return target
            y_vol: Volatility target
            train_df: Training dataframe
            n_splits: Number of CV splits
            groups: Optional group labels
            preprocess_fn: Optional preprocessing function
            preprocess_kwargs: Optional preprocessing kwargs
            q50_params: Optional Q50 parameters (for quantile models)
            feature_winsorize_k: Feature-level winsorize multiplier (<=0 disables)

        Returns:
            Tuple of (models_dict, metrics_dict, preprocess_params_dict)
        """
        pass

    def _prepare_groups(
        self,
        train_df: pd.DataFrame,
        y_return: pd.Series,
        X_df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Prepare groups array for multi-asset training."""
        groups = None
        if "symbol" in train_df.columns and len(train_df["symbol"].unique()) > 1:
            symbol_to_group = {
                symbol: idx
                for idx, symbol in enumerate(train_df["symbol"].unique())
            }
            if not y_return.index.isin(train_df.index).all():
                y_return = y_return.loc[y_return.index.isin(train_df.index)]

            if not X_df.index.equals(y_return.index):
                common_idx = X_df.index.intersection(y_return.index)
                X_df = X_df.loc[common_idx]
                y_return = y_return.loc[common_idx]

            groups = train_df.loc[y_return.index, "symbol"].map(symbol_to_group).values
        return groups

