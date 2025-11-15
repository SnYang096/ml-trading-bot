"""Trainer for classification models."""

import logging
from typing import Dict, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.pipeline.training.base_model_trainer import BaseModelTrainer
from .label_utils import (
    log_return_magnitude,
    rolling_quantile_classification_labels,
)

logger = logging.getLogger(__name__)


class ClassificationModelTrainer(BaseModelTrainer):
    """
    Trainer for classification models with return regression and volatility prediction.
    
    This trainer implements a three-model system:
    1. Classification model: Predicts direction (up/down)
    2. Return regression model: Predicts return magnitude
    3. Volatility model: Predicts future volatility
    
    This architecture enables risk-adjusted signal scoring:
    score = p_up * expected_return / expected_volatility
    """

    def __init__(
        self,
        use_gpu: bool = True,
        auto_tune_params: bool = False,
        tune_trials: int = 20,
        classification_threshold: float = 0.001,
        use_symmetric_threshold: bool = True,
        use_vol_adjusted_threshold: bool = False,
        vol_threshold_k: float = 0.5,
        auto_tune_return: bool = False,
        auto_tune_vol: bool = False,
        use_quantile_labels: bool = True,
        quantile_window: int = 5000,
        quantile_lower: float = 0.4,
        quantile_upper: float = 0.6,
        quantile_min_periods: int = 200,
    ):
        """
        Initialize the classification model trainer.
        
        Args:
            use_gpu: Enable GPU acceleration
            auto_tune_params: Auto-tune hyperparameters for classification model
            tune_trials: Number of tuning trials for classification model
            classification_threshold: Threshold for binary classification (default: 0.001)
                - If use_symmetric_threshold=True: symmetric threshold ±threshold
                - If use_symmetric_threshold=False: single threshold (values > threshold = up)
            use_symmetric_threshold: Use symmetric threshold with neutral zone (default: True)
                - y_return > +threshold → 1 (up)
                - y_return < -threshold → 0 (down)
                - |y_return| <= threshold → filtered out (neutral)
            use_vol_adjusted_threshold: Use volatility-adjusted dynamic threshold (default: False)
                - Threshold = ±k * y_vol for each sample
                - Requires y_vol to be provided
            vol_threshold_k: Multiplier for volatility-adjusted threshold (default: 0.5)
                - Only used if use_vol_adjusted_threshold=True
            auto_tune_return: Auto-tune hyperparameters for return regression model
            auto_tune_vol: Auto-tune hyperparameters for volatility model
        """
        super().__init__("classification", use_gpu, auto_tune_params,
                         tune_trials)
        self.classification_threshold = classification_threshold
        self.use_symmetric_threshold = use_symmetric_threshold
        self.use_vol_adjusted_threshold = use_vol_adjusted_threshold
        self.vol_threshold_k = vol_threshold_k
        self.auto_tune_return = auto_tune_return
        self.auto_tune_vol = auto_tune_vol
        self.use_quantile_labels = use_quantile_labels
        self.quantile_window = quantile_window
        self.quantile_lower = quantile_lower
        self.quantile_upper = quantile_upper
        self.quantile_min_periods = quantile_min_periods

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
        Train classification, return regression, and volatility models.
        
        Args:
            X_df: Feature matrix
            y_return: Future return target (used for classification and regression)
            y_vol: Future volatility target (used for volatility regression)
            train_df: Training dataframe (used for group preparation)
            n_splits: Number of CV splits
            groups: Optional group labels for multi-asset training
            preprocess_fn: Optional preprocessing function (not used, kept for compatibility)
            preprocess_kwargs: Optional preprocessing kwargs (not used, kept for compatibility)
            q50_params: Optional Q50 parameters (not used for classification, kept for compatibility)
        
        Returns:
            Tuple of (models_dict, metrics_dict, preprocess_params_dict)
            
        Raises:
            ValueError: If target series contain NaN values or classification target has only one class
        """
        logger.info(
            "Training strategy: Classification model with return regression")

        # Validate inputs
        self._validate_targets(y_return, y_vol)

        # Calculate CV splits (ensure at least 2)
        cv_splits = max(2, n_splits)

        # Create classification labels with symmetric threshold and neutral zone filtering
        y_classification, valid_mask = self._create_classification_labels(
            y_return, y_vol)

        # Log filtering statistics
        n_total = len(y_return)
        n_filtered = n_total - valid_mask.sum()
        logger.info(
            f"Label creation: {n_total} total samples, "
            f"{n_filtered} filtered (neutral zone), "
            f"{valid_mask.sum()} valid samples ({valid_mask.sum()/n_total:.1%})"
        )

        # Validate classification target
        if y_classification.nunique() < 2:
            raise ValueError(
                f"Classification target has only one class after filtering. "
                f"Check y_return distribution and threshold settings. "
                f"Consider adjusting threshold or disabling symmetric threshold."
            )

        # Log class distribution
        positive_ratio = y_classification.mean()
        logger.info(
            f"Classification target distribution (after filtering): "
            f"{y_classification.sum()} positive ({positive_ratio:.1%}), "
            f"{(1 - y_classification).sum()} negative ({1 - positive_ratio:.1%})"
        )

        # Prepare filtered data for classification model
        # Classification model uses filtered data (only clear up/down signals)
        # Return and volatility models use full data (preserve continuous information)
        if isinstance(valid_mask, pd.Series):
            valid_mask = valid_mask.values
        else:
            valid_mask = np.asarray(valid_mask)

        # Ensure valid_mask length matches X_df length
        if len(valid_mask) != len(X_df):
            raise ValueError(
                f"valid_mask length ({len(valid_mask)}) does not match X_df length ({len(X_df)})"
            )

        X_df_filtered = X_df.loc[valid_mask] if hasattr(
            X_df.index, 'is_monotonic') else X_df[valid_mask]
        y_classification_filtered = y_classification

        # Filter groups to match filtered data
        if groups is not None:
            # Ensure groups length matches X_df length
            if len(groups) != len(X_df):
                raise ValueError(
                    f"groups length ({len(groups)}) does not match X_df length ({len(X_df)})"
                )
            groups_filtered = groups[valid_mask]
        else:
            groups_filtered = None

        # Train classification model (direction prediction) with filtered data
        logger.info("Training classification model (binary: up/down)...")
        model_classification = LightGBMTrainer(model_type="classification",
                                             use_gpu=self.use_gpu)

        classification_metrics, classification_preprocess_params = model_classification.train(
            X_df_filtered,
            y_classification_filtered,
            n_splits=cv_splits,
            use_time_series_cv=True,
            groups=groups_filtered,
            auto_tune_params=self.auto_tune_params,
            tune_trials=self.tune_trials,
            feature_winsorize_k=feature_winsorize_k)

        # Train return regression model (magnitude prediction)
        logger.info(
            "Training return regression model (for magnitude prediction)...")
        model_return = LightGBMTrainer(model_type="regression",
                                     use_gpu=self.use_gpu)
        y_return_log_mag = log_return_magnitude(y_return)
        return_metrics, return_preprocess_params = model_return.train(
            X_df,
            y_return_log_mag,
            n_splits=cv_splits,
            use_time_series_cv=True,
            groups=groups,
            auto_tune_params=self.auto_tune_return,
            tune_trials=self.tune_trials if self.auto_tune_return else None,
            feature_winsorize_k=feature_winsorize_k,
        )
        return_metrics = return_metrics or {}
        return_metrics["target_space"] = "log_magnitude"

        ret_preproc = dict(return_preprocess_params or {})
        ret_preproc["target_transform"] = "log1p_abs"

        # Train volatility model (risk prediction)
        logger.info("Training volatility model (for risk prediction)...")
        model_vol = LightGBMTrainer(model_type="regression",
                                  use_gpu=self.use_gpu)
        vol_metrics, vol_preprocess_params = model_vol.train(
            X_df,
            y_vol,
            n_splits=cv_splits,
            use_time_series_cv=True,
            groups=groups,
            auto_tune_params=self.auto_tune_vol,
            tune_trials=self.tune_trials if self.auto_tune_vol else None,
            feature_winsorize_k=feature_winsorize_k,
        )

        models_dict = {
            "classification": model_classification,
            "return": model_return,
            "vol": model_vol,
        }

        metrics_dict = {
            "classification": classification_metrics,
            "return": return_metrics,
            "vol": vol_metrics,
        }

        preprocess_params_dict = {
            "classification": classification_preprocess_params,
            "return": ret_preproc,
            "vol": vol_preprocess_params,
        }

        logger.info("All models trained successfully")
        return models_dict, metrics_dict, preprocess_params_dict

    def _validate_targets(self, y_return: pd.Series, y_vol: pd.Series) -> None:
        """
        Validate target series for training.
        
        Args:
            y_return: Future return target
            y_vol: Future volatility target
            
        Raises:
            ValueError: If targets contain NaN values or have invalid data
        """
        if y_return.isna().any():
            raise ValueError(
                f"y_return contains {y_return.isna().sum()} NaN values. "
                "Please clean the data before training.")

        if y_vol.isna().any():
            raise ValueError(
                f"y_vol contains {y_vol.isna().sum()} NaN values. "
                "Please clean the data before training.")

        if len(y_return) == 0:
            raise ValueError("y_return is empty")

        if len(y_vol) == 0:
            raise ValueError("y_vol is empty")

        if not y_return.index.equals(y_vol.index):
            raise ValueError("y_return and y_vol must have the same index. "
                             f"y_return index length: {len(y_return)}, "
                             f"y_vol index length: {len(y_vol)}")

    def _create_classification_labels(
            self, y_return: pd.Series,
            y_vol: pd.Series) -> Tuple[pd.Series, np.ndarray]:
        """
        Create symmetric binary classification labels with neutral zone filtering.
        
        This method addresses the label asymmetry problem in bidirectional trading:
        - Original: y_return > threshold → 1, else → 0 (asymmetric)
        - Improved: y_return > +threshold → 1, y_return < -threshold → 0, 
                    |y_return| <= threshold → filtered (neutral)
        
        Supports two modes:
        1. Fixed symmetric threshold: ±threshold
        2. Volatility-adjusted threshold: ±k * y_vol (dynamic per sample)
        
        Args:
            y_return: Future return target
            y_vol: Future volatility target (used for dynamic threshold)
        
        Returns:
            Tuple of (y_classification, valid_mask)
            - y_classification: Binary labels (1=up, 0=down) for valid samples only
            - valid_mask: Boolean array indicating which samples are valid (not neutral)
        """
        if self.use_quantile_labels:
            y_quantile_labels, valid_mask_array, upper, lower = rolling_quantile_classification_labels(
                y_return,
                window=self.quantile_window,
                lower_quantile=self.quantile_lower,
                upper_quantile=self.quantile_upper,
                min_periods=self.quantile_min_periods,
            )

            if y_quantile_labels.nunique() < 2:
                logger.warning(
                    "Rolling quantile labelling produced fewer than two classes; "
                    "falling back to threshold-based labelling.")
            else:
                logger.info(
                    "Using rolling quantile thresholds for classification labels "
                    f"(window={self.quantile_window}, q_low={self.quantile_lower}, "
                    f"q_high={self.quantile_upper}).")
                return y_quantile_labels, valid_mask_array

        if self.use_vol_adjusted_threshold:
            # Dynamic threshold based on volatility: ±k * y_vol
            upper_threshold = self.vol_threshold_k * y_vol
            lower_threshold = -self.vol_threshold_k * y_vol

            up_mask = y_return > upper_threshold
            down_mask = y_return < lower_threshold
            valid_mask = up_mask | down_mask

            logger.info(
                f"Using volatility-adjusted threshold (k={self.vol_threshold_k}): "
                f"threshold range [{lower_threshold.min():.4f}, {upper_threshold.max():.4f}]"
            )
        elif self.use_symmetric_threshold:
            # Fixed symmetric threshold: ±threshold
            threshold = self.classification_threshold

            up_mask = y_return > threshold
            down_mask = y_return < -threshold
            valid_mask = up_mask | down_mask

            logger.info(
                f"Using symmetric threshold ±{threshold}: "
                f"up={up_mask.sum()}, down={down_mask.sum()}, neutral={~valid_mask.sum()}"
            )
        else:
            # Original single threshold (asymmetric, for backward compatibility)
            threshold = self.classification_threshold

            up_mask = y_return > threshold
            down_mask = pd.Series(
                False, index=y_return.index)  # No explicit down mask
            valid_mask = up_mask | (y_return <= threshold)  # All samples valid

            logger.info(
                f"Using single threshold {threshold} (asymmetric mode): "
                f"up={up_mask.sum()}, down={(~up_mask).sum()}")

        # Create binary labels: 1 for up, 0 for down (only for valid samples)
        # Ensure valid_mask is a numpy array for indexing
        if isinstance(valid_mask, pd.Series):
            valid_mask_array = valid_mask.values
        else:
            valid_mask_array = np.asarray(valid_mask)

        # Get labels for valid samples only
        y_classification = up_mask[valid_mask_array].astype(int)

        return y_classification, valid_mask_array
