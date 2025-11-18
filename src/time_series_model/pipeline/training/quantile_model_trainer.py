"""Trainer for quantile regression models (Q10, Q50, Q90)."""

from typing import Dict, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.pipeline.training.base_model_trainer import BaseModelTrainer


class QuantileModelTrainer(BaseModelTrainer):
    """Trainer for quantile regression models (Q10, Q50, Q90)."""

    def __init__(
        self,
        use_gpu: bool = True,
        auto_tune_params: bool = False,
        tune_trials: int = 20,
    ):
        super().__init__("quantile", use_gpu, auto_tune_params, tune_trials)

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
        Train quantile models (Q10, Q50, Q90) and volatility model.

        Returns:
            Tuple of (models_dict, metrics_dict, preprocess_params_dict)
        """
        print(f"\n   🔧 Training strategy: Staged training (Q50 first, then Q10/Q90)")

        # Stage 1: Train Q50 model
        print(f"   Stage 1: Training Q50 model (primary point estimate)...")
        model_q50 = LightGBMTrainer(
            model_type="quantile",
            quantile_alpha=0.5,
            params=q50_params,
            use_gpu=self.use_gpu,
        )

        use_auto_tune = self.auto_tune_params
        q50_metrics, q50_preprocess_params = model_q50.train(
            X_df,
            y_return,
            n_splits=max(2, n_splits),
            use_time_series_cv=True,
            preprocess_fn=preprocess_fn,
            preprocess_kwargs=preprocess_kwargs or {},
            groups=groups,
            auto_tune_params=use_auto_tune,
            tune_trials=self.tune_trials,
            feature_winsorize_k=feature_winsorize_k,
        )

        # Get Q50 predictions to calculate residuals for Q10/Q90 training
        n_pred_subset = min(50000, len(X_df))
        X_pred_subset = X_df.iloc[:n_pred_subset]
        y_pred_subset = (
            y_return.iloc[:n_pred_subset]
            if isinstance(y_return, pd.Series)
            else y_return[:n_pred_subset]
        )
        q50_pred_initial = model_q50.model.predict(X_pred_subset.values)
        q50_residuals = y_pred_subset.values - q50_pred_initial

        # Calculate residual-based sample weights
        residual_median = np.median(np.abs(q50_residuals))
        if residual_median > 0:
            delta_scale = 1.0
            q10_q90_weights = 1.0 / (
                1.0 + (np.abs(q50_residuals) / (delta_scale * residual_median + 1e-8))
            )
            q10_q90_weights = q10_q90_weights / np.mean(q10_q90_weights)
        else:
            q10_q90_weights = None

        print(f"   Stage 2: Training Q10/Q90 models (guided by Q50 residuals)...")
        if q10_q90_weights is not None:
            print(
                f"      Using residual-based weights (median abs residual: {residual_median:.6f})"
            )
            print(
                f"      Weight range: [{np.min(q10_q90_weights):.4f}, {np.max(q10_q90_weights):.4f}]"
            )

        # Extend weights to full dataset if needed
        if q10_q90_weights is not None and len(q10_q90_weights) < len(X_df):
            q50_pred_full = model_q50.model.predict(X_df.values)
            q50_residuals_full = y_return.values - q50_pred_full
            residual_median_full = np.median(np.abs(q50_residuals_full))
            if residual_median_full > 0:
                delta_scale = 1.0
                q10_q90_weights_full = 1.0 / (
                    1.0
                    + (
                        np.abs(q50_residuals_full)
                        / (delta_scale * residual_median_full + 1e-8)
                    )
                )
                q10_q90_weights_full = q10_q90_weights_full / np.mean(
                    q10_q90_weights_full
                )
            else:
                q10_q90_weights_full = None
        else:
            q10_q90_weights_full = q10_q90_weights

        # Stage 2: Train Q10 and Q90
        model_q10 = LightGBMTrainer(
            model_type="quantile", quantile_alpha=0.1, use_gpu=self.use_gpu
        )
        q10_metrics, q10_preprocess_params = model_q10.train(
            X_df,
            y_return,
            n_splits=max(2, n_splits),
            use_time_series_cv=True,
            sample_weight=(
                q10_q90_weights_full if q10_q90_weights_full is not None else None
            ),
            preprocess_fn=preprocess_fn,
            preprocess_kwargs=preprocess_kwargs or {},
            feature_winsorize_k=feature_winsorize_k,
        )

        model_q90 = LightGBMTrainer(
            model_type="quantile", quantile_alpha=0.9, use_gpu=self.use_gpu
        )
        q90_metrics, q90_preprocess_params = model_q90.train(
            X_df,
            y_return,
            n_splits=max(2, n_splits),
            use_time_series_cv=True,
            sample_weight=(
                q10_q90_weights_full if q10_q90_weights_full is not None else None
            ),
            preprocess_fn=preprocess_fn,
            preprocess_kwargs=preprocess_kwargs or {},
            feature_winsorize_k=feature_winsorize_k,
        )

        # Train volatility model
        print(f"   Training volatility model...")
        model_vol = LightGBMTrainer(model_type="regression", use_gpu=self.use_gpu)
        vol_metrics, vol_preprocess_params = model_vol.train(
            X_df,
            y_vol,
            n_splits=max(2, n_splits),
            use_time_series_cv=True,
            groups=groups,
            feature_winsorize_k=feature_winsorize_k,
        )

        models_dict = {
            "q50": model_q50,
            "q10": model_q10,
            "q90": model_q90,
            "vol": model_vol,
        }

        metrics_dict = {
            "q50": q50_metrics,
            "q10": q10_metrics,
            "q90": q90_metrics,
            "vol": vol_metrics,
        }

        preprocess_params_dict = {
            "q50": q50_preprocess_params,
            "q10": q10_preprocess_params,
            "q90": q90_preprocess_params,
            "vol": vol_preprocess_params,
        }

        return models_dict, metrics_dict, preprocess_params_dict
