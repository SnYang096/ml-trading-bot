"""Model training utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch

from time_series_model.utils.training import train_lightgbm_model


def train_production_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict | None = None,
):
    """
    Train production LightGBM model with automatic task detection.
    
    This function wraps train_lightgbm_model with production-specific defaults:
    - Uses production hyperparameters (lower learning rate, more rounds)
    - Requires validation set (not optional)
    
    Note: Neutral label filtering (0=Hold) is handled automatically by train_lightgbm_model.
    """
    print("🌲 Training production LightGBM...")

    # Basic validation
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite values detected in features (NaN/inf)")
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ValueError("Non-finite values detected in labels (NaN/inf)")

    # Check for valid labels
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 1:
        raise ValueError(
            f"y_train has only one unique value ({unique_labels[0]}); cannot train a model"
        )

    # Production defaults (override user params if provided)
    production_params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 1e-3,
        "min_split_gain": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "random_state": 42,
    }

    # Map device_type to device (for compatibility with train_lightgbm_model)
    if torch.cuda.is_available():
        production_params["device"] = "cuda"
        production_params["gpu_platform_id"] = 0
        production_params["gpu_device_id"] = 0
    else:
        production_params["device"] = "cpu"

    # Merge with user-provided params
    if params:
        production_params.update(params)

    # Use train_lightgbm_model with production defaults
    model = train_lightgbm_model(
        X_train,
        y_train,
        use_gpu=torch.cuda.is_available(),
        num_boost_round=4000,
        params=production_params,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=400,
        eval_period=200,
    )

    # Ensure best_iteration attribute is present
    if getattr(model, "best_iteration", None) in (None, 0):
        # fallback to number of trees if early stopping not triggered
        model.best_iteration = model.current_iteration()

    print(
        f"✅ Production LightGBM training complete (best_iteration={model.best_iteration})"
    )
    return model
