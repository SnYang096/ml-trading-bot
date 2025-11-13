"""Model training utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import lightgbm as lgb


def train_production_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict | None = None,
):
    """Train production LightGBM model with automatic task detection."""
    print("🌲 Training production LightGBM...")

    # Basic validation
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite values detected in features (NaN/inf)")
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ValueError("Non-finite values detected in labels (NaN/inf)")

    # Check for valid labels (for both classification and regression)
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 1:
        raise ValueError(
            f"y_train has only one unique value ({unique_labels[0]}); cannot train a model"
        )

    # Auto-detect task type based on labels
    # NOTE: data_loader.py now generates binary labels (1=Long, 0=Short) directly
    # Regression tasks (predicting returns) remain as regression - DO NOT CHANGE
    unique_labels = np.unique(y_train)
    num_unique = len(unique_labels)

    # Determine objective based on label characteristics
    # If labels are binary (0, 1) → binary classification
    # If labels are continuous values → regression (return prediction)
    if num_unique == 2 and np.all(np.isin(unique_labels, [0, 1])):
        # Binary classification (1=Long, 0=Short)
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
        print("   Using binary classification (1=Long, 0=Short)")
    else:
        # Regression for predicting continuous returns (DO NOT CHANGE - this is correct for return prediction)
        objective = "regression"
        metric = "rmse"
        task_params = {}
        print(f"   Using regression for return prediction")

    if params is None:
        params = {
            "objective": objective,
            "metric": metric,
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
            # Prefer CUDA backend if available (LightGBM built with CUDA)
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            **task_params,  # Add num_class for multiclass
        }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # Use callbacks for broad LightGBM version compatibility
    callbacks = [
        lgb.early_stopping(stopping_rounds=400, verbose=True),
        lgb.log_evaluation(period=200),
    ]
    try:
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=4000,
            valid_sets=[lgb_val],
            valid_names=["valid"],
            callbacks=callbacks,
        )
    except Exception as gpu_err:
        # Fallback: if GPU init fails (e.g., OpenCL/CUDA not available), retry on CPU
        print(f"⚠️ LightGBM GPU failed ({gpu_err}), retrying on CPU...")
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=4000,
            valid_sets=[lgb_val],
            valid_names=["valid"],
            callbacks=callbacks,
        )

    # Ensure best_iteration attribute is present
    if getattr(model, "best_iteration", None) in (None, 0):
        # fallback to number of trees if early stopping not triggered
        model.best_iteration = model.current_iteration()

    print(
        f"✅ Production LightGBM training complete (best_iteration={model.best_iteration})"
    )
    return model
