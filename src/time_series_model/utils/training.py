"""Training utilities shared across dimensionality workflows."""

from __future__ import annotations

from typing import Any, Dict

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np
import pandas as pd


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_gpu: bool = True,
    num_boost_round: int = 200,
    params: Dict[str, Any] | None = None,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    eval_period: int | None = 50,
    categorical_feature: Any | None = None,
) -> lgb.Booster:
    """Train a LightGBM model with optional validation support.
    
    Automatically detects whether to use binary classification or regression based on y_train:
    - If labels are integers in [0, 2] (3-class: 0=Hold, 1=Long, 2=Short): 
      Filters neutral labels (0=Hold) and converts to binary (1=Long, 0=Short)
    - If labels are already binary (0/1): uses binary classification
    - Otherwise: regression (for continuous return prediction)
    
    Note: Classification always uses binary (1=Long, 0=Short), neutral labels are filtered globally.
    """

    # Auto-detect task type based on labels
    unique_labels = np.unique(y_train)
    num_unique = len(unique_labels)

    # Check if labels are 3-class format (0=Hold, 1=Long, 2=Short)
    is_3class = (num_unique <= 3
                 and np.all(np.equal(np.mod(unique_labels, 1), 0))
                 and np.all(unique_labels >= 0) and np.all(unique_labels <= 2))

    # Filter neutral labels (0=Hold) and convert to binary (1=Long, 0=Short)
    if is_3class:
        # Filter out neutral labels (0=Hold), keep only Long (1) and Short (2)
        train_mask = (y_train == 1) | (y_train == 2)
        if train_mask.sum() == 0:
            raise ValueError(
                "No valid long/short samples in training set after removing neutral labels."
            )
        X_train = X_train[train_mask]
        y_train = np.where(y_train[train_mask] == 1, 1, 0).astype(int)

        # Also filter validation set if provided
        if X_val is not None and y_val is not None:
            val_mask = (y_val == 1) | (y_val == 2)
            if val_mask.sum() == 0:
                raise ValueError(
                    "No valid long/short samples in validation set after removing neutral labels."
                )
            X_val = X_val[val_mask]
            y_val = np.where(y_val[val_mask] == 1, 1, 0).astype(int)

        # Use binary classification
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
    elif num_unique == 2:
        # Already binary classification
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
    else:
        # Regression for predicting continuous returns
        objective = "regression"
        metric = "rmse"
        task_params = {}

    default_params = {
        "objective": objective,
        "metric": metric,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "force_col_wise": True,
        **task_params,  # Add num_class for multiclass
    }

    if params:
        default_params.update(params)

    if use_gpu:
        default_params.update({
            "device": "cuda",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        })

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            categorical_feature=categorical_feature,
            free_raw_data=False,
        )
        valid_sets.append(val_data)
        valid_names.append("valid")

    callbacks = []
    if eval_period is not None:
        callbacks.append(lgb.log_evaluation(period=eval_period))

    if early_stopping_rounds is not None and len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))

    try:
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    except LightGBMError as exc:
        if use_gpu and "CUDA" in str(exc).upper():
            print("⚠️  Falling back to CPU-based LightGBM training")
            return train_lightgbm_model(
                X_train,
                y_train,
                use_gpu=False,
                num_boost_round=num_boost_round,
                params=params,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_period=eval_period,
                categorical_feature=categorical_feature,
            )
        raise

    return model


__all__ = [
    "train_lightgbm_model",
]
