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

    # Diagnostic: Check if features can perfectly predict labels
    # Check for constant features
    constant_feat_indices = []
    for i in range(X_train.shape[1]):
        if np.std(X_train[:, i]) < 1e-10:
            constant_feat_indices.append(i)
    if constant_feat_indices:
        print(
            f"   ⚠️  WARNING: Found {len(constant_feat_indices)} constant features in training set!"
        )

    # Check feature-label correlation (first 10 features)
    from scipy.stats import pointbiserialr
    print(f"\n   [DEBUG] Feature-label correlation (first 10 features):")
    max_corr = 0.0
    max_corr_feat = None
    for i in range(min(10, X_train.shape[1])):
        try:
            corr, pval = pointbiserialr(X_train[:, i], y_train)
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_feat = i
            print(f"      Feature {i}: corr={corr:.4f}, p={pval:.2e}")
        except:
            pass

    if max_corr_feat is not None:
        print(
            f"   [DEBUG] Best feature correlation: Feature {max_corr_feat}, corr={max_corr:.4f}"
        )
        if abs(max_corr) < 0.1:
            print(
                f"   ⚠️  WARNING: Very weak feature-label correlation! Features may not be informative."
            )

    # Check feature distribution overlap between classes
    print(f"\n   [DEBUG] Feature distribution overlap (first 5 features):")
    for i in range(min(5, X_train.shape[1])):
        feat_0 = X_train[y_train == 0, i]
        feat_1 = X_train[y_train == 1, i]
        if len(feat_0) > 0 and len(feat_1) > 0:
            mean_0, std_0 = feat_0.mean(), feat_0.std()
            mean_1, std_1 = feat_1.mean(), feat_1.std()
            # Check overlap: if means are within 2 std of each other, there's significant overlap
            overlap = abs(mean_0 - mean_1) < 2 * (std_0 + std_1)
            print(
                f"      Feature {i}: Class 0 (mean={mean_0:.4f}, std={std_0:.4f}), Class 1 (mean={mean_1:.4f}, std={std_1:.4f}), Overlap={'Yes' if overlap else 'No'}"
            )

    # Check label distribution
    train_label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    val_label_dist = dict(zip(*np.unique(y_val, return_counts=True)))
    print(
        f"   [DEBUG] Train labels: {train_label_dist}, Val labels: {val_label_dist}"
    )

    # Calculate class weights to handle imbalance
    # LightGBM uses scale_pos_weight = negative_count / positive_count
    positive_count = (y_train == 1).sum()
    negative_count = (y_train == 0).sum()
    if positive_count > 0 and negative_count > 0:
        scale_pos_weight = negative_count / positive_count
        print(
            f"   [DEBUG] Class imbalance: positive={positive_count}, negative={negative_count}, scale_pos_weight={scale_pos_weight:.4f}"
        )
    else:
        scale_pos_weight = 1.0

    # Check if train and val are identical (would cause perfect fit)
    if X_train.shape == X_val.shape and np.allclose(X_train, X_val,
                                                    rtol=1e-10):
        print(f"   ⚠️  WARNING: Train and validation features are identical!")
    if len(y_train) == len(y_val) and np.array_equal(y_train, y_val):
        print(f"   ⚠️  WARNING: Train and validation labels are identical!")

    # Production defaults with anti-degradation parameters
    # These parameters prevent the model from degenerating to constant predictions
    # Reference: LightGBM best practices for binary classification with imbalanced data

    # Adaptive min_data_in_leaf based on dataset size
    # Rule: at least 20, but scale with dataset size (1% of samples, min 10, max 50)
    adaptive_min_data = max(10, min(50, int(len(y_train) * 0.01)))
    adaptive_min_data = max(adaptive_min_data, 20)  # Ensure at least 20

    production_params = {
        "objective": "binary",  # Explicitly set binary classification
        "metric": ["auc", "binary_logloss"
                   ],  # Use AUC for imbalanced data, also monitor logloss
        "boosting_type": "gbdt",
        "num_leaves": 31,  # Limit tree complexity (2^5-1, depth ~5)
        "learning_rate": 0.02,  # Small learning rate for stable convergence
        "feature_fraction":
        0.8,  # Random feature selection (80%) - prevents overfitting
        "bagging_fraction":
        0.9,  # Random data sampling (90%) - prevents overfitting
        "bagging_freq": 5,  # Bagging every 5 iterations
        "min_data_in_leaf":
        adaptive_min_data,  # Adaptive: prevents overfitting noise
        "min_sum_hessian_in_leaf":
        1e-3,  # Minimum sum of hessian (second-order gradient)
        "min_split_gain":
        0.0,  # Allow splits even with minimal gain (for small datasets)
        "lambda_l1": 0.5,  # L1 regularization - encourages sparsity
        "lambda_l2":
        1.0,  # L2 regularization - smooths predictions, prevents extreme outputs
        "scale_pos_weight":
        scale_pos_weight,  # Handle class imbalance (manual control)
        "max_depth": -1,  # No explicit depth limit (controlled by num_leaves)
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,  # Column-wise histogram for better performance
    }

    print(f"   [DEBUG] Anti-degradation parameters:")
    print(
        f"      min_data_in_leaf: {adaptive_min_data} (adaptive, dataset size: {len(y_train)})"
    )
    print(
        f"      scale_pos_weight: {scale_pos_weight:.4f} (class imbalance handling)"
    )
    print(
        f"      learning_rate: {production_params['learning_rate']} (stable convergence)"
    )
    print(
        f"      lambda_l1: {production_params['lambda_l1']}, lambda_l2: {production_params['lambda_l2']} (regularization)"
    )

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

    # Diagnostic: Check actual predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Check if predictions are perfect
    if len(y_pred_train) == len(y_train):
        train_acc = (np.round(y_pred_train) == y_train).mean()
        val_acc = (np.round(y_pred_val) == y_val).mean()
        print(
            f"   [DEBUG] Train accuracy: {train_acc:.6f}, Val accuracy: {val_acc:.6f}"
        )

        # Check prediction distribution (probabilities)
        print(
            f"   [DEBUG] Train prediction probs: min={y_pred_train.min():.6f}, max={y_pred_train.max():.6f}, mean={y_pred_train.mean():.6f}"
        )
        print(
            f"   [DEBUG] Val prediction probs: min={y_pred_val.min():.6f}, max={y_pred_val.max():.6f}, mean={y_pred_val.mean():.6f}"
        )

        # Check prediction distribution (classes)
        train_pred_dist = dict(
            zip(*np.unique(np.round(y_pred_train), return_counts=True)))
        val_pred_dist = dict(
            zip(*np.unique(np.round(y_pred_val), return_counts=True)))
        print(
            f"   [DEBUG] Train predictions (classes): {train_pred_dist}, Val predictions (classes): {val_pred_dist}"
        )

        # Check if predictions are constant
        if len(np.unique(np.round(y_pred_train))) == 1:
            print(
                f"   ⚠️  WARNING: Model predicts only one class in training set!"
            )
            print(
                f"      This means the model learned to always predict the majority class."
            )
            print(f"      Possible causes:")
            print(
                f"      1. Features don't contain enough information to distinguish classes"
            )
            print(
                f"      2. min_data_in_leaf is too high (current: {production_params.get('min_data_in_leaf', 'N/A')})"
            )
            print(f"      3. Class imbalance is too severe")

        if len(np.unique(np.round(y_pred_val))) == 1:
            print(
                f"   ⚠️  WARNING: Model predicts only one class in validation set!"
            )

        # Calculate actual logloss to verify
        from sklearn.metrics import log_loss
        try:
            train_logloss = log_loss(y_train, y_pred_train)
            val_logloss = log_loss(y_val, y_pred_val)
            print(
                f"   [DEBUG] Actual logloss: Train={train_logloss:.6e}, Val={val_logloss:.6e}"
            )
            if train_logloss < 1e-10:
                print(
                    f"   ⚠️  WARNING: Train logloss is extremely small! This confirms perfect fit."
                )
                print(
                    f"      This happens when model predicts probability ≈ 1.0 for all samples."
                )
        except:
            pass

        # Check if predictions match labels perfectly
        if train_acc > 0.9999:
            print(
                f"   ⚠️  WARNING: Near-perfect training accuracy ({train_acc:.6f})!"
            )
        if val_acc > 0.9999:
            print(
                f"   ⚠️  WARNING: Near-perfect validation accuracy ({val_acc:.6f})!"
            )

    print(
        f"✅ Production LightGBM training complete (best_iteration={model.best_iteration})"
    )
    return model
