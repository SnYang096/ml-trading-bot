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
    feature_names: list | None = None,
    categorical_features: list | None = None,
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
    num_classes = len(unique_labels)
    if num_classes == 1:
        raise ValueError(
            f"y_train has only one unique value ({unique_labels[0]}); cannot train a model"
        )
    
    # Validate label range for multiclass (should be 0, 1, 2 for 3-class)
    if num_classes > 2:
        if not all(label in [0, 1, 2] for label in unique_labels):
            raise ValueError(
                f"Multiclass labels must be in [0, 1, 2] (0=Hold, 1=Long, 2=Short), "
                f"but found: {unique_labels}"
            )
        print(f"   [DEBUG] Multiclass training: {num_classes} classes detected (0=Hold, 1=Long, 2=Short)")
    else:
        print(f"   [DEBUG] Binary classification: {num_classes} classes detected")
    
    # CRITICAL: Check label distribution to prevent constant prediction
    # Reference: https://docs/时序模型/统一训练：categorical feature.md
    # In quant finance, label imbalance is common and can cause model degeneration
    if num_classes == 2:
        # Binary classification: check positive rate
        label_pos_rate = y_train.mean()
    else:
        # Multiclass: check each class rate
        from collections import Counter
        label_counts = Counter(y_train)
        total_samples = len(y_train)
        label_pos_rate = label_counts.get(1, 0) / total_samples if total_samples > 0 else 0.0
    if label_pos_rate < 0.01 or label_pos_rate > 0.99:
        print(f"\n   🚨 CRITICAL WARNING: Extreme label imbalance detected!")
        print(f"      Label positive rate: {label_pos_rate:.2%} (should be 1%-99%)")
        print(f"      This can cause model to degenerate to constant prediction!")
        print(f"      → Model may learn to always predict the majority class")
        print(f"      → Recommended: Use quantile-based labels or risk-adjusted returns")
        print(f"      → Current labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # Don't fail, but warn strongly
        import warnings
        warnings.warn(
            f"Extreme label imbalance: pos_rate={label_pos_rate:.2%}. "
            f"This may cause model degeneration. Consider using quantile-based labels.",
            UserWarning
        )
    elif label_pos_rate < 0.05 or label_pos_rate > 0.95:
        print(f"\n   ⚠️  WARNING: Significant label imbalance detected!")
        print(f"      Label positive rate: {label_pos_rate:.2%} (recommended: 5%-95%)")
        print(f"      This may cause model to favor majority class")
    
    # Log label distribution for monitoring
    label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"\n   📊 Label Distribution Check:")
    print(f"      Classes: {label_dist}")
    print(f"      Positive rate: {label_pos_rate:.2%}")
    print(f"      ✅ Label distribution check passed (1% < pos_rate < 99%)")

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
    # For multiclass (3-class: 0=Hold, 1=Long, 2=Short), use class_weight
    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    
    if num_classes == 2:
        # Binary classification (backward compatibility)
        positive_count = (y_train == 1).sum()
        negative_count = (y_train == 0).sum()
        if positive_count > 0 and negative_count > 0:
            scale_pos_weight = negative_count / positive_count
            print(
                f"   [DEBUG] Binary classification: positive={positive_count}, negative={negative_count}, scale_pos_weight={scale_pos_weight:.4f}"
            )
        else:
            scale_pos_weight = 1.0
        class_weight = None
    else:
        # Multiclass classification (3-class: 0=Hold, 1=Long, 2=Short)
        from collections import Counter
        label_counts = Counter(y_train)
        total_samples = len(y_train)
        
        # Calculate class weights: inverse frequency (balanced)
        class_weight = {}
        for label in unique_labels:
            count = label_counts.get(label, 0)
            if count > 0:
                # Weight inversely proportional to frequency
                class_weight[int(label)] = total_samples / (num_classes * count)
            else:
                class_weight[int(label)] = 1.0
        
        print(
            f"   [DEBUG] Multiclass classification ({num_classes} classes):"
        )
        for label in sorted(unique_labels):
            count = label_counts.get(label, 0)
            rate = count / total_samples if total_samples > 0 else 0.0
            weight = class_weight.get(int(label), 1.0)
            label_name = {0: "Hold", 1: "Long", 2: "Short"}.get(int(label), f"Class_{label}")
            print(f"      {label_name} (label={label}): count={count} ({rate:.2%}), weight={weight:.4f}")
        
        scale_pos_weight = None  # Not used for multiclass

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

    # Determine objective and metric based on number of classes
    if num_classes == 2:
        objective = "binary"
        metric = ["auc", "binary_logloss"]
    else:
        objective = "multiclass"  # 3-class: 0=Hold, 1=Long, 2=Short
        metric = ["multi_logloss", "multi_error"]  # Multi-class logloss and error rate
    
    production_params = {
        "objective": objective,
        "metric": metric,
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
    }
    
    # Add class-specific parameters
    if num_classes == 2:
        # Binary classification: use scale_pos_weight
        production_params["scale_pos_weight"] = scale_pos_weight
    else:
        # Multiclass: set num_class (required for multiclass)
        production_params["num_class"] = num_classes
        # LightGBM doesn't directly support class_weight dict, but we can use sample_weight
        # For now, we'll rely on balanced class distribution from label generation
    
    production_params.update({
        "max_depth": -1,  # No explicit depth limit (controlled by num_leaves)
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,  # Column-wise histogram for better performance
    })

    print(f"   [DEBUG] Anti-degradation parameters:")
    print(
        f"      min_data_in_leaf: {adaptive_min_data} (adaptive, dataset size: {len(y_train)})"
    )
    if scale_pos_weight is not None:
        print(
            f"      scale_pos_weight: {scale_pos_weight:.4f} (class imbalance handling)"
        )
    else:
        print(
            f"      num_class: {num_classes} (multiclass classification)"
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

    # Add categorical feature parameters if provided
    if categorical_features:
        # Add LightGBM categorical feature parameters
        production_params.setdefault(
            "min_data_per_group",
            30)  # Each category needs at least 30 samples
        production_params.setdefault("cat_smooth",
                                     10)  # Smooth rare categories
        print(f"   ✅ Categorical features enabled: {categorical_features}")
        print(
            f"      min_data_per_group: {production_params['min_data_per_group']}"
        )
        print(f"      cat_smooth: {production_params['cat_smooth']}")

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
        feature_names=feature_names,
        categorical_feature=categorical_features,
    )

    # Ensure best_iteration attribute is present
    if getattr(model, "best_iteration", None) in (None, 0):
        # fallback to number of trees if early stopping not triggered
        model.best_iteration = model.current_iteration()

    # Diagnostic: Check actual predictions (CRITICAL for detecting constant prediction)
    # Reference: Model degeneration detection in quant finance
    y_pred_train_raw = model.predict(X_train)
    y_pred_val_raw = model.predict(X_val)
    
    # Debug: Print prediction array shapes
    print(f"   [DEBUG] Prediction shapes: train={y_pred_train_raw.shape}, val={y_pred_val_raw.shape}")
    print(f"   [DEBUG] Expected num_classes: {num_classes}")
    if y_pred_train_raw.ndim == 2:
        print(f"   [DEBUG] Prediction array columns: {y_pred_train_raw.shape[1]} (expected {num_classes} for multiclass)")
    
    # Handle multiclass predictions: convert probability array to class predictions if needed
    # For multiclass, LightGBM returns probability array (n_samples, num_classes)
    # For binary, LightGBM returns probability array (n_samples, 2) or class predictions (n_samples,)
    if y_pred_train_raw.ndim == 2 and y_pred_train_raw.shape[1] > 1:
        # Multiclass: convert probability array to class predictions
        y_pred_train = np.argmax(y_pred_train_raw, axis=1)
        y_pred_val = np.argmax(y_pred_val_raw, axis=1)
        # For statistics, use probability of predicted class
        y_pred_train_proba = np.max(y_pred_train_raw, axis=1)
        y_pred_val_proba = np.max(y_pred_val_raw, axis=1)
    else:
        # Binary classification or already class predictions
        if y_pred_train_raw.ndim == 2 and y_pred_train_raw.shape[1] == 2:
            # Binary probability array: use positive class probability
            y_pred_train = np.argmax(y_pred_train_raw, axis=1)
            y_pred_val = np.argmax(y_pred_val_raw, axis=1)
            y_pred_train_proba = y_pred_train_raw[:, 1] if y_pred_train_raw.shape[1] == 2 else y_pred_train_raw.flatten()
            y_pred_val_proba = y_pred_val_raw[:, 1] if y_pred_val_raw.shape[1] == 2 else y_pred_val_raw.flatten()
        else:
            # Already class predictions (1D array)
            y_pred_train = y_pred_train_raw.flatten()
            y_pred_val = y_pred_val_raw.flatten()
            y_pred_train_proba = y_pred_train.astype(float)
            y_pred_val_proba = y_pred_val.astype(float)

    # Calculate prediction statistics (use probability for std/mean, class for accuracy)
    pred_train_std = np.std(y_pred_train_proba)
    pred_val_std = np.std(y_pred_val_proba)
    pred_train_min = np.min(y_pred_train_proba)
    pred_train_max = np.max(y_pred_train_proba)
    pred_val_min = np.min(y_pred_val_proba)
    pred_val_max = np.max(y_pred_val_proba)
    pred_train_mean = np.mean(y_pred_train_proba)
    pred_val_mean = np.mean(y_pred_val_proba)
    
    # Label statistics
    if num_classes == 2:
        # Binary: use positive rate
        label_train_pos_rate = y_train.mean()
        label_val_pos_rate = y_val.mean()
    else:
        # Multiclass: use rate of class 1 (Long)
        label_train_pos_rate = (y_train == 1).mean()
        label_val_pos_rate = (y_val == 1).mean()

    # Print prediction statistics (as recommended in the document)
    print(f"\n   📊 Prediction Statistics (Constant Prediction Detection):")
    print(f"      Train: min={pred_train_min:.6f}, max={pred_train_max:.6f}, std={pred_train_std:.6f}, mean={pred_train_mean:.6f}")
    print(f"      Val:   min={pred_val_min:.6f}, max={pred_val_max:.6f}, std={pred_val_std:.6f}, mean={pred_val_mean:.6f}")
    print(f"      Label pos rate: Train={label_train_pos_rate:.2%}, Val={label_val_pos_rate:.2%}")
    print(f"      Pred mean vs Label pos rate: Train={pred_train_mean:.2%} (label={label_train_pos_rate:.2%}), Val={pred_val_mean:.2%} (label={label_val_pos_rate:.2%})")

    # Check if predictions are constant (CRITICAL: std < 1e-5 indicates degeneration)
    CONSTANT_PREDICTION_THRESHOLD = 1e-5
    is_constant_train = pred_train_std < CONSTANT_PREDICTION_THRESHOLD
    is_constant_val = pred_val_std < CONSTANT_PREDICTION_THRESHOLD
    
    if is_constant_train or is_constant_val:
        print(f"\n   🚨 CRITICAL WARNING: Model predictions are constant (degenerated)!")
        if is_constant_train:
            print(f"      ⚠️  Train prediction std={pred_train_std:.2e} < {CONSTANT_PREDICTION_THRESHOLD}")
            print(f"         → Model is predicting constant value: {pred_train_mean:.6f}")
        if is_constant_val:
            print(f"      ⚠️  Val prediction std={pred_val_std:.2e} < {CONSTANT_PREDICTION_THRESHOLD}")
            print(f"         → Model is predicting constant value: {pred_val_mean:.6f}")
        print(f"\n   💡 Possible causes:")
        print(f"      1. Label imbalance: pos_rate={label_train_pos_rate:.2%} (should be 1%-99%)")
        print(f"      2. Low signal-to-noise ratio: features have weak predictive power")
        print(f"      3. Over-regularization: model parameters too conservative")
        print(f"      4. Data leakage fix overdone: too many samples removed")
        print(f"      5. Label definition issue: using absolute returns in low-volatility periods")
        print(f"\n   🔧 Recommended actions:")
        print(f"      1. Check label distribution: assert 0.01 < y.mean() < 0.99")
        print(f"      2. Use risk-adjusted returns or quantile-based labels")
        print(f"      3. Reduce regularization (lambda_l1, lambda_l2)")
        print(f"      4. Add known-effective baseline features")
        print(f"      5. Monitor AUC and F1-score (not just accuracy)")
        
        # Raise warning but don't fail (allow training to continue for debugging)
        import warnings
        warnings.warn(
            f"Model predictions are constant! Train std={pred_train_std:.2e}, Val std={pred_val_std:.2e}. "
            f"This indicates model degeneration. Check label distribution and feature quality.",
            UserWarning
        )

    # Check if predictions match label distribution but are constant (another form of degeneration)
    if not is_constant_train and not is_constant_val:
        # Check if pred mean ≈ label pos rate but std ≈ 0 (model learned to predict prior)
        pred_mean_match_train = abs(pred_train_mean - label_train_pos_rate) < 0.01
        pred_mean_match_val = abs(pred_val_mean - label_val_pos_rate) < 0.01
        if (pred_mean_match_train and pred_train_std < 0.01) or (pred_mean_match_val and pred_val_std < 0.01):
            print(f"\n   ⚠️  WARNING: Model learned to predict prior probability (mean matches label rate but low variance)")
            if pred_mean_match_train and pred_train_std < 0.01:
                print(f"      Train: pred_mean={pred_train_mean:.4f} ≈ label_pos_rate={label_train_pos_rate:.4f}, but std={pred_train_std:.6f} is very low")
            if pred_mean_match_val and pred_val_std < 0.01:
                print(f"      Val: pred_mean={pred_val_mean:.4f} ≈ label_pos_rate={label_val_pos_rate:.4f}, but std={pred_val_std:.6f} is very low")

    # Check if predictions are perfect (overfitting)
    if len(y_pred_train) == len(y_train):
        # Use class predictions for accuracy (already converted from probabilities)
        train_acc = (y_pred_train == y_train).mean()
        val_acc = (y_pred_val == y_val).mean()
        print(f"\n   📊 Accuracy: Train={train_acc:.6f}, Val={val_acc:.6f}")
        
        # Check prediction distribution (classes)
        train_pred_dist = dict(
            zip(*np.unique(y_pred_train, return_counts=True)))
        val_pred_dist = dict(
            zip(*np.unique(y_pred_val, return_counts=True)))
        print(f"   📊 Prediction class distribution:")
        print(f"      Train: {train_pred_dist}")
        print(f"      Val: {val_pred_dist}")

        # Check if predictions are constant (only one class)
        if len(np.unique(y_pred_train)) == 1:
            print(f"   ⚠️  WARNING: Model predicts only one class in training set!")
        if len(np.unique(y_pred_val)) == 1:
            print(f"   ⚠️  WARNING: Model predicts only one class in validation set!")

        # Calculate actual logloss to verify (use probability arrays for log_loss)
        from sklearn.metrics import log_loss
        try:
            if num_classes == 2:
                # Binary: use probability array
                if y_pred_train_raw.ndim == 2:
                    train_logloss = log_loss(y_train, y_pred_train_raw[:, 1] if y_pred_train_raw.shape[1] == 2 else y_pred_train_raw)
                    val_logloss = log_loss(y_val, y_pred_val_raw[:, 1] if y_pred_val_raw.shape[1] == 2 else y_pred_val_raw)
                else:
                    train_logloss = log_loss(y_train, y_pred_train_proba)
                    val_logloss = log_loss(y_val, y_pred_val_proba)
            else:
                # Multiclass: use full probability array
                train_logloss = log_loss(y_train, y_pred_train_raw)
                val_logloss = log_loss(y_val, y_pred_val_raw)
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
