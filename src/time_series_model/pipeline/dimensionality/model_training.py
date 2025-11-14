"""Model training utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import lightgbm as lgb

from time_series_model.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder


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
    # IMPORTANT: Classification tasks use binary (0=Short, 1=Long)
    # Regression tasks (predicting returns) remain as regression - DO NOT CHANGE
    unique_labels = np.unique(y_train)
    num_unique = len(unique_labels)

    # Determine objective based on label characteristics
    # If labels are integers in [0, 2] → 3-class classification (signal prediction)
    # If labels are continuous values → regression (return prediction)
    if num_unique <= 3 and np.all(np.equal(
            np.mod(unique_labels, 1),
            0)) and np.all(unique_labels >= 0) and np.all(unique_labels <= 2):
        # Binary classification for signal prediction (drop neutral / map short)
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
        print(
            "   Using binary classification (1=Long, 0=Short); neutral labels (0=Hold) will be removed"
        )
    elif num_unique == 2:
        # Binary classification (fallback for compatibility)
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
        print(f"   Using binary classification")
    else:
        # Regression for predicting continuous returns (DO NOT CHANGE - this is correct for return prediction)
        objective = "regression"
        metric = "rmse"
        task_params = {}
        print(f"   Using regression for return prediction")

    if objective == "binary":

        def _filter_and_map(X, y, split_name: str):
            mask = (y == 1) | (y == 2)
            removed = int(len(y) - mask.sum())
            if mask.sum() == 0:
                raise ValueError(
                    f"No valid long/short samples remain in {split_name} after removing neutral labels."
                )
            if removed > 0:
                print(
                    f"   [{split_name}] Removed {removed} neutral samples; keeping {mask.sum()} long/short samples."
                )
            X_filtered = X[mask]
            y_filtered = np.where(y[mask] == 1, 1, 0).astype(int)
            return X_filtered, y_filtered

        X_train, y_train = _filter_and_map(X_train, y_train, "train")
        X_val, y_val = _filter_and_map(X_val, y_val, "validation")

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


def train_production_autoencoder(
    X: np.ndarray,
    encoding_dim: int = 8,
    epochs: int = 500,
    batch_size: int = 256,
    ae_type: str = "production",
    kl_weight: float = 1e-3,
    task_weight: float = 0.0,
    y_train: np.ndarray | None = None,
    task_head: torch.nn.Module | None = None,
):
    """Train autoencoder with optional VAE and task-aware loss."""
    print(f"🧠 Training {ae_type.upper()} Autoencoder for {epochs} epochs...")
    if ae_type == "vae":
        print(f"   VAE KL weight: {kl_weight}")

    autoencoder = UnifiedAutoencoder(
        input_dim=X.shape[1],
        encoding_dim=encoding_dim,
        architecture=ae_type,
    )

    # Prefer GPU if available
    ae_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device preference for AE: {ae_device}")

    trainer = AutoencoderTrainer(
        autoencoder,
        device=ae_device,
        kl_weight=kl_weight,
        task_weight=task_weight,
        task_head=task_head,
    )

    losses = trainer.train(
        X,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        y_train=y_train,
    )

    print("✅ Production Autoencoder training complete")
    return autoencoder, trainer, losses


def create_task_head(encoding_dim: int,
                     task_type: str = "classification",
                     num_classes: int = 2):
    """Create a task prediction head for multi-task learning."""
    import torch.nn as nn

    if task_type == "classification":
        return nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    else:
        # Regression
        return nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )


def auto_tune_hyperparameters(
    X_train: np.ndarray,
    X_val: np.ndarray,
    encoding_dim: int,
    ae_type: str,
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    task_weight: float = 0.0,
    task_head: torch.nn.Module | None = None,
    n_trials: int = 15,
) -> dict:
    """Automatically tune hyperparameters for autoencoder using grid search."""
    print(f"🔍 Auto-tuning hyperparameters ({n_trials} trials)...")

    # Parameter grid
    learning_rates = [0.001, 0.0005, 0.002, 0.0001]
    batch_sizes = [128, 256, 512]
    epochs_list = [300, 400, 500]
    kl_weights = [1e-4, 1e-3, 5e-3] if ae_type == "vae" else [0.0]

    best_params = None
    best_val_loss = float('inf')
    best_trainer = None
    best_ae = None

    import random
    trials = 0
    tried = set()

    while trials < n_trials:
        lr = random.choice(learning_rates)
        bs = random.choice(batch_sizes)
        ep = random.choice(epochs_list)
        kl_w = random.choice(kl_weights) if ae_type == "vae" else 1e-3

        key = (lr, bs, ep, kl_w)
        if key in tried:
            continue
        tried.add(key)
        trials += 1

        try:
            ae = UnifiedAutoencoder(
                input_dim=X_train.shape[1],
                encoding_dim=encoding_dim,
                architecture=ae_type,
            )

            ae_device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = AutoencoderTrainer(
                ae,
                device=ae_device,
                learning_rate=lr,
                kl_weight=kl_w,
                task_weight=task_weight,
                task_head=task_head,
            )

            # Train for a shorter period to evaluate
            trainer.train(
                X_train,
                epochs=min(ep, 100),  # Quick evaluation
                batch_size=bs,
                verbose=False,
                y_train=y_train,
            )

            # Evaluate on validation set
            with torch.no_grad():
                Xv_t = torch.as_tensor(X_val,
                                       dtype=torch.float32,
                                       device=ae_device)
                recon, _ = ae(Xv_t)
                val_loss = torch.nn.functional.mse_loss(recon, Xv_t).item()

                if ae_type == "vae":
                    h = ae.encoder_base(Xv_t)
                    mu = ae.encoder_mu(h)
                    logvar = ae.encoder_logvar(h)
                    kl = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp(),
                        dim=1).mean().item()
                    val_loss += kl_w * kl

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": ep,
                    "kl_weight": kl_w
                }
                best_trainer = trainer
                best_ae = ae
                print(
                    f"   Trial {trials}/{n_trials}: Val Loss = {val_loss:.6f} (lr={lr}, bs={bs}, ep={ep}, kl_w={kl_w})"
                )
            else:
                print(
                    f"   Trial {trials}/{n_trials}: Val Loss = {val_loss:.6f} (worse, skipping)"
                )
        except Exception as exc:
            print(f"   Trial {trials}/{n_trials}: Failed - {exc}")
            continue

    if best_params is None:
        print("   ⚠️ All trials failed, using defaults")
        best_params = {
            "lr": 0.001,
            "batch_size": 256,
            "epochs": 500,
            "kl_weight": 1e-3
        }
        best_ae = UnifiedAutoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=encoding_dim,
            architecture=ae_type,
        )
        ae_device = "cuda" if torch.cuda.is_available() else "cpu"
        best_trainer = AutoencoderTrainer(
            best_ae,
            device=ae_device,
            learning_rate=best_params["lr"],
            kl_weight=best_params["kl_weight"],
            task_weight=task_weight,
            task_head=task_head,
        )

    print(f"   ✓ Best params: {best_params}")
    return best_params, best_trainer, best_ae


def generate_auto_encoding_grid(num_features: int,
                                min_dim: int = 8,
                                max_ratio: float = 20.0) -> list:
    """Automatically generate encoding dimensions based on compression ratios."""
    # Generate dimensions based on compression ratios: 5x, 10x, 15x, 20x, 30x, etc.
    ratios = [5, 10, 15, 20, 30, 40]
    dims = []
    for ratio in ratios:
        dim = max(min_dim, int(num_features / ratio))
        if dim < num_features and dim >= min_dim:
            dims.append(dim)

    # Also add some fixed dimensions for fine-tuning
    fixed_dims = [64, 32, 16, 8]
    dims.extend([d for d in fixed_dims if d < num_features and d >= min_dim])

    # Remove duplicates and sort
    dims = sorted(set(dims), reverse=True)
    return dims
