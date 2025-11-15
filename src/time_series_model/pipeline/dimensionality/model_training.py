"""Model training utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch

from time_series_model.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder
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
