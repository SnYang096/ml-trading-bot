"""Dimensionality reduction comparison and research workflows."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import lightgbm as lgb

from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.data_loader import MarketDataLoader
from data_tools.rolling_data import create_labels_multi_horizon
from time_series_model.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder
from time_series_model.utils.training import train_lightgbm_model

# Import report generator for HTML report writing
from time_series_model.pipeline.dimensionality.report_generator import write_html_report

DIM_COMPARE_RESULTS_ROOT = Path("results") / "dim_compare"


def _slugify(value: str, default: str = "unknown") -> str:
    """Create a filesystem-friendly slug."""
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    # Replace commas with hyphens first to keep multi-symbol ordering visible
    value = value.replace(",", "-")
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug or default


def _get_primary_metric(perf: Dict) -> Tuple[str, Optional[float]]:
    """Return the primary evaluation metric name/value for a stage."""
    if not perf:
        return "", None

    financial = perf.get("financial_metrics") or {}
    win_rate = financial.get("win_rate")
    if win_rate is not None:
        return "win_rate", float(win_rate)

    classification = perf.get("classification_metrics") or {}
    for key in ("f1_macro", "f1_weighted", "accuracy"):
        val = classification.get(key)
        if val is not None:
            return key, float(val)

    return "r2", perf.get("r2")


def _derive_feature_insights(stage_baseline: Dict,
                             stage_candidate: Dict) -> Dict:
    """Summarise whether representative features improve over baseline."""
    metric_name_base, metric_base = _get_primary_metric(stage_baseline)
    metric_name_cand, metric_cand = _get_primary_metric(stage_candidate)

    metric_name = metric_name_cand or metric_name_base or "r2"
    delta = None
    if metric_base is not None and metric_cand is not None:
        delta = float(metric_cand) - float(metric_base)

    effective = delta is not None and delta > 0

    return {
        "metric_name":
        metric_name,
        "baseline_value":
        float(metric_base) if metric_base is not None else None,
        "candidate_value":
        float(metric_cand) if metric_cand is not None else None,
        "delta":
        delta,
        "effective":
        effective,
        "baseline_stage":
        "stage1_all_features",
        "candidate_stage":
        "stage3_representatives",
        "recommended_stage":
        "stage3_representatives" if effective else "stage1_all_features",
    }


def sanitize_features(X: np.ndarray, clip_std: float = 5.0) -> np.ndarray:
    """Replace NaN/inf and clip outliers per feature to stabilize AE training."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip per-column
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-8
    lower = means - clip_std * stds
    upper = means + clip_std * stds
    X = np.minimum(np.maximum(X, lower), upper)
    # Ensure finite again
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _generate_shap_outputs(
    model: lgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    prefix: str = "stage3",
    sample_size: int = 2000,
) -> Optional[str]:
    """Generate SHAP explainability artifacts for the provided LightGBM model."""
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"   ⚠️ SHAP not available ({exc}); skipping SHAP analysis.")
        return None

    if X.size == 0 or len(feature_names) == 0:
        print("   ⚠️ SHAP skipped: no data or feature names available.")
        return None

    sample_size = int(min(sample_size, X.shape[0]))
    if sample_size <= 0:
        print("   ⚠️ SHAP skipped: insufficient samples.")
        return None

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[sample_indices]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        print(f"   ⚠️ SHAP computation failed: {exc}")
        return None

    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_array = shap_values[0]
        else:
            # For multiclass, use the last class (typically positive class)
            shap_array = shap_values[-1]
    else:
        shap_array = shap_values

    shap_dir = Path(output_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    try:
        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
            color_bar=True,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / f"{prefix}_summary_bar.png", dpi=200)
        plt.close()

        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / f"{prefix}_summary_beeswarm.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"   ⚠️ Failed to render SHAP plots: {exc}")

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    shap_ranking = sorted(
        [{
            "feature": feat,
            "mean_abs_shap": float(val),
            "rank": idx + 1,
        } for idx, (feat, val) in enumerate(
            sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda kv: kv[1],
                reverse=True,
            ))],
        key=lambda item: item["rank"],
    )

    with open(shap_dir / f"{prefix}_shap_importance.json",
              "w",
              encoding="utf-8") as f:
        json.dump(shap_ranking, f, indent=2)

    print(f"   💾 SHAP summary saved to: {shap_dir}")
    return str(shap_dir)


def compute_selection_score(
    perf: Dict,
    metric: str,
    *,
    max_dd_threshold: float = -20.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Compute selection score from a performance dictionary using the chosen metric."""
    fm = perf.get("financial_metrics", {}) if isinstance(perf, dict) else {}
    sharpe = float(fm.get("sharpe_ratio", 0.0))
    max_dd = float(fm.get("max_drawdown", 0.0))
    f1 = float(fm.get("f1", fm.get("directional_f1", 0.0)))
    if f1 == 0.0:
        f1 = float(fm.get("win_rate", 0.0)) / 100.0

    if metric == "sharpe":
        return sharpe
    if metric == "f1":
        return f1
    if metric == "r2":
        return float(perf.get("r2", 0.0))

    # Composite score: Sharpe - alpha * penalty(DD) - beta * (1 - F1)
    dd_penalty = 0.0
    if max_dd < max_dd_threshold:
        dd_penalty = abs(max_dd - max_dd_threshold)
    return sharpe - alpha * dd_penalty - beta * (1.0 - f1)


def load_real_market_data(
    data_path: str,
    symbol: str = "ETH-USD",
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: list[int] | None = None,
    feature_type: str = "comprehensive",
) -> Tuple[np.ndarray, np.ndarray, list, list[int], pd.DataFrame]:
    """Load real market data for one or multiple symbols.
    
    Args:
        symbol: Single symbol or comma-separated symbols (e.g., "ETH-USD" or "ETH-USD,BTC-USD,SOL-USD")
    """
    # Support multiple symbols (comma-separated)
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
    symbols_str = ",".join(symbol_list) if len(
        symbol_list) > 1 else symbol_list[0] if symbol_list else "UNKNOWN"
    print(f"📊 Loading real market data for {symbols_str}...")
    print(f"   Feature type: {feature_type}")
    if len(symbol_list) > 1:
        print(f"   Multi-asset training: {len(symbol_list)} assets")

    try:
        loader = MarketDataLoader(data_path)
        # Load and resample data for all symbols, then merge
        all_dfs = []
        for sym in symbol_list:
            # Create a new loader for each symbol to ensure proper resampling
            symbol_loader = MarketDataLoader(data_path)
            df_single = symbol_loader.load_data(symbol=sym,
                                                start_date=start_date,
                                                end_date=end_date)
            if df_single is not None and not df_single.empty:
                # Resample each symbol's data before merging
                if hasattr(symbol_loader, 'resample_data'):
                    df_single = symbol_loader.resample_data("5T")
                elif isinstance(df_single.index, pd.DatetimeIndex):
                    # Fallback: resample manually
                    df_single = df_single.resample("5T").agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                if df_single is not None and not df_single.empty:
                    all_dfs.append(df_single)

        if not all_dfs:
            print(
                "⚠️ No real data found for any symbol, generating sample data..."
            )
            return create_enhanced_sample_data()

        # Merge all dataframes (already resampled)
        # For multi-asset training, all assets' data are merged together
        # All features are normalized (asset-agnostic), so the model can learn
        # common patterns across different assets
        df = pd.concat(all_dfs, axis=0).sort_index()
        if len(symbol_list) > 1:
            print(
                f"   Merged {len(all_dfs)} asset(s), total {len(df)} samples")

        comprehensive_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_type)
        df_features = comprehensive_engineer.engineer_all_features(df,
                                                                   fit=True)

        # Parse horizons
        if horizons and len(horizons) > 0:
            horizons_list = horizons
        else:
            horizons_list = [1]

        # Create multi-horizon labels
        print(
            f"   Creating multi-horizon labels for horizons: {horizons_list}")
        df_features = create_labels_multi_horizon(df_features,
                                                  horizons=horizons_list)

        # Store original df_features for multi-horizon label creation
        df_features_stored = df_features.copy()

        # Build safe feature columns (exclude targets/labels and future info)
        exclude_exact = {
            "timestamp",
            "close",
            "signal",
            "binary_signal",
            "future_return",
        }
        exclude_prefixes = (
            "signal_",
            "binary_signal_",
            "future_return_",
        )
        feature_cols = [
            col for col in df_features.columns
            if (col not in exclude_exact) and (not any(
                col.startswith(pfx) for pfx in exclude_prefixes))
        ]

        # Debug: engineered feature summary
        try:
            print(
                f"[DEBUG] Engineered features: total={len(feature_cols)} | sample={feature_cols[:10]}"
            )
        except Exception:
            pass

        X = df_features[feature_cols].values

        # Use first horizon for backward compatibility
        default_horizon = horizons_list[0]
        y = df_features[f"signal_{default_horizon}"].dropna(
        ).values  # Use 3-class signal (0=Hold, 1=Long, 2=Short)

        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        print(f"✅ Real data loaded: {X.shape}, {y.shape}")
        print(
            f"   Using horizon: {default_horizon} bars (for backward compatibility)"
        )

        # Store horizons for multi-horizon training
        if len(horizons_list) > 1:
            print(f"   Multi-horizon mode enabled: {horizons_list}")

        return X, y, feature_cols, horizons_list, df_features_stored

    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Error loading real data: {exc}")
        print("📊 Generating sample data...")
        X, y, feature_cols = create_enhanced_sample_data()
        return X, y, feature_cols, [1], pd.DataFrame()


def create_enhanced_sample_data(
    n_samples: int = 10000,
    n_factors: int = 100,
) -> Tuple[np.ndarray, np.ndarray, list]:
    print(
        f"📊 Creating enhanced sample data: {n_samples} samples, {n_factors} features"
    )

    np.random.seed(42)

    factor_names = []
    categories = [
        "momentum",
        "volatility",
        "mean_reversion",
        "trend",
        "volume",
        "liquidity",
        "sentiment",
    ]

    for i in range(n_factors):
        category = categories[i % len(categories)]
        factor_names.append(f"{category}_{i+1}")

    X = np.random.randn(n_samples, n_factors)

    for i in range(0, n_factors, 10):
        if i + 5 < n_factors:
            X[:, i + 1:i + 5] = (X[:, i:i + 4] * 0.7 +
                                 np.random.randn(n_samples, 4) * 0.3)

    momentum_factors = [
        i for i, name in enumerate(factor_names) if "momentum" in name
    ]
    volatility_factors = [
        i for i, name in enumerate(factor_names) if "volatility" in name
    ]
    trend_factors = [
        i for i, name in enumerate(factor_names) if "trend" in name
    ]

    y = (np.tanh(X[:, momentum_factors].mean(axis=1)) * 0.4 +
         np.sin(X[:, volatility_factors].mean(axis=1)) * 0.3 +
         X[:, trend_factors].mean(axis=1) * 0.2 +
         np.random.randn(n_samples) * 0.1)

    return X, y, factor_names


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
                     num_classes: int = 3):
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


def train_production_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict | None = None,
):
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
    # IMPORTANT: Classification tasks use 3-class (0=Hold, 1=Long, 2=Short)
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


def calculate_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate financial metrics for trading strategy evaluation.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
        risk_free_rate: Risk-free rate (annualized, default 0)
    
    Returns:
        Dictionary of financial metrics
    """
    metrics = {}

    try:
        # Strategy: use predicted returns as position signals
        # Simple strategy: long if pred > 0, short if pred < 0 (proportional to confidence)
        positions = np.sign(y_pred) * np.abs(y_pred)
        # Clip positions to reasonable range [-1, 1]
        positions = np.clip(positions, -1.0, 1.0)

        # Strategy returns: position * true return
        strategy_returns = positions * y_true

        # 1. Total return (cumulative)
        total_return = float(np.sum(strategy_returns))
        metrics["total_return"] = total_return

        # 2. Annualized return (assuming daily data)
        n_periods = len(strategy_returns)
        if n_periods > 0:
            # Simple annualization: multiply by ~252 trading days
            annualized_return = total_return * (
                252.0 / n_periods) if n_periods < 252 else total_return
            metrics["annualized_return"] = annualized_return
        else:
            metrics["annualized_return"] = 0.0

        # 3. Sharpe ratio
        returns_std = np.std(strategy_returns)
        if returns_std > 1e-8:
            # Annualized Sharpe: (mean_return - risk_free) / std_return * sqrt(252)
            daily_rf = risk_free_rate / 252.0
            sharpe_ratio = (np.mean(strategy_returns) -
                            daily_rf) / returns_std * np.sqrt(252.0)
            metrics["sharpe_ratio"] = float(sharpe_ratio)
        else:
            metrics["sharpe_ratio"] = 0.0

        # 4. Maximum drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        metrics["max_drawdown"] = max_drawdown
        metrics["max_drawdown_pct"] = max_drawdown / (
            1.0 + abs(running_max[-1])) if len(
                running_max) > 0 and running_max[-1] != 0 else 0.0

        # 5. Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        metrics["win_rate"] = float(winning_trades /
                                    total_trades) if total_trades > 0 else 0.0

        # 6. Average win/loss ratio
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        metrics["avg_win"] = avg_win
        metrics["avg_loss"] = avg_loss
        metrics[
            "win_loss_ratio"] = avg_win / avg_loss if avg_loss > 1e-8 else 0.0

        # 7. Volatility (annualized)
        volatility = np.std(strategy_returns) * np.sqrt(252.0)
        metrics["volatility"] = float(volatility)

        # 8. Calmar ratio (return / max_drawdown)
        if abs(max_drawdown) > 1e-8:
            metrics["calmar_ratio"] = float(annualized_return /
                                            abs(max_drawdown))
        else:
            metrics["calmar_ratio"] = 0.0

    except Exception as e:
        print(f"⚠️ Error calculating financial metrics: {e}")
        # Return defaults
        metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
        }

    return metrics


def evaluate_model_performance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    include_financial_metrics: bool = True,
):
    X_eval = X_test
    y_eval = y_test

    # Remove neutral (0 = hold) labels for classification metrics if present
    if y_eval.ndim == 1 and np.issubdtype(y_eval.dtype, np.integer):
        unique_targets = np.unique(y_eval)
        if np.any(unique_targets == 0) and np.all(
                np.isin(unique_targets, [0, 1, 2])):
            mask_active = y_eval != 0
            if mask_active.sum() == 0:
                raise ValueError(
                    "No active samples remain after removing neutral labels in evaluation set."
                )
            X_eval = X_eval[mask_active]
            y_eval = y_eval[mask_active]

    predictions = model.predict(X_eval)

    # Handle multiclass predictions: LightGBM returns probability array for multiclass
    # Shape: (n_samples, n_classes) for multiclass, (n_samples,) for binary/regression
    is_multiclass = predictions.ndim == 2 and predictions.shape[1] > 1
    if is_multiclass:
        # Multiclass: convert probability array to class predictions
        predictions_class = np.argmax(predictions, axis=1)
        # For metrics, use class predictions
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class
        probabilities = predictions
    else:
        # Binary or regression: use predictions as-is
        predictions_for_metrics = predictions
        predictions_to_store = predictions
        probabilities = None

    # Identify binary classification
    is_binary_classification = False
    if not is_multiclass and y_eval.ndim == 1 and np.issubdtype(
            y_eval.dtype, np.integer):
        unique_eval = np.unique(y_eval)
        if np.all(np.isin(unique_eval, [0, 1, 2])):
            # Map to binary labels (1=Long, 0=Short)
            y_eval_binary = np.where(y_eval == 1, 1, 0).astype(int)
            is_binary_classification = True
        elif np.all(np.isin(unique_eval, [0, 1])):
            y_eval_binary = y_eval.astype(int)
            is_binary_classification = True
        else:
            y_eval_binary = None
    else:
        y_eval_binary = None

    if is_binary_classification:
        if predictions.ndim == 1:
            probabilities = predictions
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            probabilities = predictions[:, 0]
        elif predictions.ndim == 2 and predictions.shape[1] == 2:
            probabilities = predictions[:, 1]
        else:
            probabilities = predictions
        predictions_class = (probabilities >= 0.5).astype(int)
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class

    # Basic numeric metrics (note: for multiclass these are not very meaningful)
    mse = mean_squared_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics)
    # For multiclass, R² may not be meaningful, but we'll calculate it anyway
    target_for_r2 = (y_eval if not is_binary_classification else y_eval_binary)
    r2 = r2_score(target_for_r2, predictions_for_metrics) if len(
        np.unique(target_for_r2)) > 1 else 0.0

    print(f"📊 {model_name} Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions_to_store,
    }

    # Add financial or directional metrics
    if include_financial_metrics:
        if is_multiclass:
            # Compute directional win rate among non-hold predictions (1=Long, 2=Short)
            y_pred_cls = predictions_class.astype(int)
            y_true_cls = y_eval.astype(int)
            non_hold_mask = y_pred_cls != 0
            long_mask = y_pred_cls == 1
            short_mask = y_pred_cls == 2
            active = int(np.sum(non_hold_mask))
            total = int(len(y_pred_cls))
            active_ratio = float(active / total) if total > 0 else 0.0

            if active > 0:
                correct_non_hold = ((y_pred_cls == 1) &
                                    (y_true_cls == 1)) | ((y_pred_cls == 2) &
                                                          (y_true_cls == 2))
                win_rate = float(np.sum(correct_non_hold) / active)
            else:
                win_rate = 0.0

            # Long-only win rate
            long_total = int(np.sum(long_mask))
            if long_total > 0:
                long_correct = np.sum((y_pred_cls == 1) & (y_true_cls == 1))
                long_win_rate = float(long_correct / long_total)
            else:
                long_win_rate = 0.0

            # Short-only win rate
            short_total = int(np.sum(short_mask))
            if short_total > 0:
                short_correct = np.sum((y_pred_cls == 2) & (y_true_cls == 2))
                short_win_rate = float(short_correct / short_total)
            else:
                short_win_rate = 0.0

            fm = results.setdefault("financial_metrics", {})
            fm["win_rate"] = win_rate
            fm["long_win_rate"] = long_win_rate
            fm["short_win_rate"] = short_win_rate
            fm["active_ratio"] = active_ratio

            print(f"  Directional Win Rate (non-hold): {win_rate:.4f}")
            print(f"  Long Win Rate: {long_win_rate:.4f}")
            print(f"  Short Win Rate: {short_win_rate:.4f}")
            print(f"  Active Ratio: {active_ratio:.4f}")

            # Classification diagnostics
            metrics = results.setdefault("classification_metrics", {})
            accuracy = float(np.mean(y_pred_cls == y_true_cls))
            metrics["accuracy"] = accuracy
            try:
                metrics["f1_macro"] = float(
                    f1_score(y_true_cls, y_pred_cls, average="macro"))
            except Exception:
                metrics["f1_macro"] = None
            try:
                metrics["f1_weighted"] = float(
                    f1_score(y_true_cls, y_pred_cls, average="weighted"))
            except Exception:
                metrics["f1_weighted"] = None

            active_mask_true = y_true_cls != 0
            active_mask_pred = y_pred_cls != 0
            active_mask = active_mask_true | active_mask_pred
            if np.any(active_mask):
                try:
                    metrics["f1_active_macro"] = float(
                        f1_score(
                            y_true_cls[active_mask],
                            y_pred_cls[active_mask],
                            average="macro",
                        ))
                except Exception:
                    metrics["f1_active_macro"] = None
            else:
                metrics["f1_active_macro"] = None

            class_labels = list(range(probabilities.shape[1]))
            y_true_onehot = label_binarize(y_true_cls, classes=class_labels)
            if y_true_onehot.shape[1] != probabilities.shape[1]:
                # Align shapes by padding if necessary
                diff = probabilities.shape[1] - y_true_onehot.shape[1]
                if diff > 0:
                    y_true_onehot = np.hstack([
                        y_true_onehot,
                        np.zeros((y_true_onehot.shape[0], diff))
                    ])

            if y_true_onehot.shape[1] == 1:
                # Binary case after binarize -> single column
                y_true_onehot = np.hstack((1 - y_true_onehot, y_true_onehot))

            try:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(
                        y_true_cls,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                    ))
            except Exception:
                metrics["roc_auc_macro"] = None
            try:
                metrics["pr_auc_macro"] = float(
                    average_precision_score(
                        y_true_onehot,
                        probabilities,
                        average="macro",
                    ))
            except Exception:
                metrics["pr_auc_macro"] = None
            try:
                cm = confusion_matrix(y_true_cls,
                                      y_pred_cls,
                                      labels=class_labels)
                metrics["confusion_matrix"] = cm.tolist()
                metrics["labels"] = [int(c) for c in class_labels]
            except Exception:
                metrics["confusion_matrix"] = None
                metrics["labels"] = None

            try:
                cls_report = classification_report(
                    y_true_cls,
                    y_pred_cls,
                    output_dict=True,
                    zero_division=0,
                )
                metrics["classification_report"] = cls_report
            except Exception:
                metrics["classification_report"] = None

            metrics["support"] = int(len(y_true_cls))

            print(f"  Accuracy: {accuracy:.4f}")
            if metrics["f1_macro"] is not None:
                print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            if metrics["roc_auc_macro"] is not None:
                print(f"  ROC AUC (macro): {metrics['roc_auc_macro']:.4f}")
            if metrics["pr_auc_macro"] is not None:
                print(f"  PR AUC (macro): {metrics['pr_auc_macro']:.4f}")
        elif is_binary_classification:
            y_true_bin = y_eval_binary
            y_pred_bin = predictions_for_metrics.astype(int)
            fm = results.setdefault("financial_metrics", {})
            accuracy = float(np.mean(y_pred_bin == y_true_bin))
            fm["win_rate"] = accuracy
            fm["active_ratio"] = 1.0

            long_mask = y_pred_bin == 1
            short_mask = y_pred_bin == 0
            long_total = int(np.sum(long_mask))
            short_total = int(np.sum(short_mask))
            fm["long_win_rate"] = float(np.mean(
                y_true_bin[long_mask] == 1)) if long_total > 0 else 0.0
            fm["short_win_rate"] = float(np.mean(
                y_true_bin[short_mask] == 0)) if short_total > 0 else 0.0

            metrics = results.setdefault("classification_metrics", {})
            metrics["accuracy"] = accuracy
            try:
                metrics["f1_macro"] = float(
                    f1_score(y_true_bin, y_pred_bin, average="macro"))
            except Exception:
                metrics["f1_macro"] = None
            try:
                metrics["f1_weighted"] = float(
                    f1_score(y_true_bin, y_pred_bin, average="weighted"))
            except Exception:
                metrics["f1_weighted"] = None
            try:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true_bin, probabilities))
            except Exception:
                metrics["roc_auc_macro"] = None
            try:
                metrics["pr_auc_macro"] = float(
                    average_precision_score(y_true_bin, probabilities))
            except Exception:
                metrics["pr_auc_macro"] = None
            try:
                cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
                metrics["confusion_matrix"] = cm.tolist()
                metrics["labels"] = [0, 1]
            except Exception:
                metrics["confusion_matrix"] = None
                metrics["labels"] = None
            try:
                metrics["classification_report"] = classification_report(
                    y_true_bin, y_pred_bin, output_dict=True, zero_division=0)
            except Exception:
                metrics["classification_report"] = None
            metrics["support"] = int(len(y_true_bin))

            print(f"  Accuracy: {accuracy:.4f}")
            if metrics.get("f1_macro") is not None:
                print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            if metrics.get("roc_auc_macro") is not None:
                print(f"  ROC AUC: {metrics['roc_auc_macro']:.4f}")
            if metrics.get("pr_auc_macro") is not None:
                print(f"  PR AUC: {metrics['pr_auc_macro']:.4f}")
        else:
            # Regression/binary: compute financial metrics using returns-like predictions
            financial_metrics = calculate_financial_metrics(
                y_eval, predictions_for_metrics)
            results["financial_metrics"] = financial_metrics
            print(
                f"  Sharpe Ratio: {financial_metrics.get('sharpe_ratio', 0):.4f}"
            )
            print(
                f"  Total Return: {financial_metrics.get('total_return', 0):.4f}"
            )
            print(
                f"  Max Drawdown: {financial_metrics.get('max_drawdown', 0):.4f}"
            )

    return results


def save_production_results(
    results: Dict,
    model,
    autoencoder: Optional[UnifiedAutoencoder],
    results_dir: str,
) -> str:
    print("💾 Saving production results...")
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/production_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump(model, f"{results_dir}/production_model.pkl")
    # Autoencoder disabled: only save model; skip AE artifact if not provided
    if autoencoder is not None:
        torch.save(autoencoder.state_dict(),
                   f"{results_dir}/production_autoencoder.pth")

    print(f"✅ Results saved to {results_dir}")
    return results_dir


def run_dimensionality_comparison(
    data_path: str = "/data/parquet_data",
    symbol: str = "ETH-USD",
    encoding_dim: int = 8,
    autoencoder_epochs: int = 500,
    train_start: str | None = None,
    train_end: str | None = None,
    feature_type: str = "comprehensive",
    top_k: Optional[int] = None,
    shap_analysis: bool = True,
) -> Tuple[Dict, any, Optional[UnifiedAutoencoder], str]:
    print("🚀 Dimensionality Reduction Comparison Training")
    print("=" * 60)
    start_dt = datetime.now()
    timestamp_start = start_dt.strftime("%Y%m%d_%H%M%S")
    symbol_slug = _slugify(symbol)
    feature_slug = _slugify(feature_type)

    X, y, feature_names, horizons_loaded, df_features_full = load_real_market_data(
        data_path,
        symbol,
        start_date=train_start,
        end_date=train_end,
        feature_type=feature_type,
    )

    print(f"✅ Data loaded: {X.shape}, {y.shape}")
    print(f"✅ Features: {len(feature_names)}")

    print("\n📊 Data preprocessing...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Feature/label sanitation before AE/GBM
    X_scaled = sanitize_features(X_scaled, clip_std=5.0)
    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Non-finite values remain in features after sanitation")
    if not np.isfinite(y_scaled).all():
        raise ValueError("Non-finite values found in labels after scaling")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.3,
        shuffle=False,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        shuffle=False,
    )

    print(
        f"✅ Data split: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # Autoencoder disabled: use original features as "compressed" placeholders
    autoencoder = None
    X_train_emb = X_train
    X_val_emb = X_val
    X_test_emb = X_test

    print("\n🌲 Training original features model...")
    model_original = train_production_lightgbm(X_train, y_train, X_val, y_val)

    print("\n🌲 Training compressed features model...")
    model_compressed = train_production_lightgbm(
        X_train_emb,
        y_train,
        X_val_emb,
        y_val,
    )

    print("\n📊 Evaluating performance...")
    # Evaluate on validation set
    results_original_val = evaluate_model_performance(
        model_original,
        X_val,
        y_val,
        "Original Features (Val)",
    )
    results_compressed_val = evaluate_model_performance(
        model_compressed,
        X_val_emb,
        y_val,
        "Compressed Features (Val)",
    )

    # Evaluate on test set
    results_original = evaluate_model_performance(
        model_original,
        X_test,
        y_test,
        "Original Features (Test)",
    )
    results_compressed = evaluate_model_performance(
        model_compressed,
        X_test_emb,
        y_test,
        "Compressed Features (Test)",
    )

    print("\n📋 Generating production report...")

    compression_ratio = 1.0
    performance_change = results_compressed["r2"] - results_original["r2"]

    # Format training date range for directory name (include symbol and feature_type)
    if train_start and train_end:
        # Extract date parts (YYYY-MM-DD -> YYYYMMDD)
        train_start_date = train_start.replace("-", "")[:8]
        train_end_date = train_end.replace("-", "")[:8]
        dir_date_suffix = f"{symbol_slug}_{feature_slug}_{train_start_date}_{train_end_date}"
    else:
        # Fallback to runtime timestamps if no date range provided
        train_start_date = None
        train_end_date = None
        timestamp_end = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_date_suffix = f"{symbol_slug}_{feature_slug}_{timestamp_start}_{timestamp_end}"

    stage3_feature_count = len(feature_names)
    ae_enabled = autoencoder is not None

    results = {
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "duration_sec": (datetime.now() - start_dt).total_seconds(),
        "data_info": {
            "original_features_count":
            X.shape[1],
            "compressed_dimensions":
            results_compressed.get("feature_count", X.shape[1])
            if ae_enabled else len(feature_names),
            "compression_ratio":
            compression_ratio if ae_enabled else
            ((X.shape[1] / len(feature_names)) if len(feature_names) else 1.0),
            "training_samples":
            len(X_train),
            "validation_samples":
            len(X_val),
            "test_samples":
            len(X_test),
        },
        "training_info": {
            # Autoencoder disabled
            "autoencoder_epochs": 0,
            "autoencoder_final_loss": None,
            "lightgbm_original_iterations": model_original.best_iteration,
            "lightgbm_compressed_iterations": model_compressed.best_iteration,
        },
        "performance": {
            "original_features":
            results_original,
            "compressed_features":
            results_compressed,
            "original_features_val":
            results_original_val,
            "compressed_features_val":
            results_compressed_val,
            "performance_change":
            performance_change,
            "performance_change_percent":
            (performance_change / results_original["r2"]) *
            100 if results_original["r2"] != 0 else 0,
        },
        "model_info": {
            "device_used": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "feature_names": feature_names[:10],
        },
    }

    results["insights"] = _derive_feature_insights(results_original,
                                                   results_compressed)

    # Build results directory name using training date range (if available) or runtime timestamps
    DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results_dir_path = DIM_COMPARE_RESULTS_ROOT / dir_date_suffix
    results_dir = save_production_results(
        results,
        model_compressed,
        autoencoder,
        str(results_dir_path),
    )

    print("\n" + "=" * 60)
    print("🎉 Dimensionality Reduction Comparison Complete!")
    print("=" * 60)
    print(f"📊 Compression Ratio: {compression_ratio:.1f}x")
    print(
        f"📈 Performance Change: {performance_change:.4f} ({results['performance']['performance_change_percent']:.1f}%)"
    )
    print(f"💾 Results saved to: {results_dir}")

    return results, model_compressed, autoencoder, results_dir


def main() -> Tuple[Dict, any, Optional[UnifiedAutoencoder], str]:
    parser = argparse.ArgumentParser(
        description=
        "Dimensionality reduction comparison: evaluate feature reduction stages (All → IC-filtered → Representatives → Autoencoder)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="/data/parquet_data",
        help="Parquet directory with real market data",
    )
    parser.add_argument(
        "--symbol",
        default="ETH-USD",
        help=
        "Symbol name(s) (e.g., BTC-USD, ETH-USD or BTC-USD,ETH-USD,SOL-USD for multi-asset training)",
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Autoencoder embedding dimension",
    )
    parser.add_argument(
        "--encoding-grid",
        default=None,
        help="Comma-separated list of encoding dims to try (e.g., 8,16,32,64)",
    )
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=500,
        help="Autoencoder training epochs",
    )
    parser.add_argument(
        "--ae-type",
        type=str,
        default="vae",
        choices=["production", "vae"],
        help=
        "Autoencoder type: 'production' (standard AE) or 'vae' (Variational AE)",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-3,
        help="KL divergence weight for VAE (default: 1e-3)",
    )
    parser.add_argument(
        "--auto-encoding-grid",
        action="store_true",
        default=True,
        help="Automatically generate encoding grid based on compression ratios",
    )
    parser.add_argument(
        "--ae-auto-tune",
        action="store_true",
        default=True,
        help=
        "Enable automatic hyperparameter tuning for autoencoder (learning rate, batch size, epochs)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=15,
        help="Number of trials for hyperparameter tuning (default: 15)",
    )
    parser.add_argument(
        "--ae-task-loss",
        action="store_true",
        default=True,
        help="Enable task-aware loss (reconstruction + prediction task loss)",
    )
    parser.add_argument(
        "--task-weight",
        type=float,
        default=0.1,
        help="Weight for task loss in multi-task training (default: 0.1)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=["sharpe", "f1", "r2", "composite"],
        help=
        "Metric to select best AE dimension: sharpe | f1 | r2 | composite (default)",
    )
    parser.add_argument(
        "--max-dd-threshold",
        type=float,
        default=-20.0,
        help="Max drawdown threshold (%) for composite scoring (default: -20)",
    )
    parser.add_argument(
        "--composite-alpha",
        type=float,
        default=0.5,
        help=
        "Alpha penalty weight for exceeding max drawdown in composite score (default: 0.5)",
    )
    parser.add_argument(
        "--composite-beta",
        type=float,
        default=0.5,
        help=
        "Beta penalty weight for (1 - F1) in composite score (default: 0.5)",
    )
    parser.add_argument(
        "--train-start",
        default=None,
        help="Start date (YYYY-MM-DD) for data window",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="End date (YYYY-MM-DD) for data window",
    )
    parser.add_argument(
        "--report-html",
        default=None,
        help="Path to write an HTML summary report",
    )
    parser.add_argument(
        "--export-model",
        default=None,
        help=
        "Optional path under models/ to copy the best production_model.pkl",
    )
    parser.add_argument(
        "--research-ablation",
        action="store_true",
        help=
        "Run IC filter -> representative selection -> multi-dim AE (60→32→16→8) and report reconstruction vs downstream R2",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=
        "Optional: number of top factors retained after IC ranking/representative selection",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,10,15",
        help=
        "Comma-separated list of forward bars for multi-horizon labels (e.g., 1,5,10,15)",
    )
    parser.add_argument(
        "--binary-signals",
        action="store_true",
        default=False,
        help=
        "Use binary labels (1=Long, 0=Short) without Hold. Threshold controlled by --label-threshold",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.0,
        help=
        "Threshold for future return to classify Long vs Short in binary mode (default 0.0)",
    )
    parser.set_defaults(shap_analysis=True)
    parser.add_argument(
        "--shap-analysis",
        dest="shap_analysis",
        action="store_true",
        help=
        "Generate SHAP explainability plots for representative factors (default: enabled).",
    )
    parser.add_argument(
        "--no-shap-analysis",
        dest="shap_analysis",
        action="store_false",
        help="Disable SHAP explainability plots.",
    )
    parser.add_argument(
        "--enable-autoencoder",
        action="store_true",
        help="Run Stage 4 autoencoder compression (disabled by default).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["classification", "regression", "both"],
        help=
        "Task type to evaluate: classification | regression | both (default)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help=
        "Feature type: baseline/default/enhanced/hurst/wavelet/hilbert/spectral/order_flow/dl_sequence/comprehensive or combos (default: comprehensive)",
    )

    args = parser.parse_args()
    symbol_slug = _slugify(args.symbol)
    feature_type_slug = _slugify(args.feature_type)

    # Enforce minimal training window (one quarter ~ 90 days)
    if args.train_start and args.train_end:
        try:
            start_dt_chk = pd.to_datetime(args.train_start)
            end_dt_chk = pd.to_datetime(args.train_end)
            if (end_dt_chk - start_dt_chk).days < 90:
                raise ValueError(
                    f"Training window too short: {args.train_start} → {args.train_end} (< 90 days). Please provide at least one quarter."
                )
        except Exception as _e:
            raise

    # Default behavior: if neither grid nor ablation specified, enable ablation by default
    if not args.encoding_grid and not args.research_ablation:
        args.research_ablation = True

    # If grid is provided, run multiple trials and select the best (baseline AE compare)
    grid_dims = None
    if args.encoding_grid:
        try:
            grid_dims = [
                int(x.strip()) for x in args.encoding_grid.split(',')
                if x.strip()
            ]
        except Exception:
            print(f"⚠️ Invalid --encoding-grid format: {args.encoding_grid}")

    if args.research_ablation:
        ablation_start_dt = datetime.now()
        ablation_start_ts = ablation_start_dt.strftime("%Y%m%d_%H%M%S")
        # Format training date range for directory name (if provided)
        if args.train_start and args.train_end:
            train_start_date = args.train_start.replace("-", "")[:8]
            train_end_date = args.train_end.replace("-", "")[:8]
            ablation_dir_date_suffix = f"{symbol_slug}_{feature_type_slug}_{train_start_date}_{train_end_date}"
        else:
            train_start_date = None
            train_end_date = None
            ablation_dir_date_suffix = f"{symbol_slug}_{feature_type_slug}_{ablation_start_ts}"  # Use runtime timestamps with symbol and feature_type
        # Parse horizons from args
        horizons_list = [int(h.strip()) for h in args.horizons.split(",")
                         ] if args.horizons else [1]

        # Load engineered features for IC & representative selection
        X_raw, y_raw, feature_names, horizons_loaded, df_features_original = load_real_market_data(
            args.data_path,
            args.symbol,
            args.train_start,
            args.train_end,
            horizons=horizons_list,
            feature_type=args.feature_type)

        # Use loaded horizons or fallback to parsed horizons
        horizons = horizons_loaded if horizons_loaded and len(
            horizons_loaded) > 0 else horizons_list

        original_feature_count = len(
            feature_names)  # Save original count (482)
        dfX = pd.DataFrame(X_raw, columns=feature_names)

        # For backward compatibility, use default horizon
        y_series = pd.Series(y_raw)
        # If binary mode: remap labels to 2-class using future_return threshold
        if args.binary_signals:
            try:
                # Use first horizon's future return if available
                default_h = horizons[0] if horizons else 1
                fr_col = f"future_return_{default_h}"
                if fr_col in df_features_original.columns:
                    fr = df_features_original[fr_col].values
                else:
                    # Fallback: compute from close
                    close = df_features_original["close"].values
                    fr = np.roll(close, -default_h) / close - 1.0
                thr = float(args.label_threshold)
                y_series = pd.Series((fr > thr).astype(int))
                print(
                    f"[Label] Using binary signals (thr={thr}), positives={y_series.mean():.4f}"
                )
            except Exception as exc:
                print(
                    f"⚠️ Binary label remap failed, keep original labels: {exc}"
                )

        # Stage 1: All original features (482) - missing/stability filter only
        print(f"\n[Stage 1] All original features: {len(dfX.columns)}")
        keep_all = []
        for c in dfX.columns:
            s = dfX[c]
            if s.isna().mean() < 0.2 and s.std() > 1e-8:
                keep_all.append(c)
        df_all = dfX[keep_all].fillna(method="ffill").fillna(
            method="bfill").fillna(0.0)
        X_all = df_all.values
        scaler_all = StandardScaler()
        X_all_scaled = sanitize_features(scaler_all.fit_transform(X_all))
        print(
            f"[DEBUG] Stage 1: {len(keep_all)} features after missing/stability filter"
        )

        # Stage 2: IC (Spearman) ranking - top features by |IC|
        print(f"\n[Stage 2] IC ranking...")
        ic_scores = {}
        for col in df_all.columns:
            try:
                ic = spearmanr(df_all[col].values,
                               y_series.values,
                               nan_policy="omit")[0]
            except Exception:
                ic = 0.0
            ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        top_sorted = sorted(ic_scores.items(),
                            key=lambda kv: abs(kv[1]),
                            reverse=True)
        target_top_k = args.top_k or 120
        ic_top_k = min(max(target_top_k, 1), len(top_sorted))
        if ic_top_k == 0:
            ic_top_k = min(60, len(top_sorted))
        top_cols = [c for c, _ in top_sorted[:ic_top_k]]
        df_ic = df_all[top_cols].copy()
        X_ic = df_ic.values
        scaler_ic = StandardScaler()
        X_ic_scaled = sanitize_features(scaler_ic.fit_transform(X_ic))
        print(
            f"[DEBUG] Stage 2: {len(top_cols)} features after IC ranking (target={target_top_k})"
        )

        # Stage 3: Correlation-based representative selection
        print(f"\n[Stage 3] Correlation-based representative selection...")
        # Missing and stability filter on IC-selected features
        keep_ic = []
        for c in df_ic.columns:
            s = df_ic[c]
            if s.isna().mean() < 0.2 and s.std() > 1e-8:
                keep_ic.append(c)
        df_ic_clean = df_ic[keep_ic].fillna(method="ffill").fillna(
            method="bfill").fillna(0.0)

        # Greedy representative selection by correlation threshold (0.9)
        reps: list[str] = []
        if not df_ic_clean.empty:
            corr = df_ic_clean.corr().abs().fillna(0.0)
            for c in df_ic_clean.columns:
                if all(corr.loc[c, r] < 0.9 for r in reps):
                    reps.append(c)
        # Bound reps between 60 and 100
        desired_reps = (min(target_top_k, len(df_ic_clean.columns))
                        if target_top_k and not df_ic_clean.empty else None)
        if not df_ic_clean.empty:
            if desired_reps:
                if len(reps) > desired_reps:
                    reps = reps[:desired_reps]
                elif len(reps) < desired_reps:
                    additional = [
                        c for c in df_ic_clean.columns if c not in reps
                    ][:max(desired_reps - len(reps), 0)]
                    reps.extend(additional)
            else:
                if len(reps) < 60:
                    reps = list(df_ic_clean.columns)[:60]
                elif len(reps) > 100:
                    reps = reps[:100]
        if not reps:
            fallback_source = (df_ic_clean.columns
                               if not df_ic_clean.empty else df_ic.columns)
            if len(fallback_source) == 0:
                fallback_source = df_all.columns
            reps = list(fallback_source)[:max(target_top_k or 60, 1)]
        df_reps = (df_ic_clean[reps] if set(reps).issubset(df_ic_clean.columns)
                   else df_all[reps].fillna(0.0))
        X_reps = df_reps.values
        scaler_reps = StandardScaler()
        X_reps_scaled = sanitize_features(scaler_reps.fit_transform(X_reps))
        print(
            f"[DEBUG] Stage 3: {len(reps)} representative features after correlation filtering"
        )

        # Stage 4: Autoencoder compression (optional)
        if args.enable_autoencoder:
            print(f"\n[Stage 4] Autoencoder compression (to be evaluated)...")
            print(
                f"[DEBUG] Label variance: y.std={float(np.std(y_series.values)):.6f}"
            )

        # Split data (same split for all stages - use consistent random state)
        # All stages should have the same number of samples, so we can use the same split
        n_samples = len(y_series.values)
        split_idx = int(n_samples * 0.7)
        split_idx2 = int(n_samples * 0.85)

        # Create same indices for all stages
        train_indices = np.arange(split_idx)
        val_indices = np.arange(split_idx, split_idx2)
        test_indices = np.arange(split_idx2, n_samples)

        # Split y
        y_all = y_series.values
        y_train = y_all[train_indices]
        y_val = y_all[val_indices]
        y_test = y_all[test_indices]

        # Stage 1: All features
        X_train_all = X_all_scaled[train_indices]
        X_val_all = X_all_scaled[val_indices]
        X_test_all = X_all_scaled[test_indices]

        # Stage 2: IC-filtered features
        X_train_ic = X_ic_scaled[train_indices]
        X_val_ic = X_ic_scaled[val_indices]
        X_test_ic = X_ic_scaled[test_indices]

        # Stage 3: Representative features
        X_train_reps = X_reps_scaled[train_indices]
        X_val_reps = X_reps_scaled[val_indices]
        X_test_reps = X_reps_scaled[test_indices]

        # Multi-horizon training (if enabled) - will be done after all 4 stages
        multi_horizon_results = {}
        best_horizon = None
        best_horizon_metric = float("-inf")
        best_horizon_metric_name: Optional[str] = None
        fallback_horizon = None
        fallback_metric = float("-inf")
        fallback_metric_name: Optional[str] = None

        # Train and evaluate models for the selected stages
        print("\n" + "=" * 60)
        print("Training and evaluating feature sets (Stages 1-3)")
        if args.enable_autoencoder:
            print("(Stage 4: Autoencoder compressed features enabled)")
        print("=" * 60)

        # Stage 1: All features (482 -> ~470 after filtering)
        print("\n[Stage 1] Training on ALL features...")
        model_all = train_production_lightgbm(X_train_all, y_train, X_val_all,
                                              y_val)
        perf_all = evaluate_model_performance(model_all, X_test_all, y_test,
                                              "All Features")

        # Stage 2: IC-filtered features (~120)
        print("\n[Stage 2] Training on IC-filtered features...")
        model_ic = train_production_lightgbm(X_train_ic, y_train, X_val_ic,
                                             y_val)
        perf_ic = evaluate_model_performance(model_ic, X_test_ic, y_test,
                                             "IC-Filtered Features")

        # Stage 3: Representative features (60-100)
        print("\n[Stage 3] Training on Representative features...")
        model_reps = train_production_lightgbm(X_train_reps, y_train,
                                               X_val_reps, y_val)
        perf_reps = evaluate_model_performance(model_reps, X_test_reps, y_test,
                                               "Representative Features")

        feature_insights_stage3 = _derive_feature_insights(perf_all, perf_reps)

        # Default best result to Stage 3 (representative features)
        best_model = model_reps
        best_ae = None
        best_result = {
            "timestamp_start": ablation_start_ts,
            "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "duration_sec":
            (datetime.now() - ablation_start_dt).total_seconds(),
            "data_info": {
                "stage1_all_features":
                int(len(keep_all)),
                "stage2_ic_filtered":
                int(len(top_cols)),
                "stage3_representatives":
                int(len(reps)),
                "original_features_count":
                int(original_feature_count),
                "compressed_dimensions":
                int(len(reps)),
                "compression_ratio":
                float(original_feature_count / max(len(reps), 1)),
                "training_samples":
                int(len(X_train_reps)),
                "validation_samples":
                int(len(X_val_reps)),
                "test_samples":
                int(len(X_test_reps)),
            },
            "performance": {
                "stage1_all": perf_all,
                "stage2_ic": perf_ic,
                "stage3_representatives": perf_reps,
                "stage4_compressed": None,
                "selection_metric": args.selection_metric,
            },
            "insights": feature_insights_stage3,
        }

        selection_score_stage1 = compute_selection_score(
            perf_all,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        selection_score_stage2 = compute_selection_score(
            perf_ic,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        selection_score_stage3 = compute_selection_score(
            perf_reps,
            args.selection_metric,
            max_dd_threshold=float(args.max_dd_threshold),
            alpha=float(args.composite_alpha),
            beta=float(args.composite_beta),
        )
        delta_selection_stage3 = selection_score_stage3 - selection_score_stage1
        compression_ratio_stage3 = (float(original_feature_count) /
                                    float(len(reps)) if reps else None)

        best_result = {
            "timestamp_start":
            ablation_start_ts,
            "train_start_date":
            train_start_date,
            "train_end_date":
            train_end_date,
            "task_type": ("classification_binary" if args.binary_signals else
                          "classification_multiclass"),
            "data_info": {
                "stage1_all_features": int(len(keep_all)),
                "stage2_ic_filtered": int(len(top_cols)),
                "stage3_representatives": int(len(reps)),
                "original_features_count": int(original_feature_count),
                "compressed_dimensions": int(len(reps)),
                "compression_ratio": compression_ratio_stage3,
                "training_samples": int(len(X_train_reps)),
                "validation_samples": int(len(X_val_reps)),
                "test_samples": int(len(X_test_reps)),
            },
            "performance": {
                "stage1_all_features": perf_all,
                "stage2_ic_filtered": perf_ic,
                "stage3_representatives": perf_reps,
                "selection_metric": args.selection_metric,
                "selection_scores": {
                    "stage1":
                    selection_score_stage1,
                    "stage2":
                    selection_score_stage2,
                    "stage3":
                    selection_score_stage3,
                    "delta_stage3_vs_stage1":
                    delta_selection_stage3,
                    "delta_stage3_vs_stage2":
                    selection_score_stage3 - selection_score_stage2,
                },
            },
            "training_info": {
                "autoencoder_epochs":
                0,
                "autoencoder_final_loss":
                None,
                "lightgbm_stage1_iterations":
                getattr(model_all, "best_iteration", None),
                "lightgbm_stage2_iterations":
                getattr(model_ic, "best_iteration", None),
                "lightgbm_stage3_iterations":
                getattr(model_reps, "best_iteration", None),
            },
            "model_info": {
                "device_used": "cuda" if torch.cuda.is_available() else "cpu",
                "feature_names": reps[:10] if reps else feature_names[:10],
            },
            "selection": {
                "metric": args.selection_metric,
                "best_stage": feature_insights_stage3["recommended_stage"],
            },
            "insights":
            feature_insights_stage3,
        }
        best_model = model_reps
        best_ae = None
        best_dir = None
        best_dim = None
        trial_dims: list[int] = []

        if args.enable_autoencoder:
            # Stage 4: Autoencoder compressed features
            # Try multiple compression dimensions
            num_features = len(reps)

            # Determine encoding dimensions to try
            if args.auto_encoding_grid:
                # Auto-generate grid based on compression ratios
                trial_dims = generate_auto_encoding_grid(num_features)
                print(f"   Auto-generated encoding grid: {trial_dims}")
            elif args.encoding_grid:
                # Use provided grid
                try:
                    trial_dims = [
                        int(x.strip()) for x in args.encoding_grid.split(',')
                        if x.strip()
                    ]
                    trial_dims = [
                        d for d in trial_dims if d < num_features and d >= 8
                    ]
                    trial_dims = sorted(set(trial_dims), reverse=True)
                except Exception:
                    print(
                        f"   ⚠️ Invalid --encoding-grid, using auto-generation"
                    )
                    trial_dims = generate_auto_encoding_grid(num_features)
            else:
                # Default: use compression ratios
                trial_dims = []
                for ratio in [10, 20, 30, 40]:
                    dim = max(8, int(num_features / ratio))
                    if dim < num_features and dim >= 8:
                        trial_dims.append(dim)
                trial_dims.extend([32, 16, 8])
                trial_dims = sorted(set([
                    d for d in trial_dims if d < num_features
                    and d <= X_train_reps.shape[1] and d >= 8
                ]),
                                    reverse=True)
                if not trial_dims:
                    trial_dims = [max(8, int(num_features / 10)), 16, 8]
                    trial_dims = [
                        d for d in trial_dims
                        if d <= X_train_reps.shape[1] and d >= 8
                    ]

            if trial_dims:
                print(
                    f"\n[Stage 4] Training on Autoencoder compressed features (trying dims: {trial_dims}, AE type: {args.ae_type})..."
                )
            else:
                print(
                    "   ⚠️ No valid encoding dimensions found; skipping Stage 4."
                )

        grid_rows = []
        best_row = None
        stage4_best_result = None
        stage4_best_model = None
        stage4_best_ae = None

        for dim in trial_dims:
            try:
                print(f"\n  Trying encoding_dim={dim}...")

                # Create task head for this dimension if needed
                task_head_dim = None
                if args.ae_task_loss:
                    unique_y = np.unique(y_train)
                    if len(unique_y) <= 3 and np.all(
                            np.equal(np.mod(unique_y, 1), 0)):
                        num_classes = len(unique_y)
                        task_head_dim = create_task_head(
                            dim, "classification", num_classes)
                    else:
                        task_head_dim = create_task_head(dim, "regression", 1)

                # Auto-tune hyperparameters if enabled
                if args.ae_auto_tune:
                    tuned_params, tuned_trainer, tuned_ae = auto_tune_hyperparameters(
                        X_train_reps,
                        X_val_reps,
                        dim,
                        args.ae_type,
                        y_train=y_train if args.ae_task_loss else None,
                        y_val=y_val if args.ae_task_loss else None,
                        task_weight=args.task_weight
                        if args.ae_task_loss else 0.0,
                        task_head=task_head_dim,
                        n_trials=args.tune_trials,
                    )
                    ae = tuned_ae
                    trainer = tuned_trainer
                    # Train full epochs with tuned params
                    losses = trainer.train(
                        X_train_reps,
                        epochs=tuned_params["epochs"],
                        batch_size=tuned_params["batch_size"],
                        verbose=True,
                        y_train=y_train if args.ae_task_loss else None,
                    )
                else:
                    # Standard training
                    ae, trainer, losses = train_production_autoencoder(
                        X_train_reps,
                        encoding_dim=dim,
                        epochs=args.autoencoder_epochs,
                        ae_type=args.ae_type,
                        kl_weight=args.kl_weight,
                        task_weight=args.task_weight
                        if args.ae_task_loss else 0.0,
                        y_train=y_train if args.ae_task_loss else None,
                        task_head=task_head_dim,
                    )
                # Reconstruction MSE on val
                with torch.no_grad():
                    Xv = torch.as_tensor(X_val_reps,
                                         dtype=torch.float32,
                                         device=next(ae.parameters()).device)
                    out = ae(Xv)
                    if isinstance(out, tuple) or isinstance(out, list):
                        recon = out[0].cpu().numpy()
                    else:
                        recon = out.cpu().numpy()
                recon_mse = float(np.mean((recon - X_val_reps)**2))

                Z_train = trainer.transform(X_train_reps)
                Z_val = trainer.transform(X_val_reps)
                Z_test = trainer.transform(X_test_reps)

                # Standardize AE embeddings before feeding to LightGBM
                z_scaler = StandardScaler()
                Z_train = z_scaler.fit_transform(Z_train)
                Z_val = z_scaler.transform(Z_val)
                Z_test = z_scaler.transform(Z_test)

                try:
                    z_var = float(np.var(Z_train))
                    print(
                        f"    [DEBUG] AE dim={dim} | recon_mse={recon_mse:.6e} | Z_train_var={z_var:.6e}"
                    )
                except Exception:
                    pass

                model_ae = train_production_lightgbm(Z_train, y_train, Z_val,
                                                     y_val)
                perf_ae = evaluate_model_performance(model_ae, Z_test, y_test,
                                                     f"AE{dim}")

                # Model selection based on requested metric (use test-set perf)
                score_ae = _selection_score(perf_ae, args.selection_metric)
                score_reps = _selection_score(perf_reps, args.selection_metric)
                delta_r2 = perf_ae.get("r2", 0.0) - perf_reps.get("r2", 0.0)
                row = {
                    "encoding_dim": dim,
                    "reconstruction_mse": recon_mse,
                    "selection_metric": args.selection_metric,
                    "selection_score_compressed": score_ae,
                    "selection_score_reps": score_reps,
                    "r2_stage3_reps": perf_reps["r2"],
                    "r2_compressed": perf_ae["r2"],
                    "delta_r2": delta_r2,
                    "rmse_stage3_reps": perf_reps["rmse"],
                    "rmse_compressed": perf_ae["rmse"],
                }
                grid_rows.append(row)
                # Choose best by selection score (higher is better)
                if best_row is None or score_ae > best_row.get(
                        "selection_score_compressed", -1e9):
                    best_row = row
                    best_dim = dim
                    best_model = model_ae
                    best_ae = ae

                    # Build comprehensive result struct with all 4 stages
                    results = {
                        "timestamp_start":
                        ablation_start_ts,
                        "timestamp_end":
                        datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "train_start_date":
                        train_start_date,
                        "train_end_date":
                        train_end_date,
                        "task_type":
                        ("classification_binary" if args.binary_signals else
                         "classification_multiclass"),
                        "data_info": {
                            # Feature counts at each stage
                            "stage1_all_features":
                            int(len(keep_all)),
                            "stage2_ic_filtered":
                            int(len(top_cols)),
                            "stage3_representatives":
                            int(len(reps)),
                            "stage4_compressed_dim":
                            int(dim),
                            "original_features_count":
                            int(original_feature_count),
                            "compressed_dimensions":
                            int(dim),
                            "compression_ratio":
                            (float(original_feature_count) /
                             float(dim)) if dim else None,
                            "training_samples":
                            int(len(X_train_reps)),
                            "validation_samples":
                            int(len(X_val_reps)),
                            "test_samples":
                            int(len(X_test_reps)),
                        },
                        "performance": {
                            # All 4 stages performance
                            "stage1_all_features": perf_all,
                            "stage2_ic_filtered": perf_ic,
                            "stage3_representatives": perf_reps,
                            "stage4_compressed": perf_ae,
                            # Delta comparisons
                            "stage2_vs_stage1": {
                                "delta_r2": perf_ic["r2"] - perf_all["r2"],
                                "delta_rmse":
                                perf_ic["rmse"] - perf_all["rmse"],
                            },
                            "stage3_vs_stage2": {
                                "delta_r2": perf_reps["r2"] - perf_ic["r2"],
                                "delta_rmse":
                                perf_reps["rmse"] - perf_ic["rmse"],
                            },
                            "stage4_vs_stage3": {
                                "delta_r2": perf_ae["r2"] - perf_reps["r2"],
                                "delta_rmse":
                                perf_ae["rmse"] - perf_reps["rmse"],
                            },
                            # Legacy fields for compatibility
                            "original_features": perf_all,
                            "compressed_features": perf_ae,
                            "performance_change": delta_r2,
                        },
                        # Multi-horizon results (if enabled)
                        "multi_horizon_results":
                        multi_horizon_results if multi_horizon_results else {},
                        "training_info": {
                            "autoencoder_epochs":
                            int(args.autoencoder_epochs),
                            "autoencoder_final_loss":
                            (float(losses[-1])
                             if isinstance(losses, (list, tuple))
                             and len(losses) > 0 else None),
                            "lightgbm_stage1_iterations":
                            getattr(model_all, "best_iteration", None),
                            "lightgbm_stage2_iterations":
                            getattr(model_ic, "best_iteration", None),
                            "lightgbm_stage3_iterations":
                            getattr(model_reps, "best_iteration", None),
                            "lightgbm_stage4_iterations":
                            getattr(model_ae, "best_iteration", None),
                        },
                    }
                # Proxy: map compressed predictions back to reps
                y_hat_train = model_ae.predict(Z_train)

                # Handle multiclass predictions: convert probability array to class predictions
                # For multiclass, we use class predictions (0, 1, 2) as regression targets
                if y_hat_train.ndim == 2 and y_hat_train.shape[1] > 1:
                    # Multiclass: use class predictions (0=Hold, 1=Long, 2=Short)
                    y_hat_train = np.argmax(y_hat_train,
                                            axis=1).astype(np.float64)

                # Ensure y_hat_train is 1D for Ridge regression
                if y_hat_train.ndim > 1:
                    y_hat_train = y_hat_train.flatten()

                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_reps, y_hat_train)

                # Get coefficients: coef_ is always 1D for single-output Ridge
                coef = ridge.coef_
                # Ensure coef is 1D array
                if coef.ndim > 1:
                    coef = coef.flatten()

                proxy_coefs = {
                    reps[i]: float(coef[i])
                    for i in range(len(reps)) if i < len(coef)
                }
                results["proxy_weights"] = proxy_coefs
                results["grid_search"] = grid_rows
                best_result = results
                # Use training date range for directory name if available, otherwise runtime timestamps
                if ablation_dir_date_suffix:
                    DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                    best_dir = str(DIM_COMPARE_RESULTS_ROOT /
                                   ablation_dir_date_suffix)
                else:
                    # Fallback: use symbol, feature_type, and timestamps
                    DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                    best_dir = str(
                        DIM_COMPARE_RESULTS_ROOT /
                        f"{symbol_slug}_{feature_type_slug}_{results['timestamp_start']}_{results['timestamp_end']}"
                    )
            except Exception as exc:
                print(f"⚠️ Ablation ENCODING_DIM={dim} failed: {exc}")
                continue

        if best_result is None:
            raise RuntimeError("Ablation failed for all encoding dims")

        # Multi-horizon training (if enabled) - train all 4 stages for each horizon
        if horizons and len(horizons) > 1 and not df_features_original.empty:
            print(f"\n{'=' * 80}")
            print(
                f"Multi-Horizon Training: Evaluating {len(horizons)} horizons across all 4 stages"
            )
            print(f"{'=' * 80}")

            df_multi_labels = create_labels_multi_horizon(df_features_original,
                                                          horizons=horizons)

            for horizon in horizons:
                print(f"\n{'=' * 60}")
                print(f"Training all 4 stages for Horizon: {horizon} bars")
                print(f"{'=' * 60}")

                # Get labels for this horizon (3-class: 0=Hold, 1=Long, 2=Short)
                y_horizon_col = f"signal_{horizon}"
                if y_horizon_col in df_multi_labels.columns:
                    y_horizon = df_multi_labels[y_horizon_col].values
                    y_horizon = y_horizon[:len(X_raw)]

                    # Use same split indices
                    y_train_h = y_horizon[train_indices]
                    y_val_h = y_horizon[val_indices]
                    y_test_h = y_horizon[test_indices]

                    # Stage 1: All features
                    print(
                        f"\n  [Stage 1] Horizon {horizon}: Training on ALL features..."
                    )
                    model_h_all = train_production_lightgbm(
                        X_train_all, y_train_h, X_val_all, y_val_h)
                    perf_h_all = evaluate_model_performance(
                        model_h_all, X_test_all, y_test_h,
                        f"Horizon {horizon} - All Features")

                    # Stage 2: IC-filtered features
                    print(
                        f"\n  [Stage 2] Horizon {horizon}: Training on IC-filtered features..."
                    )
                    model_h_ic = train_production_lightgbm(
                        X_train_ic, y_train_h, X_val_ic, y_val_h)
                    perf_h_ic = evaluate_model_performance(
                        model_h_ic, X_test_ic, y_test_h,
                        f"Horizon {horizon} - IC-Filtered")

                    # Stage 3: Representative features
                    print(
                        f"\n  [Stage 3] Horizon {horizon}: Training on Representative features..."
                    )
                    model_h_reps = train_production_lightgbm(
                        X_train_reps, y_train_h, X_val_reps, y_val_h)
                    perf_h_reps = evaluate_model_performance(
                        model_h_reps, X_test_reps, y_test_h,
                        f"Horizon {horizon} - Representatives")

                    # Stage 4: Autoencoder compressed features (using best_dim and best_ae)
                    if best_dim is not None and best_ae is not None:
                        print(
                            f"\n  [Stage 4] Horizon {horizon}: Training on AE-compressed features (dim={best_dim})..."
                        )
                        # Transform using best autoencoder (same method as in main training loop)
                        # Get device from best_ae model
                        ae_device = next(best_ae.parameters()).device
                        with torch.no_grad():
                            X_train_reps_t = torch.as_tensor(
                                X_train_reps,
                                dtype=torch.float32,
                                device=ae_device)
                            X_val_reps_t = torch.as_tensor(X_val_reps,
                                                           dtype=torch.float32,
                                                           device=ae_device)
                            X_test_reps_t = torch.as_tensor(
                                X_test_reps,
                                dtype=torch.float32,
                                device=ae_device)

                            _, Z_train_h_t = best_ae(X_train_reps_t)
                            _, Z_val_h_t = best_ae(X_val_reps_t)
                            _, Z_test_h_t = best_ae(X_test_reps_t)

                            Z_train_h = Z_train_h_t.cpu().numpy()
                            Z_val_h = Z_val_h_t.cpu().numpy()
                            Z_test_h = Z_test_h_t.cpu().numpy()

                        # Standardize AE embeddings before feeding to LightGBM
                        z_scaler = StandardScaler()
                        Z_train_h = z_scaler.fit_transform(Z_train_h)
                        Z_val_h = z_scaler.transform(Z_val_h)
                        Z_test_h = z_scaler.transform(Z_test_h)

                        model_h_ae = train_production_lightgbm(
                            Z_train_h, y_train_h, Z_val_h, y_val_h)
                        perf_h_ae = evaluate_model_performance(
                            model_h_ae, Z_test_h, y_test_h,
                            f"Horizon {horizon} - AE-Compressed")
                    else:
                        perf_h_ae = None

                    # Store results for this horizon
                    horizon_perf = {
                        "stage1_all_features": perf_h_all,
                        "stage2_ic_filtered": perf_h_ic,
                        "stage3_representatives": perf_h_reps,
                    }
                    if args.enable_autoencoder and perf_h_ae is not None:
                        horizon_perf["stage4_compressed"] = perf_h_ae
                    feature_insight_h = _derive_feature_insights(
                        perf_h_all, perf_h_reps)
                    horizon_perf["feature_insights"] = feature_insight_h
                    metric_val_h = feature_insight_h.get("candidate_value")
                    metric_name_h = feature_insight_h.get("metric_name")
                    if (feature_insight_h.get("effective")
                            and metric_val_h is not None
                            and metric_val_h > best_horizon_metric):
                        best_horizon_metric = float(metric_val_h)
                        best_horizon_metric_name = metric_name_h
                        best_horizon = horizon
                    if metric_val_h is not None and metric_val_h > fallback_metric:
                        fallback_metric = float(metric_val_h)
                        fallback_metric_name = metric_name_h
                        fallback_horizon = horizon
                    multi_horizon_results[f"horizon_{horizon}"] = horizon_perf

                    print(f"\n  ✅ Horizon {horizon} Complete:")
                    print(
                        f"     Stage 1 (All):      R²={perf_h_all['r2']:.4f}, RMSE={perf_h_all['rmse']:.6f}"
                    )
                    print(
                        f"     Stage 2 (IC):       R²={perf_h_ic['r2']:.4f}, RMSE={perf_h_ic['rmse']:.6f}"
                    )
                    print(
                        f"     Stage 3 (Reps):     R²={perf_h_reps['r2']:.4f}, RMSE={perf_h_reps['rmse']:.6f}"
                    )
                    if args.enable_autoencoder and perf_h_ae is not None:
                        print(
                            f"     Stage 4 (AE):       R²={perf_h_ae['r2']:.4f}, RMSE={perf_h_ae['rmse']:.6f}"
                        )
                else:
                    print(
                        f"   ⚠️  Label column {y_horizon_col} not found for horizon {horizon}"
                    )

        # Add multi-horizon results to best_result
        if multi_horizon_results:
            best_result["multi_horizon_results"] = multi_horizon_results
            insights_ref = best_result.setdefault("insights", {})
            horizon_choice = best_horizon
            horizon_metric = best_horizon_metric
            horizon_metric_name = best_horizon_metric_name
            horizon_effective = True
            if horizon_choice is None and fallback_horizon is not None:
                horizon_choice = fallback_horizon
                horizon_metric = fallback_metric
                horizon_metric_name = fallback_metric_name
                horizon_effective = False
            if horizon_choice is not None:
                insights_ref.update({
                    "recommended_horizon":
                    int(horizon_choice),
                    "recommended_horizon_metric":
                    float(horizon_metric)
                    if horizon_metric is not None else None,
                    "recommended_horizon_metric_name":
                    horizon_metric_name,
                    "recommended_horizon_effective":
                    horizon_effective,
                })

        # finalize end timestamp using actual ablation end
        ablation_end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # update results end and duration
        if best_result is not None:
            best_result["timestamp_end"] = ablation_end_ts
            try:
                start_dt_parsed = datetime.strptime(ablation_start_ts,
                                                    "%Y%m%d_%H%M%S")
                duration_sec = (datetime.now() -
                                start_dt_parsed).total_seconds()
                best_result["duration_sec"] = duration_sec
            except Exception:
                pass
            # rebuild dir using training date range if available, otherwise runtime timestamps
            if ablation_dir_date_suffix:
                DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                best_dir = str(DIM_COMPARE_RESULTS_ROOT /
                               ablation_dir_date_suffix)
            else:
                # Fallback: use symbol, feature_type, and timestamps
                DIM_COMPARE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                best_dir = str(
                    DIM_COMPARE_RESULTS_ROOT /
                    f"{symbol_slug}_{feature_type_slug}_{best_result['timestamp_start']}_{best_result['timestamp_end']}"
                )
        os.makedirs(best_dir, exist_ok=True)

        # Save representative features list (Stage 3) - after best_dir is set
        if reps:
            reps_path = os.path.join(best_dir, "representative_factors.json")
            with open(reps_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "representative_factors":
                        reps,
                        "count":
                        len(reps),
                        "stage":
                        "Stage 3: Correlation-based representative selection",
                        "description":
                        "Features selected by greedy correlation filtering (threshold=0.9)",
                        "effective":
                        feature_insights_stage3.get("effective", False),
                    },
                    f,
                    indent=2)
            print(f"   💾 Representative factors saved to: {reps_path}")

            # Also save in top_factors format for compatibility with train_model
            top_factors_path = os.path.join(best_dir, "top_factors.json")
            with open(top_factors_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "top_factors": [{
                            "name": factor
                        } for factor in reps],
                        "count":
                        len(reps),
                        "source":
                        "dim-compare",
                        "stage":
                        "Stage 3: Representative features",
                        "effective":
                        feature_insights_stage3.get("effective", False),
                    },
                    f,
                    indent=2)
            print(
                f"   💾 Top factors (compatible format) saved to: {top_factors_path}"
            )
            best_result.setdefault("data_info",
                                   {})["representatives_path"] = reps_path
            best_result["data_info"]["top_factors_path"] = top_factors_path

            shap_dir_path = None
            if args.shap_analysis:
                shap_dir_path = _generate_shap_outputs(
                    model_reps,
                    X_train_reps,
                    reps,
                    best_dir,
                    prefix="stage3_representatives",
                )
                if shap_dir_path:
                    best_result.setdefault(
                        "explainability",
                        {})["stage3_shap_dir"] = shap_dir_path

        # Save best Autoencoder model - after best_dir is set
        if best_ae is not None:
            ae_path = os.path.join(best_dir, "production_autoencoder.pth")
            torch.save(best_ae.state_dict(), ae_path)
            print(f"   💾 Best Autoencoder saved to: {ae_path}")

        # Ensure JSON-serializable (e.g., convert any numpy types)
        def _to_py(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, )):
                return float(o)
            if isinstance(o, (np.integer, )):
                return int(o)
            return o

        with open(f"{best_dir}/production_results.json", "w") as f:
            json.dump(best_result, f, indent=2, default=_to_py)

        # Generate report filename with symbol, feature_type, and time range
        def _format_date_for_filename(date_str):
            if not date_str:
                return ""
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_part = date_str.split("T")[0]
                        dt = datetime.strptime(date_part, "%Y-%m-%d")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                return ""
            except Exception:
                if isinstance(date_str, str) and len(date_str) >= 10:
                    try:
                        return date_str[:10].replace("-", "")
                    except:
                        return ""
                return ""

        train_start_str = _format_date_for_filename(
            args.train_start) if args.train_start else ""
        train_end_str = _format_date_for_filename(
            args.train_end) if args.train_end else ""

        # Build report filename
        if train_start_str and train_end_str:
            report_filename = f"{symbol_slug}_{feature_type_slug}_{train_start_str}_{train_end_str}_dimensionality_report.html"
        else:
            # Fallback to timestamps
            report_filename = f"{symbol_slug}_{feature_type_slug}_{ablation_start_ts}_dimensionality_report.html"

        default_report_path = os.path.join(best_dir, report_filename)
        write_html_report(best_result, default_report_path)
        print(f"📝 HTML report saved to: {default_report_path}")
        # optional export
        if args.export_model:
            try:
                os.makedirs(os.path.dirname(args.export_model), exist_ok=True)
                src_model = os.path.join(best_dir, "production_model.pkl")
                if os.path.exists(src_model):
                    import shutil as _sh
                    _sh.copy2(src_model, args.export_model)
                    print(f"💾 Exported best model to: {args.export_model}")
            except Exception as _exc:
                print(f"⚠️ Failed to export model: {_exc}")

        return best_result, best_model, best_ae, best_dir

    if grid_dims:
        best = None
        grid_rows = []
        for dim in grid_dims:
            try:
                trial_results, trial_model, trial_ae, trial_dir = run_dimensionality_comparison(
                    data_path=args.data_path,
                    symbol=args.symbol,
                    encoding_dim=dim,
                    autoencoder_epochs=args.autoencoder_epochs,
                    train_start=args.train_start,
                    train_end=args.train_end,
                    feature_type=args.feature_type,
                    top_k=args.top_k,
                    shap_analysis=args.shap_analysis,
                )
                perf = trial_results.get('performance', {})
                orig = perf.get('original_features', {})
                comp = perf.get('compressed_features', {})
                delta = perf.get('performance_change')
                grid_rows.append({
                    'encoding_dim': dim,
                    'r2_original': orig.get('r2'),
                    'r2_compressed': comp.get('r2'),
                    'delta_r2': delta,
                    'rmse_original': orig.get('rmse'),
                    'rmse_compressed': comp.get('rmse'),
                    'results_dir': trial_dir,
                })
                if best is None or (delta is not None
                                    and best['delta_r2'] is not None
                                    and delta > best['delta_r2']):
                    best = grid_rows[-1]
                    results = trial_results
                    model = trial_model
                    autoencoder = trial_ae
                    results_dir = trial_dir
            except Exception as exc:
                print(f"⚠️ Trial with ENCODING_DIM={dim} failed: {exc}")
                continue
        # Attach grid rows to best results and write report
        if 'grid_search' not in results:
            results['grid_search'] = grid_rows
    else:
        results, model, autoencoder, results_dir = run_dimensionality_comparison(
            data_path=args.data_path,
            symbol=args.symbol,
            encoding_dim=args.encoding_dim,
            autoencoder_epochs=args.autoencoder_epochs,
            train_start=args.train_start,
            train_end=args.train_end,
            feature_type=args.feature_type,
            top_k=args.top_k,
            shap_analysis=args.shap_analysis,
        )

    # Record Top-K hint if provided
    if args.top_k is not None:
        results.setdefault("training_info", {})["top_k"] = args.top_k

    # Always write a report into the results directory with symbol, feature_type, and time range
    try:
        # Generate report filename with symbol, feature_type, and time range
        def _format_date_for_filename(date_str):
            if not date_str:
                return ""
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_part = date_str.split("T")[0]
                        dt = datetime.strptime(date_part, "%Y-%m-%d")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                return ""
            except Exception:
                if isinstance(date_str, str) and len(date_str) >= 10:
                    try:
                        return date_str[:10].replace("-", "")
                    except:
                        return ""
                return ""

        train_start_str = _format_date_for_filename(
            args.train_start) if args.train_start else ""
        train_end_str = _format_date_for_filename(
            args.train_end) if args.train_end else ""

        # Build report filename
        if train_start_str and train_end_str:
            report_filename = f"{symbol_slug}_{feature_type_slug}_{train_start_str}_{train_end_str}_dimensionality_report.html"
        else:
            # Fallback: extract from results_dir or use timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{symbol_slug}_{feature_type_slug}_{timestamp_str}_dimensionality_report.html"

        default_report_path = os.path.join(results_dir, report_filename)
        write_html_report(results, default_report_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to write default HTML report: {exc}")

    # Optionally write an extra copy to a user-specified path
    if args.report_html:
        try:
            write_html_report(results, args.report_html)
        except Exception as exc:  # noqa: BLE001
            print(
                f"⚠️ Failed to write HTML report to {args.report_html}: {exc}")

    # Optional export in non-ablation paths
    if args.export_model:
        try:
            os.makedirs(os.path.dirname(args.export_model), exist_ok=True)
            src_model = os.path.join(results_dir, "production_model.pkl")
            if os.path.exists(src_model):
                import shutil as _sh
                _sh.copy2(src_model, args.export_model)
                print(f"💾 Exported best model to: {args.export_model}")
        except Exception as _exc:
            print(f"⚠️ Failed to export model: {_exc}")
    return results, model, autoencoder, results_dir


if __name__ == "__main__":
    try:
        results, model, autoencoder, results_dir = main()
        print("\n✅ Production training completed successfully!")
        cr = results.get('data_info', {}).get(
            'compression_dim', None) or results.get('data_info', {}).get(
                'compression_ratio', None)
        if cr is not None:
            try:
                print(f"📊 Final compression ratio: {float(cr):.1f}x")
            except Exception:
                pass
        pc = results.get('performance', {}).get('performance_change', None)
        if pc is not None:
            print(f"📈 Performance change: {pc:.4f}")
        print(f"💾 Results directory: {results_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Production training failed: {exc}")
        import traceback

        traceback.print_exc()
        raise
