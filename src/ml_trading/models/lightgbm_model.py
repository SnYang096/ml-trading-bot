"""LightGBM model implementation for classification, regression, and quantile models."""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from numbers import Number
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import optuna
from ml_trading.config.settings import DEFAULT_LGBM_PARAMS, USE_GPU, GPU_LGBM_PARAMS


class GroupTimeSeriesSplit:
    """Group-aware time series cross-validator with optional purge window.

    Highlights
    ---------
    - Preserves chronological order (builds on :class:`TimeSeriesSplit`).
    - Supports a ``purge_gap`` between training and validation windows to
      mitigate temporal leakage.
    - When ``groups`` are provided (e.g., asset symbols) and
      ``drop_same_group=True``, samples belonging to validation groups are
      removed from the training slice for that fold to avoid cross-group
      leakage.

    Notes
    -----
    - Input samples must already be sorted in chronological order.
    - ``groups`` must align with the sample axis.
    """

    def __init__(self,
                 n_splits: int = 5,
                 purge_gap: int | float = 0,
                 drop_same_group: bool = True):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 for TimeSeriesSplit")
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.drop_same_group = drop_same_group

    def split(self,
              X: pd.DataFrame | np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        n_samples = len(X)
        if n_samples <= self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits")

        base_split = TimeSeriesSplit(n_splits=self.n_splits)
        indices = np.arange(n_samples)
        groups_arr = None
        if groups is not None:
            groups_arr = np.asarray(groups)
            if len(groups_arr) != n_samples:
                raise ValueError(
                    "Length of groups must match number of samples in X")

        for fold, (train_idx,
                   test_idx) in enumerate(base_split.split(indices)):
            gap = 0
            if isinstance(self.purge_gap, float):
                gap = int(len(test_idx) * max(0.0, min(1.0, self.purge_gap)))
            elif isinstance(self.purge_gap, int):
                gap = max(0, self.purge_gap)

            if gap > 0:
                cutoff = max(0, test_idx[0] - gap)
                train_idx = train_idx[train_idx < cutoff]

            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    f"Fold {fold + 1}: insufficient samples after purging gap")

            if groups_arr is not None and self.drop_same_group:
                val_groups = set(groups_arr[test_idx])
                mask = np.array(
                    [groups_arr[idx] not in val_groups for idx in train_idx],
                    dtype=bool)
                train_idx = train_idx[mask]
                if len(train_idx) == 0:
                    raise ValueError(
                        f"Fold {fold + 1}: all training samples removed after"
                        " enforcing group separation")

            yield train_idx, test_idx


class LightGBMModel:
    """LightGBM model for classification, regression, and quantile estimation."""

    def __init__(
        self,
        model_type: str = "regression",
        params: Optional[Dict] = None,
        use_gpu: Optional[bool] = None,
        quantile_alpha: Optional[float] = None,
    ):
        """
        Initialize the LightGBM model.

        Args:
            model_type: "classification", "regression", or "quantile"
            params: LightGBM parameters (if None, use DEFAULT_LGBM_PARAMS)
            use_gpu: Enable GPU acceleration (if None, use USE_GPU from config)
            quantile_alpha: Alpha value for quantile regression (e.g., 0.1, 0.5, 0.9)
        """
        if model_type not in {"classification", "regression", "quantile"}:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Use 'classification', 'regression', or 'quantile'."
            )
        self.model_type = model_type
        self.use_gpu = use_gpu if use_gpu is not None else USE_GPU
        self.quantile_alpha = quantile_alpha

        # Start with default parameters
        self.params = params if params is not None else DEFAULT_LGBM_PARAMS.copy(
        )

        # Add GPU parameters if enabled
        if self.use_gpu:
            print("🚀 GPU acceleration enabled for LightGBM training")
            self.params.update(GPU_LGBM_PARAMS)

        self.model = None
        self.is_trained = False

        # Adjust parameters based on model type
        if model_type == "classification":
            # Binary classification (0=Down, 1=Up)
            self.params["objective"] = "binary"
            self.params["metric"] = "binary_logloss"
        elif model_type == "quantile":
            # Quantile regression (for q10, q50, q90 models)
            if quantile_alpha is None:
                raise ValueError(
                    "quantile_alpha must be provided for quantile regression")
            self.params["objective"] = "quantile"
            self.params["alpha"] = quantile_alpha
            self.params["metric"] = "quantile"
        else:
            # Regression for predicting continuous returns (e.g., volatility)
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project to numeric columns and sanitize infinities."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        features = X[numeric_columns].copy()
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    def prepare_data(self, X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix/target vector for training."""
        X_features = self._prepare_features(X)

        y_series = y.copy()
        y_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_indices = ~y_series.isna()
        X_clean = X_features.loc[valid_indices]
        y_clean = y_series.loc[valid_indices]

        return X_clean, y_clean

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        use_time_series_cv: bool = True,
        sample_weight: Optional[np.ndarray] = None,
        preprocess_fn: Optional[callable] = None,
        preprocess_kwargs: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
        auto_tune_params: bool = False,
        tune_trials: int = 20,
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], Optional[Dict]]:
        """
        Train the LightGBM model using TimeSeriesSplit for proper time series validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of time series splits (default: 5)
            use_time_series_cv: If True, use TimeSeriesSplit; if False, use train_test_split (default: True)
            sample_weight: Optional sample weights
            preprocess_fn: Optional preprocessing function called within CV loop.
                          Signature: (y_train, y_test, **kwargs) -> (y_train_processed, y_test_processed, stats_dict)
            preprocess_kwargs: Optional kwargs passed to preprocess_fn
            groups: Optional array of group labels (e.g., symbol). Used for
                   logging/diagnostics and to enable group-aware time-series
                   splits without breaking chronological order.
            auto_tune_params: If True, run Q50-aware hyperparameter search prior to training.
            tune_trials: Number of trials for hyperparameter tuning when enabled.
            verbose: If False, suppress training progress logs.
        
        Returns:
            Tuple of (training_metrics, preprocess_params)
            - training_metrics: Training metrics dictionary
            - preprocess_params: Aggregated preprocessing parameters for deployment, or None
        """
        log = print if verbose else (lambda *args, **kwargs: None)

        def aggregate_preprocess_stats(
                stats_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not stats_list:
                return None

            def _aggregate(items: List[Any]) -> Any:
                valid_items = [item for item in items if item is not None]
                if not valid_items:
                    return None
                if all(isinstance(item, dict) for item in valid_items):
                    keys: set[str] = set()
                    for item in valid_items:
                        keys.update(item.keys())
                    return {
                        key:
                        _aggregate(
                            [item[key] for item in valid_items if key in item])
                        for key in keys
                    }
                numeric_values: List[float] = []
                for item in valid_items:
                    if isinstance(item, Number):
                        if np.isfinite(item):
                            numeric_values.append(float(item))
                    elif isinstance(item, (np.floating, np.integer)):
                        if np.isfinite(item):
                            numeric_values.append(float(item))
                    else:
                        return valid_items[0]
                if numeric_values:
                    return float(np.median(numeric_values))
                return valid_items[0]

            aggregated = _aggregate(stats_list)
            return aggregated if aggregated is not None else None

        # Auto-tune parameters if requested (only for quantile models)
        if auto_tune_params:
            if self.model_type == "quantile" and self.quantile_alpha == 0.5:
                log("  🔍 Auto-tuning hyperparameters (Q50 constraint-aware)..."
                    )
                best_params = self.optimize_hyperparameters_for_q50_constraint(
                    X,
                    y,
                    n_trials=tune_trials,
                    n_splits=min(n_splits, 3),
                    groups=groups)
                if best_params:
                    self.params.update(best_params)
                    log(f"  ✅ Updated parameters: {best_params}")
            elif self.model_type == "classification":
                log("  🔍 Auto-tuning hyperparameters (classification)...")
                best_params = self.optimize_hyperparameters(
                    X, y, n_trials=tune_trials)
                if best_params:
                    self.params.update(best_params)
                    log(f"  ✅ Updated parameters: {best_params}")
            else:
                log("  ⚠️  Auto-tuning not implemented for this model type")

        # Prepare data (basic cleaning only, no target transformation)
        X_clean, y_clean = self.prepare_data(X, y)

        # Get valid indices from prepare_data (where y is not NaN)
        # This is needed to align groups and sample_weight with cleaned data
        # Note: prepare_data removes rows where y is NaN, so we need to filter groups accordingly
        y_series = y.copy()
        if isinstance(y_series, pd.Series):
            y_series = y_series.replace([np.inf, -np.inf], np.nan)
            valid_indices = ~y_series.isna(
            ).values  # Convert to numpy array for indexing
        else:
            y_series = np.array(y_series)
            y_series = np.where(np.isinf(y_series), np.nan, y_series)
            valid_indices = ~np.isnan(y_series)

        # Align groups with cleaned data if provided
        groups_clean = None
        if groups is not None:
            if len(groups) == len(y):
                # Groups based on original data, need to filter
                groups_clean = np.array(groups)[valid_indices]
            elif len(groups) == len(y_clean):
                # Groups already aligned with cleaned data
                groups_clean = groups
            else:
                log(f"  Warning: groups length ({len(groups)}) doesn't match y ({len(y)}) or y_clean ({len(y_clean)}), ignoring groups"
                    )
                groups_clean = None

        # Align sample weights with cleaned data if provided
        sample_weight_clean = None
        if sample_weight is not None:
            # Align sample_weight with valid indices
            if len(sample_weight) == len(y):
                sample_weight_clean = sample_weight[valid_indices]
            elif len(sample_weight) == len(y_clean):
                # Already aligned with cleaned data
                sample_weight_clean = sample_weight
            else:
                log(f"  Warning: sample_weight length ({len(sample_weight)}) doesn't match y ({len(y)}) or y_clean ({len(y_clean)}), ignoring weights"
                    )
                sample_weight_clean = None

        if use_time_series_cv:
            # ✅ 使用时间序列交叉验证 - 避免未来信息泄露
            if groups_clean is not None:
                log(f"  🔒 使用 GroupTimeSeriesSplit（保持时间顺序 + 可选 purge gap）")
                cv = GroupTimeSeriesSplit(n_splits=n_splits)
            else:
                log(f"  Using TimeSeriesSplit with {n_splits} folds (prevents look-ahead bias)"
                    )
                cv = TimeSeriesSplit(n_splits=n_splits)

            metrics_list = []
            best_model = None
            # For classification, maximize F1; for regression/quantile, minimize loss
            best_metric = -np.inf if self.model_type == "classification" else np.inf
            preprocess_stats_all: List[Dict[str, Any]] = []

            # Split data using the appropriate CV strategy
            cv_kwargs = {}
            if groups_clean is not None:
                cv_kwargs["groups"] = groups_clean
            cv_splits = cv.split(X_clean, **cv_kwargs)

            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                # 🚀 OPTIMIZATION: Use .values to get numpy arrays directly, avoiding DataFrame overhead
                # This reduces memory usage, especially for large datasets
                # We'll convert back to DataFrame only when needed for feature cleaning
                X_train_raw = pd.DataFrame(
                    X_clean.values[train_idx],
                    index=X_clean.index[train_idx],
                    columns=X_clean.columns,
                    copy=False  # Don't copy the underlying data
                )
                X_val_raw = pd.DataFrame(X_clean.values[val_idx],
                                         index=X_clean.index[val_idx],
                                         columns=X_clean.columns,
                                         copy=False)
                y_train_raw = pd.Series(y_clean.values[train_idx],
                                        index=y_clean.index[train_idx],
                                        copy=False)
                y_val_raw = pd.Series(y_clean.values[val_idx],
                                      index=y_clean.index[val_idx],
                                      copy=False)

                # Apply feature cleaning WITHIN CV loop (prevents lookahead bias)
                # All statistics computed ONLY from training data
                from ml_trading.pipeline.training.preprocessing import clean_features_train_test
                X_train, X_val, feature_clean_stats = clean_features_train_test(
                    X_train_raw, X_val_raw, k=4.0)
                if fold == 0 and feature_clean_stats.get(
                        "n_features_cleaned", 0) > 0:
                    log(f"    Feature cleaning (fold {fold+1}): {feature_clean_stats['n_features_cleaned']} features cleaned"
                        )

                # Apply target preprocessing WITHIN CV loop (prevents lookahead bias)
                # All statistics computed ONLY from training data
                if preprocess_fn is not None:
                    preprocess_kwargs_fold = preprocess_kwargs.copy(
                    ) if preprocess_kwargs else {}
                    # Add fold index for logging if needed
                    preprocess_kwargs_fold['fold'] = fold
                    y_train, y_val, preprocess_stats = preprocess_fn(
                        y_train_raw, y_val_raw, **preprocess_kwargs_fold)
                    if fold == 0 and verbose:
                        log(f"    Target preprocessing stats (fold {fold+1}): {preprocess_stats}"
                            )
                    preprocess_stats_all.append(preprocess_stats)
                else:
                    y_train, y_val = y_train_raw, y_val_raw

                log(f"  Fold {fold+1}/{n_splits}: Train [{train_idx[0]}:{train_idx[-1]}], Val [{val_idx[0]}:{val_idx[-1]}]"
                    )

                # Create LightGBM datasets with sample weights if provided
                if sample_weight_clean is not None:
                    train_weight = sample_weight_clean[train_idx]
                    val_weight = sample_weight_clean[val_idx]
                    train_data = lgb.Dataset(X_train,
                                             label=y_train,
                                             weight=train_weight)
                    val_data = lgb.Dataset(X_val,
                                           label=y_val,
                                           weight=val_weight,
                                           reference=train_data)
                else:
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val,
                                           label=y_val,
                                           reference=train_data)

                # Train model
                # Get num_boost_round from params if available, otherwise use default
                train_params = self.params.copy()
                num_boost_round = train_params.pop("num_boost_round", 1000)
                # Also check for n_estimators (for compatibility)
                if "n_estimators" in train_params:
                    num_boost_round = train_params.pop("n_estimators")

                model = lgb.train(
                    train_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=num_boost_round,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(0),
                    ],
                )

                # Evaluate on this fold
                if self.model_type == "classification":
                    # For classification, get probabilities
                    y_pred_proba = model.predict_proba(
                        X_val)[:, 1]  # Probability of positive class [0, 1]
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

                    # Calculate classification metrics
                    fold_accuracy = accuracy_score(y_val, y_pred_binary)
                    fold_f1 = f1_score(y_val,
                                       y_pred_binary,
                                       average='binary',
                                       zero_division=0)

                    # AUC (requires probabilities)
                    try:
                        fold_auc = roc_auc_score(y_val, y_pred_proba)
                    except ValueError:
                        # Handle case where only one class present
                        fold_auc = 0.5

                    # Confusion matrix
                    cm = confusion_matrix(y_val, y_pred_binary)
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                    elif cm.size == 1:
                        # Only one class present
                        y_val_first = y_val.iloc[0] if isinstance(
                            y_val, pd.Series) else y_val[0]
                        if y_val_first == 0:
                            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
                        else:
                            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
                    else:
                        tn, fp, fn, tp = 0, 0, 0, 0

                    metrics_list.append({
                        "fold":
                        fold + 1,
                        "accuracy":
                        fold_accuracy,
                        "f1":
                        fold_f1,
                        "auc":
                        fold_auc,
                        "confusion_matrix": {
                            "tn": int(tn),
                            "fp": int(fp),
                            "fn": int(fn),
                            "tp": int(tp),
                        },
                        "precision":
                        tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                        "recall":
                        tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    })

                    log(f"    Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}, AUC: {fold_auc:.4f}"
                        )
                    log(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}"
                        )

                    # Keep best model based on F1 score
                    if fold_f1 > best_metric:
                        best_metric = fold_f1
                        best_model = model
                elif self.model_type == "quantile":
                    if verbose and fold == 0:
                        log(f"    DEBUG: y_val range: [{np.nanmin(y_val):.6f}, {np.nanmax(y_val):.6f}], mean={np.nanmean(y_val):.6f}"
                            )
                        log(f"    DEBUG: y_pred range: [{np.nanmin(y_pred):.6f}, {np.nanmax(y_pred):.6f}], mean={np.nanmean(y_pred):.6f}"
                            )

                    # Use explicit np.where for pinball loss to avoid style warnings
                    error_fold = y_val - y_pred
                    quantile_loss = np.mean(
                        np.where(error_fold >= 0,
                                 self.quantile_alpha * error_fold,
                                 (1.0 - self.quantile_alpha) * (-error_fold)))
                    metrics_list.append({
                        "fold": fold + 1,
                        "quantile_loss": quantile_loss
                    })
                    log(f"    Quantile Loss (alpha={self.quantile_alpha}): {quantile_loss:.6f}"
                        )

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model
                else:
                    fold_mse = mean_squared_error(y_val, y_pred)
                    fold_rmse = np.sqrt(fold_mse)
                    metrics_list.append({
                        "fold": fold + 1,
                        "mse": fold_mse,
                        "rmse": fold_rmse
                    })
                    log(f"    MSE: {fold_mse:.6f}, RMSE: {fold_rmse:.6f}")

                    if fold_rmse < best_metric:
                        best_metric = fold_rmse
                        best_model = model

            # Store the best model
            self.model = best_model

            preprocess_params = aggregate_preprocess_stats(
                preprocess_stats_all)
            if preprocess_params is not None and isinstance(
                    preprocess_params, dict):
                preprocess_params.setdefault(
                    "note",
                    "Parameters aggregated (median) across CV folds for deployment consistency.",
                )

            # Return average metrics across folds
            if self.model_type == "classification":
                avg_accuracy = np.mean([m["accuracy"] for m in metrics_list])
                std_accuracy = np.std([m["accuracy"] for m in metrics_list])
                avg_f1 = np.mean([m["f1"] for m in metrics_list])
                std_f1 = np.std([m["f1"] for m in metrics_list])
                avg_auc = np.mean([m["auc"] for m in metrics_list])
                std_auc = np.std([m["auc"] for m in metrics_list])
                avg_precision = np.mean([m["precision"] for m in metrics_list])
                avg_recall = np.mean([m["recall"] for m in metrics_list])

                # Aggregate confusion matrix across folds
                total_tn = sum(m["confusion_matrix"]["tn"]
                               for m in metrics_list)
                total_fp = sum(m["confusion_matrix"]["fp"]
                               for m in metrics_list)
                total_fn = sum(m["confusion_matrix"]["fn"]
                               for m in metrics_list)
                total_tp = sum(m["confusion_matrix"]["tp"]
                               for m in metrics_list)

                metrics = {
                    "cv_accuracy": avg_accuracy,
                    "cv_accuracy_std": std_accuracy,
                    "cv_f1": avg_f1,
                    "cv_f1_std": std_f1,
                    "cv_auc": avg_auc,
                    "cv_auc_std": std_auc,
                    "cv_precision": avg_precision,
                    "cv_recall": avg_recall,
                    "confusion_matrix": {
                        "tn": int(total_tn),
                        "fp": int(total_fp),
                        "fn": int(total_fn),
                        "tp": int(total_tp),
                    },
                    "fold_details": metrics_list,
                }
                log(f"  Average CV Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}"
                    )
                log(f"  Average CV F1: {avg_f1:.4f} ± {std_f1:.4f}")
                log(f"  Average CV AUC: {avg_auc:.4f} ± {std_auc:.4f}")
                log(f"  Total Confusion Matrix: TN={total_tn}, FP={total_fp}, FN={total_fn}, TP={total_tp}"
                    )
                return metrics, preprocess_params
            elif self.model_type == "quantile":
                avg_quantile_loss = np.mean(
                    [m["quantile_loss"] for m in metrics_list])
                std_quantile_loss = np.std(
                    [m["quantile_loss"] for m in metrics_list])
                metrics = {
                    "cv_quantile_loss": avg_quantile_loss,
                    "cv_quantile_loss_std": std_quantile_loss,
                    "quantile_alpha": self.quantile_alpha,
                    "fold_details": metrics_list,
                }
                log(f"  Average CV Quantile Loss (alpha={self.quantile_alpha}): {avg_quantile_loss:.6f} ± {std_quantile_loss:.6f}"
                    )
                return metrics, preprocess_params
            else:  # regression
                avg_mse = np.mean([m["mse"] for m in metrics_list])
                avg_rmse = np.mean([m["rmse"] for m in metrics_list])
                std_mse = np.std([m["mse"] for m in metrics_list])
                metrics = {
                    "cv_mse": avg_mse,
                    "cv_rmse": avg_rmse,
                    "cv_mse_std": std_mse,
                    "fold_details": metrics_list,
                }
                log(f"  Average CV MSE: {avg_mse:.6f} ± {std_mse:.6f}")
                return metrics, preprocess_params
        else:
            # ⚠️ 传统方法（不推荐用于时间序列）- 仅用于对比
            log(f"  WARNING: Using train_test_split (random split - not recommended for time series!)"
                )
            X_train, X_val, y_train, y_val = train_test_split(X_clean,
                                                              y_clean,
                                                              test_size=0.2,
                                                              random_state=42)

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),
                ],
            )

            # Evaluate model
            y_pred = self.model.predict(X_val)

            if self.model_type == "classification":
                # Binary classification: convert probabilities to binary predictions
                y_pred_proba = y_pred  # Probabilities [0, 1]
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)

                # Calculate classification metrics
                accuracy = accuracy_score(y_val, y_pred_binary)
                f1 = f1_score(y_val,
                              y_pred_binary,
                              average='binary',
                              zero_division=0)

                # AUC (requires probabilities)
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                except ValueError:
                    auc = 0.5

                # Confusion matrix
                cm = confusion_matrix(y_val, y_pred_binary)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                metrics = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "auc": auc,
                    "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    },
                }
            elif self.model_type == "quantile":
                error = y_val - y_pred
                quantile_loss = np.mean(
                    np.where(error >= 0, self.quantile_alpha * error,
                             (1.0 - self.quantile_alpha) * (-error)))
                metrics = {
                    "quantile_loss": quantile_loss,
                    "quantile_alpha": self.quantile_alpha
                }
            else:  # regression
                metrics = {
                    "mse": mean_squared_error(y_val, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                }

        self.is_trained = True
        return metrics, None  # No preprocessing params for random split

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Predictions:
                - For classification: probabilities [0, 1] (probability of positive class)
                - For regression/quantile: predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_clean = self._prepare_features(X)

        if self.model_type == "classification":
            # For classification, return probabilities (probability of positive class)
            # predict_proba returns [prob_class_0, prob_class_1], we want prob_class_1
            predictions = self.model.predict_proba(X_clean)[:, 1]
        else:
            # For regression/quantile, return predicted values
            predictions = self.model.predict(X_clean)

        return predictions

    def optimize_hyperparameters(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials

        Returns:
            Best parameters
        """
        # Prepare data
        X_clean, y_clean = self.prepare_data(X, y)

        max_splits = max(2, min(5, len(X_clean) - 1))
        if max_splits >= len(X_clean):
            max_splits = len(X_clean) - 1
        if max_splits < 2:
            raise ValueError(
                "Not enough samples for time-series CV during hyperparameter optimization"
            )

        cv_splits = list(TimeSeriesSplit(n_splits=max_splits).split(X_clean))

        def objective(trial):
            # Suggest hyperparameters
            params = {
                "objective":
                self.params["objective"],
                "metric":
                self.params["metric"],
                "boosting_type":
                trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                "num_leaves":
                trial.suggest_int("num_leaves", 10, 1000),
                "learning_rate":
                trial.suggest_float("learning_rate", 0.001, 0.3),
                "feature_fraction":
                trial.suggest_float("feature_fraction", 0.1, 1.0),
                "bagging_fraction":
                trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq":
                trial.suggest_int("bagging_freq", 0, 10),
                "min_child_samples":
                trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight":
                trial.suggest_float("min_child_weight", 1e-5, 1.0),
                "lambda_l1":
                trial.suggest_float("lambda_l1", 1e-8, 10.0),
                "lambda_l2":
                trial.suggest_float("lambda_l2", 1e-8, 10.0),
                "verbose":
                -1,
            }

            # Add GPU parameters if GPU is enabled
            if self.use_gpu:
                params.update(GPU_LGBM_PARAMS)

            losses: List[float] = []
            for train_idx, val_idx in cv_splits:
                X_train_fold = X_clean.iloc[train_idx]
                X_val_fold = X_clean.iloc[val_idx]
                y_train_fold = y_clean.iloc[train_idx]
                y_val_fold = y_clean.iloc[val_idx]

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold,
                                       label=y_val_fold,
                                       reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20),
                        lgb.log_evaluation(0),
                    ],
                )

                if self.model_type == "classification":
                    # For classification, get probabilities
                    y_pred_proba = model.predict_proba(
                        X_val_fold)[:,
                                    1]  # Probability of positive class [0, 1]
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                    # Binary classification: use F1 score as optimization target
                    score = f1_score(y_val_fold,
                                     y_pred_binary,
                                     average='binary',
                                     zero_division=0)
                    losses.append(
                        -float(score))  # Negative because we minimize
                elif self.model_type == "quantile":
                    # For quantile, get predicted values
                    y_pred = model.predict(X_val_fold)
                    error = y_val_fold - y_pred
                    loss = np.mean(
                        np.where(error >= 0, self.quantile_alpha * error,
                                 (1.0 - self.quantile_alpha) * (-error)))
                    losses.append(float(loss))
                else:  # regression
                    # For regression, get predicted values
                    y_pred = model.predict(X_val_fold)
                    loss = mean_squared_error(y_val_fold, y_pred)
                    losses.append(float(loss))

            mean_loss = float(np.mean(losses)) if losses else np.inf
            if not np.isfinite(mean_loss):
                return -np.inf
            return -mean_loss  # Negative because we maximize (for classification F1) or minimize (for regression/quantile)

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Update model parameters
        self.params.update(study.best_params)
        return study.best_params

    def optimize_hyperparameters_for_q50_constraint(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 20,
        n_splits: int = 3,
        groups: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Optimize hyperparameters specifically for Q50 constraint compliance.
        
        This method searches for parameters that ensure Q50 loss <= max(Q10, Q90) loss.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials (default: 20)
            n_splits: Number of CV splits for evaluation (default: 3)
            groups: Optional array of group labels (e.g., symbol)
        
        Returns:
            Best parameters that satisfy Q50 constraint, or None if no good params found
        """
        if self.model_type != "quantile" or self.quantile_alpha != 0.5:
            print(
                "  ⚠️  optimize_hyperparameters_for_q50_constraint is only for Q50 models"
            )
            return None

        # Prepare data
        X_clean, y_clean = self.prepare_data(X, y)

        # Align groups with cleaned data
        groups_clean = None
        if groups is not None:
            valid_indices = ~y.isna()
            if len(groups) == len(y):
                groups_clean = np.array(groups)[valid_indices]
            elif len(groups) == len(y_clean):
                groups_clean = groups

        # Setup CV (materialize to list to avoid generator exhaustion across trials)
        if groups_clean is not None:
            cv = GroupTimeSeriesSplit(n_splits=n_splits)
            cv_splits_list = list(cv.split(X_clean, groups=groups_clean))
        else:
            cv = TimeSeriesSplit(n_splits=n_splits)
            cv_splits_list = list(cv.split(X_clean))

        def objective(trial):
            params = {
                "num_leaves":
                trial.suggest_int("num_leaves", 15, 127),
                "learning_rate":
                trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "n_estimators":
                trial.suggest_int("n_estimators", 500, 2000),
                "min_data_in_leaf":
                trial.suggest_int("min_data_in_leaf", 20, 200),
                "min_child_weight":
                trial.suggest_float("min_child_weight", 1e-5, 1.0, log=True),
                "lambda_l1":
                trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2":
                trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "feature_fraction":
                trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction":
                trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq":
                trial.suggest_int("bagging_freq", 0, 7),
                "objective":
                "quantile",
                "metric":
                "quantile",
                "alpha":
                0.5,
                "boosting_type":
                "gbdt",
                "verbose":
                -1,
            }

            if self.use_gpu:
                params.update(GPU_LGBM_PARAMS)

            losses_q50: List[float] = []

            for train_idx, val_idx in cv_splits_list:
                X_train_fold = X_clean.iloc[train_idx]
                X_val_fold = X_clean.iloc[val_idx]
                y_train_fold = y_clean.iloc[train_idx]
                y_val_fold = y_clean.iloc[val_idx]

                params_fold = params.copy()
                num_boost_round = params_fold.pop("n_estimators", 1000)

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold,
                                       label=y_val_fold,
                                       reference=train_data)

                model = lgb.train(
                    params_fold,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=num_boost_round,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred = model.predict(X_val_fold)
                error = y_val_fold - y_pred
                quantile_loss = np.mean(
                    np.where(error >= 0, 0.5 * error, 0.5 * (-error)))
                losses_q50.append(float(quantile_loss))

            avg_q50 = float(np.mean(losses_q50)) if losses_q50 else np.inf
            if not np.isfinite(avg_q50):
                return -np.inf
            return -avg_q50

        try:
            # Run optimization (maximize negative loss => minimize loss)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective,
                           n_trials=n_trials,
                           show_progress_bar=False)

            if study.best_trial:
                best_params = study.best_trial.params

                full_params = best_params.copy()
                # Convert boosting rounds
                n_est = full_params.pop("n_estimators", None)
                if n_est:
                    full_params["num_boost_round"] = n_est
                full_params.update({
                    "objective": "quantile",
                    "metric": "quantile",
                    "alpha": 0.5,
                    "boosting_type": "gbdt",
                    "verbose": -1,
                })
                if self.use_gpu:
                    full_params.update(GPU_LGBM_PARAMS)

                def evaluate_candidate(
                    candidate_params: Dict[str, Any]
                ) -> Tuple[Dict[str, float], float]:
                    losses = {"q10": [], "q50": [], "q90": []}
                    base_num_boost_round = candidate_params.get(
                        "num_boost_round", 1000)
                    base_params = {
                        k: v
                        for k, v in candidate_params.items()
                        if k != "num_boost_round"
                    }

                    for train_idx, val_idx in cv_splits_list:
                        X_train_fold = X_clean.iloc[train_idx]
                        X_val_fold = X_clean.iloc[val_idx]
                        y_train_fold = y_clean.iloc[train_idx]
                        y_val_fold = y_clean.iloc[val_idx]

                        train_data = lgb.Dataset(X_train_fold,
                                                 label=y_train_fold)
                        val_data = lgb.Dataset(X_val_fold,
                                               label=y_val_fold,
                                               reference=train_data)

                        for alpha, key in [(0.1, "q10"), (0.5, "q50"),
                                           (0.9, "q90")]:
                            params_alpha = base_params.copy()
                            params_alpha.update({
                                "alpha": alpha,
                                "objective": "quantile",
                                "metric": "quantile",
                            })

                            model = lgb.train(
                                params_alpha,
                                train_data,
                                valid_sets=[val_data],
                                num_boost_round=base_num_boost_round,
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=50),
                                    lgb.log_evaluation(0),
                                ],
                            )

                            y_pred = model.predict(X_val_fold)
                            error = y_val_fold - y_pred
                            loss_val = np.mean(
                                np.where(error >= 0, alpha * error,
                                         (1.0 - alpha) * (-error)))
                            losses[key].append(float(loss_val))

                    avg_losses = {
                        key: float(np.mean(vals)) if len(vals) > 0 else np.inf
                        for key, vals in losses.items()
                    }
                    denom = max(avg_losses["q10"], avg_losses["q90"])
                    ratio = (avg_losses["q50"] /
                             denom if denom > 0 else np.inf)
                    return avg_losses, ratio

                avg_losses, ratio = evaluate_candidate(full_params)
                if not np.isfinite(ratio) or ratio > 1.05:
                    print(
                        f"  ⚠️  Best params violated Q50 constraint (ratio={ratio:.3f}); returning None"
                    )
                    return None

                # Log summary
                print(
                    f"  ✅ Updated parameters: {{'num_leaves': {full_params.get('num_leaves')}, 'learning_rate': {full_params.get('learning_rate')}, 'num_boost_round': {full_params.get('num_boost_round', 1000)}}}"
                )
                print(
                    f"     Avg losses -> Q10: {avg_losses['q10']:.6f}, Q50: {avg_losses['q50']:.6f}, Q90: {avg_losses['q90']:.6f}, ratio={ratio:.3f}"
                )
                return full_params
        except Exception as e:
            print(f"  ⚠️  Hyperparameter optimization failed: {e}")
            return None

        return None
