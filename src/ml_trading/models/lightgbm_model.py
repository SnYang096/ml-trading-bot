"""LightGBM model implementation for trading signals and returns prediction."""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold
from sklearn.metrics import accuracy_score, mean_squared_error
import optuna
from ml_trading.config.settings import DEFAULT_LGBM_PARAMS, USE_GPU, GPU_LGBM_PARAMS


class LightGBMModel:
    """LightGBM model for trading signal classification and return regression."""

    def __init__(
        self,
        model_type: str = "classification",
        params: Optional[Dict] = None,
        use_gpu: Optional[bool] = None,
        quantile_alpha: Optional[float] = None,
    ):
        """
        Initialize the LightGBM model.

        Args:
            model_type: Either "classification", "regression", or "quantile"
            params: LightGBM parameters (if None, use DEFAULT_LGBM_PARAMS)
            use_gpu: Enable GPU acceleration (if None, use USE_GPU from config)
            quantile_alpha: Alpha value for quantile regression (e.g., 0.1, 0.5, 0.9)
        """
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
        if model_type == "quantile":
            # Quantile regression (for q10, q50, q90 models)
            if quantile_alpha is None:
                raise ValueError(
                    "quantile_alpha must be provided for quantile regression")
            self.params["objective"] = "quantile"
            self.params["alpha"] = quantile_alpha
            self.params["metric"] = "quantile"
        elif model_type == "regression":
            # Regression for predicting continuous returns (e.g., volatility)
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"
        else:
            # Classification: Use 3-class (0=Hold, 1=Long, 2=Short) instead of binary
            self.params["objective"] = "multiclass"
            self.params["metric"] = "multi_logloss"
            self.params["num_class"] = 3

    def prepare_data(self, X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Prepared X and y
        """
        # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns].copy()

        # Sanitize infinities and NaNs in features; keep rows whenever y is valid
        X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_numeric.fillna(0.0, inplace=True)

        # Clean target
        y_series = y.copy()
        y_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_indices = ~y_series.isna()
        X_clean = X_numeric.loc[valid_indices]
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
            groups: Optional array of group labels for GroupKFold (for multi-asset training).
                   If provided and use_time_series_cv=True, will use GroupKFold instead of TimeSeriesSplit.
                   This ensures samples from the same group (e.g., symbol) are not split across train/val.
        
        Returns:
            Tuple of (training_metrics, preprocess_params)
            - training_metrics: Training metrics dictionary
            - preprocess_params: Preprocessing parameters from first fold (for deployment), or None
        """
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
                print(
                    f"  Warning: groups length ({len(groups)}) doesn't match y ({len(y)}) or y_clean ({len(y_clean)}), ignoring groups"
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
                print(
                    f"  Warning: sample_weight length ({len(sample_weight)}) doesn't match y ({len(y)}) or y_clean ({len(y_clean)}), ignoring weights"
                )
                sample_weight_clean = None

        if use_time_series_cv:
            # ✅ 使用时间序列交叉验证 - 避免未来信息泄露
            # 如果提供了 groups，使用 GroupKFold（按 symbol 分组，避免跨标的数据泄露）
            # 否则使用 TimeSeriesSplit（标准时间序列交叉验证）
            if groups_clean is not None:
                print(f"  🔒 使用 GroupKFold 交叉验证（按 symbol 分组，避免跨标的数据泄露）")
                cv = GroupKFold(n_splits=n_splits)
            else:
                print(
                    f"  Using TimeSeriesSplit with {n_splits} folds (prevents look-ahead bias)"
                )
                cv = TimeSeriesSplit(n_splits=n_splits)

            metrics_list = []
            best_model = None
            best_metric = -np.inf if self.model_type == "classification" else np.inf
            preprocess_stats_first_fold = None  # Store first fold stats for deployment

            # Split data using the appropriate CV strategy
            if groups_clean is not None:
                # GroupKFold: groups parameter is required (use aligned groups)
                cv_splits = cv.split(X_clean, groups=groups_clean)
            else:
                # TimeSeriesSplit: no groups parameter
                cv_splits = cv.split(X_clean)

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
                    print(
                        f"    Feature cleaning (fold {fold+1}): {feature_clean_stats['n_features_cleaned']} features cleaned"
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
                    if fold == 0:  # Log preprocessing stats for first fold only
                        print(
                            f"    Target preprocessing stats (fold {fold+1}): {preprocess_stats}"
                        )
                        # Store first fold stats for deployment parameter extraction
                        preprocess_stats_first_fold = preprocess_stats
                else:
                    y_train, y_val = y_train_raw, y_val_raw

                print(
                    f"  Fold {fold+1}/{n_splits}: Train [{train_idx[0]}:{train_idx[-1]}], Val [{val_idx[0]}:{val_idx[-1]}]"
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
                model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(0),
                    ],
                )

                # Evaluate on this fold
                y_pred = model.predict(X_val)

                if self.model_type == "classification":
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    fold_accuracy = accuracy_score(y_val, y_pred_binary)
                    metrics_list.append({
                        "fold": fold + 1,
                        "accuracy": fold_accuracy
                    })
                    print(f"    Accuracy: {fold_accuracy:.4f}")

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model
                elif self.model_type == "quantile":
                    # For quantile regression, calculate quantile loss (pinball loss)
                    # DEBUG: Check actual values for unit issues
                    if fold == 0:  # Only print for first fold to avoid spam
                        print(
                            f"    DEBUG: y_val range: [{np.nanmin(y_val):.6f}, {np.nanmax(y_val):.6f}], mean={np.nanmean(y_val):.6f}"
                        )
                        print(
                            f"    DEBUG: y_pred range: [{np.nanmin(y_pred):.6f}, {np.nanmax(y_pred):.6f}], mean={np.nanmean(y_pred):.6f}"
                        )

                    quantile_loss = np.mean(
                        np.maximum(self.quantile_alpha * (y_val - y_pred),
                                   (1 - self.quantile_alpha) *
                                   (y_pred - y_val)))
                    metrics_list.append({
                        "fold": fold + 1,
                        "quantile_loss": quantile_loss
                    })
                    print(
                        f"    Quantile Loss (alpha={self.quantile_alpha}): {quantile_loss:.6f}"
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
                    print(f"    MSE: {fold_mse:.6f}, RMSE: {fold_rmse:.6f}")

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model

            # Store the best model
            self.model = best_model

            # Extract preprocessing parameters from first fold (for deployment)
            # These parameters will be used for consistent preprocessing in production
            preprocess_params = None
            if preprocess_fn is not None and preprocess_stats_first_fold is not None:
                first_fold_stats = preprocess_stats_first_fold
                # Extract key parameters for deployment
                winsorize_stats = first_fold_stats.get('step1_winsorize', {})
                ar1_stats = first_fold_stats.get('step2_ar1', {})
                secondary_stats = first_fold_stats.get('step2b_secondary', {})

                preprocess_params = {
                    "winsorize": {
                        "median": float(winsorize_stats.get('median', 0.0)),
                        "mad": float(winsorize_stats.get('mad', 0.0)),
                        "sigma": float(winsorize_stats.get('sigma', 0.0)),
                        "k": float(winsorize_stats.get('k', 3.5)),
                    },
                    "ar1": {
                        "ar1_phi":
                        float(ar1_stats.get('ar1_phi', 0.0)),
                        "ar1_autocorr_after":
                        float(ar1_stats.get('ar1_autocorr_after', 0.0))
                        if ar1_stats.get('ar1_autocorr_after') is not None else
                        None,
                    },
                    "secondary": {
                        "median":
                        float(secondary_stats.get('median', 0.0)),
                        "mad":
                        float(secondary_stats.get('mad', 0.0)),
                        "sigma":
                        float(secondary_stats.get('sigma', 0.0)),
                        "clip_threshold":
                        float(secondary_stats.get('clip_threshold', 0.0)),
                    },
                    "note":
                    "Parameters extracted from first CV fold. Use these for consistent preprocessing in deployment.",
                }

            # Return average metrics across folds
            if self.model_type == "classification":
                avg_accuracy = np.mean([m["accuracy"] for m in metrics_list])
                std_accuracy = np.std([m["accuracy"] for m in metrics_list])
                metrics = {
                    "cv_accuracy": avg_accuracy,
                    "cv_accuracy_std": std_accuracy,
                    "fold_details": metrics_list,
                }
                print(
                    f"  Average CV Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}"
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
                print(
                    f"  Average CV Quantile Loss (alpha={self.quantile_alpha}): {avg_quantile_loss:.6f} ± {std_quantile_loss:.6f}"
                )
                return metrics, preprocess_params
            else:
                avg_mse = np.mean([m["mse"] for m in metrics_list])
                avg_rmse = np.mean([m["rmse"] for m in metrics_list])
                std_mse = np.std([m["mse"] for m in metrics_list])
                metrics = {
                    "cv_mse": avg_mse,
                    "cv_rmse": avg_rmse,
                    "cv_mse_std": std_mse,
                    "fold_details": metrics_list,
                }
                print(f"  Average CV MSE: {avg_mse:.6f} ± {std_mse:.6f}")
                return metrics, preprocess_params
        else:
            # ⚠️ 传统方法（不推荐用于时间序列）- 仅用于对比
            print(
                f"  WARNING: Using train_test_split (random split - not recommended for time series!)"
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
            if self.model_type == "classification":
                y_pred = self.model.predict(X_val)
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics = {"accuracy": accuracy_score(y_val, y_pred_binary)}
            elif self.model_type == "quantile":
                y_pred = self.model.predict(X_val)
                quantile_loss = np.mean(
                    np.maximum(self.quantile_alpha * (y_val - y_pred),
                               (1 - self.quantile_alpha) * (y_pred - y_val)))
                metrics = {
                    "quantile_loss": quantile_loss,
                    "quantile_alpha": self.quantile_alpha
                }
            else:
                y_pred = self.model.predict(X_val)
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
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare data
        X_clean, _ = self.prepare_data(X, pd.Series(
            [0] * len(X)))  # Dummy y for consistency

        # Make predictions
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

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_clean,
                                                          y_clean,
                                                          test_size=0.2,
                                                          random_state=42)

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

            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model
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

            # Evaluate
            if self.model_type == "classification":
                y_pred = model.predict(X_val)
                y_pred_binary = (y_pred > 0.5).astype(int)
                return accuracy_score(y_val, y_pred_binary)
            else:
                y_pred = model.predict(X_val)
                return -mean_squared_error(
                    y_val, y_pred)  # Negative because we want to maximize

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Update model parameters
        self.params.update(study.best_params)
        return study.best_params
