"""LightGBM model implementation for trading signals and returns prediction."""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
        self.params = params if params is not None else DEFAULT_LGBM_PARAMS.copy()

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
                raise ValueError("quantile_alpha must be provided for quantile regression")
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

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
    ) -> Dict[str, float]:
        """
        Train the LightGBM model using TimeSeriesSplit for proper time series validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of time series splits (default: 5)
            use_time_series_cv: If True, use TimeSeriesSplit; if False, use train_test_split (default: True)

        Returns:
            Training metrics
        """
        # Prepare data
        X_clean, y_clean = self.prepare_data(X, y)

        if use_time_series_cv:
            # ✅ 使用时间序列交叉验证 - 避免未来信息泄露
            print(
                f"  Using TimeSeriesSplit with {n_splits} folds (prevents look-ahead bias)"
            )
            tscv = TimeSeriesSplit(n_splits=n_splits)

            metrics_list = []
            best_model = None
            best_metric = -np.inf if self.model_type == "classification" else np.inf

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

                print(
                    f"  Fold {fold+1}/{n_splits}: Train [{train_idx[0]}:{train_idx[-1]}], Val [{val_idx[0]}:{val_idx[-1]}]"
                )

                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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
                    metrics_list.append({"fold": fold + 1, "accuracy": fold_accuracy})
                    print(f"    Accuracy: {fold_accuracy:.4f}")

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model
                elif self.model_type == "quantile":
                    # For quantile regression, calculate quantile loss (pinball loss)
                    # DEBUG: Check actual values for unit issues
                    if fold == 0:  # Only print for first fold to avoid spam
                        print(f"    DEBUG: y_val range: [{np.nanmin(y_val):.6f}, {np.nanmax(y_val):.6f}], mean={np.nanmean(y_val):.6f}")
                        print(f"    DEBUG: y_pred range: [{np.nanmin(y_pred):.6f}, {np.nanmax(y_pred):.6f}], mean={np.nanmean(y_pred):.6f}")
                    
                    quantile_loss = np.mean(
                        np.maximum(self.quantile_alpha * (y_val - y_pred),
                                   (1 - self.quantile_alpha) * (y_pred - y_val))
                    )
                    metrics_list.append(
                        {"fold": fold + 1, "quantile_loss": quantile_loss}
                    )
                    print(f"    Quantile Loss (alpha={self.quantile_alpha}): {quantile_loss:.6f}")

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model
                else:
                    fold_mse = mean_squared_error(y_val, y_pred)
                    fold_rmse = np.sqrt(fold_mse)
                    metrics_list.append(
                        {"fold": fold + 1, "mse": fold_mse, "rmse": fold_rmse}
                    )
                    print(f"    MSE: {fold_mse:.6f}, RMSE: {fold_rmse:.6f}")

                    # Keep best model (last fold is typically best for time series)
                    if fold == n_splits - 1:  # Use last fold model
                        best_model = model

            # Store the best model
            self.model = best_model

            # Return average metrics across folds
            if self.model_type == "classification":
                avg_accuracy = np.mean([m["accuracy"] for m in metrics_list])
                std_accuracy = np.std([m["accuracy"] for m in metrics_list])
                metrics = {
                    "cv_accuracy": avg_accuracy,
                    "cv_accuracy_std": std_accuracy,
                    "fold_details": metrics_list,
                }
                print(f"  Average CV Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            elif self.model_type == "quantile":
                avg_quantile_loss = np.mean([m["quantile_loss"] for m in metrics_list])
                std_quantile_loss = np.std([m["quantile_loss"] for m in metrics_list])
                metrics = {
                    "cv_quantile_loss": avg_quantile_loss,
                    "cv_quantile_loss_std": std_quantile_loss,
                    "quantile_alpha": self.quantile_alpha,
                    "fold_details": metrics_list,
                }
                print(f"  Average CV Quantile Loss (alpha={self.quantile_alpha}): {avg_quantile_loss:.6f} ± {std_quantile_loss:.6f}")
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
        else:
            # ⚠️ 传统方法（不推荐用于时间序列）- 仅用于对比
            print(
                f"  WARNING: Using train_test_split (random split - not recommended for time series!)"
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )

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
                               (1 - self.quantile_alpha) * (y_pred - y_val))
                )
                metrics = {"quantile_loss": quantile_loss, "quantile_alpha": self.quantile_alpha}
            else:
                y_pred = self.model.predict(X_val)
                metrics = {
                    "mse": mean_squared_error(y_val, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                }

        self.is_trained = True
        return metrics

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
        X_clean, _ = self.prepare_data(
            X, pd.Series([0] * len(X))
        )  # Dummy y for consistency

        # Make predictions
        predictions = self.model.predict(X_clean)
        return predictions

    def optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50
    ) -> Dict[str, Any]:
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
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )

        def objective(trial):
            # Suggest hyperparameters
            params = {
                "objective": self.params["objective"],
                "metric": self.params["metric"],
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart"]
                ),
                "num_leaves": trial.suggest_int("num_leaves", 10, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
                "verbose": -1,
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
                    y_val, y_pred
                )  # Negative because we want to maximize

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Update model parameters
        self.params.update(study.best_params)
        return study.best_params
