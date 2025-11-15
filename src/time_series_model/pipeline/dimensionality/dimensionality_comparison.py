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
# Autoencoder removed - no longer used
from time_series_model.utils.training import train_lightgbm_model

# Import report generator for HTML report writing
from time_series_model.pipeline.dimensionality.report_generator import write_html_report
from time_series_model.backtesting.vectorbot import (
    backtest_classification_model,
    calculate_strategy_returns_from_predictions,
    calculate_financial_metrics_from_returns,
)

# Import split modules
from time_series_model.pipeline.dimensionality.data_loader import (
    load_real_market_data,
    create_enhanced_sample_data,
)
from time_series_model.pipeline.dimensionality.model_training import (
    train_production_lightgbm, )
from time_series_model.pipeline.dimensionality.evaluation import (
    evaluate_model_performance,
    calculate_financial_metrics,
    compute_selection_score,
    sanitize_features,
    _generate_shap_outputs,
)
from time_series_model.pipeline.dimensionality.utils import (
    _slugify,
    _get_primary_metric,
    _derive_feature_insights,
)

DIM_COMPARE_RESULTS_ROOT = Path("results") / "dim_compare"

# Removed duplicate function definitions - these are imported from utils.py, evaluation.py, and data_loader.py


def load_real_market_data(
    data_path: str,
    symbol: str = "ETH-USD",
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: list[int] | None = None,
    feature_type: str = "comprehensive",
    timeframe: str = "5T",
) -> Tuple[np.ndarray, np.ndarray, list, list[int], pd.DataFrame]:
    """Load real market data for one or multiple symbols.
    
    Args:
        symbol: Single symbol or comma-separated symbols (e.g., "ETH-USD" or "ETH-USD,BTC-USD,SOL-USD")
        timeframe: Timeframe for data resampling (e.g., "5T", "15T", "60T", "240T"). Default: "5T"
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
                    df_single = symbol_loader.resample_data(timeframe)
                elif isinstance(df_single.index, pd.DatetimeIndex):
                    # Fallback: resample manually
                    df_single = df_single.resample(timeframe).agg({
                        'open':
                        'first',
                        'high':
                        'max',
                        'low':
                        'min',
                        'close':
                        'last',
                        'volume':
                        'sum'
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
        # Add symbol identifier for rank-based IC calculation
        all_dfs_with_symbol = []
        for sym, df_single in zip(symbol_list, all_dfs):
            df_with_symbol = df_single.copy()
            df_with_symbol['_symbol'] = sym  # Add symbol identifier
            all_dfs_with_symbol.append(df_with_symbol)

        df = pd.concat(all_dfs_with_symbol, axis=0).sort_index()
        if len(symbol_list) > 1:
            print(
                f"   Merged {len(all_dfs)} asset(s), total {len(df)} samples")
            print(f"   Added symbol identifier for rank-based IC calculation")

        # Store symbol info before feature engineering (in case it gets dropped)
        symbol_info = df['_symbol'].copy() if '_symbol' in df.columns else None

        comprehensive_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_type)
        df_features = comprehensive_engineer.engineer_all_features(df,
                                                                   fit=True)

        # Restore symbol info if it was dropped during feature engineering
        if symbol_info is not None and '_symbol' not in df_features.columns:
            df_features['_symbol'] = symbol_info.reindex(df_features.index)

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
        # Exclude raw OHLC price features - use derived features instead
        # Exclude raw volume/order flow features - use normalized/derived features instead
        exclude_exact = {
            "timestamp",
            "close",
            "open",  # Exclude raw OHLC prices - use derived features instead
            "high",  # Exclude raw OHLC prices - use derived features instead
            "low",  # Exclude raw OHLC prices - use derived features instead
            "volume",  # Exclude raw volume - use volume_percentile, volume_anomaly, etc.
            "cvd",  # Exclude raw CVD - use cvd_normalized, cvd_spectral_*, cvd_wpt_*, etc.
            "sell_qty",  # Exclude raw sell_qty - use normalized/derived features instead
            "buy_qty",  # Exclude raw buy_qty - use normalized/derived features instead
            "binary_signal",
            "future_return",
            "_symbol",  # Exclude symbol identifier (used for rank-based IC only)
        }
        exclude_prefixes = (
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
        y = df_features[f"binary_signal_{default_horizon}"].dropna(
        ).values  # Use binary signal (0=Short, 1=Long)

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


# Autoencoder functions removed - no longer used
# train_production_lightgbm is now imported from model_training.py


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
        price_data:
    Optional[
        pd.
        DataFrame] = None,  # Optional: price data for calculating real returns
):
    X_eval = X_test
    y_eval = y_test

    predictions = model.predict(X_eval)

    # Binary classification: LightGBM returns probability array for binary classification
    # Shape: (n_samples, 2) for binary classification, (n_samples,) for regression
    is_binary = predictions.ndim == 2 and predictions.shape[1] == 2
    if is_binary:
        # Binary: convert probability array to class predictions
        predictions_class = np.argmax(predictions, axis=1)
        predictions_for_metrics = predictions_class
        predictions_to_store = predictions_class
        probabilities = predictions
    else:
        # Regression: use predictions as-is
        predictions_for_metrics = predictions
        predictions_to_store = predictions
        probabilities = None

    # Binary classification labels
    is_binary_classification = False
    if y_eval.ndim == 1 and np.issubdtype(y_eval.dtype, np.integer):
        unique_eval = np.unique(y_eval)
        if np.all(np.isin(unique_eval, [0, 1])):
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

    # Basic numeric metrics
    mse = mean_squared_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        y_eval if not is_binary_classification else y_eval_binary,
        predictions_for_metrics)
    # Calculate R² for regression metrics
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
        if is_binary_classification:
            # Binary classification: compute win rate (0=Short, 1=Long)
            y_pred_cls = predictions_class.astype(int)
            y_true_cls = y_eval_binary.astype(int)

            # Calculate win rate (accuracy for binary classification)
            win_rate = float(np.mean(y_pred_cls == y_true_cls))

            # Long predictions win rate
            long_mask = y_pred_cls == 1
            long_total = int(np.sum(long_mask))
            if long_total > 0:
                long_correct = np.sum((y_pred_cls == 1) & (y_true_cls == 1))
                long_win_rate = float(long_correct / long_total)
            else:
                long_win_rate = 0.0

            # Short predictions win rate
            short_mask = y_pred_cls == 0
            short_total = int(np.sum(short_mask))
            if short_total > 0:
                short_correct = np.sum((y_pred_cls == 0) & (y_true_cls == 0))
                short_win_rate = float(short_correct / short_total)
            else:
                short_win_rate = 0.0

            fm = results.setdefault("financial_metrics", {})
            fm["win_rate"] = win_rate
            fm["long_win_rate"] = long_win_rate
            fm["short_win_rate"] = short_win_rate

            # For classification tasks, calculate Sharpe Ratio and Max Drawdown
            # Try to use real backtest if price data is available, otherwise use win_rate as proxy
            if price_data is not None and 'close' in price_data.columns:
                try:
                    # Use real backtest with actual price data
                    if len(price_data) == len(y_pred_cls):
                        # Calculate strategy returns from predictions and actual prices
                        strategy_returns = calculate_strategy_returns_from_predictions(
                            y_pred_cls, price_data, horizon=1)
                        # Calculate financial metrics from real returns
                        backtest_metrics = calculate_financial_metrics_from_returns(
                            strategy_returns, risk_free_rate=0.0)
                        fm["sharpe_ratio"] = backtest_metrics.get(
                            "sharpe_ratio", 0.0)
                        fm["max_drawdown"] = backtest_metrics.get(
                            "max_drawdown", 0.0)
                        fm["total_return"] = backtest_metrics.get(
                            "total_return", 0.0)
                        fm["annualized_return"] = backtest_metrics.get(
                            "annualized_return", 0.0)
                        fm["volatility"] = backtest_metrics.get(
                            "volatility", 0.0)
                    else:
                        # Fallback: use win rate as proxy
                        sharpe_approx = (win_rate - 0.5) * 4.0
                        fm["sharpe_ratio"] = float(sharpe_approx)
                        max_dd_approx = -(1.0 - win_rate) * 0.1
                        fm["max_drawdown"] = float(max_dd_approx)
                        total_return_approx = (win_rate - 0.5) * 2.0
                        fm["total_return"] = float(total_return_approx)
                except Exception as e:
                    # If backtest fails, fall back to win rate approximation
                    print(
                        f"  ⚠️  Backtest calculation failed: {e}, using win_rate approximation"
                    )
                    sharpe_approx = (win_rate - 0.5) * 4.0
                    fm["sharpe_ratio"] = float(sharpe_approx)
                    max_dd_approx = -(1.0 - win_rate) * 0.1
                    fm["max_drawdown"] = float(max_dd_approx)
                    total_return_approx = (win_rate - 0.5) * 2.0
                    fm["total_return"] = float(total_return_approx)
            else:
                # No price data available: use win rate as proxy for Sharpe Ratio
                sharpe_approx = (win_rate - 0.5) * 4.0
                fm["sharpe_ratio"] = float(sharpe_approx)
                max_dd_approx = -(1.0 - win_rate) * 0.1
                fm["max_drawdown"] = float(max_dd_approx)
                total_return_approx = (win_rate - 0.5) * 2.0
                fm["total_return"] = float(total_return_approx)

            print(f"  Win Rate: {win_rate:.4f}")
            print(f"  Long Win Rate: {long_win_rate:.4f}")
            print(f"  Short Win Rate: {short_win_rate:.4f}")
            print(f"  Sharpe Ratio: {fm.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {fm.get('max_drawdown', 0):.4f}")

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

            # Binary classification: probabilities shape is (n_samples, 2) or (n_samples,)
            # For ROC AUC, we need the positive class probabilities
            # Use original predictions if it's 2D (model output), otherwise use probabilities
            probabilities_for_roc = None
            if predictions.ndim == 2 and predictions.shape[1] == 2:
                # Original predictions are 2D: use positive class probabilities
                probabilities_for_roc = predictions[:, 1]
            elif probabilities is not None:
                if hasattr(probabilities, 'ndim'):
                    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                        # 2D array: use positive class probabilities
                        probabilities_for_roc = probabilities[:, 1]
                    elif probabilities.ndim == 1:
                        # 1D array: use as-is (already positive class probabilities)
                        probabilities_for_roc = probabilities
                    elif probabilities.ndim == 0:
                        # Scalar: convert to array
                        probabilities_for_roc = np.array([probabilities])

            if probabilities_for_roc is not None:
                y_true_onehot = label_binarize(y_true_cls, classes=[0, 1])

                try:
                    # Binary classification: use positive class probabilities
                    metrics["roc_auc_macro"] = float(
                        roc_auc_score(
                            y_true_cls,
                            probabilities_for_roc,
                        ))
                except Exception:
                    metrics["roc_auc_macro"] = None
                try:
                    metrics["pr_auc_macro"] = float(
                        average_precision_score(
                            y_true_cls,
                            probabilities_for_roc,
                        ))
                except Exception:
                    metrics["pr_auc_macro"] = None
            else:
                metrics["roc_auc_macro"] = None
                metrics["pr_auc_macro"] = None

            try:
                cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
                metrics["confusion_matrix"] = cm.tolist()
                metrics["labels"] = [0, 1]
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
                print(f"  ROC AUC: {metrics['roc_auc_macro']:.4f}")
            if metrics["pr_auc_macro"] is not None:
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
    results_dir: str,
) -> str:
    print("💾 Saving production results...")
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/production_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump(model, f"{results_dir}/production_model.pkl")

    print(f"✅ Results saved to {results_dir}")
    return results_dir


def run_dimensionality_comparison(
    data_path: str = "/data/parquet_data",
    symbol: str = "ETH-USD",
    train_start: str | None = None,
    train_end: str | None = None,
    feature_type: str = "comprehensive",
    shap_analysis: bool = True,
    timeframe: str = "5T",
) -> Tuple[Dict, any, type(None), str]:
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
        timeframe=timeframe,
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

    # Use original features (autoencoder removed)
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
            len(feature_names),
            "compression_ratio":
            ((X.shape[1] / len(feature_names)) if len(feature_names) else 1.0),
            "training_samples":
            len(X_train),
            "validation_samples":
            len(X_val),
            "test_samples":
            len(X_test),
        },
        "training_info": {
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

    return results, model_compressed, results_dir


def run_single_experiment_wrapper(args) -> Dict:
    """Wrapper function to run a single experiment with given args and return result dict."""
    # We need to run the experiment logic directly
    # Since the main logic is in the if args.research_ablation block,
    # we'll create a simplified version that reuses the same code
    # For now, let's use a workaround: modify sys.argv temporarily
    import sys
    original_argv = sys.argv[:]

    try:
        # Build new argv from args object
        new_argv = ['dimensionality_comparison.py']
        if args.data_path:
            new_argv.extend(['--data-path', args.data_path])
        if args.symbol:
            new_argv.extend(['--symbol', args.symbol])
        if args.feature_type:
            new_argv.extend(['--feature-type', args.feature_type])
        if args.timeframe:
            new_argv.extend(['--timeframe', args.timeframe])
        if args.train_start:
            new_argv.extend(['--train-start', args.train_start])
        if args.train_end:
            new_argv.extend(['--train-end', args.train_end])
        # top_k parameter removed, use factor_counts instead
        if args.horizons:
            new_argv.extend(['--horizons', args.horizons])
        if args.binary_signals:
            new_argv.append('--binary-signals')
        if args.label_threshold:
            new_argv.extend(['--label-threshold', str(args.label_threshold)])
        if getattr(args, 'selection_metric', None):
            new_argv.extend(['--selection-metric', args.selection_metric])
        if getattr(args, 'max_dd_threshold', None) is not None:
            new_argv.extend(['--max-dd-threshold', str(args.max_dd_threshold)])
        if getattr(args, 'composite_alpha', None) is not None:
            new_argv.extend(['--composite-alpha', str(args.composite_alpha)])
        if getattr(args, 'composite_beta', None) is not None:
            new_argv.extend(['--composite-beta', str(args.composite_beta)])
        if args.report_html:
            new_argv.extend(['--report-html', args.report_html])
        if args.shap_analysis:
            new_argv.append('--shap-analysis')
        # Autoencoder removed - no longer used
        if args.task:
            new_argv.extend(['--task', args.task])
        if args.enable_stability_validation:
            new_argv.append('--enable-stability-validation')
        if args.validation_start:
            new_argv.extend(['--validation-start', args.validation_start])
        if args.validation_years:
            new_argv.extend(['--validation-years', str(args.validation_years)])

        # Set new argv
        sys.argv = new_argv

        # Call main() which will parse the new argv
        results, model, results_dir = main()
        # Store results_dir in results dict for later file copying
        if results and isinstance(results, dict):
            results['results_dir'] = results_dir
        return results
    except Exception as exc:
        print(f"⚠️ Single experiment failed: {exc}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def _generate_metric_3d_plot(enhanced_results: list,
                             time_windows: list,
                             factor_counts: list,
                             metric_name: str,
                             metric_label: str,
                             metric_getter,
                             color_thresholds: dict = None) -> str:
    """Generate 3D visualization for any metric across factor counts and time windows.
    
    Args:
        enhanced_results: List of result dictionaries
        time_windows: List of time window strings
        factor_counts: List of factor counts
        metric_name: Name of the metric (e.g., 'icir', 'sharpe', 'robustness')
        metric_label: Display label for the metric
        metric_getter: Function to extract metric value from result dict
        color_thresholds: Dict with 'good', 'warn', 'bad' thresholds
    """
    try:
        import plotly.graph_objects as go
        import numpy as np

        # Default color thresholds
        if color_thresholds is None:
            color_thresholds = {'good': 1.0, 'warn': 0.5, 'bad': 0.0}

        # Prepare data arrays
        x_data = []
        y_data = []
        z_data = []
        colors = []
        text_labels = []

        for i, tw in enumerate(time_windows):
            for fc in sorted(factor_counts,
                             key=lambda x:
                             (x == 'all', x
                              if isinstance(x, int) else 999999)):
                # Find result for this combination
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    metric_val = metric_getter(result)
                    # Include 0 values as well (they are valid data points)
                    # Debug: print first extraction
                    if len(x_data) == 0 and metric_val is not None:
                        print(
                            f"[DEBUG 3D] First data point: tw={tw}, fc={fc}, metric_val={metric_val}"
                        )
                    if metric_val is not None:
                        # X: factor count (numeric)
                        if isinstance(fc, int):
                            x_val = fc
                        else:
                            max_fc = max([
                                x for x in factor_counts if isinstance(x, int)
                            ],
                                         default=120)
                            x_val = max_fc * 1.2

                        x_data.append(x_val)
                        y_data.append(i)  # Time window index
                        z_data.append(metric_val)

                        # Color based on metric value
                        if metric_val > color_thresholds['good']:
                            colors.append('#167a3d')  # Green
                        elif metric_val > color_thresholds['warn']:
                            colors.append('#ffc107')  # Yellow
                        else:
                            colors.append('#dc3545')  # Red

                        text_labels.append(
                            f"Time: {tw}<br>Factors: {fc}<br>{metric_label}: {metric_val:.3f}"
                        )

        if not x_data:
            return f"""
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <h4>📊 {metric_label} 3D 可视化</h4>
                <p>⚠️ 没有可用的数据来生成3D图形。</p>
            </div>
            """

        # Create 3D scatter plot
        fig = go.Figure(
            data=go.Scatter3d(x=x_data,
                              y=y_data,
                              z=z_data,
                              mode='markers',
                              marker=dict(size=10,
                                          color=colors,
                                          opacity=0.8,
                                          line=dict(width=1, color='black')),
                              text=text_labels,
                              hovertemplate='%{text}<extra></extra>',
                              name=f'{metric_label} Points'))

        # Add surface plot to show trend
        if len(x_data) > 0 and len(set(x_data)) > 1 and len(set(y_data)) > 1:
            # Create grid for surface
            x_unique = sorted(set(x_data))
            y_unique = sorted(set(y_data))

            # Create meshgrid
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
            Z_grid = np.full_like(X_grid, np.nan, dtype=float)

            # Fill Z_grid with metric values
            for i, tw_idx in enumerate(y_unique):
                for j, fc_val in enumerate(x_unique):
                    # Find metric for this combination
                    for r in enhanced_results:
                        params = r.get('grid_search_params', {})
                        tw = time_windows[tw_idx]
                        # Find matching factor count
                        fc_match = None
                        for fc in factor_counts:
                            if isinstance(fc, int) and fc == fc_val:
                                fc_match = fc
                                break
                            elif fc == 'all' and abs(
                                    fc_val - max([
                                        x for x in factor_counts
                                        if isinstance(x, int)
                                    ],
                                                 default=120) * 1.2) < 1:
                                fc_match = fc
                                break

                        if params.get('time_window') == tw and params.get(
                                'factor_count') == fc_match:
                            metric_val = metric_getter(r)
                            if metric_val is not None:
                                Z_grid[i, j] = metric_val
                            break

            # Add surface plot
            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale='RdYlGn',
                    showscale=True,
                    opacity=0.6,
                    name=f'{metric_label} Surface',
                    hovertemplate=
                    f'Factor Count: %{{x:.0f}}<br>Time Window: %{{y}}<br>{metric_label}: %{{z:.3f}}<extra></extra>'
                ))

        # Get factor count labels
        fc_labels = []
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            if isinstance(fc, int):
                fc_labels.append(str(fc))
            else:
                fc_labels.append('all')

        # Update layout
        fig.update_layout(
            title=
            f'{metric_label} 3D 可视化 - Plateau Point 分析 ({metric_label} 3D Visualization - Plateau Point Analysis)',
            scene=dict(xaxis_title='因子数量 (Factor Count)',
                       yaxis_title='时间窗口索引 (Time Window Index)',
                       zaxis_title=f'{metric_label} 值 ({metric_label} Value)',
                       xaxis=dict(
                           tickmode='array',
                           tickvals=x_unique if 'x_unique' in locals() else [],
                           ticktext=fc_labels[:len(x_unique)]
                           if 'x_unique' in locals() else [],
                       ),
                       yaxis=dict(
                           tickmode='array',
                           tickvals=list(range(len(time_windows))),
                           ticktext=[
                               tw.split(' → ')[0] if ' → ' in tw else tw[:15]
                               for tw in time_windows
                           ],
                       ),
                       zaxis=dict(title=metric_label),
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            width=900,
            height=700,
            font=dict(size=12),
        )

        # Convert to HTML - use full HTML to ensure Plotly.js is included
        plot_html = fig.to_html(include_plotlyjs='cdn',
                                div_id=f'{metric_name}-3d-plot',
                                full_html=False)

        # Extract script and div more robustly
        import re
        # Match all script tags (may be multiple)
        script_matches = re.findall(r'<script[^>]*>.*?</script>', plot_html,
                                    re.DOTALL)
        script_content = '\n'.join(script_matches) if script_matches else ""

        # Match div with the specific ID
        div_pattern = rf'<div[^>]*id="{metric_name}-3d-plot"[^>]*>.*?</div>'
        div_match = re.search(div_pattern, plot_html, re.DOTALL)
        div_content = div_match.group(
            0
        ) if div_match else f'<div id="{metric_name}-3d-plot" class="plotly-graph-div" style="height:700px; width:900px;"></div>'

        # Debug: print if no data
        if not x_data:
            print(
                f"[DEBUG 3D] No data for {metric_label}: x_data length = {len(x_data)}"
            )
            print(
                f"[DEBUG 3D] enhanced_results count: {len(enhanced_results)}")
            if enhanced_results:
                print(
                    f"[DEBUG 3D] First result keys: {list(enhanced_results[0].keys())}"
                )
                print(
                    f"[DEBUG 3D] First result grid_search_params: {enhanced_results[0].get('grid_search_params', {})}"
                )

        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #17a2b8;">
            <h4>📊 {metric_label} 3D 可视化说明 (3D Visualization Guide):</h4>
            <ul>
                <li><strong>X轴（因子数量）：</strong>显示不同的因子数量。数值越大，使用的因子越多。</li>
                <li><strong>Y轴（时间窗口）：</strong>显示不同的时间窗口索引。每个索引对应一个时间窗口。</li>
                <li><strong>Z轴（{metric_label}值）：</strong>显示{metric_label}值，越高表示表现越好。</li>
                <li><strong>颜色含义：</strong>
                    <ul>
                        <li><span style="color: #167a3d; font-weight: 600;">绿色点</span>：{metric_label} > {color_thresholds['good']}，表现优秀</li>
                        <li><span style="color: #ffc107; font-weight: 600;">黄色点</span>：{color_thresholds['warn']} < {metric_label} ≤ {color_thresholds['good']}，表现一般</li>
                        <li><span style="color: #dc3545; font-weight: 600;">红色点</span>：{metric_label} ≤ {color_thresholds['warn']}，表现较差</li>
                    </ul>
                </li>
                <li><strong>如何识别Plateau Point：</strong>
                    <ul>
                        <li>观察3D表面图，寻找{metric_label}值不再显著上升的"平台"区域</li>
                        <li>Plateau Point通常出现在：{metric_label}值达到较高水平后，即使增加因子数量，{metric_label}也不再明显提升的位置</li>
                        <li>理想情况下，Plateau Point应该在不同时间窗口（Y轴）上都保持相对稳定的高度（Z轴）</li>
                        <li>可以通过旋转3D图形（点击并拖动）从不同角度观察，更容易识别平台区域</li>
                    </ul>
                </li>
                <li><strong>分析建议：</strong>
                    <ul>
                        <li>寻找Z轴（{metric_label}）值高且在不同Y轴（时间窗口）位置都保持稳定的X轴（因子数量）位置</li>
                        <li>如果表面图在某个因子数量后变得平坦，该位置就是Plateau Point</li>
                        <li>选择Plateau Point对应的因子数量，可以在保持高{metric_label}的同时，避免使用过多因子</li>
                    </ul>
                </li>
            </ul>
        </div>
        <div class="heatmap-container">
            {div_content}
        </div>
        {script_content}
        """
    except ImportError:
        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
            <h4>📊 {metric_label} 3D 可视化</h4>
            <p>⚠️ Plotly 未安装，无法生成3D图形。请安装: pip install plotly</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
            <h4>📊 {metric_label} 3D 可视化</h4>
            <p>⚠️ 生成3D图形时出错: {str(e)}</p>
        </div>
        """


def _generate_icir_3d_plot(enhanced_results: list, time_windows: list,
                           factor_counts: list) -> str:
    """Generate 3D visualization of ICIR distribution across factor counts and time windows."""

    def get_icir(result):
        return result.get('enhanced_metrics', {}).get('icir')

    return _generate_metric_3d_plot(enhanced_results,
                                    time_windows,
                                    factor_counts,
                                    'icir',
                                    'ICIR',
                                    get_icir,
                                    color_thresholds={
                                        'good': 1.0,
                                        'warn': 0.5,
                                        'bad': 0.0
                                    })


def _generate_icir_heatmap(enhanced_results: list, time_windows: list,
                           factor_counts: list) -> str:
    """Generate ICIR heatmap visualization using Plotly."""
    try:
        import plotly.graph_objects as go

        # Prepare data matrix for heatmap
        heatmap_data = []
        factor_count_labels = []

        # Build data matrix: rows = factor counts, columns = time windows
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            row_data = []
            factor_count_labels.append(str(fc))

            for tw in time_windows:
                # Find result for this combination
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    # Use ICIR value if available, otherwise try to get from ic_statistics
                    if icir is None:
                        ic_stats = result.get('ic_statistics', {})
                        ic_mean = ic_stats.get('ic_mean')
                        ic_std = ic_stats.get('ic_std')
                        if ic_mean is not None and ic_std is not None and ic_std > 0:
                            icir = abs(ic_mean) / ic_std
                    row_data.append(icir if icir is not None else 0)
                else:
                    row_data.append(0)

            heatmap_data.append(row_data)

        # Set time window labels (shortened for display)
        time_window_labels = [
            tw.split(' → ')[0] if ' → ' in tw else tw[:10]
            for tw in time_windows
        ]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=time_window_labels,
            y=factor_count_labels,
            colorscale='RdYlGn',  # Red-Yellow-Green scale
            colorbar=dict(title="ICIR"),
            text=[[f"{val:.3f}" if val else "-" for val in row]
                  for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate=
            'Time Window: %{x}<br>Factor Count: %{y}<br>ICIR: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title='ICIR 热力图 (ICIR Heatmap)',
            xaxis_title='时间窗口 (Time Window)',
            yaxis_title='因子数量 (Factor Count)',
            width=800,
            height=500,
            font=dict(size=12),
        )

        # Convert to HTML - use full HTML with CDN for plotly.js
        heatmap_html_full = fig.to_html(include_plotlyjs='cdn',
                                        div_id='icir-heatmap',
                                        full_html=False)

        # Extract script and div from the HTML more robustly
        import re
        # Match all script tags (may be multiple)
        script_matches = re.findall(r'<script[^>]*>.*?</script>',
                                    heatmap_html_full, re.DOTALL)
        script_content = '\n'.join(script_matches) if script_matches else ""

        # Match div with the specific ID
        div_pattern = r'<div[^>]*id="icir-heatmap"[^>]*>.*?</div>'
        div_match = re.search(div_pattern, heatmap_html_full, re.DOTALL)
        div_content = div_match.group(
            0
        ) if div_match else '<div id="icir-heatmap" class="plotly-graph-div" style="height:500px; width:800px;"></div>'

        # Debug: print if no data
        if not heatmap_data or all(
                all(val == 0 for val in row) for row in heatmap_data):
            print(
                f"[DEBUG Heatmap] No data or all zeros: heatmap_data = {heatmap_data}"
            )
            print(
                f"[DEBUG Heatmap] enhanced_results count: {len(enhanced_results)}"
            )
            if enhanced_results:
                print(
                    f"[DEBUG Heatmap] First result enhanced_metrics: {enhanced_results[0].get('enhanced_metrics', {})}"
                )

        return f"""
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>可视化不同因子数量和时间窗口的 ICIR 分布。颜色越绿表示 ICIR 越高（预测稳定性越好）。</p>
            <p>Visualization of ICIR distribution across different factor counts and time windows. Greener colors indicate higher ICIR (better predictive stability).</p>
            <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #fd7e14;">
                <h4>📖 如何阅读热力图 (How to Read the Heatmap):</h4>
                <ul>
                    <li><strong>颜色含义：</strong>
                        <ul>
                            <li><span style="color: #167a3d; font-weight: 600;">深绿色</span>：ICIR很高（> 1.5），表示因子预测能力非常稳定</li>
                            <li><span style="color: #28a745; font-weight: 600;">浅绿色</span>：ICIR较高（1.0 - 1.5），表示因子预测能力稳定</li>
                            <li><span style="color: #ffc107; font-weight: 600;">黄色</span>：ICIR中等（0.5 - 1.0），表示因子预测能力一般</li>
                            <li><span style="color: #dc3545; font-weight: 600;">红色</span>：ICIR较低（< 0.5），表示因子预测能力不稳定</li>
                        </ul>
                    </li>
                    <li><strong>分析要点：</strong>
                        <ul>
                            <li>观察颜色分布模式，找出ICIR高的区域（绿色区域）</li>
                            <li>比较不同因子数量的ICIR分布，识别最优因子数量范围</li>
                            <li>观察不同时间窗口的ICIR一致性，评估因子的时间稳定性</li>
                            <li>寻找颜色均匀的区域，表示该因子数量在不同时间窗口都表现稳定</li>
                        </ul>
                    </li>
                    <li><strong>结论：</strong>热力图提供了ICIR分布的直观可视化。理想的组合应该是在多个时间窗口都显示绿色或浅绿色，且颜色分布相对均匀，这表示该因子数量在不同市场环境下都能保持稳定的预测能力。</li>
                </ul>
            </div>
            <div class="heatmap-container">
                {div_content}
            </div>
            {script_content}
        </div>
        """
    except ImportError:
        # If plotly is not available, return a message
        return """
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>⚠️ Plotly 未安装，无法生成热力图。请安装: pip install plotly</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>⚠️ 生成热力图时出错: {str(e)}</p>
        </div>
        """


def _find_best_combination_by_robustness(
        grid_search_results: list) -> Optional[Dict]:
    """Find the best combination by Robustness Score."""
    if not grid_search_results:
        return None

    best_result = None
    best_robustness = -1

    for result in grid_search_results:
        # Calculate robustness score for this result
        perf = result.get('performance', {}).get('stage3_representatives', {})
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})

        # Get ICIR from ic_statistics or enhanced_metrics
        ic_stats = result.get('ic_statistics', {})
        icir = ic_stats.get('icir')
        if icir is None:
            metrics = result.get('enhanced_metrics', {})
            icir = metrics.get('icir', 0) or 0

        sharpe = financial.get('sharpe_ratio', 0) if financial else 0
        if sharpe == 0:
            metrics = result.get('enhanced_metrics', {})
            sharpe = metrics.get('sharpe', 0) or 0

        max_dd = financial.get('max_drawdown', 0) if financial else 0
        if max_dd == 0:
            metrics = result.get('enhanced_metrics', {})
            max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01

        robustness = (icir * sharpe) / (
            1 + abs(max_dd)) if icir > 0 and sharpe > 0 else 0

        # Store robustness in enhanced_metrics for later use
        if 'enhanced_metrics' not in result:
            result['enhanced_metrics'] = {}
        result['enhanced_metrics']['robustness'] = robustness

        if robustness > best_robustness:
            best_robustness = robustness
            best_result = result

    if best_result:
        params = best_result.get('grid_search_params', {})
        print(
            f"\n🏆 Best combination found (Robustness Score: {best_robustness:.3f}):"
        )
        print(f"   Time Window: {params.get('time_window', 'Unknown')}")
        print(f"   Factor Count: {params.get('factor_count', 'Unknown')}")

    return best_result


def _copy_best_combination_files(best_result: Dict,
                                 grid_search_dir: Path) -> None:
    """Copy best combination's model and related files to grid_search directory."""
    import shutil

    # Try to find the results directory from the best result
    # The results_dir is stored in the result dict when returned from main()
    # But we need to reconstruct it or find it from the saved files

    # Check if result has a results_dir field
    results_dir = best_result.get('results_dir')

    if not results_dir:
        # Try to reconstruct the results directory path
        # Based on the pattern used in run_dimensionality_comparison
        params = best_result.get('grid_search_params', {})
        train_start = params.get('time_window_start')
        train_end = params.get('time_window_end')

        if train_start and train_end:
            # Extract symbol and feature type from grid_search_dir name
            dir_name = grid_search_dir.name
            # Format: SYMBOL_FEATURE_grid_search_TIMESTAMP
            parts = dir_name.split('_grid_search_')
            if len(parts) == 2:
                symbol_feature = parts[0]
                # Format: SYMBOL-FEATURE or SYMBOL_FEATURE
                train_start_date = train_start.replace(
                    "-", "")[:8] if train_start else None
                train_end_date = train_end.replace(
                    "-", "")[:8] if train_end else None

                if train_start_date and train_end_date:
                    # Look for the results directory
                    potential_dir = DIM_COMPARE_RESULTS_ROOT / f"{symbol_feature}_{train_start_date}_{train_end_date}"
                    if potential_dir.exists():
                        results_dir = str(potential_dir)

    if not results_dir or not Path(results_dir).exists():
        print(
            f"⚠️ Could not find results directory for best combination, skipping file copy"
        )
        print(f"   Attempted to find: {results_dir}")
        return

    source_dir = Path(results_dir)
    target_dir = grid_search_dir / "best_combination"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = [
        "production_model.pkl",
        "production_results.json",
        # production_autoencoder.pth removed - autoencoder no longer used
    ]

    # Also copy SHAP directory if it exists
    shap_source = source_dir / "shap"
    if shap_source.exists():
        shap_target = target_dir / "shap"
        try:
            if shap_target.exists():
                shutil.rmtree(shap_target)
            shutil.copytree(shap_source, shap_target)
            print(f"   ✅ Copied SHAP directory")
        except Exception as e:
            print(f"   ⚠️ Failed to copy SHAP directory: {e}")

    # Copy individual files
    copied_count = 0
    for filename in files_to_copy:
        source_file = source_dir / filename
        if source_file.exists():
            try:
                target_file = target_dir / filename
                shutil.copy2(source_file, target_file)
                copied_count += 1
                print(f"   ✅ Copied {filename}")
            except Exception as e:
                print(f"   ⚠️ Failed to copy {filename}: {e}")

    # Also save a summary file with best combination info
    perf = best_result.get('performance', {}).get('stage3_representatives', {})
    financial = perf.get('financial_metrics', {}) if isinstance(perf,
                                                                dict) else {}
    if not financial:
        financial = best_result.get('performance',
                                    {}).get('stage3_representatives_financial',
                                            {})

    classification_metrics = perf.get('classification_metrics',
                                      {}) if isinstance(perf, dict) else {}
    data_info = best_result.get('data_info', {})

    summary = {
        "robustness_score":
        best_result.get('enhanced_metrics', {}).get('robustness', 0),
        "icir":
        best_result.get('enhanced_metrics', {}).get('icir', 0),
        "sharpe_ratio":
        best_result.get('enhanced_metrics', {}).get('sharpe', 0),
        "max_drawdown":
        best_result.get('enhanced_metrics', {}).get('max_drawdown', 0),
        "grid_search_params":
        best_result.get('grid_search_params', {}),
        "source_results_dir":
        results_dir,
        "performance_metrics": {
            "win_rate":
            financial.get('win_rate', 0) if financial else 0,
            "accuracy":
            classification_metrics.get('accuracy', 0)
            if classification_metrics else 0,
            "f1_macro":
            classification_metrics.get('f1_macro', 0)
            if classification_metrics else 0,
        },
        "factor_count":
        data_info.get('stage3_representatives', 0),
        "ic_statistics":
        best_result.get('ic_statistics', {}),
    }

    # Try to extract feature names from production_results.json if available
    production_results_file = source_dir / "production_results.json"
    if production_results_file.exists():
        try:
            with open(production_results_file, 'r') as f:
                prod_results = json.load(f)
                # Try to get feature names from various locations
                # First try selected_features (complete list)
                if 'selected_features' in prod_results:
                    summary['selected_features'] = prod_results[
                        'selected_features']
                # Then try model_info.all_selected_features
                model_info = prod_results.get('model_info', {})
                if 'all_selected_features' in model_info:
                    summary['selected_features'] = model_info[
                        'all_selected_features']
                elif 'feature_names' in model_info:
                    summary['feature_names'] = model_info['feature_names']
        except Exception as e:
            print(f"   ⚠️ Failed to read production_results.json: {e}")

    # Also try to get selected features directly from best_result
    if 'selected_features' not in summary:
        selected_features = best_result.get('selected_features')
        if selected_features:
            summary['selected_features'] = selected_features
        else:
            # Try from model_info
            model_info = best_result.get('model_info', {})
            if 'all_selected_features' in model_info:
                summary['selected_features'] = model_info[
                    'all_selected_features']

    # Generate top_factors.json for compatibility with train/rolling commands
    # Limit features to factor_count from grid_search_params
    if 'selected_features' in summary and summary['selected_features']:
        # Get target factor count from grid_search_params
        target_factor_count = None
        if 'grid_search_params' in summary:
            target_factor_count = summary['grid_search_params'].get(
                'factor_count')

        selected_features = summary['selected_features']

        # If we have a target factor_count, try to limit features using SHAP importance
        if target_factor_count and isinstance(target_factor_count,
                                              int) and target_factor_count > 0:
            # Try to load SHAP importance to get top features
            # Check both target_dir (copied) and source_dir (original)
            shap_importance_file = None
            if (target_dir / "shap" /
                    "stage3_representatives_shap_importance.json").exists():
                shap_importance_file = target_dir / "shap" / "stage3_representatives_shap_importance.json"
            elif source_dir and (
                    Path(source_dir) / "shap" /
                    "stage3_representatives_shap_importance.json").exists():
                shap_importance_file = Path(
                    source_dir
                ) / "shap" / "stage3_representatives_shap_importance.json"

            if shap_importance_file and shap_importance_file.exists():
                try:
                    with open(shap_importance_file, 'r',
                              encoding='utf-8') as f:
                        shap_importance = json.load(f)
                    # Extract top features by SHAP importance
                    top_features_by_shap = [
                        item['feature']
                        for item in shap_importance[:target_factor_count]
                    ]
                    # Filter to only include features that exist in selected_features
                    top_features = [
                        f for f in top_features_by_shap
                        if f in selected_features
                    ]
                    # If we don't have enough from SHAP, fill with remaining features
                    if len(top_features) < target_factor_count:
                        remaining = [
                            f for f in selected_features
                            if f not in top_features
                        ]
                        top_features.extend(remaining[:target_factor_count -
                                                      len(top_features)])
                    selected_features = top_features[:target_factor_count]
                    print(
                        f"   📊 Limited to top {target_factor_count} features based on SHAP importance"
                    )
                except Exception as e:
                    print(
                        f"   ⚠️ Failed to load SHAP importance, using all features: {e}"
                    )
                    # Fallback: just take first N features
                    if len(selected_features) > target_factor_count:
                        selected_features = selected_features[:
                                                              target_factor_count]
                        print(
                            f"   📊 Limited to first {target_factor_count} features"
                        )
            else:
                # No SHAP file, just take first N features
                if len(selected_features) > target_factor_count:
                    selected_features = selected_features[:target_factor_count]
                    print(
                        f"   📊 Limited to first {target_factor_count} features (no SHAP file found)"
                    )

        top_factors_file = target_dir / "top_factors.json"
        try:
            top_factors_data = {
                "top_factors": [{
                    "name": factor
                } for factor in selected_features],
                "count": len(selected_features),
                "source": "grid_search",
                "stage":
                "Stage 3: Representative features (from grid search best combination)",
                "effective": True,
            }
            # Add grid_search_params and metrics if available
            if 'grid_search_params' in summary:
                top_factors_data["grid_search_params"] = summary[
                    'grid_search_params']
            if 'robustness_score' in summary:
                top_factors_data["robustness_score"] = summary[
                    'robustness_score']
            if 'icir' in summary:
                top_factors_data["icir"] = summary['icir']
            if 'sharpe_ratio' in summary:
                top_factors_data["sharpe_ratio"] = summary['sharpe_ratio']

            with open(top_factors_file, 'w', encoding='utf-8') as f:
                json.dump(top_factors_data, f, indent=2, ensure_ascii=False)
            print(
                f"   ✅ Generated top_factors.json with {len(selected_features)} features for compatibility with train/rolling commands"
            )
        except Exception as e:
            print(f"   ⚠️ Failed to generate top_factors.json: {e}")

    summary_file = target_dir / "best_combination_summary.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   ✅ Created best_combination_summary.json")
    except Exception as e:
        print(f"   ⚠️ Failed to create summary file: {e}")

    if copied_count > 0:
        print(
            f"\n📦 Copied {copied_count} file(s) from best combination to: {target_dir}"
        )
    else:
        print(f"\n⚠️ No files were copied. Source directory: {source_dir}")


def generate_grid_search_report(grid_search_results: list, symbol_slug: str,
                                feature_type_slug: str, args) -> str:
    """Generate a comparison matrix report for grid search results."""
    from pathlib import Path
    from time_series_model.pipeline.dimensionality.report_generator import write_html_report

    # Organize results into a matrix
    # Group by time window, then by factor count
    matrix_data = {}
    for result in grid_search_results:
        params = result.get('grid_search_params', {})
        tw_key = params.get('time_window', 'Unknown')
        fc_key = params.get('factor_count', 'Unknown')

        if tw_key not in matrix_data:
            matrix_data[tw_key] = {}
        matrix_data[tw_key][fc_key] = result

    # Extract performance metrics
    # Get primary metric from first result
    first_result = grid_search_results[0]
    perf_stage3 = first_result.get('performance',
                                   {}).get('stage3_representatives', {})

    # Determine task type
    task_type = first_result.get('task_type', 'classification_binary')
    is_classification = task_type.startswith('classification')

    # Build comparison matrix
    time_windows = sorted(
        set(
            r.get('grid_search_params', {}).get('time_window', '')
            for r in grid_search_results))
    factor_counts = sorted(set(
        r.get('grid_search_params', {}).get('factor_count', '')
        for r in grid_search_results),
                           key=lambda x: (x == 'all', x
                                          if isinstance(x, int) else 999999))

    # Create comparison report
    report_data = {
        'timestamp_start': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'timestamp_end': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'symbol': args.symbol,
        'feature_type': args.feature_type,
        'grid_search_results': grid_search_results,
        'matrix_data': matrix_data,
        'time_windows': time_windows,
        'factor_counts': factor_counts,
        'task_type': task_type,
    }

    # Generate HTML report
    grid_search_dir = DIM_COMPARE_RESULTS_ROOT / f"{symbol_slug}_{feature_type_slug}_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    grid_search_dir.mkdir(parents=True, exist_ok=True)

    report_path = grid_search_dir / f"{symbol_slug}_{feature_type_slug}_grid_search_report.html"
    generate_grid_search_html_report(report_data, str(report_path))

    print(f"📊 Grid search comparison report saved to: {report_path}")

    # Calculate enhanced_metrics for all results before finding best
    # This ensures robustness scores are calculated
    for result in grid_search_results:
        perf = result.get('performance', {}).get('stage3_representatives', {})
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})

        ic_stats = result.get('ic_statistics', {})
        icir = ic_stats.get('icir')
        if icir is None:
            ic_mean = ic_stats.get('ic_mean')
            ic_std = ic_stats.get('ic_std')
            if ic_mean is not None and ic_std is not None and ic_std > 0:
                icir = abs(ic_mean) / ic_std

        sharpe = financial.get('sharpe_ratio', 0) if financial else 0
        max_dd = financial.get('max_drawdown', 0) if financial else 0

        if 'enhanced_metrics' not in result:
            result['enhanced_metrics'] = {}
        result['enhanced_metrics']['icir'] = icir
        result['enhanced_metrics']['sharpe'] = sharpe
        result['enhanced_metrics']['max_drawdown'] = max_dd

    # Find best combination by Robustness Score and copy its files
    best_result = _find_best_combination_by_robustness(grid_search_results)
    if best_result:
        _copy_best_combination_files(best_result, grid_search_dir)

    return str(report_path)


def _build_analysis_conclusions(enhanced_results: list, time_windows: list,
                                factor_counts: list,
                                is_classification: bool) -> str:
    """Build textual analysis conclusions for grid search results."""
    if not enhanced_results:
        return "<div class=\"card\"><h3>📊 Analysis Conclusions</h3><p>No results available for analysis.</p></div>"

    # Collect metrics for analysis
    results_by_fc = {}
    for result in enhanced_results:
        params = result.get('grid_search_params', {})
        fc = params.get('factor_count')
        if fc not in results_by_fc:
            results_by_fc[fc] = []
        results_by_fc[fc].append(result)

    # Find best factor count by robustness score
    best_fc = None
    best_robustness = -1
    for fc, results in results_by_fc.items():
        robustness_values = []
        for r in results:
            metrics = r.get('enhanced_metrics', {})
            icir = metrics.get('icir', 0) or 0
            sharpe = metrics.get('sharpe', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
            robustness = (icir * sharpe) / (
                1 + max_dd) if icir > 0 and sharpe > 0 else 0
            robustness_values.append(robustness)
        avg_robustness = sum(robustness_values) / len(
            robustness_values) if robustness_values else 0
        if avg_robustness > best_robustness:
            best_robustness = avg_robustness
            best_fc = fc

    # Analyze ICIR stability across time windows
    icir_stability = {}
    for fc in factor_counts:
        icir_values = []
        for result in enhanced_results:
            params = result.get('grid_search_params', {})
            if params.get('factor_count') == fc:
                icir = result.get('enhanced_metrics', {}).get('icir')
                if icir is not None:
                    icir_values.append(icir)
        if icir_values:
            mean_icir = sum(icir_values) / len(icir_values)
            std_icir = (sum((x - mean_icir)**2
                            for x in icir_values) / len(icir_values))**0.5
            icir_stability[fc] = {'mean': mean_icir, 'std': std_icir}

    # Find most stable factor count (lowest std with high mean)
    most_stable_fc = None
    best_stability_score = -1
    for fc, stats in icir_stability.items():
        if stats['mean'] > 0.5:  # Only consider factor counts with decent ICIR
            stability_score = stats['mean'] / (
                1 + stats['std'])  # Higher mean, lower std is better
            if stability_score > best_stability_score:
                best_stability_score = stability_score
                most_stable_fc = fc

    # Build conclusions HTML
    conclusions_html = "<div class=\"card\"><h3>📊 分析结论 (Analysis Conclusions)</h3>"

    # Optimal factor count
    conclusions_html += "<h4>🎯 最优因子数量 (Optimal Factor Count)</h4>"
    if best_fc is not None:
        conclusions_html += f"<p>基于稳健性得分（Robustness Score）分析，<strong>{best_fc}个因子</strong>是最优选择。</p>"
        conclusions_html += f"<p>Based on Robustness Score analysis, <strong>{best_fc} factors</strong> is the optimal choice.</p>"
        if best_robustness > 0.5:
            conclusions_html += f"<p>该因子数量的平均稳健性得分为 <strong>{best_robustness:.3f}</strong>，表现优秀（> 0.5）。</p>"
        else:
            conclusions_html += f"<p>该因子数量的平均稳健性得分为 <strong>{best_robustness:.3f}</strong>，表现一般（≤ 0.5）。</p>"
    else:
        conclusions_html += "<p>无法确定最优因子数量，请检查数据质量。</p>"

    # Factor stability across time windows
    conclusions_html += "<h4>📈 因子在不同周期的有效性 (Factor Effectiveness Across Time Windows)</h4>"
    if icir_stability:
        conclusions_html += "<ul>"
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            if fc in icir_stability:
                stats = icir_stability[fc]
                conclusions_html += f"<li><strong>{fc}个因子：</strong>"
                conclusions_html += f"平均ICIR = {stats['mean']:.3f}，标准差 = {stats['std']:.3f}。"
                if stats['mean'] > 1.0 and stats['std'] < 0.3:
                    conclusions_html += "✅ 表现优秀且稳定（高ICIR，低波动）。"
                elif stats['mean'] > 0.5:
                    conclusions_html += "⚠️ 表现一般，稳定性有待提升。"
                else:
                    conclusions_html += "❌ 表现较差，不推荐使用。"
                conclusions_html += "</li>"
        conclusions_html += "</ul>"

    if most_stable_fc is not None and most_stable_fc != best_fc:
        conclusions_html += f"<p><strong>💡 稳定性建议：</strong>如果优先考虑因子在不同时间窗口的稳定性，建议选择 <strong>{most_stable_fc}个因子</strong>（ICIR均值高且标准差低）。</p>"

    # Multi-period effectiveness
    conclusions_html += "<h4>🔄 多周期有效性分析 (Multi-Period Effectiveness Analysis)</h4>"
    if len(time_windows) > 1:
        conclusions_html += f"<p>本次测试覆盖了 <strong>{len(time_windows)}</strong> 个不同的时间窗口：</p>"
        conclusions_html += "<ul>"
        for tw in time_windows:
            conclusions_html += f"<li>{tw}</li>"
        conclusions_html += "</ul>"
        conclusions_html += "<p><strong>关键发现：</strong></p>"
        conclusions_html += "<ul>"
        conclusions_html += "<li>如果某个因子数量在所有时间窗口都表现良好（绿色单元格），说明该因子数量具有强的时间稳定性。</li>"
        conclusions_html += "<li>如果某个因子数量只在部分时间窗口表现良好，说明该因子数量可能对特定市场环境敏感。</li>"
        conclusions_html += "<li>建议优先选择在所有或大部分时间窗口都表现稳定的因子数量。</li>"
        conclusions_html += "</ul>"
    else:
        conclusions_html += "<p>本次测试仅使用单一时间窗口，无法评估多周期有效性。建议增加更多时间窗口进行测试。</p>"

    # Final recommendations
    conclusions_html += "<h4>✅ 最终建议 (Final Recommendations)</h4>"
    conclusions_html += "<ol>"
    if best_fc is not None:
        conclusions_html += f"<li><strong>推荐因子数量：{best_fc}个</strong> - 基于稳健性得分分析，这是综合表现最优的选择。</li>"
    if most_stable_fc is not None and most_stable_fc != best_fc:
        conclusions_html += f"<li><strong>备选因子数量：{most_stable_fc}个</strong> - 如果更关注时间稳定性，可以考虑此选项。</li>"
    conclusions_html += "<li><strong>验证建议：</strong>在实际使用前，建议在最新的数据上验证所选因子数量的表现。</li>"
    conclusions_html += "<li><strong>持续监控：</strong>定期重新评估因子有效性，因为市场环境会发生变化。</li>"
    conclusions_html += "</ol>"

    conclusions_html += "</div>"
    return conclusions_html


def generate_grid_search_html_report(report_data: Dict,
                                     html_path: str) -> None:
    """Generate HTML report for grid search results with enhanced metrics and visualizations."""
    import os
    import json
    from time_series_model.pipeline.dimensionality.report_generator import _format_float

    time_windows = report_data['time_windows']
    factor_counts = report_data['factor_counts']
    grid_search_results = report_data['grid_search_results']
    task_type = report_data['task_type']
    is_classification = task_type.startswith('classification')

    # Calculate ICIR and robustness metrics for each result
    enhanced_results = []
    for result in grid_search_results:
        perf = result.get('performance', {}).get('stage3_representatives', {})
        # financial_metrics is stored inside perf_reps, not as a separate field
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        # Also check the separate financial field as fallback
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})

        # Debug: print structure for first result
        if len(enhanced_results) == 0:
            print(f"\n[DEBUG] First result structure:")
            print(f"  result keys: {list(result.keys())}")
            print(
                f"  performance keys: {list(result.get('performance', {}).keys())}"
            )
            print(
                f"  perf keys: {list(perf.keys()) if isinstance(perf, dict) else 'Not a dict'}"
            )
            print(
                f"  financial keys: {list(financial.keys()) if financial else 'Empty'}"
            )
            print(f"  financial content: {financial}")
            if isinstance(perf, dict):
                print(
                    f"  perf.get('financial_metrics'): {perf.get('financial_metrics', 'NOT FOUND')}"
                )
            print(
                f"  result.get('performance', {{}}).get('stage3_representatives_financial'): {result.get('performance', {}).get('stage3_representatives_financial', 'NOT FOUND')}"
            )

        # Extract metrics
        if is_classification:
            # win_rate is stored in financial_metrics, not in performance directly
            win_rate = financial.get('win_rate', 0) if financial else 0
            # Also check performance for win_rate as fallback
            if win_rate == 0:
                win_rate = perf.get('win_rate', 0)
            # Also check classification_metrics for accuracy as fallback for win_rate
            if win_rate == 0:
                classification_metrics = perf.get('classification_metrics',
                                                  {}) if isinstance(
                                                      perf, dict) else {}
                if classification_metrics:
                    # Use accuracy as a proxy for win_rate if available
                    accuracy = classification_metrics.get('accuracy', 0)
                    if accuracy > 0:
                        win_rate = accuracy
            sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            max_dd = financial.get('max_drawdown', 0) if financial else 0

            # Debug: print extracted values for first result
            if len(enhanced_results) == 0:
                print(f"  Extracted win_rate: {win_rate}")
                print(f"  Extracted sharpe: {sharpe}")
                print(f"  Extracted max_dd: {max_dd}")
                # Check if perf is empty or None
                if not perf or (isinstance(perf, dict) and len(perf) == 0):
                    print(f"  ⚠️ WARNING: perf is empty or None!")
                if not financial or (isinstance(financial, dict)
                                     and len(financial) == 0):
                    print(f"  ⚠️ WARNING: financial is empty or None!")
        else:
            r2 = perf.get('r2', 0)
            sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            max_dd = financial.get('max_drawdown', 0) if financial else 0

        # Calculate ICIR if IC data is available
        ic_stats = result.get('ic_statistics', {})
        ic_mean = ic_stats.get('ic_mean', None)
        ic_std = ic_stats.get('ic_std', None)
        icir = ic_stats.get('icir', None)
        if icir is None and ic_mean is not None and ic_std is not None and ic_std > 0:
            icir = abs(ic_mean) / ic_std

        # Debug: print IC stats for first result
        if len(enhanced_results) == 0:
            print(f"  IC stats: {ic_stats}")
            print(f"  Calculated ICIR: {icir}")
            params = result.get('grid_search_params', {})
            print(f"  Factor count: {params.get('factor_count')}")
            print(f"  Time window: {params.get('time_window')}")

        enhanced_results.append({
            **result, 'enhanced_metrics': {
                'icir': icir,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
            }
        })

    # Build multiple comparison matrices
    # Matrix 1: Primary metric (Win Rate or R²)
    matrix_html = "<div class=\"card\"><h3>📊 Grid Search Comparison Matrix - Primary Metric</h3>"
    matrix_html += "<p>Comparison of different factor counts and time windows</p>"

    # Determine primary metric
    if is_classification:
        primary_metric = 'win_rate'
        metric_display = 'Directional Win Rate'
    else:
        primary_metric = 'r2'
        metric_display = 'R²'

    # Build table header
    matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        matrix_html += f"<th>Factors: {fc}</th>"
    matrix_html += "</tr>"

    # Build table rows
    for tw in time_windows:
        matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            # Find result for this combination
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                perf = result.get('performance',
                                  {}).get('stage3_representatives', {})
                # financial_metrics is stored inside perf_reps
                financial = perf.get('financial_metrics', {}) if isinstance(
                    perf, dict) else {}
                if not financial:
                    financial = result.get('performance', {}).get(
                        'stage3_representatives_financial', {})
                if is_classification:
                    # win_rate is stored in financial_metrics
                    metric_val = financial.get('win_rate',
                                               0) if financial else 0
                    # Fallback to performance if not in financial
                    if metric_val == 0:
                        metric_val = perf.get('win_rate', 0)
                    # Also check classification_metrics for accuracy as fallback
                    if metric_val == 0:
                        classification_metrics = perf.get(
                            'classification_metrics', {}) if isinstance(
                                perf, dict) else {}
                        if classification_metrics:
                            accuracy = classification_metrics.get(
                                'accuracy', 0)
                            if accuracy > 0:
                                metric_val = accuracy
                    cell_content = f"{_format_float(metric_val * 100, 2)}%"
                else:
                    metric_val = perf.get('r2', 0)
                    cell_content = _format_float(metric_val, 4)

                # Add color coding
                color_class = "good" if metric_val > 0.5 else "warn" if metric_val > 0 else "bad"
                matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                matrix_html += "<td>-</td>"
        matrix_html += "</tr>"

    matrix_html += "</table>"
    matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>表格结构：</strong>行表示不同的时间窗口，列表示不同的因子数量。每个单元格显示该组合下的{metric_display}值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：表现优秀（{metric_display} > 0.5）</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：表现一般（{metric_display} > 0）</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：表现较差（{metric_display} ≤ 0）</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>比较同一时间窗口下不同因子数量的表现，找出最优因子数量</li>
                    <li>比较同一因子数量下不同时间窗口的表现，评估因子在不同时期的稳定性</li>
                    <li>关注绿色单元格，这些是表现最好的组合</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格帮助识别在特定时间窗口下，使用多少因子能获得最佳{metric_display}。通常，因子数量不是越多越好，需要找到性能与复杂度的平衡点。</li>
        </ul>
    </div>
    </div>""".format(metric_display=metric_display)

    # Matrix 2: ICIR (if available)
    icir_matrix_html = ""
    if any(
            r.get('enhanced_metrics', {}).get('icir') is not None
            for r in enhanced_results):
        icir_matrix_html = "<div class=\"card\"><h3>📈 ICIR (Information Coefficient Information Ratio) Matrix</h3>"
        icir_matrix_html += "<p>ICIR = |Mean IC| / Std(IC) - Higher is better (indicates stable predictive power)</p>"
        icir_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
        icir_matrix_html += "<tr><th>Time Window</th>"
        for fc in factor_counts:
            icir_matrix_html += f"<th>Factors: {fc}</th>"
        icir_matrix_html += "</tr>"

        for tw in time_windows:
            icir_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
            for fc in factor_counts:
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    if icir is not None:
                        cell_content = _format_float(icir, 3)
                        color_class = "good" if icir > 1.0 else "warn" if icir > 0.5 else "bad"
                        icir_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
                    else:
                        icir_matrix_html += "<td>-</td>"
                else:
                    icir_matrix_html += "<td>-</td>"
            icir_matrix_html += "</tr>"

        icir_matrix_html += "</table>"
        icir_matrix_html += """
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #28a745;">
            <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
            <ul>
                <li><strong>ICIR定义：</strong>ICIR = |平均IC| / IC标准差，衡量因子的预测稳定性和有效性。ICIR越高，表示因子在不同时期的表现越稳定。</li>
                <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的ICIR值。</li>
                <li><strong>颜色编码：</strong>
                    <ul>
                        <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：ICIR > 1.0，表示因子具有稳定的预测能力</li>
                        <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0.5 < ICIR ≤ 1.0，表示因子预测能力一般</li>
                        <li><span style="color: #c53030; font-weight: 600;">红色</span>：ICIR ≤ 0.5，表示因子预测能力不稳定</li>
                    </ul>
                </li>
                <li><strong>分析要点：</strong>
                    <ul>
                        <li>ICIR > 1.0 是理想状态，表示因子的平均预测能力超过其波动性</li>
                        <li>比较不同因子数量的ICIR，找出在保持高ICIR的前提下，因子数量最少的组合</li>
                        <li>观察同一因子数量在不同时间窗口的ICIR，评估因子的时间稳定性</li>
                    </ul>
                </li>
                <li><strong>结论：</strong>此表格是选择因子的关键指标。高ICIR意味着因子不仅在历史数据上有效，而且在不同市场环境下都能保持稳定的预测能力。优先选择ICIR > 1.0且在不同时间窗口都表现稳定的因子组合。</li>
            </ul>
        </div>
        </div>"""

    # Matrix 3: Sharpe Ratio
    sharpe_matrix_html = "<div class=\"card\"><h3>💰 Sharpe Ratio Matrix</h3>"
    sharpe_matrix_html += "<p>Risk-adjusted return metric - Higher is better</p>"
    sharpe_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    sharpe_matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        sharpe_matrix_html += f"<th>Factors: {fc}</th>"
    sharpe_matrix_html += "</tr>"

    for tw in time_windows:
        sharpe_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                sharpe = result.get('enhanced_metrics', {}).get('sharpe', 0)
                cell_content = _format_float(sharpe, 3)
                color_class = "good" if sharpe > 1.0 else "warn" if sharpe > 0 else "bad"
                sharpe_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                sharpe_matrix_html += "<td>-</td>"
        sharpe_matrix_html += "</tr>"

    sharpe_matrix_html += "</table>"
    sharpe_matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #ffc107;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>Sharpe Ratio定义：</strong>夏普比率 = (策略收益率 - 无风险收益率) / 收益率标准差，衡量风险调整后的收益表现。Sharpe Ratio越高，表示在承担相同风险的情况下，获得的超额收益越多。</li>
            <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的Sharpe Ratio值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：Sharpe > 1.0，表示策略表现优秀</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0 < Sharpe ≤ 1.0，表示策略表现一般</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：Sharpe ≤ 0，表示策略表现不佳</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>Sharpe Ratio > 1.0 通常被认为是可接受的策略表现</li>
                    <li>Sharpe Ratio > 2.0 表示策略表现优秀</li>
                    <li>比较不同因子数量和时间窗口的Sharpe Ratio，找出风险调整后收益最高的组合</li>
                    <li>注意：此指标需要真实的回测数据，如果数据不可用，可能显示为0</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格帮助评估策略的实际交易表现。高Sharpe Ratio意味着策略不仅能产生收益，而且风险控制得当。结合ICIR和Sharpe Ratio，可以全面评估因子的有效性和策略的实用性。</li>
        </ul>
    </div>
    </div>"""

    # Matrix 4: Robustness Score (ICIR-weighted composite)
    robustness_matrix_html = "<div class=\"card\"><h3>🛡️ Robustness Score Matrix</h3>"
    robustness_matrix_html += "<p>Composite score: ICIR × Sharpe / (1 + |Max Drawdown|) - Higher is better</p>"
    robustness_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    robustness_matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        robustness_matrix_html += f"<th>Factors: {fc}</th>"
    robustness_matrix_html += "</tr>"

    # Calculate robustness scores
    robustness_scores = {}
    for tw in time_windows:
        robustness_scores[tw] = {}
        for fc in factor_counts:
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                metrics = result.get('enhanced_metrics', {})
                icir = metrics.get('icir', 0) or 0
                sharpe = metrics.get('sharpe', 0) or 0
                max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01

                # Robustness score: ICIR × Sharpe / (1 + |Max Drawdown|)
                robustness = (icir * sharpe) / (
                    1 + max_dd) if icir > 0 and sharpe > 0 else 0
                robustness_scores[tw][fc] = robustness
            else:
                robustness_scores[tw][fc] = None

    for tw in time_windows:
        robustness_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            score = robustness_scores[tw].get(fc)
            if score is not None:
                cell_content = _format_float(score, 3)
                color_class = "good" if score > 0.5 else "warn" if score > 0 else "bad"
                robustness_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                robustness_matrix_html += "<td>-</td>"
        robustness_matrix_html += "</tr>"

    robustness_matrix_html += "</table>"
    robustness_matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #6f42c1;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>Robustness Score定义：</strong>稳健性得分 = ICIR × Sharpe Ratio / (1 + |最大回撤|)，这是一个综合指标，同时考虑了因子的预测稳定性（ICIR）、策略的风险调整收益（Sharpe Ratio）和风险控制（最大回撤）。</li>
            <li><strong>计算公式说明：</strong>
                <ul>
                    <li>ICIR：衡量因子预测的稳定性</li>
                    <li>Sharpe Ratio：衡量策略的风险调整收益</li>
                    <li>最大回撤：衡量策略的最大风险</li>
                    <li>分母 (1 + |最大回撤|)：惩罚高回撤的策略，回撤越大，得分越低</li>
                </ul>
            </li>
            <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的Robustness Score值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：Robustness Score > 0.5，表示综合表现优秀</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0 < Robustness Score ≤ 0.5，表示综合表现一般</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：Robustness Score ≤ 0，表示综合表现不佳</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>这是最全面的评估指标，综合考虑了预测能力、收益和风险</li>
                    <li>优先选择Robustness Score最高的组合，因为它平衡了所有关键因素</li>
                    <li>比较不同因子数量的Robustness Score，找出最优的因子数量</li>
                    <li>观察不同时间窗口的Robustness Score，评估策略的长期稳定性</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格是选择最优参数组合的最重要参考。高Robustness Score意味着因子组合不仅在预测上有效，而且在实际交易中能产生稳定的风险调整收益。建议优先选择Robustness Score > 0.5且在不同时间窗口都表现稳定的组合。</li>
        </ul>
    </div>
    </div>"""

    # Build detailed results section with enhanced metrics
    details_html = "<div class=\"card\"><h3>📋 Detailed Results</h3>"
    for i, result in enumerate(enhanced_results, 1):
        params = result.get('grid_search_params', {})
        perf = result.get('performance', {}).get('stage3_representatives', {})
        # financial_metrics is stored inside perf_reps
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})
        metrics = result.get('enhanced_metrics', {})

        details_html += f"<h4>Combination {i}: {params.get('time_window')} | Factors: {params.get('factor_count')}</h4>"
        details_html += "<table class=\"metric-table\">"

        if is_classification:
            # win_rate is in financial_metrics, not directly in perf
            win_rate = financial.get('win_rate', 0) if financial else 0
            if win_rate == 0:
                win_rate = perf.get('win_rate', 0)
            # f1_macro and accuracy are in classification_metrics
            classification_metrics = perf.get('classification_metrics',
                                              {}) if isinstance(perf,
                                                                dict) else {}
            f1_macro = classification_metrics.get(
                'f1_macro', 0) if classification_metrics else 0
            accuracy = classification_metrics.get(
                'accuracy', 0) if classification_metrics else 0

            details_html += f"<tr><th>Directional Win Rate</th><td>{_format_float(win_rate * 100, 2)}%</td></tr>"
            details_html += f"<tr><th>F1 (Macro)</th><td>{_format_float(f1_macro, 4)}</td></tr>"
            details_html += f"<tr><th>Accuracy</th><td>{_format_float(accuracy * 100, 2)}%</td></tr>"
        else:
            details_html += f"<tr><th>R²</th><td>{_format_float(perf.get('r2', 0), 4)}</td></tr>"
            details_html += f"<tr><th>RMSE</th><td>{_format_float(perf.get('rmse', 0), 4)}</td></tr>"
            details_html += f"<tr><th>MAE</th><td>{_format_float(perf.get('mae', 0), 4)}</td></tr>"

        # Add financial metrics
        if financial:
            details_html += f"<tr><th>Sharpe Ratio</th><td>{_format_float(financial.get('sharpe_ratio', 0), 3)}</td></tr>"
            details_html += f"<tr><th>Max Drawdown</th><td>{_format_float(financial.get('max_drawdown', 0) * 100, 2)}%</td></tr>"
            details_html += f"<tr><th>Total Return</th><td>{_format_float(financial.get('total_return', 0) * 100, 2)}%</td></tr>"

        # Add ICIR if available
        if metrics.get('icir') is not None:
            details_html += f"<tr><th>ICIR</th><td>{_format_float(metrics.get('icir'), 3)}</td></tr>"

        # Add robustness score
        icir = metrics.get('icir', 0) or 0
        sharpe = metrics.get('sharpe', 0) or 0
        max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
        robustness = (icir *
                      sharpe) / (1 + max_dd) if icir > 0 and sharpe > 0 else 0
        details_html += f"<tr><th>Robustness Score</th><td>{_format_float(robustness, 3)}</td></tr>"

        details_html += "</table>"
        details_html += """
        <div style="margin-top: 15px; padding: 12px; background: #e9ecef; border-radius: 5px; font-size: 0.9em;">
            <strong>📖 指标说明：</strong>
            <ul style="margin: 5px 0;">
                <li><strong>Directional Win Rate / R²：</strong>主要性能指标。分类任务使用胜率，回归任务使用R²。值越高越好。</li>
                <li><strong>F1 (Macro) / RMSE / MAE：</strong>辅助性能指标。F1用于分类，RMSE/MAE用于回归。F1越高越好，RMSE/MAE越低越好。</li>
                <li><strong>Sharpe Ratio：</strong>风险调整收益。> 1.0表示表现良好，> 2.0表示表现优秀。</li>
                <li><strong>Max Drawdown：</strong>最大回撤，衡量策略的最大风险。绝对值越小越好。</li>
                <li><strong>Total Return：</strong>总收益率。正值表示盈利，负值表示亏损。</li>
                <li><strong>ICIR：</strong>因子预测稳定性。> 1.0表示因子具有稳定的预测能力。</li>
                <li><strong>Robustness Score：</strong>综合稳健性得分，综合考虑ICIR、Sharpe和回撤。> 0.5表示综合表现优秀。</li>
            </ul>
        </div>
        <br/>"""

    details_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #dc3545;">
        <h4>📖 如何阅读详细结果表格 (How to Read Detailed Results):</h4>
        <ul>
            <li><strong>表格结构：</strong>每个组合（时间窗口 + 因子数量）都有一个独立的详细结果表格，显示该组合的所有关键指标。</li>
            <li><strong>性能指标：</strong>
                <ul>
                    <li>主要关注<strong>Directional Win Rate</strong>（分类）或<strong>R²</strong>（回归），这是评估模型预测能力的主要指标</li>
                    <li>辅助指标（F1、Accuracy、RMSE、MAE）提供更全面的性能评估</li>
                </ul>
            </li>
            <li><strong>金融指标：</strong>
                <ul>
                    <li><strong>Sharpe Ratio</strong>：评估策略的风险调整收益，是实际交易中最重要的指标之一</li>
                    <li><strong>Max Drawdown</strong>：评估策略的最大风险，帮助了解最坏情况下的损失</li>
                    <li><strong>Total Return</strong>：评估策略的总收益表现</li>
                </ul>
            </li>
            <li><strong>因子质量指标：</strong>
                <ul>
                    <li><strong>ICIR</strong>：评估因子的预测稳定性，高ICIR表示因子在不同时期都能保持有效</li>
                    <li><strong>Robustness Score</strong>：综合评估因子组合的稳健性，是最全面的评估指标</li>
                </ul>
            </li>
            <li><strong>分析建议：</strong>
                <ul>
                    <li>优先查看Robustness Score，这是最全面的评估指标</li>
                    <li>结合ICIR和Sharpe Ratio，评估因子的预测能力和实际交易表现</li>
                    <li>注意Max Drawdown，确保风险在可接受范围内</li>
                    <li>比较不同组合的详细结果，找出最优参数配置</li>
                </ul>
            </li>
            <li><strong>结论：</strong>详细结果表格提供了每个参数组合的完整评估。通过对比不同组合的各项指标，可以全面了解每个组合的优势和劣势，从而做出最优的参数选择决策。</li>
        </ul>
    </div>
    </div>"""

    # Build ICIR trend analysis (Factor Count vs ICIR)
    icir_trend_html = ""
    if any(
            r.get('enhanced_metrics', {}).get('icir') is not None
            for r in enhanced_results):
        icir_trend_html = "<div class=\"card\"><h3>📈 ICIR Trend Analysis</h3>"
        icir_trend_html += "<p>ICIR vs Factor Count for each time window - Look for plateau points</p>"
        icir_trend_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
        icir_trend_html += "<tr><th>Factor Count</th>"
        for tw in time_windows:
            icir_trend_html += f"<th>{tw}</th>"
        icir_trend_html += "<th>Mean ICIR</th><th>Std(ICIR)</th></tr>"

        # Calculate mean and std ICIR across time windows for each factor count
        for fc in factor_counts:
            icir_values = []
            icir_trend_html += f"<tr><td><strong>{fc}</strong></td>"
            for tw in time_windows:
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    if icir is not None:
                        icir_values.append(icir)
                        icir_trend_html += f"<td>{_format_float(icir, 3)}</td>"
                    else:
                        icir_trend_html += "<td>-</td>"
                else:
                    icir_trend_html += "<td>-</td>"

            # Mean and std across time windows
            if icir_values:
                mean_icir = sum(icir_values) / len(icir_values)
                std_icir = (sum((x - mean_icir)**2
                                for x in icir_values) / len(icir_values))**0.5
                icir_trend_html += f"<td>{_format_float(mean_icir, 3)}</td>"
                icir_trend_html += f"<td>{_format_float(std_icir, 3)}</td>"
            else:
                icir_trend_html += "<td>-</td><td>-</td>"
            icir_trend_html += "</tr>"

        icir_trend_html += "</table>"

        # Generate multiple 3D visualizations for Plateau Point analysis
        icir_trend_html += "<h4>🎯 3D Plateau Point 分析 (3D Plateau Point Analysis)</h4>"

        # ICIR 3D plot
        icir_3d_html = _generate_icir_3d_plot(enhanced_results, time_windows,
                                              factor_counts)
        icir_trend_html += icir_3d_html

        # Robustness Score 3D plot
        def get_robustness(result):
            metrics = result.get('enhanced_metrics', {})
            icir = metrics.get('icir', 0) or 0
            sharpe = metrics.get('sharpe', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
            return (icir * sharpe) / (1 +
                                      max_dd) if icir > 0 and sharpe > 0 else 0

        robustness_3d_html = _generate_metric_3d_plot(enhanced_results,
                                                      time_windows,
                                                      factor_counts,
                                                      'robustness',
                                                      'Robustness Score',
                                                      get_robustness,
                                                      color_thresholds={
                                                          'good': 0.5,
                                                          'warn': 0.2,
                                                          'bad': 0.0
                                                      })
        icir_trend_html += robustness_3d_html

        # Sharpe Ratio 3D plot
        def get_sharpe(result):
            # Try to get from enhanced_metrics first
            sharpe = result.get('enhanced_metrics', {}).get('sharpe')
            if sharpe is None or sharpe == 0:
                # Fallback: extract from performance
                perf = result.get('performance',
                                  {}).get('stage3_representatives', {})
                financial = perf.get('financial_metrics', {}) if isinstance(
                    perf, dict) else {}
                if not financial:
                    financial = result.get('performance', {}).get(
                        'stage3_representatives_financial', {})
                sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            return sharpe

        sharpe_3d_html = _generate_metric_3d_plot(enhanced_results,
                                                  time_windows,
                                                  factor_counts,
                                                  'sharpe',
                                                  'Sharpe Ratio',
                                                  get_sharpe,
                                                  color_thresholds={
                                                      'good': 1.0,
                                                      'warn': 0.0,
                                                      'bad': -1.0
                                                  })
        icir_trend_html += sharpe_3d_html

        # Primary metric 3D plot (Win Rate or R²)
        def get_primary_metric(result):
            perf = result.get('performance', {}).get('stage3_representatives',
                                                     {})
            # financial_metrics is stored inside perf_reps
            financial = perf.get('financial_metrics', {}) if isinstance(
                perf, dict) else {}
            if not financial:
                financial = result.get('performance', {}).get(
                    'stage3_representatives_financial', {})
            if is_classification:
                # win_rate is stored in financial_metrics
                win_rate = financial.get('win_rate') if financial else None
                if win_rate is None or win_rate == 0:
                    win_rate = perf.get('win_rate')
                return win_rate
            else:
                return perf.get('r2')

        primary_metric_label = 'Directional Win Rate' if is_classification else 'R²'
        primary_3d_html = _generate_metric_3d_plot(
            enhanced_results,
            time_windows,
            factor_counts,
            'primary',
            primary_metric_label,
            get_primary_metric,
            color_thresholds={
                'good': 0.5,
                'warn': 0.0,
                'bad': -0.5
            } if is_classification else {
                'good': 0.5,
                'warn': 0.0,
                'bad': -1.0
            })
        icir_trend_html += primary_3d_html

        # Add summary section for multiple 3D visualizations
        icir_trend_html += """
        <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 5px; border-left: 4px solid #0066cc;">
            <h4>🎯 多指标Plateau Point综合分析 (Multi-Metric Plateau Point Analysis)</h4>
            <p><strong>为什么需要多个3D可视化？</strong></p>
            <ul>
                <li><strong>ICIR 3D图：</strong>识别因子预测稳定性的Plateau Point。高ICIR表示因子在不同时期都能保持有效预测。</li>
                <li><strong>Robustness Score 3D图：</strong>识别综合稳健性的Plateau Point。这是最全面的指标，综合考虑了预测能力、收益和风险。</li>
                <li><strong>Sharpe Ratio 3D图：</strong>识别风险调整收益的Plateau Point。高Sharpe Ratio表示策略在控制风险的同时获得良好收益。</li>
                <li><strong>Primary Metric 3D图：</strong>识别主要性能指标的Plateau Point。对于分类任务是胜率，对于回归任务是R²。</li>
            </ul>
            <p><strong>如何综合使用这些3D图？</strong></p>
            <ol>
                <li><strong>第一步：</strong>查看Robustness Score 3D图，找出综合表现最优的因子数量范围（绿色区域且表面平坦的位置）。</li>
                <li><strong>第二步：</strong>验证ICIR 3D图，确保该因子数量在ICIR上也表现稳定（高ICIR且在不同时间窗口都保持稳定）。</li>
                <li><strong>第三步：</strong>检查Sharpe Ratio 3D图，确认该因子数量在实际交易中能产生良好的风险调整收益。</li>
                <li><strong>第四步：</strong>参考Primary Metric 3D图，确保主要性能指标也达到预期水平。</li>
                <li><strong>第五步：</strong>选择在所有或大部分指标上都显示Plateau Point的因子数量，这表示该数量是最优选择。</li>
            </ol>
            <p><strong>💡 关键洞察：</strong>理想的Plateau Point应该在不同指标的不同3D图中都显示为平坦区域，且在不同时间窗口（Y轴）上都保持相对稳定的高度。如果某个因子数量在多个指标的3D图中都显示为Plateau Point，那么它就是最优选择。</p>
        </div>
        """

        icir_trend_html += """
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #17a2b8;">
            <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
            <ul>
                <li><strong>表格结构：</strong>行表示不同的因子数量，列表示不同的时间窗口。最后两列显示每个因子数量在所有时间窗口上的平均ICIR和标准差。</li>
                <li><strong>Mean ICIR列：</strong>计算该因子数量在所有时间窗口上的平均ICIR值。平均值越高，表示该因子数量在不同时期的表现越稳定。</li>
                <li><strong>Std(ICIR)列：</strong>计算该因子数量在所有时间窗口上的ICIR标准差。标准差越小，表示该因子数量在不同时期的表现越一致，稳定性越好。</li>
                <li><strong>分析要点：</strong>
                    <ul>
                        <li><strong>寻找平台点（Plateau Point）：</strong>找出ICIR不再显著下降的最小因子数量。例如，如果120个因子和60个因子的ICIR相近，但30个因子的ICIR明显下降，那么60个因子可能是平台点。</li>
                        <li><strong>评估稳定性：</strong>比较不同因子数量的Std(ICIR)。Std(ICIR)越小，表示该因子数量在不同时间窗口的表现越一致，越稳定。</li>
                        <li><strong>平衡性能与复杂度：</strong>在Mean ICIR高且Std(ICIR)低的前提下，选择因子数量最少的组合，以降低模型复杂度并提高可解释性。</li>
                        <li><strong>结合3D可视化：</strong>使用上方的3D图形可以更直观地识别Plateau Point。在3D图形中，Plateau Point表现为ICIR值达到较高水平后，表面变得平坦的区域。</li>
                    </ul>
                </li>
                <li><strong>结论：</strong>此表格帮助确定最优因子数量。理想的组合是：Mean ICIR高（> 1.0）、Std(ICIR)低（< 0.3），且因子数量尽可能少。这表示该因子数量既能保持高预测能力，又能在不同市场环境下保持稳定，同时避免了过度复杂化。结合3D可视化，可以更准确地识别Plateau Point。</li>
            </ul>
        </div>
        </div>"""

    # Build analysis conclusions
    analysis_conclusions_html = _build_analysis_conclusions(
        enhanced_results, time_windows, factor_counts, is_classification)

    # Generate ICIR heatmap
    heatmap_html = _generate_icir_heatmap(enhanced_results, time_windows,
                                          factor_counts)

    # Build full HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Grid Search Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .card {{ background: #fff; border-radius: 10px; padding: 18px 22px; box-shadow: 0 10px 24px rgba(27,39,53,0.1); margin: 20px 0; }}
        .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        .metric-table th, .metric-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .metric-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .good {{ color: #167a3d; font-weight: 600; }}
        .warn {{ color: #b36b00; font-weight: 600; }}
        .bad {{ color: #c53030; font-weight: 600; }}
        .heatmap-container {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <h1>🔍 Grid Search Comparison Report</h1>
    <div class="card">
        <h3>Configuration</h3>
        <p><strong>Symbol:</strong> {report_data.get('symbol', 'N/A')}</p>
        <p><strong>Feature Type:</strong> {report_data.get('feature_type', 'N/A')}</p>
        <p><strong>Task Type:</strong> {task_type}</p>
        <p><strong>Time Windows Tested:</strong> {len(time_windows)}</p>
        <p><strong>Factor Counts Tested:</strong> {len(factor_counts)}</p>
        <p><strong>Total Combinations:</strong> {len(grid_search_results)}</p>
    </div>
    
    {matrix_html}
    {icir_matrix_html}
    {sharpe_matrix_html}
    {robustness_matrix_html}
    {icir_trend_html}
    {heatmap_html}
    {details_html}
    
    {analysis_conclusions_html}
    
    <div class="card">
        <h3>💡 Interpretation Guide</h3>
        <h4>📊 How to Read the Matrices:</h4>
        <ul>
            <li><strong>Primary Metric Matrix:</strong> Compare {metric_display} across different factor counts and time windows. Higher values (green) indicate better performance.</li>
            <li><strong>ICIR Matrix:</strong> ICIR = |Mean IC| / Std(IC). Higher ICIR indicates more stable predictive power. Look for ICIR > 1.0 (green).</li>
            <li><strong>Sharpe Ratio Matrix:</strong> Risk-adjusted return metric. Higher is better. Look for Sharpe > 1.0 (green).</li>
            <li><strong>Robustness Score Matrix:</strong> Composite score combining ICIR, Sharpe, and Max Drawdown. This is the most comprehensive metric - higher is better.</li>
        </ul>
        <h4>📈 ICIR Trend Analysis:</h4>
        <ul>
            <li>Look for the <strong>plateau point</strong>: the smallest factor count where ICIR doesn't drop significantly</li>
            <li>Lower <strong>Std(ICIR)</strong> across time windows indicates better stability</li>
            <li>Optimal factor count is often where Mean ICIR is high AND Std(ICIR) is low</li>
        </ul>
        <h4>🎯 Selection Strategy:</h4>
        <ul>
            <li><strong>Step 1:</strong> Identify factor counts with high Robustness Score (green cells)</li>
            <li><strong>Step 2:</strong> Check ICIR Trend - find the plateau point</li>
            <li><strong>Step 3:</strong> Verify consistency across time windows (low Std(ICIR))</li>
            <li><strong>Step 4:</strong> Choose the smallest factor count that meets all criteria</li>
        </ul>
        <p><strong>💡 Key Insight:</strong> The optimal solution is often NOT the one with the highest primary metric, but the one with the best balance of performance and stability (high ICIR, low variance across time windows).</p>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"📝 Grid search HTML report written to: {html_path}")


def main() -> Tuple[Dict, any, str]:
    global DIM_COMPARE_RESULTS_ROOT
    parser = argparse.ArgumentParser(
        description=
        "Dimensionality reduction comparison: evaluate feature reduction stages (All → IC-filtered → Representatives)",
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
    # Autoencoder arguments removed - no longer used
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
        "--horizons",
        type=str,
        default="1,5,10,15",
        help=
        "Comma-separated list of forward bars for multi-horizon labels (e.g., 1,5,10,15)",
    )
    parser.add_argument(
        "--binary-signals",
        action="store_true",
        default=True,  # Default to binary classification (2-class)
        help=
        "Use binary labels (1=Long, 0=Short). Default: True. Threshold controlled by --label-threshold",
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
    # --enable-autoencoder removed - autoencoder no longer used
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
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5T",
        help=
        "Timeframe for data resampling (e.g., 5T, 15T, 60T, 240T). Default: 5T",
    )
    parser.add_argument(
        "--enable-stability-validation",
        action="store_true",
        help=
        "Enable stability validation: use recent data for factor selection, validate on longer historical data",
    )
    parser.add_argument(
        "--validation-start",
        default=None,
        help=
        "Start date (YYYY-MM-DD) for stability validation period. If not provided and --enable-stability-validation is set, automatically uses train-start minus 2-3 years",
    )
    parser.add_argument(
        "--validation-years",
        type=int,
        default=3,
        help=
        "Number of years to look back for stability validation (default: 3). Used when --enable-stability-validation is set and --validation-start is not provided",
    )
    parser.add_argument(
        "--factor-counts",
        type=str,
        default=None,
        help=
        "Comma-separated list of factor counts to test (e.g., 'all,120,60,30,15,8'). 'all' means use all available features. If not provided, uses --top-k or default 120",
    )
    parser.add_argument(
        "--time-windows",
        type=str,
        default=None,
        help=
        "Comma-separated list of time windows to test (e.g., '2020-01-01:2025-12-31,2022-01-01:2025-12-31,2024-01-01:2025-12-31'). Format: START:END. If not provided, uses --train-start and --train-end",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help=
        "Enable grid search mode: test all combinations of factor counts and time windows",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=["sharpe", "f1", "r2", "composite"],
        help=
        "Metric to use for feature selection scoring: sharpe | f1 | r2 | composite (default: composite)",
    )
    parser.add_argument(
        "--max-dd-threshold",
        type=float,
        default=-20.0,
        help=
        "Maximum drawdown threshold for composite score penalty (default: -20.0)",
    )
    parser.add_argument(
        "--composite-alpha",
        type=float,
        default=0.5,
        help=
        "Alpha weight for drawdown penalty in composite score (default: 0.5)",
    )
    parser.add_argument(
        "--composite-beta",
        type=float,
        default=0.5,
        help="Beta weight for F1 penalty in composite score (default: 0.5)",
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

    # Parse grid search parameters
    factor_counts_list = None
    time_windows_list = None

    # Auto-enable grid search if factor_counts or time_windows are specified
    if args.factor_counts or args.time_windows:
        args.grid_search = True

    if args.grid_search or args.factor_counts or args.time_windows:
        # Parse factor counts
        if args.factor_counts:
            factor_counts_raw = [
                x.strip() for x in args.factor_counts.split(',') if x.strip()
            ]
            factor_counts_list = []
            for fc in factor_counts_raw:
                if fc.lower() == 'all':
                    factor_counts_list.append('all')
                else:
                    try:
                        factor_counts_list.append(int(fc))
                    except ValueError:
                        print(f"⚠️ Invalid factor count: {fc}, skipping")
        else:
            # Default: use 120
            factor_counts_list = [120]

        # Parse time windows
        if args.time_windows:
            time_windows_raw = [
                x.strip() for x in args.time_windows.split(',') if x.strip()
            ]
            time_windows_list = []
            for tw in time_windows_raw:
                if ':' in tw:
                    start, end = tw.split(':', 1)
                    time_windows_list.append((start.strip(), end.strip()))
                else:
                    print(
                        f"⚠️ Invalid time window format: {tw} (expected START:END), skipping"
                    )
        else:
            # Default: use train_start and train_end
            if args.train_start and args.train_end:
                time_windows_list = [(args.train_start, args.train_end)]
            else:
                time_windows_list = [(None, None)]

        print(f"\n{'=' * 80}")
        print("🔍 Grid Search Mode Enabled")
        print(f"{'=' * 80}")
        print(f"   Factor counts to test: {factor_counts_list}")
        print(f"   Time windows to test: {time_windows_list}")
        print(
            f"   Total combinations: {len(factor_counts_list) * len(time_windows_list)}"
        )
        print(f"{'=' * 80}\n")

    # Default behavior: if ablation not specified, enable ablation by default
    if not args.research_ablation:
        args.research_ablation = True

    # Autoencoder grid removed - no longer used

    # Grid search mode: run all combinations
    # Note: If factor_counts_list or time_windows_list is set, grid_search should be enabled
    if (args.grid_search or factor_counts_list
            or time_windows_list) and factor_counts_list and time_windows_list:
        grid_search_results = []

        for time_window_idx, (tw_start,
                              tw_end) in enumerate(time_windows_list):
            for factor_count_idx, factor_count in enumerate(
                    factor_counts_list):
                print(f"\n{'=' * 80}")
                print(
                    f"🔬 Grid Search: Combination {time_window_idx * len(factor_counts_list) + factor_count_idx + 1} / {len(factor_counts_list) * len(time_windows_list)}"
                )
                print(f"   Time Window: {tw_start} → {tw_end}")
                print(f"   Factor Count: {factor_count}")
                print(f"{'=' * 80}\n")

                # Create a modified args object for this combination
                import copy
                args_comb = copy.deepcopy(args)
                args_comb.train_start = tw_start
                args_comb.train_end = tw_end
                args_comb.grid_search = False  # Prevent recursive grid search
                # Store factor count in a custom attribute for grid search
                args_comb._grid_search_factor_count = factor_count

                try:
                    # Run single experiment by calling main() with modified args
                    # We need to temporarily modify sys.argv or pass args directly
                    # Since main() uses argparse, we'll create a wrapper
                    result_dict = run_single_experiment_wrapper(args_comb)
                    if result_dict:
                        result_dict['grid_search_params'] = {
                            'time_window': f"{tw_start} → {tw_end}",
                            'factor_count': factor_count,
                            'time_window_start': tw_start,
                            'time_window_end': tw_end,
                        }
                        # Ensure results_dir is stored (should already be set by wrapper)
                        if 'results_dir' not in result_dict or not result_dict.get(
                                'results_dir'):
                            # Try to reconstruct results_dir path
                            symbol_slug = _slugify(args.symbol)
                            feature_type_slug = _slugify(args.feature_type)
                            if tw_start and tw_end:
                                train_start_date = tw_start.replace("-",
                                                                    "")[:8]
                                train_end_date = tw_end.replace("-", "")[:8]
                                dir_date_suffix = f"{symbol_slug}_{feature_type_slug}_{train_start_date}_{train_end_date}"
                                potential_dir = DIM_COMPARE_RESULTS_ROOT / dir_date_suffix
                                if potential_dir.exists():
                                    result_dict['results_dir'] = str(
                                        potential_dir)
                        grid_search_results.append(result_dict)
                except Exception as exc:
                    print(f"⚠️ Grid search combination failed: {exc}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Generate grid search comparison report
        if grid_search_results:
            print(f"\n{'=' * 80}")
            print("📊 Generating Grid Search Comparison Report")
            print(f"{'=' * 80}\n")
            report_path_str = generate_grid_search_report(
                grid_search_results, symbol_slug, feature_type_slug, args)
            # Return a summary result for grid search
            summary_result = {
                "timestamp_start": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "task_type": "grid_search",
                "grid_search_summary": {
                    "total_combinations":
                    len(factor_counts_list) * len(time_windows_list),
                    "successful_combinations":
                    len(grid_search_results),
                    "factor_counts":
                    factor_counts_list,
                    "time_windows":
                    [f"{tw[0]} → {tw[1]}" for tw in time_windows_list],
                },
                "data_info": {
                    "grid_search_mode": True,
                },
            }
            # Extract results_dir from report_path
            from pathlib import Path
            results_dir = str(
                Path(report_path_str).parent) if report_path_str else None
            # Return None for model (no autoencoder anymore)
            return summary_result, None, results_dir
        else:
            print("⚠️ No successful grid search results to report")
            # Return empty result structure
            return {}, None, None

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
            feature_type=args.feature_type,
            timeframe=args.timeframe)

        # Use loaded horizons or fallback to parsed horizons
        horizons = horizons_loaded if horizons_loaded and len(
            horizons_loaded) > 0 else horizons_list

        original_feature_count = len(
            feature_names)  # Save original count (482)
        dfX = pd.DataFrame(X_raw,
                           columns=feature_names,
                           index=df_features_original.index[:len(X_raw)])

        # For backward compatibility, use default horizon
        # Always use binary signals (0=Short, 1=Long)
        y_series = pd.Series(y_raw, index=dfX.index[:len(y_raw)])
        # Remap labels to 2-class using future_return threshold (default: always use binary)
        use_binary = getattr(args, 'binary_signals', True)  # Default to True
        if use_binary:
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
            # Skip non-numeric columns (like _symbol)
            if c == '_symbol' or not pd.api.types.is_numeric_dtype(dfX[c]):
                continue
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
        # Use rank-based method for multi-asset scenarios to avoid scale issues
        print(f"\n[Stage 2] IC ranking (rank-based for multi-asset)...")

        # Check if we have symbol information for rank-based calculation
        has_symbol_info = '_symbol' in df_features_original.columns
        is_multi_asset = has_symbol_info and df_features_original[
            '_symbol'].nunique() > 1

        if is_multi_asset:
            print(
                f"   Using rank-based IC calculation across {df_features_original['_symbol'].nunique()} assets"
            )
            # Rank-based method: rank within each asset, then compute IC on merged ranks
            ic_scores = {}

            # Get symbol column aligned with df_all
            # We need to align symbol info with df_all indices
            # df_all is created from dfX which comes from X_raw, so we need to trace back
            # Try to get symbol from df_features_original, aligned by index
            try:
                # Align symbol info with df_all indices
                # df_all index should match dfX index, which should match df_features_original index
                symbol_series = df_features_original['_symbol'].reindex(
                    df_all.index)
                # If reindex fails (different indices), try to match by position
                if symbol_series.isna().all() and len(
                        df_features_original) == len(df_all):
                    symbol_series = pd.Series(
                        df_features_original['_symbol'].values,
                        index=df_all.index)
            except Exception:
                # Fallback: if we can't align, use original method
                print(
                    f"   ⚠️ Could not align symbol info, falling back to standard IC calculation"
                )
                symbol_series = None

            if symbol_series is not None and not symbol_series.isna().all():
                for col in df_all.columns:
                    try:
                        # Group by symbol and rank within each group
                        df_ranked = df_all[[col]].copy()
                        df_ranked['_symbol'] = symbol_series.values
                        df_ranked['_y'] = y_series.values

                        # Rank within each asset
                        df_ranked['_feature_rank'] = df_ranked.groupby(
                            '_symbol')[col].rank(method='average')
                        df_ranked['_y_rank'] = df_ranked.groupby(
                            '_symbol')['_y'].rank(method='average')

                        # Compute IC on ranked data (which is already rank-based, so this is consistent)
                        ic = spearmanr(df_ranked['_feature_rank'].values,
                                       df_ranked['_y_rank'].values,
                                       nan_policy="omit")[0]
                    except Exception as e:
                        # Fallback to original method if rank-based fails
                        ic = spearmanr(df_all[col].values,
                                       y_series.values,
                                       nan_policy="omit")[0]
                        if ic is None or np.isnan(ic):
                            ic = 0.0
                    ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
            else:
                # Fallback to original method if symbol alignment failed
                print(
                    f"   ⚠️ Symbol alignment failed, using standard IC calculation"
                )
                for col in df_all.columns:
                    try:
                        ic = spearmanr(df_all[col].values,
                                       y_series.values,
                                       nan_policy="omit")[0]
                    except Exception:
                        ic = 0.0
                    ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        else:
            # Single asset or no symbol info: use original method
            print(
                f"   Using standard IC calculation (single asset or no symbol info)"
            )
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
        # Determine target factor count
        # In grid search mode, check if factor count was set via _grid_search_factor_count
        if hasattr(args, '_grid_search_factor_count'
                   ) and args._grid_search_factor_count is not None:
            if args._grid_search_factor_count == 'all':
                target_top_k = len(top_sorted)  # Use all available factors
            else:
                target_top_k = int(args._grid_search_factor_count)
        else:
            target_top_k = 120  # Default value
        ic_top_k = min(max(target_top_k, 1), len(top_sorted))
        if ic_top_k == 0:
            ic_top_k = min(60, len(top_sorted))

        # Initial selection by IC
        top_cols_initial = [c for c, _ in top_sorted[:ic_top_k]]

        # Diversity check and rebalancing
        def infer_feature_type(feature_name: str) -> str:
            """Infer feature type from feature name."""
            name_lower = feature_name.lower()
            if 'alpha101' in name_lower:
                return 'alpha101'
            elif 'hurst' in name_lower:
                return 'hurst'
            elif 'wpt' in name_lower or 'wavelet' in name_lower:
                return 'wavelet'
            elif 'hilbert' in name_lower:
                return 'hilbert'
            elif 'spectral' in name_lower:
                return 'spectral'
            elif 'cvd' in name_lower or 'ofi' in name_lower or 'order_flow' in name_lower or 'taker_buy' in name_lower:
                return 'order_flow'
            elif 'baseline' in name_lower or 'sr_' in name_lower or 'compressed' in name_lower:
                return 'baseline'
            elif 'rsi' in name_lower or 'macd' in name_lower or 'bb_' in name_lower or 'atr' in name_lower or 'ema' in name_lower or 'sma' in name_lower:
                return 'technical'
            else:
                return 'other'

        # Calculate feature type distribution
        feature_type_counts = {}
        for col in top_cols_initial:
            feat_type = infer_feature_type(col)
            feature_type_counts[feat_type] = feature_type_counts.get(
                feat_type, 0) + 1

        total_selected = len(top_cols_initial)
        max_type_ratio = max(feature_type_counts.values()
                             ) / total_selected if total_selected > 0 else 0
        diversity_threshold = 0.6  # If any type > 60%, rebalance

        print(
            f"   Feature type distribution (initial): {dict(sorted(feature_type_counts.items(), key=lambda x: x[1], reverse=True))}"
        )
        print(f"   Max type ratio: {max_type_ratio:.2%}")

        # Rebalance if needed
        if max_type_ratio > diversity_threshold and total_selected > 20:
            print(
                f"   ⚠️  Feature type imbalance detected (max ratio: {max_type_ratio:.2%} > {diversity_threshold:.0%})"
            )
            print(f"   Rebalancing features to ensure diversity...")

            # Group features by type
            features_by_type = {}
            for col, ic_val in top_sorted:
                feat_type = infer_feature_type(col)
                if feat_type not in features_by_type:
                    features_by_type[feat_type] = []
                features_by_type[feat_type].append((col, ic_val))

            # Calculate target counts per type (ensure minimum representation)
            # Strategy: allocate based on available features, but cap max per type
            type_counts_available = {
                ft: len(features)
                for ft, features in features_by_type.items()
            }
            total_available = sum(type_counts_available.values())

            # Minimum quota per type (if available)
            min_quota_per_type = max(1, int(target_top_k *
                                            0.05))  # At least 5% per type
            max_quota_per_type = int(target_top_k *
                                     0.4)  # At most 40% per type

            # Allocate quotas
            type_quotas = {}
            remaining_quota = target_top_k

            # First pass: allocate minimum quotas
            for feat_type in features_by_type.keys():
                available = type_counts_available[feat_type]
                quota = min(min_quota_per_type, available, remaining_quota)
                if quota > 0:
                    type_quotas[feat_type] = quota
                    remaining_quota -= quota

            # Second pass: allocate remaining quota proportionally (but cap at max)
            if remaining_quota > 0:
                for feat_type in sorted(features_by_type.keys(),
                                        key=lambda x: len(features_by_type[x]),
                                        reverse=True):
                    if remaining_quota <= 0:
                        break
                    current_quota = type_quotas.get(feat_type, 0)
                    available = type_counts_available[feat_type]
                    additional = min(max_quota_per_type - current_quota,
                                     available - current_quota,
                                     remaining_quota)
                    if additional > 0:
                        type_quotas[feat_type] = current_quota + additional
                        remaining_quota -= additional

            # Select features based on quotas
            top_cols = []
            for feat_type, quota in sorted(type_quotas.items(),
                                           key=lambda x: x[1],
                                           reverse=True):
                if feat_type in features_by_type:
                    selected = [
                        col for col, _ in features_by_type[feat_type][:quota]
                    ]
                    top_cols.extend(selected)
                    print(
                        f"      {feat_type}: {len(selected)}/{quota} features selected"
                    )

            # If we have less than target, fill with remaining top IC features
            if len(top_cols) < target_top_k:
                remaining_features = [(col, ic) for col, ic in top_sorted
                                      if col not in top_cols]
                needed = target_top_k - len(top_cols)
                top_cols.extend(
                    [col for col, _ in remaining_features[:needed]])

            # Recalculate distribution
            feature_type_counts_rebalanced = {}
            for col in top_cols:
                feat_type = infer_feature_type(col)
                feature_type_counts_rebalanced[
                    feat_type] = feature_type_counts_rebalanced.get(
                        feat_type, 0) + 1

            print(
                f"   Feature type distribution (rebalanced): {dict(sorted(feature_type_counts_rebalanced.items(), key=lambda x: x[1], reverse=True))}"
            )
            print(f"   Total features selected: {len(top_cols)}")
        else:
            top_cols = top_cols_initial
            print(
                f"   ✅ Feature diversity is balanced (max ratio: {max_type_ratio:.2%} <= {diversity_threshold:.0%})"
            )
        df_ic = df_all[top_cols].copy()
        X_ic = df_ic.values
        scaler_ic = StandardScaler()
        X_ic_scaled = sanitize_features(scaler_ic.fit_transform(X_ic))
        print(
            f"[DEBUG] Stage 2: {len(top_cols)} features after IC ranking (target={target_top_k})"
        )

        # Calculate IC statistics for selected factors (for ICIR calculation)
        # Note: We'll calculate this after representative selection (Stage 3) to use the final factor set
        # For now, calculate based on top_cols, but we'll recalculate after reps are selected
        selected_ic_values = [
            ic_scores.get(col, 0.0) for col in top_cols if col in ic_scores
        ]
        ic_mean = np.mean([abs(ic) for ic in selected_ic_values
                           ]) if selected_ic_values else None
        ic_std = np.std([
            abs(ic) for ic in selected_ic_values
        ]) if selected_ic_values and len(selected_ic_values) > 1 else None
        if ic_mean is not None and ic_std is not None:
            icir = ic_mean / ic_std if ic_std > 0 else None
            print(
                f"   IC Statistics for IC-filtered factors: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}, ICIR={icir:.3f}"
                if icir else
                f"   IC Statistics: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}"
            )

        # Stability validation (if enabled)
        stability_validation_results = None
        if args.enable_stability_validation and args.train_start:
            print(f"\n{'=' * 80}")
            print(
                "🔍 Stability Validation: Validating selected factors on longer historical data"
            )
            print(f"{'=' * 80}")

            # Calculate validation period
            try:
                train_start_dt = pd.to_datetime(args.train_start)
                if args.validation_start:
                    validation_start_dt = pd.to_datetime(args.validation_start)
                else:
                    # Auto-calculate: go back validation_years from train_start
                    validation_start_dt = train_start_dt - pd.DateOffset(
                        years=args.validation_years)

                validation_start_str = validation_start_dt.strftime("%Y-%m-%d")
                validation_end_str = args.train_start  # Validate up to training start

                print(
                    f"   Factor Selection Period: {args.train_start} → {args.train_end}"
                )
                print(
                    f"   Stability Validation Period: {validation_start_str} → {validation_end_str}"
                )
                print(
                    f"   This validates if factors selected on recent data are stable over longer history"
                )

                # Load validation data
                X_val_raw, y_val_raw, feature_names_val, _, df_features_val = load_real_market_data(
                    args.data_path,
                    args.symbol,
                    validation_start_str,
                    validation_end_str,
                    horizons=horizons_list,
                    feature_type=args.feature_type,
                    timeframe=args.timeframe)

                if X_val_raw is not None and len(X_val_raw) > 0:
                    dfX_val = pd.DataFrame(
                        X_val_raw,
                        columns=feature_names_val,
                        index=df_features_val.index[:len(X_val_raw)])
                    y_series_val = pd.Series(
                        y_val_raw, index=dfX_val.index[:len(y_val_raw)])

                    # Calculate IC for selected factors on validation data
                    print(
                        f"\n   Calculating IC for {len(top_cols)} selected factors on validation data..."
                    )
                    ic_scores_validation = {}

                    # Check if validation data has symbol info for rank-based
                    has_symbol_val = '_symbol' in df_features_val.columns
                    is_multi_asset_val = has_symbol_val and df_features_val[
                        '_symbol'].nunique() > 1

                    for col in top_cols:
                        if col not in dfX_val.columns:
                            continue
                        try:
                            if is_multi_asset_val:
                                # Rank-based IC for validation
                                symbol_series_val = df_features_val[
                                    '_symbol'].reindex(dfX_val.index)
                                if symbol_series_val is not None and not symbol_series_val.isna(
                                ).all():
                                    df_ranked_val = dfX_val[[col]].copy()
                                    df_ranked_val[
                                        '_symbol'] = symbol_series_val.values
                                    df_ranked_val['_y'] = y_series_val.values
                                    df_ranked_val[
                                        '_feature_rank'] = df_ranked_val.groupby(
                                            '_symbol')[col].rank(
                                                method='average')
                                    df_ranked_val[
                                        '_y_rank'] = df_ranked_val.groupby(
                                            '_symbol')['_y'].rank(
                                                method='average')
                                    ic = spearmanr(
                                        df_ranked_val['_feature_rank'].values,
                                        df_ranked_val['_y_rank'].values,
                                        nan_policy="omit")[0]
                                else:
                                    ic = spearmanr(dfX_val[col].values,
                                                   y_series_val.values,
                                                   nan_policy="omit")[0]
                            else:
                                ic = spearmanr(dfX_val[col].values,
                                               y_series_val.values,
                                               nan_policy="omit")[0]
                        except Exception:
                            ic = 0.0
                        ic_scores_validation[
                            col] = 0.0 if ic is None or np.isnan(ic) else ic

                    # Compare IC between selection period and validation period
                    ic_comparison = {}
                    stable_factors = []
                    unstable_factors = []

                    for col in top_cols:
                        if col in ic_scores and col in ic_scores_validation:
                            ic_selection = ic_scores[col]
                            ic_validation = ic_scores_validation[col]
                            ic_change = ic_validation - ic_selection
                            ic_stability = abs(ic_validation) / (
                                abs(ic_selection) +
                                1e-8) if abs(ic_selection) > 1e-8 else 0

                            ic_comparison[col] = {
                                "ic_selection": ic_selection,
                                "ic_validation": ic_validation,
                                "ic_change": ic_change,
                                "stability_ratio": ic_stability,
                            }

                            # Factor is stable if IC sign is consistent and magnitude is similar
                            if (ic_selection * ic_validation > 0
                                    and  # Same sign
                                    ic_stability > 0.5 and ic_stability
                                    < 2.0):  # Similar magnitude
                                stable_factors.append(col)
                            else:
                                unstable_factors.append(col)

                    stability_validation_results = {
                        "validation_period": {
                            "start": validation_start_str,
                            "end": validation_end_str,
                        },
                        "selection_period": {
                            "start": args.train_start,
                            "end": args.train_end,
                        },
                        "ic_comparison":
                        ic_comparison,
                        "stable_factors":
                        stable_factors,
                        "unstable_factors":
                        unstable_factors,
                        "stability_rate":
                        len(stable_factors) / len(top_cols) if top_cols else 0,
                    }

                    print(f"\n   ✅ Stability Validation Results:")
                    print(f"      Total factors tested: {len(top_cols)}")
                    print(
                        f"      Stable factors: {len(stable_factors)} ({stability_validation_results['stability_rate']:.1%})"
                    )
                    print(
                        f"      Unstable factors: {len(unstable_factors)} ({1 - stability_validation_results['stability_rate']:.1%})"
                    )

                    if len(stable_factors) > 0:
                        print(
                            f"\n   📊 Top 10 Stable Factors (IC consistent across periods):"
                        )
                        stable_sorted = sorted(stable_factors,
                                               key=lambda x: abs(ic_comparison[
                                                   x]['ic_selection']),
                                               reverse=True)[:10]
                        for i, factor in enumerate(stable_sorted, 1):
                            comp = ic_comparison[factor]
                            print(
                                f"      {i}. {factor}: IC={comp['ic_selection']:.4f} → {comp['ic_validation']:.4f} (change: {comp['ic_change']:+.4f})"
                            )

                    if len(unstable_factors) > 0:
                        print(
                            f"\n   ⚠️  Top 5 Unstable Factors (IC changed significantly):"
                        )
                        unstable_sorted = sorted(
                            unstable_factors,
                            key=lambda x: abs(ic_comparison[x]['ic_change']),
                            reverse=True)[:5]
                        for i, factor in enumerate(unstable_sorted, 1):
                            comp = ic_comparison[factor]
                            print(
                                f"      {i}. {factor}: IC={comp['ic_selection']:.4f} → {comp['ic_validation']:.4f} (change: {comp['ic_change']:+.4f})"
                            )
                else:
                    print(
                        f"   ⚠️  Could not load validation data, skipping stability validation"
                    )
            except Exception as exc:
                print(f"   ⚠️  Stability validation failed: {exc}")
                import traceback
                traceback.print_exc()

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
        # IMPORTANT: Select factors based on target_top_k FIRST, then apply correlation filtering
        # This ensures different factor counts select different factors
        desired_reps = (min(target_top_k, len(df_ic_clean.columns))
                        if target_top_k and not df_ic_clean.empty else None)

        reps: list[str] = []
        if not df_ic_clean.empty:
            # First, select top N factors by IC score (where N = target_top_k)
            # This ensures we get different factors for different target_top_k values
            if desired_reps and desired_reps > 0:
                # Sort columns by IC score (absolute value) and take top N
                cols_with_ic = [(col, abs(ic_scores.get(col, 0.0)))
                                for col in df_ic_clean.columns
                                if col in ic_scores]
                cols_with_ic.sort(key=lambda x: x[1], reverse=True)
                top_ic_cols = [col for col, _ in cols_with_ic[:desired_reps]]

                # Then apply correlation filtering on the top IC factors
                corr = df_ic_clean[top_ic_cols].corr().abs().fillna(0.0)
                for c in top_ic_cols:
                    if all(corr.loc[c, r] < 0.9 for r in reps):
                        reps.append(c)

                # If correlation filtering removed too many, add back from top IC list
                if len(reps) < desired_reps:
                    additional = [c for c in top_ic_cols if c not in reps
                                  ][:max(desired_reps - len(reps), 0)]
                    reps.extend(additional)
            else:
                # Fallback: use original correlation-based selection
                corr = df_ic_clean.corr().abs().fillna(0.0)
                for c in df_ic_clean.columns:
                    if all(corr.loc[c, r] < 0.9 for r in reps):
                        reps.append(c)
                # Bound reps between 60 and 100 if no target specified
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

        # Recalculate IC statistics for final representative factors (for accurate ICIR)
        # This ensures IC statistics reflect the actual factors used in the model
        # CRITICAL: This must be calculated AFTER reps are selected, so different factor counts get different ICIR
        final_ic_values = [
            ic_scores.get(col, 0.0) for col in reps if col in ic_scores
        ]
        if final_ic_values and len(final_ic_values) > 0:
            ic_mean = np.mean([abs(ic) for ic in final_ic_values])
            ic_std = np.std([abs(ic) for ic in final_ic_values
                             ]) if len(final_ic_values) > 1 else 0.0
            icir = ic_mean / ic_std if ic_std > 0 else None
            print(
                f"   IC Statistics for final representative factors ({len(reps)} factors, {len(final_ic_values)} with IC scores): Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}, ICIR={icir:.3f}"
                if icir else
                f"   IC Statistics for final factors: Mean(|IC|)={ic_mean:.4f}, Std(|IC|)={ic_std:.4f}"
            )
        else:
            # Fallback to previous calculation if reps don't have IC scores
            print(
                f"   ⚠️  Warning: Could not calculate IC statistics for final factors ({len(reps)} factors), using IC-filtered factors"
            )
            # Keep the previous ic_mean, ic_std, icir from Stage 2 calculation
            # (they were calculated based on top_cols, which is less accurate but better than nothing)

        # Stage 4: Autoencoder compression removed - no longer used

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
        print("=" * 60)

        # Prepare price data for backtest (if available)
        price_data_test = None
        if 'close' in df_features_original.columns:
            # Get price data aligned with test indices
            price_data_test = df_features_original[[
                'close'
            ]].iloc[test_indices].copy()
            print(
                f"  📊 Price data available for backtest: {len(price_data_test)} samples"
            )

        # Stage 1: All features (482 -> ~470 after filtering)
        print("\n[Stage 1] Training on ALL features...")
        model_all = train_production_lightgbm(X_train_all, y_train, X_val_all,
                                              y_val)
        perf_all = evaluate_model_performance(model_all,
                                              X_test_all,
                                              y_test,
                                              "All Features",
                                              price_data=price_data_test)

        # Stage 2: IC-filtered features (~120)
        print("\n[Stage 2] Training on IC-filtered features...")
        model_ic = train_production_lightgbm(X_train_ic, y_train, X_val_ic,
                                             y_val)
        perf_ic = evaluate_model_performance(model_ic,
                                             X_test_ic,
                                             y_test,
                                             "IC-Filtered Features",
                                             price_data=price_data_test)

        # Stage 3: Representative features (60-100)
        print("\n[Stage 3] Training on Representative features...")
        model_reps = train_production_lightgbm(X_train_reps, y_train,
                                               X_val_reps, y_val)
        perf_reps = evaluate_model_performance(model_reps,
                                               X_test_reps,
                                               y_test,
                                               "Representative Features",
                                               price_data=price_data_test)

        feature_insights_stage3 = _derive_feature_insights(perf_all, perf_reps)

        # Default best result to Stage 3 (representative features)
        best_model = model_reps
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
                "stage1_all":
                perf_all,
                "stage2_ic":
                perf_ic,
                "stage3_representatives":
                perf_reps,
                "stage3_representatives_financial":
                perf_reps.get("financial_metrics", {}),
                "stage4_compressed":
                None,
                "selection_metric":
                args.selection_metric,
            },
            "insights": feature_insights_stage3,
            "ic_statistics": {
                "ic_mean":
                float(ic_mean) if ic_mean is not None else None,
                "ic_std":
                float(ic_std) if ic_std is not None else None,
                "icir":
                float(icir) if ic_mean is not None and ic_std is not None
                and ic_std > 0 else None,
            },
        }

        # Add stability validation results if available
        if stability_validation_results:
            best_result.setdefault("stability_validation",
                                   stability_validation_results)

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
            "timestamp_start": ablation_start_ts,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "task_type": "classification_binary",
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
                "all_selected_features": reps
                if reps else feature_names[:10],  # Store all selected features
            },
            "selected_features":
            reps,  # Store the complete list of selected features
            "selection": {
                "metric": args.selection_metric,
                "best_stage": feature_insights_stage3["recommended_stage"],
            },
            "insights": feature_insights_stage3,
            "ic_statistics": {
                "ic_mean":
                float(ic_mean) if ic_mean is not None else None,
                "ic_std":
                float(ic_std) if ic_std is not None else None,
                "icir":
                float(icir) if ic_mean is not None and ic_std is not None
                and ic_std > 0 else None,
            },
        }
        best_model = model_reps
        best_dir = None
        # Stage 4 autoencoder removed - no longer used

        # Multi-horizon training (if enabled) - train all 3 stages for each horizon
        if horizons and len(horizons) > 1 and not df_features_original.empty:
            print(f"\n{'=' * 80}")
            print(
                f"Multi-Horizon Training: Evaluating {len(horizons)} horizons across all 3 stages"
            )
            print(f"{'=' * 80}")

            df_multi_labels = create_labels_multi_horizon(df_features_original,
                                                          horizons=horizons)

            for horizon in horizons:
                print(f"\n{'=' * 60}")
                print(f"Training all 3 stages for Horizon: {horizon} bars")
                print(f"{'=' * 60}")

                # Get labels for this horizon (binary: 0=Short, 1=Long)
                y_horizon_col = f"binary_signal_{horizon}"
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

                    # Stage 4 autoencoder removed - no longer used

                    # Store results for this horizon
                    horizon_perf = {
                        "stage1_all_features": perf_h_all,
                        "stage2_ic_filtered": perf_h_ic,
                        "stage3_representatives": perf_h_reps,
                    }
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

        # Autoencoder removed - no longer used

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

        return best_result, best_model, best_dir

    # Run dimensionality comparison
    results, model, results_dir = run_dimensionality_comparison(
        data_path=args.data_path,
        symbol=args.symbol,
        train_start=args.train_start,
        train_end=args.train_end,
        feature_type=args.feature_type,
        shap_analysis=args.shap_analysis,
        timeframe=args.timeframe,
    )

    # top_k parameter removed, factor count is now determined by factor_counts_list or default 120

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
    return results, model, results_dir


if __name__ == "__main__":
    try:
        results, model, results_dir = main()
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
