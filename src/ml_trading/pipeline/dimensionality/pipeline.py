"""Advanced dimensionality reduction pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

from ml_trading.data_tools.unified_data_loader import UnifiedDataLoader
from ml_trading.models.interpretable_factor_engine import InterpretableFactorEngine
from ml_trading.utils.sample_data import create_sample_data


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    factor_names: List[str]
    dataframe: pd.DataFrame
    target_column: str


@dataclass
class MatrixBundle:
    X: np.ndarray
    y: np.ndarray
    df: pd.DataFrame
    feature_names: List[str]


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="5T",
        )
    return df


def load_real_data(data_path: str, symbol: str = "ETH-USD") -> DatasetBundle:
    print(f"📂 Loading real data for {symbol}")

    loader = UnifiedDataLoader(data_path)
    result = loader.load_real_data(symbol=symbol, return_dataframe=True)

    X, y, feature_columns, df_features, target_column = result

    df_features = _ensure_timestamp_column(df_features)

    cleaned_df = df_features.dropna(subset=feature_columns +
                                    [target_column]).copy()
    X = cleaned_df[feature_columns].values
    y = cleaned_df[target_column].values

    print(
        f"✅ Real dataset ready: {X.shape[0]} samples, {X.shape[1]} features, target={target_column}"
    )

    return DatasetBundle(
        X=X,
        y=y,
        factor_names=feature_columns,
        dataframe=cleaned_df.reset_index(drop=True),
        target_column=target_column,
    )


def create_sample_dataset(n_samples: int, n_factors: int) -> DatasetBundle:
    X, y, factor_names, df = create_sample_data(
        n_samples=n_samples,
        n_factors=n_factors,
        return_dataframe=True,
    )

    df = _ensure_timestamp_column(df)
    target_column = "target"
    if "synthetic_future_return" in df.columns:
        df[target_column] = df.pop("synthetic_future_return")
    else:
        df[target_column] = y

    cleaned_df = df.dropna(subset=factor_names + [target_column]).copy()
    X = cleaned_df[factor_names].values
    y = cleaned_df[target_column].values

    print(
        f"📊 Using synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features"
    )

    return DatasetBundle(
        X=X,
        y=y,
        factor_names=factor_names,
        dataframe=cleaned_df.reset_index(drop=True),
        target_column=target_column,
    )


def filter_by_date_range(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if "timestamp" not in df.columns:
        return df.copy(), None, None

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    start_ts = pd.to_datetime(start) if start else timestamps.min()
    end_ts = pd.to_datetime(end) if end else timestamps.max()

    mask = (timestamps >= start_ts) & (timestamps <= end_ts)
    filtered = df.loc[mask].copy()

    return filtered, start_ts, end_ts


def create_matrix_bundle_from_df(
    df: Optional[pd.DataFrame],
    feature_names: List[str],
    target_column: str,
    min_samples: int,
    label: str,
) -> Optional[MatrixBundle]:
    if df is None or df.empty:
        print(f"⚠️ {label}: no data available")
        return None

    missing = [f for f in feature_names if f not in df.columns]
    available_features = [f for f in feature_names if f in df.columns]

    if missing:
        print(
            f"⚠️ {label}: {len(missing)} features missing from dataframe (ignored)"
        )

    if not available_features:
        print(f"❌ {label}: none of the requested features are present")
        return None

    subset = df[available_features + [target_column]].dropna()

    if len(subset) < min_samples:
        print(
            f"⚠️ {label}: insufficient samples after cleaning ({len(subset)} < {min_samples})"
        )
        return None

    return MatrixBundle(
        X=subset[available_features].values,
        y=subset[target_column].values,
        df=subset.reset_index(drop=True),
        feature_names=available_features,
    )


def build_topk_lightgbm_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "num_leaves": args.topk_num_leaves,
        "learning_rate": args.topk_learning_rate,
        "feature_fraction": args.topk_feature_fraction,
        "bagging_fraction": args.topk_bagging_fraction,
        "lambda_l1": args.topk_lambda_l1,
        "lambda_l2": args.topk_lambda_l2,
        "verbosity": -1,
    }


def train_lightgbm_on_top_factors(
    train_bundle: Optional[MatrixBundle],
    params: Dict[str, Any],
    train_ratio: float,
    early_stopping_rounds: int,
    eval_period: int,
    test_bundle: Optional[MatrixBundle] = None,
) -> Tuple[Optional[lgb.Booster], Dict[str, Any]]:
    if train_bundle is None:
        print("⚠️ Skipping top-k LightGBM training (no training bundle)")
        return None, {}

    X_train = train_bundle.X
    y_train = train_bundle.y

    if test_bundle is None:
        split_idx = int(len(X_train) * train_ratio)
        if split_idx <= 0 or split_idx >= len(X_train):
            print(
                "⚠️ Not enough samples to create validation split for top-k LightGBM"
            )
            return None, {}
        X_fit, y_fit = X_train[:split_idx], y_train[:split_idx]
        X_eval, y_eval = X_train[split_idx:], y_train[split_idx:]
    else:
        X_fit, y_fit = X_train, y_train
        X_eval, y_eval = test_bundle.X, test_bundle.y
        if len(X_eval) == 0:
            print(
                "⚠️ Evaluation bundle empty; skipping top-k LightGBM training")
            return None, {}

    lgb_train = lgb.Dataset(X_fit, label=y_fit)
    lgb_valid = lgb.Dataset(X_eval, label=y_eval, reference=lgb_train)

    callbacks = [lgb.early_stopping(early_stopping_rounds)]
    if eval_period > 0:
        callbacks.append(lgb.log_evaluation(period=eval_period))

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=callbacks,
    )

    preds = booster.predict(X_eval)
    metrics = {
        "r2": float(r2_score(y_eval, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_eval, preds))),
        "mae": float(mean_absolute_error(y_eval, preds)),
        "best_iteration": int(booster.best_iteration or len(preds)),
        "train_samples": int(len(y_fit)),
        "eval_samples": int(len(y_eval)),
        "feature_names": train_bundle.feature_names,
    }

    return booster, metrics


def generate_time_windows(
    start: Optional[str],
    end: Optional[str],
    frequency: str,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    if start is None or end is None:
        return []

    freq_map = {"quarter": "Q", "month": "M"}
    freq = freq_map.get(frequency, "Q")
    periods = pd.period_range(start=start, end=end, freq=freq)

    windows: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for period in periods:
        window_start = period.start_time
        window_end = period.end_time
        if frequency == "quarter":
            label = f"{period.start_time.year}_Q{period.quarter}"
        else:
            label = period.start_time.strftime("%Y_%m")
        windows.append((label, window_start, window_end))

    return windows


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Saved JSON to {path}")


def save_top_factors(
    output_dir: Path,
    symbol: str,
    top_factors: List[str],
    weights: List[float],
    factor_contributions: Optional[np.ndarray] = None,
) -> Path:
    payload = {
        "symbol":
        symbol,
        "timestamp":
        pd.Timestamp.now().isoformat(),
        "top_factors": [{
            "name": factor,
            "weight": float(weight)
        } for factor, weight in zip(top_factors, weights)],
    }

    if factor_contributions is not None:
        payload["factor_contributions"] = {
            factor: float(score)
            for factor, score in zip(top_factors,
                                     factor_contributions[:len(top_factors)])
        }

    path = output_dir / f"top_factors_{symbol.replace('-', '_')}.json"
    save_json(path, payload)
    return path


def run_rolling_evaluation(
    dataset: DatasetBundle,
    args: argparse.Namespace,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    if "timestamp" not in dataset.dataframe.columns:
        print(
            "⚠️ Dataset has no timestamp column. Rolling evaluation skipped.")
        return None

    windows = generate_time_windows(args.rolling_test_start,
                                    args.rolling_test_end,
                                    args.rolling_frequency)

    if not windows:
        print("⚠️ Rolling evaluation requested but no windows generated")
        return None

    base_train_start = pd.to_datetime(
        args.train_start) if args.train_start else None
    base_train_end = pd.to_datetime(args.train_end) if args.train_end else None

    if base_train_start is None:
        base_train_start = dataset.dataframe["timestamp"].min()
    if base_train_end is None:
        base_train_end = dataset.dataframe["timestamp"].max()

    base_duration = base_train_end - base_train_start

    factor_counter: Counter = Counter()
    window_results: List[Dict[str, Any]] = []
    lgb_params = build_topk_lightgbm_params(args)

    for label, window_start, window_end in windows:
        test_df, _, _ = filter_by_date_range(dataset.dataframe, window_start,
                                             window_end)

        if len(test_df) < args.min_samples:
            print(
                f"⚠️ {label}: insufficient test samples ({len(test_df)} < {args.min_samples}), skipping"
            )
            continue

        if args.rolling_train_mode == "fixed":
            train_start, train_end = base_train_start, base_train_end
        elif args.rolling_train_mode == "expanding":
            train_start, train_end = base_train_start, window_end
        else:
            train_end = window_end
            train_start = window_end - base_duration

        train_df, actual_train_start, actual_train_end = filter_by_date_range(
            dataset.dataframe, train_start, train_end)

        train_bundle = create_matrix_bundle_from_df(
            train_df,
            dataset.factor_names,
            dataset.target_column,
            args.min_samples,
            label=f"{label} training",
        )

        if train_bundle is None:
            continue

        engine = InterpretableFactorEngine(
            encoding_dim=args.encoding_dim,
            autoencoder_lr=args.learning_rate,
            autoencoder_epochs=args.epochs,
            dropout_rate=args.dropout_rate,
        )

        engine.fit(
            train_bundle.X,
            train_bundle.y,
            train_bundle.feature_names,
            top_k=args.top_k,
        )

        top_factors = list(engine.top_factors[:args.top_k])
        factor_counter.update(top_factors)

        test_bundle = create_matrix_bundle_from_df(
            test_df,
            top_factors,
            dataset.target_column,
            args.min_samples,
            label=f"{label} testing",
        )

        booster, metrics = train_lightgbm_on_top_factors(
            train_bundle=create_matrix_bundle_from_df(
                train_df,
                top_factors,
                dataset.target_column,
                args.min_samples,
                label=f"{label} top-k training",
            ),
            params=lgb_params,
            train_ratio=args.topk_train_ratio,
            early_stopping_rounds=args.topk_early_stop,
            eval_period=args.topk_eval_period,
            test_bundle=test_bundle,
        )

        window_results.append({
            "window":
            label,
            "train_range": {
                "start": actual_train_start,
                "end": actual_train_end,
                "samples": len(train_df) if train_bundle else 0,
            },
            "test_range": {
                "start": window_start,
                "end": window_end,
                "samples": len(test_df),
            },
            "top_factors":
            top_factors,
            "metrics":
            metrics,
            "best_iteration":
            metrics.get("best_iteration") if metrics else None,
        })

        if booster and args.save_topk_model:
            model_path = (
                output_dir /
                f"rolling_lightgbm_{label}_{args.symbol.replace('-', '_')}.txt"
            )
            booster.save_model(str(model_path))

    if not window_results:
        print("⚠️ No rolling evaluation windows completed")
        return None

    stability = {
        "factor_frequency":
        dict(
            sorted(factor_counter.items(),
                   key=lambda item: item[1],
                   reverse=True)),
        "stable_factors": [
            factor for factor, count in factor_counter.items()
            if count >= args.stability_min_periods
        ],
        "total_windows":
        len(window_results),
    }

    summary = {"windows": window_results, "stability": stability}

    save_json(
        output_dir /
        f"rolling_evaluation_{args.symbol.replace('-', '_')}.json",
        summary,
    )

    return summary


def generate_pipeline_report(
    output_dir: Path,
    engine: InterpretableFactorEngine,
    args: argparse.Namespace,
    predictions: np.ndarray,
    signals: np.ndarray,
    topk_metrics: Dict[str, Any],
    rolling_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    report_path = (
        output_dir /
        f"dimensionality_reduction_report_{args.symbol.replace('-', '_')}.txt")

    with open(report_path, "w") as f:
        f.write("Advanced Dimensionality Reduction Pipeline Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration\n")
        f.write("-" * 20 + "\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Encoding Dimension: {args.encoding_dim}\n")
        f.write(f"Autoencoder LR: {args.learning_rate}\n")
        f.write(f"Autoencoder Epochs: {args.epochs}\n")
        f.write(f"Dropout Rate: {args.dropout_rate}\n")
        f.write(f"Top-K Factors: {args.top_k}\n\n")

        f.write("Compression Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original Factors: {len(engine.factor_names)}\n")
        f.write(f"Compressed Dimensions: {args.encoding_dim}\n")
        f.write(
            f"Compression Ratio: {len(engine.factor_names) / args.encoding_dim:.1f}x\n\n"
        )

        f.write("Predictions\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean: {predictions.mean():.4f}\n")
        f.write(f"Std: {predictions.std():.4f}\n\n")

        f.write("Signals (Top-K Weighted)\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean: {signals.mean():.4f}\n")
        f.write(f"Std: {signals.std():.4f}\n\n")

        f.write("Top Factors\n")
        f.write("-" * 20 + "\n")
        for idx, (factor, weight) in enumerate(
                zip(engine.top_factors, engine.factor_weights)):
            f.write(f"{idx + 1:2d}. {factor:<40} {weight:.4f}\n")

        f.write("\nTop-K LightGBM Metrics\n")
        f.write("-" * 25 + "\n")
        if topk_metrics:
            for key, value in topk_metrics.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write("No top-k LightGBM model trained.\n")

        if rolling_summary:
            f.write("\nRolling Evaluation Summary\n")
            f.write("-" * 30 + "\n")
            stability = rolling_summary.get("stability", {})
            f.write(
                f"Total Windows: {stability.get('total_windows', len(rolling_summary))}\n"
            )
            f.write(
                f"Stable Factors (>= {args.stability_min_periods} windows):\n")
            for factor in stability.get("stable_factors", []):
                freq = stability.get("factor_frequency", {}).get(factor, 0)
                f.write(f"  - {factor}: {freq}\n")

    print(f"📋 Report saved to: {report_path}")
    return report_path


def run_dimensionality_reduction_pipeline(args: argparse.Namespace) -> None:
    print("🚀 Starting Dimensionality Reduction Pipeline")
    print("=" * 80)
    print("Flow: Research → AE+SHAP → Top-K → LightGBM → Rolling (optional)")
    print("=" * 80)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.use_real_data:
            dataset = load_real_data(args.data_path, args.symbol)
        else:
            dataset = create_sample_dataset(args.n_samples, args.n_factors)
    except Exception as exc:  # noqa: BLE001
        print(
            f"❌ Failed to load real data: {exc}. Falling back to sample data.")
        dataset = create_sample_dataset(args.n_samples, args.n_factors)

    train_df, train_start, train_end = filter_by_date_range(
        dataset.dataframe, args.train_start, args.train_end)

    train_bundle = create_matrix_bundle_from_df(
        train_df,
        dataset.factor_names,
        dataset.target_column,
        args.min_samples,
        label="Base training",
    )

    if train_bundle is None:
        raise RuntimeError("Training dataset insufficient after filtering")

    engine = InterpretableFactorEngine(
        encoding_dim=args.encoding_dim,
        autoencoder_lr=args.learning_rate,
        autoencoder_epochs=args.epochs,
        dropout_rate=args.dropout_rate,
    )

    print(
        f"🎯 Training AE+SHAP on {train_bundle.X.shape[0]} samples and {train_bundle.X.shape[1]} factors"
    )
    engine.fit(
        train_bundle.X,
        train_bundle.y,
        train_bundle.feature_names,
        top_k=args.top_k,
    )

    predictions = engine.predict(train_bundle.X)
    signals = engine.get_interpretable_signal(train_bundle.X)

    test_df = None
    test_bundle = None
    test_start = test_end = None
    if args.test_start and args.test_end:
        test_df, test_start, test_end = filter_by_date_range(
            dataset.dataframe, args.test_start, args.test_end)
        test_bundle = create_matrix_bundle_from_df(
            test_df,
            list(engine.top_factors[:args.top_k]),
            dataset.target_column,
            args.min_samples,
            label="Base testing",
        )

    top_factors = list(engine.top_factors[:args.top_k])
    top_weights = list(engine.factor_weights[:len(top_factors)])

    topk_train_bundle = create_matrix_bundle_from_df(
        train_df,
        top_factors,
        dataset.target_column,
        args.min_samples,
        label="Top-k training",
    )

    topk_model, topk_metrics = train_lightgbm_on_top_factors(
        train_bundle=topk_train_bundle,
        params=build_topk_lightgbm_params(args),
        train_ratio=args.topk_train_ratio,
        early_stopping_rounds=args.topk_early_stop,
        eval_period=args.topk_eval_period,
        test_bundle=test_bundle,
    )

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if args.save_model:
        model_path = (
            output_dir /
            f"interpretable_factor_engine_{args.symbol.replace('-', '_')}_{timestamp}.pkl"
        )
        engine.save_model(str(model_path))

    if args.save_topk_model and topk_model is not None:
        topk_model_path = (
            output_dir /
            f"lightgbm_topk_{args.symbol.replace('-', '_')}_{timestamp}.txt")
        topk_model.save_model(str(topk_model_path))
        print(f"💾 Top-K LightGBM saved to: {topk_model_path}")

    top_factor_path = save_top_factors(
        output_dir,
        args.symbol,
        top_factors,
        top_weights,
        getattr(engine, "factor_contributions", None),
    )

    rolling_summary = (run_rolling_evaluation(dataset, args, output_dir)
                       if args.rolling_eval else None)

    summary_payload = {
        "timestamp":
        timestamp,
        "symbol":
        args.symbol,
        "train_range": {
            "start": train_start,
            "end": train_end,
            "samples": len(train_df)
        },
        "test_range": {
            "start": test_start,
            "end": test_end,
            "samples": len(test_df) if test_df is not None else None,
        },
        "compression": {
            "original_factors":
            len(engine.factor_names),
            "encoding_dim":
            args.encoding_dim,
            "compression_ratio":
            float(len(engine.factor_names) / args.encoding_dim),
        },
        "top_factors_file":
        str(top_factor_path),
        "top_k":
        top_factors,
        "topk_weights":
        top_weights,
        "topk_lightgbm_metrics":
        topk_metrics,
        "rolling_summary":
        rolling_summary.get("stability") if rolling_summary else None,
    }

    summary_path = (
        output_dir /
        f"pipeline_summary_{args.symbol.replace('-', '_')}_{timestamp}.json")
    save_json(summary_path, summary_payload)

    if args.visualize:
        viz_path = (
            output_dir /
            f"factor_contributions_{args.symbol.replace('-', '_')}_{timestamp}.png"
        )
        engine.visualize_factor_contributions(save_path=str(viz_path),
                                              top_k=args.top_k)

    if args.generate_report:
        generate_pipeline_report(
            output_dir,
            engine,
            args,
            predictions,
            signals,
            topk_metrics,
            rolling_summary,
        )

    print("\n" + "=" * 80)
    print("🎉 Dimensionality Reduction Pipeline Complete!")
    print("=" * 80)
    print(f"📊 Compression: {len(engine.factor_names)} → {args.encoding_dim}")
    print(f"🏆 Top factors stored at: {top_factor_path}")
    if topk_metrics:
        print(f"🌲 Top-K LightGBM R²: {topk_metrics.get('r2', 'n/a'):.4f}")
    if rolling_summary:
        stable = rolling_summary.get("stability", {}).get("stable_factors", [])
        print(f"🔁 Rolling stable factors: {stable}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "Dimensionality reduction pipeline with Autoencoder + SHAP + Top-K LightGBM",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/parquet_data",
        help="Path to raw market data",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETH-USD",
        help="Trading symbol identifier",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real market data instead of synthetic sample",
    )

    parser.add_argument(
        "--train-start",
        type=str,
        default="2024-01-01",
        help="Training window start date",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2024-12-31",
        help="Training window end date",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="Optional base evaluation window start",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Optional base evaluation window end",
    )

    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Autoencoder bottleneck dimension",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Autoencoder learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Autoencoder training epochs",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Autoencoder dropout",
    )
    parser.add_argument("--top-k",
                        type=int,
                        default=10,
                        help="Number of top factors to retain")

    parser.add_argument(
        "--topk-learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate for top-k model",
    )
    parser.add_argument(
        "--topk-num-leaves",
        type=int,
        default=31,
        help="LightGBM number of leaves",
    )
    parser.add_argument(
        "--topk-feature-fraction",
        type=float,
        default=0.8,
        help="LightGBM feature fraction",
    )
    parser.add_argument(
        "--topk-bagging-fraction",
        type=float,
        default=0.8,
        help="LightGBM bagging fraction",
    )
    parser.add_argument(
        "--topk-lambda-l1",
        type=float,
        default=0.1,
        help="LightGBM L1 regularization",
    )
    parser.add_argument(
        "--topk-lambda-l2",
        type=float,
        default=0.1,
        help="LightGBM L2 regularization",
    )
    parser.add_argument(
        "--topk-train-ratio",
        type=float,
        default=0.8,
        help="Train ratio when no explicit eval set provided",
    )
    parser.add_argument(
        "--topk-early-stop",
        type=int,
        default=50,
        help="Early stopping rounds for LightGBM",
    )
    parser.add_argument(
        "--topk-eval-period",
        type=int,
        default=50,
        help="Evaluation logging period for LightGBM (0 to suppress)",
    )

    parser.add_argument(
        "--rolling-eval",
        action="store_true",
        help="Enable rolling evaluation across time windows",
    )
    parser.add_argument(
        "--rolling-test-start",
        type=str,
        default="2025-01-01",
        help="Rolling evaluation start date",
    )
    parser.add_argument(
        "--rolling-test-end",
        type=str,
        default="2025-12-31",
        help="Rolling evaluation end date",
    )
    parser.add_argument(
        "--rolling-frequency",
        choices=["quarter", "month"],
        default="quarter",
        help="Frequency for rolling evaluation windows",
    )
    parser.add_argument(
        "--rolling-train-mode",
        choices=["fixed", "expanding", "sliding"],
        default="fixed",
        help="Training window update mode for rolling evaluation",
    )
    parser.add_argument(
        "--stability-min-periods",
        type=int,
        default=3,
        help="Minimum window count for factor stability",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum samples required for any training/test window",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Synthetic data samples when not using real data",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=60,
        help="Synthetic factor count when not using real data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dimensionality_pipeline",
        help="Directory to store pipeline artefacts",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Persist the trained InterpretableFactorEngine",
    )
    parser.add_argument(
        "--save-topk-model",
        action="store_true",
        help="Persist the LightGBM model trained on top-k factors",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save factor contribution visualization",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate detailed text report",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_dimensionality_reduction_pipeline(args)


if __name__ == "__main__":
    main()
