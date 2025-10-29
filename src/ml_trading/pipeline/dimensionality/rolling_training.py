"""Rolling dimensionality training orchestrators."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from ml_trading.models.interpretable_factor_engine import InterpretableFactorEngine
from ml_trading.utils.drift import DriftDetector
from ml_trading.utils.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from ml_trading.utils.sample_data import create_sample_data

warnings.filterwarnings("ignore")


@dataclass
class RollingConfig:
    encoding_dim: int
    drift_threshold: float
    min_improvement: float
    feature_eval_ratio: float = 0.2
    model_dir: Path | None = None
    results_dir: Path | None = None

    def ensure_dirs(self) -> None:
        if self.model_dir is not None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.results_dir is not None:
            self.results_dir.mkdir(parents=True, exist_ok=True)


def _create_sample_dataset(
        n_samples: int = 1000,
        n_features: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, feature_names = create_sample_data(n_samples=n_samples,
                                             n_factors=n_features)
    return X, y, feature_names


def load_quarterly_data(
    data_path: str,
    start_date: str,
    end_date: str,
    *,
    symbol: str = "ETH-USD",
    fallback_features: int = 100,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print(f"📂 Loading data for {symbol}: {start_date} → {end_date}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Data file not found at {data_path}, using sample data")
        return _create_sample_dataset(n_features=fallback_features)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error reading {data_path}: {exc}")
        return _create_sample_dataset(n_features=fallback_features)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df = df.loc[mask]

    if df.empty:
        print("❌ No records in specified range, using sample data")
        return _create_sample_dataset(n_features=fallback_features)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    target_col = next(
        (col for col in [
            "future_return",
            "target",
            "synthetic_future_return",
        ] if col in df.columns),
        None,
    )

    if target_col is None:
        if "close" in df.columns:
            df["synthetic_future_return"] = (
                df["close"].pct_change().shift(-1))
        else:
            df["synthetic_future_return"] = np.random.randn(len(df))
        target_col = "synthetic_future_return"

    feature_cols = [col for col in numeric_cols if col != target_col]

    if not feature_cols:
        print("❌ No numeric features found, using sample data")
        return _create_sample_dataset(n_features=fallback_features)

    combined = df[feature_cols + [target_col]].dropna()

    if combined.empty:
        print("❌ Data became empty after dropping NaNs, using sample data")
        return _create_sample_dataset(n_features=len(feature_cols))

    X = combined[feature_cols].to_numpy()
    y = combined[target_col].to_numpy()

    print(
        f"✅ Loaded {len(combined)} samples with {len(feature_cols)} features")
    return X, y, feature_cols


def feature_result_to_dict(result: FeatureEvaluationResult) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(result.metrics)
    if result.best_iteration is not None:
        data["best_iteration"] = int(result.best_iteration)
    return data


def compare_performance(
    original_metrics: Dict[str, float],
    compressed_metrics: Dict[str, float],
    min_improvement: float,
) -> Dict[str, Any]:
    if "r2" in original_metrics and "r2" in compressed_metrics:
        r2_improvement = compressed_metrics["r2"] - original_metrics["r2"]
        improvement_percent = ((r2_improvement / abs(original_metrics["r2"])) *
                               100 if original_metrics["r2"] != 0 else 0.0)
        return {
            "metric": "r2",
            "original": original_metrics["r2"],
            "compressed": compressed_metrics["r2"],
            "improvement": r2_improvement,
            "improvement_percent": improvement_percent,
            "is_improved": r2_improvement > min_improvement,
        }

    if "auc" in original_metrics and "auc" in compressed_metrics:
        auc_improvement = compressed_metrics["auc"] - original_metrics["auc"]
        improvement_percent = ((auc_improvement / original_metrics["auc"]) *
                               100 if original_metrics["auc"] != 0 else 0.0)
        return {
            "metric": "auc",
            "original": original_metrics["auc"],
            "compressed": compressed_metrics["auc"],
            "improvement": auc_improvement,
            "improvement_percent": improvement_percent,
            "is_improved": auc_improvement > min_improvement,
        }

    return {"metric": None, "is_improved": False}


def evaluate_test_performance(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def save_results(results: Dict[str, Any], results_dir: Path | None) -> None:
    if results_dir is None:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"rolling_dim_results_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Results saved to: {filepath}")


def save_model(engine: InterpretableFactorEngine,
               model_dir: Path | None) -> None:
    if model_dir is None:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filepath = model_dir / f"rolling_dim_model_{timestamp}.pkl"
    engine.save_model(str(filepath))
    print(f"💾 Model saved to: {filepath}")


def summarize_window_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"error": "No results to summarize"}

    avg_original_r2 = np.mean(
        [r.get("original_performance", {}).get("r2", 0.0) for r in results])
    avg_compressed_r2 = np.mean(
        [r.get("compressed_performance", {}).get("r2", 0.0) for r in results])
    avg_test_r2 = np.mean(
        [r.get("test_performance", {}).get("r2", 0.0) for r in results])
    avg_compression_ratio = np.mean(
        [r.get("compression_ratio", 0.0) for r in results])

    return {
        "total_windows": len(results),
        "average_original_r2": float(avg_original_r2),
        "average_compressed_r2": float(avg_compressed_r2),
        "average_test_r2": float(avg_test_r2),
        "average_compression_ratio": float(avg_compression_ratio),
        "overall_improvement": float(avg_compressed_r2 - avg_original_r2),
        "results": results,
    }


def run_dimensionality_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: RollingConfig,
    *,
    symbol: str,
    train_period: str,
    test_period: str,
    drift_detector: DriftDetector | None = None,
) -> Dict[str, Any]:
    evaluator = FeatureEvaluator(
        validation_ratio=config.feature_eval_ratio,
        min_improvement=config.min_improvement,
        task_type="regression",
    )

    original_eval = evaluator.evaluate_feature_set(X_train, y_train)

    engine = InterpretableFactorEngine(encoding_dim=config.encoding_dim)
    engine.fit(X_train, y_train, feature_names)

    compressed_features = engine.embeddings
    compressed_eval = evaluator.evaluate_feature_set(compressed_features,
                                                     y_train)

    improvement = compare_performance(
        original_eval.metrics,
        compressed_eval.metrics,
        config.min_improvement,
    )

    test_predictions = engine.predict(X_test)
    test_performance = evaluate_test_performance(y_test, test_predictions)

    factor_contributions: Dict[str, float] = {}
    if engine.factor_contributions is not None:
        factor_contributions = {
            name: float(value)
            for name, value in zip(engine.factor_names,
                                   engine.factor_contributions)
        }

    drift_info: Dict[str, Any] | None = None
    if drift_detector and factor_contributions:
        drift_detector.add_importance(factor_contributions)
        _, diagnostics = drift_detector.should_trigger()
        drift_info = diagnostics

    top_factors = (list(map(str, engine.top_factors))
                   if engine.top_factors is not None else [])
    factor_weights = ([
        float(w) for w in np.asarray(engine.factor_weights).tolist()
    ] if engine.factor_weights is not None else [])

    results: Dict[str, Any] = {
        "training_period": train_period,
        "testing_period": test_period,
        "symbol": symbol,
        "original_features_count": len(feature_names),
        "compressed_dimensions": config.encoding_dim,
        "selected_factors_count": len(top_factors),
        "original_performance": feature_result_to_dict(original_eval),
        "compressed_performance": feature_result_to_dict(compressed_eval),
        "performance_improvement": improvement,
        "test_performance": test_performance,
        "top_factors": top_factors,
        "factor_weights": factor_weights,
        "factor_contributions": factor_contributions,
        "compression_ratio": len(feature_names) / max(config.encoding_dim, 1),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if drift_info is not None:
        results["drift_diagnostics"] = drift_info

    save_results(results, config.results_dir)
    save_model(engine, config.model_dir)

    return results


def run_dimensionality_training_from_paths(
    config: RollingConfig,
    *,
    train_data_path: str,
    test_data_path: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    symbol: str,
    drift_detector: DriftDetector | None = None,
) -> Dict[str, Any]:
    X_train, y_train, feature_names = load_quarterly_data(
        train_data_path,
        train_start,
        train_end,
        symbol=symbol,
    )
    X_test, y_test, _ = load_quarterly_data(
        test_data_path,
        test_start,
        test_end,
        symbol=symbol,
    )

    return run_dimensionality_training(
        X_train,
        y_train,
        feature_names,
        X_test,
        y_test,
        config,
        symbol=symbol,
        train_period=f"{train_start} to {train_end}",
        test_period=f"{test_start} to {test_end}",
        drift_detector=drift_detector,
    )


def create_quarterly_data_splits() -> Dict[str, Dict[str, str]]:
    return {
        "2024_Q1": {
            "start": "2024-01-01",
            "end": "2024-03-31"
        },
        "2024_Q2": {
            "start": "2024-04-01",
            "end": "2024-06-30"
        },
        "2024_Q3": {
            "start": "2024-07-01",
            "end": "2024-09-30"
        },
        "2024_Q4": {
            "start": "2024-10-01",
            "end": "2024-12-31"
        },
        "2025_Q1": {
            "start": "2025-01-01",
            "end": "2025-03-31"
        },
        "2025_Q2": {
            "start": "2025-04-01",
            "end": "2025-06-30"
        },
        "2025_Q3": {
            "start": "2025-07-01",
            "end": "2025-09-30"
        },
        "2025_Q4": {
            "start": "2025-10-01",
            "end": "2025-12-31"
        },
    }


def run_quarterly_rolling_training(
        args: argparse.Namespace) -> Tuple[list, Dict[str, any]]:
    print("🚀 Starting Quarterly Rolling Dimensionality Training")
    print("=" * 70)
    print("Training Period: 2024 (Full Year)")
    print("Testing Period: 2025 (Full Year)")
    print(f"Symbol: {args.symbol}")
    print("=" * 70)

    symbol_safe = args.symbol.replace("-", "_")
    config = RollingConfig(
        encoding_dim=args.encoding_dim,
        drift_threshold=args.drift_threshold,
        min_improvement=args.min_improvement,
        feature_eval_ratio=0.2,
        model_dir=Path(f"models/rolling_dim_{symbol_safe}"),
        results_dir=Path(f"results/rolling_dim_{symbol_safe}"),
    )
    config.ensure_dirs()

    quarters = create_quarterly_data_splits()
    all_results = []

    print("\n📊 Phase 1: Training on 2024 Full Year Data")
    print("-" * 50)

    train_results = run_dimensionality_training_from_paths(
        config,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-03-31",
        symbol=args.symbol,
    )

    all_results.append({
        "phase": "2024_full_year_training",
        "results": train_results
    })

    print("\n📊 Phase 2: Quarterly Testing on 2025 Data")
    print("-" * 50)

    for quarter_name, quarter_dates in quarters.items():
        if not quarter_name.startswith("2025"):
            continue

        print(f"\n🔍 Testing on {quarter_name}")
        print(f"   Period: {quarter_dates['start']} to {quarter_dates['end']}")

        quarter_results = run_dimensionality_training_from_paths(
            config,
            train_data_path=args.train_data_path,
            test_data_path=args.test_data_path,
            train_start="2024-01-01",
            train_end="2024-12-31",
            test_start=quarter_dates["start"],
            test_end=quarter_dates["end"],
            symbol=args.symbol,
        )

        all_results.append({
            "phase": f"quarterly_test_{quarter_name}",
            "results": quarter_results
        })

        print(f"✅ {quarter_name} testing complete")

    print("\n📋 Phase 3: Generating Comprehensive Report")
    print("-" * 50)

    summary_report = generate_comprehensive_report(all_results, args.symbol)
    save_final_results(all_results, summary_report, args.symbol)

    print("\n" + "=" * 70)
    print("🎉 Quarterly Rolling Dimensionality Training Complete!")
    print(f"📊 Total phases completed: {len(all_results)}")
    print("🎯 Overall performance summary available in results/")

    return all_results, summary_report


def generate_comprehensive_report(
    all_results: List[Dict[str, any]],
    symbol: str,
) -> Dict[str, any]:
    print("📋 Generating comprehensive report...")

    training_results = []
    testing_results = []

    for result in all_results:
        if "training" in result["phase"]:
            training_results.append(result["results"])
        else:
            testing_results.append(result["results"])

    if training_results:
        train_original_r2 = np.mean(
            [r["original_performance"].get("r2", 0) for r in training_results])
        train_compressed_r2 = np.mean([
            r["compressed_performance"].get("r2", 0) for r in training_results
        ])
        train_improvement = train_compressed_r2 - train_original_r2

        summary_stats = {
            "training_period": "2024 Full Year",
            "training_original_r2": train_original_r2,
            "training_compressed_r2": train_compressed_r2,
            "training_improvement": train_improvement,
            "compression_ratio": training_results[0]["compression_ratio"],
            "selected_factors": training_results[0]["selected_factors_count"],
        }
    else:
        summary_stats = {"error": "No training results found"}

    if testing_results:
        test_r2_scores = [
            r["test_performance"].get("r2", 0) for r in testing_results
        ]
        summary_stats.update({
            "testing_period": "2025 Quarters",
            "average_test_r2": np.mean(test_r2_scores),
            "test_r2_std": np.std(test_r2_scores),
            "test_r2_min": np.min(test_r2_scores),
            "test_r2_max": np.max(test_r2_scores),
        })

    report = {
        "symbol": symbol,
        "timestamp": pd.Timestamp.now().isoformat(),
        "summary_statistics": summary_stats,
        "detailed_results": all_results,
        "recommendations": generate_recommendations(summary_stats),
    }

    return report


def generate_recommendations(summary_stats: Dict[str, any]) -> List[str]:
    recommendations: List[str] = []

    if "training_improvement" in summary_stats:
        improvement = summary_stats["training_improvement"]
        if improvement > 0.01:
            recommendations.append(
                "✅ Dimensionality reduction shows significant improvement (>1%)"
            )
        elif improvement > 0.005:
            recommendations.append(
                "✅ Dimensionality reduction shows moderate improvement (>0.5%)"
            )
        else:
            recommendations.append(
                "⚠️ Dimensionality reduction shows minimal improvement")

    if "average_test_r2" in summary_stats:
        test_r2 = summary_stats["average_test_r2"]
        if test_r2 > 0.6:
            recommendations.append(
                "✅ Model shows good generalization on test data")
        elif test_r2 > 0.4:
            recommendations.append("⚠️ Model shows moderate generalization")
        else:
            recommendations.append(
                "❌ Model shows poor generalization - consider feature engineering"
            )

    if "compression_ratio" in summary_stats:
        ratio = summary_stats["compression_ratio"]
        if ratio > 5:
            recommendations.append(
                "✅ High compression ratio achieved - efficient feature reduction"
            )
        else:
            recommendations.append(
                "⚠️ Low compression ratio - consider increasing encoding dimensions"
            )

    recommendations.append(
        "🔄 Continue monitoring feature drift and retrain quarterly")
    recommendations.append(
        "📊 Track model performance across different market conditions")

    return recommendations


def save_final_results(
    all_results: List[Dict[str, any]],
    summary_report: Dict[str, any],
    symbol: str,
) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results/rolling_dim_{symbol.replace('-', '_')}"
    os.makedirs(base_dir, exist_ok=True)

    results_file = os.path.join(base_dir, f"final_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "all_results": all_results,
                "summary_report": summary_report
            },
            f,
            indent=2,
            default=str,
        )

    summary_file = os.path.join(base_dir, "summary_report.json")
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2, default=str)

    print(f"💾 Final results saved to: {results_file}")
    print(f"💾 Summary report saved to: {summary_file}")


def run_drift_triggered_training(args: argparse.Namespace):
    print("🚀 Starting Drift-Triggered Training")
    print("=" * 50)

    symbol_safe = args.symbol.replace("-", "_")
    config = RollingConfig(
        encoding_dim=args.encoding_dim,
        drift_threshold=args.drift_threshold,
        min_improvement=args.min_improvement,
        feature_eval_ratio=0.2,
        model_dir=Path(f"models/rolling_dim_{symbol_safe}"),
        results_dir=Path(f"results/rolling_dim_{symbol_safe}"),
    )
    config.ensure_dirs()

    drift_detector = DriftDetector(js_threshold=args.drift_threshold)

    X_full, y_full, feature_names = load_quarterly_data(
        args.train_data_path,
        "2024-01-01",
        "2025-12-31",
        symbol=args.symbol,
    )

    if len(X_full) < 200:
        print("⚠️ Dataset too small for drift-triggered training")
        return {"error": "insufficient_data"}

    window_size = max(len(X_full) // 4, 200)
    step = max(window_size // 2, 100)

    results: List[Dict[str, Any]] = []

    for start_idx in range(0, len(X_full) - window_size, step):
        end_idx = start_idx + window_size
        test_start = end_idx
        test_end = min(test_start + step, len(X_full))

        if test_end - test_start < 50:
            break

        X_train = X_full[start_idx:end_idx]
        y_train = y_full[start_idx:end_idx]
        X_test = X_full[test_start:test_end]
        y_test = y_full[test_start:test_end]

        train_period = f"indices {start_idx}:{end_idx}"
        test_period = f"indices {test_start}:{test_end}"

        window_result = run_dimensionality_training(
            X_train,
            y_train,
            feature_names,
            X_test,
            y_test,
            config,
            symbol=args.symbol,
            train_period=train_period,
            test_period=test_period,
            drift_detector=drift_detector,
        )

        results.append(window_result)

    summary = summarize_window_results(results)

    print("🎉 Drift-triggered training complete!")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rolling Dimensionality Training with Quarterly Data", )

    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/train_2024.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/test_2025.csv",
        help="Path to test data",
    )
    parser.add_argument("--symbol",
                        type=str,
                        default="ETH-USD",
                        help="Trading symbol")

    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Encoding dimension for dimensionality reduction",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.3,
        help="Drift detection threshold",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum improvement threshold",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["quarterly", "drift-triggered"],
        default="quarterly",
        help="Training mode",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "quarterly":
        _, summary_report = run_quarterly_rolling_training(args)

        print("\n📊 Final Summary:")
        print("-" * 30)
        if "summary_statistics" in summary_report:
            stats = summary_report["summary_statistics"]
            print(
                f"Training R²: {stats.get('training_original_r2', 0):.3f} → {stats.get('training_compressed_r2', 0):.3f}"
            )
            print(f"Improvement: {stats.get('training_improvement', 0):.3f}")
            print(f"Test R²: {stats.get('average_test_r2', 0):.3f}")
            print(
                f"Compression Ratio: {stats.get('compression_ratio', 0):.1f}x")

        print("\n🎯 Recommendations:")
        for rec in summary_report.get("recommendations", []):
            print(f"  {rec}")
    else:
        results = run_drift_triggered_training(args)
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
