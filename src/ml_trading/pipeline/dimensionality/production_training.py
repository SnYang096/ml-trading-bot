"""Production-grade dimensionality reduction training workflows."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder
from ml_trading.utils.training import train_lightgbm_model


def _format_float(val, digits: int = 4) -> str:
    try:
        if val is None or (isinstance(val, float) and
                           (np.isnan(val) or np.isinf(val))):
            return "NA"
        return f"{val:.{digits}f}"
    except Exception:
        return str(val)


def write_html_report(results: Dict, html_path: str) -> None:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    ts_start = results.get("timestamp_start", results.get("timestamp", "-"))
    ts_end = results.get("timestamp_end", "-")
    d = results.get("data_info", {})
    p = results.get("performance", {})
    train_info = results.get("training_info", {})

    orig = p.get("original_features", {})
    comp = p.get("compressed_features", {})
    delta_r2 = p.get("performance_change", None)

    conclusion = "Dimensionality reduction appears beneficial." if (
        delta_r2 is not None and delta_r2
        > 0) else "Dimensionality reduction is not beneficial under this run."

    html = f"""<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\"/><title>Dimensionality Reduction Comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;margin-top:16px;width:100%;max-width:900px}}th,td{{border:1px solid #ddd;padding:8px 10px;text-align:left}}th{{background:#f7f7f7}}.bad{{color:#b00020;font-weight:600}}.good{{color:#0a7c2f;font-weight:600}}.warn{{color:#b36b00;font-weight:600}}</style>
</head><body>
<h1>Dimensionality Reduction Comparison</h1>
<div>Start: {ts_start}  |  End: {ts_end}</div>

<h2>Data Summary</h2>
<table>
<tr><th>Original feature count</th><td>{d.get('original_features_count','-')}</td></tr>
<tr><th>Compressed dimensions</th><td>{d.get('compressed_dimensions','-')}</td></tr>
<tr><th>Compression ratio</th><td>{_format_float(d.get('compression_ratio'),2)}x</td></tr>
<tr><th>Samples (train/val/test)</th><td>{d.get('training_samples','-')} / {d.get('validation_samples','-')} / {d.get('test_samples','-')}</td></tr>
</table>

<h2>Performance (Test Set)</h2>
<table>
<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>
<tr><td>R²</td><td>{_format_float(orig.get('r2'))}</td><td>{_format_float(comp.get('r2'))}</td><td>{_format_float(delta_r2)}</td></tr>
<tr><td>RMSE</td><td>{_format_float(orig.get('rmse'))}</td><td>{_format_float(comp.get('rmse'))}</td><td>{_format_float((comp.get('rmse') or 0)-(orig.get('rmse') or 0))}</td></tr>
<tr><td>MAE</td><td>{_format_float(orig.get('mae'))}</td><td>{_format_float(comp.get('mae'))}</td><td>{_format_float((comp.get('mae') or 0)-(orig.get('mae') or 0))}</td></tr>
</table>

<h2>Training Diagnostics</h2>
<ul>
<li>Autoencoder epochs: {train_info.get('autoencoder_epochs','-')}</li>
<li>Autoencoder final loss: {_format_float(train_info.get('autoencoder_final_loss'))}</li>
<li>LightGBM iterations (original/compressed): {train_info.get('lightgbm_original_iterations','-')} / {train_info.get('lightgbm_compressed_iterations','-')}</li>
</ul>

<h2>Conclusion</h2>
<p>{conclusion}</p>
</body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📝 HTML report written to: {html_path}")


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


def load_real_market_data(
    data_path: str,
    symbol: str = "ETH-USD",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    print(f"📊 Loading real market data for {symbol}...")

    try:
        loader = MarketDataLoader(data_path)
        df = loader.load_data(symbol=symbol,
                              start_date=start_date,
                              end_date=end_date)

        if df is None or df.empty:
            print("⚠️ No real data found, generating sample data...")
            return create_enhanced_sample_data()

        df = loader.resample_data("5T")

        comprehensive_engineer = ComprehensiveFeatureEngineer()
        df_features = comprehensive_engineer.engineer_all_features(df,
                                                                   fit=True)

        feature_cols = [
            col for col in df_features.columns
            if col not in ["timestamp", "close"]
        ]

        X = df_features[feature_cols].values
        y = df_features["close"].pct_change().shift(-1).dropna().values

        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        print(f"✅ Real data loaded: {X.shape}, {y.shape}")
        return X, y, feature_cols

    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Error loading real data: {exc}")
        print("📊 Generating sample data...")
        return create_enhanced_sample_data()


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
):
    print(f"🧠 Training production Autoencoder for {epochs} epochs...")

    autoencoder = UnifiedAutoencoder(
        input_dim=X.shape[1],
        encoding_dim=encoding_dim,
        architecture="production",
    )

    trainer = AutoencoderTrainer(autoencoder, device="auto")
    losses = trainer.train(X, epochs=epochs, verbose=True)

    print("✅ Production Autoencoder training complete")
    return autoencoder, trainer, losses


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
    if float(np.std(y_train)) == 0.0:
        raise ValueError("y_train variance is zero; cannot train a regressor")

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 1e-3,
            "verbose": -1,
            "random_state": 42,
            # Prefer CUDA backend if available (LightGBM built with CUDA)
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # Use callbacks for broad LightGBM version compatibility
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(period=200),
    ]
    try:
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
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
            num_boost_round=2000,
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


def evaluate_model_performance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"📊 {model_name} Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
    }


def save_production_results(
    results: Dict,
    model,
    autoencoder: UnifiedAutoencoder,
    results_dir: str,
) -> str:
    print("💾 Saving production results...")
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/production_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump(model, f"{results_dir}/production_model.pkl")
    torch.save(autoencoder.state_dict(),
               f"{results_dir}/production_autoencoder.pth")

    print(f"✅ Results saved to {results_dir}")
    return results_dir


def run_production_training(
    data_path: str = "/data/parquet_data",
    symbol: str = "ETH-USD",
    encoding_dim: int = 8,
    autoencoder_epochs: int = 500,
    train_start: str | None = None,
    train_end: str | None = None,
) -> Tuple[Dict, any, UnifiedAutoencoder, str]:
    print("🚀 Production Dimensionality Reduction Training")
    print("=" * 60)
    start_dt = datetime.now()
    timestamp_start = start_dt.strftime("%Y%m%d_%H%M%S")

    X, y, feature_names = load_real_market_data(data_path,
                                                symbol,
                                                start_date=train_start,
                                                end_date=train_end)

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

    print("\n🧠 Training production Autoencoder...")
    autoencoder, trainer, train_losses = train_production_autoencoder(
        X_train,
        encoding_dim=encoding_dim,
        epochs=autoencoder_epochs,
    )

    print("\n📊 Extracting embeddings...")
    X_train_emb = trainer.transform(X_train)
    X_val_emb = trainer.transform(X_val)
    X_test_emb = trainer.transform(X_test)

    # Validate embeddings are finite and have variance
    if not np.isfinite(X_train_emb).all() or not np.isfinite(X_val_emb).all():
        raise ValueError("Autoencoder embeddings contain NaN/inf values")
    if float(np.std(X_train_emb)) == 0.0:
        raise ValueError(
            "Autoencoder embeddings have zero variance; model would not train")

    print(f"✅ Embeddings extracted: {X_train_emb.shape}")

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
    results_original = evaluate_model_performance(
        model_original,
        X_test,
        y_test,
        "Original Features",
    )
    results_compressed = evaluate_model_performance(
        model_compressed,
        X_test_emb,
        y_test,
        "Compressed Features",
    )

    print("\n📋 Generating production report...")

    compression_ratio = X.shape[1] / X_train_emb.shape[1]
    performance_change = results_compressed["r2"] - results_original["r2"]

    results = {
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "duration_sec": (datetime.now() - start_dt).total_seconds(),
        "data_info": {
            "original_features_count": X.shape[1],
            "compressed_dimensions": X_train_emb.shape[1],
            "compression_ratio": compression_ratio,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "training_info": {
            "autoencoder_epochs": autoencoder_epochs,
            "autoencoder_final_loss": train_losses[-1],
            "lightgbm_original_iterations": model_original.best_iteration,
            "lightgbm_compressed_iterations": model_compressed.best_iteration,
        },
        "performance": {
            "original_features":
            results_original,
            "compressed_features":
            results_compressed,
            "performance_change":
            performance_change,
            "performance_change_percent":
            (performance_change / results_original["r2"]) * 100,
        },
        "model_info": {
            "device_used": str(autoencoder.encoder[0].weight.device),
            "cuda_available": torch.cuda.is_available(),
            "feature_names": feature_names[:10],
        },
    }

    # Build results directory name using start and end timestamps
    results_dir = f"results/production_dimensionality_{results['timestamp_start']}_{results['timestamp_end']}"
    results_dir = save_production_results(
        results,
        model_compressed,
        autoencoder,
        results_dir,
    )

    print("\n" + "=" * 60)
    print("🎉 Production Dimensionality Reduction Training Complete!")
    print("=" * 60)
    print(f"📊 Compression Ratio: {compression_ratio:.1f}x")
    print(
        f"📈 Performance Change: {performance_change:.4f} ({results['performance']['performance_change_percent']:.1f}%)"
    )
    print(f"💾 Results saved to: {results_dir}")
    print("🔧 Model ready for production deployment!")

    return results, model_compressed, autoencoder, results_dir


def main() -> Tuple[Dict, any, UnifiedAutoencoder, str]:
    parser = argparse.ArgumentParser(
        description="Production-style comparison: original vs compressed/Top-K",
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
        help="Symbol name (e.g., BTC-USD, ETH-USD)",
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Autoencoder embedding dimension",
    )
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=500,
        help="Autoencoder training epochs",
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
        "--top-k",
        type=int,
        default=None,
        help=
        "Optional: number of top factors (informational; not applied in this script)",
    )

    args = parser.parse_args()

    results, model, autoencoder, results_dir = run_production_training(
        data_path=args.data_path,
        symbol=args.symbol,
        encoding_dim=args.encoding_dim,
        autoencoder_epochs=args.autoencoder_epochs,
        train_start=args.train_start,
        train_end=args.train_end,
    )

    # Record Top-K hint if provided
    if args.top_k is not None:
        results.setdefault("training_info", {})["top_k"] = args.top_k

    # Always write a report into the results directory
    try:
        default_report_path = os.path.join(results_dir,
                                           "dimensionality_report.html")
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
    return results, model, autoencoder, results_dir


if __name__ == "__main__":
    try:
        results, model, autoencoder, results_dir = main()
        print("\n✅ Production training completed successfully!")
        print(
            f"📊 Final compression ratio: {results['data_info']['compression_ratio']:.1f}x"
        )
        print(
            f"📈 Performance change: {results['performance']['performance_change']:.4f}"
        )
        print(f"💾 Results directory: {results_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Production training failed: {exc}")
        import traceback

        traceback.print_exc()
        raise
