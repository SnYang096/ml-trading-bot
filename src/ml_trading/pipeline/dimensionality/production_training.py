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
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
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

    # Optional grid table
    grid_rows = []
    grid = results.get("grid_search", [])
    if grid:
        for row in grid:
            grid_rows.append(
                f"<tr><td>{row.get('encoding_dim','-')}</td>"
                f"<td>{_format_float(row.get('r2_original'))}</td>"
                f"<td>{_format_float(row.get('r2_compressed'))}</td>"
                f"<td>{_format_float(row.get('delta_r2'))}</td>"
                f"<td>{_format_float(row.get('rmse_original'))}</td>"
                f"<td>{_format_float(row.get('rmse_compressed'))}</td>"
                "</tr>")

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

{('<h2>Encoding Grid Results</h2>'
  '<table>'
  '<tr><th>ENCODING_DIM</th><th>R² Original</th><th>R² Compressed</th><th>ΔR²</th><th>RMSE Original</th><th>RMSE Compressed</th></tr>'
  f"{''.join(grid_rows)}"
  '</table>') if grid_rows else ''}

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

        # Debug: engineered feature summary
        try:
            print(
                f"[DEBUG] Engineered features: total={len(feature_cols)} | sample={feature_cols[:10]}"
            )
        except Exception:
            pass

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
        "Optional: number of top factors (informational; not applied in this script)",
    )

    args = parser.parse_args()

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
        # Load engineered features for IC & representative selection
        X_raw, y_raw, feature_names = load_real_market_data(
            args.data_path, args.symbol, args.train_start, args.train_end)
        dfX = pd.DataFrame(X_raw, columns=feature_names)
        y_series = pd.Series(y_raw)

        # IC (Spearman) ranking
        ic_scores = {}
        for col in dfX.columns:
            try:
                ic = spearmanr(dfX[col].values,
                               y_series.values,
                               nan_policy="omit")[0]
            except Exception:
                ic = 0.0
            ic_scores[col] = 0.0 if ic is None or np.isnan(ic) else ic
        top_sorted = sorted(ic_scores.items(),
                            key=lambda kv: abs(kv[1]),
                            reverse=True)
        top_cols = [c for c, _ in top_sorted[:120]]
        df_top = dfX[top_cols].copy()

        print(
            f"[DEBUG] IC ranking done: total={len(dfX.columns)} | top_by_|IC|={len(top_cols)}"
        )

        # Missing and stability filter
        keep = []
        for c in df_top.columns:
            s = df_top[c]
            if s.isna().mean() < 0.2 and s.std() > 1e-8:
                keep.append(c)
        df_top = df_top[keep].fillna(method="ffill").fillna(
            method="bfill").fillna(0.0)

        dropped_missing = len(top_cols) - len(keep)
        print(
            f"[DEBUG] Missing/stability filter: kept={len(keep)} | dropped={dropped_missing}"
        )

        # Greedy representative selection by correlation threshold (0.9)
        reps: list[str] = []
        if not df_top.empty:
            corr = df_top.corr().abs().fillna(0.0)
            for c in df_top.columns:
                if all(corr.loc[c, r] < 0.9 for r in reps):
                    reps.append(c)
        # Bound reps between 60 and 100
        if len(reps) < 60:
            reps = top_cols[:60]
        elif len(reps) > 100:
            reps = reps[:100]

        print(
            f"[DEBUG] Representative selection: reps={len(reps)} | sample={reps[:10]}"
        )

        X_rep = df_top[reps].values if set(reps).issubset(
            df_top.columns) else dfX[reps].fillna(0.0).values

        # Scale and sanitize
        scaler_rep = StandardScaler()
        X_rep_scaled = sanitize_features(scaler_rep.fit_transform(X_rep))
        print(
            f"[DEBUG] Label variance: y.std={float(np.std(y_series.values)):.6f}"
        )

        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(X_rep_scaled,
                                                            y_series.values,
                                                            test_size=0.3,
                                                            shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                        y_temp,
                                                        test_size=0.5,
                                                        shuffle=False)

        # AE dims: 60→32→16→8 (ensure <= num reps)
        trial_dims = [60, 32, 16, 8]
        trial_dims = [d for d in trial_dims if d <= X_train.shape[1]]
        grid_rows = []
        best_row = None
        best_result = None
        best_model = None
        best_ae = None
        best_dir = None

        for dim in trial_dims:
            try:
                ae, trainer, losses = train_production_autoencoder(
                    X_train, encoding_dim=dim, epochs=args.autoencoder_epochs)
                # Reconstruction MSE on val
                with torch.no_grad():
                    Xv = torch.as_tensor(X_val,
                                         dtype=torch.float32,
                                         device=next(ae.parameters()).device)
                    out = ae(Xv)
                    if isinstance(out, tuple) or isinstance(out, list):
                        recon = out[0].cpu().numpy()
                    else:
                        recon = out.cpu().numpy()
                recon_mse = float(np.mean((recon - X_val)**2))

                Z_train = trainer.transform(X_train)
                Z_val = trainer.transform(X_val)
                Z_test = trainer.transform(X_test)

                # Standardize AE embeddings before feeding to LightGBM
                z_scaler = StandardScaler()
                Z_train = z_scaler.fit_transform(Z_train)
                Z_val = z_scaler.transform(Z_val)
                Z_test = z_scaler.transform(Z_test)

                try:
                    z_var = float(np.var(Z_train))
                    print(
                        f"[DEBUG] AE dim={dim} | recon_mse={recon_mse:.6e} | Z_train_var={z_var:.6e}"
                    )
                except Exception:
                    pass

                model_compressed = train_production_lightgbm(
                    Z_train, y_train, Z_val, y_val)
                perf_comp = evaluate_model_performance(model_compressed,
                                                       Z_test, y_test,
                                                       f"AE{dim}")

                model_orig = train_production_lightgbm(X_train, y_train, X_val,
                                                       y_val)
                perf_orig = evaluate_model_performance(model_orig, X_test,
                                                       y_test, "OriginalReps")

                try:
                    print(
                        f"[DEBUG] LightGBM iters: original={getattr(model_orig, 'best_iteration', None)} | compressed={getattr(model_compressed, 'best_iteration', None)}"
                    )
                except Exception:
                    pass

                delta_r2 = perf_comp["r2"] - perf_orig["r2"]
                row = {
                    "encoding_dim": dim,
                    "reconstruction_mse": recon_mse,
                    "r2_original": perf_orig["r2"],
                    "r2_compressed": perf_comp["r2"],
                    "delta_r2": delta_r2,
                    "rmse_original": perf_orig["rmse"],
                    "rmse_compressed": perf_comp["rmse"],
                }
                grid_rows.append(row)
                if best_row is None or delta_r2 > best_row["delta_r2"]:
                    best_row = row
                    # Build minimal result struct for report
                    results = {
                        "timestamp_start": ablation_start_ts,
                        "timestamp_end":
                        datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "data_info": {
                            "representatives": len(reps),
                            "encoding_dim": dim
                        },
                        "performance": {
                            "original_features": perf_orig,
                            "compressed_features": perf_comp,
                            "performance_change": delta_r2,
                        },
                    }
                    # Proxy: map compressed predictions back to reps
                    y_hat_train = model_compressed.predict(Z_train)
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, y_hat_train)
                    proxy_coefs = {
                        reps[i]: float(ridge.coef_[i])
                        for i in range(len(reps))
                    }
                    results["proxy_weights"] = proxy_coefs
                    results["grid_search"] = grid_rows
                    best_result = results
                    best_model = model_compressed
                    best_ae = ae
                    best_dir = f"results/production_dimensionality_{results['timestamp_start']}_{results['timestamp_end']}"
            except Exception as exc:
                print(f"⚠️ Ablation ENCODING_DIM={dim} failed: {exc}")
                continue

        if best_result is None:
            raise RuntimeError("Ablation failed for all encoding dims")

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
            # rebuild dir to include final end ts if changed
            best_dir = f"results/production_dimensionality_{best_result['timestamp_start']}_{best_result['timestamp_end']}"
        os.makedirs(best_dir, exist_ok=True)

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
        default_report_path = os.path.join(best_dir,
                                           "dimensionality_report.html")
        write_html_report(best_result, default_report_path)
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
                trial_results, trial_model, trial_ae, trial_dir = run_production_training(
                    data_path=args.data_path,
                    symbol=args.symbol,
                    encoding_dim=dim,
                    autoencoder_epochs=args.autoencoder_epochs,
                    train_start=args.train_start,
                    train_end=args.train_end,
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
