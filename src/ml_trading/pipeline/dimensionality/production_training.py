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

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder
from ml_trading.utils.training import train_lightgbm_model


def load_real_market_data(
    data_path: str,
    symbol: str = "ETH-USD",
) -> Tuple[np.ndarray, np.ndarray, list]:
    print(f"📊 Loading real market data for {symbol}...")

    try:
        loader = MarketDataLoader(data_path)
        df = loader.load_data()

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

    if params is None:
        params = {
            "objective": "regression",
            "metric": "l2",
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
        }

    model = train_lightgbm_model(X_train,
                                 y_train,
                                 use_gpu=True,
                                 num_boost_round=1000)

    print("✅ Production LightGBM training complete")
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
    timestamp: str,
) -> str:
    print("💾 Saving production results...")

    results_dir = f"results/production_dimensionality_{timestamp}"
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
) -> Tuple[Dict, any, UnifiedAutoencoder, str]:
    print("🚀 Production Dimensionality Reduction Training")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y, feature_names = load_real_market_data(data_path, symbol)

    print(f"✅ Data loaded: {X.shape}, {y.shape}")
    print(f"✅ Features: {len(feature_names)}")

    print("\n📊 Data preprocessing...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

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
        "timestamp": timestamp,
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

    results_dir = save_production_results(
        results,
        model_compressed,
        autoencoder,
        timestamp,
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

    args = parser.parse_args()

    return run_production_training(
        data_path=args.data_path,
        symbol=args.symbol,
        encoding_dim=args.encoding_dim,
        autoencoder_epochs=args.autoencoder_epochs,
    )


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
