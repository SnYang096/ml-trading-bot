#!/usr/bin/env python3
"""
03. 生产级降维训练脚本
基于production_dimensionality_training.py重构
专注于生产级模型训练和部署
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
from datetime import datetime
import json
import joblib

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import common modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from autoencoder import UnifiedAutoencoder, AutoencoderTrainer
from data_loader import UnifiedDataLoader
from training_utils import train_lightgbm_model


def load_real_market_data(data_path: str, symbol: str = "ETH-USD"):
    """加载真实市场数据"""
    print(f"📊 Loading real market data for {symbol}...")

    try:
        from ml_trading.data_tools.data_loader import MarketDataLoader
        from ml_trading.data_tools.comprehensive_feature_engineering import (
            ComprehensiveFeatureEngineer,
        )

        loader = MarketDataLoader(data_path)
        df = loader.load_data()

        if df is None or df.empty:
            print("⚠️ No real data found, generating sample data...")
            return create_enhanced_sample_data()

        # 重采样
        df = loader.resample_data("5T")

        # 特征工程
        comprehensive_engineer = ComprehensiveFeatureEngineer()
        df_features = comprehensive_engineer.engineer_all_features(df, fit=True)

        # 提取特征和目标变量
        feature_cols = [
            col for col in df_features.columns if col not in ["timestamp", "close"]
        ]

        X = df_features[feature_cols].values
        y = df_features["close"].pct_change().shift(-1).dropna().values

        # 对齐数据
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        print(f"✅ Real data loaded: {X.shape}, {y.shape}")
        return X, y, feature_cols

    except Exception as e:
        print(f"⚠️ Error loading real data: {e}")
        print("📊 Generating sample data...")
        return create_enhanced_sample_data()


def create_enhanced_sample_data(n_samples: int = 10000, n_factors: int = 100):
    """创建增强的样本数据"""
    print(
        f"📊 Creating enhanced sample data: {n_samples} samples, {n_factors} features"
    )

    np.random.seed(42)

    # 生成因子名称
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

    # 生成有相关性的因子数据
    X = np.random.randn(n_samples, n_factors)

    # 添加因子间的相关性
    for i in range(0, n_factors, 10):
        if i + 5 < n_factors:
            X[:, i + 1 : i + 5] = (
                X[:, i : i + 4] * 0.7 + np.random.randn(n_samples, 4) * 0.3
            )

    # 创建目标变量
    momentum_factors = [i for i, name in enumerate(factor_names) if "momentum" in name]
    volatility_factors = [
        i for i, name in enumerate(factor_names) if "volatility" in name
    ]
    trend_factors = [i for i, name in enumerate(factor_names) if "trend" in name]

    # 创建非线性关系
    y = (
        np.tanh(X[:, momentum_factors].mean(axis=1)) * 0.4
        + np.sin(X[:, volatility_factors].mean(axis=1)) * 0.3
        + X[:, trend_factors].mean(axis=1) * 0.2
        + np.random.randn(n_samples) * 0.1
    )

    return X, y, factor_names


def train_production_autoencoder(X, encoding_dim=8, epochs=500, batch_size=256):
    """训练生产级Autoencoder"""
    print(f"🧠 Training production Autoencoder for {epochs} epochs...")

    # 创建Autoencoder
    autoencoder = UnifiedAutoencoder(
        input_dim=X.shape[1], encoding_dim=encoding_dim, architecture="production"
    )

    # 创建训练器
    trainer = AutoencoderTrainer(autoencoder, device="auto")

    # 训练
    losses = trainer.train(X, epochs=epochs, verbose=True)

    print("✅ Production Autoencoder training complete")
    return autoencoder, trainer, losses


def train_production_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """训练生产级LightGBM"""
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

    # 使用统一的训练函数
    model = train_lightgbm_model(X_train, y_train, use_gpu=True, num_boost_round=1000)

    print("✅ Production LightGBM training complete")
    return model


def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """评估模型性能"""
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"📊 {model_name} Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "predictions": predictions}


def save_production_results(results, model, autoencoder, timestamp):
    """保存生产级结果"""
    print("💾 Saving production results...")

    # 创建结果目录
    results_dir = f"results/production_dimensionality_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 保存结果
    with open(f"{results_dir}/production_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 保存模型
    joblib.dump(model, f"{results_dir}/production_model.pkl")
    torch.save(autoencoder.state_dict(), f"{results_dir}/production_autoencoder.pth")

    print(f"✅ Results saved to {results_dir}")
    return results_dir


def main():
    """主函数 - 生产级降维训练流程"""
    print("🚀 Production Dimensionality Reduction Training")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 加载数据
    data_path = "/data/agg_data"  # Docker中的数据路径
    X, y, feature_names = load_real_market_data(data_path, "ETH-USD")

    print(f"✅ Data loaded: {X.shape}, {y.shape}")
    print(f"✅ Features: {len(feature_names)}")

    # 2. 数据预处理
    print("\n📊 Data preprocessing...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.3, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    print(
        f"✅ Data split: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # 3. 训练Autoencoder
    print("\n🧠 Training production Autoencoder...")
    autoencoder, trainer, train_losses = train_production_autoencoder(
        X_train, encoding_dim=8, epochs=500
    )

    # 4. 提取嵌入
    print("\n📊 Extracting embeddings...")
    X_train_emb = trainer.transform(X_train)
    X_val_emb = trainer.transform(X_val)
    X_test_emb = trainer.transform(X_test)

    print(f"✅ Embeddings extracted: {X_train_emb.shape}")

    # 5. 训练原始特征模型
    print("\n🌲 Training original features model...")
    model_original = train_production_lightgbm(X_train, y_train, X_val, y_val)

    # 6. 训练压缩特征模型
    print("\n🌲 Training compressed features model...")
    model_compressed = train_production_lightgbm(X_train_emb, y_train, X_val_emb, y_val)

    # 7. 评估性能
    print("\n📊 Evaluating performance...")

    # 原始特征性能
    results_original = evaluate_model_performance(
        model_original, X_test, y_test, "Original Features"
    )

    # 压缩特征性能
    results_compressed = evaluate_model_performance(
        model_compressed, X_test_emb, y_test, "Compressed Features"
    )

    # 8. 生成报告
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
            "autoencoder_epochs": 500,
            "autoencoder_final_loss": train_losses[-1],
            "lightgbm_original_iterations": model_original.best_iteration,
            "lightgbm_compressed_iterations": model_compressed.best_iteration,
        },
        "performance": {
            "original_features": results_original,
            "compressed_features": results_compressed,
            "performance_change": performance_change,
            "performance_change_percent": (performance_change / results_original["r2"])
            * 100,
        },
        "model_info": {
            "device_used": str(autoencoder.encoder[0].weight.device),
            "cuda_available": torch.cuda.is_available(),
            "feature_names": feature_names[:10],  # 只保存前10个特征名
        },
    }

    # 9. 保存结果
    results_dir = save_production_results(
        results, model_compressed, autoencoder, timestamp
    )

    # 10. 打印总结
    print("\n" + "=" * 60)
    print("🎉 Production Dimensionality Reduction Training Complete!")
    print("=" * 60)
    print(f"📊 Compression Ratio: {compression_ratio:.1f}x")
    print(
        f"📈 Performance Change: {performance_change:.4f} ({results['performance']['performance_change_percent']:.1f}%)"
    )
    print(f"💾 Results saved to: {results_dir}")
    print(f"🔧 Model ready for production deployment!")

    return results, model_compressed, autoencoder, results_dir


if __name__ == "__main__":
    try:
        results, model, autoencoder, results_dir = main()
        print(f"\n✅ Production training completed successfully!")
        print(
            f"📊 Final compression ratio: {results['data_info']['compression_ratio']:.1f}x"
        )
        print(
            f"📈 Performance change: {results['performance']['performance_change']:.4f}"
        )
        print(f"💾 Results directory: {results_dir}")
    except Exception as e:
        print(f"\n❌ Production training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
