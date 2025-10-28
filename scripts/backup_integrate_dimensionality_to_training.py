#!/usr/bin/env python3
"""
将降维结果集成到实际模型训练中
展示如何在实际交易系统中使用降维后的特征
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

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import json
import joblib

# 添加src到路径
sys.path.append("/workspace/src")


class DimensionalityIntegrationEngine:
    """降维集成引擎 - 将降维结果集成到实际训练中"""

    def __init__(self, model_path: str, autoencoder_path: str, results_path: str):
        """
        初始化降维集成引擎

        Args:
            model_path: 训练好的LightGBM模型路径
            autoencoder_path: 训练好的Autoencoder模型路径
            results_path: 训练结果JSON文件路径
        """
        self.model_path = model_path
        self.autoencoder_path = autoencoder_path
        self.results_path = results_path

        # 加载结果
        with open(results_path, "r") as f:
            self.results = json.load(f)

        # 加载模型
        self.model = joblib.load(model_path)

        # 加载Autoencoder
        self.autoencoder = self._load_autoencoder(autoencoder_path)

        print(f"✅ DimensionalityIntegrationEngine initialized")
        print(
            f"📊 Compression ratio: {self.results['data_info']['compression_ratio']:.1f}x"
        )
        print(
            f"📈 Performance: R² = {self.results['performance']['compressed_features']['r2']:.4f}"
        )

    def _load_autoencoder(self, autoencoder_path: str):
        """加载Autoencoder模型"""
        # 重建Autoencoder结构
        input_dim = self.results["data_info"]["original_features_count"]
        encoding_dim = self.results["data_info"]["compressed_dimensions"]

        autoencoder = ProductionAutoencoder(input_dim, encoding_dim)
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location="cpu"))
        autoencoder.eval()

        return autoencoder

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        将原始特征转换为降维后的特征

        Args:
            X: 原始特征矩阵 (n_samples, n_features)

        Returns:
            X_compressed: 降维后的特征矩阵 (n_samples, n_compressed_features)
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            _, X_compressed = self.autoencoder(X_tensor)
            return X_compressed.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用降维后的特征进行预测

        Args:
            X: 原始特征矩阵

        Returns:
            predictions: 预测结果
        """
        X_compressed = self.transform_features(X)
        return self.model.predict(X_compressed)

    def get_feature_importance(self) -> dict:
        """获取特征重要性"""
        if hasattr(self.model, "feature_importance"):
            importance = self.model.feature_importance(importance_type="gain")
            return {
                f"compressed_feature_{i}": importance[i] for i in range(len(importance))
            }
        return {}

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "compression_ratio": self.results["data_info"]["compression_ratio"],
            "original_features": self.results["data_info"]["original_features_count"],
            "compressed_features": self.results["data_info"]["compressed_dimensions"],
            "performance": self.results["performance"]["compressed_features"],
            "training_info": self.results["training_info"],
        }


class ProductionAutoencoder(nn.Module):
    """生产级Autoencoder"""

    def __init__(
        self, input_dim: int, encoding_dim: int = 8, dropout_rate: float = 0.2
    ):
        super(ProductionAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def integrate_with_existing_training():
    """集成到现有训练流程中"""
    print("🔗 Integrating dimensionality reduction with existing training...")

    # 1. 查找最新的降维结果
    results_dir = "results/production_dimensionality_*"
    import glob

    result_dirs = glob.glob(results_dir)

    if not result_dirs:
        print("❌ No production dimensionality results found!")
        print("Please run production_dimensionality_training.py first")
        return None

    # 使用最新的结果
    latest_dir = max(result_dirs, key=os.path.getctime)
    print(f"📁 Using latest results from: {latest_dir}")

    # 2. 初始化集成引擎
    model_path = os.path.join(latest_dir, "production_model.pkl")
    autoencoder_path = os.path.join(latest_dir, "production_autoencoder.pth")
    results_path = os.path.join(latest_dir, "production_results.json")

    integration_engine = DimensionalityIntegrationEngine(
        model_path, autoencoder_path, results_path
    )

    # 3. 加载新数据（模拟新的市场数据）
    print("\n📊 Loading new market data...")
    X_new, y_new, feature_names = load_new_market_data()

    # 4. 使用降维后的特征进行预测
    print("\n🔮 Making predictions with compressed features...")
    predictions = integration_engine.predict(X_new)

    # 5. 评估性能
    print("\n📊 Evaluating performance on new data...")
    mse = mean_squared_error(y_new, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_new, predictions)
    r2 = r2_score(y_new, predictions)

    print(f"📈 Performance on new data:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # 6. 生成集成报告
    integration_report = {
        "integration_timestamp": datetime.now().isoformat(),
        "model_info": integration_engine.get_model_info(),
        "new_data_performance": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
        "feature_importance": integration_engine.get_feature_importance(),
    }

    # 7. 保存集成报告
    integration_dir = f"results/integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(integration_dir, exist_ok=True)

    with open(os.path.join(integration_dir, "integration_report.json"), "w") as f:
        json.dump(integration_report, f, indent=2, default=str)

    print(f"✅ Integration report saved to: {integration_dir}")

    return integration_engine, integration_report


def load_new_market_data():
    """加载新的市场数据（模拟）"""
    print("📊 Loading new market data...")

    # 这里可以加载真实的新市场数据
    # 现在使用增强的样本数据模拟

    np.random.seed(123)  # 不同的随机种子模拟新数据

    n_samples = 5000
    n_factors = 100

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

    # 生成新数据
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

    y = (
        np.tanh(X[:, momentum_factors].mean(axis=1)) * 0.4
        + np.sin(X[:, volatility_factors].mean(axis=1)) * 0.3
        + X[:, trend_factors].mean(axis=1) * 0.2
        + np.random.randn(n_samples) * 0.1
    )

    return X, y, factor_names


def demonstrate_production_usage():
    """演示生产环境中的使用方式"""
    print("🏭 Demonstrating production usage...")

    # 1. 集成降维结果
    integration_engine, report = integrate_with_existing_training()

    if integration_engine is None:
        return

    # 2. 模拟实时预测
    print("\n⚡ Simulating real-time predictions...")

    # 模拟实时数据流
    for i in range(5):
        # 模拟新的市场数据点
        X_realtime = np.random.randn(1, 100)  # 单个数据点

        # 使用降维后的特征进行预测
        prediction = integration_engine.predict(X_realtime)

        print(f"  Prediction {i+1}: {prediction[0]:.6f}")

    # 3. 展示模型信息
    print("\n📊 Model information:")
    model_info = integration_engine.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("\n🎉 Production integration demonstration complete!")


def main():
    """主函数"""
    print("🚀 Dimensionality Reduction Integration Demo")
    print("=" * 60)

    try:
        demonstrate_production_usage()
        print("\n✅ Integration demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Integration demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
