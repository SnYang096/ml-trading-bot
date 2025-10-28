#!/usr/bin/env python3
"""
01. 特征工程脚本
基于enhanced_rolling_dimensionality_training.py重构
专注于特征生成、IC/IR筛选和特征分析
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import common modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from autoencoder import UnifiedAutoencoder, AutoencoderTrainer
from data_loader import UnifiedDataLoader
from training_utils import train_lightgbm_model

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
)


def calculate_ic_ir(features, target, window=252):
    """计算IC和IR"""
    print(f"📊 计算IC/IR指标...")

    ic_values = []
    ir_values = []

    for col in features.columns:
        try:
            # 计算滚动IC
            ic_rolling = features[col].rolling(window).corr(target)
            ic_mean = ic_rolling.mean()
            ic_std = ic_rolling.std()

            ic_values.append(ic_mean)

            # 计算IR
            if ic_std > 0:
                ir = ic_mean / ic_std
            else:
                ir = 0
            ir_values.append(ir)

        except Exception as e:
            ic_values.append(0)
            ir_values.append(0)

    ic_series = pd.Series(ic_values, index=features.columns)
    ir_series = pd.Series(ir_values, index=features.columns)

    return ic_series, ir_series


def filter_features_by_ic_ir(features, target, ic_threshold=0.05, ir_threshold=0.1):
    """根据IC/IR筛选特征"""
    print(f"🔍 IC/IR特征筛选...")

    ic_values, ir_values = calculate_ic_ir(features, target)

    # 筛选条件
    good_features = (ic_values.abs() > ic_threshold) & (ir_values.abs() > ir_threshold)

    print(f"  原始特征数量: {len(features.columns)}")
    print(f"  IC>0.05的特征: {(ic_values.abs() > ic_threshold).sum()}")
    print(f"  IR>0.1的特征: {(ir_values.abs() > ir_threshold).sum()}")
    print(f"  筛选后特征数量: {good_features.sum()}")

    # 显示Top 10 IC特征
    top_ic = ic_values.abs().nlargest(10)
    print(f"\n  Top 10 IC特征:")
    for i, (feature, ic) in enumerate(top_ic.items(), 1):
        print(f"    {i:2d}. {feature:<30} IC: {ic:.4f}")

    return features.loc[:, good_features], ic_values, ir_values


def apply_enhanced_dimensionality_reduction(
    X_train, X_test, method="autoencoder", encoding_dim=8
):
    """应用增强版降维"""
    print(f"🔧 应用{method}降维: {X_train.shape[1]} -> {encoding_dim}")

    if method == "autoencoder":
        return apply_autoencoder_reduction(X_train, X_test, encoding_dim)
    elif method == "pca":
        return apply_pca_reduction(X_train, X_test, encoding_dim)
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_autoencoder_reduction(X_train, X_test, encoding_dim=8):
    """应用Autoencoder降维"""
    print(f"🧠 训练Autoencoder...")

    # 创建Autoencoder
    autoencoder = UnifiedAutoencoder(
        input_dim=X_train.shape[1], encoding_dim=encoding_dim, architecture="deep"
    )

    # 创建训练器
    trainer = AutoencoderTrainer(autoencoder, device="auto")

    # 训练
    losses = trainer.train(X_train, epochs=300, verbose=True)

    # 提取嵌入
    X_train_emb = trainer.transform(X_train)
    X_test_emb = trainer.transform(X_test)

    print(f"✅ Autoencoder训练完成")
    return X_train_emb, X_test_emb, autoencoder, None


def apply_pca_reduction(X_train, X_test, n_components=8):
    """应用PCA降维"""
    print(f"📊 应用PCA降维...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_variance = pca.explained_variance_ratio_
    print(f"  解释方差: {explained_variance.sum():.3f}")

    return X_train_pca, X_test_pca, pca, scaler


def explain_compressed_features(
    original_features, compressed_features, feature_names, method="autoencoder"
):
    """解释降维后的特征"""
    print(f"🔍 分析降维后特征...")

    # 计算相关性
    correlations = []
    for i in range(compressed_features.shape[1]):
        corr_with_compressed = []
        for j in range(original_features.shape[1]):
            corr = np.corrcoef(original_features[:, j], compressed_features[:, i])[0, 1]
            corr_with_compressed.append(corr)
        correlations.append(corr_with_compressed)

    # 找到每个压缩特征最相关的原始特征
    print(f"  降维后特征解释:")
    for i in range(compressed_features.shape[1]):
        corr_values = correlations[i]
        top_indices = np.argsort(np.abs(corr_values))[-5:][::-1]

        print(f"    压缩特征 {i+1}:")
        for j, idx in enumerate(top_indices):
            if idx < len(feature_names):
                print(
                    f"      {j+1}. {feature_names[idx]:<30} 相关性: {corr_values[idx]:.4f}"
                )


def analyze_feature_types(filtered_features):
    """分析特征类型"""
    print(f"\n📊 筛选后特征类型分析:")

    hurst_count = sum(1 for col in filtered_features.columns if "hurst" in col)
    wpt_count = sum(1 for col in filtered_features.columns if "wpt_" in col)
    hilbert_count = sum(1 for col in filtered_features.columns if "hilbert" in col)
    spectral_count = sum(1 for col in filtered_features.columns if "spectral" in col)
    order_flow_count = sum(
        1
        for col in filtered_features.columns
        if any(x in col for x in ["cvd", "ofi", "tbr", "order_flow"])
    )

    print(f"  Hurst特征: {hurst_count}")
    print(f"  小波包特征: {wpt_count}")
    print(f"  Hilbert特征: {hilbert_count}")
    print(f"  光谱特征: {spectral_count}")
    print(f"  订单流特征: {order_flow_count}")

    return {
        "hurst_features": hurst_count,
        "wpt_features": wpt_count,
        "hilbert_features": hilbert_count,
        "spectral_features": spectral_count,
        "order_flow_features": order_flow_count,
    }


def main():
    """主函数"""
    print("🚀 特征工程和降维分析")
    print("=" * 60)

    # 1. 创建样本数据
    print("📊 创建样本数据...")
    dates = pd.date_range("2024-01-01", periods=2000, freq="5T")
    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.randn(2000).cumsum() + 100,
            "high": np.random.randn(2000).cumsum() + 105,
            "low": np.random.randn(2000).cumsum() + 95,
            "close": np.random.randn(2000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 2000),
            "cvd": np.random.randn(2000).cumsum(),
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, 2000),
        }
    )

    # 2. 特征工程
    print("🔧 综合特征工程...")
    comprehensive_engineer = ComprehensiveFeatureEngineer()
    df = comprehensive_engineer.engineer_all_features(sample_data, fit=True)

    # 检查特征数量
    feature_cols = [
        col
        for col in df.columns
        if col not in ["timestamp", "open", "high", "low", "close", "volume"]
    ]

    print(f"✅ 特征工程完成: {len(feature_cols)} 个特征")

    # 3. 创建目标变量
    df["label"] = (df["close"].pct_change().shift(-1) > 0).astype(int)
    df = df.dropna()

    # 4. IC/IR筛选
    features_df = df[feature_cols]
    target = df["label"]

    filtered_features, ic_values, ir_values = filter_features_by_ic_ir(
        features_df, target, ic_threshold=0.05, ir_threshold=0.1
    )

    # 5. 数据分割
    train_size = int(len(df) * 0.7)
    X_train = filtered_features.iloc[:train_size].values
    X_test = filtered_features.iloc[train_size:].values
    y_train = target.iloc[:train_size].values
    y_test = target.iloc[train_size:].values

    # 6. 降维处理
    print(f"\n🔧 降维处理...")

    methods = ["autoencoder", "pca"]
    results = {}

    for method in methods:
        print(f"\n📊 测试 {method} 方法...")

        try:
            X_train_reduced, X_test_reduced, reduction_model, scaler = (
                apply_enhanced_dimensionality_reduction(
                    X_train, X_test, method=method, encoding_dim=8
                )
            )

            # 训练模型
            model = train_lightgbm_model(X_train_reduced, y_train, use_gpu=True)

            # 评估性能
            predictions = model.predict(X_test_reduced)
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = (predictions_binary == y_test).mean()

            # 特征解释
            explain_compressed_features(
                X_train, X_train_reduced, filtered_features.columns.tolist(), method
            )

            results[method] = {
                "accuracy": accuracy,
                "compression_ratio": X_train.shape[1] / X_train_reduced.shape[1],
                "method": method,
                "encoding_dim": 8,
            }

            print(f"✅ {method} 完成: 准确率 {accuracy:.4f}")

        except Exception as e:
            print(f"❌ {method} 失败: {e}")
            results[method] = {"error": str(e)}

    # 7. 特征类型分析
    feature_type_analysis = analyze_feature_types(filtered_features)

    # 8. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/feature_engineering_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total_features": len(feature_cols),
                "filtered_features": len(filtered_features.columns),
                "feature_type_analysis": feature_type_analysis,
                "dimensionality_results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n💾 结果保存到: {results_file}")
    print(f"\n🎯 特征工程完成！")
    print(f"  1. 生成特征: {len(feature_cols)} 个")
    print(f"  2. 筛选后特征: {len(filtered_features.columns)} 个")
    print(f"  3. 降维方法对比: {len(results)} 种")


if __name__ == "__main__":
    main()
