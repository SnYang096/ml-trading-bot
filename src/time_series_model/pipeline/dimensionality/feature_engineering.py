"""Feature engineering and dimensionality analysis utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from time_series_model.models.autoencoder import AutoencoderTrainer, UnifiedAutoencoder
from time_series_model.utils.training import train_lightgbm_model


def calculate_ic_ir(
    features: pd.DataFrame,
    target: pd.Series,
    window: int = 252,
) -> Tuple[pd.Series, pd.Series]:
    print("📊 计算IC/IR指标...")

    ic_values = []
    ir_values = []

    for col in features.columns:
        try:
            ic_rolling = features[col].rolling(window).corr(target)
            ic_mean = ic_rolling.mean()
            ic_std = ic_rolling.std()

            ic_values.append(ic_mean)

            ir = ic_mean / ic_std if ic_std and ic_std > 0 else 0
            ir_values.append(ir)
        except Exception:  # noqa: BLE001
            ic_values.append(0)
            ir_values.append(0)

    ic_series = pd.Series(ic_values, index=features.columns)
    ir_series = pd.Series(ir_values, index=features.columns)

    return ic_series, ir_series


def filter_features_by_ic_ir(
    features: pd.DataFrame,
    target: pd.Series,
    ic_threshold: float = 0.05,
    ir_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    print("🔍 IC/IR特征筛选...")

    ic_values, ir_values = calculate_ic_ir(features, target)

    good_features = (ic_values.abs() > ic_threshold) & (ir_values.abs()
                                                        > ir_threshold)

    print(f"  原始特征数量: {len(features.columns)}")
    print(f"  IC>0.05的特征: {(ic_values.abs() > ic_threshold).sum()}")
    print(f"  IR>0.1的特征: {(ir_values.abs() > ir_threshold).sum()}")
    print(f"  筛选后特征数量: {good_features.sum()}")

    top_ic = ic_values.abs().nlargest(10)
    print("\n  Top 10 IC特征:")
    for i, (feature, ic_score) in enumerate(top_ic.items(), 1):
        print(f"    {i:2d}. {feature:<30} IC: {ic_score:.4f}")

    return features.loc[:, good_features], ic_values, ir_values


def apply_autoencoder_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    encoding_dim: int = 8,
):
    print("🧠 训练Autoencoder...")

    autoencoder = UnifiedAutoencoder(
        input_dim=X_train.shape[1],
        encoding_dim=encoding_dim,
        architecture="deep",
    )

    trainer = AutoencoderTrainer(autoencoder, device="auto")
    trainer.train(X_train, epochs=300, verbose=True)

    X_train_emb = trainer.transform(X_train)
    X_test_emb = trainer.transform(X_test)

    print("✅ Autoencoder训练完成")
    return X_train_emb, X_test_emb, autoencoder, None


def apply_pca_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 8,
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("📊 应用PCA降维...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_variance = pca.explained_variance_ratio_
    print(f"  解释方差: {explained_variance.sum():.3f}")

    return X_train_pca, X_test_pca, pca, scaler


def apply_enhanced_dimensionality_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "autoencoder",
    encoding_dim: int = 8,
):
    print(f"🔧 应用{method}降维: {X_train.shape[1]} -> {encoding_dim}")

    if method == "autoencoder":
        return apply_autoencoder_reduction(X_train, X_test, encoding_dim)
    if method == "pca":
        return apply_pca_reduction(X_train, X_test, encoding_dim)
    raise ValueError(f"Unknown method: {method}")


def explain_compressed_features(
    original_features: np.ndarray,
    compressed_features: np.ndarray,
    feature_names: list,
    method: str = "autoencoder",
) -> None:
    print("🔍 分析降维后特征...")

    correlations = []
    for i in range(compressed_features.shape[1]):
        corr_with_compressed = []
        for j in range(original_features.shape[1]):
            corr = np.corrcoef(original_features[:, j],
                               compressed_features[:, i])[0, 1]
            corr_with_compressed.append(corr)
        correlations.append(corr_with_compressed)

    print("  降维后特征解释:")
    for i in range(compressed_features.shape[1]):
        corr_values = correlations[i]
        top_indices = np.argsort(np.abs(corr_values))[-5:][::-1]

        print(f"    压缩特征 {i+1}:")
        for j, idx in enumerate(top_indices):
            if idx < len(feature_names):
                print(
                    f"      {j+1}. {feature_names[idx]:<30} 相关性: {corr_values[idx]:.4f}"
                )


def analyze_feature_types(filtered_features: pd.DataFrame) -> Dict[str, int]:
    print("\n📊 筛选后特征类型分析:")

    hurst_count = sum(1 for col in filtered_features.columns if "hurst" in col)
    wpt_count = sum(1 for col in filtered_features.columns if "wpt_" in col)
    hilbert_count = sum(1 for col in filtered_features.columns
                        if "hilbert" in col)
    spectral_count = sum(1 for col in filtered_features.columns
                         if "spectral" in col)
    order_flow_count = sum(1 for col in filtered_features.columns if any(
        x in col for x in ["cvd", "ofi", "tbr", "order_flow"]))

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


def run_feature_engineering() -> Dict[str, Dict[str, float]]:
    print("🚀 特征工程和降维分析")
    print("=" * 60)

    dates = pd.date_range("2024-01-01", periods=2000, freq="5T")
    sample_data = pd.DataFrame({
        "timestamp":
        dates,
        "open":
        np.random.randn(2000).cumsum() + 100,
        "high":
        np.random.randn(2000).cumsum() + 105,
        "low":
        np.random.randn(2000).cumsum() + 95,
        "close":
        np.random.randn(2000).cumsum() + 100,
        "volume":
        np.random.randint(1000, 10000, 2000),
        "cvd":
        np.random.randn(2000).cumsum(),
        "taker_buy_ratio":
        np.random.uniform(0.3, 0.7, 2000),
    })

    from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer

    print("🔧 综合特征工程...")
    comprehensive_engineer = ComprehensiveFeatureEngineer()
    df = comprehensive_engineer.engineer_all_features(sample_data, fit=True)

    feature_cols = [
        col for col in df.columns
        if col not in ["timestamp", "open", "high", "low", "close", "volume"]
    ]

    print(f"✅ 特征工程完成: {len(feature_cols)} 个特征")

    df["label"] = (df["close"].pct_change().shift(-1) > 0).astype(int)
    df = df.dropna()

    features_df = df[feature_cols]
    target = df["label"]

    filtered_features, ic_values, ir_values = filter_features_by_ic_ir(
        features_df,
        target,
        ic_threshold=0.05,
        ir_threshold=0.1,
    )

    train_size = int(len(df) * 0.7)
    X_train = filtered_features.iloc[:train_size].values
    X_test = filtered_features.iloc[train_size:].values
    y_train = target.iloc[:train_size].values
    y_test = target.iloc[train_size:].values

    print("\n🔧 降维处理...")

    methods = ["autoencoder", "pca"]
    results = {}

    for method in methods:
        print(f"\n📊 测试 {method} 方法...")

        try:
            X_train_reduced, X_test_reduced, reduction_model, scaler = (
                apply_enhanced_dimensionality_reduction(
                    X_train,
                    X_test,
                    method=method,
                    encoding_dim=8,
                ))

            model = train_lightgbm_model(X_train_reduced,
                                         y_train,
                                         use_gpu=True)

            predictions = model.predict(X_test_reduced)
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = float((predictions_binary == y_test).mean())

            explain_compressed_features(
                X_train,
                X_train_reduced,
                filtered_features.columns.tolist(),
                method,
            )

            results[method] = {
                "accuracy":
                accuracy,
                "compression_ratio":
                float(X_train.shape[1] / X_train_reduced.shape[1]),
                "method":
                method,
                "encoding_dim":
                8,
            }

            print(f"✅ {method} 完成: 准确率 {accuracy:.4f}")

        except Exception as exc:  # noqa: BLE001
            print(f"❌ {method} 失败: {exc}")
            results[method] = {"error": str(exc)}

    feature_type_analysis = analyze_feature_types(filtered_features)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/feature_engineering_{timestamp}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

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
    print("\n🎯 特征工程完成！")
    print(f"  1. 生成特征: {len(feature_cols)} 个")
    print(f"  2. 筛选后特征: {len(filtered_features.columns)} 个")
    print(f"  3. 降维方法对比: {len(results)} 种")

    return results


def main() -> None:
    run_feature_engineering()


if __name__ == "__main__":
    main()
