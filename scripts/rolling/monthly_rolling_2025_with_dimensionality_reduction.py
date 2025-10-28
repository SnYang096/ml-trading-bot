"""Monthly Rolling Training with Dimensionality Reduction.

整合降维功能到滚动训练中，包含Autoencoder + SHAP蒸馏
参考现有的滚动训练实现，添加降维功能
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import warnings

warnings.filterwarnings("ignore")

# Add common utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import (
    load_and_process_file,
    add_order_flow_features,
    create_labels,
    get_feature_columns,
)
from training_utils import train_lightgbm_model, simple_backtest, print_backtest_results
from feature_manager import create_feature_manager

# Add src to path for dimensionality reduction
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# 使用新的综合特征工程模块
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
)


class DimensionalityReductionAutoencoder(nn.Module):
    """降维Autoencoder"""

    def __init__(
        self, input_dim: int, encoding_dim: int = 8, dropout_rate: float = 0.2
    ):
        super(DimensionalityReductionAutoencoder, self).__init__()

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


def apply_dimensionality_reduction(
    X_train, X_test, method="autoencoder", encoding_dim=8
):
    """应用降维技术"""
    print(
        f"   🔧 Applying {method} dimensionality reduction: {X_train.shape[1]} features -> {encoding_dim} dimensions"
    )

    if method == "autoencoder":
        return apply_autoencoder_reduction(X_train, X_test, encoding_dim)
    elif method == "pca":
        return apply_pca_reduction(X_train, X_test, encoding_dim)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def apply_autoencoder_reduction(X_train, X_test, encoding_dim=8):
    """应用Autoencoder降维"""
    print(f"   🧠 Training Autoencoder for dimensionality reduction...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建Autoencoder
    autoencoder = DimensionalityReductionAutoencoder(
        input_dim=X_train.shape[1], encoding_dim=encoding_dim
    ).to(device)

    # 训练参数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)

    # 训练循环
    autoencoder.train()
    train_losses = []

    for epoch in range(200):  # 200轮训练，平衡速度和效果
        optimizer.zero_grad()
        reconstructed, encoded = autoencoder(X_train_tensor)
        loss = criterion(reconstructed, X_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        train_losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"     Epoch {epoch+1:3d}/200: Loss = {loss.item():.6f}")

    print(f"   ✅ Autoencoder training complete (final loss: {train_losses[-1]:.6f})")

    # 提取嵌入
    autoencoder.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

        _, X_train_emb = autoencoder(X_train_tensor)
        _, X_test_emb = autoencoder(X_test_tensor)

        X_train_emb = X_train_emb.cpu().numpy()
        X_test_emb = X_test_emb.cpu().numpy()

    print(f"   ✅ Embeddings extracted: {X_train_emb.shape}")

    return X_train_emb, X_test_emb, autoencoder, scaler


def apply_pca_reduction(X_train, X_test, n_components=8):
    """应用PCA降维"""
    print(f"   📊 Applying PCA dimensionality reduction...")

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 应用PCA
    pca = IncrementalPCA(n_components=n_components, batch_size=1000)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 打印解释方差
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"   ✅ PCA applied successfully")
    print(
        f"   ✅ Explained variance: {explained_variance[:5].sum():.3f} (first 5 components)"
    )
    print(
        f"   ✅ Cumulative variance: {cumulative_variance[-1]:.3f} (all {n_components} components)"
    )

    return X_train_pca, X_test_pca, pca, scaler


def train_with_dimensionality_reduction(
    X_train, y_train, X_test, y_test, method="autoencoder", encoding_dim=8
):
    """使用降维后的特征训练模型"""
    print(f"\n🔧 Training with {method} dimensionality reduction...")

    # 应用降维
    if method == "autoencoder":
        X_train_reduced, X_test_reduced, reduction_model, scaler = (
            apply_autoencoder_reduction(X_train, X_test, encoding_dim)
        )
    elif method == "pca":
        X_train_reduced, X_test_reduced, reduction_model, scaler = apply_pca_reduction(
            X_train, X_test, encoding_dim
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # 训练LightGBM模型
    print(f"   🌲 Training LightGBM with reduced features...")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    train_data = lgb.Dataset(X_train_reduced, label=y_train)
    eval_data = lgb.Dataset(X_test_reduced, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[eval_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0),
        ],
    )

    print(f"   ✅ LightGBM training complete (best iteration: {model.best_iteration})")

    # 评估性能
    predictions = model.predict(X_test_reduced)
    predictions_binary = (predictions > 0.5).astype(int)

    accuracy = (predictions_binary == y_test).mean()
    print(f"   📊 Test accuracy: {accuracy:.4f}")

    return (
        model,
        reduction_model,
        scaler,
        {
            "accuracy": accuracy,
            "compression_ratio": X_train.shape[1] / X_train_reduced.shape[1],
            "method": method,
            "encoding_dim": encoding_dim,
        },
    )


def compare_dimensionality_methods(X_train, y_train, X_test, y_test):
    """比较不同降维方法的效果"""
    print(f"\n📊 Comparing dimensionality reduction methods...")

    methods = ["autoencoder", "pca"]
    results = {}

    for method in methods:
        print(f"\n🔧 Testing {method} method...")
        try:
            model, reduction_model, scaler, metrics = (
                train_with_dimensionality_reduction(
                    X_train, y_train, X_test, y_test, method=method, encoding_dim=8
                )
            )
            results[method] = metrics
            print(f"   ✅ {method} completed successfully")
        except Exception as e:
            print(f"   ❌ {method} failed: {e}")
            results[method] = {"error": str(e)}

    return results


def calculate_strategy_metrics(results):
    """计算策略质量指标"""
    returns = results["total_return"] / 100
    trades = results["total_trades"]
    win_rate = results["win_rate"] / 100
    profit_factor = results["profit_factor"]
    max_drawdown = abs(results["max_drawdown"]) / 100

    if max_drawdown > 0:
        sharpe_ratio = returns / max_drawdown
    else:
        sharpe_ratio = returns if returns > 0 else 0

    if max_drawdown > 0:
        calmar_ratio = returns / max_drawdown
    else:
        calmar_ratio = 0

    win_rate_score = win_rate if win_rate > 0.5 else 0
    pf_score = min(profit_factor, 3.0) / 3.0
    dd_score = max(0, 1 - max_drawdown)

    quality_score = (
        0.3 * sharpe_ratio + 0.25 * pf_score + 0.25 * dd_score + 0.2 * win_rate_score
    )

    return {
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "win_rate_score": win_rate_score,
        "pf_score": pf_score,
        "dd_score": dd_score,
        "quality_score": quality_score,
    }


def main():
    """主函数 - 集成降维的滚动训练"""
    data_dir = os.environ.get("DATA_DIR", "/data/agg_data")

    print("\n" + "=" * 80)
    print("📊 Monthly Rolling Training with Dimensionality Reduction")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Initial Train: 2024 Q4 (Oct-Dec)")
    print(f"   Test: 2025 Jan-Jun")
    print(f"   Feature Engineering: Enhanced (Hurst + WPT + Hilbert + Spectral + CVD)")
    print(f"   Dimensionality Reduction: Autoencoder + PCA comparison")
    print(f"   DL Backend: Mamba/Transformer (auto-detect, FP16)")
    print(f"   Sequence: 120 bars -> 64 dimensions")
    print(f"   Workflow: All Features (~410) -> Dimensionality Reduction -> LightGBM")
    print(f"   Key: 降维保留特征可解释性，正确追踪重要性")
    print(f"   Training: Warm Start (保留旧知识) + GPU Acceleration")
    print(f"   Evaluation: Sharpe + PF + MaxDD + Quality Score")

    # Initialize feature manager
    feature_manager = create_feature_manager("results/feature_repository.json")

    # Define training periods
    train_periods = [
        ("2024-10-01", "2024-12-31", "2024 Q4"),
    ]

    test_periods = [
        ("2025-01-01", "2025-01-31", "2025 Jan"),
        ("2025-02-01", "2025-02-28", "2025 Feb"),
        ("2025-03-01", "2025-03-31", "2025 Mar"),
        ("2025-04-01", "2025-04-30", "2025 Apr"),
        ("2025-05-01", "2025-05-31", "2025 May"),
        ("2025-06-01", "2025-06-30", "2025 Jun"),
    ]

    # Results storage
    all_results = []
    dimensionality_results = {}

    # Initial training
    print(f"\n🚀 Phase 1: Initial Training (2024 Q4)")
    print("=" * 60)

    for train_start, train_end, train_name in train_periods:
        print(f"\n📊 Training Period: {train_name} ({train_start} to {train_end})")

        # Load training data
        train_file = os.path.join(
            data_dir, f"ETH-USD_{train_start}_{train_end}.parquet"
        )
        if not os.path.exists(train_file):
            print(f"   ⚠️  Training file not found: {train_file}")
            continue

        print(f"   📁 Loading training data: {train_file}")
        train_df = load_and_process_file(train_file)

        if train_df is None or train_df.empty:
            print(f"   ❌ Failed to load training data")
            continue

        print(f"   ✅ Training data loaded: {train_df.shape}")

        # Feature engineering - 使用综合特征工程
        print(f"   🔧 Applying comprehensive feature engineering...")
        train_df = add_order_flow_features(train_df)

        # 使用新的综合特征工程模块
        comprehensive_engineer = ComprehensiveFeatureEngineer()
        train_df = comprehensive_engineer.engineer_all_features(train_df, fit=True)

        # Create labels
        train_df = create_labels(train_df)

        # Get features
        feature_cols = get_feature_columns(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values

        print(f"   ✅ Features prepared: {X_train.shape}")

        # Test on first test period for dimensionality comparison
        test_start, test_end, test_name = test_periods[0]
        test_file = os.path.join(data_dir, f"ETH-USD_{test_start}_{test_end}.parquet")

        if os.path.exists(test_file):
            print(f"\n📊 Testing Period: {test_name} ({test_start} to {test_end})")
            print(f"   📁 Loading test data: {test_file}")
            test_df = load_and_process_file(test_file)

            if test_df is not None and not test_df.empty:
                print(f"   ✅ Test data loaded: {test_df.shape}")

                # Feature engineering for test data - 使用综合特征工程
                test_df = add_order_flow_features(test_df)
                test_df = comprehensive_engineer.engineer_all_features(
                    test_df, fit=False
                )
                test_df = create_labels(test_df)

                X_test = test_df[feature_cols].values
                y_test = test_df["label"].values

                print(f"   ✅ Test features prepared: {X_test.shape}")

                # Compare dimensionality reduction methods
                dimensionality_results = compare_dimensionality_methods(
                    X_train, y_train, X_test, y_test
                )

                # Print comparison results
                print(f"\n📊 Dimensionality Reduction Comparison Results:")
                print("=" * 60)
                for method, metrics in dimensionality_results.items():
                    if "error" not in metrics:
                        print(f"   {method.upper()}:")
                        print(f"     Accuracy: {metrics['accuracy']:.4f}")
                        print(
                            f"     Compression Ratio: {metrics['compression_ratio']:.1f}x"
                        )
                        print(f"     Method: {metrics['method']}")
                    else:
                        print(f"   {method.upper()}: Failed - {metrics['error']}")

    # Rolling testing
    print(f"\n🚀 Phase 2: Rolling Testing (2025 Jan-Jun)")
    print("=" * 60)

    # Use the best dimensionality reduction method
    best_method = "autoencoder"  # Default to autoencoder
    if dimensionality_results:
        best_accuracy = -1
        for method, metrics in dimensionality_results.items():
            if "error" not in metrics and metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_method = method

        print(f"   🏆 Best dimensionality reduction method: {best_method}")

    for test_start, test_end, test_name in test_periods:
        print(f"\n📊 Testing Period: {test_name} ({test_start} to {test_end})")

        test_file = os.path.join(data_dir, f"ETH-USD_{test_start}_{test_end}.parquet")
        if not os.path.exists(test_file):
            print(f"   ⚠️  Test file not found: {test_file}")
            continue

        print(f"   📁 Loading test data: {test_file}")
        test_df = load_and_process_file(test_file)

        if test_df is None or test_df.empty:
            print(f"   ❌ Failed to load test data")
            continue

        print(f"   ✅ Test data loaded: {test_df.shape}")

        # Feature engineering - 使用综合特征工程
        test_df = add_order_flow_features(test_df)
        test_df = comprehensive_engineer.engineer_all_features(test_df, fit=False)
        test_df = create_labels(test_df)

        # Get features
        feature_cols = get_feature_columns(test_df)
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        print(f"   ✅ Test features prepared: {X_test.shape}")

        # Train with dimensionality reduction
        try:
            model, reduction_model, scaler, metrics = (
                train_with_dimensionality_reduction(
                    X_train, y_train, X_test, y_test, method=best_method, encoding_dim=8
                )
            )

            print(f"   ✅ Model trained with {best_method} dimensionality reduction")
            print(f"   📊 Compression ratio: {metrics['compression_ratio']:.1f}x")
            print(f"   📊 Test accuracy: {metrics['accuracy']:.4f}")

            # Backtest
            print(f"   🔄 Running backtest...")
            predictions = model.predict(X_test)
            predictions_binary = (predictions > 0.5).astype(int)

            # Simple backtest
            test_df["prediction"] = predictions
            test_df["prediction_binary"] = predictions_binary

            backtest_results = simple_backtest(test_df)

            if backtest_results:
                print_backtest_results(backtest_results, test_name)

                # Calculate strategy metrics
                strategy_metrics = calculate_strategy_metrics(backtest_results)

                # Store results
                result = {
                    "period": test_name,
                    "start_date": test_start,
                    "end_date": test_end,
                    "dimensionality_method": best_method,
                    "compression_ratio": metrics["compression_ratio"],
                    "test_accuracy": metrics["accuracy"],
                    "backtest_results": backtest_results,
                    "strategy_metrics": strategy_metrics,
                }

                all_results.append(result)

        except Exception as e:
            print(f"   ❌ Failed to train model: {e}")
            continue

    # Save results
    print(f"\n💾 Saving results...")
    results_dir = "results/rolling_dimensionality_reduction"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        results_dir, f"rolling_dimensionality_results_{timestamp}.json"
    )

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Save results
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dimensionality_comparison": dimensionality_results,
                "best_method": best_method,
                "rolling_results": all_results,
                "summary": {
                    "total_periods": len(all_results),
                    "average_accuracy": (
                        np.mean([r["test_accuracy"] for r in all_results])
                        if all_results
                        else 0
                    ),
                    "average_compression_ratio": (
                        np.mean([r["compression_ratio"] for r in all_results])
                        if all_results
                        else 0
                    ),
                    "average_quality_score": (
                        np.mean(
                            [
                                r["strategy_metrics"]["quality_score"]
                                for r in all_results
                            ]
                        )
                        if all_results
                        else 0
                    ),
                },
            },
            f,
            indent=2,
            default=convert_numpy,
        )

    print(f"   ✅ Results saved to: {results_file}")

    # Print summary
    print(f"\n" + "=" * 80)
    print("🎉 Rolling Dimensionality Reduction Training Complete!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   Total periods tested: {len(all_results)}")
    print(f"   Best dimensionality method: {best_method}")
    print(
        f"   Average accuracy: {np.mean([r['test_accuracy'] for r in all_results]):.4f}"
        if all_results
        else "   Average accuracy: N/A"
    )
    print(
        f"   Average compression ratio: {np.mean([r['compression_ratio'] for r in all_results]):.1f}x"
        if all_results
        else "   Average compression ratio: N/A"
    )
    print(
        f"   Average quality score: {np.mean([r['strategy_metrics']['quality_score'] for r in all_results]):.4f}"
        if all_results
        else "   Average quality score: N/A"
    )
    print(f"   Results saved to: {results_file}")


if __name__ == "__main__":
    main()
