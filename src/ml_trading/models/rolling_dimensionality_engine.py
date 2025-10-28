"""
Rolling Dimensionality Reduction Engine
结合滚动训练和降维的协同优化系统

实现功能：
1. 季度数据滚动训练
2. 基于漂移的动态降维触发
3. 降维前后效果对比
4. 反馈闭环优化
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from .interpretable_factor_engine import InterpretableFactorEngine


class DriftDetector:
    """
    漂移检测器 - 基于特征重要性的漂移检测
    """

    def __init__(
        self,
        js_threshold: float = 0.3,
        overlap_threshold: int = 3,
        window_size: int = 5,
    ):
        """
        Args:
            js_threshold: Jensen-Shannon散度阈值
            overlap_threshold: Top特征重叠阈值
            window_size: 历史窗口大小
        """
        self.js_threshold = js_threshold
        self.overlap_threshold = overlap_threshold
        self.window_size = window_size
        self.importance_history = []

    def add_importance(self, importance_dict: Dict[str, float]):
        """添加新的特征重要性记录"""
        self.importance_history.append(importance_dict.copy())

        # 保持历史记录不超过window_size * 2
        if len(self.importance_history) > self.window_size * 2:
            self.importance_history = self.importance_history[-self.window_size * 2 :]

    def should_trigger_dimensionality_reduction(self) -> Tuple[bool, Dict[str, Any]]:
        """
        判断是否应该触发降维

        Returns:
            (should_trigger, trigger_info)
        """
        if len(self.importance_history) < self.window_size + 1:
            return False, {"reason": "insufficient_history"}

        # 获取最新和历史平均重要性
        latest = pd.Series(self.importance_history[-1])
        historical = pd.DataFrame(self.importance_history[-self.window_size - 1 : -1])
        historical_mean = historical.mean()

        # 确保索引对齐
        common_features = set(latest.index) & set(historical_mean.index)
        if len(common_features) < 10:  # 至少需要10个共同特征
            return False, {"reason": "insufficient_common_features"}

        latest_aligned = latest[common_features].fillna(0)
        historical_aligned = historical_mean[common_features].fillna(0)

        # 1. Jensen-Shannon散度检测
        js_div = jensenshannon(latest_aligned.values, historical_aligned.values)

        # 2. Top 5特征重叠检测
        top5_current = set(latest_aligned.nlargest(5).index)
        top5_historical = set(historical_aligned.nlargest(5).index)
        overlap = len(top5_current & top5_historical)

        # 3. 重要性变化检测
        importance_change = np.mean(
            np.abs(latest_aligned.values - historical_aligned.values)
        )

        trigger_info = {
            "js_divergence": js_div,
            "top5_overlap": overlap,
            "importance_change": importance_change,
            "latest_top5": list(top5_current),
            "historical_top5": list(top5_historical),
        }

        # 触发条件
        should_trigger = (
            js_div > self.js_threshold
            or overlap < self.overlap_threshold
            or importance_change > 0.1
        )

        return should_trigger, trigger_info


class FeatureEvaluator:
    """
    特征集评估器 - 评估特征集的质量
    """

    def __init__(self, validation_ratio: float = 0.2, min_improvement: float = 0.005):
        """
        Args:
            validation_ratio: 验证集比例
            min_improvement: 最小改进阈值
        """
        self.validation_ratio = validation_ratio
        self.min_improvement = min_improvement

    def evaluate_feature_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_type: str = "regression",
    ) -> Dict[str, float]:
        """
        评估特征集质量

        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
            task_type: 任务类型 ("regression" 或 "classification")

        Returns:
            评估指标字典
        """
        from sklearn.model_selection import train_test_split

        # 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_ratio, random_state=42
        )

        # 训练LightGBM模型
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            "objective": "regression" if task_type == "regression" else "binary",
            "metric": "l2" if task_type == "regression" else "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

        # 预测和评估
        y_pred = model.predict(X_val)

        if task_type == "regression":
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            return {
                "mse": mse,
                "r2": r2,
                "rmse": np.sqrt(mse),
                "best_iteration": model.best_iteration,
            }
        else:
            auc = roc_auc_score(y_val, y_pred)
            return {"auc": auc, "best_iteration": model.best_iteration}


class RollingDimensionalityEngine:
    """
    滚动降维引擎 - 结合滚动训练和降维的协同系统
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        drift_threshold: float = 0.3,
        min_improvement: float = 0.005,
        feature_evaluation_ratio: float = 0.2,
        model_save_dir: str = "models/rolling_dim",
        results_save_dir: str = "results/rolling_dim",
    ):
        """
        Args:
            encoding_dim: 降维后的维度
            drift_threshold: 漂移检测阈值
            min_improvement: 最小改进阈值
            feature_evaluation_ratio: 特征评估验证集比例
            model_save_dir: 模型保存目录
            results_save_dir: 结果保存目录
        """
        self.encoding_dim = encoding_dim
        self.drift_threshold = drift_threshold
        self.min_improvement = min_improvement
        self.feature_evaluation_ratio = feature_evaluation_ratio

        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(results_save_dir, exist_ok=True)
        self.model_save_dir = model_save_dir
        self.results_save_dir = results_save_dir

        # 初始化组件
        self.drift_detector = DriftDetector(js_threshold=drift_threshold)
        self.feature_evaluator = FeatureEvaluator(
            validation_ratio=feature_evaluation_ratio, min_improvement=min_improvement
        )
        self.dimensionality_engine = None

        # 状态跟踪
        self.current_features = None
        self.feature_history = []
        self.performance_history = []
        self.dimensionality_history = []

        print(f"🚀 RollingDimensionalityEngine initialized")
        print(f"   - Encoding dimension: {encoding_dim}")
        print(f"   - Drift threshold: {drift_threshold}")
        print(f"   - Min improvement: {min_improvement}")

    def load_quarterly_data(
        self, data_path: str, start_date: str, end_date: str, symbol: str = "ETH-USD"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        加载季度数据

        Args:
            data_path: 数据路径
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            symbol: 交易对符号

        Returns:
            (X, y, feature_names)
        """
        print(f"📂 Loading quarterly data: {start_date} to {end_date}")

        try:
            # 这里需要根据实际数据格式调整
            # 假设数据是CSV格式，包含timestamp, features, target列
            df = pd.read_csv(data_path)

            # 过滤日期范围
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            if df.empty:
                print(f"❌ No data found for date range {start_date} to {end_date}")
                return self._create_sample_data()

            # 提取特征和目标
            feature_columns = [
                col
                for col in df.columns
                if col not in ["timestamp", "target", "future_return"]
            ]
            X = df[feature_columns].values
            y = df.get(
                "future_return", df.get("target", np.random.randn(len(df)))
            ).values

            # 处理NaN值
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)

            print(f"✅ Loaded {len(df)} samples with {len(feature_columns)} features")
            return X, y, feature_columns

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self._create_sample_data()

    def _create_sample_data(
        self, n_samples: int = 1000, n_features: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """创建样本数据用于测试"""
        print(f"📊 Creating sample data: {n_samples} samples, {n_features} features")

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        # 创建有意义的特征名称
        feature_names = []
        categories = ["momentum", "volatility", "volume", "trend", "mean_reversion"]
        for i in range(n_features):
            category = categories[i % len(categories)]
            feature_names.append(f"{category}_{i+1}")

        # 创建目标变量（与某些特征相关）
        momentum_features = [
            i for i, name in enumerate(feature_names) if "momentum" in name
        ]
        volatility_features = [
            i for i, name in enumerate(feature_names) if "volatility" in name
        ]

        y = (
            X[:, momentum_features].mean(axis=1) * 0.3
            + X[:, volatility_features].mean(axis=1) * -0.2
            + np.random.randn(n_samples) * 0.1
        )

        return X, y, feature_names

    def run_rolling_training_with_dimensionality(
        self,
        train_data_path: str,
        test_data_path: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        symbol: str = "ETH-USD",
    ) -> Dict[str, Any]:
        """
        运行滚动训练与降维协同系统

        Args:
            train_data_path: 训练数据路径
            test_data_path: 测试数据路径
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            test_end: 测试结束日期
            symbol: 交易对符号

        Returns:
            训练结果字典
        """
        print("🚀 Starting Rolling Training with Dimensionality Reduction")
        print("=" * 70)

        # 1. 加载训练数据
        X_train, y_train, feature_names = self.load_quarterly_data(
            train_data_path, train_start, train_end, symbol
        )

        # 2. 加载测试数据
        X_test, y_test, _ = self.load_quarterly_data(
            test_data_path, test_start, test_end, symbol
        )

        # 3. 初始特征评估（降维前）
        print("\n📊 Evaluating original feature set...")
        original_performance = self.feature_evaluator.evaluate_feature_set(
            X_train, y_train, feature_names
        )
        print(f"Original performance: {original_performance}")

        # 4. 运行降维
        print("\n🧠 Running dimensionality reduction...")
        self.dimensionality_engine = InterpretableFactorEngine(
            encoding_dim=self.encoding_dim
        )
        self.dimensionality_engine.fit(X_train, y_train, feature_names)

        # 5. 获取降维后的特征
        compressed_features = self.dimensionality_engine.embeddings
        top_factors = self.dimensionality_engine.top_factors
        factor_weights = self.dimensionality_engine.factor_weights

        print(f"✅ Dimensionality reduction complete")
        print(
            f"   - Compressed {len(feature_names)} features to {self.encoding_dim} dimensions"
        )
        print(f"   - Selected {len(top_factors)} top factors")

        # 6. 评估降维后的性能
        print("\n📊 Evaluating compressed feature set...")
        compressed_performance = self.feature_evaluator.evaluate_feature_set(
            compressed_features, y_train, [f"dim_{i}" for i in range(self.encoding_dim)]
        )
        print(f"Compressed performance: {compressed_performance}")

        # 7. 性能对比
        performance_improvement = self._compare_performance(
            original_performance, compressed_performance
        )

        # 8. 在测试集上验证
        print("\n🔍 Testing on out-of-sample data...")
        test_predictions = self.dimensionality_engine.predict(X_test)
        test_performance = self._evaluate_test_performance(y_test, test_predictions)

        # 9. 保存结果
        results = {
            "training_period": f"{train_start} to {train_end}",
            "testing_period": f"{test_start} to {test_end}",
            "symbol": symbol,
            "original_features_count": len(feature_names),
            "compressed_dimensions": self.encoding_dim,
            "selected_factors_count": len(top_factors),
            "original_performance": original_performance,
            "compressed_performance": compressed_performance,
            "performance_improvement": performance_improvement,
            "test_performance": test_performance,
            "top_factors": top_factors.tolist(),
            "factor_weights": factor_weights.tolist(),
            "compression_ratio": len(feature_names) / self.encoding_dim,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_results(results)
        self._save_model()

        print("\n" + "=" * 70)
        print("🎉 Rolling Training with Dimensionality Reduction Complete!")
        print(f"📊 Compression ratio: {results['compression_ratio']:.1f}x")
        print(
            f"📈 Performance improvement: {performance_improvement['improvement']:.3f}"
        )
        print(f"🎯 Test performance: {test_performance['r2']:.3f}")

        return results

    def _compare_performance(
        self, original_perf: Dict[str, float], compressed_perf: Dict[str, float]
    ) -> Dict[str, Any]:
        """比较降维前后的性能"""

        # 主要指标比较
        if "r2" in original_perf and "r2" in compressed_perf:
            r2_improvement = compressed_perf["r2"] - original_perf["r2"]
            improvement_percent = (
                (r2_improvement / abs(original_perf["r2"])) * 100
                if original_perf["r2"] != 0
                else 0
            )

            return {
                "metric": "r2",
                "original": original_perf["r2"],
                "compressed": compressed_perf["r2"],
                "improvement": r2_improvement,
                "improvement_percent": improvement_percent,
                "is_improved": r2_improvement > self.min_improvement,
            }
        elif "auc" in original_perf and "auc" in compressed_perf:
            auc_improvement = compressed_perf["auc"] - original_perf["auc"]
            improvement_percent = (
                (auc_improvement / original_perf["auc"]) * 100
                if original_perf["auc"] != 0
                else 0
            )

            return {
                "metric": "auc",
                "original": original_perf["auc"],
                "compressed": compressed_perf["auc"],
                "improvement": auc_improvement,
                "improvement_percent": improvement_percent,
                "is_improved": auc_improvement > self.min_improvement,
            }
        else:
            return {"error": "No comparable metrics found"}

    def _evaluate_test_performance(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """评估测试集性能"""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {"mse": mse, "r2": r2, "rmse": rmse}

    def _save_results(self, results: Dict[str, Any]):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.results_save_dir, f"rolling_dim_results_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: {results_file}")

    def _save_model(self):
        """保存模型"""
        if self.dimensionality_engine is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(
                self.model_save_dir, f"rolling_dim_model_{timestamp}.pkl"
            )
            self.dimensionality_engine.save_model(model_file)
            print(f"💾 Model saved to: {model_file}")

    def run_drift_triggered_training(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        symbol: str = "ETH-USD",
        rolling_window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        运行基于漂移触发的训练

        Args:
            data_path: 数据路径
            start_date: 开始日期
            end_date: 结束日期
            symbol: 交易对符号
            rolling_window_days: 滚动窗口天数

        Returns:
            训练结果
        """
        print("🚀 Starting Drift-Triggered Training")
        print("=" * 50)

        # 加载完整数据
        X, y, feature_names = self.load_quarterly_data(
            data_path, start_date, end_date, symbol
        )

        # 按时间窗口分割数据
        window_size = len(X) // 4  # 假设数据是时间序列
        results = []

        for i in range(0, len(X) - window_size, window_size // 2):
            # 训练窗口
            X_train = X[i : i + window_size]
            y_train = y[i : i + window_size]

            # 测试窗口
            test_start = i + window_size
            test_end = min(test_start + window_size // 2, len(X))
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            if len(X_test) < 50:  # 测试集太小则跳过
                continue

            print(f"\n📊 Processing window {i//window_size + 1}")
            print(f"   Training: {len(X_train)} samples")
            print(f"   Testing: {len(X_test)} samples")

            # 运行降维和训练
            window_results = self.run_rolling_training_with_dimensionality(
                data_path, data_path, start_date, end_date, start_date, end_date, symbol
            )

            results.append(window_results)

        # 汇总结果
        summary = self._summarize_results(results)
        return summary

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总多个窗口的结果"""
        if not results:
            return {"error": "No results to summarize"}

        # 计算平均性能
        avg_original_r2 = np.mean(
            [r["original_performance"].get("r2", 0) for r in results]
        )
        avg_compressed_r2 = np.mean(
            [r["compressed_performance"].get("r2", 0) for r in results]
        )
        avg_test_r2 = np.mean([r["test_performance"].get("r2", 0) for r in results])

        # 计算平均压缩比
        avg_compression_ratio = np.mean([r["compression_ratio"] for r in results])

        summary = {
            "total_windows": len(results),
            "average_original_r2": avg_original_r2,
            "average_compressed_r2": avg_compressed_r2,
            "average_test_r2": avg_test_r2,
            "average_compression_ratio": avg_compression_ratio,
            "overall_improvement": avg_compressed_r2 - avg_original_r2,
            "results": results,
        }

        return summary


if __name__ == "__main__":
    # 示例用法
    engine = RollingDimensionalityEngine(
        encoding_dim=8, drift_threshold=0.3, min_improvement=0.005
    )

    # 运行季度数据训练
    results = engine.run_rolling_training_with_dimensionality(
        train_data_path="data/train_2024.csv",
        test_data_path="data/test_2025.csv",
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-03-31",
        symbol="ETH-USD",
    )

    print("🎉 Training complete!")
    print(f"Results: {results}")
