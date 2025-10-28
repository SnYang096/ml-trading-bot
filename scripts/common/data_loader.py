#!/usr/bin/env python3
"""
统一的数据加载工具
整合所有降维训练中的数据加载功能
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class UnifiedDataLoader:
    """统一的数据加载器"""

    def __init__(self, data_path: str = None):
        """
        初始化数据加载器

        Args:
            data_path: 数据路径
        """
        self.data_path = data_path
        self.feature_engineer = None

    def load_real_data(
        self,
        symbol: str = "ETH-USD",
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        加载真实市场数据

        Args:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            (X, y, feature_names): 特征矩阵、目标变量、特征名称
        """
        try:
            from ml_trading.data_tools.data_loader import MarketDataLoader
            from ml_trading.data_tools.comprehensive_feature_engineering import (
                ComprehensiveFeatureEngineer,
            )

            print(f"📊 Loading real market data for {symbol}...")

            # 加载数据
            loader = MarketDataLoader(self.data_path)
            df = loader.load_data()

            if df is None or df.empty:
                print("⚠️ No real data found, generating sample data...")
                return self._generate_sample_data()

            # 重采样
            df = loader.resample_data("5T")

            # 特征工程
            self.feature_engineer = ComprehensiveFeatureEngineer()
            df_features = self.feature_engineer.engineer_all_features(df, fit=True)

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
            return self._generate_sample_data()

    def _generate_sample_data(
        self, n_samples: int = 10000, n_factors: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        生成样本数据

        Args:
            n_samples: 样本数量
            n_factors: 因子数量

        Returns:
            (X, y, feature_names): 特征矩阵、目标变量、特征名称
        """
        print(f"📊 Generating sample data: {n_samples} samples, {n_factors} features")

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
        momentum_factors = [
            i for i, name in enumerate(factor_names) if "momentum" in name
        ]
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

        print(f"✅ Sample data generated: {X.shape}, {y.shape}")
        return X, y, factor_names

    def load_quarterly_data(
        self, symbol: str = "ETH-USD", year: int = 2024
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, list]]:
        """
        加载季度数据

        Args:
            symbol: 交易对符号
            year: 年份

        Returns:
            quarterly_data: 季度数据字典
        """
        quarters = {
            f"{year}_Q1": {"start": f"{year}-01-01", "end": f"{year}-03-31"},
            f"{year}_Q2": {"start": f"{year}-04-01", "end": f"{year}-06-30"},
            f"{year}_Q3": {"start": f"{year}-07-01", "end": f"{year}-09-30"},
            f"{year}_Q4": {"start": f"{year}-10-01", "end": f"{year}-12-31"},
        }

        quarterly_data = {}

        for quarter_name, dates in quarters.items():
            print(f"📊 Loading {quarter_name} data...")

            # 这里可以根据需要实现季度数据加载
            # 现在使用样本数据
            X, y, feature_names = self._generate_sample_data(
                n_samples=2500, n_factors=100
            )

            quarterly_data[quarter_name] = (X, y, feature_names)
            print(f"✅ {quarter_name} loaded: {X.shape}")

        return quarterly_data

    def create_time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        创建时间序列分割

        Args:
            X: 特征矩阵
            y: 目标变量
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            splits: 数据分割字典
        """
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]

        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def get_data_info(
        self, X: np.ndarray, y: np.ndarray, feature_names: list
    ) -> Dict[str, Any]:
        """
        获取数据信息

        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称

        Returns:
            info: 数据信息字典
        """
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "target_stats": {
                "mean": np.mean(y),
                "std": np.std(y),
                "min": np.min(y),
                "max": np.max(y),
            },
            "feature_stats": {"mean": np.mean(X, axis=0), "std": np.std(X, axis=0)},
        }
