"""统一的归一化解决方案.

提供统一的归一化接口，解决所有特征工程文件的归一化问题：
1. 前世偏差问题
2. 归一化方法不一致
3. 特征分组策略不统一
4. 归一化参数管理不一致
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import warnings
from abc import ABC, abstractmethod


class NormalizationStrategy(ABC):
    """归一化策略基类."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "NormalizationStrategy":
        """拟合归一化参数."""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用归一化."""
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用归一化."""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """获取归一化参数."""
        pass

    @abstractmethod
    def set_params(self, params: Dict) -> "NormalizationStrategy":
        """设置归一化参数."""
        pass


class GlobalNormalization(NormalizationStrategy):
    """全局归一化策略."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.params = {}

    def fit(self, data: np.ndarray) -> "GlobalNormalization":
        """拟合全局归一化参数."""
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        self.scaler.fit(data)
        self.params = {
            "scaler_type": self.scaler_type,
            "mean": getattr(self.scaler, "mean_", None),
            "scale": getattr(self.scaler, "scale_", None),
            "min": getattr(self.scaler, "min_", None),
            "max": getattr(self.scaler, "max_", None),
            "center": getattr(self.scaler, "center_", None),
            "scale_": getattr(self.scaler, "scale_", None),
        }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用全局归一化."""
        if self.scaler is None:
            raise ValueError("Must fit the scaler first")
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用全局归一化."""
        return self.fit(data).transform(data)

    def get_params(self) -> Dict:
        """获取归一化参数."""
        return self.params.copy()

    def set_params(self, params: Dict) -> "GlobalNormalization":
        """设置归一化参数."""
        self.params = params.copy()
        self.scaler_type = params.get("scaler_type", "standard")

        # 重建scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()

        # 设置参数
        for key, value in params.items():
            if hasattr(self.scaler, key) and value is not None:
                setattr(self.scaler, key, value)

        return self


class RollingWindowNormalization(NormalizationStrategy):
    """滚动窗口归一化策略 - 使用Welford算法优化性能."""

    def __init__(self, window_size: int, scaler_type: str = "standard"):
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.params = {}

    def fit(self, data: np.ndarray) -> "RollingWindowNormalization":
        """拟合滚动窗口归一化参数."""
        self.params = {
            "window_size": self.window_size,
            "scaler_type": self.scaler_type,
            "data_shape": data.shape,
        }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用滚动窗口归一化 - 使用Welford算法优化."""
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be >= window_size ({self.window_size})"
            )

        normalized_data = np.zeros_like(data)

        if self.scaler_type == "standard":
            # 使用Welford算法优化标准归一化
            normalized_data = self._transform_standard_welford(data)
        elif self.scaler_type == "minmax":
            # 使用滑动窗口优化最小最大归一化
            normalized_data = self._transform_minmax_optimized(data)
        elif self.scaler_type == "robust":
            # 使用滑动窗口优化鲁棒归一化
            normalized_data = self._transform_robust_optimized(data)
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        return normalized_data

    def _transform_standard_welford(self, data: np.ndarray) -> np.ndarray:
        """使用Welford算法优化标准归一化."""
        normalized_data = np.zeros_like(data)

        # 初始化Welford算法变量
        mean = np.zeros_like(data[0])
        M2 = np.zeros_like(data[0])
        count = 0

        # 窗口缓冲区
        window_buffer = np.zeros((self.window_size, data.shape[1]))
        buffer_idx = 0

        for i in range(len(data)):
            # 移除旧值（如果窗口已满）
            if count >= self.window_size:
                old_val = window_buffer[buffer_idx]
                delta = old_val - mean
                mean -= delta / count
                M2 -= delta * (old_val - mean)
                count -= 1

            # 添加新值
            new_val = data[i]
            window_buffer[buffer_idx] = new_val
            buffer_idx = (buffer_idx + 1) % self.window_size

            count += 1
            delta = new_val - mean
            mean += delta / count
            M2 += delta * (new_val - mean)

            # 计算归一化值
            if count >= 2:  # 需要至少2个点计算标准差
                std = np.sqrt(M2 / count) + 1e-8
                normalized_data[i] = (new_val - mean) / std
            else:
                normalized_data[i] = new_val  # 第一个点保持原值

        return normalized_data

    def _transform_minmax_optimized(self, data: np.ndarray) -> np.ndarray:
        """优化最小最大归一化."""
        normalized_data = np.zeros_like(data)

        # 使用滑动窗口
        window_buffer = np.zeros((self.window_size, data.shape[1]))
        buffer_idx = 0

        for i in range(len(data)):
            # 更新窗口缓冲区
            window_buffer[buffer_idx] = data[i]
            buffer_idx = (buffer_idx + 1) % self.window_size

            # 计算当前窗口的统计量
            if i < self.window_size:
                current_window = window_buffer[: i + 1]
            else:
                current_window = window_buffer

            window_min = np.min(current_window, axis=0, keepdims=True)
            window_max = np.max(current_window, axis=0, keepdims=True)
            window_range = window_max - window_min + 1e-8

            # 归一化
            normalized_data[i] = (data[i] - window_min) / window_range

        return normalized_data

    def _transform_robust_optimized(self, data: np.ndarray) -> np.ndarray:
        """优化鲁棒归一化."""
        normalized_data = np.zeros_like(data)

        # 使用滑动窗口
        window_buffer = np.zeros((self.window_size, data.shape[1]))
        buffer_idx = 0

        for i in range(len(data)):
            # 更新窗口缓冲区
            window_buffer[buffer_idx] = data[i]
            buffer_idx = (buffer_idx + 1) % self.window_size

            # 计算当前窗口的统计量
            if i < self.window_size:
                current_window = window_buffer[: i + 1]
            else:
                current_window = window_buffer

            window_median = np.median(current_window, axis=0, keepdims=True)
            window_mad = (
                np.median(np.abs(current_window - window_median), axis=0, keepdims=True)
                + 1e-8
            )

            # 归一化
            normalized_data[i] = (data[i] - window_median) / window_mad

        return normalized_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用滚动窗口归一化."""
        return self.fit(data).transform(data)

    def get_params(self) -> Dict:
        """获取归一化参数."""
        return self.params.copy()

    def set_params(self, params: Dict) -> "RollingWindowNormalization":
        """设置归一化参数."""
        self.params = params.copy()
        self.window_size = params.get("window_size", 100)
        self.scaler_type = params.get("scaler_type", "standard")
        return self


class EMANormalization(NormalizationStrategy):
    """指数移动平均归一化策略."""

    def __init__(self, alpha: float = 0.01, scaler_type: str = "standard"):
        self.alpha = alpha
        self.scaler_type = scaler_type
        self.ema_mean = None
        self.ema_var = None
        self.params = {}

    def fit(self, data: np.ndarray) -> "EMANormalization":
        """拟合EMA归一化参数."""
        if len(data) > 0:
            self.ema_mean = np.mean(data[: min(100, len(data))], axis=0, keepdims=True)
            self.ema_var = np.var(data[: min(100, len(data))], axis=0, keepdims=True)

        self.params = {
            "alpha": self.alpha,
            "scaler_type": self.scaler_type,
            "ema_mean": self.ema_mean.copy() if self.ema_mean is not None else None,
            "ema_var": self.ema_var.copy() if self.ema_var is not None else None,
        }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用EMA归一化 - 使用无偏EMA方差估计."""
        if self.ema_mean is None or self.ema_var is None:
            raise ValueError("Must fit the scaler first")

        normalized_data = np.zeros_like(data)

        for i in range(len(data)):
            # 无偏EMA方差估计
            delta = data[i] - self.ema_mean
            self.ema_mean += self.alpha * delta
            self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * delta**2)
            ema_std = np.sqrt(self.ema_var) + 1e-8

            # 归一化
            normalized_data[i] = (data[i] - self.ema_mean) / ema_std

        return normalized_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用EMA归一化."""
        return self.fit(data).transform(data)

    def get_params(self) -> Dict:
        """获取归一化参数."""
        return self.params.copy()

    def set_params(self, params: Dict) -> "EMANormalization":
        """设置归一化参数."""
        self.params = params.copy()
        self.alpha = params.get("alpha", 0.01)
        self.scaler_type = params.get("scaler_type", "standard")
        self.ema_mean = params.get("ema_mean")
        self.ema_var = params.get("ema_var")
        return self


class AdaptiveNormalization(NormalizationStrategy):
    """自适应归一化策略."""

    def __init__(
        self,
        window_size: int,
        global_weight: float = 0.3,
        local_weight: float = 0.7,
        scaler_type: str = "standard",
    ):
        self.window_size = window_size
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.scaler_type = scaler_type
        self.global_mean = None
        self.global_std = None
        self.params = {}

    def fit(self, data: np.ndarray) -> "AdaptiveNormalization":
        """拟合自适应归一化参数."""
        self.global_mean = np.mean(data, axis=0, keepdims=True)
        self.global_std = np.std(data, axis=0, keepdims=True) + 1e-8

        self.params = {
            "window_size": self.window_size,
            "global_weight": self.global_weight,
            "local_weight": self.local_weight,
            "scaler_type": self.scaler_type,
            "global_mean": self.global_mean.copy(),
            "global_std": self.global_std.copy(),
        }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用自适应归一化."""
        if self.global_mean is None or self.global_std is None:
            raise ValueError("Must fit the scaler first")

        normalized_data = np.zeros_like(data)

        for i in range(len(data) - self.window_size + 1):
            window_data = data[i : i + self.window_size]
            window_mean = np.mean(window_data, axis=0, keepdims=True)
            window_std = np.std(window_data, axis=0, keepdims=True) + 1e-8

            # 计算全局和局部统计量的加权平均
            combined_mean = (
                self.global_weight * self.global_mean + self.local_weight * window_mean
            )
            combined_std = (
                self.global_weight * self.global_std + self.local_weight * window_std
            )

            normalized_data[i : i + self.window_size] = (
                window_data - combined_mean
            ) / combined_std

        return normalized_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用自适应归一化."""
        return self.fit(data).transform(data)

    def get_params(self) -> Dict:
        """获取归一化参数."""
        return self.params.copy()

    def set_params(self, params: Dict) -> "AdaptiveNormalization":
        """设置归一化参数."""
        self.params = params.copy()
        self.window_size = params.get("window_size", 100)
        self.global_weight = params.get("global_weight", 0.3)
        self.local_weight = params.get("local_weight", 0.7)
        self.scaler_type = params.get("scaler_type", "standard")
        self.global_mean = params.get("global_mean")
        self.global_std = params.get("global_std")
        return self


class UnifiedNormalizer:
    """统一的归一化器."""

    def __init__(
        self,
        strategy: Literal["global", "rolling", "ema", "adaptive"] = "rolling",
        scaler_type: str = "standard",
        window_size: int = 100,
        alpha: float = 0.01,
        global_weight: float = 0.3,
        local_weight: float = 0.7,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            strategy: 归一化策略
            scaler_type: 归一化器类型
            window_size: 窗口大小
            alpha: EMA平滑参数
            global_weight: 全局权重
            local_weight: 局部权重
            feature_groups: 特征分组
        """
        self.strategy = strategy
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.alpha = alpha
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.feature_groups = feature_groups or {}

        # 归一化器字典
        self.normalizers: Dict[str, NormalizationStrategy] = {}
        self.is_fitted = False

    def _create_normalizer(self, group_name: str = "default") -> NormalizationStrategy:
        """创建归一化器."""
        if self.strategy == "global":
            return GlobalNormalization(self.scaler_type)
        elif self.strategy == "rolling":
            return RollingWindowNormalization(self.window_size, self.scaler_type)
        elif self.strategy == "ema":
            return EMANormalization(self.alpha, self.scaler_type)
        elif self.strategy == "adaptive":
            return AdaptiveNormalization(
                self.window_size,
                self.global_weight,
                self.local_weight,
                self.scaler_type,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _group_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """自动分组特征."""
        if self.feature_groups:
            return self.feature_groups

        # 自动分组逻辑
        feature_cols = [
            col
            for col in data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        # 定义特征分组
        price_features = [
            col
            for col in feature_cols
            if any(x in col for x in ["sma", "ema", "wma", "tema", "kama", "sar"])
        ]
        ratio_features = [
            col
            for col in feature_cols
            if "ratio" in col or "position" in col or "normalized" in col
        ]
        volatility_features = [
            col
            for col in feature_cols
            if any(x in col for x in ["atr", "natr", "trange", "bb_", "volatility"])
        ]
        volume_features = [
            col
            for col in feature_cols
            if any(x in col for x in ["volume", "obv", "ad", "vpt", "cmf"])
        ]
        momentum_features = [
            col
            for col in feature_cols
            if any(
                x in col
                for x in ["rsi", "stoch", "willr", "mom", "roc", "cci", "ultosc", "tsi"]
            )
        ]
        index_features = [
            col
            for col in feature_cols
            if any(x in col for x in ["maxindex", "minindex", "max", "min"])
        ]

        # 其他特征
        other_features = [
            col
            for col in feature_cols
            if col
            not in price_features
            + ratio_features
            + volatility_features
            + volume_features
            + momentum_features
            + index_features
        ]

        groups = {
            "price": price_features,
            "ratio": ratio_features,
            "volatility": volatility_features,
            "volume": volume_features,
            "momentum": momentum_features,
            "index": index_features,
            "other": other_features,
        }

        return groups

    def fit(
        self, data: pd.DataFrame, timeframe: str = "default"
    ) -> "UnifiedNormalizer":
        """拟合归一化器."""
        df = data.copy()

        # 获取特征分组
        feature_groups = self._group_features(df)

        # 为每个特征组创建归一化器
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue

            # 创建归一化器
            normalizer = self._create_normalizer(group_name)

            # 准备数据
            X = df[group_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 特殊处理成交量特征
            if group_name == "volume":
                X = np.log1p(np.abs(X)) * np.sign(X)

            # 拟合归一化器
            normalizer.fit(X)

            # 保存归一化器
            self.normalizers[f"{timeframe}_{group_name}"] = normalizer

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, timeframe: str = "default") -> pd.DataFrame:
        """应用归一化."""
        if not self.is_fitted:
            raise ValueError("Must fit the normalizer first")

        df = data.copy()

        # 获取特征分组
        feature_groups = self._group_features(df)

        # 为每个特征组应用归一化
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue

            # 获取归一化器
            normalizer_key = f"{timeframe}_{group_name}"
            if normalizer_key not in self.normalizers:
                raise ValueError(f"No normalizer found for {normalizer_key}")

            normalizer = self.normalizers[normalizer_key]

            # 准备数据
            X = df[group_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 特殊处理成交量特征
            if group_name == "volume":
                X = np.log1p(np.abs(X)) * np.sign(X)

            # 应用归一化
            X_scaled = normalizer.transform(X)

            # 更新DataFrame
            for i, col in enumerate(group_features):
                df[col] = X_scaled[:, i]

        return df

    def fit_transform(
        self, data: pd.DataFrame, timeframe: str = "default"
    ) -> pd.DataFrame:
        """拟合并应用归一化."""
        return self.fit(data, timeframe).transform(data, timeframe)

    def save(self, filepath: str):
        """保存归一化器."""
        normalizer_data = {
            "strategy": self.strategy,
            "scaler_type": self.scaler_type,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "global_weight": self.global_weight,
            "local_weight": self.local_weight,
            "feature_groups": self.feature_groups,
            "normalizers": {k: v.get_params() for k, v in self.normalizers.items()},
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(normalizer_data, f)

        print(f"Unified normalizer saved to {filepath}")

    def load(self, filepath: str):
        """加载归一化器."""
        with open(filepath, "rb") as f:
            normalizer_data = pickle.load(f)

        self.strategy = normalizer_data["strategy"]
        self.scaler_type = normalizer_data["scaler_type"]
        self.window_size = normalizer_data["window_size"]
        self.alpha = normalizer_data["alpha"]
        self.global_weight = normalizer_data["global_weight"]
        self.local_weight = normalizer_data["local_weight"]
        self.feature_groups = normalizer_data["feature_groups"]
        self.is_fitted = normalizer_data["is_fitted"]

        # 重建归一化器
        self.normalizers = {}
        for key, params in normalizer_data["normalizers"].items():
            normalizer = self._create_normalizer()
            normalizer.set_params(params)
            self.normalizers[key] = normalizer

        print(f"Unified normalizer loaded from {filepath}")

    def get_feature_stats(self, data: pd.DataFrame, timeframe: str = "default") -> Dict:
        """获取特征统计信息."""
        if not self.is_fitted:
            raise ValueError("Must fit the normalizer first")

        df = data.copy()
        feature_groups = self._group_features(df)

        stats = {}
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue

            group_data = df[group_features].values
            group_data = np.nan_to_num(group_data, nan=0.0, posinf=0.0, neginf=0.0)

            stats[group_name] = {
                "count": len(group_features),
                "mean": np.mean(group_data, axis=0).tolist(),
                "std": np.std(group_data, axis=0).tolist(),
                "min": np.min(group_data, axis=0).tolist(),
                "max": np.max(group_data, axis=0).tolist(),
            }

        return stats


def create_unified_normalizer(
    strategy: str = "rolling", scaler_type: str = "standard", **kwargs
) -> UnifiedNormalizer:
    """创建统一归一化器的便利函数."""
    return UnifiedNormalizer(strategy=strategy, scaler_type=scaler_type, **kwargs)


def normalize_features_unified(
    data: pd.DataFrame,
    strategy: str = "rolling",
    scaler_type: str = "standard",
    timeframe: str = "default",
    fit: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, Optional[UnifiedNormalizer]]:
    """统一归一化特征的便利函数."""
    normalizer = UnifiedNormalizer(strategy=strategy, scaler_type=scaler_type, **kwargs)

    if fit:
        normalized_data = normalizer.fit_transform(data, timeframe)
        return normalized_data, normalizer
    else:
        normalized_data = normalizer.transform(data, timeframe)
        return normalized_data, None
