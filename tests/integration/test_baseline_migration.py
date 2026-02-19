"""
集成测试：验证 baseline 特征函数的正确性

测试策略：
1. 验证函数可以被正确调用
2. 验证 registry 和 get_compute_func 可以正确获取函数
"""

import pytest
import pandas as pd
import numpy as np
from typing import Callable

from src.features.time_series.baseline_features import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_bb_width_features,
    compute_volume_anomaly,
    add_basic_indicators,
)
from src.features.registry import (
    get_registry,
    get_compute_func,
    _ensure_features_registered,
    ensure_features_registered,
)

# 在模块加载时就注册所有特征
ensure_features_registered()


@pytest.fixture
def sample_ohlcv_data():
    """生成标准 OHLCV 测试数据"""
    np.random.seed(42)
    n = 200

    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class TestBaselineFeatureFunctions:
    """测试 baseline 特征函数"""

    def setup_method(self):
        """确保每个测试前特征已注册"""
        _ensure_features_registered(force=True)

    def test_compute_rsi(self, sample_ohlcv_data):
        """测试 compute_rsi"""
        result = compute_rsi(sample_ohlcv_data["close"], period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.notna().sum() > 0
        # RSI 应该在 0-100 之间
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_rsi_via_registry(self, sample_ohlcv_data):
        """通过 registry 获取 compute_rsi"""
        func = get_compute_func("compute_rsi")
        assert func is not None

        result = func(sample_ohlcv_data["close"], period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)

    def test_compute_macd(self, sample_ohlcv_data):
        """测试 compute_macd"""
        macd, signal, hist = compute_macd(
            sample_ohlcv_data["close"], fast=12, slow=26, signal=9
        )

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_ohlcv_data)

    def test_compute_bollinger_bands(self, sample_ohlcv_data):
        """测试 compute_bollinger_bands"""
        upper, middle, lower = compute_bollinger_bands(
            sample_ohlcv_data["close"], period=20, std_dev=2
        )

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        # upper >= middle >= lower
        valid_idx = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_compute_atr(self, sample_ohlcv_data):
        """测试 compute_atr"""
        result = compute_atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            period=14,
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        # ATR 应该为正
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_compute_bb_width_features(self, sample_ohlcv_data):
        """测试 compute_bb_width_features"""
        result = compute_bb_width_features(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        # 应该添加了 bb_width 相关的列
        assert any("bb_width" in col for col in result.columns)

    def test_compute_volume_anomaly(self, sample_ohlcv_data):
        """测试 compute_volume_anomaly"""
        result = compute_volume_anomaly(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert "volume_anomaly" in result.columns

    def test_add_basic_indicators(self, sample_ohlcv_data):
        """测试 add_basic_indicators"""
        result = add_basic_indicators(sample_ohlcv_data.copy())

        assert isinstance(result, pd.DataFrame)
        # 应该有比原始数据更多的列
        assert len(result.columns) > len(sample_ohlcv_data.columns)
        # 应该包含常见指标
        assert "rsi" in result.columns


class TestRegistryIntegration:
    """测试 Registry 集成"""

    def setup_method(self):
        """确保每个测试前特征已注册"""
        _ensure_features_registered(force=True)

    def test_compute_rsi_results_match(self, sample_ohlcv_data):
        """验证直接调用和 registry 调用结果一致"""
        # 直接调用
        result1 = compute_rsi(sample_ohlcv_data["close"], period=14)

        # 通过 registry 调用
        func = get_compute_func("compute_rsi")
        result2 = func(sample_ohlcv_data["close"], period=14)

        pd.testing.assert_series_equal(result1, result2)

    def test_compute_atr_results_match(self, sample_ohlcv_data):
        """验证 compute_atr 直接调用和 registry 调用结果一致"""
        # 直接调用
        result1 = compute_atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            period=14,
        )

        # 通过 registry 调用
        func = get_compute_func("compute_atr")
        result2 = func(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            period=14,
        )

        pd.testing.assert_series_equal(result1, result2)


class TestFeatureCount:
    """测试特征数量"""

    def setup_method(self):
        """确保每个测试前特征已注册"""
        _ensure_features_registered(force=True)

    def test_total_feature_count(self):
        """验证总特征数量"""
        registry = get_registry()

        # 应该至少有 200 个特征（删除了冗余的 wrapper 函数后）
        assert registry.count >= 200, f"Expected >= 200, got {registry.count}"

    def test_key_baseline_methods_in_registry(self):
        """验证关键 baseline 方法在 registry 中"""
        registry = get_registry()
        key_methods = [
            "compute_rsi",
            "compute_atr",
            "compute_macd",
            "compute_bollinger_bands",
            "add_basic_indicators",
        ]

        for method in key_methods:
            func = registry.get(method)
            assert func is not None, f"{method} not in registry"
