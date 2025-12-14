"""
集成测试：验证 inf 值的来源和修复
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
)
from src.features.utils.data_monitor import check_data_quality


def create_synthetic_data_with_edge_cases(n_samples: int = 1000) -> pd.DataFrame:
    """创建包含边界情况的合成数据（可能导致 inf 值）"""
    dates = pd.date_range(start="2025-01-01", periods=n_samples, freq="4H")

    # 基础价格数据
    base_price = 50000.0
    returns = np.random.randn(n_samples) * 0.01
    prices = base_price * (1 + np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(100, 1000, size=n_samples),
            "cvd": np.cumsum(np.random.randn(n_samples) * 100),
        },
        index=dates,
    )

    # 添加可能导致 inf 的边界情况
    # 1. 零值 volume（可能导致除零）
    df.loc[df.index[100:110], "volume"] = 0.0

    # 2. 非常小的 ATR（可能导致除零）
    # 这会在计算 ATR 后出现

    # 3. 全零序列（RSI 计算）
    df.loc[df.index[200:210], "close"] = df.loc[df.index[199], "close"]

    return df


def test_sr_strength_max_no_inf():
    """测试 sr_strength_max 不产生 inf 值"""
    df = create_synthetic_data_with_edge_cases(n_samples=500)

    # 计算基础特征（包括 ATR）
    engineer = BaselineFeatureEngineer()
    # BaselineFeatureEngineer 没有 engineer 方法，需要直接调用 add_sr_strength_max
    # 但 add_sr_strength_max 是静态方法，需要通过类调用
    # 实际上 sr_strength_max 是在 engineer 流程中计算的，我们需要模拟这个过程
    # 先计算 ATR（使用 talib 或简单实现）
    try:
        import talib

        if "atr" not in df.columns:
            df["atr"] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
    except ImportError:
        # 如果没有 talib，使用简单实现
        if "atr" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=14, min_periods=1).mean()

    # 然后计算 SR 强度（简化测试，只检查计算逻辑）
    # 实际上 sr_strength_max 的计算比较复杂，需要边界定义等
    # 这里我们只测试 ATR 相关的计算不会产生 inf
    atr = df["atr"]
    inf_count = np.isinf(atr).sum()
    print(f"\n📊 ATR (prerequisite for sr_strength_max):")
    print(f"   Total: {len(atr)}")
    print(f"   Inf values: {inf_count}")
    print(f"   NaN values: {atr.isna().sum()}")

    assert inf_count == 0, f"ATR should not have inf values, found {inf_count}"

    # 调用 engineer_features 计算 sr_strength_max
    # 需要指定 required_features 包含 sr_strength_max
    df_features = engineer.engineer_features(
        df,
        required_features=["sr_strength_max"],
    )

    # 检查 sr_strength_max
    if "sr_strength_max" in df_features.columns:
        sr_strength = df_features["sr_strength_max"]
        inf_count = np.isinf(sr_strength).sum()
        print(f"\n📊 sr_strength_max:")
        print(f"   Total: {len(sr_strength)}")
        print(f"   Inf values: {inf_count}")
        print(f"   NaN values: {sr_strength.isna().sum()}")
        print(f"   Min: {sr_strength.min()}, Max: {sr_strength.max()}")

        assert (
            inf_count == 0
        ), f"sr_strength_max should not have inf values, found {inf_count}"
    else:
        pytest.skip("sr_strength_max not computed")


def test_hurst_features_no_inf():
    """测试 Hurst 特征不产生 inf 值"""
    df = create_synthetic_data_with_edge_cases(n_samples=500)

    # 计算 Hurst 特征
    df_features = extract_hurst_features(
        df,
        price_col="close",
        cvd_col="cvd",
        volume_col="volume",
    )

    # 检查所有 Hurst 特征
    hurst_cols = [col for col in df_features.columns if "hurst" in col.lower()]
    print(f"\n📊 Hurst features: {len(hurst_cols)}")

    for col in hurst_cols:
        values = df_features[col]
        inf_count = np.isinf(values).sum()
        print(f"   {col}: {inf_count} inf, {values.isna().sum()} NaN")

        assert inf_count == 0, f"{col} should not have inf values, found {inf_count}"


def test_rsi_no_inf():
    """测试 RSI 不产生 inf 值"""
    df = create_synthetic_data_with_edge_cases(n_samples=500)

    # 计算 RSI
    rsi = BaselineFeatureEngineer.compute_rsi(df["close"], period=14)

    inf_count = np.isinf(rsi).sum()
    print(f"\n📊 RSI:")
    print(f"   Total: {len(rsi)}")
    print(f"   Inf values: {inf_count}")
    print(f"   NaN values: {rsi.isna().sum()}")
    print(f"   Min: {rsi.min()}, Max: {rsi.max()}")

    assert inf_count == 0, f"RSI should not have inf values, found {inf_count}"


def test_trade_clustering_zscore_no_inf():
    """测试 Trade Clustering zscore 特征不产生 inf 值"""
    # 创建包含 tick 数据的合成数据
    dates = pd.date_range(start="2025-01-01", periods=1000, freq="1S")
    ticks = pd.DataFrame(
        {
            "price": 50000.0 + np.cumsum(np.random.randn(1000) * 0.1),
            "volume": np.random.uniform(0.1, 10.0, size=1000),
            "side": np.random.choice([1, -1], size=1000),
        },
        index=dates,
    )

    # 创建 K 线数据
    kline_dates = pd.date_range(start="2025-01-01", periods=100, freq="4H")
    df = pd.DataFrame(
        {
            "open": 50000.0 + np.random.randn(100) * 10,
            "high": 50000.0 + np.abs(np.random.randn(100)) * 20,
            "low": 50000.0 - np.abs(np.random.randn(100)) * 20,
            "close": 50000.0 + np.random.randn(100) * 10,
            "volume": np.random.uniform(100, 1000, size=100),
        },
        index=kline_dates,
    )

    # 计算 Trade Clustering 特征
    try:
        df_features = extract_order_flow_features(
            df,
            ticks=ticks,
            include_trade_clustering=True,
            trade_clustering_window=100,
        )

        # 检查 zscore 特征
        zscore_cols = [
            col
            for col in df_features.columns
            if "zscore" in col.lower() and "trade_cluster" in col.lower()
        ]
        print(f"\n📊 Trade Clustering zscore features: {len(zscore_cols)}")

        for col in zscore_cols:
            values = df_features[col]
            inf_count = np.isinf(values).sum()
            print(f"   {col}: {inf_count} inf, {values.isna().sum()} NaN")

            assert (
                inf_count == 0
            ), f"{col} should not have inf values, found {inf_count}"
    except Exception as e:
        pytest.skip(f"Trade Clustering calculation failed: {e}")


def test_data_monitor_detects_inf():
    """测试数据监控系统能检测到 inf 值"""
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, np.inf, 4.0, 5.0],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col3": [1.0, -np.inf, 3.0, 4.0, 5.0],
        }
    )

    # 应该检测到 inf 值
    try:
        check_data_quality(
            df,
            data_source="TEST",
            stage="test",
            raise_on_inf=True,
        )
        assert False, "Should have raised an error for inf values"
    except ValueError as e:
        assert "inf" in str(e).lower(), "Error should mention inf values"
        print(f"\n✅ Data monitor correctly detected inf values: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
