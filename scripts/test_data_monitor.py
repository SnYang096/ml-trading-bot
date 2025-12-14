#!/usr/bin/env python3
"""
测试数据监控系统

验证监控系统能够正确检测和报告 inf 值
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.utils.data_monitor import (
    check_data_quality,
    check_source_data_quality,
    monitor_feature_calculation,
)


def test_source_data_monitoring():
    """测试源数据监控"""
    print("=" * 70)
    print("测试 1: 源数据监控（正常数据）")
    print("=" * 70)

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "open": 100.0 + np.random.randn(100) * 0.1,
            "high": 100.5 + np.random.randn(100) * 0.1,
            "low": 99.5 + np.random.randn(100) * 0.1,
            "close": 100.0 + np.random.randn(100) * 0.1,
            "volume": 1000.0 + np.random.randn(100) * 100,
        },
        index=dates,
    )

    result = check_source_data_quality(df, "test_data_normal")
    assert not result["has_inf"], "正常数据不应该包含 inf"
    print("✅ 测试通过：正常数据监控正确\n")


def test_source_data_with_inf():
    """测试源数据包含 inf 的情况"""
    print("=" * 70)
    print("测试 2: 源数据监控（包含 inf）")
    print("=" * 70)

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "open": 100.0 + np.random.randn(100) * 0.1,
            "high": 100.5 + np.random.randn(100) * 0.1,
            "low": 99.5 + np.random.randn(100) * 0.1,
            "close": 100.0 + np.random.randn(100) * 0.1,
            "volume": 1000.0 + np.random.randn(100) * 100,
        },
        index=dates,
    )

    # 插入 inf 值
    df.loc[dates[50], "volume"] = np.inf
    df.loc[dates[51], "close"] = -np.inf

    result = check_source_data_quality(df, "test_data_with_inf")
    assert result["has_inf"], "应该检测到 inf 值"
    assert "volume" in result["inf_columns"], "应该检测到 volume 列的 inf"
    assert "close" in result["inf_columns"], "应该检测到 close 列的 inf"
    print("✅ 测试通过：inf 值检测正确\n")


def test_feature_calculation_monitoring():
    """测试特征计算监控"""
    print("=" * 70)
    print("测试 3: 特征计算监控（新产生的 inf）")
    print("=" * 70)

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df_before = pd.DataFrame(
        {
            "close": 100.0 + np.random.randn(100) * 0.1,
            "volume": 1000.0 + np.random.randn(100) * 100,
        },
        index=dates,
    )

    df_after = df_before.copy()
    # 模拟特征计算产生 inf
    df_after["feature_1"] = df_after["close"] / (
        df_after["volume"] - df_after["volume"].mean()
    )
    df_after.loc[dates[50], "feature_1"] = np.inf  # 手动插入 inf

    result = monitor_feature_calculation(
        df_before,
        df_after,
        feature_name="test_feature",
        stage="test_calculation",
    )

    assert "feature_1" in result["new_inf_columns"], "应该检测到新产生的 inf"
    print("✅ 测试通过：特征计算监控正确\n")


def test_data_quality_check():
    """测试数据质量检查函数"""
    print("=" * 70)
    print("测试 4: 数据质量检查函数")
    print("=" * 70)

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "col1": 100.0 + np.random.randn(100) * 0.1,
            "col2": 1000.0 + np.random.randn(100) * 100,
        },
        index=dates,
    )

    # 插入 inf 和 NaN
    df.loc[dates[50], "col1"] = np.inf
    df.loc[dates[51], "col2"] = np.nan

    result = check_data_quality(
        df,
        data_source="TEST",
        stage="test_stage",
        raise_on_inf=False,
        raise_on_nan=False,
    )

    assert result["has_inf"], "应该检测到 inf"
    assert result["has_nan"], "应该检测到 NaN"
    assert "col1" in result["inf_columns"], "应该检测到 col1 的 inf"
    assert "col2" in result["nan_columns"], "应该检测到 col2 的 NaN"
    print("✅ 测试通过：数据质量检查正确\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("数据监控系统测试")
    print("=" * 70 + "\n")

    try:
        test_source_data_monitoring()
        test_source_data_with_inf()
        test_feature_calculation_monitoring()
        test_data_quality_check()

        print("=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
