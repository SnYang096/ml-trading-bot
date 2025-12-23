"""
测试cache数据质量验证功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.loader.feature_computer import FeatureComputer


def test_validate_cache_quality():
    """测试数据质量验证功能"""

    computer = FeatureComputer()

    # 测试1: 正常数据（少量NaN，正常）
    print("\n=== 测试1: 正常数据 ===")
    normal_data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }
    )
    normal_data.loc[0:2, "feature1"] = np.nan  # 3个NaN，总共200个值，1.5% NaN
    result = computer._validate_cache_quality(
        normal_data, "test_feature", cache_type="memory"
    )
    print(
        f"   Result: nan_pct={result['nan_pct']:.2f}%, inf_pct={result['inf_pct']:.2f}%, has_issues={result['has_issues']}"
    )
    assert result["nan_pct"] < 50.0  # 应该小于50%阈值
    assert result["inf_pct"] == 0.0
    assert not result["has_issues"]  # 1.5% NaN < 50%阈值，应该没问题

    # 测试2: NaN过多（超过阈值）
    print("\n=== 测试2: NaN过多 ===")
    nan_heavy_data = pd.DataFrame(
        {
            "feature1": np.full(100, np.nan),
            "feature2": np.random.randn(100),
        }
    )
    nan_heavy_data.loc[0:59, "feature2"] = (
        np.nan
    )  # feature1全部NaN，feature2 60% NaN，总计约80% NaN
    result = computer._validate_cache_quality(
        nan_heavy_data, "nan_feature", cache_type="monthly"
    )
    print(
        f"   Result: nan_pct={result['nan_pct']:.2f}%, has_issues={result['has_issues']}"
    )
    assert result["nan_pct"] > 50.0  # 应该超过50%阈值
    assert result["has_issues"]  # 应该有问题

    # 测试3: inf过多（超过阈值）
    print("\n=== 测试3: inf过多 ===")
    inf_heavy_data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
        }
    )
    inf_heavy_data.loc[0:19, "feature1"] = np.inf  # 20% inf
    result = computer._validate_cache_quality(
        inf_heavy_data, "inf_feature", cache_type="memory"
    )
    print(
        f"   Result: inf_pct={result['inf_pct']:.2f}%, has_issues={result['has_issues']}"
    )
    assert result["inf_pct"] == 20.0
    assert result["has_issues"]  # 20% inf > 10%阈值，应该有问题

    # 测试4: Series输入
    print("\n=== 测试4: Series输入 ===")
    series_data = pd.Series(np.random.randn(100), name="series_feature")
    series_data.loc[0:9] = np.nan  # 10% NaN
    result = computer._validate_cache_quality(
        series_data, "series_feature", cache_type="computed"
    )
    print(
        f"   Result: nan_pct={result['nan_pct']:.2f}%, has_issues={result['has_issues']}"
    )
    assert result["nan_pct"] == 10.0
    assert not result["has_issues"]  # 10% NaN < 50%阈值

    # 测试5: 空数据
    print("\n=== 测试5: 空数据 ===")
    empty_data = pd.DataFrame()
    result = computer._validate_cache_quality(
        empty_data, "empty_feature", cache_type="memory"
    )
    assert result["has_issues"]
    assert "empty" in result["warnings"][0].lower()

    # 测试6: 混合数据类型（数值+非数值）
    print("\n=== 测试6: 混合数据类型 ===")
    mixed_data = pd.DataFrame(
        {
            "numeric_col": np.random.randn(100),
            "string_col": ["text"] * 100,
            "object_col": [{"key": "value"}] * 100,
        }
    )
    mixed_data.loc[0:9, "numeric_col"] = np.nan  # 10% NaN in numeric column
    result = computer._validate_cache_quality(
        mixed_data, "mixed_feature", cache_type="memory"
    )
    print(
        f"   Result: nan_pct={result['nan_pct']:.2f}%, has_issues={result['has_issues']}"
    )
    assert result["nan_pct"] == 10.0  # 只有数值列被检查
    assert not result["has_issues"]  # 10% < 50%阈值

    # 测试7: 纯非数值类型（应该跳过验证）
    print("\n=== 测试7: 纯非数值类型 ===")
    non_numeric_data = pd.DataFrame(
        {
            "string_col": ["text"] * 100,
            "object_col": [{"key": "value"}] * 100,
        }
    )
    result = computer._validate_cache_quality(
        non_numeric_data, "non_numeric_feature", cache_type="memory"
    )
    print(
        f"   Result: total_values={result['total_values']}, has_issues={result['has_issues']}"
    )
    assert result["total_values"] == 0  # 没有数值列
    assert not result["has_issues"]  # 应该跳过验证，没有问题

    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_validate_cache_quality()
