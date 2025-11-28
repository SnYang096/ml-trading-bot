"""
测试扩展波动率特征提取器
验证所有特征是否正确生成，使用模拟数据
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_series.utils_volatility_features import (
    extract_extended_volatility_features,
)


# 期望的所有特征列表（从feature_dependencies.yaml和volatility_model.yaml）
EXPECTED_FEATURES = [
    # 1. Multi-scale historical volatility (4个)
    "vol_raw_5",
    "vol_raw_10",
    "vol_raw_20",
    "vol_raw_60",
    # 2. ATR-derived features (15个)
    "vol_atr_norm",
    "vol_atr_ma_5",
    "vol_atr_ma_10",
    "vol_atr_ma_20",
    "vol_atr_std_5",
    "vol_atr_std_10",
    "vol_atr_std_20",
    "vol_atr_max_5",
    "vol_atr_max_10",
    "vol_atr_max_20",
    "vol_atr_min_5",
    "vol_atr_min_10",
    "vol_atr_min_20",
    "vol_atr_ratio_20",
    "vol_atr_change",
    "vol_atr_change_abs",
    # 3. Lag features (3个)
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
    # 4. Trend features (4个)
    "vol_slope_5",
    "vol_slope_10",
    "vol_slope_20",
    "vol_accel",
    # 5. Moving averages (6个)
    "vol_ma_5",
    "vol_ma_10",
    "vol_ma_20",
    "vol_ema_5",
    "vol_ema_10",
    "vol_ema_20",
    # 6. Regime features (2个)
    "vol_zscore",
    "vol_percentile_approx",
    # 7. Range features (4个)
    "vol_range_10",
    "vol_range_20",
    "vol_range_pos_10",
    "vol_range_pos_20",
    # 8. Momentum features (3个)
    "vol_mom_3",
    "vol_mom_5",
    "vol_mom_10",
]

# 总共应该有41个特征
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURES)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    创建模拟数据用于测试

    Args:
        n_samples: 样本数量
        seed: 随机种子

    Returns:
        包含价格和ATR数据的DataFrame
    """
    np.random.seed(seed)

    # 创建时间索引
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成价格数据（随机游走 + 趋势）
    returns = np.random.randn(n_samples) * 0.01
    # 添加一些趋势和波动率聚集
    trend = np.sin(np.arange(n_samples) / 50) * 0.001
    volatility_cluster = np.abs(np.random.randn(n_samples)) * 0.005
    returns = returns + trend + volatility_cluster

    prices = 100 * np.exp(np.cumsum(returns))

    # 生成high/low（价格上下波动）
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))

    df = pd.DataFrame(
        {
            "close": prices,
            "high": high,
            "low": low,
            "volume": np.random.lognormal(10, 1, n_samples),
        },
        index=dates,
    )

    # 计算ATR（使用真实的高低差）
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean()

    # 确保ATR不为0（避免除零错误）
    df["atr"] = df["atr"].clip(lower=1e-6)

    return df


def test_extract_extended_volatility_features_basic():
    """测试基本功能：特征是否正确生成"""
    df = create_mock_data(n_samples=500)

    # 提取特征
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 验证返回的是DataFrame
    assert isinstance(result, pd.DataFrame), "结果应该是DataFrame"

    # 验证索引匹配
    assert len(result) == len(df), "结果长度应该与输入数据相同"
    assert result.index.equals(df.index), "索引应该匹配"

    # 验证特征数量
    assert (
        len(result.columns) == EXPECTED_FEATURE_COUNT
    ), f"特征数量应该是{EXPECTED_FEATURE_COUNT}，实际是{len(result.columns)}"

    # 验证所有期望的特征都存在
    missing_features = set(EXPECTED_FEATURES) - set(result.columns)
    assert len(missing_features) == 0, f"缺少以下特征: {missing_features}"

    # 验证没有多余的特征
    extra_features = set(result.columns) - set(EXPECTED_FEATURES)
    assert len(extra_features) == 0, f"存在多余的特征: {extra_features}"


def test_extract_extended_volatility_features_values():
    """测试特征值是否合理（非NaN、非Inf、有限值）"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 检查每个特征
    for col in result.columns:
        # 前20行可能有NaN（因为滚动窗口），但之后应该都是有效值
        valid_data = result[col].iloc[100:]  # 跳过前100行

        # 检查是否有NaN
        nan_count = valid_data.isna().sum()
        assert nan_count == 0, f"特征 {col} 在有效数据中有 {nan_count} 个NaN值"

        # 检查是否有Inf
        inf_count = np.isinf(valid_data).sum()
        assert inf_count == 0, f"特征 {col} 在有效数据中有 {inf_count} 个Inf值"

        # 检查是否都是有限值
        finite_count = np.isfinite(valid_data).sum()
        assert finite_count == len(
            valid_data
        ), f"特征 {col} 在有效数据中有 {len(valid_data) - finite_count} 个非有限值"


def test_extract_extended_volatility_features_ranges():
    """测试特征值是否在合理范围内"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 跳过前100行（滚动窗口初始化）
    valid_data = result.iloc[100:]

    # 检查vol_raw_*应该在合理范围（波动率通常是0-0.1）
    for col in ["vol_raw_5", "vol_raw_10", "vol_raw_20", "vol_raw_60"]:
        if col in valid_data.columns:
            max_val = valid_data[col].max()
            min_val = valid_data[col].min()
            assert min_val >= 0, f"{col} 的最小值应该是非负的，实际是 {min_val}"
            assert max_val < 1.0, f"{col} 的最大值应该小于1.0，实际是 {max_val}"

    # 检查vol_atr_norm应该在合理范围
    if "vol_atr_norm" in valid_data.columns:
        max_val = valid_data["vol_atr_norm"].max()
        min_val = valid_data["vol_atr_norm"].min()
        assert min_val >= 0, f"vol_atr_norm 的最小值应该是非负的，实际是 {min_val}"
        assert max_val < 0.1, f"vol_atr_norm 的最大值应该小于0.1，实际是 {max_val}"

    # 检查vol_percentile_approx应该在[0, 1]范围内
    if "vol_percentile_approx" in valid_data.columns:
        max_val = valid_data["vol_percentile_approx"].max()
        min_val = valid_data["vol_percentile_approx"].min()
        assert min_val >= 0, f"vol_percentile_approx 的最小值应该>=0，实际是 {min_val}"
        assert max_val <= 1, f"vol_percentile_approx 的最大值应该<=1，实际是 {max_val}"

    # 检查vol_range_pos_*应该在[0, 1]范围内
    for col in ["vol_range_pos_10", "vol_range_pos_20"]:
        if col in valid_data.columns:
            max_val = valid_data[col].max()
            min_val = valid_data[col].min()
            assert min_val >= 0, f"{col} 的最小值应该>=0，实际是 {min_val}"
            assert max_val <= 1, f"{col} 的最大值应该<=1，实际是 {max_val}"


def test_extract_extended_volatility_features_without_atr():
    """测试没有ATR列时的行为"""
    df = create_mock_data(n_samples=500)
    # 移除ATR列
    df_no_atr = df.drop(columns=["atr"])

    result = extract_extended_volatility_features(
        df_no_atr,
        price_col="close",
        atr_col="atr",  # 指定不存在的列
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 应该仍然生成非ATR相关的特征
    assert len(result.columns) > 0, "即使没有ATR，也应该生成一些特征"

    # 验证vol_raw_*特征仍然存在
    for col in ["vol_raw_5", "vol_raw_10", "vol_raw_20", "vol_raw_60"]:
        assert col in result.columns, f"特征 {col} 应该存在（不依赖ATR）"

    # 验证ATR相关特征不存在
    atr_features = [col for col in result.columns if "atr" in col.lower()]
    assert len(atr_features) == 0, f"不应该有ATR相关特征，但找到了: {atr_features}"


def test_extract_extended_volatility_features_custom_lag_periods():
    """测试自定义lag_periods"""
    df = create_mock_data(n_samples=500)

    # 使用自定义lag_periods
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 5, 10],  # 自定义滞后
    )

    # 验证lag特征
    assert "vol_lag_1" in result.columns
    assert "vol_lag_5" in result.columns
    assert "vol_lag_10" in result.columns
    assert "vol_lag_2" not in result.columns  # 不应该存在
    assert "vol_lag_3" not in result.columns  # 不应该存在


def test_extract_extended_volatility_features_custom_window():
    """测试自定义window参数"""
    df = create_mock_data(n_samples=500)

    # 使用不同的window
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=30,  # 自定义窗口
        lag_periods=[1, 2, 3],
    )

    # 验证特征仍然正确生成
    assert len(result.columns) == EXPECTED_FEATURE_COUNT
    assert "vol_raw_5" in result.columns
    assert "vol_zscore" in result.columns


def test_extract_extended_volatility_features_edge_cases():
    """测试边界情况"""
    # 测试非常小的数据集
    df_small = create_mock_data(n_samples=50)
    result_small = extract_extended_volatility_features(
        df_small,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )
    assert len(result_small) == 50
    assert len(result_small.columns) == EXPECTED_FEATURE_COUNT

    # 测试价格全为0的情况（应该被clip处理）
    df_zero = create_mock_data(n_samples=100)
    df_zero["close"] = 0.0
    result_zero = extract_extended_volatility_features(
        df_zero,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )
    # 应该不会崩溃，但特征值可能都是0或NaN
    assert len(result_zero) == 100


def test_extract_extended_volatility_features_feature_relationships():
    """测试特征之间的逻辑关系"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    valid_data = result.iloc[100:]

    # vol_range_* 应该是 vol_max - vol_min，所以应该 >= 0
    for col in ["vol_range_10", "vol_range_20"]:
        if col in valid_data.columns:
            assert (valid_data[col] >= 0).all(), f"{col} 应该都是非负的"

    # vol_range_pos_* 应该在[0, 1]范围内
    for col in ["vol_range_pos_10", "vol_range_pos_20"]:
        if col in valid_data.columns:
            assert (valid_data[col] >= 0).all(), f"{col} 应该都是非负的"
            assert (valid_data[col] <= 1).all(), f"{col} 应该都<=1"

    # vol_atr_ratio_20 应该是当前ATR / 20期均值，应该接近1（如果波动率稳定）
    if "vol_atr_ratio_20" in valid_data.columns:
        ratio = valid_data["vol_atr_ratio_20"]
        assert (ratio > 0).all(), "vol_atr_ratio_20 应该都是正数"
        # 大部分值应该在合理范围内（比如0.5到2.0）
        reasonable_ratio = ((ratio >= 0.1) & (ratio <= 10.0)).sum() / len(ratio)
        assert reasonable_ratio > 0.8, (
            f"vol_atr_ratio_20 的大部分值应该在[0.1, 10.0]范围内，"
            f"实际只有 {reasonable_ratio*100:.1f}%"
        )


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
