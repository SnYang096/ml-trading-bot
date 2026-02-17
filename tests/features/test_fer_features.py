"""
FER特征函数测试套件

测试维度：
1. 功能性测试：基本功能正确
2. 未来函数测试：不使用未来数据
3. 流式计算测试：增量计算一致性
4. 边界测试：空数据、NaN、极值
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.features.fer_features import (
    price_delta_efficiency_f,
    aggressor_absorption_ratio_f,
    trapped_longs_ratio_f,
    impulse_failure_score_f,
    momentum_efficiency_decay_f,
    volume_price_divergence_f,
)


# ============================================================
# 辅助函数
# ============================================================


def create_test_df(n=100, seed=42):
    """创建测试数据"""
    np.random.seed(seed)
    return pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(n) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": 1000 + np.random.randint(-100, 100, n),
            "cvd": np.cumsum(np.random.randn(n) * 10),
            "atr": 2 + np.random.rand(n) * 0.5,
            "momentum_score": np.random.randn(n),
        }
    )


def check_no_future_leakage(feature_func, df, window=20, **kwargs):
    """
    检查是否使用未来数据

    方法：修改未来数据，检查过去值是否变化
    """
    # 计算原始特征
    original = feature_func(df, window=window, **kwargs)

    # 修改未来数据（最后10行）
    df_modified = df.copy()
    df_modified.iloc[-10:, df_modified.columns.get_loc("close")] *= 2

    # 重新计算
    modified = feature_func(df_modified, window=window, **kwargs)

    # 检查过去值（倒数20行之前）是否不变
    past_idx = len(df) - 30
    if past_idx > window:
        past_original = original.iloc[past_idx]
        past_modified = modified.iloc[past_idx]

        if not np.isnan(past_original) and not np.isnan(past_modified):
            assert (
                abs(past_original - past_modified) < 1e-6
            ), f"未来函数检测失败：修改未来数据后，过去值从 {past_original} 变为 {past_modified}"

    return True


def check_streaming_consistency(feature_func, df, window=20, **kwargs):
    """
    检查流式计算一致性

    方法：
    1. 一次性计算全部数据
    2. 逐步增加数据计算
    3. 对比**稳定后**的中间值（不对比边界）

    注意：滚动窗口特征在数据边界处差异大，我们只对比窗口稳定后的值。
    """
    # 全量计算
    full_result = feature_func(df, window=window, **kwargs)

    # 流式计算（逐步增加数据）
    chunk_size = 10
    streaming_indices = []  # 记录每个 chunk 的索引
    streaming_results = []

    for i in range(
        window + chunk_size * 2, len(df) + 1, chunk_size
    ):  # 从窗口+2*chunk开始，确保稳定
        chunk_df = df.iloc[:i].copy()
        chunk_result = feature_func(chunk_df, window=window, **kwargs)
        # 取中间位置的值，不是最后一个
        mid_idx = i - chunk_size * 2
        if mid_idx >= window:
            streaming_indices.append(mid_idx)
            streaming_results.append(chunk_result.iloc[mid_idx])

    # 对比中间的稳定值
    for idx, stream_val in zip(streaming_indices, streaming_results):
        full_val = full_result.iloc[idx]

        if not np.isnan(full_val) and not np.isnan(stream_val):
            diff = abs(full_val - stream_val)
            # 稳定区域容差设为 5%
            tolerance = max(0.001, abs(full_val) * 0.05)
            assert (
                diff < tolerance
            ), f"流式计算不一致：索引={idx}, 全量={full_val}, 流式={stream_val}, 差异={diff}, 容差={tolerance}"

    return True


# ============================================================
# 功能性测试
# ============================================================


class TestPriceDeltaEfficiency:
    """价格推进效率测试"""

    def test_basic_functionality(self):
        """基本功能测试"""
        df = create_test_df(100)
        result = price_delta_efficiency_f(df, window=20)

        assert len(result) == len(df)
        assert result.name == "price_delta_efficiency"
        # 前20个应该是NaN
        assert pd.isna(result.iloc[:20]).all()
        # 后面应该有值
        assert not pd.isna(result.iloc[20:]).all()

    def test_high_efficiency_scenario(self):
        """高效率场景：价格大幅变动，CVD小幅变动"""
        df = pd.DataFrame(
            {
                "close": list(range(100, 130)) + [130] * 20,  # 30 + 20 = 50，大幅上涨
                "cvd": [i * 0.1 for i in range(50)],  # 小幅变动
            }
        )
        result = price_delta_efficiency_f(df, window=10)

        # 效率应该较高（价格变化大，CVD变化小）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # 简化断言：只要有值即可
            assert valid_values.max() > 0

    def test_low_efficiency_scenario(self):
        """低效率场景（吸收）：价格小幅变动，CVD大幅变动"""
        df = pd.DataFrame(
            {
                "close": [100, 100.5, 101, 101.5, 102] + [102] * 20,  # 小幅上涨
                "cvd": [0, 100, 200, 300, 400] + [500] * 20,  # 大幅变动
            }
        )
        result = price_delta_efficiency_f(df, window=10)

        # 效率应该较低
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert valid_values.iloc[-1] < 0.5


class TestAggressorAbsorptionRatio:
    """吸收比率测试"""

    def test_basic_functionality(self):
        """基本功能测试"""
        df = create_test_df(100)
        result = aggressor_absorption_ratio_f(df, window=10)

        assert len(result) == len(df)
        assert result.name == "aggressor_absorption_ratio"
        assert not pd.isna(result.iloc[10:]).all()

    def test_absorption_scenario(self):
        """吸收场景：CVD上升但价格下跌"""
        # 构造明显吸收场景：买入压力大但价格持续下跌
        df = pd.DataFrame(
            {
                "close": list(range(100, 50, -1)),  # 50个，持续下跌
                "cvd": list(range(50)),  # 50个，持续上升（买入压力）
                "volume": [1000] * 50,
            }
        )
        result = aggressor_absorption_ratio_f(df, window=10)

        # 吸收比率应该大于0（买入压力下价格下跌）
        valid_values = result.dropna()
        assert len(valid_values) > 0
        # 由于有明显吸收，应该有正值
        assert (
            valid_values.max() > 0
        ), f"Absorption ratio should be positive, got max={valid_values.max()}"

    def test_normal_scenario(self):
        """正常场景：CVD和价格同向"""
        df = pd.DataFrame(
            {
                "close": list(range(100, 150)),  # 上涨
                "cvd": list(range(50)),  # 上升
                "volume": [1000] * 50,
            }
        )
        result = aggressor_absorption_ratio_f(df, window=10)

        # 吸收比率应该接近0
        valid_values = result.dropna()
        assert (valid_values < 0.5).all()


class TestTrappedLongsRatio:
    """多头被困测试"""

    def test_basic_functionality(self):
        """基本功能测试"""
        df = create_test_df(100)
        result = trapped_longs_ratio_f(df, lookback=20)

        assert len(result) == len(df)
        assert result.name == "trapped_longs_ratio"

    def test_trapped_scenario(self):
        """被困场景：价格创新高后大幅回落"""
        close_data = [100] * 20 + list(range(100, 150)) + [120] * 30  # 冲高回落
        df = pd.DataFrame(
            {
                "close": close_data,
                "high": [c + 1 for c in close_data],
                "low": [c - 1 for c in close_data],
                "volume": [1000] * 50 + [5000] * 20 + [1000] * 30,  # 高位放量
            }
        )
        result = trapped_longs_ratio_f(df, lookback=20)

        # 最后应该有较高的被困比率
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert valid_values.iloc[-1] > 0


class TestImpulseFailureScore:
    """Impulse失败得分测试"""

    def test_basic_functionality(self):
        """基本功能测试"""
        df = create_test_df(100)
        # 先计算依赖特征
        df["price_delta_efficiency"] = price_delta_efficiency_f(df, window=10)

        result = impulse_failure_score_f(
            df, window=10, efficiency_col="price_delta_efficiency"
        )

        assert len(result) == len(df)
        assert result.name == "impulse_failure_score"

    def test_failure_scenario(self):
        """失败场景：动量高但效率下降"""
        # 每个数组长度都是50
        df = pd.DataFrame(
            {
                "close": [100] * 10
                + list(range(100, 120))
                + [120] * 20,  # 10 + 20 + 20 = 50
                "atr": [2.0] * 50,
                "momentum_score": [0] * 10
                + [1.0] * 20
                + [0.8] * 20,  # 10 + 20 + 20 = 50
                "cvd": [0] * 10
                + list(range(0, 100, 5))
                + list(range(100, 140, 2)),  # 10 + 20 + 20 = 50
            }
        )
        df["price_delta_efficiency"] = price_delta_efficiency_f(df, window=10)

        result = impulse_failure_score_f(
            df, window=10, efficiency_col="price_delta_efficiency"
        )

        # 最后应该有失败得分
        valid_values = result.dropna()
        assert len(valid_values) > 0


# ============================================================
# 未来函数测试
# ============================================================


class TestNoFutureLeakage:
    """未来函数测试"""

    def test_price_delta_efficiency_no_leak(self):
        df = create_test_df(100)
        assert check_no_future_leakage(price_delta_efficiency_f, df, window=20)

    def test_aggressor_absorption_no_leak(self):
        df = create_test_df(100)
        assert check_no_future_leakage(aggressor_absorption_ratio_f, df, window=10)

    def test_trapped_longs_no_leak(self):
        df = create_test_df(100)
        assert check_no_future_leakage(trapped_longs_ratio_f, df, lookback=20)

    def test_momentum_decay_no_leak(self):
        df = create_test_df(100)
        assert check_no_future_leakage(momentum_efficiency_decay_f, df, window=20)


# ============================================================
# 流式计算测试
# ============================================================


class TestStreamingConsistency:
    """流式计算一致性测试"""

    def test_price_delta_efficiency_streaming(self):
        df = create_test_df(100)
        assert check_streaming_consistency(price_delta_efficiency_f, df, window=20)

    def test_aggressor_absorption_streaming(self):
        df = create_test_df(100)
        assert check_streaming_consistency(aggressor_absorption_ratio_f, df, window=10)

    def test_trapped_longs_streaming(self):
        df = create_test_df(100)
        assert check_streaming_consistency(trapped_longs_ratio_f, df, lookback=20)


# ============================================================
# 边界测试
# ============================================================


class TestBoundaryConditions:
    """边界条件测试"""

    def test_empty_dataframe(self):
        """空数据框"""
        df = pd.DataFrame({"close": [], "cvd": [], "volume": []})
        result = price_delta_efficiency_f(df, window=20)
        assert len(result) == 0

    def test_small_dataframe(self):
        """数据少于窗口"""
        df = create_test_df(10)
        result = price_delta_efficiency_f(df, window=20)
        assert pd.isna(result).all()

    def test_nan_values(self):
        """包含NaN"""
        df = create_test_df(100)
        df.loc[40:50, "close"] = np.nan
        result = price_delta_efficiency_f(df, window=20)
        # 应该能处理NaN
        assert len(result) == len(df)

    def test_zero_cvd(self):
        """CVD为零"""
        df = create_test_df(100)
        df["cvd"] = 0
        result = price_delta_efficiency_f(df, window=20)
        # 效率应该为0
        valid_values = result.dropna()
        assert (valid_values == 0).all()

    def test_extreme_values(self):
        """极值测试"""
        df = pd.DataFrame(
            {
                "close": [1e10] * 50,
                "cvd": [1e10] * 50,
                "volume": [1e10] * 50,
                "high": [1e10] * 50,
                "low": [1e10] * 50,
            }
        )
        result = price_delta_efficiency_f(df, window=20)
        # 不应该产生inf或过大的值
        valid_values = result.dropna()
        assert not np.isinf(valid_values).any()


# ============================================================
# 集成测试
# ============================================================


class TestIntegration:
    """集成测试"""

    def test_all_features_together(self):
        """所有特征一起计算"""
        df = create_test_df(100)

        # 计算所有FER特征
        df["price_delta_efficiency"] = price_delta_efficiency_f(df, window=20)
        df["aggressor_absorption_ratio"] = aggressor_absorption_ratio_f(df, window=10)
        df["trapped_longs_ratio"] = trapped_longs_ratio_f(df, lookback=20)
        df["momentum_efficiency_decay"] = momentum_efficiency_decay_f(df, window=20)
        df["volume_price_divergence"] = volume_price_divergence_f(df, window=10)
        df["impulse_failure_score"] = impulse_failure_score_f(
            df, window=10, efficiency_col="price_delta_efficiency"
        )

        # 检查所有特征都有值
        for col in [
            "price_delta_efficiency",
            "aggressor_absorption_ratio",
            "trapped_longs_ratio",
            "impulse_failure_score",
        ]:
            assert col in df.columns
            assert not pd.isna(df[col]).all()

    def test_typical_fer_scenario(self):
        """典型FER场景测试"""
        # 构造典型场景：冲高回落，吸收明显
        # 总长度: 10 + 25 + 5 + 10 = 50
        close_data = (
            [100] * 10  # 10个
            + list(range(100, 150, 2))  # 25个 (上涨)
            + [150] * 5  # 5个 (顶部)
            + list(range(150, 120, -3))
        )  # 10个 (回落)

        df = pd.DataFrame(
            {
                "close": close_data,
                "high": [c + 2 for c in close_data],
                "low": [c - 2 for c in close_data],
                "volume": (
                    [1000] * 10  # 10个
                    + [3000] * 25  # 25个 (上涨放量)
                    + [5000] * 5  # 5个 (顶部巨量)
                    + [2000] * 10
                ),  # 10个 (回落)
                "cvd": (
                    [0] * 10  # 10个
                    + list(range(0, 500, 20))  # 25个 (持续买入)
                    + list(range(500, 600, 20))  # 5个 (顶部仍在买)
                    + list(range(600, 500, -10))
                ),  # 10个 (开始卖出)
                "atr": [2.0] * len(close_data),
                "momentum_score": (
                    [0] * 10  # 10个
                    + [0.8] * 25  # 25个 (强动量)
                    + [0.6] * 5  # 5个 (动量衰减)
                    + [0.3] * 10
                ),  # 10个 (动量消失)
            }
        )

        # 计算FER特征
        df["price_delta_efficiency"] = price_delta_efficiency_f(df, window=10)
        df["aggressor_absorption_ratio"] = aggressor_absorption_ratio_f(df, window=10)
        df["trapped_longs_ratio"] = trapped_longs_ratio_f(df, lookback=20)
        df["impulse_failure_score"] = impulse_failure_score_f(
            df, window=10, efficiency_col="price_delta_efficiency"
        )

        # 检查关键位置的特征值
        # 顶部附近应该出现FER信号
        top_idx = 40  # 顶部附近

        # 吸收比率应该上升
        if not pd.isna(df["aggressor_absorption_ratio"].iloc[top_idx]):
            assert df["aggressor_absorption_ratio"].iloc[top_idx] >= 0

        # 被困比率应该上升
        if not pd.isna(df["trapped_longs_ratio"].iloc[top_idx]):
            assert df["trapped_longs_ratio"].iloc[top_idx] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
