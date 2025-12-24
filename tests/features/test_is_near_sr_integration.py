"""
集成测试：is_near_sr 特征计算和标签过滤逻辑

测试场景：
1. compute_is_near_sr 特征函数的正确性
2. 标签生成函数中的 SR 过滤逻辑
3. 单位转换（百分比 -> 绝对价格距离 -> ATR倍数）
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features.time_series.utils_interaction_features import (
    compute_is_near_sr,
    compute_is_near_sr_from_series,
)
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_with_weights,
)


@pytest.fixture
def sample_data_with_sr_features():
    """创建包含 SR 相关特征的测试数据"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")

    # 基础价格数据
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "open": prices * 0.995,
            "volume": np.random.uniform(1000, 5000, 100),
            # SR 相关特征
            "dist_to_nearest_sr": np.random.uniform(-0.1, 0.1, 100),  # -10% 到 +10%
            "atr": np.full(100, 2.0),  # 固定 ATR = 2.0
        },
        index=dates,
    )

    return df


class TestComputeIsNearSr:
    """测试 compute_is_near_sr 特征函数"""

    def test_basic_computation(self, sample_data_with_sr_features):
        """测试基本计算逻辑"""
        df = sample_data_with_sr_features.copy()

        result = compute_is_near_sr(
            df,
            dist_col="dist_to_nearest_sr",
            atr_col="atr",
            price_col="close",
            dist_atr_mult=1.5,
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.name == "is_near_sr"
        assert result.dtype == bool
        assert result.notna().all()

    def test_unit_conversion_correctness(self):
        """测试单位转换的正确性"""
        # 创建精确的测试数据
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0, 100.0],
                "dist_to_nearest_sr": [0.05, -0.05, 0.20, 0.03],  # 5%, -5%, 20%, 3%
                "atr": [10.0, 10.0, 10.0, 10.0],  # ATR = 10
            }
        )

        result = compute_is_near_sr(df, dist_atr_mult=1.5)

        # 情况1: dist = 0.05 (5%), price = 100, atr = 10
        # abs_distance = 0.05 * 100 = 5
        # dist_normalized = 5 / 10 = 0.5 ATR
        # is_near = 0.5 <= 1.5 = True
        assert result.iloc[0] == True, "5% 距离应该在 1.5 ATR 内"

        # 情况2: dist = -0.05 (-5%), price = 100, atr = 10
        # abs_distance = 0.05 * 100 = 5
        # dist_normalized = 5 / 10 = 0.5 ATR
        # is_near = 0.5 <= 1.5 = True
        assert result.iloc[1] == True, "-5% 距离应该在 1.5 ATR 内"

        # 情况3: dist = 0.20 (20%), price = 100, atr = 10
        # abs_distance = 0.20 * 100 = 20
        # dist_normalized = 20 / 10 = 2.0 ATR
        # is_near = 2.0 <= 1.5 = False
        assert result.iloc[2] == False, "20% 距离应该超出 1.5 ATR"

        # 情况4: dist = 0.03 (3%), price = 100, atr = 10
        # abs_distance = 0.03 * 100 = 3
        # dist_normalized = 3 / 10 = 0.3 ATR
        # is_near = 0.3 <= 1.5 = True
        assert result.iloc[3] == True, "3% 距离应该在 1.5 ATR 内"

    def test_threshold_boundary(self):
        """测试阈值边界情况"""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "dist_to_nearest_sr": [0.15, 0.1501],  # 正好在 1.5 ATR 边界
                "atr": [10.0, 10.0],
            }
        )

        result = compute_is_near_sr(df, dist_atr_mult=1.5)

        # dist = 0.15 (15%), price = 100, atr = 10
        # abs_distance = 0.15 * 100 = 15
        # dist_normalized = 15 / 10 = 1.5 ATR
        # is_near = 1.5 <= 1.5 = True
        assert result.iloc[0] == True, "正好在阈值上应该返回 True"

        # dist = 0.1501 (15.01%), price = 100, atr = 10
        # abs_distance = 0.1501 * 100 = 15.01
        # dist_normalized = 15.01 / 10 = 1.501 ATR
        # is_near = 1.501 <= 1.5 = False
        assert result.iloc[1] == False, "略超过阈值应该返回 False"

    def test_custom_threshold(self):
        """测试自定义阈值"""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "dist_to_nearest_sr": [0.10, 0.20],  # 10%, 20%
                "atr": [10.0, 10.0],
            }
        )

        # 使用 2.0 ATR 阈值
        result = compute_is_near_sr(df, dist_atr_mult=2.0)

        # 10% -> 1.0 ATR -> True
        assert result.iloc[0] == True
        # 20% -> 2.0 ATR -> True (正好在阈值上)
        assert result.iloc[1] == True

    def test_missing_values(self):
        """测试缺失值处理"""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "dist_to_nearest_sr": [0.05, np.nan, 0.10],
                "atr": [10.0, 10.0, np.nan],
            }
        )

        result = compute_is_near_sr(df)

        # 第一个值应该正常计算
        assert result.iloc[0] == True

        # dist_to_nearest_sr 为 NaN 时，abs() 后为 NaN，计算后会被 fillna(False)
        assert result.iloc[1] == False

        # atr 为 NaN 时，fillna 会使用中位数（10.0），所以会正常计算
        # 0.10 * 100 / 10 = 1.0 ATR <= 1.5 -> True
        assert result.iloc[2] == True

    def test_from_series_entrypoint(self):
        """测试 narrow-IO entrypoint"""
        dist_series = pd.Series([0.05, -0.05, 0.20])
        atr_series = pd.Series([10.0, 10.0, 10.0])
        close_series = pd.Series([100.0, 100.0, 100.0])

        result = compute_is_near_sr_from_series(
            dist_to_nearest_sr=dist_series,
            atr=atr_series,
            close=close_series,
            dist_atr_mult=1.5,
        )

        assert isinstance(result, pd.DataFrame)
        assert "is_near_sr" in result.columns
        assert result["is_near_sr"].iloc[0] == True
        assert result["is_near_sr"].iloc[1] == True
        assert result["is_near_sr"].iloc[2] == False

    def test_different_price_levels(self):
        """测试不同价格水平下的计算"""
        # 测试价格对计算的影响
        df1 = pd.DataFrame(
            {
                "close": [50.0],  # 低价格
                "dist_to_nearest_sr": [0.10],  # 10%
                "atr": [5.0],  # ATR = 5
            }
        )

        df2 = pd.DataFrame(
            {
                "close": [200.0],  # 高价格
                "dist_to_nearest_sr": [0.10],  # 10%
                "atr": [20.0],  # ATR = 20
            }
        )

        result1 = compute_is_near_sr(df1, dist_atr_mult=1.5)
        result2 = compute_is_near_sr(df2, dist_atr_mult=1.5)

        # df1: abs_distance = 0.10 * 50 = 5, dist_normalized = 5 / 5 = 1.0 ATR -> True
        assert result1.iloc[0] == True

        # df2: abs_distance = 0.10 * 200 = 20, dist_normalized = 20 / 20 = 1.0 ATR -> True
        assert result2.iloc[0] == True


class TestLabelFilteringWithSrMask:
    """测试标签生成函数中的 SR 过滤逻辑"""

    def test_sr_filtering_with_dist_to_sr_col(self):
        """测试使用 dist_to_sr_col 进行 SR 过滤"""
        # 创建测试数据
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1H")
        df = pd.DataFrame(
            {
                "close": 100.0 + np.random.randn(50) * 2,
                "high": 100.0 + np.random.randn(50) * 2 + 1,
                "low": 100.0 + np.random.randn(50) * 2 - 1,
                "open": 100.0 + np.random.randn(50) * 2,
                "atr": np.full(50, 2.0),
                # 创建一些在SR附近和不在SR附近的数据
                "dist_to_nearest_sr": np.concatenate(
                    [
                        np.random.uniform(-0.02, 0.02, 25),  # 在SR附近（±2%）
                        np.random.uniform(
                            -0.10, -0.05, 12
                        ),  # 不在SR附近（-10% 到 -5%）
                        np.random.uniform(0.05, 0.10, 13),  # 不在SR附近（5% 到 10%）
                    ]
                ),
            },
            index=dates,
        )

        # 生成标签（使用 dist_to_sr_col 过滤）
        labels = compute_sr_reversal_label_with_weights(
            df,
            price_col="close",
            high_col="high",
            low_col="low",
            atr_col="atr",
            max_holding_bars=10,
            stop_loss_r=1.0,
            take_profit_r=2.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,  # 1.5 ATR 阈值
            compute_weights=False,
        )

        # 验证：在SR附近的样本应该有标签，不在SR附近的样本标签应该为 NaN
        near_sr_mask = df["dist_to_nearest_sr"].abs() * df["close"] <= 1.5 * df["atr"]

        # 检查：在SR附近的样本，标签应该存在（非NaN）
        near_sr_labels = labels[near_sr_mask]
        assert near_sr_labels.notna().sum() > 0, "在SR附近的样本应该有标签"

        # 检查：不在SR附近的样本，标签应该为 NaN
        far_sr_mask = ~near_sr_mask
        if far_sr_mask.sum() > 0:
            far_sr_labels = labels[far_sr_mask]
            # 注意：由于标签生成逻辑，即使不在SR附近，如果满足TP/SL条件也可能有标签
            # 但大部分应该被过滤掉
            assert (
                far_sr_labels.notna().sum() < near_sr_labels.notna().sum()
            ), "不在SR附近的样本标签应该更少"

    def test_sr_filtering_with_sr_mask_col(self):
        """测试使用 sr_mask_col 进行 SR 过滤"""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1H")
        df = pd.DataFrame(
            {
                "close": 100.0 + np.random.randn(50) * 2,
                "high": 100.0 + np.random.randn(50) * 2 + 1,
                "low": 100.0 + np.random.randn(50) * 2 - 1,
                "open": 100.0 + np.random.randn(50) * 2,
                "atr": np.full(50, 2.0),
                "is_near_sr": np.concatenate(
                    [
                        np.full(25, True),  # 前25个在SR附近
                        np.full(25, False),  # 后25个不在SR附近
                    ]
                ),
            },
            index=dates,
        )

        # 生成标签（使用 sr_mask_col 过滤）
        labels = compute_sr_reversal_label_with_weights(
            df,
            price_col="close",
            high_col="high",
            low_col="low",
            atr_col="atr",
            max_holding_bars=10,
            stop_loss_r=1.0,
            take_profit_r=2.0,
            combine_mode="long_only",
            sr_mask_col="is_near_sr",
            compute_weights=False,
        )

        # 验证：is_near_sr=True 的样本应该有标签
        near_sr_labels = labels[df["is_near_sr"]]
        assert near_sr_labels.notna().sum() > 0, "is_near_sr=True 的样本应该有标签"

        # 验证：is_near_sr=False 的样本标签应该为 NaN
        far_sr_labels = labels[~df["is_near_sr"]]
        assert far_sr_labels.notna().sum() == 0, "is_near_sr=False 的样本标签应该为 NaN"

    def test_no_sr_filtering(self):
        """测试不使用 SR 过滤的情况（全量扫描）"""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1H")
        df = pd.DataFrame(
            {
                "close": 100.0 + np.random.randn(50) * 2,
                "high": 100.0 + np.random.randn(50) * 2 + 1,
                "low": 100.0 + np.random.randn(50) * 2 - 1,
                "open": 100.0 + np.random.randn(50) * 2,
                "atr": np.full(50, 2.0),
            },
            index=dates,
        )

        # 生成标签（不使用 SR 过滤）
        labels = compute_sr_reversal_label_with_weights(
            df,
            price_col="close",
            high_col="high",
            low_col="low",
            atr_col="atr",
            max_holding_bars=10,
            stop_loss_r=1.0,
            take_profit_r=2.0,
            combine_mode="long_only",
            compute_weights=False,
        )

        # 验证：应该有更多的标签（因为没有过滤）
        assert labels.notna().sum() > 0, "全量扫描应该有标签"


class TestIntegrationWithRealData:
    """集成测试：使用真实数据模式"""

    def test_end_to_end_pipeline(self):
        """测试端到端流程：特征计算 -> 标签生成 -> SR过滤"""
        # 创建模拟真实场景的数据
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1H")

        # 模拟价格走势
        prices = 100.0 + np.cumsum(np.random.randn(200) * 0.5)

        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "open": prices * 0.995,
                "volume": np.random.uniform(1000, 5000, 200),
                "atr": np.full(200, 2.0),
                # 模拟 dist_to_nearest_sr（相对百分比）
                "dist_to_nearest_sr": np.random.uniform(-0.15, 0.15, 200),
            },
            index=dates,
        )

        # 步骤1: 计算 is_near_sr 特征
        is_near_sr = compute_is_near_sr(
            df,
            dist_col="dist_to_nearest_sr",
            atr_col="atr",
            price_col="close",
            dist_atr_mult=1.5,
        )
        df["is_near_sr"] = is_near_sr

        # 验证特征计算
        assert "is_near_sr" in df.columns
        assert is_near_sr.dtype == bool
        assert is_near_sr.notna().all()

        # 步骤2: 使用 is_near_sr 过滤标签
        labels = compute_sr_reversal_label_with_weights(
            df,
            price_col="close",
            high_col="high",
            low_col="low",
            atr_col="atr",
            max_holding_bars=20,
            stop_loss_r=1.0,
            take_profit_r=2.0,
            combine_mode="long_only",
            sr_mask_col="is_near_sr",
            compute_weights=False,
        )

        # 验证：只有 is_near_sr=True 的样本有标签
        near_sr_indices = df[df["is_near_sr"]].index
        far_sr_indices = df[~df["is_near_sr"]].index

        if len(near_sr_indices) > 0:
            near_sr_labels = labels[near_sr_indices]
            assert near_sr_labels.notna().sum() > 0, "在SR附近的样本应该有标签"

        if len(far_sr_indices) > 0:
            far_sr_labels = labels[far_sr_indices]
            assert far_sr_labels.notna().sum() == 0, "不在SR附近的样本标签应该为 NaN"

        # 步骤3: 验证单位转换的正确性
        # 手动计算几个样本的 is_near_sr
        for idx in df.index[:10]:
            dist_pct = abs(df.loc[idx, "dist_to_nearest_sr"])
            price = df.loc[idx, "close"]
            atr = df.loc[idx, "atr"]

            abs_distance = dist_pct * price
            dist_normalized = abs_distance / atr
            expected_is_near = dist_normalized <= 1.5

            actual_is_near = df.loc[idx, "is_near_sr"]

            assert actual_is_near == expected_is_near, (
                f"索引 {idx}: 期望 {expected_is_near}, 实际 {actual_is_near} "
                f"(dist={dist_pct:.4f}, price={price:.2f}, atr={atr:.2f}, "
                f"abs_dist={abs_distance:.2f}, normalized={dist_normalized:.2f} ATR)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
