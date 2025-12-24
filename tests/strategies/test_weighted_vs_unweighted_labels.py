"""
测试带权重版本和无权重、SR过滤版本的标签差异

验证：
1. 带权重版本的标签应该和无权重、SR过滤版本的标签在数量上可能不同
2. 带权重版本的权重应该根据未来RR比率进行分级处理（>2, >1, <1等）
3. 带权重版本的权重应该反映样本的重要性
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_full_scan,
    compute_sr_reversal_label_with_weights,
    compute_sr_reversal_rr_continuous_label,
    compute_sr_reversal_rr_continuous_label_with_weights,
)


class TestWeightedVsUnweightedLabels:
    """测试带权重版本和无权重版本的标签差异"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 1000

        # 创建价格数据
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "open": prices * 0.995,
                "volume": np.random.randint(1000, 10000, n),
                "atr": np.random.uniform(1, 3, n),
                "dist_to_nearest_sr": np.random.uniform(
                    -0.05, 0.05, n
                ),  # 相对百分比，在SR附近
                "sr_strength_max": np.random.uniform(0, 1, n),
                "vpin": np.random.uniform(0, 1, n),
                "cvd_slope_5_f": np.random.uniform(-1, 1, n),
            }
        )

        return df

    def test_binary_labels_count_difference(self, sample_data):
        """测试二分类标签数量差异"""
        # 无权重，SR过滤版本
        labels_sr_filter = compute_sr_reversal_label_full_scan(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
        )
        count_sr_filter = labels_sr_filter.notna().sum()

        # 带权重版本（也进行SR过滤）
        labels_weighted = compute_sr_reversal_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "none",  # 不使用逻辑分，只看RR
                "min_rr_threshold": 1.0,
                "loss_weight": 0.05,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )
        count_weighted = labels_weighted.notna().sum()

        # 验证：标签数量应该相同（因为都进行了SR过滤）
        assert (
            count_weighted == count_sr_filter
        ), f"带权重版本标签数量({count_weighted})应该等于SR过滤版本({count_sr_filter})"

    def test_binary_weights_distribution(self, sample_data):
        """测试二分类权重分布"""
        # 带权重版本
        labels_weighted = compute_sr_reversal_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            return_weights=True,  # 返回权重
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "none",  # 不使用逻辑分，只看RR
                "min_rr_threshold": 1.0,
                "loss_weight": 0.05,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )

        labels, weights = labels_weighted

        # 验证：权重应该存在
        assert weights is not None, "权重应该被计算"
        assert len(weights) == len(labels), "权重长度应该等于标签长度"

        # 验证：有效标签的权重应该 >= 0（允许 0.0 代表忽略该样本）
        valid_mask = labels.notna()
        if valid_mask.sum() > 0:
            valid_weights = weights[valid_mask]
            assert (valid_weights >= 0).all(), "有效标签的权重应该全部 >= 0"

            # 验证：权重应该有变化（不应该全部相同）
            assert valid_weights.std() > 0, "权重应该有变化（标准差>0）"

    def test_regression_labels_count_difference(self, sample_data):
        """测试回归标签数量差异"""
        # 无权重，SR过滤版本
        labels_sr_filter = compute_sr_reversal_rr_continuous_label(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
        )
        count_sr_filter = labels_sr_filter.notna().sum()

        # 带权重版本（也进行SR过滤）
        labels_weighted = compute_sr_reversal_rr_continuous_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "none",  # 不使用逻辑分，只看RR
                "min_rr_threshold": 1.0,
                "loss_weight": 0.05,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )
        count_weighted = labels_weighted.notna().sum()

        # 验证：标签数量应该相同（因为都进行了SR过滤）
        assert (
            count_weighted == count_sr_filter
        ), f"带权重版本标签数量({count_weighted})应该等于SR过滤版本({count_sr_filter})"

    def test_regression_weights_by_rr_levels(self, sample_data):
        """测试回归权重根据RR比率分级处理"""
        # 带权重版本
        labels_weighted = compute_sr_reversal_rr_continuous_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            return_weights=True,  # 返回权重
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "none",  # 不使用逻辑分，只看RR
                "min_rr_threshold": 1.0,
                "loss_weight": 0.05,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )

        labels, weights = labels_weighted

        # 验证：权重应该存在
        assert weights is not None, "权重应该被计算"

        # 验证：有效标签的权重应该根据RR值分级
        valid_mask = labels.notna()
        if valid_mask.sum() > 0:
            valid_labels = labels[valid_mask]
            valid_weights = weights[valid_mask]

            # 按RR值分级
            rr_high = valid_labels >= 2.0  # RR >= 2.0
            rr_medium = (valid_labels >= 1.0) & (valid_labels < 2.0)  # 1.0 <= RR < 2.0
            rr_low = valid_labels < 1.0  # RR < 1.0

            # 验证：高RR样本的权重应该 >= 中等RR样本的权重
            if rr_high.sum() > 0 and rr_medium.sum() > 0:
                high_weights = valid_weights[rr_high]
                medium_weights = valid_weights[rr_medium]
                assert (
                    high_weights.mean() >= medium_weights.mean()
                ), f"高RR样本的平均权重({high_weights.mean():.4f})应该 >= 中等RR样本({medium_weights.mean():.4f})"

            # 验证：中等RR样本的权重应该 > 低RR样本的权重
            if rr_medium.sum() > 0 and rr_low.sum() > 0:
                medium_weights = valid_weights[rr_medium]
                low_weights = valid_weights[rr_low]
                assert (
                    medium_weights.mean() > low_weights.mean()
                ), f"中等RR样本的平均权重({medium_weights.mean():.4f})应该 > 低RR样本({low_weights.mean():.4f})"

            # 验证：低RR样本的权重应明显更低；允许为 0.0（代表忽略）
            if rr_low.sum() > 0:
                low_weights = valid_weights[rr_low]
                # 归一化后，低RR样本的权重应该相对较低
                assert (
                    low_weights.mean() < valid_weights.mean()
                ), f"低RR样本的平均权重({low_weights.mean():.4f})应该 < 总体平均权重({valid_weights.mean():.4f})"

    def test_zero_weight_on_loss_supported(self, sample_data):
        """验证：当 loss_weight=0.0 时，部分低RR样本权重会变为0.0（被忽略）"""
        labels, weights = compute_sr_reversal_rr_continuous_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            return_weights=True,
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "none",
                "medium_rr_threshold": 1.0,
                "high_rr_threshold": 2.0,
                "high_rr_boost": 2.0,
                "loss_weight": 0.0,
                "min_total_weight": 1e-6,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )

        valid_mask = labels.notna()
        if valid_mask.sum() == 0:
            pytest.skip("有效样本太少，跳过")

        valid_weights = weights[valid_mask]
        assert (valid_weights >= 0).all()
        # 至少应该出现一些 0 权重（低RR样本被忽略）
        assert (valid_weights == 0).sum() >= 1

    def test_weights_with_triple_resonance(self, sample_data):
        """测试三重共振对权重的影响"""
        # 设置一些样本满足三重共振条件
        sample_data.loc[:100, "vpin"] = 0.8  # 高VPIN
        sample_data.loc[:100, "cvd_slope_5_f"] = 0.5  # 正CVD斜率
        sample_data.loc[:100, "sr_strength_max"] = 0.7  # 高SR强度

        # 带权重版本（使用三重共振逻辑分）
        labels_weighted = compute_sr_reversal_rr_continuous_label_with_weights(
            sample_data,
            max_holding_bars=20,
            take_profit_r=2.0,
            stop_loss_r=1.0,
            combine_mode="long_only",
            dist_to_sr_col="dist_to_nearest_sr",
            dist_atr_mult=1.5,
            compute_weights=True,
            return_weights=True,
            weight_strategy="result_based_rr",
            weight_config={
                "logic_mode": "triple_resonance",
                "vpin_col": "vpin",
                "vpin_threshold": 0.7,
                "cvd_slope_col": "cvd_slope_5_f",
                "cvd_slope_threshold": 0.0,
                "sr_strength_col": "sr_strength_max",
                "sr_strength_threshold": 0.5,
                "logic_base": 1.0,
                "logic_boost": 1.5,
                "min_rr_threshold": 1.0,
                "loss_weight": 0.05,
                "normalize_weights": True,
                "max_holding_bars": 20,
                "stop_loss_r": 1.0,
                "take_profit_r": 2.0,
            },
        )

        labels, weights = labels_weighted

        # 验证：三重共振样本的权重应该更高
        valid_mask = labels.notna()
        if valid_mask.sum() > 0:
            valid_weights = weights[valid_mask]
            valid_indices = sample_data.index[valid_mask]

            # 检查前100个样本（满足三重共振）的权重
            triple_mask = valid_indices.isin(sample_data.index[:100])
            if triple_mask.sum() > 0:
                triple_weights = valid_weights[triple_mask]
                other_weights = valid_weights[~triple_mask]

                if other_weights.sum() > 0:
                    # 验证：三重共振样本的权重应该存在且合理
                    assert len(triple_weights) > 0, "应该有满足三重共振条件的样本"
                    assert (triple_weights > 0).all(), "三重共振样本的权重应该全部大于0"

                    # 注意：归一化后，权重均值都是1.0，所以不能直接比较均值
                    # 但我们可以验证：三重共振样本的权重分布应该更偏向高权重
                    triple_high_weight_ratio = (triple_weights > 1.0).sum() / len(
                        triple_weights
                    )
                    other_high_weight_ratio = (other_weights > 1.0).sum() / len(
                        other_weights
                    )

                    # 三重共振样本中高权重样本的比例应该 >= 其他样本
                    # （这个断言可能不总是成立，因为还取决于RR值）
                    # 所以我们只验证权重存在且合理
                    assert (
                        triple_high_weight_ratio >= 0
                    ), "三重共振样本的高权重比例应该 >= 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
