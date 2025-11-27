"""
高级特征测试（GARCH, DTW, EVT）

测试内容：
1. 归一化验证（多资产训练兼容性）
2. 实现合理性验证
3. 性能测试
4. 公共代码抽取验证
"""

import numpy as np
import pandas as pd
import pytest
import time
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# 导入特征提取函数
from src.features.time_series.utils_garch_features import extract_garch_features
from src.features.time_series.utils_dtw_features import (
    extract_dtw_features,
    create_dtw_templates,
)
from src.features.time_series.utils_evt_features import extract_evt_features

# 导入公共归一化模块
from src.features.time_series.utils_normalization import (
    normalize_series,
    normalize_by_group,
)


class TestGARCHFeatures:
    """GARCH特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 200

        # 不同价格水平的资产
        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_garch_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_garch_features(df, price_col="close", window=60)

        # 检查输出列
        expected_cols = [
            "garch_volatility",
            "garch_persistence",
            "garch_leverage_gamma",
            "garch_alpha",
            "garch_beta",
        ]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查数值合理性
        assert result["garch_persistence"].notna().sum() > 0
        assert (result["garch_persistence"] >= 0).all()
        assert (result["garch_persistence"] <= 1.5).all()  # 通常 < 1.0，允许一些误差

        # 波动率应该非负
        valid_vol = result["garch_volatility"].dropna()
        if len(valid_vol) > 0:
            assert (valid_vol >= 0).all()

    def test_garch_features_normalization_multi_asset(self, sample_data_multi_asset):
        """多资产归一化测试"""
        # 按资产分组计算特征
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_garch_features(df_asset, price_col="close", window=60)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的特征值应该在相似范围内（归一化后）
        # 由于GARCH特征已经是相对值（基于收益率），应该天然归一化
        for col in ["garch_persistence", "garch_leverage_gamma"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                # 检查不同资产的特征分布是否相似
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 均值应该在合理范围内（例如persistence在0.7-0.99之间）
                assert (by_symbol["mean"] >= 0).all()
                assert (by_symbol["mean"] <= 1.5).all()

    def test_garch_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_garch_features(df, price_col="close", window=60)
        elapsed = time.time() - start_time

        # 200个样本应该在合理时间内完成（< 10秒）
        assert elapsed < 10.0, f"GARCH特征计算耗时 {elapsed:.2f}秒，超过10秒"

        # 检查是否有有效结果
        assert result["garch_persistence"].notna().sum() > 0

    def test_garch_features_edge_cases(self):
        """边界情况测试"""
        # 数据不足
        df_short = pd.DataFrame({"close": [100, 101, 102]})
        result = extract_garch_features(df_short, price_col="close", window=60)
        assert len(result) == len(df_short)
        # 应该返回NaN或默认值
        assert (
            result["garch_persistence"].isna().all()
            or (result["garch_persistence"] == 0.0).all()
        )

        # 常数价格
        df_constant = pd.DataFrame({"close": [100.0] * 100})
        result = extract_garch_features(df_constant, price_col="close", window=60)
        assert len(result) == len(df_constant)


class TestDTWFeatures:
    """DTW特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 100

        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_dtw_normalize_series(self):
        """测试归一化函数（公共代码）"""
        # 测试Z-score归一化
        x1 = np.array([100, 101, 102, 103, 104])
        x2 = np.array([50000, 50100, 50200, 50300, 50400])

        norm1 = normalize_series(x1)
        norm2 = normalize_series(x2)

        # 归一化后应该有相似的分布
        assert np.abs(norm1.mean()) < 1e-10  # 均值应该接近0
        assert np.abs(norm1.std() - 1.0) < 1e-10  # 标准差应该接近1
        assert np.abs(norm2.mean()) < 1e-10
        assert np.abs(norm2.std() - 1.0) < 1e-10

        # 常数序列
        x_const = np.array([100.0] * 10)
        norm_const = normalize_series(x_const)
        assert np.allclose(norm_const, 0.0)

    def test_dtw_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_dtw_features(df, price_col="close", window=20)

        # 检查输出列
        templates = create_dtw_templates()
        expected_cols = [f"dtw_{name}_dist" for name in templates.keys()]
        expected_cols.extend(["dtw_min_dist", "dtw_best_match"])

        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查距离值合理性（应该非负）
        for col in expected_cols:
            if col != "dtw_best_match":
                valid_data = result[col].dropna()
                if len(valid_data) > 0:
                    assert (valid_data >= 0).all()

    def test_dtw_features_normalization_multi_asset(self, sample_data_multi_asset):
        """多资产归一化测试"""
        # DTW特征应该对价格水平不敏感（因为内部做了归一化）
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_dtw_features(df_asset, price_col="close", window=20)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的DTW距离分布应该相似（因为价格已归一化）
        for col in ["dtw_min_dist", "dtw_hammer_dist"]:
            if col in combined.columns:
                valid_data = combined[col].dropna()
                if len(valid_data) > 0:
                    by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                    # 不同资产的距离均值应该在相似范围内
                    mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                    # 如果归一化正确，不同资产的均值差异不应该太大
                    # （允许一些差异，因为价格走势不同）
                    assert mean_range < 10.0  # 合理的阈值

    def test_dtw_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_dtw_features(df, price_col="close", window=20)
        elapsed = time.time() - start_time

        # 100个样本应该在合理时间内完成（< 5秒）
        assert elapsed < 5.0, f"DTW特征计算耗时 {elapsed:.2f}秒，超过5秒"

        # 检查是否有有效结果
        assert result["dtw_min_dist"].notna().sum() > 0

    def test_dtw_templates_consistency(self):
        """测试模板一致性"""
        templates = create_dtw_templates()

        # 所有模板应该有相同的长度（20）
        for name, template in templates.items():
            assert len(template) == 20, f"模板 {name} 长度不是20"
            # 模板值应该在合理范围内
            assert template.min() >= 0.0
            assert template.max() <= 1.0


class TestEVTFeatures:
    """EVT特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 200

        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_evt_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_evt_features(df, price_col="close", window=120)

        # 检查输出列
        expected_cols = [
            "evt_tail_shape",
            "evt_tail_shape_left",
            "evt_tail_shape_right",
            "evt_scale",
            "evt_var_99",
            "evt_es_99",
        ]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查数值合理性
        # ξ 通常在 -0.5 到 1.0 之间
        valid_xi = result["evt_tail_shape"].dropna()
        if len(valid_xi) > 0:
            assert (valid_xi >= -1.0).all()  # 允许一些极端值
            assert (valid_xi <= 2.0).all()  # 允许一些极端值

        # VaR和ES应该是负数（损失）
        valid_var = result["evt_var_99"].dropna()
        if len(valid_var) > 0:
            assert (valid_var <= 0).all()  # VaR应该是负数

    def test_evt_features_normalization_multi_asset(self, sample_data_multi_asset):
        """多资产归一化测试"""
        # EVT特征基于收益率，应该天然归一化
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_evt_features(df_asset, price_col="close", window=120)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的ξ值应该在相似范围内
        for col in ["evt_tail_shape", "evt_tail_shape_left", "evt_tail_shape_right"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 均值应该在合理范围内（例如-0.5到1.0）
                assert (by_symbol["mean"] >= -1.0).all()
                assert (by_symbol["mean"] <= 2.0).all()

    def test_evt_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_evt_features(df, price_col="close", window=120)
        elapsed = time.time() - start_time

        # 200个样本应该在合理时间内完成（< 10秒）
        assert elapsed < 10.0, f"EVT特征计算耗时 {elapsed:.2f}秒，超过10秒"

        # 检查是否有有效结果
        assert result["evt_tail_shape"].notna().sum() > 0

    def test_evt_features_edge_cases(self):
        """边界情况测试"""
        # 数据不足
        df_short = pd.DataFrame({"close": [100, 101, 102]})
        result = extract_evt_features(df_short, price_col="close", window=120)
        assert len(result) == len(df_short)
        # 应该返回默认值
        assert (result["evt_tail_shape"] == 0.3).all()  # 默认安全值

        # 常数价格
        df_constant = pd.DataFrame({"close": [100.0] * 100})
        result = extract_evt_features(df_constant, price_col="close", window=120)
        assert len(result) == len(df_constant)


class TestAdvancedFeaturesIntegration:
    """高级特征集成测试"""

    def test_all_features_together(self):
        """测试所有高级特征一起使用"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(n) * 0.5)})

        # 计算所有特征
        garch_result = extract_garch_features(df, price_col="close", window=60)
        dtw_result = extract_dtw_features(df, price_col="close", window=20)
        evt_result = extract_evt_features(df, price_col="close", window=120)

        # 合并结果
        combined = pd.concat([garch_result, dtw_result, evt_result], axis=1)

        # 检查没有重复列
        assert len(combined.columns) == len(set(combined.columns))

        # 检查所有特征都有值（至少部分）
        for col in combined.columns:
            if col != "dtw_best_match":
                assert combined[col].notna().sum() > 0 or (combined[col] == 0.0).all()

    def test_features_with_missing_data(self):
        """测试缺失数据处理"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # 添加一些NaN
        prices[50:60] = np.nan
        prices[150:155] = np.nan

        df = pd.DataFrame({"close": prices})

        # 应该能正常处理
        garch_result = extract_garch_features(df, price_col="close", window=60)
        dtw_result = extract_dtw_features(df, price_col="close", window=20)
        evt_result = extract_evt_features(df, price_col="close", window=120)

        # 检查结果长度
        assert len(garch_result) == len(df)
        assert len(dtw_result) == len(df)
        assert len(evt_result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
