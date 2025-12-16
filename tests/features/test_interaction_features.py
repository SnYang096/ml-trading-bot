"""
交互特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 交互特征数学正确性验证
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

from src.features.time_series.utils_interaction_features import (
    extract_interaction_features,
    compute_liquidity_void_x_wpt_risk,
    compute_liquidity_void_x_vpin_from_series,
    compute_vpin_x_compression_from_series,
    apply_rank_transform_to_interaction_from_series,
    compute_compression_energy_x_ofi_short,
    compute_vpin_x_compression,
    compute_sr_distance_normalized,
    compute_cvd_slope,
)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成其他数据
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    volume = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame(
        {
            "close": prices,
            "high": high,
            "low": low,
            "volume": volume,
        },
        index=dates,
    )

    # 添加一些基础特征（交互特征需要这些）
    df["liquidity_void_detected"] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    df["wpt_false_breakout_risk"] = np.random.uniform(0, 1, n_samples)
    df["compression_energy"] = np.random.uniform(0, 1, n_samples)
    df["ofi_short"] = np.random.uniform(-1, 1, n_samples)
    df["vpin"] = np.random.uniform(0, 1, n_samples)
    df["vpin_zscore_50"] = np.random.uniform(-2, 2, n_samples)
    df["dist_to_nearest_sr"] = np.random.uniform(0, 10, n_samples)

    # 计算 ATR
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean()
    df["atr"] = df["atr"].fillna(1.0).clip(lower=1e-6)

    df["cvd"] = np.cumsum(np.random.randn(n_samples) * 100)

    return df


class TestInteractionFeatures:
    """交互特征测试类"""

    def test_basic_interaction_features(self):
        """测试：基础交互特征计算"""
        df = create_mock_data(n_samples=200)

        # 测试单个交互特征
        result = compute_liquidity_void_x_wpt_risk(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.notna().sum() > 0

        # 测试另一个交互特征
        result2 = compute_compression_energy_x_ofi_short(df)
        assert isinstance(result2, pd.Series)
        assert len(result2) == len(df)

        # 测试 heavy gate: liquidity_void_x_vpin（Series-in narrow）
        out_df = compute_liquidity_void_x_vpin_from_series(
            liquidity_void_detected=df["liquidity_void_detected"],
            vpin=df["vpin_zscore_50"],
        )
        assert isinstance(out_df, pd.DataFrame)
        assert "liquidity_void_x_vpin" in out_df.columns
        expected = df["liquidity_void_detected"].fillna(0.0) * df[
            "vpin_zscore_50"
        ].fillna(0.0)
        assert np.allclose(
            out_df["liquidity_void_x_vpin"].values,
            expected.values,
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )

        # 测试：vpin_x_compression（Series-in narrow）
        out2 = compute_vpin_x_compression_from_series(
            vpin=df["vpin"],
            compression_energy=df["compression_energy"],
        )
        assert "vpin_x_compression" in out2.columns
        expected2 = df["vpin"].fillna(0.0) * df["compression_energy"].fillna(0.0)
        assert np.allclose(
            out2["vpin_x_compression"].values, expected2.values, equal_nan=True
        )

        # 测试：rank transform（Series-in narrow）
        rank_df = apply_rank_transform_to_interaction_from_series(
            interaction=out2["vpin_x_compression"]
        )
        assert rank_df.shape[1] == 1
        assert rank_df.iloc[:, 0].between(0.0, 1.0).all()

    def test_extract_interaction_features(self):
        """测试：提取交互特征"""
        df = create_mock_data(n_samples=200)

        result = extract_interaction_features(df, apply_rank=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # 检查是否有交互特征列（包含 _x_）
        interaction_cols = [col for col in result.columns if "_x_" in col]
        assert len(interaction_cols) > 0, "应该有交互特征列"

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(n_samples=500, seed=42)

        # 计算第一次特征
        result1 = extract_interaction_features(df, apply_rank=False)
        # 选择一个交互特征列
        interaction_cols = [col for col in result1.columns if "_x_" in col]
        if len(interaction_cols) > 0:
            feature_col = interaction_cols[0]
            feature_1 = result1[feature_col].copy()

            # 修改未来数据（从 t=250 开始）
            df_future_modified = df.copy()
            df_future_modified.loc[df_future_modified.index[250] :, "close"] *= 2.0
            df_future_modified.loc[df_future_modified.index[250] :, "vpin"] *= 2.0
            df_future_modified.loc[
                df_future_modified.index[250] :, "compression_energy"
            ] *= 2.0

            # 重新计算特征
            result2 = extract_interaction_features(df_future_modified, apply_rank=False)
            feature_2 = result2[feature_col].copy()

            # 检查前 200 个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:200]
            feat_1_check = feature_1.loc[check_idx].dropna()
            feat_2_check = feature_2.loc[check_idx].dropna()

            if len(feat_1_check) > 0 and len(feat_2_check) > 0:
                diff = (feat_1_check - feat_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，特征分布应该对齐
        - 特征值应该在相似范围内，便于多资产训练
        """
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")

        # 创建不同价格水平的资产
        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        results = []
        for symbol, prices in assets.items():
            high = prices * (1 + np.abs(np.random.randn(n) * 0.005))
            low = prices * (1 - np.abs(np.random.randn(n) * 0.005))

            df = pd.DataFrame(
                {
                    "close": prices,
                    "high": high,
                    "low": low,
                    "volume": np.random.lognormal(10, 1, n),
                },
                index=dates,
            )

            # 添加基础特征
            df["liquidity_void_detected"] = np.random.choice([0, 1], n, p=[0.9, 0.1])
            df["wpt_false_breakout_risk"] = np.random.uniform(0, 1, n)
            df["compression_energy"] = np.random.uniform(0, 1, n)
            df["vpin"] = np.random.uniform(0, 1, n)
            df["dist_to_nearest_sr"] = np.random.uniform(0, 10, n)

            # 计算 ATR
            tr = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df["atr"] = tr.rolling(window=14, min_periods=1).mean()
            df["atr"] = df["atr"].fillna(1.0).clip(lower=1e-6)

            # 计算特征
            result = extract_interaction_features(df, apply_rank=False)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查不同资产的特征分布
        interaction_cols = [col for col in combined.columns if "_x_" in col]
        if len(interaction_cols) > 0:
            col = interaction_cols[0]
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])

                # 检查均值范围
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()

                # 交互特征应该对不同资产的价格水平不敏感（如果归一化正确）
                # 允许一定的差异，因为不同资产的基础特征可能不同
                assert mean_range < 10.0, (
                    f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(n_samples=300, seed=42)
        window = 50  # 交互特征通常不需要大的窗口

        # 批量计算（一次性计算所有数据）
        batch_result = extract_interaction_features(df, apply_rank=False)

        # 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_interaction_features(df_stream, apply_rank=False)
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            interaction_cols = [col for col in batch_result.columns if "_x_" in col]
            if len(interaction_cols) > 0:
                key_col = interaction_cols[0]
                if key_col in batch_result.columns and key_col in streaming_df.columns:
                    batch_vals = batch_result[key_col].iloc[window:].dropna()
                    stream_vals = streaming_df[key_col].dropna()

                    # 找到共同索引
                    common_idx = batch_vals.index.intersection(stream_vals.index)
                    if len(common_idx) > 10:  # 至少需要10个数据点
                        diff = (
                            batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                        ).abs()
                        max_diff = diff.max()
                        mean_diff = diff.mean()

                        # 允许一定的数值误差（由于滚动窗口计算的微小差异）
                        assert max_diff < 1e-5, (
                            f"流式与批量计算不一致 ({key_col})，最大差异: {max_diff:.8f}, "
                            f"平均差异: {mean_diff:.8f}"
                        )

    def test_interaction_math_correctness(self):
        """测试：交互特征数学正确性"""
        df = create_mock_data(n_samples=200)

        # 测试乘积交互特征
        result = compute_liquidity_void_x_wpt_risk(df)
        # 验证：result = liquidity_void_detected * wpt_false_breakout_risk
        expected = df["liquidity_void_detected"] * df["wpt_false_breakout_risk"]
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

        # 测试比率特征
        if "dist_to_nearest_sr" in df.columns and "atr" in df.columns:
            result2 = compute_sr_distance_normalized(df)
            # 验证：result = dist_to_nearest_sr / atr
            expected2 = df["dist_to_nearest_sr"] / df["atr"]
            pd.testing.assert_series_equal(
                result2, expected2, check_names=False, rtol=1e-10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
