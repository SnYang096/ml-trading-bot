"""
BPC (Breakout-Pullback-Continuation) 软阶段特征测试

测试内容：
1. 未来数据泄露验证（确保特征不使用未来信息）⭐⭐⭐⭐⭐
2. 流式 vs 批量一致性测试 ⭐⭐⭐⭐
3. 特征值范围和正确性测试 ⭐⭐⭐
4. 多资产归一化测试 ⭐⭐⭐⭐

参考规范：
- docs/tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md
- BPC特征三层输出架构规范
"""

import sys
import numpy as np
import pandas as pd
import pytest

from src.features.time_series.bpc_features import (
    compute_bpc_soft_phase_from_series,
    compute_bpc_pullback_depth_pct_from_series,
    compute_bpc_pullback_duration_from_series,
    compute_bpc_impulse_return_atr_from_series,
    compute_bpc_dir_consistency_multi_from_series,
    compute_bpc_breakout_context_from_series,
    compute_bpc_pullback_structure_from_series,
    compute_bpc_continuation_target_from_series,
    compute_bpc_compression_state_from_series,
    compute_bpc_phase_transition_from_series,
    FEATURE_VERSION,
    DEFAULT_LOOKBACK_BREAKOUT,
)

# =============================================================================
# 📊 测试数据生成器
# =============================================================================


def create_ohlcv_data(
    n_samples: int = 500,
    base_price: float = 50000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    创建模拟 OHLCV 数据

    Args:
        n_samples: 样本数量
        base_price: 基础价格水平
        volatility: 波动率
        seed: 随机种子

    Returns:
        包含 open, high, low, close, volume, atr 的 DataFrame
    """
    np.random.seed(seed)

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_samples, freq="5min")

    # 生成价格序列（随机游走 + 趋势）
    returns = np.random.randn(n_samples) * volatility
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # 生成 OHLC
    high_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    low_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    open_offset = np.random.randn(n_samples) * volatility * base_price * 0.5

    high = close + high_spread
    low = close - low_spread
    open_price = close + open_offset

    # 确保 OHLC 关系正确
    high = np.maximum(high, np.maximum(close, open_price))
    low = np.minimum(low, np.minimum(close, open_price))

    # 生成成交量
    volume = np.random.uniform(100, 1000, n_samples)

    # 计算 ATR
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]  # 第一个值
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
        },
        index=timestamps,
    )

    return df


def create_orderflow_data(
    df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    创建模拟订单流数据

    Args:
        df: OHLCV DataFrame
        seed: 随机种子

    Returns:
        包含 cvd_change_5, vpin, ofci_pct 的字典
    """
    np.random.seed(seed)
    n = len(df)

    # CVD 变化（与价格方向相关）
    price_dir = np.sign(df["close"].diff().fillna(0).values)
    cvd_change_5 = (
        price_dir * np.abs(np.random.randn(n)) * 100 + np.random.randn(n) * 30
    )

    # VPIN（0-1 范围）
    vpin = np.clip(np.random.uniform(0.3, 0.8, n), 0, 1)

    # OFCI 百分位（0-1 范围）
    ofci_pct = np.clip(np.random.uniform(0.2, 0.8, n), 0, 1)

    # BB 宽度归一化
    bb_width_normalized = np.clip(np.random.uniform(0.2, 0.6, n), 0, 1)

    return {
        "cvd_change_5": pd.Series(cvd_change_5, index=df.index),
        "vpin": pd.Series(vpin, index=df.index),
        "ofci_pct": pd.Series(ofci_pct, index=df.index),
        "bb_width_normalized": pd.Series(bb_width_normalized, index=df.index),
    }


# =============================================================================
# 📋 测试类：未来函数检测
# =============================================================================


class TestBPCFeaturesNoFutureLeak:
    """
    BPC 特征无未来函数测试（No Lookahead Bias）⭐⭐⭐⭐⭐

    验证方法：修改未来数据，确认历史特征值不变
    """

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_bpc_soft_phase_no_future_leak(self, sample_data):
        """
        测试 BPC 软阶段分数：修改未来数据不应影响历史特征值

        关键验证：
        - 修改 t=300 之后的数据
        - t=250 之前的所有特征值应该完全相同
        """
        print("\n" + "=" * 70)
        print("测试：BPC 软阶段分数无未来函数 (No Future Leak)")
        print("=" * 70)

        df, orderflow = sample_data

        # 1. 计算第一次特征（原始数据）
        result1 = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
            bb_width_normalized=orderflow["bb_width_normalized"],
        )

        # 2. 修改未来数据（t=300 之后）
        df_future = df.copy()
        future_idx = df_future.index[300:]
        df_future.loc[future_idx, "close"] *= 2.0  # 价格翻倍
        df_future.loc[future_idx, "high"] *= 2.0
        df_future.loc[future_idx, "low"] *= 2.0
        df_future.loc[future_idx, "volume"] *= 3.0  # 成交量翻三倍

        orderflow_future = {k: v.copy() for k, v in orderflow.items()}
        orderflow_future["cvd_change_5"].loc[future_idx] *= 5.0

        # 3. 重新计算特征
        result2 = compute_bpc_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            atr=df_future["atr"],
            volume=df_future["volume"],
            cvd_change_5=orderflow_future["cvd_change_5"],
            vpin=orderflow_future["vpin"],
            bb_width_normalized=orderflow_future["bb_width_normalized"],
        )

        # 4. 验证历史特征值完全相同（t=250 之前，留出足够的回看窗口）
        check_idx = df.index[:250]

        # 检查所有输出特征
        key_features = [
            "bpc_score_breakout",
            "bpc_score_pullback",
            "bpc_score_continuation",
            "bpc_score_neutral",
            "bpc_price_breakout_strength",
            "bpc_pullback_depth",
            "bpc_direction_confidence",
        ]

        for feat in key_features:
            if feat in result1.columns and feat in result2.columns:
                vals1 = result1.loc[check_idx, feat].dropna()
                vals2 = result2.loc[check_idx, feat].dropna()

                common_idx = vals1.index.intersection(vals2.index)
                if len(common_idx) > 0:
                    diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs()
                    max_diff = diff.max()

                    assert max_diff < 1e-6, (
                        f"未来数据变化影响了历史 {feat} 值，"
                        f"最大差异: {max_diff:.8f}"
                    )

        print("  ✅ 所有特征通过无未来函数验证")

    def test_bpc_rolling_window_no_lookahead(self, sample_data):
        """
        测试滚动窗口计算不使用未来数据

        验证：
        - 在 t 时刻的特征只依赖 [t-lookback, t-1] 的数据
        - 不使用 t 及之后的数据
        """
        print("\n" + "=" * 70)
        print("测试：BPC 滚动窗口无 Lookahead")
        print("=" * 70)

        df, orderflow = sample_data

        # 在 t=200 处制造价格突变
        df_shock = df.copy()
        shock_idx = df_shock.index[200]
        df_shock.loc[shock_idx, "close"] *= 1.5  # 50% 跳涨
        df_shock.loc[shock_idx, "high"] *= 1.5

        # 计算特征
        result = compute_bpc_soft_phase_from_series(
            close=df_shock["close"],
            high=df_shock["high"],
            low=df_shock["low"],
            atr=df_shock["atr"],
            volume=df_shock["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # t=199 的特征应该只基于 t=179-198 的数据（lookback=20）
        # 不应该包含 t=200 的跳涨信息
        feat_199 = result.loc[df.index[199], "bpc_score_breakout"]

        # 用原始数据计算 t=199 的特征
        result_original = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )
        feat_199_original = result_original.loc[df.index[199], "bpc_score_breakout"]

        # 两者应该相同（因为 t=199 的计算不应该使用 t=200 的数据）
        diff = abs(feat_199 - feat_199_original)
        assert diff < 1e-6, f"t=199 的特征受到 t=200 数据影响，差异: {diff:.8f}"

        print("  ✅ 滚动窗口计算验证通过")


# =============================================================================
# 📋 测试类：流式 vs 批量一致性
# =============================================================================


class TestBPCFeaturesStreamingVsBatch:
    """
    BPC 特征流式 vs 批量一致性测试 ⭐⭐⭐⭐

    对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
    """

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_streaming_vs_batch_consistency(self, sample_data):
        """
        测试：流式计算与批量计算应该一致

        方法：
        1. 批量计算：一次性处理所有数据
        2. 流式计算：分块处理，每块带足够的 warmup 窗口
        3. 比较相同时间戳的特征值
        """
        print("\n" + "=" * 70)
        print("测试：BPC 流式 vs 批量一致性")
        print("=" * 70)

        df, orderflow = sample_data

        # 1. 批量计算（一次性处理所有数据）
        batch_result = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
            bb_width_normalized=orderflow["bb_width_normalized"],
        )

        # 2. 流式计算（分块处理，带 warmup）
        chunk_size = 100
        warmup_size = DEFAULT_LOOKBACK_BREAKOUT * 2  # 确保足够的预热数据
        streaming_results = []

        for i in range(warmup_size, len(df), chunk_size):
            # 包含 warmup 窗口的数据
            start_idx = max(0, i - warmup_size)
            end_idx = min(i + chunk_size, len(df))

            chunk_df = df.iloc[start_idx:end_idx]
            chunk_orderflow = {
                k: v.iloc[start_idx:end_idx] for k, v in orderflow.items()
            }

            # 计算当前块的特征
            chunk_result = compute_bpc_soft_phase_from_series(
                close=chunk_df["close"],
                high=chunk_df["high"],
                low=chunk_df["low"],
                atr=chunk_df["atr"],
                volume=chunk_df["volume"],
                cvd_change_5=chunk_orderflow["cvd_change_5"],
                vpin=chunk_orderflow["vpin"],
                bb_width_normalized=chunk_orderflow["bb_width_normalized"],
            )

            # 只保留非 warmup 部分的结果
            actual_start = i - start_idx
            actual_result = chunk_result.iloc[actual_start:]
            streaming_results.append(actual_result)

        # 3. 合并流式结果
        if streaming_results:
            streaming_combined = pd.concat(streaming_results, axis=0)
            # 去重（可能有重叠）
            streaming_combined = streaming_combined[
                ~streaming_combined.index.duplicated(keep="first")
            ]
        else:
            streaming_combined = pd.DataFrame()

        # 4. 比较关键特征
        key_features = [
            "bpc_score_breakout",
            "bpc_score_pullback",
            "bpc_score_continuation",
            "bpc_score_neutral",
        ]

        for feat in key_features:
            if feat in batch_result.columns and feat in streaming_combined.columns:
                batch_vals = batch_result[feat].dropna()
                stream_vals = streaming_combined[feat].dropna()

                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 10:
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()
                    mean_diff = diff.mean()

                    print(f"  {feat}:")
                    print(f"    共同索引数: {len(common_idx)}")
                    print(f"    最大差异: {max_diff:.8f}")
                    print(f"    平均差异: {mean_diff:.8f}")

                    # 允许一定的数值误差（由于边界处理）
                    assert max_diff < 1e-5, (
                        f"流式与批量计算 {feat} 不一致，"
                        f"最大差异: {max_diff:.8f}, 平均差异: {mean_diff:.8f}"
                    )

        print("  ✅ 流式 vs 批量一致性验证通过")

    def test_incremental_append_consistency(self, sample_data):
        """
        测试：增量追加数据时特征计算一致性

        模拟生产环境中逐步追加新数据的场景
        """
        print("\n" + "=" * 70)
        print("测试：BPC 增量追加一致性")
        print("=" * 70)

        df, orderflow = sample_data

        # 前 300 条数据
        df_partial = df.iloc[:300].copy()
        orderflow_partial = {k: v.iloc[:300] for k, v in orderflow.items()}

        result_partial = compute_bpc_soft_phase_from_series(
            close=df_partial["close"],
            high=df_partial["high"],
            low=df_partial["low"],
            atr=df_partial["atr"],
            volume=df_partial["volume"],
            cvd_change_5=orderflow_partial["cvd_change_5"],
            vpin=orderflow_partial["vpin"],
        )

        # 全部数据
        result_full = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # 前 250 条的结果应该相同（留出 warmup）
        check_idx = df.index[50:250]  # 跳过前 50 条（warmup 不完整）

        for feat in ["bpc_score_breakout", "bpc_score_pullback"]:
            vals_partial = result_partial.loc[check_idx, feat].dropna()
            vals_full = result_full.loc[check_idx, feat].dropna()

            common_idx = vals_partial.index.intersection(vals_full.index)
            if len(common_idx) > 0:
                diff = (vals_partial.loc[common_idx] - vals_full.loc[common_idx]).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"增量追加数据影响了历史 {feat} 值，差异: {max_diff:.8f}"

        print("  ✅ 增量追加一致性验证通过")


# =============================================================================
# 📋 测试类：特征值范围和正确性
# =============================================================================


class TestBPCFeaturesCorrectness:
    """
    BPC 特征值范围和正确性测试 ⭐⭐⭐
    """

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_feature_value_ranges(self, sample_data):
        """
        测试：所有特征值在预期范围内

        验证：
        - bpc_score_* 在 [0, 1] 范围内
        - bpc_direction 在 {-1, 0, 1} 范围内
        - 无 NaN/Inf 溢出
        """
        print("\n" + "=" * 70)
        print("测试：BPC 特征值范围")
        print("=" * 70)

        df, orderflow = sample_data

        result = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
            bb_width_normalized=orderflow["bb_width_normalized"],
        )

        # 检查 [0, 1] 范围的特征
        bounded_features = [
            "bpc_score_breakout",
            "bpc_score_pullback",
            "bpc_score_continuation",
            "bpc_score_neutral",
            "bpc_price_breakout_strength",
            "bpc_pullback_depth",
            "bpc_pullback_quality",
            "bpc_vol_breakout_confirm",
            "bpc_direction_confidence",
        ]

        for feat in bounded_features:
            if feat in result.columns:
                vals = result[feat].dropna()
                if len(vals) > 0:
                    min_val = vals.min()
                    max_val = vals.max()

                    assert min_val >= -0.01, f"{feat} 最小值 {min_val:.4f} < 0"
                    assert max_val <= 1.01, f"{feat} 最大值 {max_val:.4f} > 1"

                    print(f"  {feat}: [{min_val:.4f}, {max_val:.4f}] ✓")

        # 检查方向特征
        if "bpc_breakout_direction" in result.columns:
            direction_vals = result["bpc_breakout_direction"].dropna().unique()
            assert all(
                d in [-1, 0, 1] for d in direction_vals
            ), f"方向值超出范围: {direction_vals}"
            print(f"  bpc_breakout_direction: {set(direction_vals)} ✓")

        # 检查无 Inf
        for col in result.columns:
            assert not np.isinf(result[col]).any(), f"{col} 包含 Inf 值"

        print("  ✅ 所有特征值在预期范围内")

    def test_feature_completeness(self, sample_data):
        """
        测试：所有预期输出特征都存在
        """
        print("\n" + "=" * 70)
        print("测试：BPC 特征完整性")
        print("=" * 70)

        df, orderflow = sample_data

        result = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
            bb_width_normalized=orderflow["bb_width_normalized"],
        )

        # 预期的 25 个输出特征
        expected_features = [
            # ATOMIC: Breakout
            "bpc_price_breakout_strength",
            "bpc_vol_breakout_confirm",
            "bpc_cvd_breakout_confirm",
            "bpc_vpin_breakout_confirm",
            # ATOMIC: Pullback
            "bpc_pullback_depth",
            "bpc_pullback_quality",
            "bpc_vol_pullback_confirm",
            "bpc_cvd_absorption",
            # ATOMIC: Continuation
            "bpc_recovery_strength",
            "bpc_momentum_confirm",
            "bpc_vol_continuation_confirm",
            "bpc_cvd_momentum",
            "bpc_vpin_rising",
            # ATOMIC: Neutral
            "bpc_bb_compression",
            "bpc_vol_compression",
            # COMPOSITE
            "bpc_score_breakout",
            "bpc_score_pullback",
            "bpc_score_continuation",
            "bpc_score_neutral",
            # CONTEXTUAL
            "bpc_breakout_direction",
            "bpc_direction_confidence",
            "bpc_is_after_breakout",
            "bpc_was_in_pullback",
            "bpc_vol_ratio",
            "bpc_cvd_z",
        ]

        missing = [f for f in expected_features if f not in result.columns]
        extra = [f for f in result.columns if f not in expected_features]

        if missing:
            print(f"  ⚠️ 缺失特征: {missing}")
        if extra:
            print(f"  ⚠️ 额外特征: {extra}")

        assert len(missing) == 0, f"缺失特征: {missing}"

        print(f"  ✅ 所有 {len(expected_features)} 个特征都存在")

    def test_metadata_output(self, sample_data):
        """
        测试：元数据正确输出
        """
        print("\n" + "=" * 70)
        print("测试：BPC 特征元数据")
        print("=" * 70)

        df, orderflow = sample_data

        result = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
        )

        # 检查元数据
        assert hasattr(result, "attrs"), "结果应该有 attrs 属性"
        assert "feature_version" in result.attrs, "应该有 feature_version"
        assert (
            result.attrs["feature_version"] == FEATURE_VERSION
        ), f"版本不匹配: {result.attrs['feature_version']} != {FEATURE_VERSION}"

        print(f"  feature_version: {result.attrs['feature_version']} ✓")
        print(
            f"  param_pullback_decay: {result.attrs.get('param_pullback_decay', 'N/A')} ✓"
        )

        if "thresholds" in result.attrs:
            print(f"  thresholds: {list(result.attrs['thresholds'].keys())} ✓")

        print("  ✅ 元数据输出正确")


# =============================================================================
# 📋 测试类：多资产归一化
# =============================================================================


class TestBPCFeaturesMultiAsset:
    """
    BPC 特征多资产归一化测试 ⭐⭐⭐⭐

    验证：不同价格水平的资产，BPC 分数应该在相似范围内
    """

    def test_multi_asset_comparability(self):
        """
        测试：多资产 BPC 分数可比性

        验证：
        - 不同价格水平的资产（BTC ~50000, ETH ~3000, SOL ~100）
        - BPC 分数应该在 [0, 1] 范围内
        - 均值和方差应该相似（因为是归一化的）
        """
        print("\n" + "=" * 70)
        print("测试：多资产 BPC 分数可比性")
        print("=" * 70)

        results = {}

        # 测试不同价格水平的资产
        assets = {
            "BTC": (50000.0, 0.02),  # 高价格，中等波动
            "ETH": (3000.0, 0.025),  # 中价格，略高波动
            "SOL": (100.0, 0.03),  # 低价格，高波动
        }

        for symbol, (base_price, volatility) in assets.items():
            df = create_ohlcv_data(
                n_samples=300,
                base_price=base_price,
                volatility=volatility,
                seed=42,
            )
            orderflow = create_orderflow_data(df, seed=42)

            result = compute_bpc_soft_phase_from_series(
                close=df["close"],
                high=df["high"],
                low=df["low"],
                atr=df["atr"],
                volume=df["volume"],
                cvd_change_5=orderflow["cvd_change_5"],
                vpin=orderflow["vpin"],
            )

            results[symbol] = result

            print(f"\n  {symbol} (价格水平 ~{base_price:.0f}):")
            for feat in [
                "bpc_score_breakout",
                "bpc_score_pullback",
                "bpc_score_neutral",
            ]:
                vals = result[feat].dropna()
                if len(vals) > 0:
                    print(f"    {feat}:")
                    print(f"      均值: {vals.mean():.4f}")
                    print(f"      标准差: {vals.std():.4f}")
                    print(f"      范围: [{vals.min():.4f}, {vals.max():.4f}]")

                    # 验证范围
                    assert (vals >= 0).all(), f"{symbol} {feat} 有负值"
                    assert (vals <= 1).all(), f"{symbol} {feat} > 1"

        # 验证不同资产的分数分布相似
        for feat in ["bpc_score_breakout", "bpc_score_neutral"]:
            means = [results[s][feat].dropna().mean() for s in assets.keys()]
            stds = [results[s][feat].dropna().std() for s in assets.keys()]

            # 均值差异不应过大（考虑随机性，允许一定差异）
            mean_range = max(means) - min(means)
            std_range = max(stds) - min(stds)

            print(f"\n  {feat} 跨资产统计:")
            print(f"    均值范围: {mean_range:.4f}")
            print(f"    标准差范围: {std_range:.4f}")

            assert mean_range < 0.3, f"{feat} 不同资产均值差异过大: {mean_range:.4f}"

        print("\n  ✅ 多资产可比性验证通过")


# =============================================================================
# 📋 测试类：辅助特征函数
# =============================================================================


class TestBPCAuxiliaryFeatures:
    """
    BPC 辅助特征函数测试
    """

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        df = create_ohlcv_data(n_samples=200, seed=42)
        return df

    def test_pullback_depth_pct(self, sample_data):
        """测试回踩深度计算"""
        df = sample_data

        result = compute_bpc_pullback_depth_pct_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            lookback=20,
        )

        assert "bpc_pullback_depth_long" in result.columns
        assert "bpc_pullback_depth_short" in result.columns
        assert "bpc_pullback_depth_pct" in result.columns

        # 验证范围 [0, 1]
        for col in result.columns:
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"{col} 有负值"
            assert (vals <= 1).all(), f"{col} > 1"

        print("  ✅ pullback_depth_pct 测试通过")

    def test_pullback_duration(self, sample_data):
        """测试回踩持续时间计算"""
        df = sample_data

        result = compute_bpc_pullback_duration_from_series(
            close=df["close"],
            high=df["high"],
            lookback=20,
        )

        assert "bpc_pullback_duration" in result.columns

        vals = result["bpc_pullback_duration"].dropna()
        assert (vals >= 0).all(), "duration 有负值"
        assert (vals <= 1).all(), "duration > 1"

        print("  ✅ pullback_duration 测试通过")

    def test_impulse_return_atr(self, sample_data):
        """测试脉冲收益 ATR 归一化"""
        df = sample_data

        result = compute_bpc_impulse_return_atr_from_series(
            close=df["close"],
            atr=df["atr"],
            lookback=20,
        )

        assert "bpc_impulse_return_atr" in result.columns
        assert "bpc_impulse_direction_match" in result.columns

        # impulse_return_atr 在 [-1, 1] 范围内
        vals = result["bpc_impulse_return_atr"].dropna()
        assert (vals >= -1.01).all(), "impulse < -1"
        assert (vals <= 1.01).all(), "impulse > 1"

        # direction_match 是 0 或 1
        match_vals = result["bpc_impulse_direction_match"].dropna().unique()
        assert all(
            v in [0, 1] for v in match_vals
        ), f"direction_match 超出范围: {match_vals}"

        print("  ✅ impulse_return_atr 测试通过")

    def test_dir_consistency_multi(self, sample_data):
        """测试多尺度方向一致性"""
        df = sample_data

        result = compute_bpc_dir_consistency_multi_from_series(
            close=df["close"],
            window_short=5,
            window_mid=20,
            window_long=50,
        )

        assert "bpc_dir_consistency_short" in result.columns
        assert "bpc_dir_consistency_mid" in result.columns
        assert "bpc_dir_consistency_long" in result.columns

        # 一致性应该在 [0, 1] 范围内
        for col in result.columns:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all(), f"{col} 有负值"
                assert (vals <= 1).all(), f"{col} > 1"

        print("  ✅ dir_consistency_multi 测试通过")


# =============================================================================
# 🎯 运行入口
# =============================================================================


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print(f"BPC 特征测试 (版本 {FEATURE_VERSION})")
    print("=" * 70)

    # 创建 fixtures
    df = create_ohlcv_data(n_samples=500, seed=42)
    orderflow = create_orderflow_data(df, seed=42)
    sample_data = (df, orderflow)

    tests = []

    # 未来函数测试
    test_no_leak = TestBPCFeaturesNoFutureLeak()
    tests.append(
        (
            "无未来函数 - 软阶段分数",
            test_no_leak.test_bpc_soft_phase_no_future_leak,
            [sample_data],
        )
    )
    tests.append(
        (
            "无未来函数 - 滚动窗口",
            test_no_leak.test_bpc_rolling_window_no_lookahead,
            [sample_data],
        )
    )

    # 流式 vs 批量测试
    test_streaming = TestBPCFeaturesStreamingVsBatch()
    tests.append(
        (
            "流式 vs 批量一致性",
            test_streaming.test_streaming_vs_batch_consistency,
            [sample_data],
        )
    )
    tests.append(
        (
            "增量追加一致性",
            test_streaming.test_incremental_append_consistency,
            [sample_data],
        )
    )

    # 正确性测试
    test_correctness = TestBPCFeaturesCorrectness()
    tests.append(
        ("特征值范围", test_correctness.test_feature_value_ranges, [sample_data])
    )
    tests.append(
        ("特征完整性", test_correctness.test_feature_completeness, [sample_data])
    )
    tests.append(("元数据输出", test_correctness.test_metadata_output, [sample_data]))

    # 多资产测试
    test_multi = TestBPCFeaturesMultiAsset()
    tests.append(("多资产可比性", test_multi.test_multi_asset_comparability, []))

    # 辅助特征测试
    test_aux = TestBPCAuxiliaryFeatures()
    tests.append(("pullback_depth_pct", test_aux.test_pullback_depth_pct, [df]))
    tests.append(("pullback_duration", test_aux.test_pullback_duration, [df]))
    tests.append(("impulse_return_atr", test_aux.test_impulse_return_atr, [df]))
    tests.append(("dir_consistency_multi", test_aux.test_dir_consistency_multi, [df]))

    passed = 0
    failed = 0

    for test_name, test_func, args in tests:
        try:
            test_func(*args)
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 失败: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

# =============================================================================
# 📋 测试类：BPC 上下文特征（新增）
# =============================================================================


class TestBPCContextFeatures:
    """
    BPC 上下文特征测试（breakout_context, pullback_structure, continuation_target,
    compression_state, phase_transition）
    """

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_breakout_context_basic(self, sample_data):
        """测试 breakout_context 基本功能"""
        df, orderflow = sample_data

        # 模拟必要输入
        bpc_breakout_direction = pd.Series(
            np.sign(df["close"].diff(5).fillna(0)), index=df.index
        ).astype(int)
        vp_poc = df["close"].rolling(20, min_periods=1).mean()
        vp_hal_high = df["high"].rolling(20, min_periods=1).max()
        vp_hal_low = df["low"].rolling(20, min_periods=1).min()
        liquidity_void_detected = pd.Series(
            np.random.randint(0, 2, len(df)), index=df.index
        )
        ofci_pct = pd.Series(np.random.uniform(0.3, 0.7, len(df)), index=df.index)

        result = compute_bpc_breakout_context_from_series(
            close=df["close"],
            bpc_breakout_direction=bpc_breakout_direction,
            vp_poc=vp_poc,
            vp_hal_high=vp_hal_high,
            vp_hal_low=vp_hal_low,
            liquidity_void_detected=liquidity_void_detected,
            ofci_pct=ofci_pct,
        )

        expected_cols = [
            "bpc_breakout_above_poc",
            "bpc_breakout_above_hal",
            "bpc_liquidity_void_ahead",
            "bpc_false_breakout_risk",
            "bpc_reflex_confirm",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        print("✅ breakout_context 基本功能测试通过")

    def test_pullback_structure_basic(self, sample_data):
        """测试 pullback_structure 基本功能"""
        df, _ = sample_data

        bpc_breakout_direction = pd.Series(
            np.sign(df["close"].diff(5).fillna(0)), index=df.index
        ).astype(int)
        vp_poc = df["close"].rolling(20, min_periods=1).mean()
        vp_hal_high = df["high"].rolling(20, min_periods=1).max()
        vp_hal_low = df["low"].rolling(20, min_periods=1).min()
        vpvr_volume_density = pd.Series(
            np.random.uniform(0.3, 0.7, len(df)), index=df.index
        )

        result = compute_bpc_pullback_structure_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            bpc_breakout_direction=bpc_breakout_direction,
            vp_poc=vp_poc,
            vp_hal_high=vp_hal_high,
            vp_hal_low=vp_hal_low,
            vpvr_volume_density=vpvr_volume_density,
        )

        expected_cols = [
            "bpc_pullback_fib_382",
            "bpc_pullback_fib_500",
            "bpc_pullback_fib_618",
            "bpc_pullback_to_poc",
            "bpc_pullback_in_hal",
            "bpc_pullback_volume_support",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        print("✅ pullback_structure 基本功能测试通过")

    def test_continuation_target_basic(self, sample_data):
        """测试 continuation_target 基本功能"""
        df, _ = sample_data

        bpc_breakout_direction = pd.Series(
            np.sign(df["close"].diff(5).fillna(0)), index=df.index
        ).astype(int)
        vpvr_lvn_distance = pd.Series(
            np.random.uniform(0.0, 1.0, len(df)), index=df.index
        )
        vpvr_lvn_count = pd.Series(np.random.randint(0, 5, len(df)), index=df.index)
        shd_pct = pd.Series(np.random.uniform(0.3, 0.7, len(df)), index=df.index)

        result = compute_bpc_continuation_target_from_series(
            close=df["close"],
            atr=df["atr"],
            bpc_breakout_direction=bpc_breakout_direction,
            vpvr_lvn_distance=vpvr_lvn_distance,
            vpvr_lvn_count=vpvr_lvn_count,
            shd_pct=shd_pct,
        )

        expected_cols = [
            "bpc_target_lvn_distance",
            "bpc_target_lvn_count",
            "bpc_momentum_divergence",
            "bpc_reflex_momentum",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        print("✅ continuation_target 基本功能测试通过")

    def test_compression_state_basic(self, sample_data):
        """测试 compression_state 基本功能"""
        df, orderflow = sample_data

        result = compute_bpc_compression_state_from_series(
            close=df["close"],
            volume=df["volume"],
            bb_width_normalized=orderflow["bb_width_normalized"],
        )

        expected_cols = [
            "bpc_vol_compression_state",
            "bpc_bb_compression_state",
            "bpc_garch_compression",
            "bpc_wpt_energy_low",
            "bpc_pre_breakout_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in ["bpc_pre_breakout_score"]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ compression_state 基本功能测试通过")

    def test_phase_transition_basic(self, sample_data):
        """测试 phase_transition 基本功能"""
        df, _ = sample_data

        # 模拟 BPC 主函数输出
        bpc_main = compute_bpc_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
            volume=df["volume"],
        )

        shd_pct = pd.Series(np.random.uniform(0.3, 0.7, len(df)), index=df.index)

        result = compute_bpc_phase_transition_from_series(
            bpc_score_breakout=bpc_main["bpc_score_breakout"],
            bpc_score_pullback=bpc_main["bpc_score_pullback"],
            bpc_score_continuation=bpc_main["bpc_score_continuation"],
            bpc_score_neutral=bpc_main["bpc_score_neutral"],
            shd_pct=shd_pct,
        )

        expected_cols = [
            "bpc_phase_dominant",
            "bpc_phase_confidence",
            "bpc_transition_b_to_p",
            "bpc_transition_p_to_c",
            "bpc_transition_speed",
            "bpc_structure_health",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        print("✅ phase_transition 基本功能测试通过")

    def test_context_features_no_future_leak(self, sample_data):
        """测试上下文特征无未来数据泄露"""
        df, orderflow = sample_data

        # 创建输入数据
        bpc_breakout_direction = pd.Series(
            np.sign(df["close"].diff(5).fillna(0)), index=df.index
        ).astype(int)
        vp_poc = df["close"].rolling(20, min_periods=1).mean()
        vp_hal_high = df["high"].rolling(20, min_periods=1).max()
        vp_hal_low = df["low"].rolling(20, min_periods=1).min()
        liquidity_void_detected = pd.Series(
            np.random.randint(0, 2, len(df)), index=df.index
        )
        ofci_pct = pd.Series(np.random.uniform(0.3, 0.7, len(df)), index=df.index)

        # 第一次计算
        result1 = compute_bpc_breakout_context_from_series(
            close=df["close"],
            bpc_breakout_direction=bpc_breakout_direction,
            vp_poc=vp_poc,
            vp_hal_high=vp_hal_high,
            vp_hal_low=vp_hal_low,
            liquidity_void_detected=liquidity_void_detected,
            ofci_pct=ofci_pct,
        )

        # 修改未来数据
        df_future = df.copy()
        future_idx = df_future.index[300:]
        df_future.loc[future_idx, "close"] *= 2.0

        # 第二次计算
        result2 = compute_bpc_breakout_context_from_series(
            close=df_future["close"],
            bpc_breakout_direction=bpc_breakout_direction,
            vp_poc=vp_poc,
            vp_hal_high=vp_hal_high,
            vp_hal_low=vp_hal_low,
            liquidity_void_detected=liquidity_void_detected,
            ofci_pct=ofci_pct,
        )

        # 验证历史特征值相同
        check_idx = df.index[:200]
        for col in ["bpc_breakout_above_poc", "bpc_reflex_confirm"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ 上下文特征无未来数据泄露测试通过")


# =============================================================================
# 📊 BB 二阶压缩 vs GARCH 压缩等效性测试
# =============================================================================


class TestBpcCompressionVsGarchEquivalence:
    """
    验证 bpc_garch_compression（已改为 BB 二阶压缩）与原始 GARCH 压缩的语义等效性。

    BB 二阶压缩定义：当前 BB 宽度在过去 pct_window 周期内的历史百分位（低 = 压缩）。
    GARCH 压缩定义：当前 GARCH 波动率在过去 pct_window 周期内的历史百分位（低 = 压缩）。

    两者都是衡量「当前波动率相对历史的压缩程度」，预期 Spearman 相关性 ≥ 0.70。
    """

    @staticmethod
    def _compute_garch_compression_reference(
        close: pd.Series, pct_window: int = 100
    ) -> pd.Series:
        """
        参考 GARCH 压缩实现（使用 arch 库）。
        若 arch 不可用则回退到 GARCH 近似：指数加权标准差的历史百分位。
        """
        try:
            from arch import arch_model  # type: ignore

            returns = close.pct_change().dropna() * 100
            am = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            garch_vol = res.conditional_volatility
            garch_vol = garch_vol.reindex(close.index).fillna(method="ffill").fillna(0)
        except Exception:
            # 回退：EWMA 近似 GARCH
            returns = close.pct_change().fillna(0)
            garch_vol = returns.ewm(span=20).std()

        garch_pct = (
            garch_vol.rolling(pct_window, min_periods=20).rank(pct=True).fillna(0.5)
        )
        return (1 - garch_pct).rename("garch_compression_ref")

    @staticmethod
    def _compute_ewma_compression(close: pd.Series, pct_window: int = 100) -> pd.Series:
        """EWMA 波动率压缩：与 compute_bpc_compression_state_from_series 内部实现完全一致。"""
        rets = close.pct_change().fillna(0)
        ewma_vol = rets.ewm(span=20, min_periods=5).std().fillna(0)
        ewma_pct = (
            ewma_vol.rolling(pct_window, min_periods=20).rank(pct=True).fillna(0.5)
        )
        return (1 - ewma_pct).rename("ewma_compression")

    @staticmethod
    def _make_bb_width_normalized(close: pd.Series, window: int = 20) -> pd.Series:
        """计算布林带宽度归一化值（与 bb_width_normalized_pct_f 输出一致）。"""
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().fillna(0)
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std
        bb_width = (bb_upper - bb_lower) / ma.clip(lower=1e-8)
        # 百分位归一化
        bb_pct = bb_width.rolling(100, min_periods=20).rank(pct=True).fillna(0.5)
        return bb_pct.rename("bb_width_normalized")

    def test_spearman_correlation_on_synthetic_data(self):
        """
        合成数据上验证 EWMA 波动率压缩与 GARCH 压缩的 Spearman 相关性 ≥ 0.85。

        EWMA 是 GARCH(1,1) 的标准近似，两者相关性在 GARCH 过程生成的数据上预期 ≥ 0.85。
        """
        np.random.seed(42)
        n = 2000

        # 生成 GARCH(1,1) 过程
        omega, alpha, beta = 0.00001, 0.1, 0.85
        sigma2 = np.zeros(n)
        eps = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            eps[t] = np.random.randn() * np.sqrt(sigma2[t])

        idx = pd.date_range("2022-01-01", periods=n, freq="4h")
        close = pd.Series(50000 * np.exp(np.cumsum(eps)), index=idx)
        # 真实 GARCH 压缩（用 sigma2 直接计算）
        garch_vol = pd.Series(np.sqrt(sigma2), index=idx)
        garch_pct = garch_vol.rolling(100, min_periods=20).rank(pct=True).fillna(0.5)
        garch_compression = (1 - garch_pct).rename("garch_compression")

        ewma_compression = self._compute_ewma_compression(close)

        # 对齐并计算相关性（跳过前 120 个 warmup 周期）
        common = close.index[120:]
        g = garch_compression.loc[common].dropna()
        e = ewma_compression.loc[common].reindex(g.index).dropna()
        common_idx = g.index.intersection(e.index)

        from scipy.stats import spearmanr

        corr, pvalue = spearmanr(g.loc[common_idx], e.loc[common_idx])

        print(f"\nEWMA压缩 vs GARCH压缩 Spearman 相关性: {corr:.4f} (p={pvalue:.2e})")
        assert (
            corr >= 0.85
        ), f"EWMA 压缩与 GARCH 压缩 Spearman 相关性 {corr:.4f} < 0.85，语义等效性不足"
        assert pvalue < 0.01, f"相关性不显著 (p={pvalue:.2e})"
        print("✅ EWMA压缩与GARCH压缩语义等效性验证通过")

    def test_bpc_garch_compression_output_equals_ewma(self):
        """
        验证 compute_bpc_compression_state_from_series 的 bpc_garch_compression
        输出列确实等于 EWMA 波动率压缩（而不是 GARCH）。
        """
        np.random.seed(123)
        n = 500
        idx = pd.date_range("2023-01-01", periods=n, freq="4h")
        close = pd.Series(
            np.exp(np.cumsum(np.random.randn(n) * 0.01)) * 50000, index=idx
        )
        volume = pd.Series(np.random.uniform(100, 1000, n), index=idx)

        bb_width_norm = self._make_bb_width_normalized(close)

        result = compute_bpc_compression_state_from_series(
            close=close,
            volume=volume,
            bb_width_normalized=bb_width_norm,
        )

        # bpc_garch_compression 应等于 ewma_compression
        expected = self._compute_ewma_compression(close)
        actual = result["bpc_garch_compression"]

        common_idx = actual.dropna().index.intersection(expected.dropna().index)
        assert len(common_idx) > 100, "有效样本不足"

        diff = (actual.loc[common_idx] - expected.loc[common_idx]).abs().max()
        assert diff < 1e-10, f"bpc_garch_compression 不等于 EWMA 压缩，最大差异: {diff}"
        print("✅ bpc_garch_compression 输出等于 EWMA 波动率压缩")

    def test_output_columns_unchanged(self):
        """
        验证输出列名集合未发生变化（向后兼容性）。
        """
        df = create_ohlcv_data(n_samples=300)
        close = df["close"]
        volume = df["volume"]
        bb_width_norm = self._make_bb_width_normalized(close)

        result = compute_bpc_compression_state_from_series(
            close=close,
            volume=volume,
            bb_width_normalized=bb_width_norm,
        )

        expected_cols = {
            "bpc_vol_compression_state",
            "bpc_bb_compression_state",
            "bpc_garch_compression",
            "bpc_wpt_energy_low",
            "bpc_pre_breakout_score",
        }
        assert (
            set(result.columns) == expected_cols
        ), f"输出列名变化: 期望 {expected_cols}，实际 {set(result.columns)}"
        print("✅ 输出列名向后兼容性验证通过")

    def test_no_garch_dependency_in_feature_nodes(self):
        """
        验证 FER/BPC/ME 的 feature nodes 中不再包含 garch_features_f（依赖链已清除）。
        """
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        for strategy in ["bpc", "me", "fer"]:
            archetypes_dir = f"config/strategies/{strategy}/archetypes"
            _, nodes = extract_features_from_archetypes(archetypes_dir)
            assert (
                "garch_features_f" not in nodes
            ), f"{strategy} 的 feature nodes 中仍包含 garch_features_f: {nodes}"
        print("✅ 所有策略的 feature nodes 中不含 garch_features_f")

    def test_bounded_output_range(self):
        """
        验证 bpc_garch_compression（BB二阶压缩）输出值域 [0, 1]。
        """
        df = create_ohlcv_data(n_samples=500)
        close = df["close"]
        volume = df["volume"]
        bb_width_norm = self._make_bb_width_normalized(close)

        result = compute_bpc_compression_state_from_series(
            close=close,
            volume=volume,
            bb_width_normalized=bb_width_norm,
        )

        col = result["bpc_garch_compression"].dropna()
        assert col.min() >= 0.0, f"bpc_garch_compression 最小值 {col.min()} < 0"
        assert col.max() <= 1.0, f"bpc_garch_compression 最大值 {col.max()} > 1"
        print(
            f"✅ bpc_garch_compression 值域 [{col.min():.4f}, {col.max():.4f}] ⊆ [0, 1]"
        )
