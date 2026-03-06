"""
FER (FailureExhaustionReversal) v2.1 特征测试

测试内容：
1. 未来函数检测：修改未来数据不影响历史特征值 ⭐⭐⭐⭐⭐
2. 流式一致性：分段计算 ≈ 全量计算 ⭐⭐⭐⭐
3. 功能正确性：值域、CVD 掩码、方向语义 ⭐⭐⭐
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.fer_features import (
    compute_fer_failure_signals_from_series,
    FEATURE_VERSION,
)

# =============================================================================
# 📊 测试数据生成器
# =============================================================================


def create_ohlcv_with_cvd(n=500, seed=42):
    """生成带 CVD 的 OHLCV 测试数据"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")

    close = pd.Series(50000 + np.cumsum(rng.randn(n) * 100), index=dates)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    volume = pd.Series(rng.uniform(100, 5000, n), index=dates)
    atr = pd.Series(rng.uniform(100, 500, n), index=dates)
    cvd = pd.Series(np.cumsum(rng.randn(n) * 500), index=dates)
    cvd_change_5 = cvd.diff(5)

    return dict(
        close=close,
        high=high,
        low=low,
        volume=volume,
        atr=atr,
        cvd=cvd,
        cvd_change_5=cvd_change_5,
    )


# CVD 依赖列（无流动性时应为 0.0 中性值）
CVD_DEPENDENT_COLS = [
    "fer_signed_efficiency",
    "fer_signed_efficiency_pct",
    "fer_efficiency_flip",
    "fer_efficiency_flip_strength",
    "fer_aggressor_absorption",
    "fer_absorption_streak",
    "fer_trapped_longs_score",
    "fer_trapped_shorts_score",
    "fer_impulse_failure_score",
    "fer_impulse_failure_direction",
]

# 纯价量列（不受 CVD 影响）
PRICE_ONLY_COLS = [
    "fer_momentum_efficiency_decay",
    "fer_volume_price_divergence",
]

ALL_COLS = CVD_DEPENDENT_COLS + PRICE_ONLY_COLS

# =============================================================================
# 1️⃣ 未来函数检测
# =============================================================================


class TestFERNoFutureLeak:
    """修改未来数据不应影响历史特征值"""

    @pytest.fixture
    def data(self):
        return create_ohlcv_with_cvd(n=500, seed=42)

    def test_no_future_leak_all_columns(self, data):
        """核心测试：修改 t=350 之后数据，t≤250 之前的值必须完全一致"""
        result_orig = compute_fer_failure_signals_from_series(**data)

        # 修改未来数据 (t=350 之后)
        data_mod = {k: v.copy() for k, v in data.items()}
        data_mod["close"].iloc[350:] *= 2.0
        data_mod["high"].iloc[350:] *= 2.0
        data_mod["low"].iloc[350:] *= 2.0
        data_mod["volume"].iloc[350:] *= 5.0
        data_mod["cvd"].iloc[350:] = 999999.0
        data_mod["cvd_change_5"].iloc[350:] = 999999.0

        result_mod = compute_fer_failure_signals_from_series(**data_mod)

        # 检查 t≤250 之前（留 100 bar 安全边距）
        check_end = 250
        for col in ALL_COLS:
            orig = result_orig[col].iloc[:check_end].dropna()
            mod = result_mod[col].iloc[:check_end].dropna()
            common = orig.index.intersection(mod.index)
            if len(common) == 0:
                continue
            diff = (orig.loc[common] - mod.loc[common]).abs()
            max_diff = diff.max()
            assert max_diff < 1e-8, (
                f"未来函数检测失败 [{col}]: 修改 t>350 数据后, "
                f"t<{check_end} 最大差异={max_diff:.2e}"
            )

    def test_no_future_leak_cvd_only_change(self, data):
        """仅修改未来 CVD，价量特征不受影响"""
        result_orig = compute_fer_failure_signals_from_series(**data)

        data_mod = {k: v.copy() for k, v in data.items()}
        data_mod["cvd"].iloc[300:] = -999999.0
        data_mod["cvd_change_5"].iloc[300:] = -999999.0

        result_mod = compute_fer_failure_signals_from_series(**data_mod)

        check_end = 200
        for col in ALL_COLS:
            orig = result_orig[col].iloc[:check_end].dropna()
            mod = result_mod[col].iloc[:check_end].dropna()
            common = orig.index.intersection(mod.index)
            if len(common) == 0:
                continue
            diff = (orig.loc[common] - mod.loc[common]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-8
            ), f"未来函数检测失败(CVD only) [{col}]: max_diff={max_diff:.2e}"


# =============================================================================
# 2️⃣ 流式一致性：分段计算 ≈ 全量计算
# =============================================================================


class TestFERStreamingConsistency:
    """分段计算结果 ≈ 全量计算结果（warmup 之后）"""

    @pytest.fixture
    def data(self):
        return create_ohlcv_with_cvd(n=500, seed=123)

    def test_two_part_streaming(self, data):
        """
        将 500 bar 分为 [0:300] 和 [0:500]
        后半段重叠区域的值应一致
        """
        # 全量
        result_full = compute_fer_failure_signals_from_series(**data)

        # 只取前 300
        data_part = {k: v.iloc[:300].copy() for k, v in data.items()}
        result_part = compute_fer_failure_signals_from_series(**data_part)

        # 检查重叠区域 [100:300]（前 100 是 warmup）
        check_start, check_end = 100, 300
        for col in ALL_COLS:
            full_slice = result_full[col].iloc[check_start:check_end].dropna()
            part_slice = result_part[col].iloc[check_start:check_end].dropna()
            common = full_slice.index.intersection(part_slice.index)
            if len(common) < 10:
                continue
            diff = (full_slice.loc[common] - part_slice.loc[common]).abs()
            max_diff = diff.max()
            assert max_diff < 1e-6, f"流式一致性失败 [{col}]: max_diff={max_diff:.2e}"

    def test_incremental_append(self, data):
        """追加新数据后，旧数据特征不变"""
        # 计算前 400 bar
        data_400 = {k: v.iloc[:400].copy() for k, v in data.items()}
        result_400 = compute_fer_failure_signals_from_series(**data_400)

        # 计算全部 500 bar
        result_500 = compute_fer_failure_signals_from_series(**data)

        # [100:400] 结果必须一致
        for col in ALL_COLS:
            r400 = result_400[col].iloc[100:400].dropna()
            r500 = result_500[col].iloc[100:400].dropna()
            common = r400.index.intersection(r500.index)
            if len(common) < 10:
                continue
            diff = (r400.loc[common] - r500.loc[common]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"增量追加一致性失败 [{col}]: max_diff={max_diff:.2e}"


# =============================================================================
# 3️⃣ 功能正确性
# =============================================================================


class TestFERFunctionalCorrectness:
    """值域、CVD 掩码、方向语义"""

    @pytest.fixture
    def data(self):
        return create_ohlcv_with_cvd(n=500, seed=77)

    def test_output_columns(self, data):
        """输出应有 12 列"""
        result = compute_fer_failure_signals_from_series(**data)
        assert set(result.columns) == set(ALL_COLS), (
            f"列不匹配: 缺={set(ALL_COLS)-set(result.columns)}, "
            f"多={set(result.columns)-set(ALL_COLS)}"
        )

    def test_bounded_columns(self, data):
        """bounded [0,1] 列的值域检查"""
        result = compute_fer_failure_signals_from_series(**data)
        bounded_cols = [
            "fer_signed_efficiency_pct",
            "fer_impulse_failure_score",
            "fer_momentum_efficiency_decay",
            "fer_volume_price_divergence",
        ]
        for col in bounded_cols:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            assert valid.min() >= -1e-9, f"{col} 下界越界: {valid.min()}"
            assert valid.max() <= 1.0 + 1e-9, f"{col} 上界越界: {valid.max()}"

    def test_efficiency_flip_discrete(self, data):
        """efficiency_flip 只有 -1, 0, 1"""
        result = compute_fer_failure_signals_from_series(**data)
        valid = result["fer_efficiency_flip"].dropna()
        assert set(valid.unique()).issubset(
            {-1.0, 0.0, 1.0}
        ), f"fer_efficiency_flip 应为 {{-1,0,1}}, 实际: {sorted(valid.unique())}"

    def test_impulse_failure_direction_discrete(self, data):
        """impulse_failure_direction 只有 -1, 0, 1"""
        result = compute_fer_failure_signals_from_series(**data)
        valid = result["fer_impulse_failure_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0}), (
            f"fer_impulse_failure_direction 应为 {{-1,0,1}}, "
            f"实际: {sorted(valid.unique())}"
        )

    def test_trapped_scores_non_negative(self, data):
        """trapped scores ≥ 0"""
        result = compute_fer_failure_signals_from_series(**data)
        for col in ["fer_trapped_longs_score", "fer_trapped_shorts_score"]:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            assert valid.min() >= -1e-9, f"{col} 出现负值: {valid.min()}"

    def test_absorption_streak_non_negative(self, data):
        """absorption_streak ≥ 0"""
        result = compute_fer_failure_signals_from_series(**data)
        valid = result["fer_absorption_streak"].dropna()
        if len(valid) > 0:
            assert valid.min() >= -1e-9, f"absorption_streak 出现负值"


# =============================================================================
# 4️⃣ CVD 活跃度掩码测试
# =============================================================================


class TestFERCVDActivityMask:
    """CVD 不活跃 / 缺失时的 0.0 中性值行为 (不是 NaN)"""

    def test_no_cvd_all_dependent_zero(self):
        """无 CVD 输入 → 10 个 CVD 相关列全 0.0 (中性值，不是 NaN)"""
        data = create_ohlcv_with_cvd(n=300, seed=99)
        del data["cvd"]
        del data["cvd_change_5"]

        result = compute_fer_failure_signals_from_series(**data)

        for col in CVD_DEPENDENT_COLS:
            assert (result[col] == 0.0).all(), (
                f"无 CVD 时 {col} 应全 0.0, "
                f"实际 non-zero={( result[col] != 0.0).sum()}"
            )

        # 纯价量列不应受影响
        for col in PRICE_ONLY_COLS:
            assert result[col].notna().sum() > 0, f"无 CVD 时 {col} 不应全 NaN"

    def test_zero_cvd_all_dependent_zero(self):
        """CVD 全零 → CVD 相关列全 0.0 (不是假的效率=0)"""
        data = create_ohlcv_with_cvd(n=300, seed=88)
        data["cvd"] = pd.Series(0.0, index=data["close"].index)
        data["cvd_change_5"] = pd.Series(0.0, index=data["close"].index)

        result = compute_fer_failure_signals_from_series(**data)

        for col in CVD_DEPENDENT_COLS:
            assert (result[col] == 0.0).all(), (
                f"CVD 全零时 {col} 应全 0.0, "
                f"实际 non-zero={( result[col] != 0.0).sum()}"
            )

    def test_cvd_active_has_valid_values(self):
        """正常 CVD → CVD 相关列应有有效值"""
        data = create_ohlcv_with_cvd(n=500, seed=42)
        result = compute_fer_failure_signals_from_series(**data)

        for col in CVD_DEPENDENT_COLS:
            valid_count = result[col].notna().sum()
            assert valid_count > 100, f"{col} 有效值太少: {valid_count}/500"

    def test_partial_cvd_dropout(self):
        """CVD 中间段归零 → 该段 CVD 相关列为 0.0"""
        data = create_ohlcv_with_cvd(n=500, seed=55)
        # 将 200-300 段 CVD 设为常数（模拟丢数据）
        data["cvd"].iloc[200:300] = data["cvd"].iloc[199]
        data["cvd_change_5"].iloc[200:300] = 0.0

        result = compute_fer_failure_signals_from_series(**data)

        # 该段 signed_efficiency 应为 0.0 (被 activity mask 置为中性值)
        zero_in_dropout = (result["fer_signed_efficiency"].iloc[220:300] == 0.0).sum()
        assert (
            zero_in_dropout > 30
        ), f"CVD 静止段应产生较多 0.0, 实际 zero={zero_in_dropout}/80"


# =============================================================================
# 4b️⃣ CVD 中性值实盘安全性测试 (bug fix 回归测试)
# =============================================================================


class TestFERCVDNeutralValueLiveSafety:
    """CVD 不活跃时输出 0.0，不被实盘 pd.isna() 丢弃"""

    def test_inactive_cvd_no_nan_in_output(self):
        """CVD 不活跃 → 12 列全部无 NaN (保证实盘不丢特征)"""
        data = create_ohlcv_with_cvd(n=300, seed=99)
        del data["cvd"]
        del data["cvd_change_5"]

        result = compute_fer_failure_signals_from_series(**data)

        for col in ALL_COLS:
            nan_count = result[col].isna().sum()
            assert nan_count == 0, (
                f"{col} 存在 {nan_count} 个 NaN，" f"实盘 pd.isna() 会丢弃这些特征"
            )

    def test_live_feature_extraction_preserves_all_fer(self):
        """模拟实盘特征提取流程：即使 CVD 不活跃，12 列全部进入 features dict"""
        data = create_ohlcv_with_cvd(n=300, seed=99)
        del data["cvd"]
        del data["cvd_change_5"]

        result = compute_fer_failure_signals_from_series(**data)

        # 模拟 incremental_feature_computer.py L1478-1485 的提取逻辑
        last_row = result.iloc[-1].to_dict()
        features = {}
        for k, v in last_row.items():
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[k] = float(v)

        for col in ALL_COLS:
            assert col in features, (
                f"{col} 未进入 features dict，" f"模型将收不到该特征"
            )

    def test_zero_cvd_live_extraction(self):
        """CVD 全零 → 实盘提取保留所有 12 列"""
        data = create_ohlcv_with_cvd(n=300, seed=88)
        data["cvd"] = pd.Series(0.0, index=data["close"].index)
        data["cvd_change_5"] = pd.Series(0.0, index=data["close"].index)

        result = compute_fer_failure_signals_from_series(**data)

        last_row = result.iloc[-1].to_dict()
        features = {}
        for k, v in last_row.items():
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[k] = float(v)

        for col in ALL_COLS:
            assert col in features, f"CVD 全零时 {col} 未进入 features dict"

    def test_active_cvd_values_unchanged(self):
        """CVD 活跃时，特征值不应被替换为 0.0"""
        data = create_ohlcv_with_cvd(n=500, seed=42)
        result = compute_fer_failure_signals_from_series(**data)

        # warmup 之后的活跃区间应有非零值
        # 离散稀疏信号 (flip/direction) 阈值较低
        sparse_cols = {
            "fer_efficiency_flip",
            "fer_efficiency_flip_strength",
            "fer_impulse_failure_direction",
        }
        for col in CVD_DEPENDENT_COLS:
            valid = result[col].iloc[100:]
            nonzero = (valid != 0.0).sum()
            threshold = 10 if col in sparse_cols else 50
            assert nonzero > threshold, (
                f"CVD 活跃时 {col} 非零值太少: {nonzero}/400，"
                f"可能误将有效值替换为 0.0"
            )

    def test_cvd_dependent_neutral_value_is_zero(self):
        """CVD 不活跃时的中性值应为 0.0"""
        data = create_ohlcv_with_cvd(n=300, seed=77)
        del data["cvd"]
        del data["cvd_change_5"]

        result = compute_fer_failure_signals_from_series(**data)

        for col in CVD_DEPENDENT_COLS:
            unique_vals = result[col].unique()
            assert len(unique_vals) == 1 and unique_vals[0] == 0.0, (
                f"无 CVD 时 {col} 应只有 0.0，" f"实际: {sorted(unique_vals)[:5]}"
            )


# =============================================================================
# 5️⃣ 方向语义验证
# =============================================================================


class TestFERDirectionSemantics:
    """验证 FER 方向信号的因果语义"""

    def test_impulse_failure_direction_semantics(self):
        """
        CVD↑ + price↓ → direction=-1 (多头失败→做空)
        CVD↓ + price↑ → direction=+1 (空头失败→做多)
        """
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")
        rng = np.random.RandomState(42)

        close = pd.Series(50000 + np.cumsum(rng.randn(n) * 50), index=dates)
        high = close + 100
        low = close - 100
        volume = pd.Series(1000 + rng.rand(n) * 500, index=dates)
        atr = pd.Series(200.0, index=dates)

        # 构造: CVD 持续上升 + 价格下跌
        cvd = pd.Series(np.linspace(0, 10000, n), index=dates)
        close_down = pd.Series(np.linspace(50000, 45000, n), index=dates)
        high_down = close_down + 100
        low_down = close_down - 100
        cvd_change_5 = cvd.diff(5)

        result = compute_fer_failure_signals_from_series(
            close=close_down,
            high=high_down,
            low=low_down,
            volume=volume,
            atr=atr,
            cvd=cvd,
            cvd_change_5=cvd_change_5,
        )

        # 应有 direction=-1 (多头失败) 出现
        dir_vals = result["fer_impulse_failure_direction"].dropna()
        neg_count = (dir_vals == -1).sum()
        assert neg_count > 0, "CVD↑ + price↓ 场景应检测到多头失败 (direction=-1)"

    def test_efficiency_flip_semantics(self):
        """构造效率翻转场景并检查 flip 触发"""
        n = 500
        rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")

        # CVD 持续上升 + 加噪声（避免 ΔCVD 恒定导致掩码全 False）
        cvd_base = np.linspace(0, 20000, n)
        cvd = pd.Series(cvd_base + np.cumsum(rng.randn(n) * 100), index=dates)

        # 前半段: price↑ (正效率), 后半段: price↓ (负效率)
        close = pd.Series(index=dates, dtype=float)
        close.iloc[:250] = np.linspace(50000, 55000, 250) + rng.randn(250) * 30
        close.iloc[250:] = np.linspace(55000, 48000, 250) + rng.randn(250) * 30
        high = close + 100
        low = close - 100
        volume = pd.Series(1000 + rng.rand(n) * 500, index=dates)
        atr = pd.Series(200.0, index=dates)
        cvd_change_5 = cvd.diff(5)

        result = compute_fer_failure_signals_from_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            atr=atr,
            cvd=cvd,
            cvd_change_5=cvd_change_5,
        )

        # 应在 ~250 附近检测到 flip（效率从正翻负）
        flips = result["fer_efficiency_flip"].dropna()
        flip_events = flips[flips != 0]
        assert len(flip_events) > 0, "效率翻转场景应检测到 flip 事件"


# =============================================================================
# 🏃 直接运行入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
