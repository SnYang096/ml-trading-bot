"""
MomentumExpansion v3.0 特征测试

测试内容：
1. 基本功能：输出列存在、值范围正确
2. 未来函数检测：修改未来数据不影响历史计算
3. 流式一致性：全量 vs 增量结果一致
4. NaN 安全：缺失输入时输出中性值
5. 所有层（Core/Gate/Evidence/Entry/Failure/Context）
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.momentum_expansion_features import (
    compute_momentum_expansion_soft_phase_from_series,
    compute_me_gate_from_series,
    compute_me_evidence_from_series,
    compute_me_entry_from_series,
    compute_momentum_expansion_failure_from_series,
    compute_momentum_expansion_context_from_series,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """生成 500 根 K 线的模拟数据"""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")

    # 模拟价格走势（含趋势 + 噪声）
    returns = np.random.normal(0.0002, 0.005, n)
    # 在 200-300 区间注入加速段
    returns[200:250] += 0.003  # 加速上涨
    returns[250:280] += 0.001  # 减速
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volume = np.random.lognormal(10, 0.5, n)
    # 加速段放量
    volume[200:250] *= 2.0

    atr = pd.Series(high - low, index=idx).rolling(14, min_periods=1).mean()

    df = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "atr": atr,
        },
        index=idx,
    )

    # 订单流数据
    cvd_change_5 = pd.Series(
        np.random.normal(0, 100, n) + returns * 50000,  # 与价格方向相关
        index=idx,
    )
    delta = pd.Series(
        np.random.normal(0, 50, n) + returns * 20000,
        index=idx,
    )

    return df, cvd_change_5, delta


# =============================================================================
# 1. Core function tests
# =============================================================================


class TestMECoreFunction:
    """测试核心三因子函数"""

    EXPECTED_COLUMNS = [
        "me_atr_pct",
        "me_vol_regime",
        "me_accel_2k",
        "me_accel_5k",
        "me_accel_persistence",
        "me_multi_tf_alignment",
        "me_cvd_alignment",
        "me_cvd_strength",
        "me_volume_surge",
        "me_volume_accel",
        "me_delta_net_flow",
    ]

    def test_basic_output(self, sample_data):
        """11 个输出列全部存在"""
        df, cvd, delta = sample_data
        result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )
        for col in self.EXPECTED_COLUMNS:
            assert col in result.columns, f"缺少列: {col}"
        assert len(result) == len(df)

    def test_value_ranges(self, sample_data):
        """值范围检查"""
        df, cvd, delta = sample_data
        result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )

        # [0, 1] 范围
        bounded_01 = [
            "me_atr_pct",
            "me_vol_regime",
            "me_accel_persistence",
            "me_multi_tf_alignment",
            "me_cvd_alignment",
            "me_cvd_strength",
            "me_volume_surge",
            "me_volume_accel",
        ]
        for col in bounded_01:
            vals = result[col].dropna()
            assert vals.min() >= 0 - 1e-9, f"{col} min={vals.min()} < 0"
            assert vals.max() <= 1 + 1e-9, f"{col} max={vals.max()} > 1"

        # [-3, 3] 范围
        for col in ["me_accel_2k", "me_accel_5k"]:
            vals = result[col].dropna()
            assert vals.min() >= -3 - 1e-9, f"{col} min={vals.min()} < -3"
            assert vals.max() <= 3 + 1e-9, f"{col} max={vals.max()} > 3"

        # [-1, 1] 范围
        vals = result["me_delta_net_flow"].dropna()
        assert vals.min() >= -1 - 1e-9, f"delta_net_flow min={vals.min()} < -1"
        assert vals.max() <= 1 + 1e-9, f"delta_net_flow max={vals.max()} > 1"

    def test_no_future_leak(self, sample_data):
        """未来函数检测：修改 bar 300+ 的数据不影响 bar 0~199"""
        df, cvd, delta = sample_data

        result1 = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )

        # 篡改未来数据
        df2 = df.copy()
        df2.loc[df2.index[300:], "close"] *= 2.0
        df2.loc[df2.index[300:], "volume"] *= 5.0
        cvd2 = cvd.copy()
        cvd2.iloc[300:] *= 10.0
        delta2 = delta.copy()
        delta2.iloc[300:] *= 10.0

        result2 = compute_momentum_expansion_soft_phase_from_series(
            close=df2["close"],
            high=df2["high"],
            low=df2["low"],
            volume=df2["volume"],
            atr=df2["atr"],
            cvd_change_5=cvd2,
            delta=delta2,
        )

        # 历史部分（前200根）应完全一致
        check_idx = df.index[:200]
        for col in self.EXPECTED_COLUMNS:
            v1 = result1.loc[check_idx, col].dropna()
            v2 = result2.loc[check_idx, col].dropna()
            common = v1.index.intersection(v2.index)
            if len(common) > 0:
                diff = (v1.loc[common] - v2.loc[common]).abs().max()
                assert diff < 1e-10, f"{col} 存在未来泄露，差异: {diff}"

    def test_streaming_consistency(self, sample_data):
        """流式一致性：全量 vs 前 N 根结果应一致"""
        df, cvd, delta = sample_data

        # 全量计算
        full_result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )

        # 只取前 300 根
        n = 300
        partial_result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"].iloc[:n],
            high=df["high"].iloc[:n],
            low=df["low"].iloc[:n],
            volume=df["volume"].iloc[:n],
            atr=df["atr"].iloc[:n],
            cvd_change_5=cvd.iloc[:n],
            delta=delta.iloc[:n],
        )

        # 前 300 根结果应完全一致
        for col in self.EXPECTED_COLUMNS:
            v_full = full_result[col].iloc[:n].dropna()
            v_part = partial_result[col].dropna()
            common = v_full.index.intersection(v_part.index)
            if len(common) > 0:
                diff = (v_full.loc[common] - v_part.loc[common]).abs().max()
                assert diff < 1e-10, f"{col} 流式不一致，差异: {diff}"

    def test_no_orderflow_fallback(self, sample_data):
        """无订单流时应返回中性值"""
        df, _, _ = sample_data
        result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        # CVD 相关应为中性 0.5
        assert (result["me_cvd_alignment"] == 0.5).all()
        assert (result["me_cvd_strength"] == 0.5).all()
        # Delta 相关应为 0
        assert (result["me_delta_net_flow"] == 0.0).all()

    def test_acceleration_not_trend(self, sample_data):
        """稳定趋势中加速度应接近 0"""
        n = 500
        idx = pd.date_range("2025-01-01", periods=n, freq="15min")
        # 创建匀速趋势：每根 K 线涨 0.1%
        close = 100 * np.exp(np.arange(n) * 0.001)
        high = close * 1.002
        low = close * 0.998
        volume = np.full(n, 1000.0)
        atr = pd.Series(high - low, index=idx).rolling(14, min_periods=1).mean()

        result = compute_momentum_expansion_soft_phase_from_series(
            close=pd.Series(close, index=idx),
            high=pd.Series(high, index=idx),
            low=pd.Series(low, index=idx),
            volume=pd.Series(volume, index=idx),
            atr=atr,
        )

        # 匀速趋势中，5K加速度应接近 0（不触发 ME）
        accel_5k = result["me_accel_5k"].iloc[50:].dropna()
        assert (
            accel_5k.abs().mean() < 0.3
        ), f"匀速趋势中 accel_5k 均值过高: {accel_5k.abs().mean():.3f}"


# =============================================================================
# 2. Gate / Evidence / Entry tests
# =============================================================================


class TestMELayerFunctions:
    """测试 Gate / Evidence / Entry 层"""

    def _get_core_result(self, sample_data):
        df, cvd, delta = sample_data
        return compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )

    def test_gate_basic(self, sample_data):
        """Gate 输出正确"""
        core = self._get_core_result(sample_data)
        gate = compute_me_gate_from_series(
            me_atr_pct=core["me_atr_pct"],
            me_cvd_alignment=core["me_cvd_alignment"],
            me_volume_surge=core["me_volume_surge"],
        )
        expected = [
            "me_gate_expansion_ok",
            "me_gate_flow_ok",
            "me_gate_volume_ok",
            "me_gate_pass",
        ]
        for col in expected:
            assert col in gate.columns, f"Gate 缺少列: {col}"
        # 所有值应为 0 或 1
        for col in expected:
            unique_vals = set(gate[col].dropna().unique())
            assert unique_vals.issubset({0.0, 1.0}), f"{col} 非二值: {unique_vals}"

    def test_evidence_basic(self, sample_data):
        """Evidence 乘法信号在 [0, 1]"""
        core = self._get_core_result(sample_data)
        evidence = compute_me_evidence_from_series(
            me_atr_pct=core["me_atr_pct"],
            me_accel_5k=core["me_accel_5k"],
            me_cvd_alignment=core["me_cvd_alignment"],
            me_cvd_strength=core["me_cvd_strength"],
            me_volume_surge=core["me_volume_surge"],
        )
        assert "me_evidence_signal" in evidence.columns
        vals = evidence["me_evidence_signal"].dropna()
        assert vals.min() >= 0 - 1e-9
        assert vals.max() <= 1 + 1e-9

    def test_entry_basic(self, sample_data):
        """Entry 输出正确"""
        core = self._get_core_result(sample_data)
        entry = compute_me_entry_from_series(
            me_accel_2k=core["me_accel_2k"],
            me_cvd_strength=core["me_cvd_strength"],
            me_volume_surge=core["me_volume_surge"],
        )
        expected = ["me_entry_micro_accel", "me_entry_flow_burst", "me_entry_confirm"]
        for col in expected:
            assert col in entry.columns, f"Entry 缺少列: {col}"

    def test_evidence_multiplicative(self, sample_data):
        """Evidence 乘法：任一因子为 0 → 信号为 0"""
        core = self._get_core_result(sample_data)

        # 强制 atr_pct 为 0
        zero_energy = pd.Series(0.0, index=core.index)
        evidence = compute_me_evidence_from_series(
            me_atr_pct=zero_energy,
            me_accel_5k=core["me_accel_5k"],
            me_cvd_alignment=core["me_cvd_alignment"],
            me_cvd_strength=core["me_cvd_strength"],
            me_volume_surge=core["me_volume_surge"],
        )
        # NaN 来自 rolling 操作的前几行，fillna(0) 后检查
        vals = evidence["me_evidence_signal"].fillna(0.0)
        assert (vals == 0.0).all(), "Energy=0 时 Evidence 应为 0"


# =============================================================================
# 3. Failure / Context tests
# =============================================================================


class TestMEFailureContext:
    """测试失败信号和上下文特征"""

    def test_failure_basic(self, sample_data):
        """Failure 输出正确"""
        df, cvd, delta = sample_data
        core = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=cvd,
            delta=delta,
        )
        failure = compute_momentum_expansion_failure_from_series(
            close=df["close"],
            me_atr_pct=core["me_atr_pct"],
            me_accel_5k=core["me_accel_5k"],
            me_cvd_alignment=core["me_cvd_alignment"],
            volume=df["volume"],
        )
        expected = [
            "me_false_expansion",
            "me_vol_divergence",
            "me_flow_exhaustion",
            "me_failure_score",
        ]
        for col in expected:
            assert col in failure.columns, f"Failure 缺少列: {col}"
        # failure_score 在 [0, 1]
        vals = failure["me_failure_score"].dropna()
        assert vals.min() >= 0 - 1e-9
        assert vals.max() <= 1 + 1e-9

    def test_context_basic(self, sample_data):
        """Context 输出正确"""
        df, _, _ = sample_data
        context = compute_momentum_expansion_context_from_series(
            close=df["close"],
        )
        expected = ["me_jump_risk_suitable", "me_reflex_risk", "me_regime_suitable"]
        for col in expected:
            assert col in context.columns, f"Context 缺少列: {col}"

    def test_context_with_inputs(self, sample_data):
        """Context 带输入时值范围正确"""
        df, _, _ = sample_data
        n = len(df)
        idx = df.index
        context = compute_momentum_expansion_context_from_series(
            close=df["close"],
            jump_risk_pct=pd.Series(np.random.uniform(0, 1, n), index=idx),
            shd_pct=pd.Series(np.random.uniform(0, 1, n), index=idx),
            ofci_pct=pd.Series(np.random.uniform(0, 1, n), index=idx),
        )
        for col in context.columns:
            vals = context[col].dropna()
            assert vals.min() >= 0 - 1e-9, f"{col} < 0"
            assert vals.max() <= 1 + 1e-9, f"{col} > 1"
