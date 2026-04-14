"""
SRB 特征测试（srb_features）

覆盖：
1. 未来函数：修改未来 bar 不影响足够早的历史输出
2. 流式：前缀窗口重算与全量在重叠区间一致
3. 语义：与 FER 失败对齐条件互斥的 impulse 方向；假突破降权
4. 功能：值域、无 CVD 时全零、注册可加载
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.registry import ensure_features_registered, get_feature_func
from src.features.time_series.srb_features import (
    FEATURE_VERSION,
    compute_srb_sr_success_breakout_from_series,
)

SRB_COLS = [
    "srb_sr_success_breakout_score",
    "srb_sr_success_breakout_score_pct",
    "srb_sr_success_breakout_direction_signed",
]


def create_srb_inputs(n: int = 500, seed: int = 42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = pd.Series(50000 + np.cumsum(rng.randn(n) * 100), index=dates)
    cvd = pd.Series(np.cumsum(rng.randn(n) * 500), index=dates)
    cvd_change_5 = cvd.diff(5).fillna(0.0)
    dist = pd.Series(rng.uniform(0.05, 0.8, n), index=dates)
    direction = pd.Series(np.where(rng.rand(n) > 0.5, 1.0, -1.0), index=dates)
    return dict(
        close=close,
        cvd=cvd,
        cvd_change_5=cvd_change_5,
        dist_to_nearest_sr=dist,
        direction_to_nearest_sr=direction,
    )


# =============================================================================
# 1) 未来函数
# =============================================================================


class TestSRBNoFutureLeak:
    def test_future_price_volume_no_effect_on_past(self):
        data = create_srb_inputs(n=500, seed=42)
        orig = compute_srb_sr_success_breakout_from_series(**data)
        mod = {k: v.copy() for k, v in data.items()}
        mod["close"].iloc[350:] *= 2.0
        out = compute_srb_sr_success_breakout_from_series(**mod)
        check_end = 250
        for col in SRB_COLS:
            a = orig[col].iloc[:check_end]
            b = out[col].iloc[:check_end]
            common = a.index.intersection(b.index)
            diff = (a.loc[common] - b.loc[common]).abs().max()
            assert diff < 1e-8, f"{col} future leak via close: max_diff={diff}"

    def test_future_sr_context_no_effect_on_past(self):
        data = create_srb_inputs(n=500, seed=7)
        orig = compute_srb_sr_success_breakout_from_series(**data)
        mod = {k: v.copy() for k, v in data.items()}
        mod["dist_to_nearest_sr"].iloc[320:] = 9.0
        mod["direction_to_nearest_sr"].iloc[320:] = -1.0
        out = compute_srb_sr_success_breakout_from_series(**mod)
        check_end = 220
        for col in SRB_COLS:
            a = orig[col].iloc[:check_end]
            b = out[col].iloc[:check_end]
            common = a.index.intersection(b.index)
            diff = (a.loc[common] - b.loc[common]).abs().max()
            assert diff < 1e-8, f"{col} future leak via SR cols: max_diff={diff}"


# =============================================================================
# 2) 流式一致性
# =============================================================================


class TestSRBStreamingConsistency:
    def test_prefix_matches_full_in_overlap(self):
        data = create_srb_inputs(n=500, seed=123)
        full = compute_srb_sr_success_breakout_from_series(**data)
        part = {k: v.iloc[:300].copy() for k, v in data.items()}
        pre = compute_srb_sr_success_breakout_from_series(**part)
        check_start, check_end = 120, 300
        for col in SRB_COLS:
            a = full[col].iloc[check_start:check_end]
            b = pre[col].iloc[check_start:check_end]
            common = a.index.intersection(b.index)
            if len(common) < 20:
                continue
            diff = (a.loc[common] - b.loc[common]).abs().max()
            assert diff < 1e-6, f"streaming [{col}] max_diff={diff}"

    def test_incremental_extend_unchanged_prefix(self):
        data = create_srb_inputs(n=500, seed=5)
        r400 = compute_srb_sr_success_breakout_from_series(
            **{k: v.iloc[:400].copy() for k, v in data.items()}
        )
        r500 = compute_srb_sr_success_breakout_from_series(**data)
        for col in SRB_COLS:
            a = r400[col].iloc[100:400]
            b = r500[col].iloc[100:400]
            common = a.index.intersection(b.index)
            diff = (a.loc[common] - b.loc[common]).abs().max()
            assert diff < 1e-6, f"incremental [{col}] max_diff={diff}"


# =============================================================================
# 3) 值域与无 CVD
# =============================================================================


class TestSRBFunctionalBounds:
    def test_bounded_outputs_with_cvd(self):
        data = create_srb_inputs(n=400, seed=0)
        out = compute_srb_sr_success_breakout_from_series(**data)
        assert out["srb_sr_success_breakout_score"].between(0.0, 1.0).all()
        assert out["srb_sr_success_breakout_score_pct"].between(0.0, 1.0).all()
        ds = out["srb_sr_success_breakout_direction_signed"]
        assert ds.isin([-1.0, 0.0, 1.0]).all()

    def test_no_cvd_all_zero(self):
        data = create_srb_inputs(n=200, seed=1)
        del data["cvd"]
        del data["cvd_change_5"]
        out = compute_srb_sr_success_breakout_from_series(**data)
        for col in SRB_COLS:
            assert (out[col].abs().max() == 0.0) or out[col].isna().all()


# =============================================================================
# 4) 语义：对齐 / 与 FER 失败模式互斥 / 假突破
# =============================================================================


class TestSRBSemantics:
    def test_aligned_bullish_impulse_below_sr_positive_score(self):
        """direction=+1（价在 SR 下方）且近 SR：CVD↑ 与 close↑ 同向 → 应有非零成功得分。"""
        n = 260
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.RandomState(99)
        close = pd.Series(100.0, index=idx)
        # 随机游走 CVD，保证活跃度分位阈值下仍有足够 bar 为 active
        cvd = pd.Series(np.cumsum(rng.randn(n) * 40.0 + 3.0), index=idx)
        for t in range(220, n):
            close.iloc[t] = 100.0 + (t - 219) * 0.15
        dist = pd.Series(0.25, index=idx)
        direction = pd.Series(1.0, index=idx)
        out = compute_srb_sr_success_breakout_from_series(
            close=close,
            cvd=cvd,
            cvd_change_5=cvd.diff(5).fillna(0.0),
            dist_to_nearest_sr=dist,
            direction_to_nearest_sr=direction,
        )
        tail = out["srb_sr_success_breakout_score"].iloc[-5:]
        assert tail.max() > 0.01, "expected some SRB success mass near end"
        last_dir = out["srb_sr_success_breakout_direction_signed"].iloc[-1]
        assert last_dir in (0.0, 1.0)

    def test_failure_pattern_no_sr_success_alignment(self):
        """dir=+1 但 CVD↑、5bar 价跌（FER 类多头失败）→ 与成功 impulse 正交，得分应接近 0。"""
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = pd.Series(100.0, index=idx)
        rng = np.random.RandomState(3)
        cvd = pd.Series(np.cumsum(rng.randn(n) * 25.0 + 2.0), index=idx)
        close.iloc[-6:] = [100.3, 100.2, 100.1, 100.05, 100.0, 99.9]
        dist = pd.Series(0.2, index=idx)
        direction = pd.Series(1.0, index=idx)
        out = compute_srb_sr_success_breakout_from_series(
            close=close,
            cvd=cvd,
            cvd_change_5=cvd.diff(5).fillna(0.0),
            dist_to_nearest_sr=dist,
            direction_to_nearest_sr=direction,
        )
        assert out["srb_sr_success_breakout_score"].iloc[-1] < 1e-6

    def test_fake_breakout_reduces_score(self):
        data = create_srb_inputs(n=350, seed=11)
        base = compute_srb_sr_success_breakout_from_series(**data)
        fake = pd.Series(0.0, index=data["close"].index)
        fake.iloc[200:340] = 1.0
        penalized = compute_srb_sr_success_breakout_from_series(
            **data, fake_breakout=fake
        )
        t = 250
        assert (
            penalized["srb_sr_success_breakout_score"].iloc[t]
            <= base["srb_sr_success_breakout_score"].iloc[t] + 1e-9
        )


# =============================================================================
# 5) 注册
# =============================================================================


def test_feature_registered():
    ensure_features_registered()
    f = get_feature_func("compute_srb_sr_success_breakout_from_series")
    assert f is compute_srb_sr_success_breakout_from_series


def test_feature_version_string():
    assert isinstance(FEATURE_VERSION, str) and len(FEATURE_VERSION) > 0
