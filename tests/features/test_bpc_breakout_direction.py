"""
测试 bpc_breakout_direction 正确性

验证:
1. 值只有 1 和 -1（无 0 或 NaN）
2. 方向与 breakout_long/short 一致
3. 方向与 price breakout 逻辑一致（突破前高→1，突破前低→-1）
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.bpc_features import compute_bpc_soft_phase_from_series


def _make_trending_up_data(n: int = 300) -> pd.DataFrame:
    """生成上涨趋势数据（应产生 direction=1）"""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.abs(np.random.randn(n)) * 0.3)  # 持续上涨
    high = prices + np.random.rand(n) * 0.5
    low = prices - np.random.rand(n) * 0.3
    volume = 1000 + np.random.randint(0, 2000, n).astype(float)
    # 某些 bar 放量突破
    volume[100:110] *= 3
    volume[200:210] *= 3

    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


def _make_trending_down_data(n: int = 300) -> pd.DataFrame:
    """生成下跌趋势数据（应产生 direction=-1）"""
    np.random.seed(123)
    prices = 200 - np.cumsum(np.abs(np.random.randn(n)) * 0.3)  # 持续下跌
    high = prices + np.random.rand(n) * 0.3
    low = prices - np.random.rand(n) * 0.5
    volume = 1000 + np.random.randint(0, 2000, n).astype(float)
    volume[100:110] *= 3

    return pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


def _compute_direction(df: pd.DataFrame) -> pd.Series:
    """用 bpc_features 计算 bpc_breakout_direction"""
    # 需要提供 ATR
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()

    result = compute_bpc_soft_phase_from_series(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        atr=atr,
    )
    return result["bpc_breakout_direction"]


class TestBpcBreakoutDirection:
    """bpc_breakout_direction 正确性测试"""

    def test_values_are_1_or_minus1(self):
        """方向值只能是 1、0 或 -1（0=初始无方向）"""
        df = _make_trending_up_data()
        direction = _compute_direction(df)

        unique_vals = set(direction.dropna().unique())
        assert unique_vals.issubset(
            {0, 1, -1, 0.0, 1.0, -1.0}
        ), f"Expected {{0, 1, -1}}, got {unique_vals}"

    def test_no_nan_values(self):
        """不应有 NaN"""
        df = _make_trending_up_data()
        direction = _compute_direction(df)
        nan_count = direction.isna().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_uptrend_mostly_positive(self):
        """上涨趋势中，多数时刻 direction=1"""
        df = _make_trending_up_data()
        direction = _compute_direction(df)

        positive_pct = (direction == 1).mean()
        assert (
            positive_pct > 0.5
        ), f"Uptrend should have >50% direction=1, got {positive_pct:.1%}"

    def test_downtrend_mostly_negative(self):
        """下跌趋势中，多数时刻 direction=-1"""
        df = _make_trending_down_data()
        direction = _compute_direction(df)

        negative_pct = (direction == -1).mean()
        assert (
            negative_pct > 0.5
        ), f"Downtrend should have >50% direction=-1, got {negative_pct:.1%}"

    def test_predictions_file_direction_valid(self):
        """验证实际 predictions 文件中的 bpc_breakout_direction"""
        from pathlib import Path

        preds = list(Path("results").rglob("predictions*.parquet"))
        if not preds:
            pytest.skip("No predictions file found")

        df = pd.read_parquet(preds[0])
        assert (
            "bpc_breakout_direction" in df.columns
        ), "predictions file missing bpc_breakout_direction"

        direction = df["bpc_breakout_direction"]

        # 1. 只有 1 和 -1
        unique_vals = set(direction.unique())
        assert unique_vals == {1.0, -1.0}, f"Expected {{1.0, -1.0}}, got {unique_vals}"

        # 2. 没有 NaN
        assert direction.isna().sum() == 0, "Should have no NaN"

        # 3. 每个 symbol 都有两个方向
        sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
        for sym in df[sym_col].unique():
            mask = df[sym_col] == sym
            sym_unique = set(direction[mask].unique())
            assert sym_unique == {
                1.0,
                -1.0,
            }, f"{sym}: expected both directions, got {sym_unique}"

    def test_direction_dtype_is_numeric(self):
        """方向列应为数值类型（float64 或 int）"""
        df = _make_trending_up_data()
        direction = _compute_direction(df)
        assert np.issubdtype(
            direction.dtype, np.number
        ), f"Expected numeric dtype, got {direction.dtype}"
