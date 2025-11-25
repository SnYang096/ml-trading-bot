"""
标签生成器测试

测试四个策略的标签生成器：
1. SR Reversal: 二元标签（≥2R 成功率）
2. SR Breakout: 连续标签（实现 R/R）
3. Compression Breakout: 三元标签（方向+质量）
4. Trend Following: 百分位标签（Rank）
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label,
)
from src.time_series_model.strategies.labels.sr_breakout_label import (
    compute_sr_breakout_label,
)
from src.time_series_model.strategies.labels.compression_breakout_label import (
    compute_compression_breakout_label,
)
from src.time_series_model.strategies.labels.trend_following_label import (
    compute_trend_following_label,
)


class TestLabelGenerators(unittest.TestCase):
    """标签生成器测试类"""

    @classmethod
    def setUpClass(cls):
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 200

        # 创建价格数据
        price_base = 100.0
        returns = np.random.randn(n_samples) * 0.02
        prices = price_base + np.cumsum(returns)

        cls.test_df = pd.DataFrame(
            {
                "open": prices + np.random.randn(n_samples) * 0.1,
                "high": prices + np.abs(np.random.randn(n_samples) * 0.2),
                "low": prices - np.abs(np.random.randn(n_samples) * 0.2),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n_samples),
            }
        )

        # 确保 high >= close >= low
        cls.test_df["high"] = np.maximum(
            cls.test_df["high"], cls.test_df[["open", "close"]].max(axis=1)
        )
        cls.test_df["low"] = np.minimum(
            cls.test_df["low"], cls.test_df[["open", "close"]].min(axis=1)
        )

        # 计算 ATR
        high_low = cls.test_df["high"] - cls.test_df["low"]
        high_close = np.abs(cls.test_df["high"] - cls.test_df["close"].shift(1))
        low_close = np.abs(cls.test_df["low"] - cls.test_df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        cls.test_df["atr"] = tr.rolling(window=14, min_periods=1).mean()

        # 创建信号列
        cls.test_df["signal"] = 0
        # 在几个位置添加信号
        cls.test_df.loc[50, "signal"] = 1  # Long signal
        cls.test_df.loc[100, "signal"] = -1  # Short signal
        cls.test_df.loc[150, "signal"] = 1  # Long signal

    def test_sr_reversal_label_basic(self):
        """测试 SR Reversal 标签基本功能"""
        df = self.test_df.copy()

        labels = compute_sr_reversal_label(
            df,
            signal_col="signal",
            max_holding_bars=20,
            rr_ratio=2.0,
            auto_generate_signals=False,
        )

        # 检查返回类型
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(df))

        # 检查标签值范围
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            self.assertTrue(
                valid_labels.isin([0.0, 1.0]).all(),
                f"Labels should be 0.0 or 1.0, got: {valid_labels.unique()}",
            )

        # 检查有信号的位置应该有标签（或 NaN）
        signal_mask = df["signal"] != 0
        for idx in df[signal_mask].index:
            label_val = labels.loc[idx]
            # 应该要么是 0.0, 1.0，要么是 NaN（如果超时）
            self.assertTrue(
                pd.isna(label_val) or label_val in [0.0, 1.0],
                f"Label at {idx} should be 0.0, 1.0, or NaN, got {label_val}",
            )

    def test_sr_breakout_label_basic(self):
        """测试 SR Breakout 标签基本功能"""
        df = self.test_df.copy()

        labels = compute_sr_breakout_label(
            df,
            signal_col="signal",
            max_holding_bars=20,
            max_rr=3.0,
        )

        # 检查返回类型
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(df))

        # 检查标签值范围（应该是 0 到 max_rr，或 NaN）
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            self.assertTrue(
                (valid_labels >= 0).all() and (valid_labels <= 3.0).all(),
                f"Labels should be in [0, 3.0], got: min={valid_labels.min()}, max={valid_labels.max()}",
            )

    def test_compression_breakout_label_basic(self):
        """测试 Compression Breakout 标签基本功能"""
        df = self.test_df.copy()

        labels = compute_compression_breakout_label(
            df,
            lookback_window=10,
            confirmation_bars=3,
        )

        # 检查返回类型
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(df))

        # 检查标签值范围（应该是 -1, 0, 1，或 NaN）
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            self.assertTrue(
                valid_labels.isin([-1.0, 0.0, 1.0]).all(),
                f"Labels should be -1.0, 0.0, or 1.0, got: {valid_labels.unique()}",
            )

    def test_trend_following_label_basic(self):
        """测试 Trend Following 标签基本功能"""
        df = self.test_df.copy()

        labels = compute_trend_following_label(
            df,
            horizon=20,
            rank_window=50,
            min_periods=20,
        )

        # 检查返回类型
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(df))

        # 检查标签值范围（应该是 0.0 到 1.0，或 NaN）
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            self.assertTrue(
                (valid_labels >= 0.0).all() and (valid_labels <= 1.0).all(),
                f"Labels should be in [0.0, 1.0], got: min={valid_labels.min()}, max={valid_labels.max()}",
            )

    def test_sr_reversal_label_no_signal(self):
        """测试 SR Reversal 标签在没有信号时返回 NaN"""
        df = self.test_df.copy()
        df["signal"] = 0  # 没有信号

        labels = compute_sr_reversal_label(
            df,
            signal_col="signal",
            auto_generate_signals=False,
        )

        # 所有标签应该是 NaN
        self.assertTrue(
            labels.isna().all(),
            "All labels should be NaN when there are no signals",
        )

    def test_sr_reversal_label_missing_atr(self):
        """测试 SR Reversal 标签在缺少 ATR 时自动计算"""
        df = self.test_df.copy()
        df = df.drop(columns=["atr"])

        labels = compute_sr_reversal_label(
            df,
            signal_col="signal",
            atr_window=14,
            auto_generate_signals=False,
        )

        # 应该能正常计算
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(df))

    def test_sr_reversal_label_auto_signal_generation(self):
        """测试自动生成 SR 信号后能正确得到标签"""
        df = pd.DataFrame(
            {
                "open": [100.0, 99.8, 100.2, 101.5, 102.0, 102.3, 102.6, 103.0],
                "high": [100.5, 100.6, 102.8, 103.2, 103.5, 103.6, 104.0, 104.2],
                "low": [99.7, 99.3, 99.9, 100.9, 101.2, 101.8, 102.0, 102.5],
                "close": [100.2, 100.4, 102.0, 102.8, 103.2, 103.4, 103.8, 104.0],
                "atr": [1.0] * 8,
                "sr_strength_max": [0.0, 1.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
                "sqs_hal_low": [0.0, 0.9, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                "sqs_hal_high": [0.0] * 8,
                "vpvr_pvp": [99.5] * 8,
            }
        )

        labels = compute_sr_reversal_label(
            df,
            signal_col="signal",
            max_holding_bars=4,
        )

        valid_labels = labels.dropna()
        self.assertTrue(
            (valid_labels == 1.0).all(),
            f"Auto-generated SR signals should yield success labels, got {valid_labels.unique()}",
        )

    def test_label_index_alignment(self):
        """测试标签索引对齐"""
        df = self.test_df.copy()

        # 测试所有标签生成器
        label_funcs = [
            lambda d: compute_sr_reversal_label(
                d, signal_col="signal", auto_generate_signals=False
            ),
            lambda d: compute_sr_breakout_label(d, signal_col="signal"),
            lambda d: compute_compression_breakout_label(d),
            lambda d: compute_trend_following_label(d, horizon=20),
        ]

        for label_func in label_funcs:
            labels = label_func(df)
            # 检查索引对齐
            self.assertTrue(
                labels.index.equals(df.index),
                f"Label index should match DataFrame index for {label_func}",
            )


if __name__ == "__main__":
    unittest.main()
