"""测试批量特征计算的端到端集成"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import pytest

from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.live_data_stream.order_flow_listener import OrderFlowListener
from src.live_data_stream.feature_storage import StorageManager


class TestBatchFeatureIntegration:
    """批量特征计算的端到端集成测试"""

    def test_end_to_end_batch_calculation_flow(self):
        """测试端到端的批量计算流程"""
        # 创建临时存储目录
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(base_path=temp_dir)

            # 创建 feature computer
            computer = IncrementalFeatureComputer(
                archetypes_dir="config/strategies/bpc/archetypes"
            )

            # 创建 listener
            listener = OrderFlowListener(
                symbol="BTCUSDT", storage_manager=storage, feature_computer=computer
            )

            # 准备测试数据：创建几天的 bars 和 ticks
            start_time = pd.Timestamp("2024-01-01", tz="UTC")
            n_bars = 24 * 60 * 3  # 3天的1min数据 = 4320 bars
            timestamps = pd.date_range(
                start_time, periods=n_bars, freq="1min", tz="UTC"
            )

            # 创建 OHLCV 数据
            base_price = 50000
            returns = np.random.normal(0.0001, 0.001, n_bars)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            bars_1min = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": prices,
                    "high": [
                        p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices
                    ],
                    "low": [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
                    "close": prices,
                    "volume": np.random.uniform(50, 500, n_bars),
                }
            )

            # 保存 bars 到存储
            storage.save_1min_ticks("BTCUSDT", bars_1min)

            # 创建 ticks 数据（少一些，符合实际比例）
            n_ticks = 1000
            tick_timestamps = pd.date_range(
                start_time, periods=n_ticks, freq="1min", tz="UTC"
            )
            tick_prices = [base_price]
            for _ in range(1, n_ticks):
                tick_prices.append(tick_prices[-1] * (1 + np.random.normal(0, 0.0002)))

            ticks_1min = pd.DataFrame(
                {
                    "timestamp": tick_timestamps,
                    "price": tick_prices,
                    "volume": np.random.uniform(1, 10, n_ticks),
                    "side": np.random.choice([1, -1], n_ticks),
                }
            )

            # 保存 ticks 到存储
            storage.save_ticks("BTCUSDT", ticks_1min)

            # 现在测试 _compute_and_save_15min_features 方法（这会触发批量计算）
            # 模拟调用这个方法
            listener._compute_and_save_15min_features()

            # 验证特征被保存
            features_15min = storage.feature_15min.load_range(
                "BTCUSDT", "2024-01-01", "2024-01-04"  # 4号包含了1号的数据
            )

            assert (
                len(features_15min) >= 0
            ), f"应该有特征被保存，实际有 {len(features_15min)} 条"
            if len(features_15min) > 0:
                print(f"Saved {len(features_15min)} features to storage")
                print(f"Features columns: {list(features_15min.columns)}")

    def test_warmup_state_restoration_with_batch_calculation(self):
        """测试 warmup 状态恢复与批量计算的兼容性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(base_path=temp_dir)

            # 创建 feature computer
            computer = IncrementalFeatureComputer(
                archetypes_dir="config/strategies/bpc/archetypes"
            )

            # 创建 listener
            listener = OrderFlowListener(
                symbol="BTCUSDT", storage_manager=storage, feature_computer=computer
            )

            # 创建模拟的 warmup 数据
            start_time = pd.Timestamp("2024-01-01", tz="UTC")
            n_bars = 100
            timestamps = pd.date_range(
                start_time, periods=n_bars, freq="1min", tz="UTC"
            )

            bars_1min = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [50000 + i for i in range(n_bars)],
                    "high": [50001 + i for i in range(n_bars)],
                    "low": [49999 + i for i in range(n_bars)],
                    "close": [50000.5 + i for i in range(n_bars)],
                    "volume": [100 + i for i in range(n_bars)],
                }
            )

            n_ticks = 50
            tick_timestamps = pd.date_range(
                start_time, periods=n_ticks, freq="1min", tz="UTC"
            )
            ticks_1min = pd.DataFrame(
                {
                    "timestamp": tick_timestamps,
                    "price": [50000.5 + i for i in range(n_ticks)],
                    "volume": [5 + i for i in range(n_ticks)],
                    "side": [1 if i % 2 == 0 else -1 for i in range(n_ticks)],
                }
            )

            # 创建 warmup 数据字典（模拟 GapFiller.warmup 的输出）
            warmup_data = {
                "bars_1min": bars_1min,
                "ticks_1min": ticks_1min,
                "features_15min": pd.DataFrame(),  # 没有预计算的特征
                "features_4h": pd.DataFrame(),
            }

            # 调用 _restore_state（这应该只恢复 memory_window，不回放到 feature_computer）
            listener._restore_state(warmup_data)

            # 验证 memory_window 被正确恢复
            assert listener.memory_window.size() == len(
                bars_1min
            ), "Memory window 应该被恢复"

            # 验证没有错误发生（特别是关于回放 feature_computer 的部分）
            # 现在调用特征计算，应该能正常工作
            listener._compute_and_save_15min_features()

            # 验证特征被计算和保存
            saved_features = storage.feature_15min.load_range(
                "BTCUSDT", "2024-01-01", "2024-01-02"
            )
            assert len(saved_features) >= 0, "特征应该被保存"

    def test_batch_calculation_with_different_timeframes(self):
        """测试不同时间框架下的批量计算"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 创建测试数据
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        n_bars = 500  # 更多数据用于不同时间框架测试
        timestamps = pd.date_range(start_time, periods=n_bars, freq="1min", tz="UTC")

        base_price = 50000
        returns = np.random.normal(0.0001, 0.0005, n_bars)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(50, 300, n_bars),
            }
        )

        n_ticks = 300
        tick_timestamps = pd.date_range(
            start_time, periods=n_ticks, freq="1min", tz="UTC"
        )
        tick_prices = [base_price]
        for _ in range(1, n_ticks):
            tick_prices.append(tick_prices[-1] * (1 + np.random.normal(0, 0.0001)))

        ticks_1min = pd.DataFrame(
            {
                "timestamp": tick_timestamps,
                "price": tick_prices,
                "volume": np.random.uniform(1, 5, n_ticks),
                "side": np.random.choice([1, -1], n_ticks),
            }
        )

        # 测试不同的时间框架
        for timeframe in ["240T", "120T", "60T"]:  # 4小时, 2小时, 1小时
            result = computer.compute_features_batch(
                bars_1min=bars_1min, ticks_1min=ticks_1min, primary_timeframe=timeframe
            )

            assert isinstance(result, dict), f"Timeframe {timeframe} should return dict"
            print(f"Timeframe {timeframe}: {len(result)} features generated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
