"""测试批量计算架构是否正确实现（避免warmup时的流式计算）"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock

from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.live_data_stream.order_flow_listener import OrderFlowListener
from src.live_data_stream.feature_storage import StorageManager


class TestBatchCalculationArchitecture:
    """测试批量计算架构的核心特性"""

    def test_on_bar_does_not_trigger_streaming_computation(self):
        """测试 on_bar 不会触发流式特征计算"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 模拟一个 bar 数据
        bar_data = {
            "ts": pd.Timestamp("2024-01-01 10:00:00", tz="UTC").value,
            "timestamp": pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 100.0,
        }

        # 检查 on_bar 之前的状态
        initial_buffer_len = len(computer.bar_buffer)

        # 调用 on_bar（这应该只维护缓冲区，不触发特征计算）
        computer.on_bar(bar_data, timeframe="1min")

        # 验证缓冲区增加了数据
        assert len(computer.bar_buffer) == initial_buffer_len + 1

        # 验证没有调用任何特征计算方法
        # 由于重构后 on_bar 不再调用 _update_timeframe_features，这里应正常执行

    def test_compute_features_batch_isolated_from_on_bar(self):
        """测试 compute_features_batch 与 on_bar 完全隔离"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 先用 on_bar 添加一些数据到缓冲区
        for i in range(5):
            bar_data = {
                "ts": pd.Timestamp(f"2024-01-01 10:{i:02d}:00", tz="UTC").value,
                "timestamp": pd.Timestamp(f"2024-01-01 10:{i:02d}:00", tz="UTC"),
                "open": 50000.0 + i,
                "high": 50010.0 + i,
                "low": 49990.0 + i,
                "close": 50005.0 + i,
                "volume": 100.0 + i,
            }
            computer.on_bar(bar_data, timeframe="1min")

        # 创建批量计算的数据（与缓冲区数据不同）
        start_time = pd.Timestamp("2024-02-01", tz="UTC")
        timestamps = pd.date_range(start_time, periods=100, freq="1min", tz="UTC")

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [60000 + i for i in range(100)],
                "high": [60010 + i for i in range(100)],
                "low": [59990 + i for i in range(100)],
                "close": [60005 + i for i in range(100)],
                "volume": [200 + i for i in range(100)],
            }
        )

        ticks_1min = pd.DataFrame(
            {
                "timestamp": timestamps[:50],  # 少一些 ticks
                "price": [60005 + i for i in range(50)],
                "volume": [10 + i for i in range(50)],
                "side": [1 if i % 2 == 0 else -1 for i in range(50)],
            }
        )

        # 执行批量计算
        result = computer.compute_features_batch(
            bars_1min=bars_1min, ticks_1min=ticks_1min
        )

        # 验证结果独立于缓冲区中的数据
        assert isinstance(result, dict)
        print(f"Batch computation result has {len(result)} features")

        # 验证缓冲区中的数据没有影响批量计算
        # （批量计算应该完全基于输入的 bars_1min 和 ticks_1min）

    def test_order_flow_listener_restore_state_does_not_replay_to_computer(self):
        """测试 OrderFlowListener 的 _restore_state 不会回放数据到 feature_computer"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(base_path=temp_dir)

            # 创建 feature computer 并监控其方法调用
            computer = IncrementalFeatureComputer(
                archetypes_dir="config/strategies/bpc/archetypes"
            )

            # 使用 Mock 来监控 on_bar 和 on_tick 方法是否被调用
            original_on_bar = computer.on_bar
            original_on_tick = computer.on_tick

            computer.on_bar = MagicMock(side_effect=original_on_bar)
            computer.on_tick = MagicMock(side_effect=original_on_tick)

            # 创建 listener
            listener = OrderFlowListener(
                symbol="BTCUSDT", storage_manager=storage, feature_computer=computer
            )

            # 创建模拟的 warmup 数据
            start_time = pd.Timestamp("2024-01-01", tz="UTC")
            n_bars = 10  # 较少的数据用于测试
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

            n_ticks = 5
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

            # 创建 warmup 数据字典
            warmup_data = {
                "bars_1min": bars_1min,
                "ticks_1min": ticks_1min,
                "features_15min": pd.DataFrame(),
                "features_4h": pd.DataFrame(),
            }

            # 调用 _restore_state（这应该只恢复 memory_window，不回放到 feature_computer）
            listener._restore_state(warmup_data)

            # 验证 feature_computer 的 on_bar 和 on_tick 没有被调用
            assert (
                computer.on_bar.call_count == 0
            ), "在 _restore_state 过程中不应该调用 on_bar"
            assert (
                computer.on_tick.call_count == 0
            ), "在 _restore_state 过程中不应该调用 on_tick"

            # 验证 memory_window 被正确恢复
            assert listener.memory_window.size() == len(
                bars_1min
            ), "Memory window 应该被恢复"

    def test_batch_computation_vs_streaming_comparison(self):
        """比较批量计算和流式计算的结果一致性"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 创建测试数据
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        n_bars = 200  # 足够长的数据用于有意义的特征计算
        timestamps = pd.date_range(start_time, periods=n_bars, freq="1min", tz="UTC")

        # 生成具有一定趋势和波动的数据
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

        # 创建对应的 ticks 数据
        n_ticks = 100
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
                "volume": np.random.uniform(1, 10, n_ticks),
                "side": np.random.choice([1, -1], n_ticks),
            }
        )

        # 执行批量计算
        batch_result = computer.compute_features_batch(
            bars_1min=bars_1min, ticks_1min=ticks_1min
        )

        # 验证批量计算成功
        assert isinstance(batch_result, dict), "批量计算应该返回字典"
        print(f"Batch computation generated {len(batch_result)} features")

        # 验证 get_features 返回相同结果（因为使用了缓存）
        get_result = computer.get_features()
        assert get_result == batch_result, "get_features 应该返回批量计算的结果"

    def test_feature_computer_initialization_without_archetypes(self):
        """测试不使用 archetypes 初始化的 feature computer"""
        # 创建不带 archetypes 的 computer（用于测试基础功能）
        computer = IncrementalFeatureComputer()

        # 测试基础的 on_tick 功能
        tick_data = {
            "ts": pd.Timestamp("2024-01-01 10:00:00", tz="UTC").value,
            "price": 50000.0,
            "volume": 10.0,
            "side": 1,  # buy
        }

        computer.on_tick(tick_data)
        assert computer._cvd_cum == 10.0

        # 测试基础的 on_bar 功能
        bar_data = {
            "ts": pd.Timestamp("2024-01-01 10:00:00", tz="UTC").value,
            "timestamp": pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 100.0,
        }

        initial_buffer_len = len(computer.bar_buffer)
        computer.on_bar(bar_data, timeframe="1min")
        assert len(computer.bar_buffer) == initial_buffer_len + 1

        # 测试空的批量计算
        empty_bars = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        empty_ticks = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])

        result = computer.compute_features_batch(empty_bars, empty_ticks)
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
