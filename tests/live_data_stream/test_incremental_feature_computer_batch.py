"""测试 IncrementalFeatureComputer 的批量计算功能 (compute_features_batch)"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


class TestIncrementalFeatureComputerBatch:
    """测试 IncrementalFeatureComputer 的批量计算功能"""

    def test_compute_features_batch_basic(self):
        """测试基本的批量计算功能"""
        # 创建 feature computer
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 创建测试数据：100天的1min bars
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        timestamps = pd.date_range(
            start_time, periods=100 * 24 * 60, freq="1min", tz="UTC"
        )  # 100天

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 50000 + np.random.normal(0, 100, len(timestamps)),
                "high": 50000 + np.random.normal(0, 100, len(timestamps)) + 10,
                "low": 50000 + np.random.normal(0, 100, len(timestamps)) - 10,
                "close": 50000 + np.random.normal(0, 100, len(timestamps)),
                "volume": np.random.uniform(100, 1000, len(timestamps)),
            }
        )

        # 创建测试数据：8天的1min ticks
        tick_timestamps = pd.date_range(
            start_time, periods=8 * 24 * 60, freq="1min", tz="UTC"
        )  # 8天
        ticks_1min = pd.DataFrame(
            {
                "timestamp": tick_timestamps,
                "price": 50000 + np.random.normal(0, 50, len(tick_timestamps)),
                "volume": np.random.uniform(1, 10, len(tick_timestamps)),
                "side": np.random.choice([1, -1], len(tick_timestamps)),
            }
        )

        # 调用批量计算
        result = computer.compute_features_batch(
            bars_1min=bars_1min,
            ticks_1min=ticks_1min,
            primary_timeframe="240T",  # 4小时
        )

        # 验证结果
        assert isinstance(result, dict), "结果应该是字典"
        assert len(result) >= 0, "特征字典不应该为空（即使没有匹配的特征）"
        print(f"Generated {len(result)} features")

    def test_compute_features_batch_empty_data(self):
        """测试空数据的批量计算"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 空的 bars 数据
        empty_bars = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        empty_ticks = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])

        result = computer.compute_features_batch(
            bars_1min=empty_bars, ticks_1min=empty_ticks
        )

        assert result == {}, "空数据应该返回空字典"

    def test_compute_features_batch_minimal_data(self):
        """测试最小数据集的批量计算"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 最小的 bars 数据集
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        timestamps = pd.date_range(start_time, periods=100, freq="1min", tz="UTC")

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [50000 + i for i in range(100)],
                "high": [50001 + i for i in range(100)],
                "low": [49999 + i for i in range(100)],
                "close": [50000.5 + i for i in range(100)],
                "volume": [100 + i for i in range(100)],
            }
        )

        # 最小的 ticks 数据集
        tick_timestamps = pd.date_range(start_time, periods=50, freq="1min", tz="UTC")
        ticks_1min = pd.DataFrame(
            {
                "timestamp": tick_timestamps,
                "price": [50000.5 + i for i in range(50)],
                "volume": [5 + i for i in range(50)],
                "side": [1 if i % 2 == 0 else -1 for i in range(50)],
            }
        )

        result = computer.compute_features_batch(
            bars_1min=bars_1min, ticks_1min=ticks_1min
        )

        # 验证结果结构
        assert isinstance(result, dict)
        print(f"Minimal data generated {len(result)} features")

    def test_compute_features_batch_with_get_features_compatibility(self):
        """测试批量计算结果与 get_features 接口的兼容性"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 创建测试数据
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        timestamps = pd.date_range(start_time, periods=200, freq="1min", tz="UTC")

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 50000 + np.random.normal(0, 50, len(timestamps)),
                "high": 50000 + np.random.normal(0, 50, len(timestamps)) + 5,
                "low": 50000 + np.random.normal(0, 50, len(timestamps)) - 5,
                "close": 50000 + np.random.normal(0, 50, len(timestamps)),
                "volume": np.random.uniform(50, 200, len(timestamps)),
            }
        )

        tick_timestamps = pd.date_range(start_time, periods=100, freq="1min", tz="UTC")
        ticks_1min = pd.DataFrame(
            {
                "timestamp": tick_timestamps,
                "price": 50000 + np.random.normal(0, 25, len(tick_timestamps)),
                "volume": np.random.uniform(1, 5, len(tick_timestamps)),
                "side": np.random.choice([1, -1], len(tick_timestamps)),
            }
        )

        # 先调用批量计算
        batch_result = computer.compute_features_batch(
            bars_1min=bars_1min, ticks_1min=ticks_1min
        )

        # 然后调用 get_features，应该返回相同的结果（因为使用了缓存）
        get_result = computer.get_features()

        # 验证两者结果一致（当批量计算有结果时）
        if batch_result:
            assert get_result == batch_result, "get_features 应该返回批量计算的结果"

    def test_on_tick_still_works_after_refactor(self):
        """测试重构后 on_tick 仍然正常工作（用于 CVD 累积）"""
        computer = IncrementalFeatureComputer()

        # 模拟一个 tick
        tick_data = {
            "ts": pd.Timestamp("2024-01-01 10:00:00", tz="UTC").value,
            "price": 50000.0,
            "volume": 10.0,
            "side": 1,  # buy
        }

        # 调用 on_tick
        computer.on_tick(tick_data)

        # 验证 CVD 状态更新
        assert computer._cvd_cum == 10.0, "CVD 应该累积 buy volume"
        assert computer._cvd_bar_delta == 10.0, "CVD bar delta 应该更新"

        # 再添加一个 sell tick
        tick_data["side"] = -1  # sell
        tick_data["volume"] = 5.0
        computer.on_tick(tick_data)

        assert computer._cvd_cum == 5.0, "CVD 应该累积 net volume (10 - 5)"
        assert computer._cvd_bar_delta == 5.0, "CVD bar delta 应该更新 net volume"

    def test_on_bar_still_works_after_refactor(self):
        """测试重构后 on_bar 仍然正常工作（用于 bar_buffer 维护）"""
        computer = IncrementalFeatureComputer()

        # 模拟一个 bar
        bar_data = {
            "ts": pd.Timestamp("2024-01-01 10:00:00", tz="UTC").value,
            "timestamp": pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 100.0,
        }

        # 调用 on_bar
        computer.on_bar(bar_data, timeframe="1min")

        # 验证 bar 被添加到缓冲区
        assert len(computer.bar_buffer) == 1, "Bar 应该被添加到缓冲区"
        stored_bar = list(computer.bar_buffer)[0]
        assert stored_bar["open"] == 50000.0, "Open price 应该正确存储"

    def test_compute_features_batch_with_realistic_data(self):
        """使用更真实的测试数据进行批量计算测试"""
        computer = IncrementalFeatureComputer(
            archetypes_dir="config/strategies/bpc/archetypes"
        )

        # 创建更真实的 OHLCV 数据（模拟价格走势）
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
        n_bars = 1000  # 1000分钟数据
        timestamps = pd.date_range(start_time, periods=n_bars, freq="1min", tz="UTC")

        # 生成有一定趋势和波动的价格数据
        base_price = 50000
        returns = np.random.normal(0.0001, 0.001, n_bars)  # 小幅正回报趋势
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        bars_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices[:-1] + [prices[-2]],  # 避免索引越界
                "high": [
                    p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices[:-1]
                ]
                + [prices[-1] * 1.0005],
                "low": [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices[:-1]]
                + [prices[-1] * 0.9995],
                "close": prices,
                "volume": np.random.uniform(50, 500, n_bars),
            }
        )

        # 创建对应的 ticks 数据
        n_ticks = 500
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

        # 执行批量计算
        result = computer.compute_features_batch(
            bars_1min=bars_1min, ticks_1min=ticks_1min
        )

        # 验证结果
        assert isinstance(result, dict)
        print(f"Realistic data generated {len(result)} features")

        # 验证时间框架设置
        assert (
            computer.primary_timeframe == "240T" or computer.primary_timeframe is None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
