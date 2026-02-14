#!/usr/bin/env python3
"""
测试批量计算的磁盘+Buffer合并逻辑

验证 OrderFlowListener 的 v2 优化：
1. _merge_bars(): 合并磁盘bars + memory_window，按timestamp去重
2. _merge_ticks(): 合并磁盘ticks + tick_buffer，保留全部
3. _get_tick_buffer_df(): 从feature_computer.tick_buffer提取DataFrame
4. _compute_and_save_15min_features(): 完整流程集成测试
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.live_data_stream.order_flow_listener import OrderFlowListener
from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.memory_window import MemoryWindow
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


class TestBatchMergeBuffer:
    """测试批量计算的Buffer合并逻辑"""

    @pytest.fixture
    def base_timestamp(self):
        """基准时间戳"""
        return pd.Timestamp("2026-02-13 10:00:00", tz="UTC")

    @pytest.fixture
    def mock_storage_manager(self):
        """Mock存储管理器"""
        storage = Mock(spec=StorageManager)
        storage.bar_1min = Mock()
        storage.ticks = Mock()
        storage.feature_15min = Mock()
        return storage

    @pytest.fixture
    def mock_feature_computer(self):
        """Mock特征计算器"""
        computer = Mock(spec=IncrementalFeatureComputer)
        computer.primary_timeframe = "240T"
        computer.tick_buffer = []
        computer.compute_features_batch = Mock(return_value={})
        return computer

    @pytest.fixture
    def listener(self, mock_storage_manager, mock_feature_computer):
        """创建OrderFlowListener实例"""
        return OrderFlowListener(
            symbol="BTCUSDT",
            storage_manager=mock_storage_manager,
            feature_computer=mock_feature_computer,
            memory_window_hours=4.0,
        )

    # ================================================================
    # Test _merge_bars()
    # ================================================================

    def test_merge_bars_empty_both(self, listener):
        """测试：磁盘和buffer都为空"""
        result = listener._merge_bars(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_merge_bars_empty_disk(self, listener, base_timestamp):
        """测试：磁盘为空，只有buffer"""
        buffer_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "open": 100, "close": 101},
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 101,
                    "close": 102,
                },
            ]
        )

        result = listener._merge_bars(pd.DataFrame(), buffer_df)

        assert len(result) == 2
        assert result["close"].tolist() == [101, 102]

    def test_merge_bars_empty_buffer(self, listener, base_timestamp):
        """测试：buffer为空，只有磁盘"""
        disk_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "open": 100, "close": 101},
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 101,
                    "close": 102,
                },
            ]
        )

        result = listener._merge_bars(disk_df, pd.DataFrame())

        assert len(result) == 2
        assert result["close"].tolist() == [101, 102]

    def test_merge_bars_no_overlap(self, listener, base_timestamp):
        """测试：磁盘和buffer无重叠（buffer更新）"""
        disk_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "open": 100, "close": 101},
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 101,
                    "close": 102,
                },
            ]
        )

        buffer_df = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp + timedelta(minutes=2),
                    "open": 102,
                    "close": 103,
                },
                {
                    "timestamp": base_timestamp + timedelta(minutes=3),
                    "open": 103,
                    "close": 104,
                },
            ]
        )

        result = listener._merge_bars(disk_df, buffer_df)

        assert len(result) == 4
        assert result["close"].tolist() == [101, 102, 103, 104]

    def test_merge_bars_with_overlap(self, listener, base_timestamp):
        """测试：磁盘和buffer有重叠时间戳（keep='last'保留buffer）"""
        disk_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "open": 100, "close": 101},
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 101,
                    "close": 102,
                },
                {
                    "timestamp": base_timestamp + timedelta(minutes=2),
                    "open": 102,
                    "close": 103,
                },
            ]
        )

        # buffer有更新的数据（同一时间戳，不同close价格）
        buffer_df = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp + timedelta(minutes=2),
                    "open": 102,
                    "close": 103.5,
                },
                {
                    "timestamp": base_timestamp + timedelta(minutes=3),
                    "open": 103.5,
                    "close": 104,
                },
            ]
        )

        result = listener._merge_bars(disk_df, buffer_df)

        # 应该有4条记录（去重后）
        assert len(result) == 4
        # 第3条应该是buffer的数据（close=103.5）
        assert result.iloc[2]["close"] == 103.5
        assert result.iloc[3]["close"] == 104

    def test_merge_bars_timestamp_conversion(self, listener, base_timestamp):
        """测试：timestamp格式转换（字符串 -> Timestamp）"""
        disk_df = pd.DataFrame(
            [
                {"timestamp": "2026-02-13 10:00:00", "open": 100, "close": 101},
            ]
        )

        buffer_df = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 101,
                    "close": 102,
                },
            ]
        )

        result = listener._merge_bars(disk_df, buffer_df)

        assert len(result) == 2
        assert isinstance(result.iloc[0]["timestamp"], pd.Timestamp)
        assert result.iloc[0]["timestamp"].tz is not None  # UTC

    # ================================================================
    # Test _merge_ticks()
    # ================================================================

    def test_merge_ticks_empty_both(self, listener):
        """测试：磁盘和buffer都为空"""
        result = listener._merge_ticks(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_merge_ticks_no_dedup(self, listener, base_timestamp):
        """测试：ticks不去重（同一毫秒多笔交易）"""
        disk_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "price": 100, "volume": 1.0, "side": 1},
                {
                    "timestamp": base_timestamp,
                    "price": 100.1,
                    "volume": 2.0,
                    "side": -1,
                },
            ]
        )

        buffer_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp, "price": 100.2, "volume": 1.5, "side": 1},
                {
                    "timestamp": base_timestamp + timedelta(milliseconds=1),
                    "price": 100.3,
                    "volume": 3.0,
                    "side": -1,
                },
            ]
        )

        result = listener._merge_ticks(disk_df, buffer_df)

        # 应该有4条（不去重）
        assert len(result) == 4
        # 按时间排序
        assert result["price"].tolist() == [100, 100.1, 100.2, 100.3]

    def test_merge_ticks_sorted_by_timestamp(self, listener, base_timestamp):
        """测试：ticks按timestamp排序"""
        # 故意打乱顺序
        disk_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp + timedelta(seconds=2), "price": 102},
                {"timestamp": base_timestamp, "price": 100},
            ]
        )

        buffer_df = pd.DataFrame(
            [
                {"timestamp": base_timestamp + timedelta(seconds=3), "price": 103},
                {"timestamp": base_timestamp + timedelta(seconds=1), "price": 101},
            ]
        )

        result = listener._merge_ticks(disk_df, buffer_df)

        # 应该按时间排序
        assert result["price"].tolist() == [100, 101, 102, 103]

    # ================================================================
    # Test _get_tick_buffer_df()
    # ================================================================

    def test_get_tick_buffer_df_empty(self, listener):
        """测试：tick_buffer为空"""
        listener.feature_computer.tick_buffer = []
        result = listener._get_tick_buffer_df()
        assert result.empty

    def test_get_tick_buffer_df_no_attribute(self, listener):
        """测试：feature_computer没有tick_buffer属性"""
        delattr(listener.feature_computer, "tick_buffer")
        result = listener._get_tick_buffer_df()
        assert result.empty

    def test_get_tick_buffer_df_valid(self, listener, base_timestamp):
        """测试：从tick_buffer提取DataFrame"""
        # 模拟tick_buffer数据（纳秒时间戳格式）
        ts_ns = base_timestamp.value  # 纳秒
        listener.feature_computer.tick_buffer = [
            {"ts": ts_ns, "price": 100.0, "volume": 1.0, "side": 1},
            {"ts": ts_ns + 1_000_000_000, "price": 101.0, "volume": 2.0, "side": -1},
        ]

        result = listener._get_tick_buffer_df()

        assert len(result) == 2
        assert "timestamp" in result.columns
        assert result["price"].tolist() == [100.0, 101.0]
        assert isinstance(result.iloc[0]["timestamp"], pd.Timestamp)

    # ================================================================
    # Test _compute_and_save_15min_features() 集成测试
    # ================================================================

    def test_compute_features_with_merge(self, listener, base_timestamp):
        """测试：完整流程 - 磁盘+buffer合并后批量计算"""
        # 模拟磁盘数据
        disk_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "open": 100 - i,
                    "high": 101 - i,
                    "low": 99 - i,
                    "close": 100 - i,
                    "volume": 1000,
                }
                for i in range(100, 0, -1)
            ]
        )
        # 提供足够的tick数据（>=20160条）
        disk_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 10,
                    "side": 1,
                }
                for i in range(25000, 0, -1)
            ]
        )

        listener.storage_manager.bar_1min.load_range = Mock(return_value=disk_bars)
        listener.storage_manager.ticks.load_range = Mock(return_value=disk_ticks)

        # 模拟内存buffer
        buffer_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp,
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100.5,
                    "volume": 1200,
                },
                {
                    "timestamp": base_timestamp + timedelta(minutes=1),
                    "open": 100.5,
                    "high": 101.5,
                    "low": 100,
                    "close": 101,
                    "volume": 1100,
                },
            ]
        )
        listener.memory_window._data = []
        for _, row in buffer_bars.iterrows():
            listener.memory_window.add(row.to_dict())

        # 模拟tick_buffer
        listener.feature_computer.tick_buffer = [
            {"ts": base_timestamp.value, "price": 100.5, "volume": 5, "side": 1},
            {
                "ts": (base_timestamp + timedelta(seconds=30)).value,
                "price": 101.0,
                "volume": 3,
                "side": -1,
            },
        ]

        # Mock批量计算返回
        listener.feature_computer.compute_features_batch = Mock(
            return_value={
                "close": 101.0,
                "atr": 0.5,
                "rsi": 55.0,
            }
        )

        # Mock保存
        listener.storage_manager.save_15min_features = Mock()
        listener._handle_features = Mock()

        # 执行
        with patch("pandas.Timestamp.now", return_value=base_timestamp):
            listener._compute_and_save_15min_features()

        # 验证：compute_features_batch被调用
        assert listener.feature_computer.compute_features_batch.called

        # 验证：传入的bars包含磁盘+buffer数据
        call_args = listener.feature_computer.compute_features_batch.call_args
        bars_merged = call_args.kwargs["bars_1min"]
        ticks_merged = call_args.kwargs["ticks_1min"]

        # bars应该有102条（100磁盘 + 2buffer）
        assert len(bars_merged) == 102
        # 最后一条应该是buffer的数据（close=101）
        assert bars_merged.iloc[-1]["close"] == 101

        # ticks应该有25002条（25000磁盘 + 2buffer）
        assert len(ticks_merged) == 25002

        # 验证：保存被调用
        assert listener.storage_manager.save_15min_features.called
        assert listener._handle_features.called

    def test_compute_features_disk_only(self, listener, base_timestamp):
        """测试：只有磁盘数据，buffer为空"""
        disk_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "open": 100 - i,
                    "high": 101 - i,
                    "low": 99 - i,
                    "close": 100 - i,
                    "volume": 1000,
                }
                for i in range(10, 0, -1)
            ]
        )
        # 提供足够的tick数据（>=20160条）
        disk_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 10,
                    "side": 1,
                }
                for i in range(25000, 0, -1)
            ]
        )

        listener.storage_manager.bar_1min.load_range = Mock(return_value=disk_bars)
        listener.storage_manager.ticks.load_range = Mock(return_value=disk_ticks)

        # buffer为空
        listener.memory_window._data = []
        listener.feature_computer.tick_buffer = []

        listener.feature_computer.compute_features_batch = Mock(
            return_value={"close": 100}
        )
        listener.storage_manager.save_15min_features = Mock()
        listener._handle_features = Mock()

        with patch("pandas.Timestamp.now", return_value=base_timestamp):
            listener._compute_and_save_15min_features()

        # 验证：只用磁盘数据
        call_args = listener.feature_computer.compute_features_batch.call_args
        bars_merged = call_args.kwargs["bars_1min"]

        assert len(bars_merged) == 10  # 只有磁盘数据

    def test_compute_features_buffer_only(self, listener, base_timestamp):
        """测试：只有buffer数据，磁盘为空（首次启动场景）—— 应该报错"""
        listener.storage_manager.bar_1min.load_range = Mock(return_value=pd.DataFrame())
        # 两次调用都返回空（近8天和100天扩展）
        listener.storage_manager.ticks.load_range = Mock(return_value=pd.DataFrame())

        # 只有buffer
        buffer_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp,
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 1000,
                },
            ]
        )
        listener.memory_window._data = []
        for _, row in buffer_bars.iterrows():
            listener.memory_window.add(row.to_dict())

        listener.feature_computer.tick_buffer = [
            {"ts": base_timestamp.value, "price": 100, "volume": 5, "side": 1},
        ]

        # 应该抛出RuntimeError（tick数据不足或bars为空）
        with pytest.raises(RuntimeError, match=r"(tick数据不足|磁盘bars数据为空)"):
            with patch("pandas.Timestamp.now", return_value=base_timestamp):
                listener._compute_and_save_15min_features()


class TestTickSave:
    """测试tick数据保存逻辑"""

    @pytest.fixture
    def base_timestamp(self):
        return pd.Timestamp("2026-02-13 10:00:00", tz="UTC")

    @pytest.fixture
    def mock_storage_manager(self):
        storage = Mock(spec=StorageManager)
        storage.bar_1min = Mock()
        storage.ticks = Mock()
        storage.feature_15min = Mock()
        return storage

    @pytest.fixture
    def mock_feature_computer(self):
        computer = Mock(spec=IncrementalFeatureComputer)
        computer.primary_timeframe = "240T"
        computer.tick_buffer = []
        return computer

    @pytest.fixture
    def listener(self, mock_storage_manager, mock_feature_computer):
        return OrderFlowListener(
            symbol="BTCUSDT",
            storage_manager=mock_storage_manager,
            feature_computer=mock_feature_computer,
            memory_window_hours=4.0,
        )

    def test_save_1min_ticks_calls_storage_append(self, listener, base_timestamp):
        """测试：_save_1min_ticks() 正确调用 storage_manager.ticks.append()"""
        # 设置tick_1min_buffer
        listener.tick_1min_buffer = {
            "start_time": base_timestamp,
            "buy_ticks": [
                {"timestamp": base_timestamp, "price": 100.0, "volume": 1.0},
                {
                    "timestamp": base_timestamp + timedelta(seconds=10),
                    "price": 100.5,
                    "volume": 2.0,
                },
            ],
            "sell_ticks": [
                {
                    "timestamp": base_timestamp + timedelta(seconds=5),
                    "price": 100.2,
                    "volume": 1.5,
                },
            ],
        }

        # 执行
        listener._save_1min_ticks()

        # 验证：ticks.append被调用
        assert listener.storage_manager.ticks.append.called

        call_args = listener.storage_manager.ticks.append.call_args
        symbol = call_args[0][0]
        trading_date = call_args[0][1]
        tick_df = call_args[0][2]

        assert symbol == "BTCUSDT"
        assert trading_date == "2026-02-13"
        assert len(tick_df) == 2  # 1 buy + 1 sell
        assert tick_df[tick_df["side"] == 1]["volume"].iloc[0] == 3.0  # buy total
        assert tick_df[tick_df["side"] == -1]["volume"].iloc[0] == 1.5  # sell total

    def test_save_1min_ticks_empty_buffer(self, listener):
        """测试：空buffer不调用保存"""
        listener.tick_1min_buffer = {"start_time": None}

        listener._save_1min_ticks()

        # 不应该调用append
        assert not listener.storage_manager.ticks.append.called

    def test_save_1min_ticks_only_buy(self, listener, base_timestamp):
        """测试：只有买方tick"""
        listener.tick_1min_buffer = {
            "start_time": base_timestamp,
            "buy_ticks": [
                {"timestamp": base_timestamp, "price": 100.0, "volume": 2.0},
            ],
            "sell_ticks": [],
        }

        listener._save_1min_ticks()

        call_args = listener.storage_manager.ticks.append.call_args
        tick_df = call_args[0][2]

        assert len(tick_df) == 1
        assert tick_df.iloc[0]["side"] == 1


class TestTickLoadExtend:
    """测试tick数据加载扩展逻辑"""

    @pytest.fixture
    def base_timestamp(self):
        return pd.Timestamp("2026-02-13 10:00:00", tz="UTC")

    @pytest.fixture
    def mock_storage_manager(self):
        storage = Mock(spec=StorageManager)
        storage.bar_1min = Mock()
        storage.ticks = Mock()
        storage.feature_15min = Mock()
        return storage

    @pytest.fixture
    def mock_feature_computer(self):
        computer = Mock(spec=IncrementalFeatureComputer)
        computer.primary_timeframe = "240T"
        computer.tick_buffer = []
        computer.compute_features_batch = Mock(return_value={})
        return computer

    @pytest.fixture
    def listener(self, mock_storage_manager, mock_feature_computer):
        return OrderFlowListener(
            symbol="BTCUSDT",
            storage_manager=mock_storage_manager,
            feature_computer=mock_feature_computer,
            memory_window_hours=4.0,
        )

    def test_tick_load_extends_when_insufficient(self, listener, base_timestamp):
        """测试：ticks不足时自动扩展加载范围"""
        # 近8天只有166条（数据缺口）
        recent_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 1,
                    "side": 1,
                }
                for i in range(166)
            ]
        )

        # 100天范围有足够数据（25000条）
        extended_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 1,
                    "side": 1,
                }
                for i in range(25000)
            ]
        )

        # 第一次调用返回166条，第二次返回25000条
        listener.storage_manager.ticks.load_range = Mock(
            side_effect=[recent_ticks, extended_ticks]
        )

        # 准备其他数据
        disk_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 1000,
                }
                for i in range(100)
            ]
        )
        listener.storage_manager.bar_1min.load_range = Mock(return_value=disk_bars)
        listener.memory_window._data = []
        listener.feature_computer.tick_buffer = []
        listener.storage_manager.save_15min_features = Mock()
        listener._handle_features = Mock()

        with patch("pandas.Timestamp.now", return_value=base_timestamp):
            listener._compute_and_save_15min_features()

        # 验证：ticks.load_range被调用两次（第二次是扩展加载）
        assert listener.storage_manager.ticks.load_range.call_count == 2

        # 验证：传给9compute_features_batch的ticks是扩展后的数据
        call_args = listener.feature_computer.compute_features_batch.call_args
        ticks_merged = call_args.kwargs["ticks_1min"]
        assert len(ticks_merged) == 25000

    def test_tick_load_raises_when_insufficient_after_extend(
        self, listener, base_timestamp
    ):
        """测试：扩展后ticks仍不足时报错退出"""
        # 近8天只有100条
        recent_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 1,
                    "side": 1,
                }
                for i in range(100)
            ]
        )

        # 100天范围也只有1000条（不足20160）
        extended_ticks = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "price": 100,
                    "volume": 1,
                    "side": 1,
                }
                for i in range(1000)
            ]
        )

        listener.storage_manager.ticks.load_range = Mock(
            side_effect=[recent_ticks, extended_ticks]
        )

        # 准备其他数据
        disk_bars = pd.DataFrame(
            [
                {
                    "timestamp": base_timestamp - timedelta(minutes=i),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 1000,
                }
                for i in range(100)
            ]
        )
        listener.storage_manager.bar_1min.load_range = Mock(return_value=disk_bars)
        listener.memory_window._data = []
        listener.feature_computer.tick_buffer = []

        # 应该抛出RuntimeError
        with pytest.raises(RuntimeError, match=r"tick数据不足.*fill-gap"):
            with patch("pandas.Timestamp.now", return_value=base_timestamp):
                listener._compute_and_save_15min_features()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
