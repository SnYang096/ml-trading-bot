#!/usr/bin/env python3
"""
测试短时断线补 ticks 功能

测试要点：
1. fill_missing_ticks 正确下载缺失数据
2. 已有的 ticks 不会被覆盖
3. merge_ticks 正确去重和排序
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from unittest.mock import Mock, patch

from src.live_data_stream.gap_filler import GapFiller
from src.live_data_stream.data_gap_filler import DataGapFiller


class TestFillMissingTicks:
    """测试 fill_missing_ticks 方法"""

    def test_skip_existing_ticks(self):
        """测试：已有的 ticks 不会被重复添加"""
        # 准备已有数据
        existing_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00.100",
                        "2024-01-01 10:00:00.200",
                        "2024-01-01 10:00:00.300",
                    ],
                    utc=True,
                ),
                "price": [50000.0, 50001.0, 50002.0],
                "volume": [1.0, 2.0, 3.0],
                "side": [1, -1, 1],
            }
        )

        # 模拟下载的数据（包含部分重复）
        downloaded_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00.200",  # 重复
                        "2024-01-01 10:00:00.400",  # 新的
                        "2024-01-01 10:00:00.500",  # 新的
                    ],
                    utc=True,
                ),
                "price": [50001.0, 50003.0, 50004.0],
                "volume": [2.0, 4.0, 5.0],
                "side": [-1, 1, -1],
            }
        )

        # Mock GapFiller
        mock_storage = Mock()
        mock_exchange = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage, exchange=mock_exchange)

        # Mock data_gap_filler.fill_missing_trades 返回下载的数据
        gap_filler.data_gap_filler = Mock()
        gap_filler.data_gap_filler.fill_missing_trades.return_value = downloaded_ticks

        # 执行
        result = gap_filler.fill_missing_ticks(
            symbol="BTCUSDT",
            start_time=pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            end_time=pd.Timestamp("2024-01-01 10:00:01", tz="UTC"),
            existing_ticks=existing_ticks,
        )

        # 验证：只返回 2 条新数据（跳过重复的 1 条）
        assert len(result) == 2
        assert result.iloc[0]["price"] == 50003.0
        assert result.iloc[1]["price"] == 50004.0

    def test_no_existing_ticks(self):
        """测试：没有已有数据时，返回所有下载的数据"""
        downloaded_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00.100",
                        "2024-01-01 10:00:00.200",
                    ],
                    utc=True,
                ),
                "price": [50000.0, 50001.0],
                "volume": [1.0, 2.0],
                "side": [1, -1],
            }
        )

        mock_storage = Mock()
        mock_exchange = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage, exchange=mock_exchange)
        gap_filler.data_gap_filler = Mock()
        gap_filler.data_gap_filler.fill_missing_trades.return_value = downloaded_ticks

        # existing_ticks=None
        result = gap_filler.fill_missing_ticks(
            symbol="BTCUSDT",
            start_time=pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            end_time=pd.Timestamp("2024-01-01 10:00:01", tz="UTC"),
            existing_ticks=None,
        )

        # 验证：返回所有数据
        assert len(result) == 2

    def test_empty_download(self):
        """测试：下载为空时返回空 DataFrame"""
        mock_storage = Mock()
        mock_exchange = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage, exchange=mock_exchange)
        gap_filler.data_gap_filler = Mock()
        gap_filler.data_gap_filler.fill_missing_trades.return_value = pd.DataFrame()

        result = gap_filler.fill_missing_ticks(
            symbol="BTCUSDT",
            start_time=pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            end_time=pd.Timestamp("2024-01-01 10:00:01", tz="UTC"),
        )

        assert len(result) == 0


class TestMergeTicks:
    """测试 merge_ticks 方法"""

    def test_merge_with_duplicates(self):
        """测试：合并时去除重复数据"""
        existing = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00.100",
                        "2024-01-01 10:00:00.200",
                    ],
                    utc=True,
                ),
                "price": [50000.0, 50001.0],
                "volume": [1.0, 2.0],
                "side": [1, -1],
            }
        )

        new_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00.200",  # 重复
                        "2024-01-01 10:00:00.300",  # 新的
                    ],
                    utc=True,
                ),
                "price": [50001.0, 50002.0],
                "volume": [2.0, 3.0],
                "side": [-1, 1],
            }
        )

        mock_storage = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage)

        result = gap_filler.merge_ticks(existing, new_ticks)

        # 验证：3 条（去除 1 条重复）
        assert len(result) == 3
        # 验证时间顺序
        assert result.iloc[0]["timestamp"] == pd.Timestamp(
            "2024-01-01 10:00:00.100", tz="UTC"
        )
        assert result.iloc[1]["timestamp"] == pd.Timestamp(
            "2024-01-01 10:00:00.200", tz="UTC"
        )
        assert result.iloc[2]["timestamp"] == pd.Timestamp(
            "2024-01-01 10:00:00.300", tz="UTC"
        )

    def test_merge_empty_existing(self):
        """测试：existing 为空时返回 new_ticks"""
        new_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00"], utc=True),
                "price": [50000.0],
                "volume": [1.0],
                "side": [1],
            }
        )

        mock_storage = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage)

        result = gap_filler.merge_ticks(pd.DataFrame(), new_ticks)
        assert len(result) == 1

    def test_merge_empty_new(self):
        """测试：new_ticks 为空时返回 existing"""
        existing = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00"], utc=True),
                "price": [50000.0],
                "volume": [1.0],
                "side": [1],
            }
        )

        mock_storage = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage)

        result = gap_filler.merge_ticks(existing, pd.DataFrame())
        assert len(result) == 1

    def test_keep_existing_on_duplicate(self):
        """测试：重复时保留 existing 的数据（keep='first'）"""
        existing = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00.100"], utc=True),
                "price": [50000.0],
                "volume": [1.0],
                "side": [1],  # existing 的 side=1
            }
        )

        new_ticks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00.100"], utc=True),
                "price": [50000.0],
                "volume": [1.0],
                "side": [-1],  # new 的 side=-1
            }
        )

        mock_storage = Mock()
        gap_filler = GapFiller(storage_manager=mock_storage)

        result = gap_filler.merge_ticks(existing, new_ticks)

        # 验证：保留 existing 的 side=1
        assert len(result) == 1
        assert result.iloc[0]["side"] == 1


class TestDataGapFillerFillMissingTrades:
    """测试 DataGapFiller.fill_missing_trades 方法"""

    def test_method_exists(self):
        """测试：方法存在且参数正确"""
        import inspect

        assert hasattr(DataGapFiller, "fill_missing_trades")

        sig = inspect.signature(DataGapFiller.fill_missing_trades)
        params = list(sig.parameters.keys())

        assert "symbol" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "max_retries" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
