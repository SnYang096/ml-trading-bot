#!/usr/bin/env python3
"""
补数据逻辑场景测试

模拟各种数据缺失场景，测试补数据功能。
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.live_data_stream.data_gap_filler import DataGapFiller


def test_detect_missing_scenarios():
    """测试检测缺失数据的各种场景"""
    print("=" * 60)
    print("测试数据缺失检测场景")
    print("=" * 60)

    # 创建模拟exchange
    mock_exchange = Mock()

    with patch("src.live_data_stream.data_gap_filler.CCXT_AVAILABLE", True):
        gap_filler = DataGapFiller(mock_exchange)

        # 场景1: 无缺失数据
        print("\n场景1: 无缺失数据")
        df1 = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00"),
                    pd.Timestamp("2024-01-01 10:15:00"),
                    pd.Timestamp("2024-01-01 10:30:00"),
                ],
                "open": [50000, 50100, 50200],
                "high": [50100, 50200, 50300],
                "low": [49900, 50000, 50100],
                "close": [50050, 50150, 50250],
                "volume": [100, 110, 120],
            }
        )
        missing1 = gap_filler.detect_missing_bars(df1, timeframe="15T")
        assert len(missing1) == 0
        print("   ✅ 无缺失数据检测正确")

        # 场景2: 单个缺失
        print("\n场景2: 单个缺失")
        df2 = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00"),
                    pd.Timestamp("2024-01-01 10:15:00"),
                    # 缺失 10:30
                    pd.Timestamp("2024-01-01 10:45:00"),
                ],
                "open": [50000, 50100, 50200],
                "high": [50100, 50200, 50300],
                "low": [49900, 50000, 50100],
                "close": [50050, 50150, 50250],
                "volume": [100, 110, 120],
            }
        )
        missing2 = gap_filler.detect_missing_bars(df2, timeframe="15T")
        assert len(missing2) == 1
        assert pd.Timestamp("2024-01-01 10:30:00") in missing2
        print(f"   ✅ 检测到 {len(missing2)} 个缺失")

        # 场景3: 连续多个缺失
        print("\n场景3: 连续多个缺失")
        df3 = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00"),
                    # 缺失 10:15, 10:30, 10:45
                    pd.Timestamp("2024-01-01 11:00:00"),
                ],
                "open": [50000, 50200],
                "high": [50100, 50300],
                "low": [49900, 50100],
                "close": [50050, 50250],
                "volume": [100, 120],
            }
        )
        missing3 = gap_filler.detect_missing_bars(df3, timeframe="15T")
        assert len(missing3) == 3
        print(f"   ✅ 检测到 {len(missing3)} 个连续缺失")

        # 场景4: 大间隔缺失（超过1小时）
        print("\n场景4: 大间隔缺失")
        df4 = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00"),
                    # 缺失超过1小时的数据
                    pd.Timestamp("2024-01-01 12:00:00"),
                ],
                "open": [50000, 51000],
                "high": [50100, 51100],
                "low": [49900, 50900],
                "close": [50050, 51050],
                "volume": [100, 200],
            }
        )
        missing4 = gap_filler.detect_missing_bars(df4, timeframe="15T")
        assert len(missing4) >= 7  # 至少7个15分钟间隔
        print(f"   ✅ 检测到 {len(missing4)} 个大间隔缺失")

        # 场景5: 边界情况 - 空DataFrame
        print("\n场景5: 空DataFrame")
        df5 = pd.DataFrame()
        missing5 = gap_filler.detect_missing_bars(df5, timeframe="15T")
        assert len(missing5) == 0
        print("   ✅ 空DataFrame处理正确")

        # 场景6: 边界情况 - 单行数据
        print("\n场景6: 单行数据")
        df6 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01 10:00:00")],
                "open": [50000],
                "high": [50100],
                "low": [49900],
                "close": [50050],
                "volume": [100],
            }
        )
        missing6 = gap_filler.detect_missing_bars(df6, timeframe="15T")
        assert len(missing6) == 0
        print("   ✅ 单行数据处理正确")

        print("\n" + "=" * 60)
        print("✅ 所有缺失检测场景测试通过")
        print("=" * 60)


def test_validate_data_scenarios():
    """测试数据验证场景"""
    print("\n" + "=" * 60)
    print("测试数据验证场景")
    print("=" * 60)

    mock_exchange = Mock()

    with patch("src.live_data_stream.data_gap_filler.CCXT_AVAILABLE", True):
        gap_filler = DataGapFiller(mock_exchange)

        # 场景1: 有效数据
        print("\n场景1: 有效数据验证")
        valid_bar = pd.Series(
            {
                "open": 50000,
                "high": 50100,
                "low": 49900,
                "close": 50050,
                "volume": 100,
            }
        )
        assert gap_filler._validate_bar(valid_bar) is True
        print("   ✅ 有效数据通过验证")

        # 场景2: 价格范围错误
        print("\n场景2: 价格范围错误")
        invalid_bar1 = pd.Series(
            {
                "open": 50000,
                "high": 49900,  # high < low
                "low": 50000,
                "close": 50050,
                "volume": 100,
            }
        )
        assert gap_filler._validate_bar(invalid_bar1) is False
        print("   ✅ 价格范围错误被正确拒绝")

        # 场景3: 缺少字段
        print("\n场景3: 缺少字段")
        invalid_bar2 = pd.Series(
            {
                "open": 50000,
                "high": 50100,
                # 缺少 low
                "close": 50050,
                "volume": 100,
            }
        )
        assert gap_filler._validate_bar(invalid_bar2) is False
        print("   ✅ 缺少字段被正确拒绝")

        # 场景4: 负成交量
        print("\n场景4: 负成交量")
        invalid_bar3 = pd.Series(
            {
                "open": 50000,
                "high": 50100,
                "low": 49900,
                "close": 50050,
                "volume": -100,  # 负成交量
            }
        )
        assert gap_filler._validate_bar(invalid_bar3) is False
        print("   ✅ 负成交量被正确拒绝")

        print("\n" + "=" * 60)
        print("✅ 所有数据验证场景测试通过")
        print("=" * 60)


def test_download_retry_scenarios():
    """测试下载重试场景"""
    print("\n" + "=" * 60)
    print("测试下载重试场景")
    print("=" * 60)

    mock_exchange = Mock()

    with patch("src.live_data_stream.data_gap_filler.CCXT_AVAILABLE", True):
        gap_filler = DataGapFiller(mock_exchange)

        # 场景1: API失败后重试
        print("\n场景1: API失败重试")
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            [],  # 第三次返回空数据
        ]

        missing_timestamps = [pd.Timestamp("2024-01-01 10:15:00")]
        result = gap_filler.download_missing_bars(
            symbol="BTC/USDT:USDT",
            missing_timestamps=missing_timestamps,
            timeframe="15T",
            max_retries=3,
        )

        assert len(result) == 0
        assert mock_exchange.fetch_ohlcv.call_count == 3
        print(f"   ✅ 重试了 {mock_exchange.fetch_ohlcv.call_count} 次")

        # 重置mock
        mock_exchange.reset_mock()

        # 场景2: 第一次失败，第二次成功
        print("\n场景2: 失败后成功")
        ts_10_15 = int(pd.Timestamp("2024-01-01 10:15:00").timestamp() * 1000)
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("Network error"),
            [[ts_10_15, 50100, 50200, 50000, 50150, 110]],
        ]

        result = gap_filler.download_missing_bars(
            symbol="BTC/USDT:USDT",
            missing_timestamps=[pd.Timestamp("2024-01-01 10:15:00")],
            timeframe="15T",
            max_retries=3,
        )

        assert len(result) > 0
        assert mock_exchange.fetch_ohlcv.call_count == 2
        print(f"   ✅ 第2次尝试成功，下载了 {len(result)} 条数据")

        print("\n" + "=" * 60)
        print("✅ 所有下载重试场景测试通过")
        print("=" * 60)


def test_integration_scenario():
    """集成测试场景：完整的补数据流程"""
    print("\n" + "=" * 60)
    print("集成测试：完整补数据流程")
    print("=" * 60)

    mock_exchange = Mock()

    with patch("src.live_data_stream.data_gap_filler.CCXT_AVAILABLE", True):
        gap_filler = DataGapFiller(mock_exchange)

        # 模拟已有数据（有缺失）
        print("\n1. 检测缺失数据")
        df_existing = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00"),
                    pd.Timestamp("2024-01-01 10:15:00"),
                    # 缺失 10:30, 10:45
                    pd.Timestamp("2024-01-01 11:00:00"),
                ],
                "open": [50000, 50100, 50200],
                "high": [50100, 50200, 50300],
                "low": [49900, 50000, 50100],
                "close": [50050, 50150, 50250],
                "volume": [100, 110, 120],
            }
        )

        missing = gap_filler.detect_missing_bars(df_existing, timeframe="15T")
        print(f"   检测到 {len(missing)} 个缺失时间戳")
        assert len(missing) == 2

        # 模拟下载缺失数据
        print("\n2. 下载缺失数据")
        ts_10_30 = int(pd.Timestamp("2024-01-01 10:30:00").timestamp() * 1000)
        ts_10_45 = int(pd.Timestamp("2024-01-01 10:45:00").timestamp() * 1000)

        mock_exchange.fetch_ohlcv.return_value = [
            [ts_10_30, 50200, 50300, 50100, 50250, 120],
            [ts_10_45, 50300, 50400, 50200, 50350, 130],
        ]

        downloaded = gap_filler.download_missing_bars(
            symbol="BTC/USDT:USDT",
            missing_timestamps=missing,
            timeframe="15T",
        )
        print(f"   下载了 {len(downloaded)} 条数据")
        assert len(downloaded) > 0

        # 验证下载的数据
        print("\n3. 验证下载的数据")
        validated = gap_filler.validate_downloaded_data(
            downloaded, missing, timeframe="15T"
        )
        print(f"   验证通过 {len(validated)} 条数据")
        assert len(validated) > 0

        # 合并数据
        print("\n4. 合并数据")
        df_complete = (
            pd.concat([df_existing, validated])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        print(f"   合并后共有 {len(df_complete)} 条数据")

        # 再次检测，应该没有缺失了
        missing_after = gap_filler.detect_missing_bars(df_complete, timeframe="15T")
        print(f"   补全后缺失: {len(missing_after)} 条")
        assert len(missing_after) == 0

        print("\n" + "=" * 60)
        print("✅ 集成测试通过")
        print("=" * 60)


if __name__ == "__main__":
    try:
        test_detect_missing_scenarios()
        test_validate_data_scenarios()
        test_download_retry_scenarios()
        test_integration_scenario()

        print("\n" + "=" * 60)
        print("✅ 所有补数据逻辑测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
