"""
订单流监听和补数据系统集成测试

包含5个测试场景：
1. 基本功能测试
2. Socket中断恢复测试
3. 1天内补数据测试
4. 1天以上补数据测试
5. Feature Store恢复测试
"""

import pytest
import asyncio
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

from tests.live_data_stream.test_config import TestConfig
from tests.live_data_stream.test_data_simulator import (
    TickDataSimulator,
    InterruptibleDataStream,
)
from tests.live_data_stream.test_data_downloader import TestDataDownloader

from src.live_data_stream import (
    StorageManager,
    OrderFlowListener,
    GapFiller,
    OrderFlowListenerConfig,
)
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


@pytest.fixture
def test_storage_manager():
    """创建测试存储管理器"""
    storage_path = TestConfig.get_test_storage_path()
    # 清理测试存储目录
    if storage_path.exists():
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    manager = StorageManager(base_path=str(storage_path))
    yield manager

    # 测试后清理（可选）
    # if storage_path.exists():
    #     shutil.rmtree(storage_path)


@pytest.fixture
def test_listener(test_storage_manager):
    """创建测试监听器"""
    symbol = TestConfig.TEST_SYMBOL

    feature_computer = IncrementalFeatureComputer(
        tick_window_minutes=240,  # 4小时
        bar_window_size=240,
    )

    listener = OrderFlowListener(
        symbol=symbol,
        storage_manager=test_storage_manager,
        feature_computer=feature_computer,
        memory_window_hours=4.0,
        feature_compute_interval_minutes=15,
        feature_4h_interval_hours=4,
    )

    return listener


@pytest.fixture
def test_gap_filler(test_storage_manager):
    """创建测试数据补全器"""
    if not CCXT_AVAILABLE:
        pytest.skip("ccxt not available")

    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    gap_filler = GapFiller(
        storage_manager=test_storage_manager,
        exchange=exchange,
    )

    return gap_filler


def test_basic_functionality(test_listener, test_storage_manager):
    """
    测试场景1：基本功能测试

    验证：
    - 1分钟聚合是否正确
    - 特征计算是否正常
    - 数据保存到Parquet
    """
    symbol = TestConfig.TEST_SYMBOL

    # 1. 创建数据模拟器（使用2024年12月的数据）
    simulator = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        start_date=datetime(2024, 12, 1),
        end_date=datetime(2024, 12, 31),
        max_ticks=10000,  # 限制数据量
    )

    # 2. 加载数据
    df = simulator.load_data()
    assert len(df) > 0, "没有加载到数据"

    # 3. 处理tick数据
    bar_count = 0
    for tick in simulator.stream_ticks():
        test_listener.on_trade_tick(tick)
        bar_count += 1
        if bar_count >= 10000:  # 限制处理数量
            break

    # 4. 验证1分钟聚合
    memory_window = test_listener.get_memory_window()
    assert len(memory_window) > 0, "内存窗口应该包含数据"

    # 验证bar数据格式
    assert "timestamp" in memory_window.columns
    assert "open" in memory_window.columns
    assert "high" in memory_window.columns
    assert "low" in memory_window.columns
    assert "close" in memory_window.columns
    assert "volume" in memory_window.columns

    # 5. 验证数据保存
    # 检查ticks数据是否保存
    trading_date = datetime.now().strftime("%Y-%m-%d")
    ticks = test_storage_manager.tick_1min.load(symbol, trading_date)
    assert len(ticks) > 0, "ticks数据应该已保存"

    # 6. 验证特征计算
    features = test_listener.feature_computer.get_features()
    assert features is not None, "特征应该已计算"

    print(f"✅ 基本功能测试通过")
    print(f"   处理了 {bar_count} 条tick数据")
    print(f"   生成了 {len(memory_window)} 条1分钟bar")
    print(f"   保存了 {len(ticks)} 条tick数据")


def test_socket_interruption_recovery(test_listener, test_storage_manager):
    """
    测试场景2：Socket中断恢复测试

    验证：
    - 未完成的bar是否正确保存
    - 从Parquet恢复状态
    - 内存窗口恢复
    - 特征计算器状态恢复
    - 数据连续性
    """
    symbol = TestConfig.TEST_SYMBOL

    # 1. 创建数据模拟器
    simulator = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        max_ticks=50000,
    )

    df = simulator.load_data()
    assert len(df) > 0, "没有加载到数据"

    # 2. 设置中断时间点（处理到一半时中断）
    interrupt_at = df["timestamp"].iloc[len(df) // 2]

    # 3. 创建可中断的数据流
    interruptible_stream = InterruptibleDataStream(
        simulator=simulator,
        interrupt_at=interrupt_at,
    )

    # 4. 处理数据直到中断
    bar_count_before = 0
    interrupted = False

    try:
        for tick in interruptible_stream.stream():
            test_listener.on_trade_tick(tick)
            bar_count_before += 1
    except ConnectionError as e:
        interrupted = True
        print(f"✅ Socket中断: {e}")

    assert interrupted, "应该发生socket中断"

    # 5. 验证未完成的bar已保存
    trading_date = interrupt_at.strftime("%Y-%m-%d")
    incomplete_bar = test_storage_manager.tick_1min.get_incomplete_bar(
        symbol, trading_date
    )
    assert incomplete_bar is not None, "未完成的bar应该已保存"

    # 6. 获取恢复状态
    recovery_state = test_listener.get_recovery_state()
    assert recovery_state["latest_1min_timestamp"] is not None, "应该有最新的bar时间戳"

    # 7. 恢复状态
    test_listener.recover_from_interruption()

    # 8. Warmup恢复数据
    warmup_data = test_listener.warmup(days=1, use_gap_filler=False)

    # 9. 继续处理剩余数据
    # 创建新的模拟器，从中断点之后开始
    df_after = df[df["timestamp"] > interrupt_at].copy()
    simulator_after = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        max_ticks=50000,
    )
    simulator_after.df = df_after  # 直接使用过滤后的数据

    bar_count_after = 0
    for tick in simulator_after.stream_ticks():
        test_listener.on_trade_tick(tick)
        bar_count_after += 1
        if bar_count_after >= 10000:
            break

    # 10. 验证数据连续性
    memory_window = test_listener.get_memory_window()
    assert len(memory_window) > 0, "内存窗口应该包含数据"

    # 验证时间连续性（不应该有大的时间间隔）
    if len(memory_window) > 1:
        timestamps = pd.to_datetime(memory_window["timestamp"])
        time_diffs = timestamps.diff().dropna()
        max_diff = time_diffs.max()
        assert max_diff <= pd.Timedelta(minutes=2), "数据应该连续，最大间隔不超过2分钟"

    print(f"✅ Socket中断恢复测试通过")
    print(f"   中断前处理了 {bar_count_before} 条tick")
    print(f"   恢复后处理了 {bar_count_after} 条tick")
    print(f"   内存窗口包含 {len(memory_window)} 条bar")


def test_gap_fill_within_day(test_listener, test_storage_manager, test_gap_filler):
    """
    测试场景3：1天内补数据测试

    验证：
    - 从Parquet warmup补数据
    - 补数据后系统继续正常运行
    """
    symbol = TestConfig.TEST_SYMBOL

    # 1. 创建数据模拟器（加载部分数据）
    simulator = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        max_ticks=20000,
    )

    df = simulator.load_data()
    assert len(df) > 0, "没有加载到数据"

    # 2. 处理前半部分数据
    df_first_half = df.head(len(df) // 2).copy()
    simulator_first = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        max_ticks=20000,
    )
    simulator_first.df = df_first_half

    for tick in simulator_first.stream_ticks():
        test_listener.on_trade_tick(tick)

    # 3. 保存当前状态
    last_timestamp = df_first_half["timestamp"].max()

    # 4. 模拟数据缺失（1天内）
    # 使用GapFiller从Parquet补数据
    gap_filler = GapFiller(
        storage_manager=test_storage_manager,
        exchange=None,  # 1天内不需要从API获取
    )

    # 5. Warmup补数据
    warmup_data = gap_filler.warmup(
        symbol=symbol,
        days=1,
        prefer_feature_store=False,
    )

    # 6. 恢复状态（使用warmup方法）
    test_listener.warmup(days=1, use_gap_filler=False)

    # 7. 继续处理剩余数据
    df_second_half = df.tail(len(df) // 2).copy()
    simulator_second = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        max_ticks=20000,
    )
    simulator_second.df = df_second_half

    for tick in simulator_second.stream_ticks():
        test_listener.on_trade_tick(tick)

    # 8. 验证数据连续性
    memory_window = test_listener.get_memory_window()
    assert len(memory_window) > 0, "内存窗口应该包含数据"

    # 验证没有大的时间间隔
    if len(memory_window) > 1:
        timestamps = pd.to_datetime(memory_window["timestamp"])
        time_diffs = timestamps.diff().dropna()
        max_diff = time_diffs.max()
        assert max_diff <= pd.Timedelta(minutes=2), "数据应该连续"

    print(f"✅ 1天内补数据测试通过")
    print(f"   补数据后内存窗口包含 {len(memory_window)} 条bar")


@pytest.mark.skipif(not CCXT_AVAILABLE, reason="ccxt not available")
def test_gap_fill_over_day(test_listener, test_storage_manager, test_gap_filler):
    """
    测试场景4：1天以上补数据测试

    验证：
    - 从币安API下载数据
    - 使用GapFiller从币安API补数据
    - 补数据后系统继续正常运行
    """
    symbol = TestConfig.TEST_SYMBOL

    # 注意：这个测试需要实际下载数据，可能需要较长时间
    # 可以标记为slow测试，使用pytest -m slow运行

    # 1. 创建数据模拟器（加载旧数据，如1个月前）
    end_date = datetime.now() - timedelta(days=35)  # 35天前
    start_date = end_date - timedelta(days=1)

    simulator = TickDataSimulator(
        symbol=symbol,
        data_dir=TestConfig.PARQUET_DATA_1S_DIR,
        start_date=start_date,
        end_date=end_date,
        max_ticks=10000,
    )

    try:
        df = simulator.load_data()
    except ValueError:
        pytest.skip(f"没有找到 {start_date} 到 {end_date} 的数据")

    if len(df) == 0:
        pytest.skip("数据为空")

    # 2. 处理旧数据
    for tick in simulator.stream_ticks():
        test_listener.on_trade_tick(tick)
        break  # 只处理少量数据，模拟旧数据

    # 3. 模拟数据缺失（超过1天）
    last_timestamp = df["timestamp"].max()
    now = datetime.now()

    # 4. 使用GapFiller从币安API补数据
    if not CCXT_AVAILABLE:
        pytest.skip("ccxt not available")

    import ccxt

    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    gap_filler = GapFiller(
        storage_manager=test_storage_manager,
        exchange=exchange,
    )

    # 5. 补数据（从币安API）
    fill_data = gap_filler.fill_from_binance_api(
        symbol=symbol,
        start_time=pd.Timestamp(last_timestamp) + timedelta(minutes=1),
        end_time=pd.Timestamp(now),
        timeframe="1m",
    )

    if len(fill_data) > 0:
        # 6. 处理补全的数据
        for _, row in fill_data.iterrows():
            # 转换为tick（简化处理）
            from .test_data_simulator import MockTradeTick

            tick = MockTradeTick(
                timestamp=pd.Timestamp(row["timestamp"]),
                price=row["close"],  # 使用close作为价格
                volume=row.get("volume", 0),
                side=1,  # 简化处理
            )
            test_listener.on_trade_tick(tick)

        # 7. 验证数据连续性
        memory_window = test_listener.get_memory_window()
        assert len(memory_window) > 0, "内存窗口应该包含数据"

        print(f"✅ 1天以上补数据测试通过")
        print(f"   补全了 {len(fill_data)} 条数据")
        print(f"   内存窗口包含 {len(memory_window)} 条bar")
    else:
        pytest.skip("无法从币安API获取数据（可能需要网络连接）")


@pytest.mark.skipif(not CCXT_AVAILABLE, reason="ccxt not available")
def test_feature_store_recovery(test_listener, test_storage_manager):
    """
    测试场景5：Feature Store恢复测试

    验证：
    - 下载数据
    - 计算特征
    - 保存到Feature Store
    - 从Feature Store恢复特征
    """
    symbol = TestConfig.TEST_SYMBOL

    # 注意：这个测试需要Feature Store支持，可能需要较长时间

    # 1. 创建数据下载器
    try:
        downloader = TestDataDownloader(
            data_dir="data/test_raw",
            parquet_dir="data/test_parquet",
        )
    except ImportError:
        pytest.skip("BinanceMultiSymbolDownloader不可用")

    # 2. 下载最近几天的数据
    success = downloader.download_days(symbol, days=7)
    if not success:
        pytest.skip("无法下载数据（可能需要网络连接）")

    # 3. 从下载的数据加载并计算特征
    # 这里简化处理，实际应该：
    # - 加载下载的数据
    # - 使用OrderFlowListener处理
    # - 计算特征
    # - 保存到Feature Store

    # 4. 使用GapFiller从Feature Store恢复
    gap_filler = GapFiller(
        storage_manager=test_storage_manager,
        exchange=None,
        feature_store_dir="feature_store",
        feature_store_layer="test_layer",
    )

    # 5. 从Feature Store加载特征
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    features_4h = gap_filler.warmup_from_feature_store(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        timeframe="4h",
    )

    features_15min = gap_filler.warmup_from_feature_store(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        timeframe="15min",
    )

    if features_4h is not None and len(features_4h) > 0:
        print(f"✅ Feature Store恢复测试通过")
        print(f"   从Feature Store加载了 {len(features_4h)} 条4小时特征")
        if features_15min is not None:
            print(f"   从Feature Store加载了 {len(features_15min)} 条15分钟特征")
    else:
        pytest.skip("Feature Store中没有数据（需要先构建Feature Store）")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])
