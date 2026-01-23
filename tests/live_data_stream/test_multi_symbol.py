"""
多Symbol测试

测试多个symbol同时处理时的正确性，包括：
1. 多symbol基本功能测试
2. 多symbol并发处理测试
3. 多symbol socket中断恢复测试
4. 多symbol补数据测试
"""

import pytest
import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

from tests.live_data_stream.test_config import TestConfig
from tests.live_data_stream.test_data_simulator import (
    TickDataSimulator,
    InterruptibleDataStream,
)

from src.live_data_stream import (
    StorageManager,
    OrderFlowListener,
    GapFiller,
    MultiSymbolManager,
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


@pytest.fixture
def test_symbols():
    """获取可用的测试symbol列表"""
    available = TestConfig.get_available_test_symbols()
    # 快速测试：只使用2个symbol
    max_symbols = 2
    # 至少使用2个symbol进行测试
    if len(available) < 2:
        # 如果可用symbol少于2个，使用默认symbol重复（仅用于测试）
        return [TestConfig.TEST_SYMBOL, TestConfig.TEST_SYMBOL]
    return available[:max_symbols]  # 快速模式最多使用2个symbol


@pytest.fixture
def test_multi_symbol_manager(test_storage_manager, test_symbols):
    """创建多symbol管理器"""
    # 测试优化：使用更小的内存窗口和更长的特征计算间隔（减少计算量）
    memory_window_hours = 1.0  # 从4小时降到1小时
    feature_compute_interval = 60  # 测试时60分钟计算一次（实际不会触发，减少计算）
    feature_4h_interval = 4

    manager = MultiSymbolManager(
        symbols=test_symbols,
        storage_manager=test_storage_manager,
        memory_window_hours=memory_window_hours,
        feature_compute_interval_minutes=feature_compute_interval,
        feature_4h_interval_hours=feature_4h_interval,
    )
    return manager


def test_multi_symbol_basic_functionality(
    test_multi_symbol_manager, test_storage_manager, test_symbols
):
    """
    测试场景1：多symbol基本功能测试

    验证：
    - 每个symbol的数据正确聚合
    - 数据隔离（不同symbol的数据不会混淆）
    - 特征计算正确性
    """
    # 1. 为每个symbol创建数据模拟器
    simulators = {}
    for symbol in test_symbols:
        simulator = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            start_date=datetime(2024, 12, 1),
            end_date=datetime(2024, 12, 31),
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        simulators[symbol] = simulator

    # 2. 加载所有symbol的数据
    for symbol, simulator in simulators.items():
        df = simulator.load_data()
        assert len(df) > 0, f"没有加载到 {symbol} 的数据"

    # 3. 处理所有symbol的tick数据
    tick_counts = {}
    for symbol, simulator in simulators.items():
        count = 0
        for tick in simulator.stream_ticks():
            test_multi_symbol_manager.on_trade_tick(symbol, tick)
            count += 1
            # 使用配置值（CI/CD模式会自动使用100 ticks）
            max_ticks = TestConfig.get_max_ticks_per_symbol()
            if count >= max_ticks:
                break
        tick_counts[symbol] = count

    # 4. 验证每个symbol的数据隔离和正确性
    for symbol in test_symbols:
        listener = test_multi_symbol_manager.get_listener(symbol)
        assert listener is not None, f"{symbol} 的listener应该存在"

        # 验证内存窗口
        memory_window = listener.get_memory_window()
        # CI模式下数据量少，可能没有完整的bar，至少验证处理了数据
        if TestConfig.IS_CI:
            # CI模式下，至少验证处理了tick数据
            assert tick_counts.get(symbol, 0) > 0, f"{symbol} 应该至少处理1条tick"
            # CI模式下不强制要求有完整的bar（因为100条tick可能不足以生成1分钟bar）
        else:
            assert len(memory_window) > 0, f"{symbol} 的内存窗口应该包含数据"

            # 验证数据格式（仅在非CI模式下）
            assert "timestamp" in memory_window.columns
            assert "open" in memory_window.columns
            assert "high" in memory_window.columns
            assert "low" in memory_window.columns
            assert "close" in memory_window.columns
            assert "volume" in memory_window.columns

            # 验证数据保存（仅在非CI模式下）
            trading_date = datetime.now().strftime("%Y-%m-%d")
            ticks = test_storage_manager.tick_1min.load(symbol, trading_date)
            assert len(ticks) > 0, f"{symbol} 的ticks数据应该已保存"

        # 验证特征计算（所有模式）
        features = listener.feature_computer.get_features()
        assert features is not None, f"{symbol} 的特征应该已计算"

    # 5. 验证数据隔离（不同symbol的数据不会混淆）
    # 检查不同symbol的数据是否保存在不同路径
    for symbol in test_symbols:
        trading_date = datetime.now().strftime("%Y-%m-%d")
        ticks = test_storage_manager.tick_1min.load(symbol, trading_date)

        # 验证数据中的symbol字段（如果有）
        if "symbol" in ticks.columns:
            assert all(
                ticks["symbol"] == symbol
            ), f"{symbol} 的数据不应该包含其他symbol"

    print(f"✅ 多symbol基本功能测试通过")
    print(f"   测试了 {len(test_symbols)} 个symbol: {test_symbols}")
    for symbol, count in tick_counts.items():
        memory_window = test_multi_symbol_manager.get_listener(
            symbol
        ).get_memory_window()
        print(f"   {symbol}: 处理了 {count} 条tick，生成了 {len(memory_window)} 条bar")


async def process_symbol_ticks(
    symbol: str,
    simulator: TickDataSimulator,
    manager: MultiSymbolManager,
    max_ticks: int,
):
    """异步处理单个symbol的tick数据"""
    count = 0
    for tick in simulator.stream_ticks():
        manager.on_trade_tick(symbol, tick)
        count += 1
        if count >= max_ticks:
            break
        # 添加小延迟，模拟实时数据流
        await asyncio.sleep(0.001)
    return count


def test_multi_symbol_concurrent_processing(test_multi_symbol_manager, test_symbols):
    """
    测试场景2：多symbol并发处理测试

    验证：
    - 多个symbol同时接收tick数据不会出错
    - 并发处理正确性
    - 数据隔离
    """
    # 1. 为每个symbol创建数据模拟器
    simulators = {}
    for symbol in test_symbols:
        simulator = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            start_date=datetime(2024, 12, 1),
            end_date=datetime(2024, 12, 31),
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        simulators[symbol] = simulator

    # 2. 并发处理所有symbol的tick数据
    async def run_concurrent():
        tasks = []
        for symbol, simulator in simulators.items():
            # 使用配置值（CI/CD模式会自动使用100 ticks）
            max_ticks = TestConfig.get_max_ticks_per_symbol()
            task = process_symbol_ticks(
                symbol,
                simulator,
                test_multi_symbol_manager,
                max_ticks,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return dict(zip(test_symbols, results))

    # 运行并发处理
    tick_counts = asyncio.run(run_concurrent())

    # 3. 验证每个symbol的数据正确性
    for symbol in test_symbols:
        listener = test_multi_symbol_manager.get_listener(symbol)
        memory_window = listener.get_memory_window()

        assert len(memory_window) > 0, f"{symbol} 的内存窗口应该包含数据"

        # 验证数据连续性（不应该有大的时间间隔）
        if len(memory_window) > 1:
            timestamps = pd.to_datetime(memory_window["timestamp"])
            time_diffs = timestamps.diff().dropna()
            max_diff = time_diffs.max()
            assert max_diff <= pd.Timedelta(minutes=2), f"{symbol} 的数据应该连续"

    print(f"✅ 多symbol并发处理测试通过")
    print(f"   并发处理了 {len(test_symbols)} 个symbol")
    for symbol, count in tick_counts.items():
        memory_window = test_multi_symbol_manager.get_listener(
            symbol
        ).get_memory_window()
        print(f"   {symbol}: 处理了 {count} 条tick，生成了 {len(memory_window)} 条bar")


def test_multi_symbol_interruption_recovery(
    test_multi_symbol_manager, test_storage_manager, test_symbols
):
    """
    测试场景3：多symbol socket中断恢复测试

    验证：
    - 每个symbol独立恢复
    - 恢复后数据连续性
    """
    # 1. 为每个symbol创建数据模拟器
    simulators = {}
    interrupt_times = {}

    for symbol in test_symbols:
        simulator = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            start_date=datetime(2024, 12, 1),
            end_date=datetime(2024, 12, 31),
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        df = simulator.load_data()
        # 设置中断时间点（处理到一半时中断）
        interrupt_at_raw = df["timestamp"].iloc[len(df) // 2]
        # 转换为Timestamp并确保时区感知
        if isinstance(interrupt_at_raw, pd.Timestamp):
            interrupt_at = interrupt_at_raw
        else:
            interrupt_at = pd.Timestamp(interrupt_at_raw)
        if interrupt_at.tz is None:
            interrupt_at = interrupt_at.tz_localize("UTC")
        interrupt_times[symbol] = interrupt_at
        simulators[symbol] = simulator

    # 2. 处理数据直到中断
    interrupted_symbols = set()

    for symbol, simulator in simulators.items():
        interrupt_at = interrupt_times[symbol]
        interruptible_stream = InterruptibleDataStream(
            simulator=simulator,
            interrupt_at=interrupt_at,
        )

        try:
            for tick in interruptible_stream.stream():
                test_multi_symbol_manager.on_trade_tick(symbol, tick)
        except ConnectionError:
            interrupted_symbols.add(symbol)
            print(f"✅ {symbol} Socket中断")

    assert len(interrupted_symbols) > 0, "应该有symbol发生中断"

    # 3. 验证未完成的bar已保存（通过检查恢复状态）
    for symbol in interrupted_symbols:
        listener = test_multi_symbol_manager.get_listener(symbol)
        recovery_state = listener.get_recovery_state()
        # 检查是否有未完成的bar（通过检查current_1min_bar）
        has_incomplete = (
            listener.current_1min_bar is not None
            or recovery_state.get("incomplete_bar") is not None
        )
        assert has_incomplete, f"{symbol} 应该有未完成的bar"

    # 4. 恢复所有symbol
    recovery_states = test_multi_symbol_manager.recover_all_from_interruption()

    # 5. Warmup所有symbol
    warmup_results = asyncio.run(
        test_multi_symbol_manager.warmup_all(days=1, use_gap_filler=False)
    )

    # 6. 继续处理剩余数据
    for symbol, simulator in simulators.items():
        if symbol in interrupted_symbols:
            interrupt_at = interrupt_times[symbol]
            # 创建新的模拟器，从中断点之后开始
            df = simulator.load_data()
            # 确保时间戳比较正确（时区感知）
            df_timestamps = pd.to_datetime(df["timestamp"])
            if df_timestamps.dt.tz is None:
                df_timestamps = df_timestamps.dt.tz_localize("UTC")
            # 确保interrupt_at也是时区感知的
            if interrupt_at.tz is None:
                interrupt_at = interrupt_at.tz_localize("UTC")
            df_after = df[df_timestamps > interrupt_at].copy()

            if len(df_after) > 0:
                simulator_after = TickDataSimulator(
                    symbol=symbol,
                    data_dir=TestConfig.PARQUET_DATA_1S_DIR,
                    max_ticks=TestConfig.get_max_ticks_per_symbol(),
                )
                simulator_after.df = df_after

                # 继续处理
                for tick in simulator_after.stream_ticks():
                    test_multi_symbol_manager.on_trade_tick(symbol, tick)
                    break  # 只处理少量数据验证恢复

    # 7. 验证数据连续性
    for symbol in interrupted_symbols:
        listener = test_multi_symbol_manager.get_listener(symbol)
        memory_window = listener.get_memory_window()

        assert len(memory_window) > 0, f"{symbol} 的内存窗口应该包含数据"

        # 验证时间连续性
        if len(memory_window) > 1:
            timestamps = pd.to_datetime(memory_window["timestamp"])
            time_diffs = timestamps.diff().dropna()
            max_diff = time_diffs.max()
            assert max_diff <= pd.Timedelta(minutes=2), f"{symbol} 的数据应该连续"

    print(f"✅ 多symbol socket中断恢复测试通过")
    print(f"   {len(interrupted_symbols)} 个symbol发生中断并成功恢复")


def test_multi_symbol_gap_fill(
    test_multi_symbol_manager, test_storage_manager, test_symbols
):
    """
    测试场景4：多symbol补数据测试

    验证：
    - 多个symbol同时需要补数据时逻辑正确
    - 不同symbol的补数据互不影响
    """
    # 1. 为每个symbol创建数据模拟器（加载部分数据）
    simulators_first = {}
    simulators_second = {}

    for symbol in test_symbols:
        simulator = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            start_date=datetime(2024, 12, 1),
            end_date=datetime(2024, 12, 31),
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        df = simulator.load_data()

        # 分割数据
        df_first = df.head(len(df) // 2).copy()
        df_second = df.tail(len(df) // 2).copy()

        simulator_first = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        simulator_first.df = df_first
        simulators_first[symbol] = simulator_first

        simulator_second = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            max_ticks=TestConfig.get_max_ticks_per_symbol(),
        )
        simulator_second.df = df_second
        simulators_second[symbol] = simulator_second

    # 2. 处理前半部分数据
    for symbol, simulator in simulators_first.items():
        for tick in simulator.stream_ticks():
            test_multi_symbol_manager.on_trade_tick(symbol, tick)

    # 3. Warmup补数据（所有symbol）
    warmup_results = asyncio.run(
        test_multi_symbol_manager.warmup_all(days=1, use_gap_filler=False)
    )

    # 4. 继续处理剩余数据
    for symbol, simulator in simulators_second.items():
        for tick in simulator.stream_ticks():
            test_multi_symbol_manager.on_trade_tick(symbol, tick)
            break  # 只处理少量数据验证

    # 5. 验证每个symbol的数据连续性
    for symbol in test_symbols:
        listener = test_multi_symbol_manager.get_listener(symbol)
        memory_window = listener.get_memory_window()

        assert len(memory_window) > 0, f"{symbol} 的内存窗口应该包含数据"

        # 验证数据连续性
        if len(memory_window) > 1:
            timestamps = pd.to_datetime(memory_window["timestamp"])
            time_diffs = timestamps.diff().dropna()
            max_diff = time_diffs.max()
            assert max_diff <= pd.Timedelta(minutes=2), f"{symbol} 的数据应该连续"

    # 6. 验证不同symbol的数据隔离
    # 检查不同symbol的数据是否保存在不同路径
    for symbol in test_symbols:
        trading_date = datetime.now().strftime("%Y-%m-%d")
        ticks = test_storage_manager.tick_1min.load(symbol, trading_date)
        assert len(ticks) > 0, f"{symbol} 的数据应该已保存"

    print(f"✅ 多symbol补数据测试通过")
    print(f"   测试了 {len(test_symbols)} 个symbol的补数据功能")


def test_multi_symbol_status_summary(test_multi_symbol_manager, test_symbols):
    """
    测试多symbol状态摘要功能
    """
    # 处理一些数据
    for symbol in test_symbols:
        simulator = TickDataSimulator(
            symbol=symbol,
            data_dir=TestConfig.PARQUET_DATA_1S_DIR,
            start_date=datetime(2024, 12, 1),
            end_date=datetime(2024, 12, 31),
            max_ticks=1000,
        )
        for tick in simulator.stream_ticks():
            test_multi_symbol_manager.on_trade_tick(symbol, tick)
            break  # 只处理少量数据

    # 获取状态摘要
    summary = test_multi_symbol_manager.get_status_summary()

    assert "symbols" in summary
    assert "listeners" in summary
    assert len(summary["symbols"]) == len(test_symbols)
    assert len(summary["listeners"]) == len(test_symbols)

    for symbol in test_symbols:
        assert symbol in summary["listeners"]
        listener_status = summary["listeners"][symbol]
        assert "is_running" in listener_status
        assert "memory_window_size" in listener_status

    print(f"✅ 多symbol状态摘要测试通过")
    print(f"   状态摘要: {summary}")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])
