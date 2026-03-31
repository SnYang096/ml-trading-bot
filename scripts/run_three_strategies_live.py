#!/usr/bin/env python3
"""
三策略实盘启动脚本 (观察模式)

在真实环境中的完整三策略启动流程验证
不执行实际交易，仅验证信号生成和仲裁逻辑

使用方法:
PYTHONPATH=. python scripts/run_three_strategies_live.py --mode demo
PYTHONPATH=. python scripts/run_three_strategies_live.py --mode observe
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.portfolio.live_pcm import LivePCM
from src.live_data_stream import StorageManager, MultiSymbolManager, GapFiller
from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick
from src.live_data_stream.order_manager_factory import init_order_manager_from_env
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


# =============================================================================
# 配置设置
# =============================================================================


def setup_environment(args):
    """设置环境变量"""

    # 基础配置
    os.environ.setdefault("MLBOT_LIVE_SYMBOLS", "BTCUSDT,ETHUSDT")
    os.environ.setdefault("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    os.environ.setdefault("MLBOT_LIVE_STORAGE_BASE", "live/highcap/data")
    os.environ.setdefault("MLBOT_LIVE_USE_FUTURES", "true")
    os.environ.setdefault("MLBOT_LIVE_WARMUP_DAYS", "7")
    os.environ.setdefault("MLBOT_CAPACITY_LIMIT", "2")

    # 确保 tick 数据阈值正确传递
    os.environ.setdefault("MLBOT_MIN_TICKS_REQUIRED", "20160")

    # 交易大小控制
    if args.mode == "demo":
        # 演示模式 - 不交易
        os.environ["MLBOT_LIVE_TRADE_SIZE"] = "0.0"
        trade_desc = "演示模式 (观察)"
    elif args.mode == "observe":
        # 观察模式 - 不交易
        os.environ["MLBOT_LIVE_TRADE_SIZE"] = "0.0"
        trade_desc = "观察模式 (不交易)"
    else:
        # 实盘模式 - 实际交易
        trade_desc = f"实盘模式 (交易大小: {args.trade_size})"
        os.environ["MLBOT_LIVE_TRADE_SIZE"] = str(args.trade_size)

    return trade_desc


# =============================================================================
# 策略初始化
# =============================================================================


def create_three_strategies():
    """创建并初始化三个策略"""

    strategies_root = os.environ["MLBOT_STRATEGIES_ROOT"]
    strategies = {}

    print("=== 初始化三策略 ===")

    # BPC 策略
    print("1. 初始化 BPC 策略...")
    try:
        bpc = GenericLiveStrategy(strategy_name="bpc", strategies_root=strategies_root)
        bpc.load_configs()
        strategies["bpc"] = bpc
        print("   ✅ BPC 策略初始化成功")
    except Exception as e:
        print(f"   ❌ BPC 策略初始化失败: {e}")
        return None

    # ME 策略
    print("2. 初始化 ME 策略...")
    try:
        me = GenericLiveStrategy(
            strategy_name="me-long", strategies_root=strategies_root
        )
        me.load_configs()
        strategies["me-long"] = me
        print("   ✅ ME 策略初始化成功")
    except Exception as e:
        print(f"   ❌ ME 策略初始化失败: {e}")
        return None

    # FER 策略
    print("3. 初始化 FER 策略...")
    try:
        fer = GenericLiveStrategy(strategy_name="fer", strategies_root=strategies_root)
        fer.load_configs()
        strategies["fer"] = fer
        print("   ✅ FER 策略初始化成功")
    except Exception as e:
        print(f"   ❌ FER 策略初始化失败: {e}")
        return None

    return strategies


def setup_pcm(strategies):
    """设置 PCM 仲裁层"""

    print("\n=== 设置 PCM 仲裁层 ===")

    capacity_limit = int(os.environ.get("MLBOT_CAPACITY_LIMIT", "2"))
    pcm = LivePCM(
        archetype_priority=["fer", "me-long", "bpc"],  # FER 最高优先级
        capacity_limit=capacity_limit,
    )

    # 注册所有策略
    for name, strategy in strategies.items():
        pcm.register(name, strategy)
        print(f"✅ 注册策略: {name.upper()}")

    print(f"\n📊 PCM 配置:")
    print(f"   优先级顺序: {' > '.join(pcm.archetype_priority)}")
    print(f"   最大容量上限: {pcm._capacity_limit}")
    print(f"   已注册策略: {', '.join(pcm.registered_archetypes)}")

    return pcm


# =============================================================================
# 数据流设置
# =============================================================================


def setup_data_stream(symbols, storage, pcm):
    """设置数据流和特征计算"""

    print("\n=== 设置数据流 ===")

    bar_minutes = int(os.getenv("MLBOT_BPC_BAR_MINUTES", "240"))
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # 创建 GapFiller
    try:
        import ccxt

        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        gap_filler = GapFiller(
            storage_manager=storage,
            exchange=exchange,
            feature_store_dir="feature_store",
        )
        print("✅ GapFiller 初始化成功")
    except Exception as e:
        print(f"⚠️  GapFiller 初始化失败: {e}")
        gap_filler = None

    # 创建特征计算器
    def _make_feature_computer(symbol: str) -> IncrementalFeatureComputer:
        archetypes_dir = os.path.join(
            os.environ["MLBOT_STRATEGIES_ROOT"], "bpc", "archetypes"
        )
        return IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes,
            bar_window_size=bar_minutes * 2,
            archetypes_dir=archetypes_dir,
            primary_timeframe=f"{bar_minutes}T",
        )

    # 创建多币种管理器
    order_manager = init_order_manager_from_env()

    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=_make_feature_computer,
        gap_filler=gap_filler,
        feature_compute_interval_minutes=window_minutes,
        orderflow_window_minutes=window_minutes,
        order_manager=order_manager,
    )

    # 为每个监听器设置决策处理器
    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = pcm
        listener.order_manager = order_manager
        trade_size = float(os.environ.get("MLBOT_LIVE_TRADE_SIZE", "0.0"))
        if trade_size > 0:
            listener.trade_size = trade_size

    print(f"✅ 数据流设置完成: {len(symbols)} 个币种")
    return manager


# =============================================================================
# 信号监控和日志
# =============================================================================


class SignalLogger:
    """信号日志记录器"""

    def __init__(self):
        self.signal_count = 0
        self.archetype_stats = {"bpc": 0, "me-long": 0, "fer": 0}
        self.start_time = datetime.now()

    def log_signal(self, intent, symbol):
        """记录信号"""
        self.signal_count += 1
        self.archetype_stats[intent.archetype] += 1

        elapsed = datetime.now() - self.start_time
        print(f"\n🔔 信号 #{self.signal_count} (运行时间: {elapsed})")
        print(f"   币种: {symbol}")
        print(f"   策略: {intent.archetype.upper()}")
        print(f"   动作: {intent.action}")
        print(f"   置信度: {intent.confidence:.3f}")
        print(f"   仓位倍数: {intent.size_multiplier}")

        # 统计信息
        total = sum(self.archetype_stats.values())
        if total > 0:
            print(f"\n📊 策略统计:")
            for arch, count in self.archetype_stats.items():
                pct = (count / total) * 100
                print(f"   {arch.upper()}: {count} ({pct:.1f}%)")


# =============================================================================
# 主函数
# =============================================================================


async def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="三策略实盘启动")
    parser.add_argument(
        "--mode", choices=["demo", "observe", "live"], default="demo", help="运行模式"
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=0.1,
        help="实盘交易大小 (仅在 --mode live 时有效)",
    )
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT", help="交易币种 (逗号分隔)"
    )
    args = parser.parse_args()

    # 设置环境
    trade_desc = setup_environment(args)

    print("=" * 80)
    print("🚀 三策略实盘启动")
    print("=" * 80)
    print(f"模式: {trade_desc}")
    print(f"币种: {args.symbols}")
    print(f"时间: {datetime.now()}")
    print()

    # 解析币种
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # 1. 初始化策略
    strategies = create_three_strategies()
    if not strategies:
        print("❌ 策略初始化失败")
        return 1

    # 2. 设置 PCM
    pcm = setup_pcm(strategies)

    # 3. 设置数据流
    # 使用实盘数据目录而不是默认的live_storage
    storage_base = os.environ.get("MLBOT_LIVE_STORAGE_BASE", "live/highcap/data")
    storage = StorageManager(base_path=storage_base)
    manager = setup_data_stream(symbols, storage, pcm)

    # 4. 初始化信号日志
    signal_logger = SignalLogger()

    # 5. 启动数据流
    print(f"\n=== 启动数据流 ===")
    await manager.start_all()

    # 6. 连接 WebSocket
    use_futures = os.getenv("MLBOT_LIVE_USE_FUTURES", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    ws_client = BinanceWebSocketClient(symbols=symbols, use_futures=use_futures)

    def _handle_tick(tick: BinanceTick) -> None:
        """处理 tick 数据"""
        from types import SimpleNamespace
        import pandas as pd

        listener_tick = SimpleNamespace(
            price=float(tick.price),
            size=float(tick.volume),
            side=int(tick.side),
            timestamp=pd.Timestamp(tick.timestamp_ms, unit="ms", tz="UTC"),
            trade_id=tick.trade_id,
        )
        manager.on_trade_tick(tick.symbol, listener_tick)

    ws_client.add_callback(_handle_tick)

    # 7. 运行主循环
    print(f"\n✅ 系统启动完成，开始监听...")
    print(f"按 Ctrl+C 停止")

    stop_event = asyncio.Event()

    try:
        await ws_client.run(stop_event)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断")
        stop_event.set()
    finally:
        await manager.stop_all()
        print(f"\n📊 运行统计:")
        print(f"   总信号数: {signal_logger.signal_count}")
        print(f"   运行时间: {datetime.now() - signal_logger.start_time}")
        if signal_logger.signal_count > 0:
            print(f"   策略分布:")
            total = sum(signal_logger.archetype_stats.values())
            for arch, count in signal_logger.archetype_stats.items():
                pct = (count / total) * 100
                print(f"     {arch.upper()}: {count} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
