#!/usr/bin/env python3
"""
币安实盘测试运行脚本

使用空策略测试多symbol数据接收和订单流特征计算。

功能：
- 从 config/local/binance_mainnet.env 加载API key
- 创建Nautilus Trader节点
- 运行空策略（不执行交易）
- 计算并输出订单流特征（vpin, cvd, tradecluster, volprofile, vwap）

使用方法：
    python scripts/run_live_test.py --symbols BTCUSDT ETHUSDT SOLUSDT --duration 10

环境变量：
    BINANCE_API_KEY: 币安API key（主网）
    BINANCE_API_SECRET: 币安API secret（主网）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from nautilus_trader.config import TradingNodeConfig
    from nautilus_trader.config import CacheConfig
    from nautilus_trader.config import LiveDataEngineConfig
    from nautilus_trader.config import LiveExecEngineConfig
    from nautilus_trader.config import LiveRiskEngineConfig
    from nautilus_trader.config import PortfolioConfig
    from nautilus_trader.adapters.binance import BINANCE
    from nautilus_trader.adapters.binance import BinanceDataClientConfig
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.common.config import InstrumentProviderConfig
    from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model import InstrumentId

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    print("❌ Nautilus Trader is not installed.")
    print("Install it with: pip install nautilus-trader")
    sys.exit(1)

from src.live_data_stream.live_test_strategy import LiveTestStrategy
from src.live_data_stream.multi_symbol_manager import MultiSymbolManager
from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


def load_api_keys(env_file: Path) -> tuple[str, str]:
    """
    从环境变量文件加载API key

    Args:
        env_file: 环境变量文件路径

    Returns:
        (api_key, api_secret)
    """
    api_key = ""
    api_secret = ""

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        # 支持多种key格式
                        # 主网: BINANCE_API_KEY 或 BINANCE_FUTURES_API_KEY
                        # 测试网: BINANCE_FUTURES_TESTNET_API_KEY
                        if "API_KEY" in key.upper():
                            api_key = value
                        elif "API_SECRET" in key.upper() or "SECRET" in key.upper():
                            api_secret = value

        # 同时设置到环境变量（供Nautilus Trader使用）
        # 主网使用 BINANCE_API_KEY，测试网使用 BINANCE_FUTURES_TESTNET_API_KEY
        if api_key:
            os.environ["BINANCE_API_KEY"] = api_key
            # 如果key名称包含TESTNET，也设置测试网环境变量
            if "TESTNET" in str(env_file) or "TESTNET" in api_key:
                os.environ["BINANCE_FUTURES_TESTNET_API_KEY"] = api_key
        if api_secret:
            os.environ["BINANCE_API_SECRET"] = api_secret
            if "TESTNET" in str(env_file) or "TESTNET" in api_secret:
                os.environ["BINANCE_FUTURES_TESTNET_API_SECRET"] = api_secret

    return api_key, api_secret


def create_live_test_node(
    symbols: List[str],
    api_key: str,
    api_secret: str,
    storage_path: str = "data/live_storage",
    testnet: bool = False,
) -> TradingNode:
    """
    创建实盘测试节点

    Args:
        symbols: 交易对符号列表
        api_key: 币安API key
        api_secret: 币安API secret
        storage_path: 存储路径
        testnet: 是否使用测试网

    Returns:
        TradingNode实例
    """
    # 1. 创建存储管理器
    storage_manager = StorageManager(base_path=storage_path)

    # 2. 创建数据补全器（如果需要）
    gap_filler = None
    if CCXT_AVAILABLE:
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        gap_filler = GapFiller(
            storage_manager=storage_manager,
            exchange=exchange,
        )

    # 3. 创建 MultiSymbolManager
    multi_manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage_manager,
        gap_filler=gap_filler,
        memory_window_hours=4.0,
        feature_compute_interval_minutes=15,
        feature_4h_interval_hours=4,
    )

    # 4. 创建 InstrumentId 列表
    instrument_ids = []
    for symbol in symbols:
        if "USDT" in symbol:
            instrument_str = f"{symbol}-PERP.BINANCE"
        else:
            instrument_str = f"{symbol}.BINANCE"
        instrument_id = InstrumentId.from_str(instrument_str)
        instrument_ids.append(instrument_id)

    # 5. 创建策略
    strategy = LiveTestStrategy(
        instrument_ids=instrument_ids,
        multi_symbol_manager=multi_manager,
        feature_output_interval_minutes=15,  # 每15分钟输出一次特征
    )

    # 6. 配置 Binance 数据客户端
    # 根据Nautilus Trader版本，使用正确的账户类型
    if testnet:
        # 测试网可能没有专门的枚举，使用主网但配置为测试网
        account_type = BinanceAccountType.USDT_FUTURES  # 测试网使用相同的枚举
    else:
        account_type = BinanceAccountType.USDT_FUTURES

    # 创建instrument_id列表用于加载
    instrument_ids_to_load = []
    for symbol in symbols:
        if "USDT" in symbol:
            instrument_str = f"{symbol}-PERP.BINANCE"
        else:
            instrument_str = f"{symbol}.BINANCE"
        instrument_ids_to_load.append(InstrumentId.from_str(instrument_str))

    # 配置instrument provider
    # 使用load_ids而不是load_all，因为load_all需要账户信息权限
    # load_ids只需要查询交易所的instrument列表，不需要账户权限
    instrument_provider_config = InstrumentProviderConfig(
        load_all=False,  # 不使用load_all（需要账户权限）
        load_ids=frozenset(
            instrument_ids_to_load
        ),  # 只加载需要的instruments（不需要账户权限）
    )

    # 配置Binance数据客户端
    # 根据Nautilus Trader文档：
    # - account_type: USDT_FUTURES for futures
    # - testnet: True for testnet, False for mainnet
    # - use_agg_trade_ticks: False (default) uses raw trades, True uses aggregated trades
    # - instrument_provider: 配置instrument加载
    binance_config = BinanceDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=account_type,
        testnet=testnet,  # 设置testnet参数以使用测试网URL
        use_agg_trade_ticks=False,  # 使用原始交易数据（非聚合）
        instrument_provider=instrument_provider_config,  # 传递instrument provider配置
    )

    # 验证配置
    print(
        f"📋 Instrument Provider配置: load_all={instrument_provider_config.load_all}, load_ids={len(instrument_ids_to_load)}个instruments"
    )

    # 7. 创建交易节点配置（在构造时传入data_clients）
    config = TradingNodeConfig(
        trader_id="LIVE-TEST-TRADER",
        cache=CacheConfig(tick_capacity=10000, bar_capacity=1000),
        data_engine=LiveDataEngineConfig(),
        exec_engine=LiveExecEngineConfig(),
        risk_engine=LiveRiskEngineConfig(),
        portfolio=PortfolioConfig(),
        data_clients={BINANCE: binance_config},
    )

    # 8. 创建交易节点
    node = TradingNode(config=config)

    # 9. 添加数据客户端工厂
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)

    # 10. 添加策略
    node.trader.add_strategy(strategy)

    # 11. 构建节点（连接客户端）
    node.build()

    return node


async def run_live_test(
    symbols: List[str],
    duration_minutes: int = 10,
    testnet: bool = False,
    storage_path: str = "data/live_storage",
) -> None:
    """
    运行实盘测试

    Args:
        symbols: 交易对符号列表
        duration_minutes: 运行时长（分钟）
        testnet: 是否使用测试网
        storage_path: 存储路径
    """
    # 加载API key
    if testnet:
        env_file = project_root / "config" / "local" / "binance_testnet.env"
        # 测试网也可以从环境变量读取
        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
    else:
        env_file = project_root / "config" / "local" / "binance_mainnet.env"
        # 主网也可以从环境变量读取（支持多种格式）
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv(
            "BINANCE_FUTURES_API_KEY", ""
        )
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
            "BINANCE_FUTURES_API_SECRET", ""
        )

    # 如果环境变量没有，从文件加载
    if not api_key or not api_secret:
        api_key, api_secret = load_api_keys(env_file)

        # 如果文件中的key是testnet格式但我们在用主网，给出警告
        if not testnet and env_file.exists():
            file_content = env_file.read_text()
            if "TESTNET" in file_content and "BINANCE_FUTURES_TESTNET" in file_content:
                print("⚠️  警告: 主网配置文件中包含测试网格式的API key")
                print("   如果这是错误的，请检查配置文件")

    if not api_key or not api_secret:
        print(f"❌ 错误: 无法加载API key")
        print(f"   请确保文件存在: {env_file}")
        print(f"   或设置环境变量:")
        if testnet:
            print(f"     BINANCE_FUTURES_TESTNET_API_KEY")
            print(f"     BINANCE_FUTURES_TESTNET_API_SECRET")
        else:
            print(f"     BINANCE_API_KEY / BINANCE_FUTURES_API_KEY")
            print(f"     BINANCE_API_SECRET / BINANCE_FUTURES_API_SECRET")
        sys.exit(1)

    print(f"✅ 已加载API key: {api_key[:20]}...")
    print(f"📊 测试symbol: {symbols}")
    print(f"⏱️  运行时长: {duration_minutes} 分钟")
    network_type = "testnet" if testnet else "mainnet"
    print(f"🌐 使用{'测试网' if testnet else '主网'}")
    print(f"   网络类型: {network_type}")
    print()

    # 创建节点
    node = create_live_test_node(
        symbols=symbols,
        api_key=api_key,
        api_secret=api_secret,
        storage_path=storage_path,
        testnet=testnet,
    )

    try:
        # 启动节点（使用run_async方法）
        print("🚀 启动Nautilus Trader节点...")

        # 在后台运行节点
        node_task = asyncio.create_task(node.run_async())

        # 等待一小段时间让节点完全启动
        await asyncio.sleep(2)

        # 等待instruments自动加载（如果配置了load_all=True，会自动加载）
        print("📥 等待instruments加载...")
        # 给instrument provider一些时间加载instruments
        # 同时等待WebSocket连接建立
        await asyncio.sleep(10)  # 等待10秒让instruments加载完成和WebSocket连接建立

        # 检查instruments是否已加载（详细检查）
        try:
            cache = node.cache
            instruments = cache.instruments()
            if instruments:
                print(f"✅ 已加载 {len(instruments)} 个instruments")
                # 检查我们需要的instruments是否存在
                for symbol in symbols:
                    if "USDT" in symbol:
                        instrument_str = f"{symbol}-PERP.BINANCE"
                    else:
                        instrument_str = f"{symbol}.BINANCE"
                    instrument_id = InstrumentId.from_str(instrument_str)
                    cached_instrument = cache.instrument(instrument_id)
                    if cached_instrument:
                        print(f"   ✅ {symbol} ({instrument_str}) 已加载")
                        # 输出instrument详细信息用于调试
                        print(f"      Symbol: {cached_instrument.id.symbol}")
                        print(f"      Venue: {cached_instrument.id.venue}")
                        print(f"      Base Currency: {cached_instrument.base_currency}")
                        print(
                            f"      Quote Currency: {cached_instrument.quote_currency}"
                        )
                    else:
                        print(f"   ⚠️  {symbol} ({instrument_str}) 未找到")
                        print(f"      这可能导致订阅失败！")
                        # 列出所有已加载的instruments，帮助调试
                        print(f"      已加载的instruments (前10个):")
                        for inst in list(instruments)[:10]:
                            print(f"        - {inst}")
            else:
                print("⚠️  未找到已加载的instruments，继续运行...")
        except Exception as e:
            print(f"⚠️  检查instruments时出错: {e}")
            import traceback

            traceback.print_exc()
            print("   继续运行...")

        print("✅ 节点已启动，开始接收数据...")
        print("   每15分钟会输出一次特征摘要")
        print("   按 Ctrl+C 可提前停止")
        print()

        # 运行指定时长
        await asyncio.sleep(duration_minutes * 60)

        print(f"\n⏱️  运行时间已到（{duration_minutes} 分钟），停止测试...")

    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在停止...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 停止节点
        print("🛑 停止节点...")
        try:
            node.stop()
            # 等待节点任务完成
            if "node_task" in locals():
                await asyncio.wait_for(node_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("⚠️  节点停止超时，强制取消任务")
            if "node_task" in locals():
                node_task.cancel()
        except Exception as e:
            print(f"⚠️  停止节点时出错: {e}")
        print("✅ 节点已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="币安实盘测试：测试多symbol数据接收和订单流特征计算"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="交易对符号列表（默认: BTCUSDT ETHUSDT SOLUSDT）",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="运行时长（分钟，默认: 10）",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="使用测试网（默认: 主网）",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="data/live_storage",
        help="存储路径（默认: data/live_storage）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    # 设置日志级别
    # 为了调试订阅问题，默认启用DEBUG级别日志
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # 即使不verbose，也启用DEBUG以便查看订阅问题
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # 特别启用Nautilus Trader的WebSocket相关日志
    logging.getLogger("nautilus_trader.adapters.binance").setLevel(logging.DEBUG)
    logging.getLogger("nautilus_trader.adapters.binance.futures").setLevel(
        logging.DEBUG
    )
    logging.getLogger("nautilus_trader.adapters.binance.websocket").setLevel(
        logging.DEBUG
    )

    # 运行测试
    asyncio.run(
        run_live_test(
            symbols=args.symbols,
            duration_minutes=args.duration,
            testnet=args.testnet,
            storage_path=args.storage,
        )
    )


if __name__ == "__main__":
    main()
