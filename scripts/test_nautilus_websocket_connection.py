"""
测试Nautilus Trader WebSocket连接
测试在没有VPN的情况下Nautilus Trader能否连接

注意：直接WebSocket测试请使用：
- scripts/test_binance_testnet_websocket.py
- scripts/test_binance_mainnet_websocket.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 尝试导入Nautilus Trader
try:
    from nautilus_trader.config import TradingNodeConfig
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.adapters.binance.config import (
        BinanceDataClientConfig,
    )
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.adapters.binance import BINANCE
    from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
    from nautilus_trader.common.config import InstrumentProviderConfig
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.config import (
        CacheConfig,
        LiveDataEngineConfig,
        LiveExecEngineConfig,
        LiveRiskEngineConfig,
        PortfolioConfig,
    )

    NAUTILUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Nautilus Trader未安装: {e}")
    NAUTILUS_AVAILABLE = False
    # 定义占位符以避免NameError
    Strategy = None
    TradingNode = None


def load_api_keys(testnet: bool = False):
    """加载API密钥"""
    if testnet:
        env_file = project_root / "config" / "local" / "binance_testnet.env"
    else:
        env_file = project_root / "config" / "local" / "binance_mainnet.env"

    if not env_file.exists():
        logger.warning(f"API密钥文件不存在: {env_file}，将使用空密钥")
        return None, None

    api_key = None
    api_secret = None

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BINANCE_FUTURES_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("BINANCE_FUTURES_API_SECRET="):
                api_secret = line.split("=", 1)[1].strip()
            elif testnet and line.startswith("BINANCE_FUTURES_TESTNET_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif testnet and line.startswith("BINANCE_FUTURES_TESTNET_API_SECRET="):
                api_secret = line.split("=", 1)[1].strip()

    return api_key, api_secret


if NAUTILUS_AVAILABLE:

    class TestStrategy(Strategy):
        """测试策略，用于接收tick数据"""

        def __init__(self):
            super().__init__()
            self.instrument_ids = []
            self.tick_count = 0
            self.received_symbols = set()
            self.first_tick_time = None
            self.last_tick_time = None

        def on_start(self):
            """策略启动"""
            self.log.info("策略已启动")
            # 订阅交易tick
            for instrument_id in self.instrument_ids:
                try:
                    self.subscribe_trade_ticks(instrument_id)
                    self.log.info(f"已订阅交易tick: {instrument_id}")
                except Exception as e:
                    self.log.error(f"订阅tick失败 {instrument_id}: {e}")

        def on_instrument(self, instrument):
            """Instrument加载回调"""
            self.log.info(f"Instrument已加载: {instrument.id}")

        def on_trade_tick(self, tick: TradeTick):
            """接收交易tick"""
            self.tick_count += 1
            symbol = str(tick.instrument_id.symbol)
            self.received_symbols.add(symbol)

            if self.first_tick_time is None:
                self.first_tick_time = tick.ts_event

            self.last_tick_time = tick.ts_event

            if self.tick_count <= 5:
                self.log.info(
                    f"收到tick [{self.tick_count}]: "
                    f"交易对={symbol}, 价格={tick.price}, "
                    f"数量={tick.size}, 时间={tick.ts_event}"
                )
            elif self.tick_count % 10 == 0:
                self.log.info(f"已收到 {self.tick_count} 条tick数据")

        def get_summary(self):
            """获取摘要"""
            return {
                "tick_count": self.tick_count,
                "symbols": sorted(self.received_symbols),
                "first_tick": self.first_tick_time,
                "last_tick": self.last_tick_time,
            }


async def test_nautilus_connection(
    symbols: list[str], testnet: bool = False, duration_seconds: int = 30
):
    """
    测试Nautilus Trader连接

    Args:
        symbols: 交易对列表
        testnet: 是否使用测试网
        duration_seconds: 测试持续时间（秒）
    """
    if not NAUTILUS_AVAILABLE:
        logger.error("❌ Nautilus Trader未安装，无法测试")
        return False

    logger.info(f"测试Nautilus Trader连接 ({'测试网' if testnet else '主网'})")
    logger.info(f"交易对: {symbols}")
    logger.info(f"持续时间: {duration_seconds}秒")

    # 加载API密钥
    api_key, api_secret = load_api_keys(testnet)

    # 配置Binance数据客户端
    account_type = BinanceAccountType.USDT_FUTURES

    # 创建instrument_id列表
    instrument_ids = []
    for symbol in symbols:
        if "USDT" in symbol:
            instrument_str = f"{symbol}-PERP.BINANCE"
        else:
            instrument_str = f"{symbol}.BINANCE"
        instrument_ids.append(InstrumentId.from_str(instrument_str))

    instrument_provider_config = InstrumentProviderConfig(
        load_all=False,
        load_ids=frozenset(instrument_ids) if instrument_ids else None,
    )

    data_client_config = BinanceDataClientConfig(
        api_key=api_key or "",
        api_secret=api_secret or "",
        account_type=account_type,
        testnet=testnet,
        use_agg_trade_ticks=False,  # 使用原始trade ticks
        instrument_provider=instrument_provider_config,
    )

    # 配置交易节点
    node_config = TradingNodeConfig(
        trader_id="TEST-001",
        cache=CacheConfig(tick_capacity=10000, bar_capacity=1000),
        data_engine=LiveDataEngineConfig(),
        exec_engine=LiveExecEngineConfig(),
        risk_engine=LiveRiskEngineConfig(),
        portfolio=PortfolioConfig(),
        data_clients={
            BINANCE: data_client_config,
        },
        exec_clients={},
        strategies=[],  # 先不添加策略，后面手动添加
    )

    # 创建节点
    node = TradingNode(config=node_config)

    # 添加数据客户端工厂
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)

    # 创建并添加策略
    strategy = TestStrategy()
    strategy.instrument_ids = instrument_ids
    node.trader.add_strategy(strategy)

    try:
        logger.info("启动Nautilus Trader节点...")
        node.build()

        # 启动节点（run_async返回一个任务，需要在后台运行）
        import asyncio as aio

        run_task = aio.create_task(node.run_async())
        logger.info("等待节点启动和连接...")
        await asyncio.sleep(5)  # 等待节点启动

        logger.info("等待instrument加载和策略启动...")
        await asyncio.sleep(5)  # 等待instrument加载

        logger.info("等待数据订阅生效...")
        await asyncio.sleep(10)  # 增加等待时间，让WebSocket连接建立

        logger.info("节点已启动，等待接收数据...")

        # 检查数据客户端连接状态
        try:
            data_client = node.trader.data_engine.get_client(BINANCE)
            if data_client:
                logger.info(
                    f"数据客户端状态: {data_client.is_connected if hasattr(data_client, 'is_connected') else 'unknown'}"
                )
        except Exception as e:
            logger.warning(f"检查数据客户端状态时出错: {e}")

        # 检查instrument是否已加载
        try:
            for instrument_id in instrument_ids:
                instrument = node.cache.instrument(instrument_id)
                if instrument:
                    logger.info(f"✅ Instrument已加载: {instrument_id}")
                else:
                    logger.warning(f"⚠️ Instrument未加载: {instrument_id}")
        except Exception as e:
            logger.warning(f"检查instrument时出错: {e}")

        # 等待指定时间
        await asyncio.sleep(duration_seconds)

        # 获取摘要
        summary = strategy.get_summary()

        logger.info("=" * 60)
        logger.info("测试结果:")
        logger.info(f"  收到tick数量: {summary['tick_count']}")
        logger.info(f"  收到数据的交易对: {summary['symbols']}")
        if summary["first_tick"]:
            logger.info(f"  第一条tick时间: {summary['first_tick']}")
        if summary["last_tick"]:
            logger.info(f"  最后一条tick时间: {summary['last_tick']}")
        logger.info("=" * 60)

        if summary["tick_count"] > 0:
            logger.info("✅ Nautilus Trader连接成功，可以接收数据")
            return True
        else:
            logger.warning("⚠️ Nautilus Trader连接成功，但未收到数据")
            return False

    except Exception as e:
        logger.error(f"❌ Nautilus Trader连接失败: {e}", exc_info=True)
        return False
    finally:
        try:
            logger.info("停止节点...")
            node.stop()
            # 等待run_task完成
            try:
                await asyncio.wait_for(run_task, timeout=5)
            except asyncio.TimeoutError:
                run_task.cancel()
            await asyncio.sleep(2)
            node.dispose()
        except Exception as e:
            logger.error(f"停止节点时出错: {e}")


async def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Nautilus Trader WebSocket连接测试")
    logger.info("=" * 60)

    if not NAUTILUS_AVAILABLE:
        logger.error("❌ Nautilus Trader未安装")
        logger.info("请安装: pip install nautilus_trader")
        return

    # 测试1: 测试网
    logger.info("\n测试1: Nautilus Trader - 测试网")
    logger.info("-" * 60)
    success1 = await test_nautilus_connection(
        symbols=["BTCUSDT"], testnet=True, duration_seconds=40
    )

    # 测试2: 主网
    logger.info("\n测试2: Nautilus Trader - 主网")
    logger.info("-" * 60)
    success2 = await test_nautilus_connection(
        symbols=["BTCUSDT"], testnet=False, duration_seconds=20
    )

    # 测试3: 多交易对（测试网）
    logger.info("\n测试3: Nautilus Trader - 多交易对（测试网）")
    logger.info("-" * 60)
    success3 = await test_nautilus_connection(
        symbols=["BTCUSDT", "ETHUSDT"], testnet=True, duration_seconds=20
    )

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"测试网单交易对: {'✅ 成功' if success1 else '❌ 失败'}")
    logger.info(f"主网单交易对: {'✅ 成功' if success2 else '❌ 失败'}")
    logger.info(f"测试网多交易对: {'✅ 成功' if success3 else '❌ 失败'}")

    if success1 or success2 or success3:
        logger.info("\n✅ 至少有一个Nautilus Trader连接成功！")
    else:
        logger.error("\n❌ 所有Nautilus Trader连接都失败！")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
