"""
Nautilus Trader 集成示例

展示如何使用 Nautilus Trader 的数据客户端订阅实时 tick 数据流，
并集成到 OrderFlowListener。
"""

from __future__ import annotations

import asyncio
from typing import Optional
from pathlib import Path

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
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.trading.strategy import Strategy

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    Strategy = None
    TradeTick = None
    InstrumentId = None
    TradingNodeConfig = None
    CacheConfig = None

from .order_flow_listener import OrderFlowListener
from .multi_symbol_manager import MultiSymbolManager
from .feature_storage import StorageManager
from .gap_filler import GapFiller
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class OrderFlowStrategy(Strategy):
    """
    订单流策略（集成 OrderFlowListener）
    
    使用 Nautilus Trader 订阅实时 tick 数据，并传递给 OrderFlowListener 处理。
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        order_flow_listener: OrderFlowListener,
    ):
        """
        Args:
            instrument_id: 交易对ID
            order_flow_listener: OrderFlowListener实例
        """
        super().__init__()
        self.instrument_id = instrument_id
        self.listener = order_flow_listener
    
    def on_start(self) -> None:
        """策略启动时调用"""
        self.log.info(f"🚀 OrderFlowStrategy started for {self.instrument_id}")
        
        # 订阅 trade ticks
        self.subscribe_trade_ticks(self.instrument_id)
        self.log.info(f"✅ Subscribed to trade ticks: {self.instrument_id}")
        
        # Warmup（加载历史数据）
        try:
            warmup_data = self.listener.warmup(days=30, use_gap_filler=True)
            self.log.info(f"✅ Warmup completed: {len(warmup_data.get('ticks_1min', []))} ticks loaded")
        except Exception as e:
            self.log.warning(f"⚠️ Warmup failed: {e}")
        
        # 启动监听器
        asyncio.create_task(self.listener.start())
    
    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        处理 trade tick 事件（由 Nautilus Trader 数据客户端调用）
        
        Args:
            tick: TradeTick 对象
        """
        try:
            # 传递给 OrderFlowListener 处理
            self.listener.on_trade_tick(tick)
        except Exception as e:
            self.log.error(f"❌ Error processing trade tick: {e}")
            import traceback
            self.log.error(traceback.format_exc())
    
    def on_stop(self) -> None:
        """策略停止时调用"""
        self.log.info("🛑 OrderFlowStrategy stopping...")
        # 停止监听器
        asyncio.create_task(self.listener.stop())


def create_order_flow_node(
    symbol: str,
    storage_path: str = "data/live_storage",
    feature_store_dir: Optional[str] = None,
    feature_store_layer: Optional[str] = None,
    testnet: bool = True,
) -> TradingNode:
    """
    创建集成了 OrderFlowListener 的 Nautilus Trader 交易节点
    
    Args:
        symbol: 交易对符号（如 "BTCUSDT"）
        storage_path: 存储路径
        feature_store_dir: Feature Store目录（可选）
        feature_store_layer: Feature Store层名称（可选）
        testnet: 是否使用测试网
    
    Returns:
        TradingNode实例
    """
    if not NAUTILUS_AVAILABLE:
        raise ImportError("Nautilus Trader is not installed")
    
    # 1. 创建存储管理器
    storage_manager = StorageManager(base_path=storage_path)
    
    # 2. 创建特征计算器
    feature_computer = IncrementalFeatureComputer(
        tick_window_minutes=240,  # 4小时
        bar_window_size=240,
    )
    
    # 3. 创建数据补全器（如果需要）
    gap_filler = None
    if CCXT_AVAILABLE:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        gap_filler = GapFiller(
            storage_manager=storage_manager,
            exchange=exchange,
            feature_store_dir=feature_store_dir,
            feature_store_layer=feature_store_layer,
        )
    
    # 4. 创建 OrderFlowListener
    listener = OrderFlowListener(
        symbol=symbol,
        storage_manager=storage_manager,
        feature_computer=feature_computer,
        gap_filler=gap_filler,
        memory_window_hours=4.0,
        feature_compute_interval_minutes=15,
        feature_4h_interval_hours=4,
    )
    
    # 5. 创建 InstrumentId
    if "USDT" in symbol:
        instrument_str = f"{symbol}-PERP.BINANCE"
    else:
        instrument_str = f"{symbol}.BINANCE"
    instrument_id = InstrumentId.from_str(instrument_str)
    
    # 6. 创建策略
    strategy = OrderFlowStrategy(
        instrument_id=instrument_id,
        order_flow_listener=listener,
    )
    
    # 7. 创建交易节点配置
    import os
    
    config = TradingNodeConfig(
        trader_id="ORDER_FLOW_TRADER",
        cache=CacheConfig(tick_capacity=10000, bar_capacity=1000),
        data_engine=LiveDataEngineConfig(
            buffer_capacity=10000,
        ),
        exec_engine=LiveExecEngineConfig(),
        risk_engine=LiveRiskEngineConfig(),
        portfolio=PortfolioConfig(),
    )
    
    # 8. 配置 Binance 数据客户端
    account_type = BinanceAccountType.FUTURES_USDT_TESTNET if testnet else BinanceAccountType.FUTURES_USDT
    
    config.add_data_client(
        BINANCE,
        BinanceDataClientConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            account_type=account_type,
        ),
    )
    
    # 9. 创建交易节点
    node = TradingNode(config=config)
    
    # 10. 添加数据客户端工厂
    from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    
    # 11. 添加策略
    node.trader.add_strategy(strategy)
    
    return node


class MultiSymbolOrderFlowStrategy(Strategy):
    """
    多Symbol订单流策略（集成 MultiSymbolManager）
    
    使用 Nautilus Trader 订阅多个symbol的实时 tick 数据，并传递给 MultiSymbolManager 处理。
    """
    
    def __init__(
        self,
        instrument_ids: List[InstrumentId],
        multi_symbol_manager: MultiSymbolManager,
    ):
        """
        Args:
            instrument_ids: 交易对ID列表
            multi_symbol_manager: MultiSymbolManager实例
        """
        super().__init__()
        self.instrument_ids = instrument_ids
        self.manager = multi_symbol_manager
        # 创建symbol到instrument_id的映射
        self.symbol_map = {}
        for instrument_id in instrument_ids:
            # 从instrument_id提取symbol（如 BTCUSDT-PERP.BINANCE -> BTCUSDT）
            symbol = str(instrument_id).split("-")[0].split(".")[0]
            self.symbol_map[instrument_id] = symbol
    
    def on_start(self) -> None:
        """策略启动时调用"""
        self.log.info(f"🚀 MultiSymbolOrderFlowStrategy started for {len(self.instrument_ids)} symbols")
        
        # 订阅所有symbol的 trade ticks
        for instrument_id in self.instrument_ids:
            self.subscribe_trade_ticks(instrument_id)
            symbol = self.symbol_map[instrument_id]
            self.log.info(f"✅ Subscribed to trade ticks: {instrument_id} ({symbol})")
        
        # Warmup所有symbol（加载历史数据）
        try:
            warmup_results = asyncio.run(self.manager.warmup_all(days=30, use_gap_filler=True))
            total_ticks = sum(len(data.get('ticks_1min', [])) for data in warmup_results.values())
            self.log.info(f"✅ Warmup completed: {total_ticks} total ticks loaded across {len(warmup_results)} symbols")
        except Exception as e:
            self.log.warning(f"⚠️ Warmup failed: {e}")
        
        # 启动所有listener
        asyncio.create_task(self.manager.start_all())
    
    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        处理 trade tick 事件（由 Nautilus Trader 数据客户端调用）
        
        Args:
            tick: TradeTick 对象
        """
        try:
            # 从tick中获取symbol
            instrument_id = tick.instrument_id
            symbol = self.symbol_map.get(instrument_id)
            
            if symbol:
                # 传递给 MultiSymbolManager 处理
                self.manager.on_trade_tick(symbol, tick)
            else:
                self.log.warning(f"⚠️ Unknown instrument_id: {instrument_id}")
        except Exception as e:
            self.log.error(f"❌ Error processing trade tick: {e}")
            import traceback
            self.log.error(traceback.format_exc())
    
    def on_stop(self) -> None:
        """策略停止时调用"""
        self.log.info("🛑 MultiSymbolOrderFlowStrategy stopping...")
        # 停止所有listener
        asyncio.create_task(self.manager.stop_all())


def create_multi_symbol_order_flow_node(
    symbols: List[str],
    storage_path: str = "data/live_storage",
    feature_store_dir: Optional[str] = None,
    feature_store_layer: Optional[str] = None,
    testnet: bool = True,
) -> TradingNode:
    """
    创建集成了 MultiSymbolManager 的 Nautilus Trader 交易节点
    
    Args:
        symbols: 交易对符号列表（如 ["BTCUSDT", "ETHUSDT", "SOLUSDT"]）
        storage_path: 存储路径
        feature_store_dir: Feature Store目录（可选）
        feature_store_layer: Feature Store层名称（可选）
        testnet: 是否使用测试网
    
    Returns:
        TradingNode实例
    """
    if not NAUTILUS_AVAILABLE:
        raise ImportError("Nautilus Trader is not installed")
    
    import os
    
    # 1. 创建存储管理器
    storage_manager = StorageManager(base_path=storage_path)
    
    # 2. 创建数据补全器（如果需要）
    gap_filler = None
    if CCXT_AVAILABLE:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        gap_filler = GapFiller(
            storage_manager=storage_manager,
            exchange=exchange,
            feature_store_dir=feature_store_dir,
            feature_store_layer=feature_store_layer,
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
    strategy = MultiSymbolOrderFlowStrategy(
        instrument_ids=instrument_ids,
        multi_symbol_manager=multi_manager,
    )
    
    # 6. 创建交易节点配置
    config = TradingNodeConfig(
        trader_id="MULTI_SYMBOL_ORDER_FLOW_TRADER",
        cache=CacheConfig(tick_capacity=10000, bar_capacity=1000),
        data_engine=LiveDataEngineConfig(
            buffer_capacity=10000,
        ),
        exec_engine=LiveExecEngineConfig(),
        risk_engine=LiveRiskEngineConfig(),
        portfolio=PortfolioConfig(),
    )
    
    # 7. 配置 Binance 数据客户端
    account_type = BinanceAccountType.FUTURES_USDT_TESTNET if testnet else BinanceAccountType.FUTURES_USDT
    
    config.add_data_client(
        BINANCE,
        BinanceDataClientConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            account_type=account_type,
        ),
    )
    
    # 8. 创建交易节点
    node = TradingNode(config=config)
    
    # 9. 添加数据客户端工厂
    from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    
    # 10. 添加策略
    node.trader.add_strategy(strategy)
    
    return node


async def run_order_flow_listener(
    symbol: str,
    storage_path: str = "data/live_storage",
    feature_store_dir: Optional[str] = None,
    feature_store_layer: Optional[str] = None,
    testnet: bool = True,
) -> None:
    """
    运行订单流监听器（使用 Nautilus Trader 订阅实时数据）
    
    Args:
        symbol: 交易对符号
        storage_path: 存储路径
        feature_store_dir: Feature Store目录（可选）
        feature_store_layer: Feature Store层名称（可选）
        testnet: 是否使用测试网
    """
    # 创建交易节点
    node = create_order_flow_node(
        symbol=symbol,
        storage_path=storage_path,
        feature_store_dir=feature_store_dir,
        feature_store_layer=feature_store_layer,
        testnet=testnet,
    )
    
    try:
        # 启动节点
        node.start()
        print(f"✅ OrderFlowListener started for {symbol}")
        print("   Listening for trade ticks from Nautilus Trader...")
        
        # 运行直到停止
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping OrderFlowListener...")
    finally:
        node.stop()
        print("✅ OrderFlowListener stopped")


async def run_multi_symbol_order_flow_listener(
    symbols: List[str],
    storage_path: str = "data/live_storage",
    feature_store_dir: Optional[str] = None,
    feature_store_layer: Optional[str] = None,
    testnet: bool = True,
) -> None:
    """
    运行多symbol订单流监听器（使用 Nautilus Trader 订阅多个symbol的实时数据）
    
    Args:
        symbols: 交易对符号列表（如 ["BTCUSDT", "ETHUSDT", "SOLUSDT"]）
        storage_path: 存储路径
        feature_store_dir: Feature Store目录（可选）
        feature_store_layer: Feature Store层名称（可选）
        testnet: 是否使用测试网
    """
    # 创建交易节点
    node = create_multi_symbol_order_flow_node(
        symbols=symbols,
        storage_path=storage_path,
        feature_store_dir=feature_store_dir,
        feature_store_layer=feature_store_layer,
        testnet=testnet,
    )
    
    try:
        # 启动节点
        node.start()
        print(f"✅ MultiSymbolOrderFlowListener started for {len(symbols)} symbols: {symbols}")
        print("   Listening for trade ticks from Nautilus Trader...")
        
        # 运行直到停止
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping MultiSymbolOrderFlowListener...")
    finally:
        node.stop()
        print("✅ MultiSymbolOrderFlowListener stopped")


if __name__ == "__main__":
    import os
    
    # 示例1：单symbol订单流监听器
    # 注意：需要设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET
    # 或使用测试网：BINANCE_FUTURES_TESTNET_API_KEY 和 BINANCE_FUTURES_TESTNET_API_SECRET
    asyncio.run(run_order_flow_listener(
        symbol="BTCUSDT",
        storage_path="data/live_storage",
        testnet=True,  # 使用测试网
    ))
    
    # 示例2：多symbol订单流监听器
    # asyncio.run(run_multi_symbol_order_flow_listener(
    #     symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    #     storage_path="data/live_storage",
    #     testnet=True,
    # ))