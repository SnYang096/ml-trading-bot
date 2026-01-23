"""
实盘测试策略（空策略）

用于测试多symbol数据接收和订单流特征计算，不执行任何交易。
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.trading.strategy import Strategy

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    Strategy = None
    TradeTick = None
    InstrumentId = None

from .multi_symbol_manager import MultiSymbolManager
from .feature_storage import StorageManager

logger = logging.getLogger(__name__)


class LiveTestStrategy(Strategy):
    """
    实盘测试策略（空策略）
    
    功能：
    - 监听多个symbol的实时tick数据
    - 计算订单流特征（vpin, cvd, tradecluster, volprofile, vwap）
    - 定期输出特征摘要
    - 不执行任何交易
    """
    
    def __init__(
        self,
        instrument_ids: List[InstrumentId],
        multi_symbol_manager: MultiSymbolManager,
        feature_output_interval_minutes: int = 15,
    ):
        """
        Args:
            instrument_ids: 交易对ID列表
            multi_symbol_manager: MultiSymbolManager实例
            feature_output_interval_minutes: 特征输出间隔（分钟）
        """
        super().__init__()
        self.instrument_ids = instrument_ids
        self.manager = multi_symbol_manager
        self.feature_output_interval = feature_output_interval_minutes
        
        # 创建symbol到instrument_id的映射
        self.symbol_map = {}
        for instrument_id in instrument_ids:
            # 从instrument_id提取symbol（如 BTCUSDT-PERP.BINANCE -> BTCUSDT）
            symbol = str(instrument_id).split("-")[0].split(".")[0]
            self.symbol_map[instrument_id] = symbol
        
        # 统计信息
        self.tick_counts: Dict[str, int] = {symbol: 0 for symbol in self.symbol_map.values()}
        self.last_feature_output_time: Optional[datetime] = None
        self._feature_output_task: Optional[asyncio.Task] = None
    
    def on_start(self) -> None:
        """策略启动时调用"""
        self.log.info(f"🚀 LiveTestStrategy started for {len(self.instrument_ids)} symbols")
        self.log.info(f"   Symbols: {list(self.symbol_map.values())}")
        
        # 检查数据客户端是否已连接
        try:
            from nautilus_trader.adapters.binance import BINANCE
            data_engine = self.trader.data_engine
            data_client = data_engine.get_client(BINANCE)
            if data_client:
                if data_client.is_connected:
                    self.log.info(f"✅ 数据客户端已连接: {data_client}")
                else:
                    self.log.warning(f"⚠️  数据客户端未连接: {data_client}")
            else:
                self.log.warning(f"⚠️  无法获取数据客户端")
        except Exception as e:
            self.log.warning(f"⚠️  检查数据客户端时出错: {e}")
        
        # 订阅所有symbol的 trade ticks
        for instrument_id in self.instrument_ids:
            symbol = self.symbol_map[instrument_id]
            self.log.info(f"📡 准备订阅: {instrument_id} ({symbol})")
            self.log.debug(f"   Instrument ID详情: symbol={instrument_id.symbol}, venue={instrument_id.venue}")
            
            # 检查instrument是否在cache中
            try:
                cache = self.cache
                cached_instrument = cache.instrument(instrument_id)
                if cached_instrument:
                    self.log.debug(f"   ✅ Instrument在cache中")
                    self.log.debug(f"      Symbol: {cached_instrument.id.symbol}")
                    self.log.debug(f"      Venue: {cached_instrument.id.venue}")
                else:
                    self.log.error(f"   ❌ Instrument不在cache中: {instrument_id}")
                    self.log.error(f"      订阅需要instrument在cache中才能工作！")
                    continue
            except Exception as e:
                self.log.error(f"   ❌ 检查cache时出错: {e}")
                continue
            
            try:
                # 订阅trade ticks
                # 根据Nautilus Trader文档，订阅trade ticks是标准功能
                # client_id可以是None，会自动从instrument_id推断venue
                # params参数可以传递额外的订阅参数（根据API文档支持）
                
                # 记录订阅前的状态
                self.log.debug(f"   订阅前检查:")
                self.log.debug(f"      Instrument ID: {instrument_id}")
                self.log.debug(f"      Instrument在cache: {cached_instrument is not None}")
                
                # 检查数据客户端连接状态（在订阅前）
                try:
                    from nautilus_trader.adapters.binance import BINANCE
                    if hasattr(self, '_trader'):
                        trader = self._trader
                    elif hasattr(self, 'trader'):
                        trader = self.trader
                    else:
                        trader = None
                    
                    if trader:
                        data_engine = trader.data_engine
                        data_client = data_engine.get_client(BINANCE)
                        if data_client:
                            self.log.debug(f"      数据客户端状态: {data_client.state}")
                            self.log.debug(f"      数据客户端已连接: {data_client.is_connected}")
                            if hasattr(data_client, 'base_url_ws'):
                                self.log.debug(f"      WebSocket URL: {data_client.base_url_ws}")
                except Exception as e:
                    self.log.debug(f"      检查数据客户端状态时出错: {e}")
                
                # 发送订阅命令
                # 注意：根据Nautilus Trader文档，Strategy.subscribe_trade_ticks内部会创建SubscribeTradeTicks命令
                # 并传递给相应的DataClient
                self.subscribe_trade_ticks(
                    instrument_id, 
                    client_id=None,
                    params=None  # 可以传递额外的订阅参数
                )
                
                self.log.info(f"✅ 订阅命令已发送: {instrument_id} ({symbol})")
                self.log.debug(f"   订阅参数: instrument_id={instrument_id}, client_id=None, params=None")
                self.log.debug(f"   订阅命令已通过Strategy.subscribe_trade_ticks发送")
                self.log.debug(f"   注意：Strategy.subscribe_trade_ticks会创建SubscribeTradeTicks命令并传递给DataClient")
            except Exception as e:
                self.log.error(f"❌ 订阅失败: {instrument_id} ({symbol}), 错误: {e}")
                import traceback
                self.log.error(traceback.format_exc())
        
        # Warmup所有symbol（加载历史数据）
        # 注意：在异步环境中，不能使用asyncio.run()，应该使用create_task
        try:
            # 创建warmup任务（异步执行）
            warmup_task = asyncio.create_task(self.manager.warmup_all(days=1, use_gap_filler=False))
            # 不等待完成，让它在后台运行
            self.log.info("⏳ Warmup started in background...")
        except Exception as e:
            self.log.warning(f"⚠️ Warmup failed: {e}")
        
        # 启动所有listener
        asyncio.create_task(self.manager.start_all())
        
        # 启动特征输出任务
        self._feature_output_task = asyncio.create_task(self._periodic_feature_output())
        
        # 尝试检查订阅状态（作为调试辅助）
        # 注意：这需要等待一段时间让节点完全启动
        # 使用asyncio.create_task在后台运行
        try:
            asyncio.create_task(self._check_subscription_status())
        except AttributeError:
            # 如果方法不存在，忽略错误（向后兼容）
            self.log.debug("_check_subscription_status方法不可用，跳过订阅状态检查")
    
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
            
            # 如果直接查找失败，尝试通过字符串匹配（处理InstrumentId对象比较问题）
            if not symbol:
                # 尝试通过字符串匹配查找
                instrument_id_str = str(instrument_id)
                for key, value in self.symbol_map.items():
                    if str(key) == instrument_id_str:
                        symbol = value
                        break
                
                # 如果还是找不到，尝试从instrument_id直接提取symbol
                if not symbol:
                    # 从instrument_id提取symbol（如 ETHUSDT-PERP.BINANCE -> ETHUSDT）
                    symbol = str(instrument_id).split("-")[0].split(".")[0]
                    # 检查这个symbol是否在我们的symbol列表中
                    if symbol not in self.symbol_map.values():
                        symbol = None
            
            if symbol:
                # 传递给 MultiSymbolManager 处理
                self.manager.on_trade_tick(symbol, tick)
                
                # 更新统计
                self.tick_counts[symbol] = self.tick_counts.get(symbol, 0) + 1
                
                # 前10条tick输出日志（用于调试）
                if self.tick_counts[symbol] <= 10:
                    self.log.info(f"📊 {symbol}: 收到第 {self.tick_counts[symbol]} 条tick (price={tick.price}, size={tick.size})")
                    self.log.debug(f"   Instrument ID: {instrument_id}")
                    self.log.debug(f"   Symbol映射: {self.symbol_map.get(instrument_id, 'Not found')}")
                
                # 每1000条tick输出一次统计
                if self.tick_counts[symbol] % 1000 == 0:
                    self.log.info(f"📊 {symbol}: 已处理 {self.tick_counts[symbol]} 条tick")
            else:
                self.log.warning(f"⚠️ Unknown instrument_id: {instrument_id}")
                self.log.warning(f"   Instrument ID type: {type(instrument_id)}")
                self.log.warning(f"   Instrument ID string: {str(instrument_id)}")
                self.log.debug(f"   已知的symbol映射: {self.symbol_map}")
                self.log.debug(f"   已知的instrument IDs: {list(self.symbol_map.keys())}")
        except Exception as e:
            self.log.error(f"❌ Error processing trade tick: {e}")
            import traceback
            self.log.error(traceback.format_exc())
    
    async def _periodic_feature_output(self) -> None:
        """定期输出特征摘要"""
        while True:
            try:
                await asyncio.sleep(self.feature_output_interval * 60)  # 转换为秒
                
                # 输出特征摘要
                self._output_feature_summary()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"❌ Error in feature output: {e}")
    
    def _output_feature_summary(self) -> None:
        """输出特征摘要"""
        self.log.info("=" * 80)
        self.log.info("📊 订单流特征摘要")
        self.log.info("=" * 80)
        
        for symbol in self.symbol_map.values():
            listener = self.manager.get_listener(symbol)
            if listener is None:
                continue
            
            self.log.info(f"\n🔹 {symbol}:")
            self.log.info(f"   已处理tick数: {self.tick_counts.get(symbol, 0)}")
            
            # 获取特征
            features = listener.feature_computer.get_features() or {}
            
            # 获取时间框架特征（包含更多订单流特征）
            timeframe_features = listener.feature_computer.timeframe_features.get("1min", {})
            
            # 获取订单流特征（15分钟窗口）
            try:
                orderflow_features = listener.feature_computer.get_orderflow_features(window_minutes=15)
            except Exception as e:
                self.log.warning(f"⚠️ 获取订单流特征失败: {e}")
                orderflow_features = {}
            
            # 合并所有特征
            all_features = {
                **features,
                **timeframe_features,
                **orderflow_features,
            }
            
            # 输出关键特征
            self._log_feature_group(all_features, "VPIN", ["vpin"])
            self._log_feature_group(all_features, "CVD", ["cvd", "cvd_change"])
            self._log_feature_group(all_features, "Trade Cluster", ["trade_cluster"])
            self._log_feature_group(all_features, "Volume Profile", ["vp_", "vpvr", "volume_profile"])
            self._log_feature_group(all_features, "VWAP", ["vwap"])
            
            # 内存窗口状态
            memory_window = listener.get_memory_window()
            self.log.info(f"   内存窗口: {len(memory_window)} 条bar")
            
            # 数据保存状态
            recovery_state = listener.get_recovery_state()
            if recovery_state.get("latest_1min_timestamp"):
                self.log.info(f"   最新1分钟bar: {recovery_state['latest_1min_timestamp']}")
            if recovery_state.get("latest_15min_timestamp"):
                self.log.info(f"   最新15分钟特征: {recovery_state['latest_15min_timestamp']}")
        
        self.log.info("=" * 80)
        self.last_feature_output_time = datetime.now()
    
    def _log_feature_group(self, features: Dict[str, Any], group_name: str, keywords: List[str]) -> None:
        """输出特征组"""
        group_features = {}
        for key, value in features.items():
            if any(kw in key.lower() for kw in keywords):
                group_features[key] = value
        
        if group_features:
            self.log.info(f"   {group_name}:")
            for key, value in sorted(group_features.items()):
                if isinstance(value, (int, float)):
                    self.log.info(f"     {key}: {value:.6f}" if isinstance(value, float) else f"     {key}: {value}")
                else:
                    self.log.info(f"     {key}: {value}")
    
    async def _check_subscription_status(self) -> None:
        """
        检查订阅状态（作为调试辅助）
        等待节点完全启动后再检查
        
        这个方法的目的是：
        1. 检查订阅任务是否成功创建
        2. 检查订阅任务是否被取消
        3. 分析订阅失败的原因
        """
        try:
            # 等待一段时间让节点完全启动
            await asyncio.sleep(10)
            
            from nautilus_trader.adapters.binance import BINANCE
            
            # 获取trader和data_client
            if hasattr(self, '_trader'):
                trader = self._trader
            elif hasattr(self, 'trader'):
                trader = self.trader
            else:
                self.log.debug("无法访问trader对象，跳过订阅状态检查")
                return
            
            data_engine = trader.data_engine
            data_client = data_engine.get_client(BINANCE)
            
            if not data_client:
                self.log.warning("⚠️  无法获取数据客户端，订阅可能失败")
                return
            
            # 检查数据客户端状态
            self.log.info(f"📊 数据客户端状态检查:")
            self.log.info(f"   状态: {data_client.state}")
            self.log.info(f"   已连接: {data_client.is_connected}")
            
            if hasattr(data_client, 'base_url_ws'):
                self.log.info(f"   WebSocket URL: {data_client.base_url_ws}")
            
            # 检查已订阅的trade ticks
            try:
                subscribed = data_client.subscribed_trade_ticks()
                self.log.info(f"📊 当前已订阅的trade ticks: {len(subscribed)} 个")
                
                # 检查每个instrument的订阅状态
                for instrument_id in self.instrument_ids:
                    symbol = self.symbol_map.get(instrument_id, "Unknown")
                    if instrument_id in subscribed:
                        self.log.info(f"   ✅ {symbol} ({instrument_id}) 已订阅")
                    else:
                        self.log.warning(f"   ⚠️  {symbol} ({instrument_id}) 未订阅")
                        self.log.warning(f"      可能的原因:")
                        self.log.warning(f"      1. 订阅任务被取消")
                        self.log.warning(f"      2. WebSocket连接未建立")
                        self.log.warning(f"      3. Instrument未正确加载")
                        self.log.warning(f"      4. 订阅命令未正确传递到DataClient")
                
                # 如果没有任何订阅，输出警告
                if len(subscribed) == 0:
                    self.log.error("❌ 没有任何trade ticks被订阅！")
                    self.log.error("   这可能是订阅任务被取消或WebSocket连接问题")
                    
            except Exception as e:
                self.log.warning(f"⚠️  检查订阅状态时出错: {e}")
                import traceback
                self.log.debug(traceback.format_exc())
                
            # 检查是否有其他订阅（quote ticks, bars等）
            try:
                if hasattr(data_client, 'subscribed_quote_ticks'):
                    quote_subscribed = data_client.subscribed_quote_ticks()
                    if quote_subscribed:
                        self.log.debug(f"   已订阅的quote ticks: {len(quote_subscribed)} 个")
            except:
                pass
                
        except Exception as e:
            self.log.warning(f"⚠️  检查订阅状态时出错: {e}")
            import traceback
            self.log.debug(traceback.format_exc())
    
    def on_stop(self) -> None:
        """策略停止时调用"""
        self.log.info("🛑 LiveTestStrategy stopping...")
        
        # 停止特征输出任务
        if self._feature_output_task:
            self._feature_output_task.cancel()
        
        # 输出最终统计
        self._output_feature_summary()
        
        # 输出最终统计信息
        self.log.info("\n" + "=" * 80)
        self.log.info("📊 最终统计")
        self.log.info("=" * 80)
        for symbol, count in self.tick_counts.items():
            self.log.info(f"   {symbol}: {count} 条tick")
        self.log.info("=" * 80)
        
        # 停止所有listener
        asyncio.create_task(self.manager.stop_all())
