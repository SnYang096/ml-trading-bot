"""
监控服务
实时监控仓位和订单状态，提供告警功能，集成Prometheus和Grafana
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from threading import Thread

from .storage import Storage
from .position_manager import PositionManager
from .order_manager import OrderManager
from .binance_api import BinanceAPI
from . import metrics

logger = logging.getLogger(__name__)


class MonitoringService:
    """监控服务"""
    
    def __init__(
        self,
        storage: Storage,
        position_manager: PositionManager,
        order_manager: OrderManager,
        binance_api: BinanceAPI,
        update_interval: int = 5
    ):
        """
        初始化监控服务
        
        Args:
            storage: 存储层实例
            position_manager: 仓位管理器实例
            order_manager: 订单管理器实例
            binance_api: Binance API实例
            update_interval: 更新间隔（秒）
        """
        self.storage = storage
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.binance_api = binance_api
        self.update_interval = update_interval
        
        self._running = False
        self._monitor_thread: Optional[Thread] = None
        self._alert_callbacks: List[Callable[[str, str], None]] = []
        self._last_reconcile_time: Optional[float] = None
        self._reconcile_interval = 60  # seconds
    
    def start(self):
        """启动监控服务"""
        if self._running:
            logger.warning("监控服务已在运行")
            return
        
        self._running = True
        # 启动前先进行一次对账
        try:
            self.order_manager.reconcile_open_orders()
        except Exception as e:
            logger.warning(f"启动对账失败: {e}")
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("监控服务已启动")
    
    def stop(self):
        """停止监控服务"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("监控服务已停止")
    
    def register_alert_callback(self, callback: Callable[[str, str], None]):
        """
        注册告警回调函数
        
        Args:
            callback: 回调函数，参数为(告警类型, 告警消息)
        """
        self._alert_callbacks.append(callback)
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._update_metrics()
                self._check_alerts()
                self._maybe_reconcile_orders()
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            # 等待更新间隔
            import time
            time.sleep(self.update_interval)

    def _maybe_reconcile_orders(self) -> None:
        """定期对账订单状态"""
        import time
        now = time.time()
        if self._last_reconcile_time is None or now - self._last_reconcile_time >= self._reconcile_interval:
            try:
                self.order_manager.reconcile_open_orders()
            except Exception as e:
                logger.warning(f"定期对账失败: {e}")
            self._last_reconcile_time = now
    
    def _update_metrics(self):
        """更新Prometheus指标"""
        try:
            # 更新仓位指标
            open_positions = self.position_manager.get_open_positions()
            total_unrealized_pnl = 0.0
            total_value = 0.0
            
            # 按symbol和side统计
            position_counts: Dict[tuple, int] = {}
            
            for position in open_positions:
                # 更新仓位数量
                key = (position.symbol, position.side.value)
                position_counts[key] = position_counts.get(key, 0) + 1
                
                # 更新未实现盈亏
                if position.unrealized_pnl:
                    metrics.position_unrealized_pnl.labels(symbol=position.symbol).set(
                        position.unrealized_pnl
                    )
                    total_unrealized_pnl += position.unrealized_pnl
                
                # 更新总价值
                if position.total_value:
                    metrics.position_total_value.labels(symbol=position.symbol).set(
                        position.total_value
                    )
                    total_value += position.total_value
            
            # 更新仓位计数
            for (symbol, side), count in position_counts.items():
                metrics.position_count.labels(symbol=symbol, side=side).set(count)
            
            # 更新风险指标
            account_info = self.binance_api.get_account_info()
            total_balance = account_info.get('total_balance', 0)
            used_balance = account_info.get('used_balance', 0)
            
            if total_balance > 0:
                margin_ratio = used_balance / total_balance
                metrics.margin_usage_ratio.set(margin_ratio)
            
            # 更新每日盈亏（简化实现，实际应该从数据库查询）
            metrics.daily_pnl.set(total_unrealized_pnl)
            
            # 同步订单状态并更新订单指标
            self._update_order_metrics()
            
        except Exception as e:
            logger.error(f"更新指标失败: {e}")
    
    def _update_order_metrics(self):
        """更新订单指标"""
        try:
            # 同步所有未完成订单
            updated_orders = self.order_manager.sync_all_orders()
            
            # 统计订单状态
            for order in updated_orders:
                metrics.orders_total.labels(
                    status=order.status.value,
                    type=order.order_type.value,
                    symbol=order.symbol
                ).inc()
        except Exception as e:
            logger.error(f"更新订单指标失败: {e}")
    
    def _check_alerts(self):
        """检查告警条件"""
        try:
            # 检查止损触发
            self._check_stop_loss_alerts()
            
            # 检查保证金不足
            self._check_margin_alerts()
            
            # 检查API错误率
            # self._check_api_error_rate()
            
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
    
    def _check_stop_loss_alerts(self):
        """检查止损触发告警"""
        open_positions = self.position_manager.get_open_positions()
        
        for position in open_positions:
            if not position.stop_loss_price:
                continue
            
            # 获取当前价格
            binance_position = self.binance_api.get_position(position.symbol)
            if not binance_position:
                continue
            
            current_price = binance_position.get('mark_price')
            if not current_price:
                continue
            
            # 检查是否触发止损
            if position.side.value == 'long':
                if current_price <= position.stop_loss_price:
                    self._trigger_alert(
                        'stop_loss_triggered',
                        f"止损触发: {position.symbol}, 当前价格={current_price}, 止损价={position.stop_loss_price}"
                    )
            else:  # short
                if current_price >= position.stop_loss_price:
                    self._trigger_alert(
                        'stop_loss_triggered',
                        f"止损触发: {position.symbol}, 当前价格={current_price}, 止损价={position.stop_loss_price}"
                    )
    
    def _check_margin_alerts(self):
        """检查保证金告警"""
        account_info = self.binance_api.get_account_info()
        total_balance = account_info.get('total_balance', 0)
        used_balance = account_info.get('used_balance', 0)
        
        if total_balance > 0:
            margin_ratio = used_balance / total_balance
            
            # 保证金使用率超过80%告警
            if margin_ratio > 0.8:
                self._trigger_alert(
                    'high_margin_usage',
                    f"保证金使用率过高: {margin_ratio:.2%}, 可用余额={total_balance - used_balance:.2f} USDT"
                )
            
            # 可用余额不足告警
            available_balance = total_balance - used_balance
            if available_balance < 100:  # 小于100 USDT告警
                self._trigger_alert(
                    'low_available_balance',
                    f"可用余额不足: {available_balance:.2f} USDT"
                )
    
    def _trigger_alert(self, alert_type: str, message: str):
        """
        触发告警
        
        Args:
            alert_type: 告警类型
            message: 告警消息
        """
        logger.warning(f"告警 [{alert_type}]: {message}")
        
        # 调用所有注册的回调函数
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                logger.error(f"告警回调函数执行失败: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        获取监控摘要
        
        Returns:
            监控摘要信息
        """
        try:
            open_positions = self.position_manager.get_open_positions()
            open_orders = self.order_manager.get_open_orders()
            account_info = self.binance_api.get_account_info()
            
            total_unrealized_pnl = sum(
                p.unrealized_pnl or 0 for p in open_positions
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'positions': {
                    'count': len(open_positions),
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'by_symbol': {
                        p.symbol: {
                            'side': p.side.value,
                            'size': p.current_size,
                            'unrealized_pnl': p.unrealized_pnl
                        }
                        for p in open_positions
                    }
                },
                'orders': {
                    'open_count': len(open_orders),
                    'by_status': {}
                },
                'account': {
                    'total_balance': account_info.get('total_balance', 0),
                    'available_balance': account_info.get('free_balance', 0),
                    'used_balance': account_info.get('used_balance', 0)
                }
            }
        except Exception as e:
            logger.error(f"获取监控摘要失败: {e}")
            return {'error': str(e)}
