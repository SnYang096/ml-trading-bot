"""
订单管理系统集成示例
展示如何与现有WebSocket系统集成
"""
import os
import sys
import logging
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.order_management.storage import Storage
from src.order_management.binance_api import BinanceAPI
from src.order_management.position_manager import PositionManager
from src.order_management.order_manager import OrderManager
from src.order_management.risk_controller import RiskController
from src.order_management.monitoring import MonitoringService
from src.order_management.performance_metrics import PerformanceMetricsCalculator
from src.order_management.models import OrderSide, OrderType, PositionSide
from src.order_management import metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_api_keys(testnet: bool = False) -> tuple[str, str]:
    """
    加载API密钥
    
    Args:
        testnet: 是否使用测试网
    
    Returns:
        (api_key, api_secret)
    """
    if testnet:
        env_file = project_root / "config" / "local" / "binance_testnet.env"
    else:
        env_file = project_root / "config" / "local" / "binance_mainnet.env"
    
    if not env_file.exists():
        raise FileNotFoundError(f"API密钥文件不存在: {env_file}")
    
    api_key = None
    api_secret = None
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BINANCE_FUTURES_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
            elif line.startswith('BINANCE_FUTURES_API_SECRET='):
                api_secret = line.split('=', 1)[1].strip()
    
    if not api_key or not api_secret:
        raise ValueError("无法从环境文件加载API密钥")
    
    return api_key, api_secret


def setup_order_management_system(testnet: bool = False):
    """
    设置订单管理系统
    
    Args:
        testnet: 是否使用测试网
    
    Returns:
        系统组件字典
    """
    # 加载API密钥
    api_key, api_secret = load_api_keys(testnet)
    
    # 初始化存储
    db_path = project_root / "data" / "order_management.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = Storage(str(db_path))
    
    # 初始化Binance API
    binance_api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
    
    # 初始化管理器
    position_manager = PositionManager(storage, binance_api)
    order_manager = OrderManager(storage, binance_api)
    risk_controller = RiskController(storage, binance_api)
    
    # 初始化监控服务
    monitoring_service = MonitoringService(
        storage=storage,
        position_manager=position_manager,
        order_manager=order_manager,
        binance_api=binance_api,
        update_interval=5
    )
    
    # 初始化性能指标计算器
    performance_calculator = PerformanceMetricsCalculator(storage)
    
    return {
        'storage': storage,
        'binance_api': binance_api,
        'position_manager': position_manager,
        'order_manager': order_manager,
        'risk_controller': risk_controller,
        'monitoring_service': monitoring_service,
        'performance_calculator': performance_calculator
    }


def example_basic_usage():
    """基本使用示例"""
    logger.info("=== 基本使用示例 ===")
    
    # 设置系统
    system = setup_order_management_system(testnet=True)
    pm = system['position_manager']
    om = system['order_manager']
    rc = system['risk_controller']
    
    # 1. 创建仓位
    logger.info("1. 创建仓位")
    position = pm.create_position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=50000.0,
        size=0.1,
        stop_loss_price=49000.0,
        take_profit_price=52000.0
    )
    logger.info(f"创建仓位: {position.position_id}")
    
    # 2. 下单
    logger.info("2. 下单")
    order = Order(
        order_id="example_order_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    
    # 风险验证
    passed, error = rc.validate_order_before_submit(order, leverage=1)
    if passed:
        logger.info("订单通过风险验证")
        # 实际下单
        # placed_order = om.place_order(...)
    else:
        logger.warning(f"订单未通过风险验证: {error}")
    
    # 3. 查询仓位
    logger.info("3. 查询仓位")
    open_positions = pm.get_open_positions()
    logger.info(f"当前开仓数量: {len(open_positions)}")
    
    # 4. 查询订单
    logger.info("4. 查询订单")
    open_orders = om.get_open_orders()
    logger.info(f"当前未完成订单数量: {len(open_orders)}")


def example_with_monitoring():
    """带监控的使用示例"""
    logger.info("=== 带监控的使用示例 ===")
    
    # 设置系统
    system = setup_order_management_system(testnet=True)
    monitoring = system['monitoring_service']
    
    # 启动Prometheus metrics服务器
    metrics.start_metrics_server(port=8000)
    logger.info("Prometheus metrics服务器已启动在端口8000")
    
    # 启动监控服务
    monitoring.start()
    logger.info("监控服务已启动")
    
    # 注册告警回调
    def alert_callback(alert_type: str, message: str):
        logger.warning(f"告警 [{alert_type}]: {message}")
        # 这里可以发送邮件、Telegram通知等
    
    monitoring.register_alert_callback(alert_callback)
    
    try:
        # 运行一段时间
        import time
        time.sleep(30)
        
        # 获取监控摘要
        summary = monitoring.get_monitoring_summary()
        logger.info(f"监控摘要: {summary}")
    finally:
        monitoring.stop()
        logger.info("监控服务已停止")


def example_with_websocket_integration():
    """
    与WebSocket系统集成示例
    
    说明：这个示例展示如何将订单管理系统与现有的WebSocket数据流集成
    """
    logger.info("=== WebSocket集成示例 ===")
    
    # 设置系统
    system = setup_order_management_system(testnet=True)
    pm = system['position_manager']
    om = system['order_manager']
    
    # 模拟WebSocket消息处理
    async def handle_trade_tick(symbol: str, price: float, size: float):
        """
        处理交易tick数据
        
        在实际集成中，这个函数会被WebSocket客户端调用
        """
        # 1. 检查是否有开仓需要更新止损
        open_positions = pm.get_open_positions(symbol)
        for position in open_positions:
            # 检查止损触发
            if position.stop_loss_price:
                if position.side.value == 'long' and price <= position.stop_loss_price:
                    logger.warning(f"止损触发: {position.position_id}, 价格={price}")
                    # 这里可以触发平仓逻辑
                elif position.side.value == 'short' and price >= position.stop_loss_price:
                    logger.warning(f"止损触发: {position.position_id}, 价格={price}")
                    # 这里可以触发平仓逻辑
            
            # 检查止盈触发
            if position.take_profit_price:
                if position.side.value == 'long' and price >= position.take_profit_price:
                    logger.info(f"止盈触发: {position.position_id}, 价格={price}")
                    # 这里可以触发平仓逻辑
                elif position.side.value == 'short' and price <= position.take_profit_price:
                    logger.info(f"止盈触发: {position.position_id}, 价格={price}")
                    # 这里可以触发平仓逻辑
    
    async def handle_order_update(order_id: str, status: str):
        """
        处理订单更新
        
        在实际集成中，这个函数会被WebSocket用户数据流调用
        """
        # 同步订单状态
        order = om.get_order(order_id)
        if order:
            updated = om.sync_order_status(order_id)
            logger.info(f"订单状态更新: {order_id}, 状态={updated.status}")
    
    # 模拟WebSocket事件
    logger.info("模拟WebSocket事件处理...")
    # 在实际使用中，这些函数会被WebSocket客户端注册为回调


if __name__ == "__main__":
    # 运行示例
    try:
        example_basic_usage()
        print("\n" + "="*50 + "\n")
        example_with_monitoring()
        print("\n" + "="*50 + "\n")
        example_with_websocket_integration()
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
