"""
订单管理系统测试Demo

测试场景：
1. 开多单 + 止损/止盈
2. 加仓（独立多单 + 独立止损/止盈）
3. 移动止损
4. 反向开仓（平多 → 开空）
"""

import os
import sys
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.order_management.storage import Storage
from src.order_management.binance_api import BinanceAPI
from src.order_management.position_manager import PositionManager
from src.order_management.order_manager import OrderManager
from src.order_management.binance_user_stream import BinanceUserStream
from src.order_management.database_backup import DatabaseBackup
from src.order_management.models import OrderSide, OrderType, PositionSide

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        # 也支持从环境变量读取
        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
    else:
        env_file = project_root / "config" / "local" / "binance_mainnet.env"
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv(
            "BINANCE_FUTURES_API_KEY", ""
        )
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
            "BINANCE_FUTURES_API_SECRET", ""
        )

    # 如果环境变量没有，从文件加载
    if not api_key or not api_secret:
        if not env_file.exists():
            raise FileNotFoundError(f"API密钥文件不存在: {env_file}")

        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        if "API_KEY" in key.upper() and not api_key:
                            api_key = value
                        elif (
                            "API_SECRET" in key.upper() or "SECRET" in key.upper()
                        ) and not api_secret:
                            api_secret = value

    if not api_key or not api_secret:
        raise ValueError("无法加载API密钥，请检查配置文件或环境变量")

    return api_key, api_secret


def setup_system(testnet: bool = True) -> dict:
    """
    设置订单管理系统

    Args:
        testnet: 是否使用测试网

    Returns:
        系统组件字典
    """
    logger.info("=" * 80)
    logger.info("初始化订单管理系统")
    logger.info("=" * 80)

    # 加载API密钥
    api_key, api_secret = load_api_keys(testnet)
    logger.info(f"✅ API密钥已加载（测试网: {testnet}）")

    # 初始化存储
    db_path = project_root / "data" / "order_management_demo.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = Storage(str(db_path))
    logger.info(f"✅ 数据库已初始化: {db_path}")

    # 初始化Binance API（支持代理配置）
    # 主网时，如果设置了USE_SOCKS5_PROXY环境变量，会自动使用代理
    binance_api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        use_proxy=None,  # None表示从环境变量读取
    )
    logger.info("✅ Binance API已初始化")
    if binance_api.use_proxy:
        logger.info(
            f"   代理: {binance_api.proxy_type}://{binance_api.proxy_host}:{binance_api.proxy_port}"
        )

    # 初始化管理器
    position_manager = PositionManager(storage, binance_api)
    order_manager = OrderManager(storage, binance_api)
    logger.info("✅ 管理器和订单管理器已初始化")

    # 初始化 User Data Stream
    def on_execution_report(report: Dict[str, Any]) -> None:
        """处理订单执行回报"""
        try:
            order_manager.handle_execution_report(report)
            logger.info(
                f"📨 收到订单回报: order_id={report.get('order_id')}, "
                f"status={report.get('status')}, symbol={report.get('symbol')}"
            )
        except Exception as e:
            logger.error(f"处理订单回报失败: {e}", exc_info=True)

    user_stream = BinanceUserStream(
        binance_api=binance_api,
        on_execution_report=on_execution_report,
        keepalive_interval=30 * 60,  # 30分钟续期一次
    )
    logger.info("✅ User Data Stream已初始化")

    # 初始化数据库备份
    backup_manager = DatabaseBackup(
        db_path=str(db_path),
        retention_days=30,
    )
    logger.info("✅ 数据库备份管理器已初始化")

    return {
        "storage": storage,
        "binance_api": binance_api,
        "position_manager": position_manager,
        "order_manager": order_manager,
        "user_stream": user_stream,
        "backup_manager": backup_manager,
    }


def get_current_price(binance_api: BinanceAPI, symbol: str) -> float:
    """
    获取当前市场价格

    Args:
        binance_api: Binance API实例
        symbol: 交易对符号

    Returns:
        当前价格
    """
    price = binance_api.get_ticker_price(symbol)
    if price is None:
        raise ValueError(f"无法获取 {symbol} 的当前价格")
    return price


def open_short_position(
    system: dict, symbol: str, size: float
) -> tuple[Optional[str], Optional[str]]:
    """
    开空单（SHORT）

    Args:
        system: 系统组件字典
        symbol: 交易对符号
        size: 仓位大小

    Returns:
        (position_id, order_id) 如果成功，否则 (None, None)
    """
    logger.info("=" * 80)
    logger.info(f"开空单（SHORT）: {symbol}, size={size}")
    logger.info("=" * 80)

    binance_api = system["binance_api"]
    order_manager = system["order_manager"]
    position_manager = system["position_manager"]

    try:
        # 1. 获取当前价格
        current_price = get_current_price(binance_api, symbol)
        logger.info(f"📊 当前价格: {current_price}")

        # 2. 下市价卖单（SELL, MARKET）
        logger.info(f"📤 下市价卖单: {symbol}, size={size}")
        order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=size,
        )
        logger.info(
            f"✅ 订单已提交: order_id={order.order_id}, binance_order_id={order.binance_order_id}"
        )

        # 等待订单成交
        logger.info("⏳ 等待订单成交...")
        time.sleep(2)  # 等待订单成交

        # 同步订单状态
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"📋 订单状态: {order.status.value}")

        if order.status.value != "filled":
            logger.warning(f"⚠️ 订单未完全成交: status={order.status.value}")
            # 继续执行，使用已成交数量

        # 3. 创建SHORT仓位记录
        filled_size = order.filled_quantity or size
        entry_price = order.average_price or current_price

        logger.info(
            f"📝 创建SHORT仓位记录: size={filled_size}, entry_price={entry_price}"
        )
        position = position_manager.create_position(
            symbol=symbol,
            side=PositionSide.SHORT,
            entry_price=entry_price,
            size=filled_size,
            strategy_id="demo_strategy",
            notes="基线demo - 开空单",
        )
        logger.info(f"✅ SHORT仓位已创建: position_id={position.position_id}")

        return position.position_id, order.order_id

    except Exception as e:
        logger.error(f"❌ 开空单失败: {e}", exc_info=True)
        return None, None


def open_long_position(
    system: dict, symbol: str, size: float
) -> tuple[Optional[str], Optional[str]]:
    """
    开多单（LONG）

    Args:
        system: 系统组件字典
        symbol: 交易对符号
        size: 仓位大小

    Returns:
        (position_id, order_id) 如果成功，否则 (None, None)
    """
    logger.info("=" * 80)
    logger.info(f"开多单（LONG）: {symbol}, size={size}")
    logger.info("=" * 80)

    binance_api = system["binance_api"]
    order_manager = system["order_manager"]
    position_manager = system["position_manager"]

    try:
        # 1. 获取当前价格
        current_price = get_current_price(binance_api, symbol)
        logger.info(f"📊 当前价格: {current_price}")

        # 2. 下市价买单（BUY, MARKET）
        logger.info(f"📤 下市价买单: {symbol}, size={size}")
        order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=size,
        )
        logger.info(
            f"✅ 订单已提交: order_id={order.order_id}, binance_order_id={order.binance_order_id}"
        )

        # 等待订单成交
        logger.info("⏳ 等待订单成交...")
        time.sleep(2)  # 等待订单成交

        # 同步订单状态
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"📋 订单状态: {order.status.value}")

        if order.status.value != "filled":
            logger.warning(f"⚠️ 订单未完全成交: status={order.status.value}")
            # 继续执行，使用已成交数量

        # 3. 创建LONG仓位记录
        filled_size = order.filled_quantity or size
        entry_price = order.average_price or current_price

        logger.info(
            f"📝 创建LONG仓位记录: size={filled_size}, entry_price={entry_price}"
        )
        position = position_manager.create_position(
            symbol=symbol,
            side=PositionSide.LONG,
            entry_price=entry_price,
            size=filled_size,
            strategy_id="demo_strategy",
            notes="基线demo - 开多单",
        )
        logger.info(f"✅ LONG仓位已创建: position_id={position.position_id}")

        return position.position_id, order.order_id

    except Exception as e:
        logger.error(f"❌ 开多单失败: {e}", exc_info=True)
        return None, None


def close_position(system: dict, position_id: str, symbol: str) -> bool:
    """
    关闭仓位

    Args:
        system: 系统组件字典
        position_id: 仓位ID
        symbol: 交易对符号

    Returns:
        是否成功
    """
    logger.info("=" * 80)
    logger.info(f"关闭仓位: position_id={position_id}, symbol={symbol}")
    logger.info("=" * 80)

    binance_api = system["binance_api"]
    order_manager = system["order_manager"]
    position_manager = system["position_manager"]

    try:
        # 1. 获取当前价格
        current_price = get_current_price(binance_api, symbol)
        logger.info(f"📊 当前价格: {current_price}")

        # 2. 获取仓位信息
        position = position_manager.get_position(position_id)
        if not position:
            logger.error(f"❌ 仓位不存在: {position_id}")
            return False

        logger.info(
            f"📋 仓位信息: side={position.side.value}, size={position.current_size}, entry_price={position.entry_price}"
        )

        # 3. 确定平仓方向
        if position.side == PositionSide.LONG:
            # 平多单：下卖单
            close_side = OrderSide.SELL
        else:  # SHORT
            # 平空单：下买单
            close_side = OrderSide.BUY

        # 4. 下市价平仓单
        logger.info(
            f"📤 下市价平仓单: {symbol}, side={close_side.value}, size={position.current_size}"
        )
        order = order_manager.place_order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.current_size,
            position_id=position_id,
            reduce_only=True,  # 只减仓
        )
        logger.info(f"✅ 平仓订单已提交: order_id={order.order_id}")

        # 等待订单成交
        logger.info("⏳ 等待平仓订单成交...")
        time.sleep(2)

        # 同步订单状态
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"📋 平仓订单状态: {order.status.value}")

        # 5. 关闭仓位记录
        exit_price = order.average_price or current_price
        logger.info(f"📝 关闭仓位记录: exit_price={exit_price}")
        closed_position = position_manager.close_position(
            position_id=position_id,
            price=exit_price,
            order_id=order.order_id,
            reason="基线demo - 平仓",
        )

        # 计算盈亏
        pnl = closed_position.realized_pnl or 0.0
        logger.info(f"✅ 仓位已关闭: position_id={position_id}")
        logger.info(f"💰 实现盈亏: {pnl:.4f} USDT")

        return True

    except Exception as e:
        logger.error(f"❌ 关闭仓位失败: {e}", exc_info=True)
        return False


def print_summary(system: dict):
    """
    打印统计信息

    Args:
        system: 系统组件字典
    """
    logger.info("=" * 80)
    logger.info("📊 统计信息")
    logger.info("=" * 80)

    position_manager = system["position_manager"]
    order_manager = system["order_manager"]

    # 获取开仓仓位
    open_positions = position_manager.get_open_positions()

    # 获取所有订单和仓位（通过查询数据库）
    storage = system["storage"]
    import sqlite3

    # storage.db_path是数据库路径
    conn = sqlite3.connect(storage.db_path)
    try:
        cursor = conn.cursor()
        # 获取所有订单
        cursor.execute("SELECT * FROM orders")
        all_order_rows = cursor.fetchall()

        # 获取所有仓位
        cursor.execute("SELECT * FROM positions")
        all_position_rows = cursor.fetchall()

        logger.info(f"📈 总仓位数: {len(all_position_rows)}")

        open_count = sum(
            1 for row in all_position_rows if row[12] == "open"
        )  # status字段
        closed_count = sum(1 for row in all_position_rows if row[12] == "closed")

        logger.info(f"   开仓: {open_count}")
        logger.info(f"   已平仓: {closed_count}")

        # 计算总盈亏（从已平仓仓位）
        total_realized_pnl = 0.0
        total_unrealized_pnl = 0.0

        for row in all_position_rows:
            status = row[12]  # status
            realized_pnl = row[11] or 0.0  # realized_pnl
            unrealized_pnl = row[10] or 0.0  # unrealized_pnl

            if status == "closed":
                total_realized_pnl += realized_pnl
            elif status == "open":
                total_unrealized_pnl += unrealized_pnl

        logger.info(f"💰 总实现盈亏: {total_realized_pnl:.4f} USDT")
        logger.info(f"💰 总未实现盈亏: {total_unrealized_pnl:.4f} USDT")
        logger.info(f"💰 总盈亏: {total_realized_pnl + total_unrealized_pnl:.4f} USDT")

        # 统计订单
        logger.info(f"📋 总订单数: {len(all_order_rows)}")

        filled_count = sum(
            1 for row in all_order_rows if row[9] == "filled"
        )  # status字段
        logger.info(f"   已成交: {filled_count}")

    finally:
        conn.close()

    logger.info("=" * 80)


def place_stop_loss(system: dict, symbol: str, position_size: float, stop_price: float) -> Optional[str]:
    """
    下止损单（STOP_MARKET, reduceOnly=true）
    
    Args:
        system: 系统组件
        symbol: 交易对
        position_size: 持仓数量
        stop_price: 止损价格
    
    Returns:
        订单ID
    """
    order_manager = system["order_manager"]
    
    try:
        # 多单止损 = 下跌时触发卖出
        order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=position_size,
            stop_price=stop_price,
            reduce_only=True,
        )
        logger.info(f"✅ 止损单已下: order_id={order.order_id}, stop_price={stop_price}")
        return order.order_id
    except Exception as e:
        logger.error(f"❌ 下止损单失败: {e}", exc_info=True)
        return None


def place_take_profit(system: dict, symbol: str, position_size: float, take_profit_price: float) -> Optional[str]:
    """
    下止盈单（TAKE_PROFIT_MARKET, reduceOnly=true）
    
    Args:
        system: 系统组件
        symbol: 交易对
        position_size: 持仓数量
        take_profit_price: 止盈价格
    
    Returns:
        订单ID
    """
    order_manager = system["order_manager"]
    
    try:
        # 多单止盈 = 上涨时触发卖出
        order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=position_size,
            stop_price=take_profit_price,
            reduce_only=True,
        )
        logger.info(f"✅ 止盈单已下: order_id={order.order_id}, take_profit_price={take_profit_price}")
        return order.order_id
    except Exception as e:
        logger.error(f"❌ 下止盈单失败: {e}", exc_info=True)
        return None


def get_binance_positions(binance_api: BinanceAPI, symbol: str) -> dict:
    """
    查询 Binance 真实持仓
    
    Returns:
        {'long': float, 'short': float}
    """
    positions = binance_api.get_positions(symbol)
    result = {'long': 0.0, 'short': 0.0}
    
    for pos in positions:
        side = pos.get('side', '').lower()
        amount = abs(pos.get('contracts', 0.0) or pos.get('amount', 0.0))
        
        if side == 'long':
            result['long'] = amount
        elif side == 'short':
            result['short'] = amount
    
    return result


async def run_demo_async(symbol: str = "XRPUSDT", size: float = 50.0, testnet: bool = False):
    """
    测试订单管理系统的完整流程

    Args:
        symbol: 交易对符号（默认: XRPUSDT）
        size: 仓位大小（默认: 50）
        testnet: 是否使用测试网（默认: False，使用主网）
    """
    logger.info("=" * 80)
    logger.info("🚀 启动订单管理测试Demo")
    logger.info("=" * 80)
    logger.info(f"交易对: {symbol}")
    logger.info(f"仓位大小: {size}")
    logger.info(f"使用测试网: {testnet}")
    if not testnet:
        logger.warning("⚠️  警告: 使用主网，将使用真实资金！")
    logger.info("=" * 80)

    system = None
    user_stream = None
    backup_manager = None

    try:
        # 1. 初始化系统
        system = setup_system(testnet=testnet)
        user_stream = system["user_stream"]
        backup_manager = system.get("backup_manager")
        binance_api = system["binance_api"]
        order_manager = system["order_manager"]

        # 2. 启动 User Data Stream
        await user_stream.start()
        logger.info("✅ User Data Stream已启动")

        # 3. 启动数据库备份
        if backup_manager:
            await backup_manager.start()
            logger.info("✅ 数据库备份任务已启动")

        # ===== 场景 1: 开多单 + 止损/止盈 =====
        logger.info("\n" + "=" * 80)
        logger.info("场景 1: 开多单 + 止损/止盈")
        logger.info("=" * 80)
        
        long_position_id, _ = open_long_position(system, symbol, size)
        if not long_position_id:
            logger.error("❌ 开多单失败，终止测试")
            return
        
        time.sleep(2)
        
        # 获取当前价格
        current_price = get_current_price(binance_api, symbol)
        stop_loss_price = current_price * 0.98  # 止损 -2%
        take_profit_price = current_price * 1.02  # 止盈 +2%
        
        logger.info(f"当前价格: {current_price}")
        logger.info(f"止损价格: {stop_loss_price}")
        logger.info(f"止盈价格: {take_profit_price}")
        
        sl_order_id = place_stop_loss(system, symbol, size, stop_loss_price)
        tp_order_id = place_take_profit(system, symbol, size, take_profit_price)
        
        time.sleep(2)
        
        # 验证挂单状态
        open_orders = order_manager.get_open_orders(symbol)
        logger.info(f"\n📋 当前挂单数: {len(open_orders)}")
        for order in open_orders:
            logger.info(f"  - {order.order_type.value}: {order.side.value}, stop_price={order.stop_price}")
        
        # ===== 场景 2: 加仓 =====
        logger.info("\n" + "=" * 80)
        logger.info("场景 2: 加仓（独立多单 + 独立止损/止盈）")
        logger.info("=" * 80)
        
        add_size = size * 0.6  # 加仓 60%
        long_position_id_2, _ = open_long_position(system, symbol, add_size)
        if not long_position_id_2:
            logger.error("❌ 加仓失败")
        else:
            time.sleep(2)
            current_price = get_current_price(binance_api, symbol)
            sl_order_id_2 = place_stop_loss(system, symbol, add_size, current_price * 0.98)
            tp_order_id_2 = place_take_profit(system, symbol, add_size, current_price * 1.02)
        
        time.sleep(2)
        
        # 查询 Binance 真实持仓
        positions = get_binance_positions(binance_api, symbol)
        logger.info(f"\n📊 Binance 真实持仓:")
        logger.info(f"  多单: {positions['long']}")
        logger.info(f"  空单: {positions['short']}")
        
        # ===== 场景 3: 移动止损 =====
        logger.info("\n" + "=" * 80)
        logger.info("场景 3: 移动止损")
        logger.info("=" * 80)
        
        if sl_order_id:
            try:
                # 取消旧止损单
                order_manager.cancel_order(sl_order_id)
                logger.info(f"✅ 已取消旧止损单: {sl_order_id}")
                time.sleep(1)
                
                # 下新止损单（价格上移）
                current_price = get_current_price(binance_api, symbol)
                new_stop_loss = current_price * 0.99  # 新止损 -1%
                logger.info(f"新止损价格: {new_stop_loss}")
                sl_order_id = place_stop_loss(system, symbol, size, new_stop_loss)
            except Exception as e:
                logger.error(f"❌ 移动止损失败: {e}", exc_info=True)
        
        time.sleep(2)
        
        # ===== 场景 4: 反向开仓（平多 → 开空）=====
        logger.info("\n" + "=" * 80)
        logger.info("场景 4: 反向开仓（平多 → 开空）")
        logger.info("=" * 80)
        
        # 先取消所有挂单
        open_orders = order_manager.get_open_orders(symbol)
        logger.info(f"取消 {len(open_orders)} 个挂单...")
        for order in open_orders:
            try:
                order_manager.cancel_order(order.order_id)
                logger.info(f"  ✅ 已取消: {order.order_id}")
            except Exception as e:
                logger.warning(f"  ⚠️ 取消失败: {e}")
        
        time.sleep(2)
        
        # 查询真实持仓
        positions = get_binance_positions(binance_api, symbol)
        total_long = positions['long']
        
        if total_long > 0:
            logger.info(f"平掉所有多单: {total_long}")
            
            # 市价平多（reduceOnly=true）
            order = order_manager.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=total_long,
                reduce_only=True,
            )
            logger.info(f"✅ 平多单已提交: {order.order_id}")
            
            time.sleep(3)
            
            # 验证持仓已平
            positions = get_binance_positions(binance_api, symbol)
            logger.info(f"平仓后持仓: 多={positions['long']}, 空={positions['short']}")
        
        # 开空单
        logger.info(f"\n开空单: {size}")
        short_position_id, _ = open_short_position(system, symbol, size)
        if short_position_id:
            time.sleep(2)
            current_price = get_current_price(binance_api, symbol)
            
            # 空单止损 = 上涨时触发买入
            sl_short = order_manager.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.STOP_MARKET,
                quantity=size,
                stop_price=current_price * 1.02,
                reduce_only=True,
            )
            logger.info(f"✅ 空单止损已下: {sl_short.order_id}")
        
        time.sleep(2)
        
        # 最终清理：平掉所有持仓
        logger.info("\n" + "=" * 80)
        logger.info("最终清理：平掉所有持仓")
        logger.info("=" * 80)
        
        positions = get_binance_positions(binance_api, symbol)
        if positions['short'] > 0:
            order = order_manager.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=positions['short'],
                reduce_only=True,
            )
            logger.info(f"✅ 平空单已提交: {order.order_id}")
        
        time.sleep(2)
        
        # 7. 打印统计信息
        print_summary(system)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}", exc_info=True)
    finally:
        # 停止数据库备份
        if backup_manager:
            await backup_manager.stop()
        
        # 停止 User Data Stream
        if user_stream:
            await user_stream.stop()
            logger.info("✅ User Data Stream已停止")


def run_demo(symbol: str = "XRPUSDT", size: float = 50.0, testnet: bool = False):
    """
    运行测试demo（同步包装器）

    Args:
        symbol: 交易对符号（默认: XRPUSDT）
        size: 仓位大小（默认: 50）
        testnet: 是否使用测试网（默认: False，使用主网）
    """
    asyncio.run(run_demo_async(symbol, size, testnet))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="订单管理测试Demo")
    parser.add_argument(
        "--symbol", type=str, default="XRPUSDT", help="交易对符号（默认: XRPUSDT）"
    )
    parser.add_argument(
        "--size", type=float, default=50.0, help="仓位大小（默认: 50）"
    )
    parser.add_argument(
        "--testnet", action="store_true", help="使用测试网（默认: False，使用主网）"
    )

    args = parser.parse_args()

    run_demo(symbol=args.symbol, size=args.size, testnet=args.testnet)
