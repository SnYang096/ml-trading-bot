"""
高级订单测试：限价单、部分成交、移动止损

测试场景：
1. 限价单开仓（挂单 → 等待成交）
2. 部分成交处理（查询填充数量）
3. 移动止损（取消旧止损 → 新止损）
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.order_management.binance_api import BinanceAPI
from src.order_management.order_manager import OrderManager
from src.order_management.storage import Storage
from src.order_management.models import OrderSide, OrderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_api_keys() -> tuple[str, str]:
    """加载 API 密钥"""
    env_file = project_root / "config" / "local" / "binance_mainnet.env"
    api_key = ""
    api_secret = ""
    
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    
                    if "API_KEY" in k.upper():
                        api_key = v
                    elif "SECRET" in k.upper():
                        api_secret = v
    
    if not api_key or not api_secret:
        raise ValueError("无法加载API密钥")
    
    return api_key, api_secret


def test_limit_order(binance_api: BinanceAPI, order_manager: OrderManager, symbol: str, ccxt_symbol: str, size: float):
    """
    测试 1: 限价单开仓
    
    策略：挂一个低于市价的限价买单，等待成交
    """
    logger.info("\n" + "=" * 80)
    logger.info("测试 1: 限价单开仓")
    logger.info("=" * 80)
    
    try:
        # 获取当前市价（使用 Binance 格式）
        current_price = binance_api.get_ticker_price(symbol)
        logger.info(f"当前市价: {current_price}")
        
        if current_price is None:
            logger.error("无法获取市价")
            return None
        
        # 挂限价买单（低于市价 0.5%）
        limit_price = round(current_price * 0.995, 4)
        logger.info(f"限价买单: {size} @ {limit_price}")
        
        order = order_manager.place_order(
            symbol=ccxt_symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=size,
            price=limit_price,
        )
        logger.info(f"✅ 限价单已挂: order_id={order.order_id}")
        
        # 等待 5 秒查询状态
        logger.info("⏳ 等待 5 秒...")
        time.sleep(5)
        
        # 同步订单状态
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"订单状态: {order.status.value}")
        logger.info(f"已成交数量: {order.filled_quantity}/{order.quantity}")
        logger.info(f"平均成交价: {order.average_price}")
        
        # 如果未完全成交，取消订单
        if order.status.value in ["open", "partially_filled"]:
            logger.info("取消限价单...")
            order_manager.cancel_order(order.order_id)
            logger.info("✅ 限价单已取消")
        
        return order.order_id
        
    except Exception as e:
        logger.error(f"❌ 限价单测试失败: {e}", exc_info=True)
        return None


def test_partial_fill(binance_api: BinanceAPI, order_manager: OrderManager, symbol: str, size: float):
    """
    测试 2: 部分成交处理
    
    策略：先用市价单开仓，然后分批平仓观察部分成交
    """
    logger.info("\n" + "=" * 80)
    logger.info("测试 2: 部分成交处理")
    logger.info("=" * 80)
    
    try:
        # 1. 市价开多仓
        logger.info(f"市价买入 {size}")
        order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=size,
        )
        time.sleep(2)
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"✅ 开仓成功: {order.filled_quantity} @ {order.average_price}")
        
        # 2. 分批平仓（卖出一半）
        half_size = size / 2
        logger.info(f"\n分批平仓: 卖出 {half_size}")
        close_order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=half_size,
            reduce_only=True,
        )
        time.sleep(2)
        close_order = order_manager.sync_order_status(close_order.order_id)
        logger.info(f"✅ 部分平仓: {close_order.filled_quantity} @ {close_order.average_price}")
        
        # 3. 查询剩余持仓
        positions = binance_api.get_positions(symbol)
        logger.info(f"\n剩余持仓:")
        for pos in positions:
            if pos.get('side') == 'long':
                logger.info(f"  多单: {pos.get('size', 0)}")
        
        # 4. 清空剩余持仓
        logger.info(f"\n清空剩余持仓: 卖出 {half_size}")
        final_order = order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=half_size,
            reduce_only=True,
        )
        time.sleep(2)
        logger.info("✅ 持仓已清空")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 部分成交测试失败: {e}", exc_info=True)
        return False


def test_move_stop_loss(binance_api: BinanceAPI, order_manager: OrderManager, symbol: str, ccxt_symbol: str, size: float):
    """
    测试 3: 移动止损
    
    策略：开仓 → 设置止损 → 移动止损（取消旧 + 新止损）
    """
    logger.info("\n" + "=" * 80)
    logger.info("测试 3: 移动止损")
    logger.info("=" * 80)
    
    try:
        # 1. 市价开多仓
        logger.info(f"市价买入 {size}")
        order = order_manager.place_order(
            symbol=ccxt_symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=size,
        )
        time.sleep(2)
        order = order_manager.sync_order_status(order.order_id)
        logger.info(f"✅ 开仓成功: {order.filled_quantity} @ {order.average_price}")
        
        # 2. 设置初始止损（-2%）
        current_price = binance_api.get_ticker_price(symbol)
        if current_price is None:
            logger.error("无法获取市价")
            return False
            
        stop_loss_1 = round(current_price * 0.98, 4)
        logger.info(f"\n设置初始止损: {stop_loss_1}")
        
        sl_order = order_manager.place_order(
            symbol=ccxt_symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=size,
            stop_price=stop_loss_1,
            reduce_only=True,
        )
        logger.info(f"✅ 止损单已下: {sl_order.order_id}")
        
        time.sleep(2)
        
        # 3. 查询挂单
        open_orders = order_manager.get_open_orders(ccxt_symbol)
        logger.info(f"\n当前挂单数: {len(open_orders)}")
        for o in open_orders:
            logger.info(f"  - {o.order_type.value}: stop_price={o.stop_price}")
        
        # 4. 移动止损（取消旧止损，设置新止损 -1%）
        logger.info(f"\n移动止损...")
        
        # 先查询 Binance 服务器上的真实挂单
        binance_open_orders = binance_api.exchange.fetch_open_orders(ccxt_symbol)
        logger.info(f"Binance 挂单数: {len(binance_open_orders)}")
        
        if binance_open_orders:
            # 取消第一个止损单
            binance_order_id = str(binance_open_orders[0]['id'])
            logger.info(f"取消止损单: {binance_order_id}")
            binance_api.cancel_order(binance_order_id, ccxt_symbol)
            logger.info("✅ 旧止损单已取消")
            time.sleep(1)
        
        # 设置新止损（-1%）
        stop_loss_2 = round(current_price * 0.99, 4)
        logger.info(f"设置新止损: {stop_loss_2}")
        
        sl_order_2 = order_manager.place_order(
            symbol=ccxt_symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=size,
            stop_price=stop_loss_2,
            reduce_only=True,
        )
        logger.info(f"✅ 新止损单已下: {sl_order_2.order_id}")
        
        time.sleep(2)
        
        # 5. 验证新止损
        open_orders = order_manager.get_open_orders(ccxt_symbol)
        logger.info(f"\n移动后挂单数: {len(open_orders)}")
        for o in open_orders:
            logger.info(f"  - {o.order_type.value}: stop_price={o.stop_price}")
        
        # 6. 清空持仓
        logger.info(f"\n清空持仓...")
        # 先取消所有挂单
        for o in open_orders:
            try:
                # 使用 Binance order ID
                if o.binance_order_id:
                    binance_api.cancel_order(str(o.binance_order_id), ccxt_symbol)
            except Exception as e:
                logger.warning(f"取消挂单失败: {e}")
        
        time.sleep(1)
        
        # 市价平仓
        order_manager.place_order(
            symbol=ccxt_symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=size,
            reduce_only=True,
        )
        logger.info("✅ 持仓已清空")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 移动止损测试失败: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("🚀 高级订单测试")
    logger.info("=" * 80)
    
    # 加载 API 密钥
    api_key, api_secret = load_api_keys()
    logger.info("✅ API 密钥已加载")
    
    # 初始化
    db_path = project_root / "data" / "test_advanced_orders.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = Storage(str(db_path))
    
    binance_api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
    )
    logger.info("✅ Binance API 已初始化")
    
    order_manager = OrderManager(storage, binance_api)
    logger.info("✅ OrderManager 已初始化")
    
    # 测试参数
    symbol = "XRPUSDT"  # Binance 格式
    ccxt_symbol = "XRP/USDT:USDT"  # ccxt 格式
    size = 50.0
    
    # 执行测试
    test_limit_order(binance_api, order_manager, symbol, ccxt_symbol, size)
    test_partial_fill(binance_api, order_manager, ccxt_symbol, size)
    test_move_stop_loss(binance_api, order_manager, symbol, ccxt_symbol, size)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 所有测试完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
