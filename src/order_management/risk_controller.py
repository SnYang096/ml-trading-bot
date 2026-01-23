"""
风险控制器
专注于Binance API层面的风险控制，作为PCM的补充
"""
import logging
from typing import Optional, Dict, Any, Tuple
from decimal import Decimal, ROUND_DOWN

from .models import Order, OrderSide, OrderType
from .storage import Storage
from .binance_api import BinanceAPI

logger = logging.getLogger(__name__)


class RiskController:
    """
    风险控制器
    
    注意：现有系统已有PCM (Portfolio Capital Management)实现。
    本RiskController专注于Binance API层面的风险控制，而不是重复PCM的功能。
    """
    
    def __init__(self, storage: Storage, binance_api: BinanceAPI):
        """
        初始化风险控制器
        
        Args:
            storage: 存储层实例
            binance_api: Binance API实例
        """
        self.storage = storage
        self.binance_api = binance_api
    
    def check_binance_margin_requirement(
        self,
        symbol: str,
        quantity: float,
        price: float,
        leverage: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        检查Binance保证金要求
        
        Args:
            symbol: 交易对符号
            quantity: 订单数量
            price: 订单价格
            leverage: 杠杆倍数
        
        Returns:
            (是否通过, 错误信息)
        """
        try:
            # 获取账户余额
            account_info = self.binance_api.get_account_info()
            available_margin = account_info.get('free_balance', 0)
            
            # 计算所需保证金
            required_margin = (quantity * price) / leverage
            
            if available_margin < required_margin:
                error_msg = (
                    f"保证金不足: 可用={available_margin:.2f} USDT, "
                    f"需要={required_margin:.2f} USDT"
                )
                logger.warning(error_msg)
                return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查保证金要求失败: {e}")
            return False, str(e)
    
    def check_binance_position_limits(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide
    ) -> Tuple[bool, Optional[str]]:
        """
        检查Binance单品种仓位限制
        
        Args:
            symbol: 交易对符号
            quantity: 订单数量
            side: 订单方向
        
        Returns:
            (是否通过, 错误信息)
        """
        try:
            # 获取当前仓位
            current_position = self.binance_api.get_position(symbol)
            
            # 获取交易对信息
            symbol_info = self.binance_api.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"无法获取交易对信息: {symbol}"
            
            # 检查最大仓位限制（Binance通常有单品种最大仓位限制）
            max_position_size = symbol_info.get('limits', {}).get('amount', {}).get('max')
            if max_position_size:
                if current_position:
                    current_size = abs(current_position.get('size', 0))
                    if side == OrderSide.BUY:
                        new_size = current_size + quantity
                    else:  # SELL
                        new_size = current_size - quantity
                    
                    if abs(new_size) > max_position_size:
                        error_msg = (
                            f"超过单品种最大仓位限制: "
                            f"当前={current_size}, 新增={quantity}, "
                            f"最大={max_position_size}"
                        )
                        logger.warning(error_msg)
                        return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查仓位限制失败: {e}")
            return False, str(e)
    
    def check_order_size_limits(
        self,
        symbol: str,
        quantity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        检查订单大小限制（最小/最大订单量）
        
        Args:
            symbol: 交易对符号
            quantity: 订单数量
        
        Returns:
            (是否通过, 错误信息)
        """
        try:
            symbol_info = self.binance_api.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"无法获取交易对信息: {symbol}"
            
            limits = symbol_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            
            min_amount = amount_limits.get('min')
            max_amount = amount_limits.get('max')
            
            if min_amount and quantity < min_amount:
                error_msg = f"订单数量小于最小值: {quantity} < {min_amount}"
                logger.warning(error_msg)
                return False, error_msg
            
            if max_amount and quantity > max_amount:
                error_msg = f"订单数量大于最大值: {quantity} > {max_amount}"
                logger.warning(error_msg)
                return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查订单大小限制失败: {e}")
            return False, str(e)
    
    def check_leverage_limits(
        self,
        symbol: str,
        leverage: int
    ) -> Tuple[bool, Optional[str]]:
        """
        检查杠杆倍数限制
        
        Args:
            symbol: 交易对符号
            leverage: 杠杆倍数
        
        Returns:
            (是否通过, 错误信息)
        """
        try:
            # Binance合约通常支持1-125倍杠杆
            min_leverage = 1
            max_leverage = 125
            
            if leverage < min_leverage or leverage > max_leverage:
                error_msg = f"杠杆倍数超出范围: {leverage}, 允许范围: {min_leverage}-{max_leverage}"
                logger.warning(error_msg)
                return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查杠杆限制失败: {e}")
            return False, str(e)
    
    def check_order_price_precision(
        self,
        symbol: str,
        price: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        检查订单价格精度
        
        Args:
            symbol: 交易对符号
            price: 订单价格（限价单需要）
        
        Returns:
            (是否通过, 错误信息)
        """
        if price is None:
            return True, None
        
        try:
            symbol_info = self.binance_api.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"无法获取交易对信息: {symbol}"
            
            precision = symbol_info.get('precision', {}).get('price')
            if precision is None:
                return True, None
            
            # 检查价格精度
            price_str = f"{price:.{precision}f}"
            price_rounded = float(price_str)
            
            if abs(price - price_rounded) > 1e-10:
                error_msg = (
                    f"订单价格精度不符合要求: {price}, "
                    f"精度={precision}位小数"
                )
                logger.warning(error_msg)
                return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查价格精度失败: {e}")
            return False, str(e)
    
    def check_order_quantity_precision(
        self,
        symbol: str,
        quantity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        检查订单数量精度
        
        Args:
            symbol: 交易对符号
            quantity: 订单数量
        
        Returns:
            (是否通过, 错误信息)
        """
        try:
            symbol_info = self.binance_api.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"无法获取交易对信息: {symbol}"
            
            precision = symbol_info.get('precision', {}).get('amount')
            if precision is None:
                return True, None
            
            # 检查数量精度
            quantity_str = f"{quantity:.{precision}f}"
            quantity_rounded = float(quantity_str)
            
            if abs(quantity - quantity_rounded) > 1e-10:
                error_msg = (
                    f"订单数量精度不符合要求: {quantity}, "
                    f"精度={precision}位小数"
                )
                logger.warning(error_msg)
                return False, error_msg
            
            return True, None
        except Exception as e:
            logger.error(f"检查数量精度失败: {e}")
            return False, str(e)
    
    def validate_order_before_submit(
        self,
        order: Order,
        leverage: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        下单前的综合验证（复用PCM结果+Binance限制）
        
        注意：此方法假设PCM已经验证了仓位大小和预算分配。
        这里只进行Binance API层面的验证。
        
        Args:
            order: 订单对象
            leverage: 杠杆倍数
        
        Returns:
            (是否通过, 错误信息)
        """
        # 1. 检查订单大小限制
        passed, error = self.check_order_size_limits(order.symbol, order.quantity)
        if not passed:
            return False, error
        
        # 2. 检查订单数量精度
        passed, error = self.check_order_quantity_precision(order.symbol, order.quantity)
        if not passed:
            return False, error
        
        # 3. 检查订单价格精度（限价单需要）
        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                return False, "限价单必须指定价格"
            passed, error = self.check_order_price_precision(order.symbol, order.price)
            if not passed:
                return False, error
        
        # 4. 检查杠杆倍数限制
        passed, error = self.check_leverage_limits(order.symbol, leverage)
        if not passed:
            return False, error
        
        # 5. 检查保证金要求
        price = order.price or 0  # 市价单使用当前价格估算
        if price == 0:
            # 对于市价单，需要获取当前价格
            symbol_info = self.binance_api.get_symbol_info(order.symbol)
            if symbol_info:
                # 使用ticker价格作为估算
                try:
                    ticker = self.binance_api.exchange.fetch_ticker(order.symbol)
                    price = ticker.get('last', 0)
                except:
                    pass
        
        if price > 0:
            passed, error = self.check_binance_margin_requirement(
                order.symbol,
                order.quantity,
                price,
                leverage
            )
            if not passed:
                return False, error
        
        # 6. 检查仓位限制
        passed, error = self.check_binance_position_limits(
            order.symbol,
            order.quantity,
            order.side
        )
        if not passed:
            return False, error
        
        return True, None
