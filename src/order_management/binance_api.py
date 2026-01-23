"""
Binance API封装
使用ccxt实现REST API调用
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import ccxt

from .models import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class BinanceAPI:
    """Binance API封装（使用ccxt）"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        sandbox: bool = False
    ):
        """
        初始化Binance API客户端
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网
            sandbox: 是否使用沙箱环境（与testnet相同）
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet or sandbox
        
        # 创建ccxt交易所实例
        exchange_options = {
            'defaultType': 'future',  # 使用合约交易
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        }
        
        if self.testnet:
            # 测试网配置
            exchange_options['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com',
                    'private': 'https://testnet.binancefuture.com',
                }
            }
        
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            **exchange_options
        })
    
    # ========== 账户信息 ==========
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        获取账户余额
        
        Returns:
            账户余额信息
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息
        """
        try:
            # ccxt的fetch_balance已经包含账户信息
            balance = self.get_account_balance()
            return {
                'total_balance': balance.get('USDT', {}).get('total', 0),
                'free_balance': balance.get('USDT', {}).get('free', 0),
                'used_balance': balance.get('USDT', {}).get('used', 0),
                'info': balance.get('info', {})
            }
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            raise
    
    def get_margin_info(self) -> Dict[str, Any]:
        """
        获取保证金信息
        
        Returns:
            保证金信息
        """
        try:
            balance = self.get_account_balance()
            return {
                'total_margin': balance.get('USDT', {}).get('total', 0),
                'available_margin': balance.get('USDT', {}).get('free', 0),
                'used_margin': balance.get('USDT', {}).get('used', 0),
                'margin_ratio': 0.0  # 需要从info中计算
            }
        except Exception as e:
            logger.error(f"获取保证金信息失败: {e}")
            raise
    
    # ========== 仓位查询 ==========
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取仓位信息
        
        Args:
            symbol: 交易对符号，None表示获取所有仓位
        
        Returns:
            仓位列表
        """
        try:
            positions = self.exchange.fetch_positions(symbols=[symbol] if symbol else None)
            result = []
            for pos in positions:
                if pos['contracts'] != 0:  # 只返回有仓位的
                    result.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': pos['contracts'],
                        'entry_price': pos['entryPrice'],
                        'mark_price': pos['markPrice'],
                        'unrealized_pnl': pos['unrealizedPnl'],
                        'percentage': pos['percentage'],
                        'leverage': pos.get('leverage', 1),
                        'notional': pos.get('notional', 0),
                        'margin_mode': pos.get('marginMode', 'isolated'),
                        'liquidation_price': pos.get('liquidationPrice')
                    })
            return result
        except Exception as e:
            logger.error(f"获取仓位信息失败: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取指定交易对的仓位
        
        Args:
            symbol: 交易对符号
        
        Returns:
            仓位信息，如果没有仓位返回None
        """
        positions = self.get_positions(symbol)
        return positions[0] if positions else None
    
    # ========== 订单操作 ==========
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False
    ) -> Dict[str, Any]:
        """
        下单
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
            reduce_only: 是否只减仓
            close_position: 是否平仓
        
        Returns:
            订单信息
        """
        try:
            # 转换订单类型
            ccxt_side = 'buy' if side == OrderSide.BUY else 'sell'
            ccxt_type = self._convert_order_type(order_type)
            
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            if close_position:
                params['closePosition'] = True
            
            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(
                    symbol, ccxt_side, quantity, params=params
                )
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("限价单需要指定价格")
                order = self.exchange.create_limit_order(
                    symbol, ccxt_side, quantity, price, params=params
                )
            elif order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
                if stop_price is None:
                    raise ValueError("止损单需要指定止损价格")
                params['stopPrice'] = stop_price
                if order_type == OrderType.STOP_MARKET:
                    params['type'] = 'STOP_MARKET'
                order = self.exchange.create_order(
                    symbol, ccxt_type, ccxt_side, quantity, price, params=params
                )
            elif order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                if stop_price is None:
                    raise ValueError("止盈单需要指定止盈价格")
                params['stopPrice'] = stop_price
                if order_type == OrderType.TAKE_PROFIT_MARKET:
                    params['type'] = 'TAKE_PROFIT_MARKET'
                order = self.exchange.create_order(
                    symbol, ccxt_type, ccxt_side, quantity, price, params=params
                )
            else:
                raise ValueError(f"不支持的订单类型: {order_type}")
            
            return {
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'status': order['status'],
                'quantity': order['amount'],
                'price': order.get('price'),
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'info': order.get('info', {})
            }
        except Exception as e:
            logger.error(f"下单失败: {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        撤单
        
        Args:
            order_id: 订单ID
            symbol: 交易对符号
        
        Returns:
            是否成功
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            return result.get('status') == 'canceled'
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            raise
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        撤销所有订单
        
        Args:
            symbol: 交易对符号，None表示撤销所有交易对的订单
        
        Returns:
            撤销的订单列表
        """
        try:
            if symbol:
                result = self.exchange.cancel_all_orders(symbol)
            else:
                # 需要获取所有交易对的订单
                open_orders = self.get_open_orders()
                result = []
                for order in open_orders:
                    try:
                        canceled = self.cancel_order(order['id'], order['symbol'])
                        if canceled:
                            result.append(order)
                    except Exception as e:
                        logger.warning(f"撤销订单 {order['id']} 失败: {e}")
            return result
        except Exception as e:
            logger.error(f"撤销所有订单失败: {e}")
            raise
    
    def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        查询订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对符号
        
        Returns:
            订单信息
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return {
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'status': order['status'],
                'quantity': order['amount'],
                'price': order.get('price'),
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'average_price': order.get('average'),
                'created_at': order.get('timestamp'),
                'info': order.get('info', {})
            }
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取未完成订单
        
        Args:
            symbol: 交易对符号，None表示获取所有交易对的订单
        
        Returns:
            订单列表
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            result = []
            for order in orders:
                result.append({
                    'order_id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'status': order['status'],
                    'quantity': order['amount'],
                    'price': order.get('price'),
                    'filled': order.get('filled', 0),
                    'remaining': order.get('remaining', 0),
                    'created_at': order.get('timestamp'),
                    'info': order.get('info', {})
                })
            return result
        except Exception as e:
            logger.error(f"获取未完成订单失败: {e}")
            raise
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """转换订单类型为ccxt格式"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_MARKET: 'stop_market',
            OrderType.TAKE_PROFIT: 'take_profit',
            OrderType.TAKE_PROFIT_MARKET: 'take_profit_market'
        }
        return mapping.get(order_type, 'market')
    
    # ========== 交易对信息 ==========
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取交易对信息
        
        Args:
            symbol: 交易对符号
        
        Returns:
            交易对信息
        """
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                market = markets[symbol]
                return {
                    'symbol': symbol,
                    'base': market['base'],
                    'quote': market['quote'],
                    'precision': {
                        'amount': market['precision']['amount'],
                        'price': market['precision']['price']
                    },
                    'limits': {
                        'amount': market['limits']['amount'],
                        'price': market['limits']['price'],
                        'cost': market['limits']['cost']
                    },
                    'active': market.get('active', True),
                    'contract': market.get('contract', False),
                    'info': market.get('info', {})
                }
            return None
        except Exception as e:
            logger.error(f"获取交易对信息失败: {e}")
            return None
    
    def get_leverage(self, symbol: str) -> Optional[int]:
        """
        获取杠杆倍数
        
        Args:
            symbol: 交易对符号
        
        Returns:
            杠杆倍数
        """
        try:
            # ccxt可能不支持直接获取杠杆，需要从仓位信息中获取
            position = self.get_position(symbol)
            if position:
                return position.get('leverage', 1)
            return None
        except Exception as e:
            logger.error(f"获取杠杆倍数失败: {e}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        设置杠杆倍数
        
        Args:
            symbol: 交易对符号
            leverage: 杠杆倍数
        
        Returns:
            是否成功
        """
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error(f"设置杠杆倍数失败: {e}")
            return False
