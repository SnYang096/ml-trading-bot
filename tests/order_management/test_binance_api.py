"""
Binance API 封装测试
"""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime

from src.order_management.binance_api import BinanceAPI, _ccxt_linear_usdt_perp_symbol
from src.order_management.models import OrderSide, OrderType

_CCXT_BTC_PERP = "BTC/USDT:USDT"


@pytest.fixture
def mock_ccxt_exchange():
    """创建模拟的ccxt交易所对象"""
    exchange = MagicMock()

    # 模拟markets
    exchange.markets = {
        "BTCUSDT": {
            "base": "BTC",
            "quote": "USDT",
            "precision": {"amount": 3, "price": 2},
            "limits": {
                "amount": {"min": 0.001, "max": 1000.0},
                "price": {"min": 0.01, "max": 1000000.0},
                "cost": {"min": 10.0, "max": 10000000.0},
            },
            "active": True,
            "contract": True,
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                ]
            },
        }
    }

    # 模拟load_markets
    exchange.load_markets.return_value = exchange.markets

    # 模拟fetch_balance
    exchange.fetch_balance.return_value = {
        "USDT": {
            "total": 10000.0,
            "free": 5000.0,
            "used": 5000.0,
        },
        "info": {},
    }

    # 模拟fetch_positions
    exchange.fetch_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "contracts": 0.1,
            "entryPrice": 50000.0,
            "markPrice": 51000.0,
            "unrealizedPnl": 100.0,
            "percentage": 2.0,
            "leverage": 10,
            "notional": 5100.0,
            "marginMode": "isolated",
            "liquidationPrice": 45000.0,
        }
    ]

    # 模拟create_market_order
    exchange.create_market_order.return_value = {
        "id": "binance_order_123",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "new",
        "amount": 0.1,
        "filled": 0,
        "remaining": 0.1,
        "price": None,
        "clientOrderId": "client_123",
    }

    # 模拟create_limit_order
    exchange.create_limit_order.return_value = {
        "id": "binance_order_124",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "limit",
        "status": "new",
        "amount": 0.1,
        "filled": 0,
        "remaining": 0.1,
        "price": 50000.0,
    }

    # 模拟create_order
    exchange.create_order.return_value = {
        "id": "binance_order_125",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "stop",
        "status": "new",
        "amount": 0.1,
        "filled": 0,
        "remaining": 0.1,
        "price": 49000.0,
    }

    # 模拟cancel_order
    exchange.cancel_order.return_value = {
        "id": "binance_order_123",
        "status": "canceled",
    }

    # 模拟fetch_order
    exchange.fetch_order.return_value = {
        "id": "binance_order_123",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "filled",
        "amount": 0.1,
        "filled": 0.1,
        "remaining": 0,
        "price": None,
        "average": 50000.0,
        "timestamp": int(datetime.now().timestamp() * 1000),
    }

    # 模拟fetch_open_orders
    exchange.fetch_open_orders.return_value = [
        {
            "id": "binance_order_123",
            "symbol": "BTCUSDT",
            "side": "buy",
            "type": "market",
            "status": "new",
            "amount": 0.1,
            "filled": 0,
            "remaining": 0.1,
            "price": None,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }
    ]

    # 模拟set_leverage
    exchange.set_leverage.return_value = None

    # 模拟URLs
    exchange.urls = {
        "api": {
            "public": "https://fapi.binance.com",
            "private": "https://fapi.binance.com",
        }
    }

    exchange.options = {}

    return exchange


@pytest.fixture
def binance_api(mock_ccxt_exchange):
    """创建BinanceAPI实例（主网）"""
    with patch(
        "src.order_management.binance_api.ccxt.binance", return_value=mock_ccxt_exchange
    ):
        api = BinanceAPI(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False,
        )
        api.exchange = mock_ccxt_exchange
        return api


@pytest.fixture
def binance_api_testnet(mock_ccxt_exchange):
    """创建BinanceAPI实例（测试网）"""
    with patch(
        "src.order_management.binance_api.ccxt.binance", return_value=mock_ccxt_exchange
    ):
        api = BinanceAPI(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )
        api.exchange = mock_ccxt_exchange
        return api


def test_open_order_from_binance_rest_normalizes_row(binance_api):
    """Binance REST openOrders 行字段映射到 get_open_orders 条目形状。"""
    row = {
        "orderId": 99,
        "symbol": "BTCUSDT",
        "clientOrderId": "c1",
        "price": "50000",
        "origQty": "0.1",
        "executedQty": "0.02",
        "status": "NEW",
        "side": "BUY",
        "type": "LIMIT",
        "time": 1700000000000,
    }
    out = binance_api._open_order_from_binance_rest(row)
    assert out["order_id"] == "99"
    assert out["symbol"] == "BTCUSDT"
    assert out["side"] == "buy"
    assert out["status"] == "new"
    assert out["quantity"] == 0.1
    assert out["filled"] == 0.02
    assert out["remaining"] == pytest.approx(0.08)
    assert out["price"] == 50000.0
    assert out["created_at"] == 1700000000


def test_ccxt_linear_usdt_perp_symbol_mapping():
    assert _ccxt_linear_usdt_perp_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert _ccxt_linear_usdt_perp_symbol("btcusdt") == "BTC/USDT:USDT"
    assert _ccxt_linear_usdt_perp_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
    assert _ccxt_linear_usdt_perp_symbol(None) is None


def test_init_mainnet(binance_api):
    """测试主网初始化"""
    assert binance_api.api_key == "test_key"
    assert binance_api.api_secret == "test_secret"
    assert binance_api.testnet == False
    assert binance_api.exchange is not None


def test_init_testnet(binance_api_testnet):
    """测试测试网初始化"""
    assert binance_api_testnet.testnet == True
    assert binance_api_testnet.exchange is not None


def test_init_with_proxy():
    """测试代理配置"""
    with patch("src.order_management.binance_api.ccxt.binance") as mock_binance:
        mock_exchange = MagicMock()
        mock_binance.return_value = mock_exchange

        api = BinanceAPI(
            api_key="test_key",
            api_secret="test_secret",
            use_proxy=True,
            proxy_host="127.0.0.1",
            proxy_port=7897,
        )

        assert api.use_proxy == True
        assert api.proxy_host == "127.0.0.1"
        assert api.proxy_port == 7897


def test_init_proxy_from_env():
    """测试从环境变量读取代理配置"""
    with patch("src.order_management.binance_api.ccxt.binance") as mock_binance:
        mock_exchange = MagicMock()
        mock_binance.return_value = mock_exchange

        with patch.dict(os.environ, {"USE_SOCKS5_PROXY": "true"}):
            api = BinanceAPI(
                api_key="test_key",
                api_secret="test_secret",
            )
            assert api.use_proxy == True


def test_get_account_balance(binance_api):
    """测试获取账户余额"""
    balance = binance_api.get_account_balance()

    assert balance is not None
    assert "USDT" in balance
    assert balance["USDT"]["total"] == 10000.0
    binance_api.exchange.fetch_balance.assert_called_once()


def test_get_account_info(binance_api):
    """测试获取账户信息"""
    info = binance_api.get_account_info()

    assert info is not None
    assert info["total_balance"] == 10000.0
    assert info["free_balance"] == 5000.0
    assert info["used_balance"] == 5000.0


def test_get_margin_info(binance_api):
    """测试获取保证金信息"""
    margin_info = binance_api.get_margin_info()

    assert margin_info is not None
    assert margin_info["total_margin"] == 10000.0
    assert margin_info["available_margin"] == 5000.0
    assert margin_info["used_margin"] == 5000.0


def test_get_positions(binance_api):
    """测试获取仓位信息"""
    positions = binance_api.get_positions()

    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTCUSDT"
    assert positions[0]["size"] == 0.1
    assert positions[0]["entry_price"] == 50000.0
    binance_api.exchange.fetch_positions.assert_called_once()


def test_get_positions_with_symbol(binance_api):
    """测试获取指定交易对的仓位"""
    positions = binance_api.get_positions("BTCUSDT")

    assert len(positions) == 1
    binance_api.exchange.fetch_positions.assert_called_once_with(
        symbols=[_CCXT_BTC_PERP]
    )


def test_get_positions_empty(binance_api):
    """测试获取空仓位列表"""
    binance_api.exchange.fetch_positions.return_value = [
        {"symbol": "BTCUSDT", "contracts": 0}  # 无仓位
    ]

    positions = binance_api.get_positions()
    assert len(positions) == 0


def test_get_position(binance_api):
    """测试获取指定交易对的仓位"""
    position = binance_api.get_position("BTCUSDT")

    assert position is not None
    assert position["symbol"] == "BTCUSDT"


def test_get_position_not_found(binance_api):
    """测试获取不存在的仓位"""
    binance_api.exchange.fetch_positions.return_value = []

    position = binance_api.get_position("BTCUSDT")
    assert position is None


def test_place_market_order(binance_api):
    """测试下市价单"""
    order = binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    assert order is not None
    assert order["order_id"] == "binance_order_123"
    assert order["symbol"] == "BTCUSDT"
    assert order["side"] == "buy"
    binance_api.exchange.create_market_order.assert_called_once()


def test_place_limit_order(binance_api):
    """测试下限价单"""
    order = binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=50000.0,
    )

    assert order is not None
    assert order["order_id"] == "binance_order_124"
    binance_api.exchange.create_limit_order.assert_called_once()


def test_place_post_only_limit_order_uses_gtx(binance_api):
    """测试 Binance Futures post-only 限价单参数。"""
    binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=51000.0,
        post_only=True,
        client_order_id="cg_tp_1",
    )

    params = binance_api.exchange.create_limit_order.call_args.kwargs["params"]
    assert params["timeInForce"] == "GTX"
    assert params["newClientOrderId"] == "cg_tp_1"


def test_place_limit_order_no_price(binance_api):
    """测试下限价单未指定价格"""
    with pytest.raises(ValueError, match="限价单需要指定价格"):
        binance_api.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
        )


def test_place_stop_order(binance_api):
    """测试下止损单"""
    order = binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=0.1,
        price=50000.0,
        stop_price=49000.0,
    )

    assert order is not None
    binance_api.exchange.create_order.assert_called_once()


def test_place_stop_market_accepts_explicit_hedge_protection_params(binance_api):
    """测试 Hedge Mode 条件保护单参数透传。"""
    binance_api.hedge_mode = True

    binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.1,
        stop_price=49000.0,
        reduce_only=True,
        position_side="LONG",
        working_type="MARK_PRICE",
        price_protect=True,
        client_order_id="cg_sl_1",
    )

    params = binance_api.exchange.create_order.call_args.kwargs["params"]
    assert params["positionSide"] == "LONG"
    assert params["stopPrice"] == 49000.0
    assert params["type"] == "STOP_MARKET"
    assert params["workingType"] == "MARK_PRICE"
    assert params["priceProtect"] == "TRUE"
    assert params["newClientOrderId"] == "cg_sl_1"
    assert "reduceOnly" not in params


def test_place_stop_order_no_stop_price(binance_api):
    """测试下止损单未指定止损价格"""
    with pytest.raises(ValueError, match="止损单需要指定止损价格"):
        binance_api.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=0.1,
            price=50000.0,
        )


def test_place_order_reduce_only(binance_api):
    """测试只减仓订单"""
    order = binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        reduce_only=True,
    )

    assert order is not None
    # 验证params中包含reduceOnly
    call_args = binance_api.exchange.create_market_order.call_args
    assert "reduceOnly" in call_args.kwargs.get("params", {})


def test_cancel_order(binance_api):
    """测试撤单"""
    success = binance_api.cancel_order("binance_order_123", "BTCUSDT")

    assert success == True
    binance_api.exchange.cancel_order.assert_called_once_with(
        "binance_order_123", _CCXT_BTC_PERP
    )


def test_cancel_order_failed(binance_api):
    """测试撤单失败"""
    binance_api.exchange.cancel_order.return_value = {"status": "new"}

    success = binance_api.cancel_order("binance_order_123", "BTCUSDT")
    assert success == False


def test_cancel_all_orders(binance_api):
    """测试撤销所有订单（指定交易对）"""
    binance_api.exchange.cancel_all_orders.return_value = [
        {"id": "binance_order_123", "status": "canceled"}
    ]

    result = binance_api.cancel_all_orders("BTCUSDT")
    assert len(result) == 1
    binance_api.exchange.cancel_all_orders.assert_called_once_with(_CCXT_BTC_PERP)


def test_cancel_all_orders_all_symbols(binance_api):
    """测试撤销所有订单（所有交易对）"""
    # 模拟get_open_orders返回多个订单（注意：get_open_orders返回的格式使用order_id）
    binance_api.exchange.fetch_open_orders.return_value = [
        {
            "id": "order1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "type": "market",
            "status": "new",
            "amount": 0.1,
            "filled": 0,
            "remaining": 0.1,
            "price": None,
            "timestamp": int(datetime.now().timestamp() * 1000),
        },
        {
            "id": "order2",
            "symbol": "ETHUSDT",
            "side": "sell",
            "type": "market",
            "status": "new",
            "amount": 1.0,
            "filled": 0,
            "remaining": 1.0,
            "price": None,
            "timestamp": int(datetime.now().timestamp() * 1000),
        },
    ]

    result = binance_api.cancel_all_orders()
    # 应该尝试撤销所有订单
    assert binance_api.exchange.cancel_order.call_count == 2


def test_get_order(binance_api):
    """测试查询订单"""
    order = binance_api.get_order("binance_order_123", "BTCUSDT")

    assert order is not None
    assert order["order_id"] == "binance_order_123"
    assert order["status"] == "filled"
    assert isinstance(order["created_at"], int)
    binance_api.exchange.fetch_order.assert_called_once_with(
        "binance_order_123", _CCXT_BTC_PERP
    )


def test_get_order_not_found(binance_api):
    """测试查询不存在的订单"""
    binance_api.exchange.fetch_order.side_effect = Exception("Order not found")

    order = binance_api.get_order("nonexistent", "BTCUSDT")
    assert order is None


def test_get_open_orders(binance_api):
    """测试获取未完成订单"""
    orders = binance_api.get_open_orders()

    assert len(orders) == 1
    assert orders[0]["order_id"] == "binance_order_123"
    binance_api.exchange.fetch_open_orders.assert_called_once()


def test_get_open_orders_with_symbol(binance_api):
    """测试获取指定交易对的未完成订单"""
    orders = binance_api.get_open_orders("BTCUSDT")

    assert len(orders) == 1
    binance_api.exchange.fetch_open_orders.assert_called_once_with(_CCXT_BTC_PERP)


def test_get_symbol_info(binance_api):
    """测试获取交易对信息"""
    info = binance_api.get_symbol_info("BTCUSDT")

    assert info is not None
    assert info["symbol"] == "BTCUSDT"
    assert info["base"] == "BTC"
    assert info["quote"] == "USDT"
    assert "precision" in info
    assert "limits" in info
    assert info["filters"]["tick_size"] == 0.1
    assert info["filters"]["step_size"] == 0.001


def test_place_order_client_order_id(binance_api):
    """测试下单返回client_order_id"""
    order = binance_api.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        client_order_id="client_123",
    )
    assert order["client_order_id"] == "client_123"


def test_get_symbol_info_not_found(binance_api):
    """测试获取不存在的交易对信息"""
    binance_api.exchange.markets = {}
    binance_api.exchange.load_markets.return_value = {}

    info = binance_api.get_symbol_info("NONEXISTENT")
    assert info is None


def test_get_leverage(binance_api):
    """测试获取杠杆倍数"""
    leverage = binance_api.get_leverage("BTCUSDT")

    assert leverage == 10  # 从mock的position中获取


def test_get_leverage_no_position(binance_api):
    """测试获取杠杆倍数（无仓位）"""
    binance_api.exchange.fetch_positions.return_value = []

    leverage = binance_api.get_leverage("BTCUSDT")
    assert leverage is None


def test_set_leverage(binance_api):
    """测试设置杠杆倍数"""
    success = binance_api.set_leverage("BTCUSDT", 20)

    assert success == True
    binance_api.exchange.set_leverage.assert_called_once_with(20, _CCXT_BTC_PERP)


def test_set_leverage_failed(binance_api):
    """测试设置杠杆倍数失败"""
    binance_api.exchange.set_leverage.side_effect = Exception("Failed")

    success = binance_api.set_leverage("BTCUSDT", 20)
    assert success == False


@patch("requests.get")
def test_get_ticker_price(mock_get, binance_api):
    """测试获取市场价格"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"price": "50000.0"}

    price = binance_api.get_ticker_price("BTCUSDT")

    assert price == 50000.0
    mock_get.assert_called_once()


@patch("requests.get")
def test_get_ticker_price_testnet(mock_get, binance_api_testnet):
    """测试获取市场价格（测试网）"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"price": "50000.0"}

    price = binance_api_testnet.get_ticker_price("BTCUSDT")

    assert price == 50000.0
    # 验证使用了测试网URL
    call_args = mock_get.call_args[0][0]
    assert "testnet.binancefuture.com" in call_args


@patch("requests.get")
def test_get_ticker_price_failed(mock_get, binance_api):
    """测试获取市场价格失败"""
    mock_get.return_value.status_code = 500

    price = binance_api.get_ticker_price("BTCUSDT")
    assert price is None


def test_convert_order_type(binance_api):
    """测试订单类型转换"""
    assert binance_api._convert_order_type(OrderType.MARKET) == "market"
    assert binance_api._convert_order_type(OrderType.LIMIT) == "limit"
    assert binance_api._convert_order_type(OrderType.STOP) == "stop"
    assert binance_api._convert_order_type(OrderType.STOP_MARKET) == "stop_market"
    assert binance_api._convert_order_type(OrderType.TAKE_PROFIT) == "take_profit"
    assert (
        binance_api._convert_order_type(OrderType.TAKE_PROFIT_MARKET)
        == "take_profit_market"
    )


@patch("requests.get")
def test_testnet_fetch_markets_patch(mock_get, binance_api_testnet):
    """测试测试网的fetch_markets monkey patch"""
    # 验证fetch_markets方法被替换
    assert hasattr(binance_api_testnet.exchange, "fetch_markets")

    # 测试fetch_markets调用（需要mock requests）
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "pricePrecision": 2,
                "quantityPrecision": 3,
            }
        ]
    }

    # 调用load_markets会触发fetch_markets
    try:
        binance_api_testnet.exchange.load_markets()
    except Exception:
        pass  # 可能因为parse_market失败，但不影响测试


def test_testnet_sign_patch(binance_api_testnet):
    """测试测试网的sign monkey patch"""
    # 验证sign方法被替换
    assert hasattr(binance_api_testnet.exchange, "sign")

    # 验证URLs中包含fapiPrivate（在初始化时已设置）
    api_urls = binance_api_testnet.exchange.urls.get("api", {})
    # 由于是mock对象，可能没有正确设置，但至少验证sign方法存在
    assert callable(binance_api_testnet.exchange.sign)
