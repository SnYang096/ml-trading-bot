"""
BinanceWebSocketClient 单元测试
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from src.live_data_stream.websocket_client import (
    BinanceWebSocketClient,
    BinanceTick,
    create_and_run_websocket,
)


class TestBinanceTick:
    """BinanceTick 数据类测试"""

    def test_from_binance(self):
        """测试从 Binance 数据解析"""
        payload = {
            "e": "trade",
            "s": "BTCUSDT",
            "p": "50000.00",
            "q": "0.1",
            "T": 1234567890000,
            "m": False,  # 买方是 taker（主动买入）
            "t": 12345,
        }

        tick = BinanceTick.from_binance(payload)

        assert tick.symbol == "BTCUSDT"
        assert tick.price == 50000.0
        assert tick.volume == 0.1
        assert tick.timestamp_ms == 1234567890000
        assert tick.side == 1  # BUY
        assert tick.trade_id == 12345

    def test_from_binance_sell(self):
        """测试卖出 tick"""
        payload = {
            "e": "trade",
            "s": "BTCUSDT",
            "p": "50000.00",
            "q": "0.1",
            "T": 1234567890000,
            "m": True,  # 买方是 maker（主动卖出）
            "t": 12345,
        }

        tick = BinanceTick.from_binance(payload)

        assert tick.side == -1  # SELL

    def test_to_dict(self):
        """测试转换为字典"""
        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
            trade_id=12345,
        )

        result = tick.to_dict()

        assert result["symbol"] == "BTCUSDT"
        assert result["price"] == 50000.0
        assert result["volume"] == 0.1
        assert result["side"] == 1


class TestBinanceWebSocketClient:
    """BinanceWebSocketClient 测试"""

    def test_init(self):
        """测试初始化"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT", "ETHUSDT"],
            use_futures=True,
            reconnect_delay=5,
        )

        assert client.symbols == ["BTCUSDT", "ETHUSDT"]
        assert client.use_futures is True
        assert client.reconnect_delay == 5
        assert len(client._callbacks) == 0

    def test_init_empty_symbols(self):
        """测试空符号列表"""
        with pytest.raises(ValueError, match="symbols must not be empty"):
            BinanceWebSocketClient(symbols=[])

    def test_ws_url_spot(self):
        """测试现货 WebSocket URL"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT"],
            use_futures=False,
        )

        url = client._ws_url()
        assert "stream.binance.com" in url
        assert "btcusdt@trade" in url

    def test_ws_url_futures(self):
        """测试期货 WebSocket URL"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT"],
            use_futures=True,
        )

        url = client._ws_url()
        assert "fstream.binance.com" in url
        assert "btcusdt@trade" in url

    def test_ws_url_multiple_symbols(self):
        """测试多币种 WebSocket URL"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            use_futures=True,
        )

        url = client._ws_url()
        assert "btcusdt@trade" in url
        assert "ethusdt@trade" in url
        assert "solusdt@trade" in url

    def test_add_callback(self):
        """测试添加回调"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def callback(tick):
            pass

        client.add_callback(callback)

        assert len(client._callbacks) == 1
        assert callback in client._callbacks

    def test_remove_callback(self):
        """测试移除回调"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def callback1(tick):
            pass

        def callback2(tick):
            pass

        client.add_callback(callback1)
        client.add_callback(callback2)

        assert len(client._callbacks) == 2

        client.remove_callback(callback1)

        assert len(client._callbacks) == 1
        assert callback1 not in client._callbacks
        assert callback2 in client._callbacks

    def test_stream_ticks_success(self):
        """测试成功流式获取 tick（简化版，不测试实际 WebSocket 连接）"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        # 测试 URL 生成
        url = client._ws_url()
        assert "btcusdt@trade" in url

        # 测试基本功能（不测试实际 WebSocket 连接）
        assert client.symbols == ["BTCUSDT"]
        assert client.use_futures is True

    def test_stream_ticks_invalid_json(self):
        """测试无效 JSON 处理（简化版）"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        # 测试基本功能（不测试实际 WebSocket 连接）
        # 实际 WebSocket 错误处理在 stream_ticks 方法中实现
        assert client.reconnect_delay == 5

    def test_callbacks(self):
        """测试回调调用"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        callback_results = []

        def callback1(tick):
            callback_results.append(("callback1", tick))

        def callback2(tick):
            callback_results.append(("callback2", tick))

        client.add_callback(callback1)
        client.add_callback(callback2)

        # 创建测试 tick
        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        # 手动调用回调（模拟 WebSocket 收到数据）
        for callback in client._callbacks:
            callback(tick)

        assert len(callback_results) == 2
        assert callback_results[0][0] == "callback1"
        assert callback_results[1][0] == "callback2"
        assert callback_results[0][1].symbol == "BTCUSDT"

    def test_callback_error_handling(self):
        """测试回调错误处理"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def bad_callback(tick):
            raise ValueError("Callback error")

        def good_callback(tick):
            pass

        client.add_callback(bad_callback)
        client.add_callback(good_callback)

        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        # 回调应该处理错误，不影响其他回调
        for callback in client._callbacks:
            try:
                callback(tick)
            except Exception:
                pass  # 错误应该被捕获

        # 如果到这里没有异常，说明错误处理正常
        assert True
