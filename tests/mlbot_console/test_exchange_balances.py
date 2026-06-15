"""Tests for exchange_balances margin and open orders parsing."""

import pytest
from mlbot_console.services.exchange_balances import (
    parse_futures_account,
    parse_open_orders_margin,
    futures_open_positions,
)


class TestFuturesOpenPositions:
    def test_liquidation_price_parsing(self):
        data = {
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.1",
                    "entryPrice": "50000",
                    "markPrice": "51000",
                    "leverage": "10",
                    "positionInitialMargin": "500",
                    "maintMargin": "50",
                    "marginType": "cross",
                    "unRealizedProfit": "100",
                    "liquidationPrice": "45000",
                }
            ]
        }
        positions = futures_open_positions(data)
        assert len(positions) == 1
        assert positions[0]["liquidation_price"] == 45000.0

    def test_missing_liquidation_price(self):
        data = {
            "positions": [
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "1.0",
                    "entryPrice": "3000",
                    "markPrice": "3100",
                }
            ]
        }
        positions = futures_open_positions(data)
        assert positions[0]["liquidation_price"] is None


class TestParseFuturesAccount:
    def test_basic_fields(self):
        data = {
            "totalMarginBalance": "10000",
            "totalWalletBalance": "9800",
            "availableBalance": "8000",
            "totalMaintMargin": "200",
            "totalPositionInitialMargin": "1500",
            "totalOpenOrderInitialMargin": "500",
            "totalUnrealizedProfit": "200",
        }
        result = parse_futures_account(data)
        assert result["equity_usdt"] == 10000.0
        assert result["margin_locked_usdt"] == 2000.0  # 10000 - 8000
        assert result["gross_leverage"] is None  # No positions provided


class TestParseOpenOrdersMargin:
    def test_parse_single_order(self):
        orders = [
            {
                "orderId": 12345,
                "clientOrderId": "test_order",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "positionSide": "LONG",
                "type": "LIMIT",
                "price": "60000",
                "origQty": "0.1",
                "initialMargin": "100",
                "status": "NEW",
            }
        ]
        result = parse_open_orders_margin(orders)
        assert len(result) == 1
        assert result[0]["order_id"] == "12345"
        assert result[0]["initial_margin_usdt"] == 100.0
        assert result[0]["position_side"] == "LONG"

    def test_parse_zero_margin(self):
        orders = [{"orderId": 1, "initialMargin": "0"}]
        result = parse_open_orders_margin(orders)
        assert result[0]["initial_margin_usdt"] is None
