"""Tests for open_positions_list enrichment logic."""

import pytest

from mlbot_console.services.open_positions_list import _enrich_with_exchange_legs


class TestEnrichWithExchangeLegs:
    def test_enrich_single_position(self):
        rows = [
            {
                "scope": "trend",
                "symbol": "BTCUSDT",
                "side": "long",
                "quantity": 0.1,
            }
        ]
        exchange_ledger = {
            "accounts": [
                {
                    "scope": "trend",
                    "exchange_open_positions": [
                        {
                            "symbol": "BTCUSDT",
                            "position_amt": "0.1",
                            "leverage": 10,
                            "notional_usdt": 5000,
                            "initial_margin_usdt": 500,
                            "maint_margin_usdt": 50,
                            "liquidation_price": 45000,
                            "margin_type": "cross",
                        }
                    ],
                }
            ]
        }
        _enrich_with_exchange_legs(rows, exchange_ledger)
        assert rows[0]["exchange_leverage"] == 10
        assert rows[0]["exchange_initial_margin_usdt"] == 500
        assert rows[0]["exchange_liquidation_price"] == 45000

    def test_no_match_keeps_fields_none(self):
        rows = [{"scope": "trend", "symbol": "ETHUSDT", "side": "long"}]
        exchange_ledger = {"accounts": []}
        _enrich_with_exchange_legs(rows, exchange_ledger)
        assert (
            "exchange_leverage" not in rows[0]
            or rows[0].get("exchange_leverage") is None
        )

    def test_multi_leg_fallback(self):
        """Test that multi-leg positions can find data in trend/multi_leg accounts."""
        rows = [{"scope": "multi_leg", "symbol": "SOLUSDT", "side": "short"}]
        exchange_ledger = {
            "accounts": [
                {
                    "scope": "multi_leg",
                    "exchange_open_positions": [
                        {
                            "symbol": "SOLUSDT",
                            "position_amt": "-5",
                            "leverage": 5,
                            "initial_margin_usdt": 200,
                            "liquidation_price": 200,
                        }
                    ],
                }
            ]
        }
        _enrich_with_exchange_legs(rows, exchange_ledger)
        assert rows[0]["exchange_leverage"] == 5
        assert rows[0]["exchange_liquidation_price"] == 200

    def test_multi_leg_prorata_margin(self):
        rows = [
            {
                "scope": "multi_leg",
                "symbol": "XRPUSDT",
                "side": "short",
                "quantity": 100,
            },
            {
                "scope": "multi_leg",
                "symbol": "XRPUSDT",
                "side": "short",
                "quantity": 300,
            },
        ]
        exchange_ledger = {
            "accounts": [
                {
                    "scope": "multi_leg",
                    "exchange_open_positions": [
                        {
                            "symbol": "XRPUSDT",
                            "position_amt": "-400",
                            "leverage": 5,
                            "initial_margin_usdt": 1942.87,
                        }
                    ],
                }
            ]
        }
        _enrich_with_exchange_legs(rows, exchange_ledger)
        assert rows[0]["exchange_initial_margin_usdt"] == pytest.approx(
            485.7175, rel=1e-3
        )
        assert rows[1]["exchange_initial_margin_usdt"] == pytest.approx(
            1457.1525, rel=1e-3
        )
        assert rows[0]["exchange_margin_allocated"] is True
