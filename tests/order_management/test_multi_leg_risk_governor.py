from __future__ import annotations

from src.order_management.multi_leg_risk_governor import (
    ExposureSnapshot,
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)


def test_rejects_place_when_portfolio_gross_would_exceed_limit() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
    )

    result = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.02,
                "price": 60_000.0,
            }
        ]
    )

    assert result.approved_actions == []
    assert "max_gross_notional exceeded" in result.rejected[0].reason


def test_rejects_place_when_net_exposure_would_exceed_limit() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(max_gross_notional=10_000.0, max_net_notional=500.0)
    )

    result = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.02,
                "price": 60_000.0,
            }
        ],
        positions=[
            ExposureSnapshot("BTCUSDT", "LONG", 0.01, 60_000.0),
            ExposureSnapshot("BTCUSDT", "SHORT", 0.01, 60_000.0),
        ],
    )

    assert not result.ok
    assert "max_net_notional exceeded" in result.rejected[0].reason


def test_allows_market_exit_even_when_existing_exposure_is_over_limit() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(max_gross_notional=100.0, max_net_notional=100.0)
    )

    result = governor.check_actions(
        [
            {
                "action": "market_exit",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.01,
            }
        ],
        positions=[ExposureSnapshot("BTCUSDT", "LONG", 0.1, 60_000.0)],
    )

    assert result.ok
    assert result.approved_actions[0]["action"] == "market_exit"


def test_rejects_when_resting_order_limit_would_be_exceeded() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=10_000.0,
            max_net_notional=10_000.0,
            max_resting_orders=1,
        )
    )

    result = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "quantity": 0.01,
                "price": 60_000.0,
            }
        ],
        open_orders=[{"order_id": "existing"}],
    )

    assert not result.ok
    assert "max_resting_orders exceeded" in result.rejected[0].reason


def test_processes_batch_incrementally_for_symbol_caps() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=10_000.0,
            max_net_notional=10_000.0,
            max_symbol_gross_notional=1_100.0,
        )
    )

    result = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 60_000.0,
            },
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "quantity": 0.01,
                "price": 60_000.0,
            },
        ]
    )

    assert len(result.approved_actions) == 1
    assert len(result.rejected) == 1
    assert "max_symbol_gross_notional exceeded" in result.rejected[0].reason
