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


def test_rejects_place_when_account_risk_limit_exceeded() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=10_000.0,
            max_net_notional=10_000.0,
            account_equity_usdt=1000.0,
            account_risk_limits={
                "enabled": True,
                "max_gross_leverage": 3.0,
                "margin_stress_leverage": 5.0,
            },
        )
    )

    result = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.05,
                "price": 60_000.0,
            }
        ],
        positions=[
            ExposureSnapshot("ETHUSDT", "SHORT", 1.0, 2900.0),
        ],
    )

    assert result.approved_actions == []
    assert "account_risk_limit" in result.rejected[0].reason


def test_rejects_new_places_when_drawdown_limit_is_reached() -> None:
    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=10_000.0,
            max_net_notional=10_000.0,
            account_equity_usdt=10_000.0,
            max_drawdown_pct=0.12,
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
            }
        ],
        drawdown_pct=0.12,
    )

    assert result.approved_actions == []
    assert "max_drawdown_pct exceeded" in result.rejected[0].reason


def test_rejects_place_when_kill_switch_halted(tmp_path) -> None:
    from datetime import datetime, timezone

    from src.order_management.multi_leg_kill_switch import (
        MultiLegKillSwitchConfig,
        MultiLegKillSwitchTracker,
    )
    from src.time_series_model.core.constitution.account_risk_guard import (
        AccountRiskSnapshot,
    )

    tracker = MultiLegKillSwitchTracker(
        config=MultiLegKillSwitchConfig(
            enabled=True,
            daily_loss_limit=0.06,
            max_dd=0.20,
            cooldown_minutes=0,
        ),
        state_path=tmp_path / "kill_switch_state.json",
    )
    now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
    tracker.begin_batch()
    tracker.update_from_equity(10_000.0, now=now)
    tracker.begin_batch()
    tracker.update_from_equity(9_300.0, now=now)
    assert tracker.is_halted()

    governor = MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=1_000_000.0, max_net_notional=1_000_000.0
        ),
        account_snapshot_provider=lambda: AccountRiskSnapshot(
            equity=9_300.0, gross_notional=0.0
        ),
        kill_switch_tracker=tracker,
    )

    place = governor.check_actions(
        [
            {
                "action": "place",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "price": 60_000.0,
            }
        ]
    )
    assert not place.ok
    assert "kill_switch" in place.rejected[0].reason

    exit_ok = governor.check_actions(
        [
            {
                "action": "market_exit",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.01,
            }
        ]
    )
    assert exit_ok.ok
