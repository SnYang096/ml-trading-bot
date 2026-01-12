import pytest

from src.time_series_model.portfolio.capital_escalation import (
    eligible_for_escalation,
    late_bull_leverage_allowed,
)


@pytest.mark.unit
def test_eligible_for_escalation_requires_ath_and_profit_factor():
    ok, reason = eligible_for_escalation(
        {
            "portfolio_equity_ath": False,
            "portfolio_profit_factor": 2.0,
            "rule_avg_max_dd": 0.01,
        }
    )
    assert ok is False
    assert reason == "equity_not_ath"

    ok2, reason2 = eligible_for_escalation(
        {
            "portfolio_equity_ath": True,
            "portfolio_profit_factor": 0.5,
            "rule_avg_max_dd": 0.01,
        }
    )
    assert ok2 is False
    assert reason2 == "profit_factor_too_low"

    ok3, reason3 = eligible_for_escalation(
        {
            "portfolio_equity_ath": True,
            "portfolio_profit_factor": 2.0,
            "rule_avg_max_dd": 0.01,
        }
    )
    assert ok3 is True
    assert reason3 == "eligible"


@pytest.mark.unit
def test_late_bull_leverage_allowed_only_in_mature_trend():
    ok, reason = late_bull_leverage_allowed(
        {
            "bull_phase": 2,
            "trend_duration_bars": 999,
            "trend_stability": 0.99,
            "portfolio_equity_ath": True,
        }
    )
    assert ok is False
    assert reason == "phase_not_mature"

    ok2, reason2 = late_bull_leverage_allowed(
        {
            "bull_phase": 3,
            "trend_duration_bars": 100,
            "trend_stability": 0.8,
            "portfolio_equity_ath": True,
        }
    )
    assert ok2 is True
    assert reason2 == "allowed"
