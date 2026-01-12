import pytest

from src.time_series_model.portfolio.withdrawal_policy import (
    WithdrawalRulesV1,
    validate_withdrawal_request,
)


@pytest.mark.unit
def test_withdrawal_monthly_cap():
    rules = WithdrawalRulesV1(max_monthly_withdrawal_ratio=0.02)
    ok, reason, _ = validate_withdrawal_request(
        rules=rules,
        equity_usd=10_000,
        withdraw_usd=300,  # cap=200
        withdrawn_this_month_usd=0,
        realized_profit_ytd_usd=0,
        withdrawn_profit_ytd_usd=0,
        global_pause=False,
    )
    assert ok is False
    assert reason == "monthly_cap_exceeded"


@pytest.mark.unit
def test_withdrawal_profit_cap():
    rules = WithdrawalRulesV1(max_annual_profit_withdrawal_ratio=0.4)
    ok, reason, _ = validate_withdrawal_request(
        rules=rules,
        equity_usd=100_000,  # avoid hitting monthly cap first
        withdraw_usd=500,
        withdrawn_this_month_usd=0,
        realized_profit_ytd_usd=1000,  # cap=400
        withdrawn_profit_ytd_usd=0,
        global_pause=False,
    )
    assert ok is False
    assert reason == "annual_profit_cap_exceeded"


@pytest.mark.unit
def test_withdrawal_forbidden_on_global_pause():
    rules = WithdrawalRulesV1(forbid_when_global_pause=True)
    ok, reason, _ = validate_withdrawal_request(
        rules=rules,
        equity_usd=10_000,
        withdraw_usd=100,
        withdrawn_this_month_usd=0,
        realized_profit_ytd_usd=0,
        withdrawn_profit_ytd_usd=0,
        global_pause=True,
    )
    assert ok is False
    assert reason == "forbid_when_global_pause"
