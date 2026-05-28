from __future__ import annotations

from scripts.event_backtest.simulator.position import PositionSimulator
from scripts.event_backtest.sizing import sync_event_backtest_sizing_equity


def test_sync_compound_updates_risk_per_slot() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    sims = {"BTC": sim}
    sync_event_backtest_sizing_equity(
        simulators=sims,
        equity_usdt=20_000.0,
        risk_per_slot=0.01,
        compound_sizing=True,
        initial_cash_usdt=10_000.0,
    )
    assert sim._risk_per_slot_usdt == 200.0
    assert sim._account_risk_equity == 20_000.0


def test_sync_fixed_base_ignores_equity_growth() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    sims = {"BTC": sim}
    sync_event_backtest_sizing_equity(
        simulators=sims,
        equity_usdt=25_000.0,
        risk_per_slot=0.01,
        compound_sizing=False,
        initial_cash_usdt=10_000.0,
    )
    assert sim._risk_per_slot_usdt == 100.0
    assert sim._account_risk_equity == 10_000.0


def test_sync_updates_spot_budget_equity() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    budget = {"equity_usdt": 10_000.0}
    sync_event_backtest_sizing_equity(
        simulators={"x": sim},
        equity_usdt=15_000.0,
        risk_per_slot=0.01,
        compound_sizing=True,
        initial_cash_usdt=10_000.0,
        spot_capital_budget=budget,
    )
    assert budget["equity_usdt"] == 15_000.0
