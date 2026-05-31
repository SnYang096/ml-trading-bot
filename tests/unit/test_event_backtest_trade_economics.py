from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.event_backtest.simulator.position import PositionSimulator


def _sim_with_budget(*, risk_per_slot_usdt: float = 100.0) -> PositionSimulator:
    sim = PositionSimulator(fee_rate=0.0)
    sim._risk_per_slot_usdt = float(risk_per_slot_usdt)
    return sim


def test_pnl_r_matches_realized_over_risk_budget_on_add_leg() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=100.0)
    pos = {
        "side": "LONG",
        "_size_multiplier": 3.0,
        "_entry_notional_usdt": 300.0,
        "_qty_base": 300.0 / 0.39,
        "entry_price": 0.39,
        "initial_risk_distance": 0.0065,
        "atr_at_entry": 0.0065,
        "effective_stop_pct": 0.0,
    }
    econ = sim._build_close_economics(pos=pos, exit_price=1.22)
    pnl_r = sim._pnl_r_from_economics(
        pos=pos,
        econ=econ,
        entry_price=0.39,
        exit_price=1.22,
    )
    realized = float(econ["pnl_usd_realized"])
    assert realized > 500.0
    assert pnl_r == pytest.approx(realized / 300.0, rel=1e-6)


def test_price_path_fallback_when_no_risk_budget() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=0.0)
    pos = {
        "side": "LONG",
        "_size_multiplier": 1.0,
        "initial_risk_distance": 10.0,
        "atr_at_entry": 10.0,
    }
    econ = {"pnl_usd_realized": 50.0}
    pnl_r = sim._pnl_r_from_economics(
        pos=pos,
        econ=econ,
        entry_price=100.0,
        exit_price=110.0,
    )
    assert pnl_r == pytest.approx(1.0)
