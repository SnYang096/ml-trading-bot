"""Spot new-buy gate: skip exchange order when free USDT is insufficient."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.time_series_model.live.decision_chain_debug import collect_spot_new_buy_report


def _minimal_report(*, planned_usdt: float, free_usdt: float | None) -> dict:
    intent = SimpleNamespace(action="LONG", size_multiplier=1.0)
    strategy = SimpleNamespace(archetype=None, _last_funnel={})
    budget = SimpleNamespace(max_new_entries_per_day=99, min_order_interval_minutes=0)
    ledger = SimpleNamespace(buy_entries_today=lambda _day: 0)
    return collect_spot_new_buy_report(
        symbol="ETHUSDT",
        ts=pd.Timestamp("2024-01-01", tz="UTC"),
        features={"weekly_ema_200_position": -0.1, "close": 3000.0},
        strategy=strategy,
        deploy_schedule_cfg={},
        budget=budget,
        positions={},
        ledger=ledger,
        day_key="2024-01-01",
        intents=[intent],
        om_shadow=False,
        planned_usdt=planned_usdt,
        free_usdt=free_usdt,
    )


def test_blocks_buy_when_free_usdt_below_planned():
    report = _minimal_report(planned_usdt=125.0, free_usdt=10.0)
    assert report["can_submit_new_buy"] is False
    assert any("insufficient_free_usdt" in b for b in report["blockers"])


def test_allows_buy_when_free_usdt_covers_planned():
    report = _minimal_report(planned_usdt=125.0, free_usdt=200.0)
    assert report["can_submit_new_buy"] is True
    assert not any("insufficient_free_usdt" in b for b in report["blockers"])
