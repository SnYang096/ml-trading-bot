from __future__ import annotations

from scripts.run_spot_accum_live import (
    SpotBudgetConfig,
    _spot_process_pending_buys,
)
from src.order_management.spot_live_recovery import (
    has_blocking_pending_buy,
    new_position_shell,
    set_pending_buy,
)


class _DummyApi:
    def fetch_order(self, _symbol: str, _exchange_order_id: str):
        raise RuntimeError("temporary api failure")


class _DummyOrderManager:
    def __init__(self) -> None:
        self.api = _DummyApi()
        self.shadow = False

    def cancel_exchange_order(self, _symbol: str, _exchange_order_id: str) -> None:
        raise RuntimeError("cancel failed")

    def update_order_record(self, *_args, **_kwargs) -> None:
        return


def _budget() -> SpotBudgetConfig:
    return SpotBudgetConfig(
        equity_anchor_usdt=1000.0,
        target_deploy_pct=0.5,
        max_gross_notional_pct=1.0,
        max_daily_deploy_pct=1.0,
        min_order_interval_minutes=10,
        max_new_entries_per_day=10,
        symbol_budgets_usdt={"BTCUSDT": 500.0},
        symbol_units_usdt={"BTCUSDT": 100.0},
        entry_order_type="limit",
        entry_limit_offset_bps=10.0,
        deploy_decay_cfg={},
        deploy_schedule_cfg={},
        profit_take_ladder_cfg={},
    )


def test_spot_pending_stale_cancel_failure_is_reported() -> None:
    pos = new_position_shell("BTCUSDT", profit_take_ladder_cfg={})
    set_pending_buy(
        pos,
        local_order_id="spot_1",
        exchange_order_id="ex_1",
        client_order_id="sa_1",
        quantity=0.01,
        price=50000.0,
        quote_reserved=500.0,
        placed_at="2026-01-01T00:00:00+00:00",
    )
    positions = {"BTCUSDT": pos}
    stats = _spot_process_pending_buys(
        om=_DummyOrderManager(),
        positions=positions,
        ledger=object(),
        budget=_budget(),
        symbols=["BTCUSDT"],
        day_key="2026-01-02",
    )
    assert has_blocking_pending_buy(positions["BTCUSDT"])
    assert stats["stale_local_order"] >= 1
    assert stats["api_error"] >= 1
