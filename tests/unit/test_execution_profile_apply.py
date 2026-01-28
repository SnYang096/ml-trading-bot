from datetime import datetime, timedelta, timezone

from src.time_series_model.live.execution_profile_apply import (
    compute_rr_prices,
    compute_trailing_stop,
    holding_expired,
)


def test_compute_rr_prices_long() -> None:
    sl, tp = compute_rr_prices(
        side="LONG", entry_price=100.0, atr=2.0, stop_loss_r=1.0, take_profit_r=2.5
    )
    assert sl == 98.0
    assert tp == 105.0


def test_compute_rr_prices_short() -> None:
    sl, tp = compute_rr_prices(
        side="SHORT", entry_price=100.0, atr=2.0, stop_loss_r=1.0, take_profit_r=2.5
    )
    assert sl == 102.0
    assert tp == 95.0


def test_compute_trailing_stop_long() -> None:
    stop = compute_trailing_stop(
        side="LONG", current_price=110.0, atr=2.0, trailing_atr=0.5
    )
    assert stop == 109.0


def test_compute_trailing_stop_short() -> None:
    stop = compute_trailing_stop(
        side="SHORT", current_price=90.0, atr=2.0, trailing_atr=0.5
    )
    assert stop == 91.0


def test_holding_expired() -> None:
    entry = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = entry + timedelta(hours=8)
    assert holding_expired(
        entry_time=entry, now=now, max_holding_bars=2, bar_minutes=240
    )
