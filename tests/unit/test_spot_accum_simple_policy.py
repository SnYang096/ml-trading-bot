"""Unit tests for spot_accum_simple deploy decay and profit ladder acceleration."""

import pytest

from datetime import datetime, timezone

from src.time_series_model.live.spot_accum_simple import (
    deep_bear_allows_buy,
    deploy_decay_multiplier,
    deploy_schedule_allows_new_buy,
    profit_ladder_speed_multiplier,
)


def test_deep_bear_weekly_ema_below_zero():
    policy = {
        "enabled": True,
        "regime_feature": "weekly_ema_200_position",
        "deep_bear_max_position": 0.0,
    }
    assert deep_bear_allows_buy({"weekly_ema_200_position": -0.05}, policy) == (
        True,
        -0.05,
    )
    assert deep_bear_allows_buy({"weekly_ema_200_position": 0.02}, policy) == (
        False,
        0.02,
    )


def test_deploy_decay_tiers():
    cfg = {
        "enabled": True,
        "tiers": [
            {"max_deployed_pct_exclusive": 30.0, "unit_multiplier": 1.0},
            {"max_deployed_pct_exclusive": 60.0, "unit_multiplier": 0.7},
            {"max_deployed_pct_exclusive": 80.0, "unit_multiplier": 0.4},
            {"max_deployed_pct_exclusive": 999.0, "unit_multiplier": 0.2},
        ],
    }
    assert deploy_decay_multiplier(0.0, 1000.0, cfg) == pytest.approx(1.0)
    assert deploy_decay_multiplier(350.0, 1000.0, cfg) == pytest.approx(0.7)
    assert deploy_decay_multiplier(850.0, 1000.0, cfg) == pytest.approx(0.2)


def test_deploy_schedule_london_window():
    schedule = {
        "enabled": True,
        "timezone": "Europe/London",
        "new_order_local_start": "08:00",
        "new_order_local_end": "11:00",
    }
    inside = datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc)
    # 09:30 UTC = 09:30 London (winter)
    ok, _ = deploy_schedule_allows_new_buy(inside, schedule)
    assert ok
    outside = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
    ok2, reason = deploy_schedule_allows_new_buy(outside, schedule)
    assert not ok2
    assert "outside_deploy_window" in reason


def test_profit_ladder_power_acceleration():
    accel = {"type": "power", "exponent": 0.75, "max_speed_multiplier": 4.0}
    assert profit_ladder_speed_multiplier(4.0, 5.0, accel) == 0.0
    assert profit_ladder_speed_multiplier(5.0, 5.0, accel) == pytest.approx(1.0)
    assert profit_ladder_speed_multiplier(10.0, 5.0, accel) == pytest.approx(2.0**0.75)
    assert profit_ladder_speed_multiplier(100.0, 5.0, accel) == pytest.approx(4.0)
