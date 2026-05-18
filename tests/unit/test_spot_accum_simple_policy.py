"""Unit tests for spot_accum_simple deploy decay and profit ladder acceleration."""

import pytest

from src.time_series_model.live.spot_accum_simple import (
    deep_bear_allows_buy,
    deploy_decay_multiplier,
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


def test_profit_ladder_power_acceleration():
    accel = {"type": "power", "exponent": 0.75, "max_speed_multiplier": 4.0}
    assert profit_ladder_speed_multiplier(4.0, 5.0, accel) == 0.0
    assert profit_ladder_speed_multiplier(5.0, 5.0, accel) == pytest.approx(1.0)
    assert profit_ladder_speed_multiplier(10.0, 5.0, accel) == pytest.approx(2.0**0.75)
    assert profit_ladder_speed_multiplier(100.0, 5.0, accel) == pytest.approx(4.0)
