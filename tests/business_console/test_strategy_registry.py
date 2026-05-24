"""Strategy registry + funnel aggregation."""

from __future__ import annotations

from mlbot_console.services.strategy_registry import (
    account_layer_label,
    strategies_for_layer,
    strategy_account_layer,
)
from mlbot_console.services.trend_funnel import aggregate_funnel_by_strategy


def test_strategy_account_layer_mapping():
    assert strategy_account_layer("bpc") == "trend"
    assert strategy_account_layer("spot_accum_simple") == "spot"
    assert strategy_account_layer("chop_grid") == "multi_leg"
    assert account_layer_label("spot") == "A·Spot"


def test_strategies_for_layer():
    assert "bpc" in strategies_for_layer("trend")
    assert "chop_grid" in strategies_for_layer("multi_leg")


def test_aggregate_funnel_by_strategy_filters_layer():
    snaps = [
        {
            "symbol": "ETHUSDT",
            "by_strategy": {
                "bpc": {"regime_passed": 2, "prefilter_passed": 1},
                "chop_grid": {"regime_passed": 5},
            },
        },
        {
            "symbol": "ETHUSDT",
            "by_strategy": {
                "bpc": {"regime_passed": 1},
            },
        },
    ]
    trend = aggregate_funnel_by_strategy(snaps, symbol="ETHUSDT", account_layer="trend")
    assert trend["bpc"]["regime_passed"] == 3
    assert "chop_grid" not in trend
    multileg = aggregate_funnel_by_strategy(
        snaps, symbol="ETHUSDT", account_layer="multi_leg"
    )
    assert multileg["chop_grid"]["regime_passed"] == 5
