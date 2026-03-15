#!/usr/bin/env python3

from scripts.tune_locked_prefilter_thresholds import (
    CaseParams,
    aggregate_case,
    parse_float_list,
)
from scripts.locked_prefilter_utils import apply_locked_thresholds


def test_parse_float_list():
    assert parse_float_list("0.1, 0.2,0.3") == [0.1, 0.2, 0.3]


def test_apply_locked_thresholds_updates_expected_rules():
    raw = {
        "rules": [
            {
                "feature": "fer_signed_efficiency_pct",
                "operator": ">=",
                "value": 0.0,
                "locked": True,
            },
            {
                "feature": "fer_signed_efficiency_pct",
                "operator": "<=",
                "value": 0.35,
                "locked": True,
            },
            {
                "feature": "sr_strength_max",
                "operator": ">=",
                "value": 0.55,
                "locked": True,
            },
            {
                "feature": "dist_to_nearest_sr",
                "operator": ">=",
                "value": -1.2,
                "locked": True,
            },
            {
                "feature": "dist_to_nearest_sr",
                "operator": "<=",
                "value": 1.2,
                "locked": True,
            },
        ]
    }
    params = CaseParams(fer_lower=0.05, fer_upper=0.4, sr_min=0.6, dist_max=1.0)
    out = apply_locked_thresholds(
        raw,
        fer_lower=params.fer_lower,
        fer_upper=params.fer_upper,
        sr_min=params.sr_min,
        dist_max=params.dist_max,
    )
    rules = out["rules"]
    assert rules[0]["value"] == 0.05
    assert rules[1]["value"] == 0.4
    assert rules[2]["value"] == 0.6
    assert rules[3]["value"] == -1.0
    assert rules[4]["value"] == 1.0


def test_aggregate_case_with_trade_penalty():
    case_results = [
        {"metrics": {"sharpe_per_trade": 0.4, "total_trades": 30}},
        {"metrics": {"sharpe_per_trade": 0.2, "total_trades": 50}},
        {"metrics": {"sharpe_per_trade": -0.1, "total_trades": 70}},
    ]
    agg = aggregate_case(case_results, min_trades_target=60, trade_penalty=0.01)
    assert round(agg["median_sharpe"], 6) == 0.2
    assert round(agg["positive_ratio"], 6) == round(2 / 3, 6)
    assert agg["median_trades"] == 50.0
    # score = 0.2 - 0.01 * (60 - 50)
    assert round(agg["score"], 6) == 0.1
