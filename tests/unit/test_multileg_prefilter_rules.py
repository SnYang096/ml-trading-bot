import pandas as pd

from scripts.pipeline.multileg_prefilter_rules import apply_prefilter_rules


def test_apply_simple_rule():
    df = pd.DataFrame({"atr_percentile": [0.3, 0.9]})
    rules = [{"feature": "atr_percentile", "operator": "<=", "value": 0.8}]
    mask = apply_prefilter_rules(df, rules)
    assert mask.tolist() == [True, False]


def test_flat_box_pos_range_matches_all_of():
    df = pd.DataFrame({"box_pos_60": [0.35, 0.50, 0.65, 0.70]})
    flat = [
        {"feature": "box_pos_60", "operator": ">=", "value": 0.40},
        {"feature": "box_pos_60", "operator": "<=", "value": 0.60},
    ]
    nested = [
        {
            "all_of": [
                {"feature": "box_pos_60", "operator": ">=", "value": 0.40},
                {"feature": "box_pos_60", "operator": "<=", "value": 0.60},
            ]
        }
    ]
    assert (
        apply_prefilter_rules(df, flat).tolist()
        == apply_prefilter_rules(df, nested).tolist()
    )


def test_apply_any_of_and_all_of_rules():
    df = pd.DataFrame(
        {
            "ema_1200_position": [-0.05, 0.0, 0.04],
            "volatility_regime": [0.8, 2.0, 1.2],
        }
    )
    rules = [
        {
            "any_of": [
                {"feature": "ema_1200_position", "operator": "<=", "value": -0.03},
                {"feature": "ema_1200_position", "operator": ">=", "value": 0.03},
            ]
        },
        {
            "all_of": [
                {"feature": "volatility_regime", "operator": ">=", "value": 0.6},
                {"feature": "volatility_regime", "operator": "<=", "value": 1.6},
            ]
        },
    ]
    mask = apply_prefilter_rules(df, rules)
    assert mask.tolist() == [True, False, True]
