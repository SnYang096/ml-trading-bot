"""Unit tests for dual_position_agree_deadband scalar/series helpers."""

import pandas as pd
import pytest

from src.time_series_model.live.direction_rule_ops import (
    direction_rule_ft_key,
    dual_position_agree_deadband_scalar,
    dual_position_agree_deadband_series,
    is_direction_rule_enabled,
    parse_dual_rule,
    parse_signal_match_position_band_rule,
    parse_single_position_band_rule,
    signal_match_position_band_series,
    single_position_band_scalar,
    single_position_band_series,
)


@pytest.mark.parametrize(
    "v1,v2,eps,expected",
    [
        (0.01, 0.02, 0.005, 1),
        (-0.02, -0.01, 0.005, -1),
        (0.01, 0.002, 0.005, 0),
        (0.01, -0.02, 0.005, 0),
        (float("nan"), 1.0, 0.005, 0),
    ],
)
def test_dual_scalar(v1, v2, eps, expected):
    assert dual_position_agree_deadband_scalar(v1, v2, eps) == expected


def test_dual_scalar_negative_epsilon_clamped():
    assert dual_position_agree_deadband_scalar(0.01, 0.02, -1.0) == 1


def test_dual_series_matches_scalar():
    df = pd.DataFrame(
        {
            "a": [0.01, -0.02, 0.0],
            "b": [0.02, -0.01, 0.5],
        }
    )
    s = dual_position_agree_deadband_series(df, "a", "b", 0.005)
    assert s.iloc[0] == 1.0
    assert s.iloc[1] == -1.0
    assert s.iloc[2] == 0.0


def test_dual_series_missing_columns():
    df = pd.DataFrame({"a": [1.0]})
    s = dual_position_agree_deadband_series(df, "a", "missing", 0.01)
    assert (s == 0).all()


def test_epsilon_grid_values_from_config():
    from scripts.tune_direction_macro_epsilon import epsilon_grid_values_from_config

    assert epsilon_grid_values_from_config({"epsilon_grid": "0.1, 0.2"}) == [0.1, 0.2]
    assert epsilon_grid_values_from_config(
        {"epsilon_min": 0.0, "epsilon_max": 1.0, "epsilon_steps": 3}
    ) == [0.0, 0.5, 1.0]


def test_inner_outer_abs_grid_values_from_config():
    from scripts.tune_direction_macro_epsilon import (
        inner_abs_grid_values_from_config,
        outer_abs_grid_values_from_config,
        pick_best_median_band,
    )

    assert inner_abs_grid_values_from_config({"inner_abs_grid": "0.01, 0.02"}) == [
        0.01,
        0.02,
    ]
    assert outer_abs_grid_values_from_config({}) == []
    assert outer_abs_grid_values_from_config({"outer_abs_grid": "0.4,0.5"}) == [
        0.4,
        0.5,
    ]
    rows = [
        {
            "status": "OK",
            "median_in_direction": 1.0,
            "inner_abs": 0.01,
            "outer_abs": 0.1,
        },
        {
            "status": "OK",
            "median_in_direction": 2.0,
            "inner_abs": 0.02,
            "outer_abs": 0.1,
        },
    ]
    assert pick_best_median_band(rows) == (0.02, 0.1)


def test_single_position_band_scalar():
    assert single_position_band_scalar(0.0, 0.01, 0.1) == 0
    assert single_position_band_scalar(0.02, 0.01, 0.1) == 1
    assert single_position_band_scalar(-0.02, 0.01, 0.1) == -1
    assert single_position_band_scalar(0.15, 0.01, 0.1) == 0


def test_single_position_band_series():
    df = pd.DataFrame({"p": [0.0, 0.02, -0.03, 0.2]})
    s = single_position_band_series(df, "p", 0.01, 0.1)
    assert s.iloc[0] == 0.0
    assert s.iloc[1] == 1.0
    assert s.iloc[2] == -1.0
    assert s.iloc[3] == 0.0


def test_parse_single_position_band_rule():
    r = {
        "method": "single_position_band",
        "feature": "macro_tp_vwap_1200_position",
        "inner_abs": 0.005,
        "outer_abs": 0.05,
    }
    assert parse_single_position_band_rule(r) == (
        "macro_tp_vwap_1200_position",
        0.005,
        0.05,
    )
    assert direction_rule_ft_key(r)[0] == "single_position_band"


def test_parse_dual_rule():
    r = {
        "method": "dual_position_agree_deadband",
        "features": ["x", "y"],
        "epsilon": 0.1,
    }
    assert parse_dual_rule(r) == ("x", "y", 0.1)
    assert parse_dual_rule({"method": "other"}) is None


def test_direction_rule_ft_key_dual_and_sign():
    dual = {
        "method": "dual_position_agree_deadband",
        "features": ["a", "b"],
        "epsilon": 0.05,
    }
    assert direction_rule_ft_key(dual) == (
        "dual_position_agree_deadband",
        "a",
        "b",
        0.05,
    )
    assert direction_rule_ft_key({"feature": "z", "transform": "sign"}) == ("z", "sign")


def test_is_direction_rule_enabled():
    assert is_direction_rule_enabled({}) is True
    assert is_direction_rule_enabled({"enabled": True}) is True
    assert is_direction_rule_enabled({"enabled": False}) is False


def test_compute_direction_series_from_rules_single_position_band():
    from scripts.direction_strict_validation import compute_direction_series_from_rules

    df = pd.DataFrame({"macro_tp_vwap_1200_position": [0.0, 0.02, -0.02, 0.2]})
    rules = [
        {
            "method": "single_position_band",
            "feature": "macro_tp_vwap_1200_position",
            "inner_abs": 0.01,
            "outer_abs": 0.1,
            "id": "b",
        }
    ]
    s = compute_direction_series_from_rules(df, rules)
    assert s.iloc[0] == 0.0
    assert s.iloc[1] == 1.0
    assert s.iloc[2] == -1.0
    assert s.iloc[3] == 0.0


def test_parse_signal_match_position_band_rule():
    r = {
        "method": "signal_match_position_band",
        "id": "x",
        "signal_rules": [
            {"feature": "bpc_breakout_direction", "transform": "raw"},
            {"feature": "macd_atr", "transform": "sign"},
        ],
        "position_band": {
            "feature": "macro_tp_vwap_1200_position",
            "inner_abs": 0.005,
            "outer_abs": 0.95,
        },
    }
    p = parse_signal_match_position_band_rule(r)
    assert p is not None
    assert p["band_feature"] == "macro_tp_vwap_1200_position"
    assert p["inner_abs"] == 0.005
    assert p["outer_abs"] == 0.95
    assert len(p["signal_rules"]) == 2
    assert direction_rule_ft_key(r)[0] == "signal_match_position_band"


def test_signal_match_position_band_series_aligns_long_short():
    df = pd.DataFrame(
        {
            "bpc_breakout_direction": [1.0, 1.0, -1.0, -1.0],
            "macd_atr": [0.0, 0.0, 0.0, 0.0],
            "macro_tp_vwap_1200_position": [0.02, -0.02, -0.02, 0.02],
        }
    )
    s = signal_match_position_band_series(
        df,
        signal_rules=[
            {"feature": "bpc_breakout_direction", "transform": "raw"},
            {"feature": "macd_atr", "transform": "sign"},
        ],
        band_feature="macro_tp_vwap_1200_position",
        inner_abs=0.005,
        outer_abs=0.95,
    )
    assert s.iloc[0] == 1.0
    assert s.iloc[1] == 0.0
    assert s.iloc[2] == -1.0
    assert s.iloc[3] == 0.0


def test_signal_match_position_band_series_macd_fallback():
    df = pd.DataFrame(
        {
            "bpc_breakout_direction": [0.0, 0.0],
            "macd_atr": [0.5, -0.3],
            "macro_tp_vwap_1200_position": [0.02, -0.03],
        }
    )
    s = signal_match_position_band_series(
        df,
        signal_rules=[
            {"feature": "bpc_breakout_direction", "transform": "raw"},
            {"feature": "macd_atr", "transform": "sign"},
        ],
        band_feature="macro_tp_vwap_1200_position",
        inner_abs=0.005,
        outer_abs=0.95,
    )
    assert s.iloc[0] == 1.0
    assert s.iloc[1] == -1.0


def test_compute_direction_series_skips_disabled_then_applies_next():
    from scripts.direction_strict_validation import compute_direction_series_from_rules

    df = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [-2.0, 3.0]})
    rules = [
        {
            "method": "dual_position_agree_deadband",
            "features": ["a", "b"],
            "epsilon": 0.01,
            "enabled": False,
        },
        {"method": "feature_sign", "feature": "c", "transform": "sign"},
    ]
    s = compute_direction_series_from_rules(df, rules)
    assert s.iloc[0] == -1.0
    assert s.iloc[1] == 1.0
