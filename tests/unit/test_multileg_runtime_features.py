from __future__ import annotations

from src.time_series_model.live.multileg_runtime_features import (
    coerce_trend_direction_for_bus,
    enrich_multileg_runtime_features,
    live_feature_satisfied,
    trend_direction_label,
)


def test_enrich_adds_semantic_chop_and_confidence_alias() -> None:
    features: dict = {"bpc_semantic_chop": 0.42, "trend_confidence": 0.88}
    enrich_multileg_runtime_features(features)
    assert features["semantic_chop"] == 0.42
    assert features["trend_confidence_f"] == 0.88


def test_enrich_box_prefilter_from_box_structure_columns() -> None:
    features: dict = {
        "box_stability_120": 0.9,
        "box_width_pct_120": 0.1,
        "box_touches_hi_120": 6.0,
        "box_touches_lo_120": 6.0,
    }
    enrich_multileg_runtime_features(features)
    assert features["box_prefilter"] == 1.0


def test_coerce_trend_direction_from_string_and_raw() -> None:
    features: dict = {"trend_direction": "DOWN", "trend_direction_raw": 1.0}
    coerce_trend_direction_for_bus(features)
    assert features["trend_direction"] == 1.0

    features2: dict = {"trend_direction": "UP"}
    coerce_trend_direction_for_bus(features2)
    assert features2["trend_direction"] == 1.0

    features3: dict = {"trend_direction_raw": -0.5}
    coerce_trend_direction_for_bus(features3)
    assert features3["trend_direction"] == -1.0


def test_trend_direction_label_decodes_numeric_bus() -> None:
    assert trend_direction_label({"trend_direction": 1.0}) == "UP"
    assert trend_direction_label({"trend_direction": -1.0}) == "DOWN"
    assert trend_direction_label({"trend_direction": "DOWN"}) == "DOWN"


def test_live_feature_satisfied_covers_aliases() -> None:
    features = {"trend_confidence": 0.7, "bpc_semantic_chop": 0.2}
    assert live_feature_satisfied("semantic_chop", features)
    assert live_feature_satisfied("trend_confidence_f", features)
    assert live_feature_satisfied("trend_direction", {"trend_direction_raw": 1.0})
