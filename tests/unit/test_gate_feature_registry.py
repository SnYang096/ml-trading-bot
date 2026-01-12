import pytest

from src.time_series_model.gating.feature_registry import (
    FeatureMeta,
    validate_features_allowed,
    validate_gate_feature_registry,
)


@pytest.mark.unit
def test_gate_feature_registry_validation_and_allowlist():
    reg = {
        "x": FeatureMeta(
            feature_name="x",
            semantic_group="commitment",
            time_scale="5s",
            applicable_scope="trend",
            allowed_layers=["gate"],
            drift_sensitivity="high",
        )
    }
    errs = validate_gate_feature_registry(reg)
    assert errs == []

    # Allowed in gate
    e1 = validate_features_allowed(registry=reg, requested_features=["x"], layer="gate")
    assert e1 == []

    # Not allowed in model
    e2 = validate_features_allowed(
        registry=reg, requested_features=["x"], layer="model"
    )
    assert e2 and "not allowed" in e2[0]

    # Missing registry is hard error
    e3 = validate_features_allowed(
        registry=reg, requested_features=["missing"], layer="gate"
    )
    assert e3 and "no metadata" in e3[0]
