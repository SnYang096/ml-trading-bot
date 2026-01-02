import yaml

from src.features.normalization.feature_contract import (
    collect_feature_normalization_meta,
    validate_feature_dependencies_normalization,
)


def test_feature_dependencies_normalization_contract_has_no_missing_methods():
    """
    Code contract: every output column must have an explicit normalization method.

    This test prevents 'FEATURE_CATALOG says normalized' drifting away from actual
    config/code reality.
    """
    with open("config/feature_dependencies.yaml", "r", encoding="utf-8") as f:
        deps = yaml.safe_load(f)

    report = validate_feature_dependencies_normalization(deps, mode="error")
    assert report["ok"]


def test_feature_dependencies_normalization_contract_has_no_raw_columns_global():
    """
    Stronger code contract: global feature registry must have 0 raw output columns.
    This is the repo-wide prerequisite for multi-asset training and NN stability.
    """
    with open("config/feature_dependencies.yaml", "r", encoding="utf-8") as f:
        deps = yaml.safe_load(f)

    rows = collect_feature_normalization_meta(deps, only_features=None)
    raw = [r for r in rows if r["method"] == "raw"]
    assert raw == []
