from src.time_series_model.live.live_feature_contract import (
    LiveFeatureContractV1,
    validate_live_features_v1,
)


def test_live_feature_contract_requires_orderflow_keys() -> None:
    c = LiveFeatureContractV1(
        required_keys_any=["vpin", "imbalance", "total_vol"],
        required_pred_keys=["pred_dir_prob"],
        on_violation="NO_TRADE",
    )
    ok, reasons = validate_live_features_v1(
        contract=c, features={"vpin": 0.1}, nn_inference_enabled=False
    )
    assert not ok
    assert any("missing_required_keys_any" in r for r in reasons)


def test_live_feature_contract_requires_pred_keys_when_inference_enabled() -> None:
    c = LiveFeatureContractV1(
        required_keys_any=["vpin"],
        required_pred_keys=["pred_dir_prob", "pred_mfe_atr"],
        on_violation="NO_TRADE",
    )
    ok, reasons = validate_live_features_v1(
        contract=c,
        features={"vpin": 0.1, "pred_dir_prob": 0.5},
        nn_inference_enabled=True,
    )
    assert not ok
    assert any("missing_required_pred_keys" in r for r in reasons)
