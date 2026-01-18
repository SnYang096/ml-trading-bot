from src.time_series_model.diagnostics.live_dashboard import (
    build_live_dashboard_payload,
    validate_live_dashboard_payload,
)
from src.time_series_model.diagnostics.ood_config import load_ood_config


def test_live_dashboard_payload_has_required_keys() -> None:
    cfg = load_ood_config("config/ood/ood_config.yaml")
    m = build_live_dashboard_payload(
        ood_cfg=cfg,
        ood_score=0.2,
        top_archetype_survival_prob=0.8,
        active_archetype="TrendContinuationTC",
        size_cap=0.5,
        kill_switch_state="TRADEABLE",
    )
    d = m.as_dict()
    ok, reasons = validate_live_dashboard_payload(ood_cfg=cfg, payload=d)
    assert ok, reasons
    for k in cfg.dashboard_keys:
        assert k in d
