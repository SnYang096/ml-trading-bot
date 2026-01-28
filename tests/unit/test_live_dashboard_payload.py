from src.time_series_model.diagnostics.live_dashboard import (
    build_live_dashboard_payload,
    validate_live_dashboard_payload,
)


def test_live_dashboard_payload_has_required_keys() -> None:
    m = build_live_dashboard_payload(
        active_archetype="TrendContinuationTC",
        size_cap=0.5,
        kill_switch_state="TRADEABLE",
        drawdown=0.1,
        daily_loss=0.02,
    )
    d = m.as_dict()
    ok, reasons = validate_live_dashboard_payload(payload=d)
    assert ok, reasons
    assert "active_archetype" in d
    assert "kill_switch_state" in d
    assert "drawdown" in d
