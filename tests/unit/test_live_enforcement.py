import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.diagnostics.ood_config import load_ood_config_v1
from src.time_series_model.diagnostics.live_dashboard import (
    build_live_dashboard_payload,
)


@pytest.mark.unit
def test_live_enforcement_reserves_slot_and_writes_snapshot(tmp_path):
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "C_LIVE"
kill_switch:
  enabled: true
  max_dd: 0.5
  daily_loss_limit: 1.0
  weekly_loss_limit: 1.0
  monthly_loss_limit: 1.0
  kill_on_any_hard_violation: true
slots:
  enabled: true
  slot_count: 1
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/slots.json"
add_position:
  enabled: false
replacement_policy:
  enabled: false
capital_escalation:
  enabled: false
extreme_tail:
  enabled: false
""",
        encoding="utf-8",
    )
    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()
    out_path = tmp_path / "snap.json"
    ood_cfg = load_ood_config_v1("config/ood/ood_config_v1.yaml")
    dash = build_live_dashboard_payload(
        ood_cfg=ood_cfg,
        ood_score=None,
        top_archetype_survival_prob=None,
        active_archetype=None,
        size_cap=None,
        kill_switch_state=None,
    ).as_dict()
    res = enforce_before_order(
        executor=ex,
        runtime_state=st,
        position_id="p1",
        symbol="BTCUSDT",
        mode="TREND",
        execution_strategy="TrendContinuationTC",
        execution_evidence={
            "has_orderflow": True,
            "has_trend_context": True,
            "has_volatility": True,
        },
        drawdown=0.1,
        snapshot_out=str(out_path),
        snapshot_extra={"live_dashboard": dash},
    )
    assert res.ok is True
    assert out_path.exists()

    # Second order should violate slot cap
    st2 = ex.load_runtime_state()
    with pytest.raises(ConstitutionViolation):
        enforce_before_order(
            executor=ex,
            runtime_state=st2,
            position_id="p2",
            symbol="ETHUSDT",
            mode="MEAN",
            drawdown=0.1,
        )
