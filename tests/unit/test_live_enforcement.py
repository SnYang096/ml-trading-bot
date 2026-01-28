import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.diagnostics.ood_config import load_ood_config
from src.time_series_model.diagnostics.live_dashboard import (
    build_live_dashboard_payload,
)
from src.order_management.storage import Storage


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
""",
        encoding="utf-8",
    )
    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()
    out_path = tmp_path / "snap.json"
    ood_cfg = load_ood_config("config/ood/ood_config.yaml")
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
        archetype="TREND",
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
            archetype="MEAN",
            drawdown=0.1,
        )


@pytest.mark.unit
def test_live_enforcement_persists_safety_state(tmp_path):
    cy = tmp_path / "constitution.yaml"
    db_path = tmp_path / "safety.db"
    cy.write_text(
        f"""
version: 1
name: "C_SAFETY"
kill_switch:
  enabled: true
  max_dd: 0.5
  daily_loss_limit: 0.01
  weekly_loss_limit: 1.0
  monthly_loss_limit: 1.0
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  cooldown_minutes: 0
  daily_reset_timezone: "UTC"
  kill_on_any_hard_violation: true
safety_state:
  persist_to: "{db_path.as_posix()}"
slots:
  enabled: false
add_position:
  enabled: false
replacement_policy:
  enabled: false
""",
        encoding="utf-8",
    )
    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    with pytest.raises(ConstitutionViolation):
        enforce_before_order(
            executor=ex,
            runtime_state=st,
            position_id="p1",
            symbol="BTCUSDT",
            archetype="TREND",
            execution_strategy="TrendContinuationTC",
            daily_loss=0.05,
            evt_risk_flag=True,
        )

    storage = Storage(str(db_path))
    saved = storage.get_safety_state(state_id="global")
    assert saved is not None
    assert saved.get("halted") is True
    assert "daily_loss_limit" in (saved.get("halt_reason") or [])
    assert (saved.get("last_metrics") or {}).get("evt_risk_flag") is True
