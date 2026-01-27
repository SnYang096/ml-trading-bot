import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation


def _write_constitution(tmp_path):
    p = tmp_path / "constitution.yaml"
    p.write_text(
        f"""
version: 1
name: "C_TEST"
kill_switch:
  enabled: false
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/slots.json"
add_position:
  enabled: true
  state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/add_position.json"
replacement_policy:
  enabled: true
  auditability:
    log_every_replacement: true
    log_path: "{tmp_path.as_posix()}/logs/replacements/"
    required_fields: ["closed_position_id","close_reason","new_position_signal","expected_rr_improvement","timestamp"]
capital_escalation:
  enabled: true
  auto_degradation:
    on_exit:
      - action: "lock_new_escalation"
        duration_days: 30
    state_persistence:
      persist_to: "{tmp_path.as_posix()}/state/escalation.json"
""",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.unit
def test_slots_state_roundtrip(tmp_path):
    cy = _write_constitution(tmp_path)
    ex = ConstitutionExecutor(constitution_yaml=cy)
    st = ex.load_runtime_state()
    assert st.slots.active_count() == 0

    ex.reserve_slot(
        st=st,
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TREND",
        opened_at="2026-01-01T00:00:00+00:00",
    )
    ex.reserve_slot(
        st=st,
        position_id="p2",
        symbol="ETHUSDT",
        archetype="MEAN",
        opened_at="2026-01-01T00:00:00+00:00",
    )
    assert st.slots.active_count() == 2

    ex.save_runtime_state(st)
    ex2 = ConstitutionExecutor(constitution_yaml=cy)
    st2 = ex2.load_runtime_state()
    assert st2.slots.active_count() == 2

    with pytest.raises(ConstitutionViolation):
        ex2.reserve_slot(st=st2, position_id="p3", symbol="SOLUSDT", archetype="TREND")


@pytest.mark.unit
def test_escalation_lockout(tmp_path):
    cy = _write_constitution(tmp_path)
    ex = ConstitutionExecutor(constitution_yaml=cy)
    st = ex.load_runtime_state()
    assert ex.is_escalation_locked(st=st) is False

    ex.record_escalation_exit(
        st=st,
        exit_reason="test",
        equity_at_exit=1.0,
        exited_at="2026-01-01T00:00:00+00:00",
    )
    # Locked for 30 days per constitution
    assert ex.is_escalation_locked(st=st, now_iso="2026-01-10T00:00:00+00:00") is True
    assert ex.is_escalation_locked(st=st, now_iso="2026-02-15T00:00:00+00:00") is False
