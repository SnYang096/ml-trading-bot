import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation


@pytest.mark.unit
def test_extreme_tail_caps(tmp_path):
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "C_XT"
kill_switch: {{enabled: false}}
slots: {{enabled: false}}
replacement_policy: {{enabled: false}}
add_position: {{enabled: false}}
capital_escalation: {{enabled: false}}
extreme_tail:
  enabled: true
  max_budget: 0.02
  hard_limits:
    single_event_max_usd: 1000
    annual_total_ratio: 0.02
  state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/extreme_tail.json"
""",
        encoding="utf-8",
    )
    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    # Too large single entry
    with pytest.raises(ConstitutionViolation):
        ex.validate_extreme_tail_entry(
            equity_usd=10_000, entry_usd=2000, st=st, year=2026
        )

    # Ok entry (budget cap: 2% of 10k = 200)
    ex.validate_extreme_tail_entry(equity_usd=10_000, entry_usd=100, st=st, year=2026)
    ex.record_extreme_tail_entry(st=st, entry_usd=100, position_id="p1", year=2026)

    # Annual cap: equity 10k, annual 2% => 200; already used 100 => next can be at most 100 more
    with pytest.raises(ConstitutionViolation):
        ex.validate_extreme_tail_entry(
            equity_usd=10_000, entry_usd=150, st=st, year=2026
        )
