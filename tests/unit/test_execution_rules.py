from src.time_series_model.live.execution_rules import (
    ExecutionRules,
    apply_execution_rules,
)


def test_execution_rules_veto_on_missing_keys() -> None:
    rules = ExecutionRules(
        required_keys_by_archetype={"TrendExpansionTE": ["need_a", "need_b"]}
    )
    ok, reasons = apply_execution_rules(
        rules=rules, archetype_name="TrendExpansionTE", features={"need_a": 1}
    )
    assert not ok
    assert reasons and "exec_rules_missing_keys" in reasons[0]


def test_execution_rules_allow_when_present() -> None:
    rules = ExecutionRules(required_keys_by_archetype={"TrendExpansionTE": ["need_a"]})
    ok, reasons = apply_execution_rules(
        rules=rules, archetype_name="TrendExpansionTE", features={"need_a": 1}
    )
    assert ok
    assert reasons == []
