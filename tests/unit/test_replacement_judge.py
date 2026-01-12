import pytest

from src.time_series_model.core.constitution.replacement_judge import (
    ReplacementDecision,
    ReplacementInputs,
    decide_replacement_v1,
)


@pytest.mark.unit
def test_replacement_free_slot_allows_replace():
    res = decide_replacement_v1(
        ReplacementInputs(
            has_free_slot=True,
            old_position_id="old",
            old_remaining_rr=1.0,
            old_failure_reasons=[],
            new_signal_id="new",
            new_expected_rr=0.1,
        )
    )
    assert res.decision == ReplacementDecision.REPLACE
    assert res.reason == "free_slot"


@pytest.mark.unit
def test_replacement_requires_guilty_reason():
    res = decide_replacement_v1(
        ReplacementInputs(
            has_free_slot=False,
            old_position_id="old",
            old_remaining_rr=1.0,
            old_failure_reasons=["Some Other Reason"],
            new_signal_id="new",
            new_expected_rr=999.0,
        )
    )
    assert res.decision == ReplacementDecision.REJECT
    assert res.reason == "old_not_guilty"


@pytest.mark.unit
def test_replacement_rr_threshold_single_dimension():
    # old is guilty, but new RR not better enough
    res1 = decide_replacement_v1(
        ReplacementInputs(
            has_free_slot=False,
            old_position_id="old",
            old_remaining_rr=1.0,
            old_failure_reasons=["Time Decay"],
            new_signal_id="new",
            new_expected_rr=1.2,  # <= 1.25
            beta=1.25,
        )
    )
    assert res1.decision == ReplacementDecision.REJECT
    assert res1.reason == "rr_not_better"

    # better enough -> replace
    res2 = decide_replacement_v1(
        ReplacementInputs(
            has_free_slot=False,
            old_position_id="old",
            old_remaining_rr=1.0,
            old_failure_reasons=["Time Decay"],
            new_signal_id="new",
            new_expected_rr=1.26,
            beta=1.25,
        )
    )
    assert res2.decision == ReplacementDecision.REPLACE
    assert res2.reason == "rr_improvement"
