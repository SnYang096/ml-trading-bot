"""rolling_sim CLI --month filtering preserves global cadence indices."""

import pytest

from scripts.pipeline.config import rolling_sim_iteration_schedule


def test_rolling_sim_iteration_schedule_all_months_equals_enumerate():
    full = ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert rolling_sim_iteration_schedule(full, None) == [
        (0, "2024-01"),
        (1, "2024-02"),
        (2, "2024-03"),
        (3, "2024-04"),
    ]


def test_rolling_sim_iteration_schedule_preserves_global_index_for_filters():
    full = ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert rolling_sim_iteration_schedule(full, ["2024-04", "2024-03"]) == [
        (3, "2024-04"),
        (2, "2024-03"),
    ]


def test_rolling_sim_iteration_schedule_dedupes_requested():
    full = ["2024-01", "2024-02"]
    assert rolling_sim_iteration_schedule(full, ["2024-02", "2024-02"]) == [
        (1, "2024-02")
    ]


def test_rolling_sim_iteration_schedule_rejects_unknown_month():
    full = ["2024-01", "2024-02"]
    with pytest.raises(ValueError, match="不在 holdout"):
        rolling_sim_iteration_schedule(full, ["2024-12"])
