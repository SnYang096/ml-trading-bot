import pytest

from src.time_series_model.live.timers import compute_next_aligned_delay_seconds


@pytest.mark.unit
def test_compute_next_aligned_delay_seconds_basic():
    # 00:00:00 -> next 10-min boundary is 00:10:00
    assert compute_next_aligned_delay_seconds(now_ns=0, interval_minutes=10) == 600

    # 00:09:59 -> next 10-min boundary is 00:10:00 (1s)
    now_ns = (9 * 60 + 59) * 1_000_000_000
    assert compute_next_aligned_delay_seconds(now_ns=now_ns, interval_minutes=10) == 1

    # 00:10:00 -> next 10-min boundary is 00:20:00
    now_ns = (10 * 60) * 1_000_000_000
    assert compute_next_aligned_delay_seconds(now_ns=now_ns, interval_minutes=10) == 600


@pytest.mark.unit
def test_compute_next_aligned_delay_seconds_invalid():
    with pytest.raises(ValueError):
        compute_next_aligned_delay_seconds(now_ns=0, interval_minutes=0)
