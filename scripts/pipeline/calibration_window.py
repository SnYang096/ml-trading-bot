"""Rolling calibration window helpers (month token + calibration_months).

Extracted from ``auto_research_pipeline`` for reuse by PCM cutoff logic without
import cycles.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timedelta
from typing import Dict, Tuple


def parse_month_token(month_token: str) -> Tuple[int, int]:
    """Parse YYYY-MM month token."""
    token = str(month_token or "").strip()
    try:
        dt = datetime.strptime(token, "%Y-%m")
        return dt.year, dt.month
    except Exception as exc:
        raise ValueError(f"非法月份格式: {month_token}, 期望 YYYY-MM") from exc


def month_token_to_range(month_token: str) -> Tuple[str, str]:
    """Convert YYYY-MM to inclusive start/end date strings."""
    y, m = parse_month_token(month_token)
    last_day = monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last_day:02d}"


def add_months(date_str: str, months: int) -> str:
    """Shift YYYY-MM-DD by month delta; clamps day to valid month end."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y = dt.year
    m = dt.month + int(months)
    while m > 12:
        y += 1
        m -= 12
    while m <= 0:
        y -= 1
        m += 12
    d = min(dt.day, monthrange(y, m)[1])
    return f"{y:04d}-{m:02d}-{d:02d}"


def month_start(month_token: str) -> str:
    y, m = parse_month_token(month_token)
    return f"{y:04d}-{m:02d}-01"


def month_prev_end(month_token: str) -> str:
    ms = month_start(month_token)
    prev_month_day = datetime.strptime(ms, "%Y-%m-%d") - timedelta(days=1)
    return prev_month_day.strftime("%Y-%m-%d")


def calib_and_test_windows(
    *,
    month_token: str,
    calibration_months: int,
    step_months: int = 1,
) -> Dict[str, str]:
    """For target month M: calib=[M-k, M-1 end], test spans ``step_months``."""
    step = max(int(step_months or 1), 1)
    test_start, _ = month_token_to_range(month_token)
    test_end_dt = datetime.strptime(
        add_months(test_start, step), "%Y-%m-%d"
    ) - timedelta(days=1)
    test_end = test_end_dt.strftime("%Y-%m-%d")
    calib_end = month_prev_end(month_token)
    calib_start = add_months(test_start, -int(calibration_months))
    return {
        "calib_start": calib_start,
        "calib_end": calib_end,
        "test_start": test_start,
        "test_end": test_end,
    }
