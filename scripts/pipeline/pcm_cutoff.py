"""PCM Prefilter/Gate/EntryFilter cutoff date resolution (ADR §14.3)."""

from __future__ import annotations

from typing import Optional

from scripts.pipeline.calibration_window import calib_and_test_windows


def resolve_pcm_cutoff_date(
    policy: str,
    *,
    month_token: Optional[str],
    calibration_months: int,
    holdout_start: str,
    test_start: str,
) -> Optional[str]:
    """Return YYYY-MM-DD for ``--cutoff-date``, or ``None`` to omit the flag.

    - ``static_holdout`` (default): use global Val/Test split — cutoff at
      ``test_start`` when it differs from ``holdout_start`` (unchanged legacy).
    - ``walk_forward_monthly``: when ``month_token`` is set, cutoff at the end
      of the rolling calibration window for that month (``calib_end``), aligned
      with ``fast_month`` / ``calib_and_test_windows``. Without ``month_token``,
      falls back to the static rule so full-pipeline CLI runs stay stable.
    """
    pol = (policy or "static_holdout").strip().lower()
    if pol == "walk_forward_monthly" and month_token:
        wins = calib_and_test_windows(
            month_token=str(month_token).strip(),
            calibration_months=int(calibration_months),
        )
        return str(wins["calib_end"])
    if test_start and holdout_start and test_start != holdout_start:
        return test_start
    return None
