from __future__ import annotations

from typing import Callable, Dict


def _noop(*args, **kwargs):
    return None


# Stage registry scaffold for future unification.
STAGES: Dict[str, Callable] = {
    "prefilter": _noop,
    "gate": _noop,
    "entry_filter": _noop,
    "execution_opt": _noop,
    "event_backtest": _noop,
    "fast_month": _noop,
    "rolling_sim": _noop,
    "pcm_joint": _noop,
}
