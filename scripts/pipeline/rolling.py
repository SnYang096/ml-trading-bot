from __future__ import annotations

from typing import Any, Dict


def run_fast_month_stage(*args, **kwargs) -> Dict[str, Any]:
    from scripts import auto_research_pipeline as legacy

    return legacy._run_fast_month_stage(*args, **kwargs)


def run_slow_structure_snapshot_for_month(*args, **kwargs) -> Dict[str, Any]:
    from scripts import auto_research_pipeline as legacy

    return legacy._run_slow_structure_snapshot_for_month(*args, **kwargs)
