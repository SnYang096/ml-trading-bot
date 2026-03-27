from __future__ import annotations

from typing import Any, Dict


def run_event_backtest_step(*args, **kwargs) -> Dict[str, Any]:
    from scripts import auto_research_pipeline as legacy

    return legacy._run_event_backtest_step(*args, **kwargs)


def run_event_execution_opt_only(*args, **kwargs) -> Dict[str, Any]:
    from scripts import auto_research_pipeline as legacy

    return legacy._run_event_execution_opt_only(*args, **kwargs)


def run_pcm_joint_backtest(*args, **kwargs):
    from scripts import auto_research_pipeline as legacy

    return legacy._run_pcm_joint_backtest(*args, **kwargs)
