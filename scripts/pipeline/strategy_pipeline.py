from __future__ import annotations

from typing import Any, Dict


def run_strategy_pipeline(*args, **kwargs) -> Dict[str, Any]:
    # Runtime bridge to keep backward compatibility during staged migration.
    from scripts import auto_research_pipeline as legacy

    return legacy.run_strategy_pipeline(*args, **kwargs)
