from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipelines" / "pcm_orchestrate_2h.yaml"


@dataclass
class PipelineContext:
    cfg: Dict[str, Any]
    dry_run: bool
    project_root: Path = PROJECT_ROOT
