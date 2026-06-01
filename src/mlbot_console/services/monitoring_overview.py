"""Read monitor index + SQLite for CMS drift cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.monitoring.dashboard import build_monitoring_dashboard, load_monitor_events
from src.monitoring.store import load_monitoring_index


def monitoring_overview(repo_root: Path, registry_db: Path) -> Dict[str, Any]:
    index = load_monitoring_index(repo_root)
    events = load_monitor_events(registry_db, limit=100)
    return {
        "index": index,
        "recent_events": events,
        "dashboard": build_monitoring_dashboard(repo_root, registry_db),
        "index_path": str(repo_root / "results" / "monitoring" / "index.json"),
        "registry_db": str(registry_db),
    }
