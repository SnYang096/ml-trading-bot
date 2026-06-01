"""Monitor drift overview API for CMS."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.monitoring_overview import monitoring_overview
from src.monitoring.dashboard import build_monitoring_dashboard, load_monitor_events
from src.monitoring.store import load_monitoring_index

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/index")
def get_index():
    """Latest per-cadence status from results/monitoring/index.json."""
    return ok(load_monitoring_index(SETTINGS.repo_root))


@router.get("/events")
def get_events(
    cadence: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Per-strategy monitor_event rows from rd_registry.sqlite."""
    db = SETTINGS.repo_root / "results" / "rd_registry.sqlite"
    return ok(load_monitor_events(db, cadence=cadence, limit=limit))


@router.get("/overview")
def get_overview():
    """Combined index + recent SQLite events."""
    db = SETTINGS.repo_root / "results" / "rd_registry.sqlite"
    return ok(monitoring_overview(SETTINGS.repo_root, db))


@router.get("/dashboard")
def get_dashboard():
    """CMS cards: OK / ALERT / MISSED per cadence + strategy alerts."""
    db = SETTINGS.repo_root / "results" / "rd_registry.sqlite"
    return ok(build_monitoring_dashboard(SETTINGS.repo_root, db))
