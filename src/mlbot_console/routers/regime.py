from __future__ import annotations

from fastapi import APIRouter

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.regime_ops import (
    fetch_regime_ops_snapshot,
    regime_drift_meta,
)

router = APIRouter(tags=["regime"])


@router.get("/api/trend/regime-ops")
def trend_regime_ops_api() -> dict:
    rows = fetch_regime_ops_snapshot(
        SETTINGS.strategies_root,
        project_root=SETTINGS.repo_root,
    )
    return ok(
        rows,
        meta={
            "count": len(rows),
            "strategies_root": str(SETTINGS.strategies_root),
            **regime_drift_meta(SETTINGS.repo_root),
        },
    )
