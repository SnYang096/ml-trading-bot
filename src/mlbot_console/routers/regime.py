from __future__ import annotations

from fastapi import APIRouter, Query

from mlbot_console.config import SETTINGS
from mlbot_console.responses import ok
from mlbot_console.services.rebalance_advisor import build_regime_cockpit
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


@router.get("/api/regime/cockpit")
def regime_cockpit_api(
    symbol: str = Query("BTCUSDT", description="Primary reference symbol"),
    window_days: int = Query(7, ge=1, le=90),
) -> dict:
    payload = build_regime_cockpit(
        strategies_root=SETTINGS.strategies_root,
        project_root=SETTINGS.repo_root,
        feature_bus_root=SETTINGS.feature_bus_root,
        symbol=symbol,
        window_days=window_days,
    )
    from src.monitoring.rebalance_cockpit_run import load_latest_rebalance_event

    latest = load_latest_rebalance_event(SETTINGS.repo_root / "results" / "rd_registry.sqlite")
    if latest and latest.get("detail_json"):
        try:
            import json as _json

            latest["detail"] = _json.loads(str(latest["detail_json"]))
        except _json.JSONDecodeError:
            latest["detail"] = None
    payload["last_scheduled"] = latest
    return ok(
        payload,
        meta={
            "strategies_root": str(SETTINGS.strategies_root),
            "feature_bus_root": str(SETTINGS.feature_bus_root),
            "symbol": symbol.upper(),
            "window_days": window_days,
        },
    )
