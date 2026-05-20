from __future__ import annotations

from fastapi import APIRouter

from app.config import SETTINGS
from app.responses import ok
from app.services import ohlcv_reader
from app.services.db import db_status
from app.services.universe import load_universe_symbols

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    return ok(
        {
            "service": "mlbot-business-console",
            "symbols_count": len(symbols),
            "paths": {
                "feature_bus_root": str(SETTINGS.feature_bus_root),
                "live_data_root": str(SETTINGS.live_data_root),
                "engine_data_root": str(SETTINGS.engine_data_root),
            },
            "databases": {
                "trend_order": db_status(SETTINGS.trend_order_db),
                "live_monitor": db_status(SETTINGS.live_monitor_db),
                "spot_order": db_status(SETTINGS.spot_order_db),
                "spot_ledger": db_status(SETTINGS.spot_ledger_db),
                "multi_leg": db_status(SETTINGS.multi_leg_db),
            },
        }
    )


@router.get("/api/overview")
def overview() -> dict:
    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    latest: dict = {}
    for sym in symbols[:12]:
        meta = ohlcv_reader.latest_bar_meta(SETTINGS.feature_bus_root, sym)
        if meta:
            latest[sym] = meta
    return ok(
        {
            "symbols": symbols,
            "latest_bars": latest,
            "poll_seconds": SETTINGS.map_poll_seconds,
        }
    )
