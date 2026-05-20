from __future__ import annotations

from fastapi import APIRouter

from app.config import SETTINGS
from app.responses import ok

router = APIRouter(tags=["links"])


@router.get("/api/links")
def external_links() -> dict:
    links = [
        {"id": "grafana", "label": "Grafana", "url": SETTINGS.grafana_url},
    ]
    if SETTINGS.rolling_backtest_url:
        links.append(
            {
                "id": "rolling_backtest",
                "label": "Rolling backtest maps",
                "url": SETTINGS.rolling_backtest_url,
            }
        )
    return ok({"links": links})
