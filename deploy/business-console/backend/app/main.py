"""ML Trading Bot — read-only business console (Trade Map + ops views)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import BasicAuthMiddleware
from app.config import SETTINGS
from app.routers import bus, constitution, health, links, orders, spot, trade_map

app = FastAPI(
    title="MLBot Business Console",
    description="Read-only business CMS: Trade Map, orders, feature bus OHLCV",
    version="0.2.0",
)

if SETTINGS.basic_auth_user and SETTINGS.basic_auth_password:
    app.add_middleware(
        BasicAuthMiddleware,
        user=SETTINGS.basic_auth_user,
        password=SETTINGS.basic_auth_password,
    )

app.include_router(health.router)
app.include_router(constitution.router)
app.include_router(bus.router)
app.include_router(trade_map.router)
app.include_router(spot.router)
app.include_router(orders.router)
app.include_router(links.router)

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_FRONTEND / "index.html")


@app.get("/trade-map")
def trade_map_page() -> FileResponse:
    return FileResponse(_FRONTEND / "index.html")
