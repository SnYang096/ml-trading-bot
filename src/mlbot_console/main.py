"""ML Trading Bot — read-only business console (Trade Map + ops views)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from mlbot_console.auth import BasicAuthMiddleware
from mlbot_console.config import SETTINGS
from mlbot_console.services.env_bootstrap import load_console_env_files

load_console_env_files(SETTINGS.repo_root)

from mlbot_console.routers import (
    account,
    bus,
    constitution,
    health,
    links,
    monitoring,
    orders,
    regime,
    spot,
    trade_map,
)

app = FastAPI(
    title="MLBot Business Console",
    description="Read-only business CMS: Trade Map, orders, feature bus OHLCV",
    version="0.3.0",
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
app.include_router(account.router)
app.include_router(regime.router)
app.include_router(monitoring.router)
app.include_router(links.router)

_FRONTEND = Path(__file__).resolve().parent / "static"
_DIST = _FRONTEND / "dist"
_SPA_INDEX = _DIST / "index.html"

if _DIST.is_dir():
    app.mount("/static", StaticFiles(directory=str(_DIST)), name="static")

_logger = logging.getLogger(__name__)


def _verify_spa_bundle() -> None:
    """index.html and hashed JS chunks must come from the same vite build."""
    if not _SPA_INDEX.is_file():
        return
    html = _SPA_INDEX.read_text(encoding="utf-8")
    match = re.search(r'src="(/static/assets/[^"]+\.js)"', html)
    if not match:
        _logger.warning("SPA index.html has no JS bundle reference")
        return
    rel = match.group(1).removeprefix("/static/")
    bundle = _DIST / rel
    if not bundle.is_file():
        _logger.error(
            "SPA bundle missing: index.html -> %s not found under %s; "
            "run `make frontend-build` and restart the console process",
            match.group(1),
            _DIST,
        )


_verify_spa_bundle()


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/trade-map")


def _spa_index() -> FileResponse:
    if not _SPA_INDEX.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built. Run: make frontend-build",
        )
    return FileResponse(_SPA_INDEX)


@app.get("/trade-map")
def trade_map_page() -> FileResponse:
    return _spa_index()


@app.get("/orders")
def orders_page() -> FileResponse:
    return _spa_index()


@app.get("/signals")
def signals_page() -> FileResponse:
    return _spa_index()


@app.get("/account")
def account_page() -> FileResponse:
    return _spa_index()


@app.get("/regime")
def regime_page() -> FileResponse:
    return _spa_index()


@app.get("/monitoring")
def monitoring_page() -> FileResponse:
    return _spa_index()


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    """Client-side routes (React Router) — must be registered after /api routers."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return _spa_index()
