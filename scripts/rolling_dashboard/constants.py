"""Paths and tuning knobs for the rolling dashboard package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]

# Fewer DOM nodes + fewer `_ledger_summary` reads per request (分页卡片接口).
DASHBOARD_CARD_PAGE_DEFAULT = 80

DASHBOARD_ASSET_PREFIX = "/__dashboard__"


class DashboardVisibility(TypedDict):
    """Which dashboard surfaces are enabled (can hide research or prod via env)."""

    research: bool
    prod: bool


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def dashboard_api_cache_ttl_s() -> float:
    """API / 卡片片段内存缓存秒数。``ROLLING_DASHBOARD_CACHE_TTL_S``，默认 15；``0`` 关闭。

    删除目录成功后会整包清空缓存，避免列表与统计滞后过久。
    """
    raw = os.environ.get("ROLLING_DASHBOARD_CACHE_TTL_S", "15").strip()
    try:
        v = float(raw)
    except ValueError:
        return 15.0
    return max(0.0, min(v, 600.0))


def dashboard_visibility() -> DashboardVisibility:
    """Hide one of the two pipeline pages for ops / UI clutter.

    - ``ROLLING_DASHBOARD_HIDE_RESEARCH=1`` — disable ``/dashboard/research`` (404).
    - ``ROLLING_DASHBOARD_HIDE_PROD=1`` — disable ``/dashboard/prod`` (404).

    If both are set, only ``/dashboard`` hub returns 503 when neither route exists.
    """
    return {
        "research": not _env_truthy("ROLLING_DASHBOARD_HIDE_RESEARCH"),
        "prod": not _env_truthy("ROLLING_DASHBOARD_HIDE_PROD"),
    }
