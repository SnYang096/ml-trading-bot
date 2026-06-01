"""Paths and tuning knobs for the rolling dashboard package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

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


def experiments_root_path() -> Path:
    """Git-tracked experiment cards under ``config/experiments/``."""
    raw = os.environ.get("ROLLING_DASHBOARD_EXPERIMENTS_ROOT", "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / raw
        return p.resolve()
    return (PROJECT_ROOT / "config" / "experiments").resolve()


def dashboard_visibility() -> DashboardVisibility:
    """Hide legacy research/prod pipeline pages by default (local /rd is the R&D entry).

    - Legacy surfaces require ``ROLLING_DASHBOARD_ENABLE_LEGACY=1``.
    - ``ROLLING_DASHBOARD_HIDE_RESEARCH=1`` / ``HIDE_PROD=1`` still apply when legacy is on.
    """
    legacy = _env_truthy("ROLLING_DASHBOARD_ENABLE_LEGACY")
    return {
        "research": legacy and not _env_truthy("ROLLING_DASHBOARD_HIDE_RESEARCH"),
        "prod": legacy and not _env_truthy("ROLLING_DASHBOARD_HIDE_PROD"),
    }
