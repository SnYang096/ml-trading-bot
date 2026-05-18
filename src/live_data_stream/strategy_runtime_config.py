"""Shared strategy disk layout + meta.yaml timeframe resolution (live + feature bus)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.config.strategy_layout import resolve_strategy_package_under_root

logger = logging.getLogger(__name__)


def load_strategy_timeframe(strategies_root: str, strategy_name: str) -> str:
    """从 meta.yaml 读取策略的 timeframe，缺失时 fallback 到 240T。"""
    import yaml

    pkg = resolve_strategy_package_under_root(
        Path(strategies_root), strategy_name, allow_bad_candidates=False
    )
    meta_path = pkg / "meta.yaml"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        tf = (meta.get("strategy") or {}).get("timeframe")
        if tf:
            return str(tf)
    except FileNotFoundError:
        logger.warning("meta.yaml 不存在: %s，使用默认 240T", meta_path)
    except Exception as e:
        logger.warning("读取 meta.yaml 失败: %s — %s，使用默认 240T", meta_path, e)
    return "240T"


def me_enabled_in_allowlist(enabled_archetypes: List[str]) -> bool:
    """宪法 enabled_archetypes 中含 me / me-long / me-short 或 ``me-*`` 即启用 ME 包。"""
    for raw in enabled_archetypes:
        a = str(raw).lower().strip()
        if a in {"me", "me-long", "me-short"}:
            return True
        if a.startswith(("me-long-", "me-short-", "me-")):
            return True
    return False
