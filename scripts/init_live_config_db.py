#!/usr/bin/env python3
"""
初始化实盘配置到数据库（显式列）

从 config/live/live_config_defaults.yaml 读取默认值。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.order_management.storage import Storage
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def init_live_config(
    *,
    db_path: str,
    archetype_registry_path: str = "config/nnmultihead/execution_archetypes.yaml",
    defaults_path: str = "config/live/live_config_defaults.yaml",
) -> None:
    """初始化数据库配置（若已有则不覆盖）"""
    storage = Storage(db_path=db_path)

    existing = storage.get_live_config()
    if existing is not None:
        print("⏭️  live_config already exists, skipping")
        return

    defaults_file = Path(defaults_path)
    if not defaults_file.exists():
        raise FileNotFoundError(f"Defaults file not found: {defaults_path}")

    defaults = yaml.safe_load(defaults_file.read_text(encoding="utf-8")) or {}

    # enabled_archetypes
    enabled_raw = defaults.get("enabled_archetypes", "ALL")
    if isinstance(enabled_raw, str) and enabled_raw.strip().upper() == "ALL":
        arches = load_execution_archetypes_registry(archetype_registry_path)
        enabled_archetypes = list(arches.keys())
    elif isinstance(enabled_raw, list):
        enabled_archetypes = [str(x) for x in enabled_raw]
    else:
        enabled_archetypes = []

    # size_multipliers: ensure all enabled archetypes have a multiplier
    size_multipliers = defaults.get("size_multipliers") or {}
    # Fill missing archetypes with default 1.0
    for arch in enabled_archetypes:
        if arch not in size_multipliers:
            size_multipliers[arch] = 1.0

    # window + min interval
    window_minutes = int(defaults.get("window_minutes", 15))
    min_order_interval_minutes = int(defaults.get("min_order_interval_minutes", 10))

    # nnmultihead_inference
    nnmultihead_inference = defaults.get("nnmultihead_inference") or {}

    storage.upsert_live_config(
        enabled_archetypes=enabled_archetypes,
        size_multipliers=size_multipliers,
        window_minutes=window_minutes,
        min_order_interval_minutes=min_order_interval_minutes,
        nnmultihead_inference=nnmultihead_inference,
        updated_by=os.getenv("MLBOT_USER"),
    )
    print("✅ Live config initialization complete!")


if __name__ == "__main__":
    db_path = os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db")
    archetype_registry_path = os.getenv(
        "MLBOT_ARCHETYPE_REGISTRY",
        "config/nnmultihead/execution_archetypes.yaml",
    )
    defaults_path = os.getenv(
        "MLBOT_LIVE_CONFIG_DEFAULTS_YAML",
        "config/live/live_config_defaults.yaml",
    )

    init_live_config(
        db_path=db_path,
        archetype_registry_path=archetype_registry_path,
        defaults_path=defaults_path,
    )
