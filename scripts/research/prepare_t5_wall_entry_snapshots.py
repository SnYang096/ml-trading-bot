#!/usr/bin/env python3
"""Materialize variant trees for 20260615_t5_wall_entry_validate."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SRC_TPC = REPO / "config" / "strategies" / "tpc"
EXP = REPO / "config" / "experiments" / "20260615_t5_wall_entry_validate" / "variants"

_WALL_LONG_2 = {
    "id": "t5_wall_near_support_long",
    "enabled": True,
    "direction": "long",
    "description": "T5α 多单：近端买盘墙 ≤2 ATR（Phase 1c scan |z|=3.1）",
    "conditions": [
        {"feature": "wall_nearest_dist_atr", "operator": "<=", "value": 2.0}
    ],
    "tier": "EXPERIMENT",
}
_WALL_SHORT_25 = {
    "id": "t5_wall_near_resistance_short",
    "enabled": True,
    "direction": "short",
    "description": "T5α 空单：近端卖盘墙 ≤2.5 ATR（Phase 1c scan |z|=2.8）",
    "conditions": [
        {"feature": "wall_nearest_dist_atr", "operator": "<=", "value": 2.5}
    ],
    "tier": "EXPERIMENT",
}
_WALL_BOTH_2 = {
    "id": "t5_wall_near_2atr",
    "enabled": True,
    "description": "T5α 对称：近端墙 ≤2 ATR（对照）",
    "conditions": [
        {"feature": "wall_nearest_dist_atr", "operator": "<=", "value": 2.0}
    ],
    "tier": "EXPERIMENT",
}

SNAPSHOTS: dict[str, dict[str, object]] = {
    "tpc_wall_W1_bull2_strategies": {
        "entry_filters_patch": {
            "combination_mode": "and",
            "filters_append": [_WALL_LONG_2],
        },
    },
    "tpc_wall_W2_asym_strategies": {
        "entry_filters_patch": {
            "combination_mode": "and",
            "filters_append": [_WALL_LONG_2, _WALL_SHORT_25],
        },
    },
    "tpc_wall_W4_sym2_strategies": {
        "entry_filters_patch": {
            "combination_mode": "and",
            "filters_append": [_WALL_BOTH_2],
        },
    },
}

_FEATURE_APPEND = [
    "ema_1200_value_f",
    "wall_features_f",
]


def _copy_tpc_snapshot(name: str) -> Path:
    """Copy prod ``tpc/`` only — event_backtest reads ``strategies_root/tpc/*``."""
    dst = EXP / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC_TPC, dst / "tpc")
    return dst


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _patch_entry_filters(tree: Path, patch: dict) -> None:
    path = tree / "tpc" / "archetypes" / "entry_filters.yaml"
    data = _load_yaml(path)
    if "combination_mode" in patch:
        data["combination_mode"] = patch["combination_mode"]
    filters = list(data.get("filters") or [])
    filters.extend(patch.get("filters_append") or [])
    data["filters"] = filters
    _dump_yaml(path, data)


def _patch_features(tree: Path) -> None:
    path = tree / "tpc" / "features.yaml"
    data = _load_yaml(path)
    pipe = data.get("feature_pipeline") or {}
    requested = list(pipe.get("requested_features") or [])
    for f in _FEATURE_APPEND:
        if f not in requested:
            requested.append(f)
    pipe["requested_features"] = requested
    data["feature_pipeline"] = pipe
    _dump_yaml(path, data)


def main() -> int:
    if not SRC_TPC.is_dir():
        print(f"missing {SRC_TPC}", file=sys.stderr)
        return 1
    EXP.mkdir(parents=True, exist_ok=True)

    for name, spec in SNAPSHOTS.items():
        print(f"building {name} ...")
        tree = _copy_tpc_snapshot(name)
        if "entry_filters_patch" in spec:
            _patch_entry_filters(tree, spec["entry_filters_patch"])  # type: ignore[arg-type]
        _patch_features(tree)
        print(f"  -> {tree}")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
