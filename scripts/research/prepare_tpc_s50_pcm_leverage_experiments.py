#!/usr/bin/env python3
"""Materialize strategy trees + constitution overrides for S50 PCM / leverage experiments."""

from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "config" / "strategies"
VARIANTS = REPO / "config/experiments/20260607_tpc_s50_pcm_leverage/variants"
S50_TREE = VARIANTS / "tpc_semantic_depth_gt50_strategies"
EXP_DIR = REPO / "config" / "experiments" / "20260607_tpc_s50_pcm_leverage"
BASE_CONST = REPO / "config" / "constitution" / "constitution.yaml"

S50_PREFILTER_APPEND = [
    {
        "feature": "tpc_pullback_depth",
        "operator": ">",
        "value": 0.5,
        "rationale": "高语义深回踩：仅 depth>0.5 才进入后续层（实验 S50）",
    }
]

REGIME_EXEC_3X = {
    "enabled": True,
    "buckets": {
        "default": {
            "size_multiplier": 3.0,
            "rationale": "S50 深回踩入场：静态 3x 仓位（Phase-1 杠杆实验）",
        }
    },
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _copy_tree(name: str, *, src: Path | None = None) -> Path:
    dst = VARIANTS / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src or SRC, dst)
    return dst


def _patch_prefilter_append(tree: Path, append_rules: list) -> None:
    path = tree / "tpc" / "archetypes" / "prefilter.yaml"
    data = _load_yaml(path)
    rules = list(data.get("rules") or [])
    rules.extend(append_rules)
    data["rules"] = rules
    _dump_yaml(path, data)


def _patch_execution_regime(tree: Path, regime_exec: dict) -> None:
    path = tree / "tpc" / "archetypes" / "execution.yaml"
    data = _load_yaml(path)
    data["regime_execution"] = regime_exec
    _dump_yaml(path, data)


def _write_constitution(name: str, *, tpc_risk: float, bpc_risk: float) -> Path:
    out_dir = EXP_DIR / "constitution"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = deepcopy(_load_yaml(BASE_CONST))
    limits = (data.get("resource_allocation") or {}).get("per_strategy_limits") or {}
    if "tpc" in limits:
        limits["tpc"]["max_risk_per_trade"] = float(tpc_risk)
    if "bpc" in limits:
        limits["bpc"]["max_risk_per_trade"] = float(bpc_risk)
    data.setdefault("resource_allocation", {})["per_strategy_limits"] = limits
    out_path = out_dir / name
    _dump_yaml(out_path, data)
    return out_path


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1

    print("building tpc_s50_bpc_pcm_strategies (full tree, S50 prefilter) ...")
    pcm_tree = _copy_tree("tpc_s50_bpc_pcm_strategies")
    _patch_prefilter_append(pcm_tree, S50_PREFILTER_APPEND)

    print("building tpc_semantic_depth_gt50_3x_strategies ...")
    if S50_TREE.is_dir():
        lev_tree = _copy_tree("tpc_semantic_depth_gt50_3x_strategies", src=S50_TREE)
    else:
        lev_tree = _copy_tree("tpc_semantic_depth_gt50_3x_strategies")
        _patch_prefilter_append(lev_tree, S50_PREFILTER_APPEND)
    _patch_execution_regime(lev_tree, REGIME_EXEC_3X)

    print("building tpc_s50_bpc_pcm_3x_strategies ...")
    pcm3_tree = _copy_tree("tpc_s50_bpc_pcm_3x_strategies", src=pcm_tree)
    _patch_execution_regime(pcm3_tree, REGIME_EXEC_3X)

    print("writing constitution overrides ...")
    _write_constitution("pcm_equal.yaml", tpc_risk=0.01, bpc_risk=0.01)
    _write_constitution("pcm_tpc_heavy.yaml", tpc_risk=0.01, bpc_risk=0.005)
    _write_constitution("pcm_bpc_heavy.yaml", tpc_risk=0.005, bpc_risk=0.01)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
