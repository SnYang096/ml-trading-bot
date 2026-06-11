#!/usr/bin/env python3
"""Materialize variant trees for 20260604_tpc_entry_semantic_validate."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "config" / "strategies"
TURBO_EXEC = (
    REPO
    / "results/tpc/turbo-rolling-sim/_rolling_sim/20260424_191639"
    / "fast_month_2024-01/strategies_calibrated/tpc/archetypes/execution.yaml"
)

_ANTI_CHASE_APPEND = [
    {
        "id": "tpc_anti_chase_not_at_high",
        "enabled": True,
        "direction": "long",
        "description": "做多：距局部高点足够远，避免贴顶追",
        "conditions": [
            {"feature": "bars_since_local_high", "operator": ">=", "value": 0.10}
        ],
    },
    {
        "id": "tpc_anti_chase_not_at_low",
        "enabled": True,
        "direction": "short",
        "description": "做空：距局部低点足够远，避免贴底追",
        "conditions": [
            {"feature": "bars_since_local_low", "operator": ">=", "value": 0.10}
        ],
    },
]

_E2A_OR_BUNDLE = [
    "tpc_deep_pullback_vol_confirm",
    "tpc_deep_pullback_delta_absorb",
]

SNAPSHOTS: dict[str, dict[str, object]] = {
    "tpc_semantic_depth_gt50_strategies": {
        "prefilter_rules_append": [
            {
                "feature": "tpc_pullback_depth",
                "operator": ">",
                "value": 0.5,
                "rationale": "高语义深回踩：仅 depth>0.5 才进入后续层（实验 S50）",
            }
        ],
    },
    "tpc_semantic_depth_gt50_ema_near_strategies": {
        "prefilter_rules_append": [
            {
                "feature": "tpc_pullback_depth",
                "operator": ">",
                "value": 0.5,
                "rationale": "高语义深回踩：仅 depth>0.5（实验 S51）",
            },
            {
                "feature": "ema_1200_position",
                "operator": ">=",
                "value": -0.1,
                "rationale": "允许价格略低于 EMA1200（不低于 -10%），配合深回踩",
            },
        ],
        "regime_any_of_replace": [
            {
                "feature": "ema_1200_position",
                "operator": ">=",
                "value": 0.03,
            },
            {
                "feature": "ema_1200_position",
                "operator": "<=",
                "value": -0.03,
            },
        ],
        "direction_inner_abs": -0.1,
    },
    "tpc_entry_depth_ge15_strategies": {
        "prefilter_rules_append": [
            {
                "feature": "tpc_pullback_depth",
                "operator": ">=",
                "value": 0.15,
                "rationale": "浅回踩下界：过滤 depth≈0 追高/追低（实验 E1）",
            }
        ],
    },
    "tpc_entry_anti_chase_strategies": {
        "entry_filters_patch": {
            "combination_mode": "and",
            "filters_append": _ANTI_CHASE_APPEND,
        },
    },
    "tpc_entry_e2a_or_anti_chase_strategies": {
        "entry_filters_patch": {
            "or_bundle_ids": _E2A_OR_BUNDLE,
            "filters_append": _ANTI_CHASE_APPEND,
        },
    },
    "tpc_entry_e1e2_band_or_anti_strategies": {
        "prefilter_rules_append": [
            {
                "feature": "tpc_pullback_depth",
                "operator": ">=",
                "value": 0.15,
                "rationale": "双边带下界 0.15（与 prod <=0.85 组成 band）",
            }
        ],
        "entry_filters_patch": {
            "or_bundle_ids": _E2A_OR_BUNDLE,
            "filters_append": _ANTI_CHASE_APPEND,
        },
    },
    "tpc_entry_gate_pe_strategies": {
        "gate_append": [
            {
                "id": "gate_path_efficiency_continuation",
                "tag": "HARD_PATH_EFFICIENCY_CONTINUATION",
                "phase": "hard_gate",
                "priority": 7,
                "reason": "高 path_efficiency = 延续区，留给 BPC（实验 E3）",
                "when": {"path_efficiency_pct": {"value_gt": 0.15}},
                "then": {"action": "deny"},
                "comment": "deny 高 PE；低 PE 回踩区保留给 TPC",
            }
        ],
    },
    "tpc_exec_turbo_20260424_strategies": {
        "execution_copy": str(TURBO_EXEC),
    },
}


def _copy_tree(name: str) -> Path:
    dst = (
        REPO / "config/experiments/20260604_tpc_entry_semantic_validate/variants" / name
    )
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    return dst


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _patch_prefilter(tree: Path, append_rules: list) -> None:
    path = tree / "tpc" / "archetypes" / "prefilter.yaml"
    data = _load_yaml(path)
    rules = list(data.get("rules") or [])
    rules.extend(append_rules)
    data["rules"] = rules
    _dump_yaml(path, data)


def _patch_entry_filters(tree: Path, patch: dict) -> None:
    path = tree / "tpc" / "archetypes" / "entry_filters.yaml"
    data = _load_yaml(path)
    if "combination_mode" in patch:
        data["combination_mode"] = patch["combination_mode"]
    if "or_bundle_ids" in patch:
        data["or_bundle_ids"] = list(patch["or_bundle_ids"])
    filters = list(data.get("filters") or [])
    filters.extend(patch.get("filters_append") or [])
    data["filters"] = filters
    _dump_yaml(path, data)


def _patch_gate(tree: Path, append_rules: list) -> None:
    path = tree / "tpc" / "archetypes" / "gate.yaml"
    data = _load_yaml(path)
    hard = list(data.get("hard_gates") or [])
    hard.extend(append_rules)
    data["hard_gates"] = hard
    _dump_yaml(path, data)


def _patch_regime_any_of(tree: Path, any_of_rules: list) -> None:
    path = tree / "tpc" / "archetypes" / "regime.yaml"
    data = _load_yaml(path)
    rules = list(data.get("rules") or [])
    for rule in rules:
        if isinstance(rule, dict) and "any_of" in rule:
            rule["any_of"] = any_of_rules
            rule["rationale"] = (
                "S51：收窄 EMA1200 死区至 |pos|>0.03，允许略低于均线深回踩"
            )
            break
    data["rules"] = rules
    _dump_yaml(path, data)


def _patch_direction_inner_abs(tree: Path, inner_abs: float) -> None:
    path = tree / "tpc" / "archetypes" / "direction.yaml"
    data = _load_yaml(path)
    for rule in data.get("direction_rules") or []:
        pb = rule.get("position_band")
        if isinstance(pb, dict):
            pb["inner_abs"] = inner_abs
    desc = str(data.get("description") or "")
    if "S51" not in desc:
        data["description"] = (
            desc + " | S51: inner_abs=-0.10 允许 ema_1200 略下方做多"
        ).strip()
    for rule in data.get("direction_rules") or []:
        if rule.get("id") == "tpc_macd_ema1200_align":
            rule["description"] = (
                "MACD sign；long pos>-0.10，short pos<0.10（S51 宏回踩带）"
            )
    _dump_yaml(path, data)


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1

    for name, spec in SNAPSHOTS.items():
        print(f"building {name} ...")
        tree = _copy_tree(name)
        if "prefilter_rules_append" in spec:
            _patch_prefilter(tree, spec["prefilter_rules_append"])  # type: ignore[arg-type]
        if "entry_filters_patch" in spec:
            _patch_entry_filters(tree, spec["entry_filters_patch"])  # type: ignore[arg-type]
        if "gate_append" in spec:
            _patch_gate(tree, spec["gate_append"])  # type: ignore[arg-type]
        if "regime_any_of_replace" in spec:
            _patch_regime_any_of(tree, spec["regime_any_of_replace"])  # type: ignore[arg-type]
        if "direction_inner_abs" in spec:
            _patch_direction_inner_abs(tree, float(spec["direction_inner_abs"]))
        if "execution_copy" in spec:
            src = Path(spec["execution_copy"])  # type: ignore[arg-type]
            if not src.is_file():
                print(f"missing turbo execution: {src}", file=sys.stderr)
                return 1
            dst = tree / "tpc" / "archetypes" / "execution.yaml"
            shutil.copy2(src, dst)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
