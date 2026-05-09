from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from src.config.strategy_layout import (
    deep_merge_dicts,
    load_yaml_dict,
    load_yaml_extends_chain,
    resolve_strategy_profile_path,
)


def _write_yaml_dict(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

def load_multileg_layers(
    *,
    config_dir: Path,
    strategy_type: str,
    profile: str = "turbo",
    profile_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any]]:
    """Load profile YAML + optional legacy engine YAML + archetype layers."""
    root: Dict[str, Any] = {
        "strategy_type": str(strategy_type).strip().lower(),
        "status": "research",
    }
    prof_path = profile_path or resolve_strategy_profile_path(config_dir, profile)
    if not prof_path.is_absolute():
        prof_path = config_dir / prof_path
    if prof_path.exists():
        root = deep_merge_dicts(root, load_yaml_extends_chain(prof_path, strict=True))
    elif profile_path is not None:
        raise ValueError(f"missing multileg profile yaml: {prof_path}")

    # Backward compatibility: allow passing a legacy engine YAML path.
    if engine_path is not None:
        if not engine_path.is_absolute():
            engine_path = config_dir / engine_path
        if engine_path.exists():
            root = deep_merge_dicts(root, load_yaml_dict(engine_path, strict=True))
    else:
        engine_path = prof_path

    arch = root.get("archetypes", {}) or {}
    prefilter_rel = str(arch.get("prefilter", "archetypes/prefilter.yaml") or "").strip()
    execution_rel = str(arch.get("execution", "archetypes/execution.yaml") or "").strip()
    prefilter_path = (
        config_dir / prefilter_rel
        if prefilter_rel
        else config_dir / "archetypes/prefilter.yaml"
    )
    execution_path = (
        config_dir / execution_rel
        if execution_rel
        else config_dir / "archetypes/execution.yaml"
    )
    prefilter = load_yaml_dict(prefilter_path, strict=False)
    execution = load_yaml_dict(execution_path, strict=False)
    return engine_path, root, prefilter_path, prefilter, execution_path, execution


def load_multileg_effective_config(
    *,
    config_dir: Path,
    strategy_type: str,
    profile: str = "turbo",
    profile_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build effective runtime config from research profile + archetype overlays."""
    _, root, _, prefilter, _, execution = load_multileg_layers(
        config_dir=config_dir,
        strategy_type=strategy_type,
        profile=profile,
        profile_path=profile_path,
        engine_path=engine_path,
    )
    merged = dict(root)
    if prefilter:
        merged = deep_merge_dicts(merged, prefilter)
    if execution:
        merged = deep_merge_dicts(merged, execution)
    return merged


def update_multileg_calibration_candidate(
    *,
    config_dir: Path,
    strategy_type: str,
    candidate: Dict[str, Any],
    profile: str = "turbo",
    profile_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
) -> None:
    """Apply calibration candidate to layer files (prefer archetype overlays)."""
    (
        engine_path,
        root,
        prefilter_path,
        prefilter,
        execution_path,
        execution,
    ) = load_multileg_layers(
        config_dir=config_dir,
        strategy_type=strategy_type,
        profile=profile,
        profile_path=profile_path,
        engine_path=engine_path,
    )
    pre = prefilter if prefilter else root
    exe = execution if execution else root

    if strategy_type == "grid":
        regime = pre.setdefault("regime", {})
        inv = exe.setdefault("inventory", {})
        spacing = inv.setdefault("spacing", {})
        if "entry_chop_min" in candidate:
            regime["entry_chop_min"] = float(candidate["entry_chop_min"])
        if "exit_chop_below" in candidate:
            regime["exit_chop_below"] = float(candidate["exit_chop_below"])
        if "exclude_box_prefilter" in candidate:
            regime["exclude_box_prefilter"] = bool(candidate["exclude_box_prefilter"])
        if "atr_mult" in candidate:
            spacing["atr_mult"] = float(candidate["atr_mult"])
        if "min_pct" in candidate:
            spacing["min_pct"] = float(candidate["min_pct"])
    elif strategy_type == "dual_add_trend":
        regime = pre.setdefault("regime", {})
        inv = exe.setdefault("inventory", {})
        spacing = exe.setdefault("add_spacing", {})
        tp = exe.setdefault("take_profit", {})
        if "entry_min" in candidate:
            regime["entry_min"] = float(candidate["entry_min"])
        if "exit_below" in candidate:
            regime["exit_below"] = float(candidate["exit_below"])
        if "max_semantic_chop_entry" in candidate:
            regime["max_semantic_chop_entry"] = float(
                candidate["max_semantic_chop_entry"]
            )
        if "max_semantic_chop_hold" in candidate:
            regime["max_semantic_chop_hold"] = float(
                candidate["max_semantic_chop_hold"]
            )
        if "step_atr_mult" in candidate:
            spacing["atr_mult"] = float(candidate["step_atr_mult"])
        if "tp_atr_mult" in candidate:
            tp["atr_mult"] = float(candidate["tp_atr_mult"])
        if "tp_pct" in candidate:
            tp["min_pct"] = float(candidate["tp_pct"])
        if "flip_action" in candidate:
            inv["flip_action"] = str(candidate["flip_action"])
    else:
        raise ValueError(f"unsupported multi-leg strategy_type={strategy_type!r}")

    if prefilter:
        _write_yaml_dict(prefilter_path, pre)
    if execution:
        _write_yaml_dict(execution_path, exe)
