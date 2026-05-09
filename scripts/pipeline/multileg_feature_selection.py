from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEFAULT_REQUIRED_NODES_BY_TYPE = {
    "grid": {"bpc_soft_phase_f", "atr_f"},
    "dual_add_trend": {"trend_confidence_f", "bpc_soft_phase_f", "atr_f"},
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_str_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    return set()


def _default_keep_nodes(
    *,
    strategy_type: str,
    tuned_candidate: Mapping[str, Any],
) -> set[str]:
    st = str(strategy_type or "").strip().lower()
    keep = set(_DEFAULT_REQUIRED_NODES_BY_TYPE.get(st, set()))
    if st == "grid":
        # Only keep box structure when the calibrated regime actually uses the
        # box prefilter. Otherwise it stays in the source candidate pool but not
        # in the adopted slow snapshot.
        if bool(tuned_candidate.get("exclude_box_prefilter", False)):
            keep.add("box_structure_f")
    elif st == "dual_add_trend":
        # dual_add_trend currently uses explicit box exclusion in the execution
        # path, so box_structure_f is part of the selected slow structure.
        keep.add("box_structure_f")
    return keep


def select_multileg_feature_subset(
    *,
    strategy: str,
    strategy_type: str,
    config_dir: Path,
    output_dir: Path,
    strategy_cfg: Mapping[str, Any] | None,
    best_calibration: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write a slow-snapshot feature subset for a multi-leg strategy.

    This is intentionally model-free: multi-leg strategies do not own the
    TradeIntent label stack used by BPC SHAP. The selector is still
    config-driven and polymorphic by strategy_type, then strategy YAML can add
    or remove nodes through ``multileg_feature_selection``.
    """
    scfg = strategy_cfg or {}
    fs_cfg = scfg.get("multileg_feature_selection") or {}
    if not isinstance(fs_cfg, Mapping):
        fs_cfg = {}
    if not _as_bool(fs_cfg.get("enabled"), True):
        return {
            "strategy": strategy,
            "strategy_type": strategy_type,
            "enabled": False,
            "reason": "disabled_by_config",
        }

    best = best_calibration or {}
    tuned_candidate = best.get("tuned_candidate") or best.get("candidate") or {}
    if not isinstance(tuned_candidate, Mapping):
        tuned_candidate = {}

    keep_nodes = _default_keep_nodes(
        strategy_type=strategy_type,
        tuned_candidate=tuned_candidate,
    )
    keep_nodes.update(_as_str_set(fs_cfg.get("protected_nodes")))
    keep_nodes.update(_as_str_set(fs_cfg.get("keep_nodes")))
    keep_nodes.difference_update(_as_str_set(fs_cfg.get("drop_nodes")))

    output_dir.mkdir(parents=True, exist_ok=True)
    changed_files: list[dict[str, Any]] = []
    for yaml_name in ("features.yaml", "features_prefilter.yaml"):
        path = config_dir / yaml_name
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fp = data.get("feature_pipeline") or {}
        requested = fp.get("requested_features") or []
        if not isinstance(requested, list) or not requested:
            continue

        selected = [str(n) for n in requested if str(n) in keep_nodes]
        removed = [str(n) for n in requested if str(n) not in keep_nodes]
        if not selected:
            changed_files.append(
                {
                    "file": yaml_name,
                    "original": [str(n) for n in requested],
                    "selected": [],
                    "removed": [str(n) for n in requested],
                    "skipped": "empty_selection",
                }
            )
            continue

        fp["requested_features"] = selected
        data["feature_pipeline"] = fp
        data["_multileg_feature_selection"] = {
            "timestamp": datetime.now().isoformat(),
            "strategy_type": strategy_type,
            "source": "slow_snapshot",
            "selected_nodes": list(selected),
            "removed_nodes": list(removed),
            "tuned_candidate": dict(tuned_candidate),
        }
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        changed_files.append(
            {
                "file": yaml_name,
                "original": [str(n) for n in requested],
                "selected": selected,
                "removed": removed,
            }
        )

    result = {
        "strategy": strategy,
        "strategy_type": strategy_type,
        "enabled": True,
        "selector": "config_polymorphic_multileg",
        "keep_nodes": sorted(keep_nodes),
        "best_calibration": best,
        "metrics": dict(metrics or {}),
        "files": changed_files,
    }
    artifact_path = output_dir / "multileg_feature_selection.json"
    artifact_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    result["artifact_path"] = str(artifact_path)
    return result
