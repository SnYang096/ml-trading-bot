from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def _load_yaml_list(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return []


def _node_to_output_columns(
    *,
    nodes: Iterable[str],
    feature_deps: Dict[str, Any],
) -> Set[str]:
    out: Set[str] = set()
    feats = feature_deps.get("features") or {}
    for node in nodes:
        info = feats.get(str(node))
        if isinstance(info, dict):
            cols = info.get("output_columns") or []
            for c in cols:
                out.add(str(c))
            continue
        # fallback: strip _f
        if str(node).endswith("_f"):
            out.add(str(node)[:-2])
        else:
            out.add(str(node))
    return out


def load_live_feature_plan(
    *,
    plan_path: str | Path = "config/nnmultihead/live_feature_plan.yaml",
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> Set[str]:
    cfg = _load_yaml(plan_path)
    base_plan_path = str(cfg.get("base_feature_plan") or "")
    mode = str(cfg.get("base_feature_plan_mode") or "tiers")
    overlay = cfg.get("overlay") or {}

    base_features: Set[str] = set()
    base_nodes: List[str] = []
    if base_plan_path:
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        if mode == "minimal_required_cols":
            base_features = {
                str(x)
                for x in (fp.get("feature_contract") or {}).get(
                    "minimal_required_cols", []
                )
            }
        else:
            tiers = fp.get("tiers_enabled") or []
            tier_files = fp.get("tier_feature_files") or {}
            nodes: List[str] = []
            for t in tiers:
                f = tier_files.get(t)
                if f:
                    nodes.extend(_load_yaml_list(f))
            base_nodes = list(nodes)
            # optional blocks (base plan only)
            blocks = fp.get("optional_blocks_library") or {}
            enabled = fp.get("optional_blocks_enabled") or []
            for b in enabled:
                nodes.extend(list(blocks.get(b) or []))
            deps = _load_yaml(feature_deps_path)
            base_features = _node_to_output_columns(nodes=nodes, feature_deps=deps)

    # Overlay adjustments
    add_features = {str(x) for x in (overlay.get("add_features") or [])}
    add_nodes = [str(x) for x in (overlay.get("add_feature_nodes") or [])]
    add_blocks = [str(x) for x in (overlay.get("add_optional_blocks") or [])]
    drop_features = {str(x) for x in (overlay.get("drop_features") or [])}

    if add_nodes or add_blocks:
        deps = _load_yaml(feature_deps_path)
        nodes = list(add_nodes)
        if add_blocks:
            base = _load_yaml(base_plan_path)
            fp = base.get("feature_plan") or {}
            blocks = fp.get("optional_blocks_library") or {}
            for b in add_blocks:
                nodes.extend(list(blocks.get(b) or []))
        base_nodes.extend(nodes)
        base_features |= _node_to_output_columns(nodes=nodes, feature_deps=deps)

    base_features |= add_features
    base_features -= drop_features
    return base_features


def load_live_feature_nodes(
    *,
    plan_path: str | Path = "config/nnmultihead/live_feature_plan.yaml",
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> List[str]:
    cfg = _load_yaml(plan_path)
    base_plan_path = str(cfg.get("base_feature_plan") or "")
    mode = str(cfg.get("base_feature_plan_mode") or "tiers")
    overlay = cfg.get("overlay") or {}
    nodes: List[str] = []
    if base_plan_path and mode == "tiers":
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        tiers = fp.get("tiers_enabled") or []
        tier_files = fp.get("tier_feature_files") or {}
        for t in tiers:
            f = tier_files.get(t)
            if f:
                nodes.extend(_load_yaml_list(f))
        blocks = fp.get("optional_blocks_library") or {}
        enabled = fp.get("optional_blocks_enabled") or []
        for b in enabled:
            nodes.extend(list(blocks.get(b) or []))
    add_nodes = [str(x) for x in (overlay.get("add_feature_nodes") or [])]
    add_blocks = [str(x) for x in (overlay.get("add_optional_blocks") or [])]
    if add_nodes or add_blocks:
        base = _load_yaml(base_plan_path)
        fp = base.get("feature_plan") or {}
        blocks = fp.get("optional_blocks_library") or {}
        nodes.extend(add_nodes)
        for b in add_blocks:
            nodes.extend(list(blocks.get(b) or []))
    # de-dup while preserving order
    seen = set()
    out = []
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out
