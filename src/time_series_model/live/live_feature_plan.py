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
    plan_path: str | Path = "config/live/live_feature_plan.yaml",
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

    # Always include raw OHLCV (filled by IncrementalFeatureComputer from bar data)
    base_features |= {"open", "high", "low", "close", "volume"}

    # Merge gate-required columns (same auto-detect as load_live_feature_nodes)
    # so that _want(gate_column) is True and computed gate columns are kept.
    try:
        from src.cli.auto_detect_compute_requirements import (
            extract_required_features_from_execution_archetypes,
            map_features_to_tier_nodes,
            resolve_feature_dependencies,
        )

        deps_path = Path(feature_deps_path)
        project_root = (
            deps_path.resolve().parents[1]
            if deps_path.is_absolute()
            else Path(__file__).resolve().parents[3]
        )
        gate_features = extract_required_features_from_execution_archetypes(
            project_root / "config/strategies/bpc/archetypes/gate.yaml"
        )
        if gate_features:
            base_features |= gate_features  # gate rule keys (e.g. bpc_dir_consistency_long) in case filled by IncrementalFeatureComputer
            deps = _load_yaml(feature_deps_path)
            if deps:
                gate_nodes = map_features_to_tier_nodes(gate_features, deps)
                if gate_nodes:
                    all_gate_nodes = resolve_feature_dependencies(gate_nodes, deps)
                    gate_cols = _node_to_output_columns(
                        nodes=all_gate_nodes, feature_deps=deps
                    )
                    base_features |= gate_cols
    except Exception:
        pass

    return base_features


def load_live_feature_nodes(
    *,
    plan_path: str | Path = "config/live/live_feature_plan.yaml",
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
    # AUTO-DETECT: 自动检测gate规则需要的特征并添加到nodes
    try:
        from src.cli.auto_detect_compute_requirements import (
            extract_required_features_from_execution_archetypes,
            map_features_to_tier_nodes,
            resolve_feature_dependencies,
        )
        from pathlib import Path

        # 获取项目根目录（从feature_deps_path推断）
        deps_path = Path(feature_deps_path)
        if not deps_path.is_absolute():
            # 假设相对于项目根目录
            project_root = (
                Path(__file__).resolve().parents[3]
            )  # src/time_series_model/live -> project root
        else:
            project_root = deps_path.parents[
                1
            ]  # config/feature_dependencies.yaml -> project root

        # 提取gate规则需要的特征列名
        gate_features = extract_required_features_from_execution_archetypes(
            project_root / "config/strategies/bpc/archetypes/gate.yaml"
        )

        if gate_features:
            # 读取feature_dependencies
            deps = _load_yaml(feature_deps_path)
            if deps:
                # 映射到feature nodes
                gate_nodes = map_features_to_tier_nodes(gate_features, deps)

                if gate_nodes:
                    # 递归解析依赖关系
                    all_gate_nodes = resolve_feature_dependencies(gate_nodes, deps)

                    # 添加gate nodes到nodes列表
                    existing_nodes_set = set(nodes)
                    new_gate_nodes = [
                        n for n in all_gate_nodes if n not in existing_nodes_set
                    ]
                    if new_gate_nodes:
                        nodes.extend(new_gate_nodes)
    except Exception:
        # 如果自动检测失败，不影响正常流程
        pass

    # de-dup while preserving order
    seen = set()
    out = []
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out
