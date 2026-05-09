"""Strategy directory layout and default pipeline resolution (ADR §3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# 全局默认（无 --strategy）：多策略 PCM 编排；历史遗留单体研究包仍可用 ``pipelines/research_pipeline.yaml`` 显式传入。
DEFAULT_PCM_ORCHESTRATE_REL = Path("config/pipelines/pcm_orchestrate_2h.yaml")
LEGACY_RESEARCH_PIPELINE_REL = Path("config/pipelines/research_pipeline.yaml")
RESEARCH_PIPELINE_PROBE_NAMES = ("turbo.yaml", "slow.yaml", "pipeline.yaml")
LEGACY_MULTILEG_ENGINE_NAMES = frozenset({"grid.yaml", "dual_add.yaml"})


def load_yaml_dict(path: Path, *, strict: bool = True) -> Dict[str, Any]:
    """Load a YAML dict from disk.

    - ``strict=True``: missing file / parse errors raise ``ValueError``.
    - ``strict=False``: returns ``{}`` on missing file / parse errors.
    """
    if not path.exists():
        if strict:
            raise ValueError(f"配置文件不存在: {path}")
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        if strict:
            raise ValueError(f"读取配置失败: {path}") from exc
        return {}
    if not isinstance(raw, dict):
        if strict:
            raise ValueError(f"配置文件格式错误(期望 dict): {path}")
        return {}
    return raw


def load_yaml_extends_chain(path: Path, *, strict: bool = True) -> Dict[str, Any]:
    """Load YAML with ``extends`` chain (child overlays parent)."""
    if not path.exists() and not strict:
        return {}
    chain: List[Dict[str, Any]] = []
    cur = path.resolve()
    visited: Set[Path] = set()
    for _ in range(64):
        if cur in visited:
            raise ValueError(f"extends 循环引用: {cur}")
        visited.add(cur)
        raw = load_yaml_dict(cur, strict=strict)
        ext = raw.pop("extends", None)
        chain.append(raw)
        if not ext:
            break
        nxt = (cur.parent / str(ext).strip()).resolve()
        if not nxt.is_file():
            raise ValueError(f"extends 指向的文件不存在: {ext!r}（自 {cur}）")
        cur = nxt
    merged: Dict[str, Any] = {}
    for layer in reversed(chain):
        merged = deep_merge_dicts(merged, layer)
    return merged


def resolve_strategy_profile_path(config_dir: Path, profile: str = "turbo") -> Path:
    p = str(profile or "turbo").strip().lower().replace("-", "_")
    if not p:
        p = "turbo"
    return config_dir / "research" / f"{p}.yaml"


def resolve_strategy_config_input(
    path: Path, *, default_profile: str = "turbo"
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve ``config_dir/profile_path/engine_path`` from a user input path.

    Supports:
    - strategy directory
    - research profile yaml (``.../research/*.yaml``)
    - legacy engine yaml (``grid.yaml`` / ``dual_add.yaml``)
    - generic yaml file path (treated as profile-like root layer)
    """
    if path.is_dir():
        cfg_dir = path
        prof_path = resolve_strategy_profile_path(cfg_dir, default_profile)
        return cfg_dir, prof_path if prof_path.exists() else None, None
    if path.name in LEGACY_MULTILEG_ENGINE_NAMES:
        return path.parent, None, path
    if path.parent.name == "research":
        return path.parent.parent, path, None
    return path.parent, path, None


def strategy_packaged_root(project_root: Path, strategy_slug: str) -> Path:
    """Return ``config/strategies/<strategy_slug>/`` (folder name as in pipeline YAML keys)."""
    return (project_root / "config" / "strategies" / strategy_slug).resolve()


def resolve_default_pipeline_config(
    project_root: Path,
    strategy_slug: Optional[str],
    explicit_config: Optional[Path],
) -> Tuple[Path, List[str]]:
    """Resolve pipeline YAML path for ``mlbot pipeline`` when ``--config`` is omitted.

    Order (per ADR §3.2): ``research/turbo.yaml`` → ``research/slow.yaml`` →
    ``research/pipeline.yaml``; each may be a thin pointer via top-level ``extends: <relative path>``.

    Returns ``(absolute_path, warning_messages)``.
    """
    warnings: List[str] = []
    if explicit_config is not None:
        p = explicit_config if explicit_config.is_absolute() else (project_root / explicit_config)
        return p.resolve(), warnings
    if not strategy_slug or not str(strategy_slug).strip():
        fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
        warnings.append(
            f"No --strategy/--config; using default {fb.relative_to(project_root)}"
        )
        return fb, warnings

    slug = str(strategy_slug).strip()
    research = strategy_packaged_root(project_root, slug) / "research"
    for name in RESEARCH_PIPELINE_PROBE_NAMES:
        candidate = research / name
        if not candidate.is_file():
            continue
        resolved, w = _resolve_research_pipeline_marker(candidate, project_root)
        warnings.extend(w)
        return resolved, warnings

    fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
    warnings.append(
        f"No {', '.join(RESEARCH_PIPELINE_PROBE_NAMES)} under "
        f"{research.relative_to(project_root)}; falling back to "
        f"{fb.relative_to(project_root)}"
    )
    return fb, warnings


def _resolve_research_pipeline_marker(
    marker: Path, project_root: Path
) -> Tuple[Path, List[str]]:
    raw = load_yaml_dict(marker, strict=False)
    ext = raw.get("extends")
    if isinstance(ext, str) and ext.strip():
        target = (marker.parent / ext.strip()).resolve()
        if target.is_file():
            rel_m = marker.relative_to(project_root)
            rel_t = target.relative_to(project_root)
            return target, [f"Resolved pipeline via {rel_m} → {rel_t}"]
        fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
        return fb, [
            f"{marker.relative_to(project_root)}: extends {ext!r} missing; "
            f"falling back to {fb.relative_to(project_root)}"
        ]
    return marker.resolve(), []


def is_research_turbo_or_slow_yaml(config_path: Path) -> bool:
    """True for ``config/strategies/*/research/turbo.yaml`` or ``slow.yaml``."""
    try:
        parts = config_path.resolve().parts
    except Exception:
        parts = config_path.parts
    if len(parts) < 2:
        return False
    return parts[-1] in ("turbo.yaml", "slow.yaml") and parts[-2] == "research"


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts (override wins); lists and scalars are replaced."""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

