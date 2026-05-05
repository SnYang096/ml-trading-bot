"""Strategy directory layout and default pipeline resolution (ADR §3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_RESEARCH_PIPELINE_REL = Path("config/research_pipeline.yaml")
RESEARCH_PIPELINE_PROBE_NAMES = ("turbo.yaml", "slow.yaml", "pipeline.yaml")


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
        fb = (project_root / DEFAULT_RESEARCH_PIPELINE_REL).resolve()
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

    fb = (project_root / DEFAULT_RESEARCH_PIPELINE_REL).resolve()
    warnings.append(
        f"No {', '.join(RESEARCH_PIPELINE_PROBE_NAMES)} under "
        f"{research.relative_to(project_root)}; falling back to "
        f"{fb.relative_to(project_root)}"
    )
    return fb, warnings


def _resolve_research_pipeline_marker(
    marker: Path, project_root: Path
) -> Tuple[Path, List[str]]:
    raw = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
    ext = raw.get("extends")
    if isinstance(ext, str) and ext.strip():
        target = (marker.parent / ext.strip()).resolve()
        if target.is_file():
            rel_m = marker.relative_to(project_root)
            rel_t = target.relative_to(project_root)
            return target, [f"Resolved pipeline via {rel_m} → {rel_t}"]
        fb = (project_root / DEFAULT_RESEARCH_PIPELINE_REL).resolve()
        return fb, [
            f"{marker.relative_to(project_root)}: extends {ext!r} missing; "
            f"falling back to {fb.relative_to(project_root)}"
        ]
    return marker.resolve(), []


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts (override wins); lists and scalars are replaced."""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_strategy_study_and_threshold_search(
    project_root: Path, strategy_slug: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return ``(study_dict, threshold_search_dict)`` from ``research/turbo.yaml`` if present."""
    turbo = strategy_packaged_root(project_root, strategy_slug) / "research" / "turbo.yaml"
    if not turbo.is_file():
        return None, None
    raw = yaml.safe_load(turbo.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None, None
    study = raw.get("study")
    th = raw.get("threshold_search")
    out_s = study if isinstance(study, dict) else None
    out_t = th if isinstance(th, dict) else None
    return out_s, out_t
