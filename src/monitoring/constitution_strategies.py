"""Resolve monitor strategy slugs from constitution + support allowlist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    load_constitution_dict,
    multi_leg_strategies_from_constitution,
    resolve_constitution_yaml,
)

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_SUPPORT = Path("config/monitoring/strategy_support.yaml")
DEFAULT_CONSTITUTION = Path("live/highcap/config/constitution/constitution.yaml")


def load_strategy_support(repo_root: Path) -> Dict[str, Any]:
    path = (repo_root / DEFAULT_STRATEGY_SUPPORT).resolve()
    if not path.is_file():
        return {"pcm_drift_ready": ["tpc"]}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def resolve_constitution_path(
    manifest: Dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    raw = manifest.get("constitution")
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    strategies_root = str(
        manifest.get("strategies_root") or "live/highcap/config/strategies"
    )
    rel = resolve_constitution_yaml(strategies_root)
    p = Path(rel)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def resolve_manifest_strategies(
    manifest: Dict[str, Any],
    *,
    repo_root: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return (strategies, meta) where meta documents constitution source and skips."""
    layer = str(manifest.get("strategies_layer") or "pcm").strip().lower()
    raw_source = manifest.get("strategies_source")
    meta: Dict[str, Any] = {
        "strategies_layer": layer,
        "skipped_not_ready": [],
    }

    if raw_source is None and manifest.get("strategies"):
        explicit = manifest.get("strategies")
        if isinstance(explicit, list) and explicit:
            slugs = [str(s).strip().lower() for s in explicit if str(s).strip()]
            if layer == "pcm":
                slugs, skipped = _filter_pcm_ready(slugs, repo_root=repo_root)
                meta["skipped_not_ready"] = skipped
            meta["strategies_source"] = "explicit"
            meta["strategies"] = slugs
            return slugs, meta

    source = str(raw_source or "constitution").strip().lower()
    meta["strategies_source"] = source
    if source != "constitution":
        raise ValueError(
            f"unsupported strategies_source {source!r}; use constitution or omit for legacy explicit list"
        )

    constitution_path = resolve_constitution_path(manifest, repo_root=repo_root)
    meta["constitution"] = str(constitution_path)
    cfg = load_constitution_dict(str(constitution_path))

    if layer == "multi_leg":
        slugs = multi_leg_strategies_from_constitution(cfg)
        if not slugs:
            logger.warning(
                "constitution %s has no multi_leg.strategies; C monitor will be empty",
                constitution_path,
            )
        meta["strategies"] = slugs
        return slugs, meta

    if layer != "pcm":
        raise ValueError(f"unknown strategies_layer {layer!r}; use pcm or multi_leg")

    enabled = enabled_archetypes_from_constitution(cfg)
    slugs, skipped = _filter_pcm_ready(enabled, repo_root=repo_root)
    meta["enabled_archetypes"] = enabled
    meta["skipped_not_ready"] = skipped
    meta["strategies"] = slugs
    if skipped:
        logger.info(
            "PCM monitor skips (not drift-ready): %s; monitoring: %s",
            ", ".join(skipped),
            ", ".join(slugs) or "(none)",
        )
    return slugs, meta


def _filter_pcm_ready(
    slugs: List[str],
    *,
    repo_root: Path,
) -> Tuple[List[str], List[str]]:
    support = load_strategy_support(repo_root)
    ready = {
        str(s).strip().lower()
        for s in (support.get("pcm_drift_ready") or ["tpc"])
        if str(s).strip()
    }
    ordered: List[str] = []
    skipped: List[str] = []
    seen: set[str] = set()
    for s in slugs:
        key = str(s).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        if key in ready:
            ordered.append(key)
        else:
            skipped.append(key)
    return ordered, skipped
