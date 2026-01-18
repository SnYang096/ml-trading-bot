from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class StrategyExecutionProfile:
    router_mode: str
    execution_strategy_id: str
    evidence_rules: List[Dict[str, Any]]


@dataclass(frozen=True)
class StrategyProfile:
    version: int
    strategy_id: str
    archetype: str


@dataclass(frozen=True)
class ExecutionArchetype:
    name: str
    regime: str
    required_conditions: List[str]
    required_evidence: List[str]
    evidence_rules: List[Dict[str, Any]]
    gate_rules: Dict[str, Any]
    execution_constraints: Dict[str, Any]


def load_execution_archetypes_registry(
    path: str | Path = "config/nnmultihead/execution_archetypes.yaml",
) -> Dict[str, ExecutionArchetype]:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    regimes = obj.get("regimes") or {}
    out: Dict[str, ExecutionArchetype] = {}
    if isinstance(regimes, dict):
        for regime, rr in regimes.items():
            if not isinstance(rr, dict):
                continue
            arch = rr.get("archetypes") or {}
            if not isinstance(arch, dict):
                continue
            for name, a in arch.items():
                if not isinstance(a, dict):
                    continue
                out[str(name)] = ExecutionArchetype(
                    name=str(name),
                    regime=str(regime).upper(),
                    required_conditions=[
                        str(x) for x in (a.get("required_conditions") or [])
                    ],
                    required_evidence=[
                        str(x) for x in (a.get("required_evidence") or [])
                    ],
                    evidence_rules=list(a.get("evidence_rules") or []),
                    gate_rules=dict(a.get("gate_rules") or {}),
                    execution_constraints=dict(a.get("execution_constraints") or {}),
                )

    # Optional overlays
    overlays = obj.get("overlays") or {}
    if isinstance(overlays, dict):
        for name, a in overlays.items():
            if not isinstance(a, dict):
                continue
            out[str(name)] = ExecutionArchetype(
                name=str(name),
                regime=str(a.get("regime") or "MEAN").upper(),
                required_conditions=[
                    str(x) for x in (a.get("required_conditions") or [])
                ],
                required_evidence=[str(x) for x in (a.get("required_evidence") or [])],
                evidence_rules=list(a.get("evidence_rules") or []),
                gate_rules=dict(a.get("gate_rules") or {}),
                execution_constraints=dict(a.get("execution_constraints") or {}),
            )
    return out


def load_execution_profile_runtime_config(
    path: str | Path = "config/nnmultihead/execution_profile_runtime.yaml",
) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def resolve_execution_profile_paths(
    *,
    default_profile_root: str = "config/nnmultihead/strategies",
    default_archetype_registry_path: str = "config/nnmultihead/execution_archetypes.yaml",
    runtime_config_path: Optional[str | Path] = None,
) -> Tuple[str, str]:
    cfg_path = (
        runtime_config_path
        if runtime_config_path is not None
        else os.getenv(
            "MLBOT_NNMH_EXEC_PROFILE_CONFIG",
            "config/nnmultihead/execution_profile_runtime.yaml",
        )
    )
    cfg = load_execution_profile_runtime_config(cfg_path)
    profile_root = os.getenv(
        "MLBOT_NNMH_STRATEGY_PROFILE_ROOT",
        str(cfg.get("strategy_profile_root") or default_profile_root),
    )
    archetype_registry = os.getenv(
        "MLBOT_NNMH_EXEC_ARCHETYPE_REGISTRY",
        str(cfg.get("execution_archetype_registry") or default_archetype_registry_path),
    )
    return profile_root, archetype_registry


def load_strategy_profile_yaml(path: str | Path) -> StrategyProfile:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return StrategyProfile(
        version=int(obj.get("version", 1)),
        strategy_id=str(obj.get("strategy_id") or p.parent.name),
        archetype=str(obj.get("archetype") or "").strip(),
    )


def load_strategy_profile(
    *,
    strategy_id: str,
    root_dir: str | Path = "config/nnmultihead/strategies",
) -> Optional[StrategyProfile]:
    sid = str(strategy_id).strip()
    if not sid:
        return None
    root = Path(root_dir)
    p = root / sid / "profile.yaml"
    if not p.exists():
        return None
    return load_strategy_profile_yaml(p)


def resolve_strategy_profile_path(
    *,
    strategy_name: str,
    root_dir: str | Path = "config/nnmultihead/strategies",
) -> Optional[Path]:
    sid = str(strategy_name).strip()
    if not sid:
        return None
    root = Path(root_dir)
    direct = root / sid / "profile.yaml"
    return direct if direct.exists() else None


def resolve_execution_profile(
    *,
    strategy_id: str,
    profile_root: str | Path = "config/nnmultihead/strategies",
    archetype_registry_path: (
        str | Path
    ) = "config/nnmultihead/execution_archetypes.yaml",
) -> Optional[StrategyExecutionProfile]:
    pp = resolve_strategy_profile_path(strategy_name=strategy_id, root_dir=profile_root)
    prof = load_strategy_profile_yaml(pp) if pp else None
    if prof is None:
        return None
    arches = load_execution_archetypes_registry(archetype_registry_path)
    arch = arches.get(prof.archetype)
    if arch is None:
        return None
    return StrategyExecutionProfile(
        router_mode=str(arch.regime).upper(),
        execution_strategy_id=str(arch.name),
        evidence_rules=list(arch.evidence_rules or []),
    )
