from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class FeatureMeta:
    feature_name: str
    semantic_group: str
    time_scale: str
    applicable_scope: str
    allowed_layers: List[str]
    drift_sensitivity: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": str(self.feature_name),
            "semantic_group": str(self.semantic_group),
            "time_scale": str(self.time_scale),
            "applicable_scope": str(self.applicable_scope),
            "allowed_layers": list(self.allowed_layers),
            "drift_sensitivity": str(self.drift_sensitivity),
        }


def load_gate_feature_registry(path: str | Path) -> Dict[str, FeatureMeta]:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    feats = obj.get("features") if isinstance(obj, dict) else None
    if not isinstance(feats, dict):
        return {}
    out: Dict[str, FeatureMeta] = {}
    for k, v in feats.items():
        if not isinstance(v, dict):
            continue
        name = str(v.get("feature_name") or k).strip()
        if not name:
            continue
        out[name] = FeatureMeta(
            feature_name=name,
            semantic_group=str(v.get("semantic_group") or "").strip(),
            time_scale=str(v.get("time_scale") or "").strip(),
            applicable_scope=str(v.get("applicable_scope") or "").strip(),
            allowed_layers=[
                str(x).strip()
                for x in (v.get("allowed_layers") or [])
                if str(x).strip()
            ],
            drift_sensitivity=str(v.get("drift_sensitivity") or "unknown").strip(),
        )
    return out


def validate_gate_feature_registry(reg: Dict[str, FeatureMeta]) -> List[str]:
    errs: List[str] = []
    for name, meta in (reg or {}).items():
        if not meta.feature_name:
            errs.append(f"{name}: missing feature_name")
        if not meta.semantic_group:
            errs.append(f"{name}: missing semantic_group")
        if not meta.time_scale:
            errs.append(f"{name}: missing time_scale")
        if not meta.applicable_scope:
            errs.append(f"{name}: missing applicable_scope")
        if not meta.allowed_layers:
            errs.append(f"{name}: missing allowed_layers")
        else:
            # Must be gate-only or include gate.
            if "gate" not in [x.lower() for x in meta.allowed_layers]:
                errs.append(f"{name}: allowed_layers must include 'gate'")
    return errs


def validate_features_allowed(
    *,
    registry: Dict[str, FeatureMeta],
    requested_features: List[str],
    layer: str,
) -> List[str]:
    """
    Hard rule: "no metadata => cannot enter system".
    """
    layer = str(layer).lower().strip()
    errs: List[str] = []
    for f in requested_features or []:
        name = str(f).strip()
        if not name:
            continue
        meta = (registry or {}).get(name)
        if meta is None:
            errs.append(f"{name}: not in registry (no metadata)")
            continue
        allowed = [x.lower() for x in (meta.allowed_layers or [])]
        if layer not in allowed:
            errs.append(
                f"{name}: layer '{layer}' not allowed (allowed_layers={meta.allowed_layers})"
            )
    return errs
