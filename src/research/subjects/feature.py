from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class Feature:
    col: str


@dataclass(frozen=True)
class RuleExpr:
    expr: str


@dataclass(frozen=True)
class ModelScore:
    col: str
    model_path: Optional[Path] = None


@dataclass(frozen=True)
class FeaturePool:
    features: List[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeaturePool":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        fp = raw.get("feature_pipeline", {}) or {}
        requested = fp.get("requested_features", []) or []
        return cls(features=[str(x) for x in requested])
