"""Simple model registry for logging model artifacts and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RegistryEntry:
    timestamp: str
    pipeline: str
    symbol: str
    artifact_path: str
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    notes: Optional[str] = None


class ModelRegistry:
    def __init__(self, registry_path: str) -> None:
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8") as f:
                json.dump({"entries": []}, f, indent=2)

    def log(
        self,
        pipeline: str,
        symbol: str,
        artifact_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        metrics = metrics or {}
        params = params or {}
        entry = RegistryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline=pipeline,
            symbol=symbol,
            artifact_path=artifact_path,
            metrics=metrics,
            params=params,
            notes=notes,
        )
        data = self._load()
        data["entries"].append(asdict(entry))
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)


