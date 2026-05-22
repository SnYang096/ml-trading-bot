"""Read-only regime calibration / drift status for CMS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_regime_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _latest_drift_report(project_root: Path) -> Optional[Path]:
    base = project_root / "results" / "regime_drift_monitor"
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("**/report.json"), reverse=True)
    return candidates[0] if candidates else None


def fetch_regime_ops_snapshot(
    strategies_root: Path,
    *,
    project_root: Path,
    strategies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Per-strategy regime.yaml summary + optional latest drift monitor row."""
    slugs = strategies or ["tpc"]
    drift_path = _latest_drift_report(project_root)
    drift_by_strategy: Dict[str, Any] = {}
    if drift_path and drift_path.is_file():
        try:
            drift_doc = json.loads(drift_path.read_text(encoding="utf-8"))
            for item in drift_doc.get("strategies") or []:
                if isinstance(item, dict) and item.get("strategy"):
                    drift_by_strategy[str(item["strategy"])] = item
        except json.JSONDecodeError:
            pass

    rows: List[Dict[str, Any]] = []
    for slug in slugs:
        regime_path = strategies_root / slug / "archetypes" / "regime.yaml"
        data = _load_regime_yaml(regime_path)
        rules = data.get("rules") or []
        lc = data.get("last_calibration") or {}
        rows.append(
            {
                "strategy": slug,
                "regime_path": str(regime_path),
                "present": regime_path.is_file(),
                "n_rules": len(rules) if isinstance(rules, list) else 0,
                "allowed_sides": list(data.get("allowed_sides") or []),
                "allowed_regimes": list(data.get("allowed_regimes") or []),
                "last_calibration": lc if isinstance(lc, dict) else {},
                "drift": drift_by_strategy.get(slug),
                "drift_report_path": str(drift_path) if drift_path else None,
            }
        )
    return rows
