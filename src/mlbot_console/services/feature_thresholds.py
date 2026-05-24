"""Archetype threshold reference lines and semantic hints for Trade Map sub-charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Built-in fallbacks when YAML is missing (match locked archetype values).
_BUILTIN_REFERENCE_LINES: Dict[str, List[Dict[str, Any]]] = {
    "tpc_semantic_chop": [
        {"y": 0.40, "label": "regime ≤0.40", "operator": "<="},
    ],
    "bpc_semantic_chop": [
        {"y": 0.50, "label": "chop grid ≥0.50", "operator": ">="},
    ],
    "bpc_volume_compression_pct": [
        {"y": 0.9295, "label": "prefilter ≥0.9295", "operator": ">="},
    ],
}


def _extract_rule_thresholds(rules: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(rules, list):
        return out
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if "feature" in rule and "value" in rule:
            feat = str(rule.get("feature") or "").strip()
            op = str(rule.get("operator") or "").strip()
            try:
                val = float(rule["value"])
            except (TypeError, ValueError):
                continue
            if feat:
                out.append(
                    {
                        "feature": feat,
                        "y": val,
                        "operator": op,
                        "label": f"{feat} {op}{val:g}",
                    }
                )
        for sub in rule.get("any_of") or []:
            if isinstance(sub, dict) and "feature" in sub and "value" in sub:
                feat = str(sub.get("feature") or "").strip()
                op = str(sub.get("operator") or "").strip()
                try:
                    val = float(sub["value"])
                except (TypeError, ValueError):
                    continue
                if feat:
                    out.append(
                        {
                            "feature": feat,
                            "y": val,
                            "operator": op,
                            "label": f"{feat} {op}{val:g}",
                        }
                    )
    return out


def _load_yaml_rules(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return _extract_rule_thresholds(doc.get("rules"))


def build_reference_lines_by_column(
    strategies_root: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Map feature column -> [{y, label, operator}] from archetype YAML + builtins."""
    out: Dict[str, List[Dict[str, Any]]] = {
        k: [dict(x) for x in v] for k, v in _BUILTIN_REFERENCE_LINES.items()
    }
    if strategies_root is None or not strategies_root.is_dir():
        return out
    for strat_dir in sorted(strategies_root.iterdir()):
        if not strat_dir.is_dir():
            continue
        for stage in ("regime", "prefilter", "gate", "execution"):
            path = strat_dir / "archetypes" / f"{stage}.yaml"
            for item in _load_yaml_rules(path):
                feat = str(item.get("feature") or "")
                if not feat:
                    continue
                line = {
                    "y": item["y"],
                    "label": item.get("label")
                    or f"{feat} {item.get('operator', '')}{item['y']:g}",
                    "operator": item.get("operator") or "",
                }
                existing = out.setdefault(feat, [])
                if not any(abs(x.get("y", 0) - line["y"]) < 1e-9 for x in existing):
                    existing.append(line)
    return out


def semantic_hint_for_column(column: str, value: Optional[float]) -> str:
    """Short UI hint for latest feature value."""
    if value is None or value != value:
        return ""
    col = str(column or "")
    v = float(value)
    if col == "tpc_semantic_chop":
        if v > 0.40:
            return f"chop高({v:.2f}), regime禁入, 阈≤0.40"
        return f"chop低({v:.2f}), regime可入场, 阈≤0.40"
    if col == "bpc_semantic_chop":
        if v >= 0.50:
            return f"chop区({v:.2f}), 阈≥0.50"
        return f"非chop({v:.2f}), 阈≥0.50"
    if col == "bpc_volume_compression_pct":
        ok = v >= 0.9295
        return f"{'通过' if ok else '未过'}({v:.3f}), 阈≥0.9295"
    if col == "weekly_ema_200_position":
        if v < 0.0:
            return f"深熊({v:.3f}), 阈<0"
        return f"EMA上方({v:.3f}), 阈<0"
    if col == "ema_1200_position":
        if v < 0.0:
            return f"EMA下方({v:.3f}), 阈<0"
        return f"EMA上方({v:.3f}), 阈<0"
    return ""
