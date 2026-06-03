"""Archetype threshold reference lines and semantic hints for Trade Map sub-charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config.regime_layer import multileg_regime_section

# Built-in fallbacks when YAML is missing (match locked archetype values).
_BUILTIN_REFERENCE_LINES: Dict[str, List[Dict[str, Any]]] = {
    "tpc_semantic_chop": [
        {"y": 0.40, "label": "regime ≤0.40", "operator": "<="},
    ],
    "bpc_semantic_chop": [
        {"y": 0.50, "label": "regime enter ≥0.50", "operator": ">="},
        {"y": 0.32, "label": "regime exit <0.32", "operator": "<"},
    ],
    "bpc_volume_compression_pct": [
        {"y": 0.9295, "label": "prefilter ≥0.9295", "operator": ">="},
    ],
}


def _append_rule_threshold(
    out: List[Dict[str, Any]],
    *,
    feature: str,
    operator: str,
    value: Any,
    label: Optional[str] = None,
) -> None:
    feat = str(feature or "").strip()
    if not feat:
        return
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    op = str(operator or "").strip()
    out.append(
        {
            "feature": feat,
            "y": val,
            "operator": op,
            "label": label or f"{feat} {op}{val:g}",
        }
    )


def _extract_rule_thresholds(rules: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(rules, list):
        return out
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if "feature" in rule and "value" in rule:
            _append_rule_threshold(
                out,
                feature=str(rule.get("feature") or ""),
                operator=str(rule.get("operator") or ""),
                value=rule["value"],
            )
        for key in ("any_of", "all_of"):
            for sub in rule.get(key) or []:
                if isinstance(sub, dict) and "feature" in sub and "value" in sub:
                    _append_rule_threshold(
                        out,
                        feature=str(sub.get("feature") or ""),
                        operator=str(sub.get("operator") or ""),
                        value=sub["value"],
                    )
    return out


def _load_yaml_doc(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _load_yaml_rules(path: Path) -> List[Dict[str, Any]]:
    return _extract_rule_thresholds(_load_yaml_doc(path).get("rules"))


def _load_multileg_regime_thresholds(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Multileg regime hysteresis from regime.yaml (TPC or legacy nested block)."""
    regime = multileg_regime_section(doc)
    if not regime:
        return []
    out: List[Dict[str, Any]] = []
    entry_feat = str(regime.get("entry_feature") or "bpc_semantic_chop").strip()
    entry_val = regime.get("entry_min", regime.get("entry_chop_min"))
    if entry_val is not None:
        _append_rule_threshold(
            out,
            feature=entry_feat,
            operator=">=",
            value=entry_val,
            label=f"regime enter ≥{float(entry_val):g}",
        )
    exit_val = regime.get("exit_below", regime.get("exit_chop_below"))
    if exit_val is not None:
        _append_rule_threshold(
            out,
            feature=entry_feat,
            operator="<",
            value=exit_val,
            label=f"regime exit <{float(exit_val):g}",
        )
    return out


def _load_multileg_box_prefilter_thresholds(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Box-quality gates from regime.yaml ``extensions.multileg.box_prefilter``."""
    regime = multileg_regime_section(doc)
    if not regime:
        return []
    box = regime.get("box_prefilter")
    if not isinstance(box, dict):
        return []
    out: List[Dict[str, Any]] = []
    if box.get("stability_min") is not None:
        _append_rule_threshold(
            out,
            feature="box_stability_60",
            operator=">=",
            value=box["stability_min"],
            label=f"box stability ≥{float(box['stability_min']):g}",
        )
    if box.get("width_min") is not None:
        _append_rule_threshold(
            out,
            feature="box_width_pct_60",
            operator=">=",
            value=box["width_min"],
            label=f"box width ≥{float(box['width_min']):g}",
        )
    if box.get("width_max") is not None:
        _append_rule_threshold(
            out,
            feature="box_width_pct_60",
            operator="<=",
            value=box["width_max"],
            label=f"box width ≤{float(box['width_max']):g}",
        )
    if box.get("touches_min") is not None:
        for feat in ("box_touches_hi_60", "box_touches_lo_60"):
            _append_rule_threshold(
                out,
                feature=feat,
                operator=">=",
                value=box["touches_min"],
                label=f"{feat} ≥{float(box['touches_min']):g}",
            )
    return out


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
        # Multi-leg chop_grid / trend_scalp: regime hysteresis in regime.yaml.
        reg_doc = _load_yaml_doc(strat_dir / "archetypes" / "regime.yaml")
        pre_doc = _load_yaml_doc(strat_dir / "archetypes" / "prefilter.yaml")
        regime_src = reg_doc if multileg_regime_section(reg_doc) else pre_doc
        for item in _load_multileg_regime_thresholds(regime_src):
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
        for item in _load_multileg_box_prefilter_thresholds(regime_src):
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
            return f"chop区({v:.2f}), enter≥0.50 exit<0.32"
        if v >= 0.32:
            return f"滞回区({v:.2f}), enter≥0.50 exit<0.32"
        return f"非chop({v:.2f}), exit<0.32"
    if col == "box_pos_60":
        if 0.35 <= v <= 0.65:
            return f"箱中({v:.3f}), 阈0.35–0.65"
        return f"箱外({v:.3f}), 阈0.35–0.65"
    if col == "box_stability_60":
        ok = v >= 0.85
        return (
            f"{'稳定' if ok else '不稳'}({v:.3f}), regime.box≥0.85 "
            f"(非 rules 段)"
        )
    if col == "box_width_pct_60":
        ok = 0.04 <= v <= 0.30
        return (
            f"{'宽度OK' if ok else '宽度异常'}({v:.3f}), regime.box 0.04–0.30 "
            f"(非 rules)"
        )
    if col in ("box_touches_hi_60", "box_touches_lo_60"):
        ok = v >= 5
        return (
            f"触边{int(v) if v == v else 0}次, regime.box≥5 "
            f"(非 rules; rules 只看 box_pos_60)"
        )
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
