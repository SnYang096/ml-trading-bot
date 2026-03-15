from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _rule_signature(rule: Dict[str, Any]) -> Tuple[Any, ...]:
    if not isinstance(rule, dict):
        return ("raw", json.dumps(rule, sort_keys=True, ensure_ascii=False))

    if "feature" in rule:
        return (
            "simple",
            rule.get("feature"),
            rule.get("operator"),
            json.dumps(rule.get("value"), sort_keys=True, ensure_ascii=False),
        )

    if "any_of" in rule and isinstance(rule.get("any_of"), list):
        sub_sigs = []
        for sub in rule["any_of"]:
            if not isinstance(sub, dict):
                continue
            sub_sigs.append(
                (
                    sub.get("feature"),
                    sub.get("operator"),
                    json.dumps(sub.get("value"), sort_keys=True, ensure_ascii=False),
                )
            )
        return ("any_of", tuple(sorted(sub_sigs)))

    return ("raw", json.dumps(rule, sort_keys=True, ensure_ascii=False))


def load_locked_prefilter_rules(prefilter_path: Path) -> List[Dict[str, Any]]:
    if not prefilter_path.exists():
        return []
    raw = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules = raw.get("rules") or []
    if not isinstance(rules, list):
        return []
    return [copy.deepcopy(r) for r in rules if isinstance(r, dict) and r.get("locked")]


def merge_locked_prefilter_rules(
    prefilter_path: Path, locked_rules: List[Dict[str, Any]]
) -> Dict[str, int]:
    if not locked_rules:
        return {"added": 0, "total": 0}

    raw: Dict[str, Any] = {}
    if prefilter_path.exists():
        raw = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}

    current_rules = raw.get("rules") or []
    if not isinstance(current_rules, list):
        current_rules = []

    existing = {_rule_signature(r) for r in current_rules if isinstance(r, dict)}
    merged_rules = [copy.deepcopy(r) for r in current_rules if isinstance(r, dict)]
    added = 0
    for lr in locked_rules:
        sig = _rule_signature(lr)
        if sig in existing:
            continue
        merged_rules.append(copy.deepcopy(lr))
        existing.add(sig)
        added += 1

    if added > 0 or not prefilter_path.exists():
        raw["rules"] = merged_rules
        prefilter_path.parent.mkdir(parents=True, exist_ok=True)
        prefilter_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    return {"added": added, "total": len(merged_rules)}


def apply_locked_thresholds(
    prefilter_raw: Dict[str, Any],
    *,
    fer_lower: float,
    fer_upper: float,
    sr_min: float,
    dist_max: float,
) -> Dict[str, Any]:
    out = json.loads(json.dumps(prefilter_raw))
    rules = out.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("prefilter.yaml rules 必须为 list")

    seen = {
        "fer_lower": False,
        "fer_upper": False,
        "sr_min": False,
        "dist_lower": False,
        "dist_upper": False,
    }

    for r in rules:
        if not isinstance(r, dict) or not r.get("locked"):
            continue
        feat = r.get("feature")
        op = r.get("operator")
        if feat == "fer_signed_efficiency_pct" and op == ">=":
            r["value"] = float(fer_lower)
            seen["fer_lower"] = True
        elif feat == "fer_signed_efficiency_pct" and op == "<=":
            r["value"] = float(fer_upper)
            seen["fer_upper"] = True
        elif feat == "sr_strength_max" and op == ">=":
            r["value"] = float(sr_min)
            seen["sr_min"] = True
        elif feat == "dist_to_nearest_sr" and op == ">=":
            r["value"] = float(-dist_max)
            seen["dist_lower"] = True
        elif feat == "dist_to_nearest_sr" and op == "<=":
            r["value"] = float(dist_max)
            seen["dist_upper"] = True

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise ValueError(f"prefilter.yaml 缺少必要 locked 规则: {missing}")
    return out


def build_override_prefilter(
    prod_prefilter_path: Path, output_path: Path, params: Dict[str, float]
) -> Path:
    base = yaml.safe_load(prod_prefilter_path.read_text(encoding="utf-8")) or {}
    tuned = apply_locked_thresholds(
        base,
        fer_lower=float(params["fer_lower"]),
        fer_upper=float(params["fer_upper"]),
        sr_min=float(params["sr_min"]),
        dist_max=float(params["dist_max"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(tuned, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path
