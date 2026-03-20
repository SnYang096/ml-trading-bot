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


def detect_locked_template(prefilter_raw: Dict[str, Any]) -> str:
    rules = prefilter_raw.get("rules") or []
    if not isinstance(rules, list):
        return "unknown"
    feats = {r.get("feature") for r in rules if isinstance(r, dict) and r.get("locked")}
    if {
        "fer_signed_efficiency_pct",
        "sr_strength_max",
        "dist_to_nearest_sr",
    }.issubset(feats):
        return "fer"
    if {
        "me_atr_pct",
        "me_cvd_alignment",
    }.issubset(feats) and (
        "me_accel_5k_long" in feats
        or "me_accel_5k_short" in feats
        or "me_accel_5k" in feats
    ):
        return "me"
    if {
        "atr_percentile",
        "recent_compression_decay",
        "compression_duration",
        "oi_compression_score",
    }.issubset(feats):
        return "me"
    if {
        "bpc_score_pullback",
        "bpc_pullback_depth",
        "bpc_recovery_strength",
    }.issubset(feats):
        return "bpc"
    return "unknown"


def apply_locked_thresholds(
    prefilter_raw: Dict[str, Any],
    *,
    fer_lower: float | None = None,
    fer_upper: float | None = None,
    sr_min: float | None = None,
    dist_max: float | None = None,
    fer_sqs_min: float | None = None,
    atr_lower: float | None = None,
    atr_upper: float | None = None,
    me_accel_abs_min: float | None = None,
    me_cvd_min: float | None = None,
    compression_min: float | None = None,
    decay_upper: float | None = None,
    oi_min: float | None = None,
    bpc_pullback_score_min: float | None = None,
    bpc_pullback_depth_max: float | None = None,
    bpc_recovery_min: float | None = None,
    template: str | None = None,
) -> Dict[str, Any]:
    out = json.loads(json.dumps(prefilter_raw))
    rules = out.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("prefilter.yaml rules 必须为 list")

    tpl = (template or detect_locked_template(out)).lower()
    if tpl == "unknown":
        raise ValueError("无法识别 locked 规则模板，请显式传入 template")

    if tpl == "fer":
        required = {
            "fer_lower": fer_lower,
            "fer_upper": fer_upper,
            "sr_min": sr_min,
            "dist_max": dist_max,
            "fer_sqs_min": fer_sqs_min,
        }
        missing_params = [k for k, v in required.items() if v is None]
        if missing_params:
            raise ValueError(f"FER tuned 参数缺失: {missing_params}")

        seen = {
            "fer_lower": False,
            "fer_upper": False,
            "sr_min": False,
            "dist_lower": False,
            "dist_upper": False,
            "sqs_min": False,
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
                r["value"] = float(-float(dist_max))
                seen["dist_lower"] = True
            elif feat == "dist_to_nearest_sr" and op == "<=":
                r["value"] = float(dist_max)
                seen["dist_upper"] = True
            elif (
                feat
                in (
                    "sqs_hal_high",
                    "sqs_hal_low",
                    "sqs_hal_high_pct",
                    "sqs_hal_low_pct",
                )
                and op == ">="
            ):
                r["value"] = float(fer_sqs_min)
                seen["sqs_min"] = True
        missing = [k for k, v in seen.items() if not v]
        if missing:
            raise ValueError(f"prefilter.yaml 缺少必要 FER locked 规则: {missing}")
        return out

    if tpl == "me":
        required = {
            "atr_lower": atr_lower,
            "atr_upper": atr_upper,
            "me_accel_abs_min": me_accel_abs_min,
            "me_cvd_min": me_cvd_min,
        }
        missing_params = [k for k, v in required.items() if v is None]
        if missing_params:
            raise ValueError(f"ME tuned 参数缺失: {missing_params}")

        seen = {
            "atr_lower": False,
            "atr_upper": False,
            "accel": False,
            "cvd": False,
        }
        for r in rules:
            if not isinstance(r, dict) or not r.get("locked"):
                continue
            feat = r.get("feature")
            op = r.get("operator")
            if feat in ("atr_percentile", "me_atr_pct") and op == ">=":
                r["value"] = float(atr_lower)
                seen["atr_lower"] = True
            elif feat in ("atr_percentile", "me_atr_pct") and op == "<=":
                r["value"] = float(atr_upper)
                seen["atr_upper"] = True
            elif feat == "me_accel_5k_long" and op == ">=":
                r["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k_short" and op == ">=":
                r["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k" and op == ">=":
                # Backward compatibility for old long-side configs.
                r["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k" and op == "<=":
                # Backward compatibility for old short-side configs.
                r["value"] = float(-abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_cvd_alignment" and op == ">=":
                r["value"] = float(me_cvd_min)
                seen["cvd"] = True
        missing = [k for k, v in seen.items() if not v]
        if missing:
            raise ValueError(f"prefilter.yaml 缺少必要 ME locked 规则: {missing}")
        return out

    if tpl == "bpc":
        required = {
            "bpc_pullback_score_min": bpc_pullback_score_min,
            "bpc_pullback_depth_max": bpc_pullback_depth_max,
            "bpc_recovery_min": bpc_recovery_min,
        }
        missing_params = [k for k, v in required.items() if v is None]
        if missing_params:
            raise ValueError(f"BPC tuned 参数缺失: {missing_params}")

        seen = {
            "pullback_score_min": False,
            "pullback_depth_max": False,
            "recovery_min": False,
        }
        for r in rules:
            if not isinstance(r, dict) or not r.get("locked"):
                continue
            feat = r.get("feature")
            op = r.get("operator")
            if feat == "bpc_score_pullback" and op == ">=":
                r["value"] = float(bpc_pullback_score_min)
                seen["pullback_score_min"] = True
            elif feat == "bpc_pullback_depth" and op == "<=":
                r["value"] = float(bpc_pullback_depth_max)
                seen["pullback_depth_max"] = True
            elif feat == "bpc_recovery_strength" and op == ">=":
                r["value"] = float(bpc_recovery_min)
                seen["recovery_min"] = True
        missing = [k for k, v in seen.items() if not v]
        if missing:
            raise ValueError(f"prefilter.yaml 缺少必要 BPC locked 规则: {missing}")
        return out

    raise ValueError(f"不支持的 template: {tpl}")


def _infer_template_from_params(params: Dict[str, float]) -> str:
    if {
        "fer_lower",
        "fer_upper",
        "sr_min",
        "dist_max",
        "fer_sqs_min",
    }.issubset(params.keys()):
        return "fer"
    if {"atr_lower", "atr_upper", "me_accel_abs_min", "me_cvd_min"}.issubset(
        params.keys()
    ) or {
        "atr_lower",
        "atr_upper",
        "compression_min",
        "decay_upper",
        "oi_min",
    }.issubset(
        params.keys()
    ):
        return "me"
    if {
        "bpc_pullback_score_min",
        "bpc_pullback_depth_max",
        "bpc_recovery_min",
    }.issubset(params.keys()):
        return "bpc"
    return "unknown"


def _normalize_params_for_template(
    params: Dict[str, float], template: str
) -> Dict[str, float]:
    if template == "fer":
        return {
            "fer_lower": float(params["fer_lower"]),
            "fer_upper": float(params["fer_upper"]),
            "sr_min": float(params["sr_min"]),
            "dist_max": float(params["dist_max"]),
            "fer_sqs_min": float(params["fer_sqs_min"]),
        }
    if template == "me":
        if {"me_accel_abs_min", "me_cvd_min"}.issubset(params.keys()):
            return {
                "atr_lower": float(params["atr_lower"]),
                "atr_upper": float(params["atr_upper"]),
                "me_accel_abs_min": float(params["me_accel_abs_min"]),
                "me_cvd_min": float(params["me_cvd_min"]),
            }
        # Backward compatibility: old ME knobs map to loose defaults.
        return {
            "atr_lower": float(params["atr_lower"]),
            "atr_upper": float(params["atr_upper"]),
            "me_accel_abs_min": 0.0,
            "me_cvd_min": 0.0,
        }
    if template == "bpc":
        return {
            "bpc_pullback_score_min": float(params["bpc_pullback_score_min"]),
            "bpc_pullback_depth_max": float(params["bpc_pullback_depth_max"]),
            "bpc_recovery_min": float(params["bpc_recovery_min"]),
        }
    raise ValueError(f"不支持的 template: {template}")


def build_override_prefilter(
    prod_prefilter_path: Path,
    output_path: Path,
    params: Dict[str, float],
    *,
    template: str | None = None,
) -> Path:
    base = yaml.safe_load(prod_prefilter_path.read_text(encoding="utf-8")) or {}
    tpl = (template or _infer_template_from_params(params)).lower()
    if tpl == "unknown":
        tpl = detect_locked_template(base)
    norm = _normalize_params_for_template(params, tpl)
    tuned = apply_locked_thresholds(base, template=tpl, **norm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(tuned, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path
