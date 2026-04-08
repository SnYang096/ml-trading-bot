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


def _locked_prefilter_features(rules: List[Any]) -> set:
    """Top-level locked rules: include ``feature`` and features inside ``any_of`` children."""
    feats: set = set()
    for r in rules:
        if not isinstance(r, dict) or not r.get("locked"):
            continue
        if r.get("feature"):
            feats.add(r["feature"])
        subs = r.get("any_of")
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict) and sub.get("feature"):
                    feats.add(sub["feature"])
    return feats


def detect_locked_template(prefilter_raw: Dict[str, Any]) -> str:
    rules = prefilter_raw.get("rules") or []
    if not isinstance(rules, list):
        return "unknown"
    feats = _locked_prefilter_features(rules)
    if {
        "fer_signed_efficiency_pct",
        "sr_strength_max",
        "dist_to_nearest_sr",
    }.issubset(feats):
        return "fer"
    # ME：ATR 带 + 加速度（signed 或 long/short 拆分）；不要求 me_cvd_alignment（精简 prefilter 时仍识别）
    if "me_atr_pct" in feats and (
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


def _apply_value_transform(value: float, transform: str) -> float:
    t = str(transform or "identity").strip().lower()
    if t in {"identity", "none"}:
        return float(value)
    if t == "abs":
        return float(abs(value))
    if t in {"negate", "neg"}:
        return float(-value)
    if t in {"neg_abs", "negative_abs"}:
        return float(-abs(value))
    raise ValueError(f"unsupported value_transform: {transform}")


def _apply_config_bindings(
    out: Dict[str, Any],
    *,
    params: Dict[str, float],
    bindings: List[Dict[str, Any]],
    strict: bool = True,
) -> Dict[str, Any]:
    rules = out.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("prefilter.yaml rules 必须为 list")
    if not bindings:
        raise ValueError("writeback bindings 为空")

    seen_match: Dict[str, bool] = {}
    seen_param: Dict[str, bool] = {}
    for b in bindings:
        if not isinstance(b, dict):
            continue
        p = str(b.get("param", "")).strip()
        if not p:
            continue
        seen_match.setdefault(p, False)
        seen_param.setdefault(p, False)
        if p in params:
            seen_param[p] = True

    for r in rules:
        if not isinstance(r, dict) or not r.get("locked"):
            continue
        targets: List[Tuple[Dict[str, Any], bool]] = [(r, False)]
        subs = r.get("any_of")
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    targets.append((sub, True))
        for node, is_sub in targets:
            feat = node.get("feature")
            op = node.get("operator")
            if not feat or not op:
                continue
            for b in bindings:
                if not isinstance(b, dict):
                    continue
                p = str(b.get("param", "")).strip()
                if not p:
                    continue
                if p not in params:
                    continue
                bf = b.get("feature")
                bo = b.get("operator")
                target = str(b.get("target", "any")).strip().lower()
                if bf and str(bf) != str(feat):
                    continue
                if bo and str(bo) != str(op):
                    continue
                if target == "rule" and is_sub:
                    continue
                if target in {"any_of", "sub"} and not is_sub:
                    continue
                transformed = _apply_value_transform(
                    float(params[p]), str(b.get("value_transform", "identity"))
                )
                node["value"] = transformed
                seen_match[p] = True

    if strict:
        missing_params = [k for k, v in seen_param.items() if not v]
        if missing_params:
            raise ValueError(
                f"writeback bindings 需要的参数缺失: {missing_params}; got={sorted(params.keys())}"
            )
        missing_match = [k for k, v in seen_match.items() if not v]
        if missing_match:
            raise ValueError(
                f"writeback bindings 未命中任何 locked 规则: {missing_match}"
            )

    return out


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
    params: Dict[str, float] | None = None,
    bindings: List[Dict[str, Any]] | None = None,
    strict_bindings: bool = True,
    template: str | None = None,
) -> Dict[str, Any]:
    out = json.loads(json.dumps(prefilter_raw))
    rules = out.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("prefilter.yaml rules 必须为 list")

    if bindings:
        return _apply_config_bindings(
            out,
            params={k: float(v) for k, v in (params or {}).items()},
            bindings=bindings,
            strict=strict_bindings,
        )

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
        # 旧版「压缩语义」ME prefilter（atr + compression_duration + …），与动量 ME 共用 template 名 me
        is_compression = any(
            isinstance(r, dict)
            and r.get("locked")
            and r.get("feature") == "compression_duration"
            for r in rules
        )
        if is_compression:
            required = {
                "atr_lower": atr_lower,
                "atr_upper": atr_upper,
                "compression_min": compression_min,
                "decay_upper": decay_upper,
                "oi_min": oi_min,
            }
            missing_params = [k for k, v in required.items() if v is None]
            if missing_params:
                raise ValueError(f"ME(compression) tuned 参数缺失: {missing_params}")
            seen = {
                "atr_lower": False,
                "atr_upper": False,
                "compression": False,
                "decay": False,
                "oi": False,
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
                elif feat == "compression_duration" and op == ">=":
                    r["value"] = float(compression_min)
                    seen["compression"] = True
                elif feat == "recent_compression_decay" and op == "<=":
                    r["value"] = float(decay_upper)
                    seen["decay"] = True
                elif feat == "oi_compression_score" and op == ">=":
                    r["value"] = float(oi_min)
                    seen["oi"] = True
            missing = [k for k, v in seen.items() if not v]
            if missing:
                raise ValueError(
                    f"prefilter.yaml 缺少必要 ME(compression) locked 规则: {missing}"
                )
            return out

        wants_cvd = any(
            isinstance(r, dict)
            and r.get("locked")
            and r.get("feature") == "me_cvd_alignment"
            for r in rules
        )
        required: Dict[str, Any] = {
            "atr_lower": atr_lower,
            "atr_upper": atr_upper,
            "me_accel_abs_min": me_accel_abs_min,
        }
        if wants_cvd:
            required["me_cvd_min"] = me_cvd_min
        missing_params = [k for k, v in required.items() if v is None]
        if missing_params:
            raise ValueError(f"ME tuned 参数缺失: {missing_params}")

        seen: Dict[str, bool] = {
            "atr_lower": False,
            "atr_upper": False,
            "accel": False,
        }
        if wants_cvd:
            seen["cvd"] = False

        def _apply_me_accel_to_rule(feat: str, op: str, rdict: Dict[str, Any]) -> None:
            if feat == "me_accel_5k_long" and op == ">=":
                rdict["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k_short" and op == ">=":
                rdict["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k" and op == ">=":
                rdict["value"] = float(abs(float(me_accel_abs_min)))
                seen["accel"] = True
            elif feat == "me_accel_5k" and op == "<=":
                rdict["value"] = float(-abs(float(me_accel_abs_min)))
                seen["accel"] = True

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
            elif feat == "me_cvd_alignment" and op == ">=":
                r["value"] = float(me_cvd_min)
                seen["cvd"] = True
            elif r.get("any_of") and isinstance(r["any_of"], list):
                for sub in r["any_of"]:
                    if not isinstance(sub, dict):
                        continue
                    sf, so = sub.get("feature"), sub.get("operator")
                    if sf and so:
                        _apply_me_accel_to_rule(str(sf), str(so), sub)
            else:
                if feat and op:
                    _apply_me_accel_to_rule(str(feat), str(op), r)
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
    bindings: List[Dict[str, Any]] | None = None,
    strict_bindings: bool = True,
    template: str | None = None,
) -> Path:
    base = yaml.safe_load(prod_prefilter_path.read_text(encoding="utf-8")) or {}
    if bindings:
        tuned = apply_locked_thresholds(
            base,
            params=params,
            bindings=bindings,
            strict_bindings=strict_bindings,
            template=template,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(tuned, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return output_path

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
