"""
Locked Gate Utils — 慢-快分离的 Gate 治理工具。

慢变量（人工锁定）：哪些特征进 gate（locked: true）
快变量（自动优化）：阈值、方向（区间/单侧）、是否 disable

核心 API:
  - load_locked_gate_rules(gate_path) → List[dict]
  - merge_locked_gate_rules(gate_path, locked_rules) → dict
  - calibrate_locked_gate_rule(rule, df, rr_col, ...) → dict  (单规则阈值校准)
  - calibrate_all_locked_gates(gate_path, df, rr_col, ...) → List[dict]
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Signature / identity
# ---------------------------------------------------------------------------


def _gate_rule_signature(rule: Dict[str, Any]) -> str:
    """从 gate rule 中提取特征签名用于去重/匹配。"""
    rid = rule.get("id", "")
    if rid:
        return rid
    features = _extract_features_from_when(rule.get("when", {}))
    return "|".join(sorted(features))


def _extract_features_from_when(when: Dict[str, Any]) -> List[str]:
    """从 when 子句中提取所有涉及的特征名。"""
    features: List[str] = []
    if "all_of" in when:
        for sub in when["all_of"]:
            if isinstance(sub, dict):
                for k in sub:
                    if k not in ("all_of", "any_of", "min_matches"):
                        features.append(k)
    elif "any_of" in when:
        for sub in when["any_of"]:
            if isinstance(sub, dict):
                for k in sub:
                    if k not in ("all_of", "any_of", "min_matches"):
                        features.append(k)
    else:
        for k in when:
            if k not in ("all_of", "any_of", "min_matches"):
                features.append(k)
    return features


# ---------------------------------------------------------------------------
# Load / Merge
# ---------------------------------------------------------------------------


def load_locked_gate_rules(gate_path: Path) -> List[Dict[str, Any]]:
    """从 gate.yaml 加载所有 locked: true 的规则。"""
    if not gate_path.exists():
        return []
    raw = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    locked: List[Dict[str, Any]] = []
    for section in ("hard_gates", "system_safety", "guardrails"):
        for rule in raw.get(section) or []:
            if isinstance(rule, dict) and rule.get("locked"):
                locked.append(copy.deepcopy(rule))
    return locked


def merge_locked_gate_rules(
    gate_path: Path,
    locked_rules: List[Dict[str, Any]],
) -> Dict[str, int]:
    """将 locked 规则合并回 gate.yaml（保证不丢失）。

    Returns: {"added": int, "total": int}
    """
    if not locked_rules:
        return {"added": 0, "total": 0}

    raw: Dict[str, Any] = {}
    if gate_path.exists():
        raw = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}

    existing_by_sig: Dict[str, Dict[str, Any]] = {}
    for section in ("hard_gates", "system_safety", "guardrails"):
        for rule in raw.get(section) or []:
            if isinstance(rule, dict):
                existing_by_sig[_gate_rule_signature(rule)] = rule

    added = 0
    updated = 0
    target_section = "hard_gates"
    if target_section not in raw or not isinstance(raw.get(target_section), list):
        raw[target_section] = []

    for lr in locked_rules:
        sig = _gate_rule_signature(lr)
        if sig in existing_by_sig:
            existing = existing_by_sig[sig]
            for key in (
                "locked",
                "frozen",
                "promote_never_disable",
                "disabled",
                "disabled_reason",
                "lock_reason",
            ):
                if key in lr and existing.get(key) != lr.get(key):
                    existing[key] = copy.deepcopy(lr[key])
                    updated += 1
            continue
        raw[target_section].append(copy.deepcopy(lr))
        existing_by_sig[sig] = raw[target_section][-1]
        added += 1

    total = sum(
        len(raw.get(s) or []) for s in ("hard_gates", "system_safety", "guardrails")
    )

    if added > 0 or updated > 0 or not gate_path.exists():
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    return {"added": added, "updated": updated, "total": total}


# ---------------------------------------------------------------------------
# Gate Score (Youden's J) computation
# ---------------------------------------------------------------------------


def _compute_gate_score(
    deny_mask: np.ndarray,
    bad_mask: np.ndarray,
) -> Dict[str, float]:
    """计算 gate_score = tail_capture - good_deny_rate (Youden's J)。"""
    total_bad = int(bad_mask.sum())
    total_good = int((~bad_mask).sum())

    bad_in_deny = int((deny_mask & bad_mask).sum())
    good_in_deny = int((deny_mask & (~bad_mask)).sum())

    tail_capture = bad_in_deny / max(total_bad, 1)
    good_deny_rate = good_in_deny / max(total_good, 1)
    gate_score = tail_capture - good_deny_rate

    return {
        "gate_score": round(gate_score, 4),
        "tail_capture": round(tail_capture, 4),
        "good_deny_rate": round(good_deny_rate, 4),
        "deny_count": int(deny_mask.sum()),
        "total": len(deny_mask),
    }


# ---------------------------------------------------------------------------
# Single-rule threshold calibration
# ---------------------------------------------------------------------------


def _scan_single_threshold(
    col: np.ndarray,
    bad: np.ndarray,
    direction: str,
    n_steps: int = 50,
) -> Optional[Dict[str, Any]]:
    """扫描单侧阈值 (> thresh deny 或 < thresh deny)。"""
    lo, hi = float(np.nanpercentile(col, 2)), float(np.nanpercentile(col, 98))
    if hi - lo < 1e-9:
        return None
    thresholds = np.linspace(lo, hi, n_steps)
    best: Optional[Dict[str, Any]] = None
    for t in thresholds:
        if direction == "gt":
            deny = col > t
        else:
            deny = col < t
        stats = _compute_gate_score(deny, bad)
        if stats["gate_score"] > (best["gate_score"] if best else 0):
            best = {**stats, "threshold": round(float(t), 4), "direction": direction}
    return best


def _scan_range_threshold(
    col: np.ndarray,
    bad: np.ndarray,
    n_steps: int = 30,
) -> Optional[Dict[str, Any]]:
    """扫描区间阈值 (lo < x < hi deny)。"""
    lo, hi = float(np.nanpercentile(col, 2)), float(np.nanpercentile(col, 98))
    if hi - lo < 1e-9:
        return None
    grid = np.linspace(lo, hi, n_steps)
    best: Optional[Dict[str, Any]] = None
    for i, t_lo in enumerate(grid[:-3]):
        for t_hi in grid[i + 2 :]:
            deny = (col > t_lo) & (col < t_hi)
            deny_rate = deny.mean()
            if deny_rate < 0.02 or deny_rate > 0.60:
                continue
            stats = _compute_gate_score(deny, bad)
            if stats["gate_score"] > (best["gate_score"] if best else 0):
                best = {
                    **stats,
                    "threshold_low": round(float(t_lo), 4),
                    "threshold_high": round(float(t_hi), 4),
                    "direction": "range",
                }
    return best


def calibrate_locked_gate_rule(
    rule: Dict[str, Any],
    df: pd.DataFrame,
    bad_col: str = "is_bad",
    min_gate_score: float = 0.0,
) -> Dict[str, Any]:
    """对单条 locked gate 规则进行阈值校准。

    自动尝试三种模式: gt / lt / range，选择 gate_score 最高的。
    如果所有模式都 <= min_gate_score，设置 disabled: true。

    Returns: 更新后的 rule (deepcopy, 不修改原始)
    """
    out = copy.deepcopy(rule)
    features = _extract_features_from_when(rule.get("when", {}))
    if not features:
        out["disabled"] = True
        out["disabled_reason"] = "no features in when clause"
        return out

    feature = features[0]
    if feature not in df.columns:
        out["disabled"] = True
        out["disabled_reason"] = f"feature '{feature}' not in DataFrame"
        return out

    col = df[feature].values.astype(float)
    valid = ~np.isnan(col)
    if valid.sum() < 50:
        out["disabled"] = True
        out["disabled_reason"] = f"insufficient valid samples ({valid.sum()})"
        return out

    col_v = col[valid]
    bad_v = df[bad_col].values.astype(bool)[valid]

    candidates: List[Dict[str, Any]] = []

    gt_result = _scan_single_threshold(col_v, bad_v, "gt")
    if gt_result and gt_result["gate_score"] > min_gate_score:
        candidates.append(gt_result)

    lt_result = _scan_single_threshold(col_v, bad_v, "lt")
    if lt_result and lt_result["gate_score"] > min_gate_score:
        candidates.append(lt_result)

    range_result = _scan_range_threshold(col_v, bad_v)
    if range_result and range_result["gate_score"] > min_gate_score:
        candidates.append(range_result)

    if not candidates:
        out["disabled"] = True
        out["disabled_reason"] = f"gate_score <= {min_gate_score} for all directions"
        out.pop("when", None)
        out["when"] = _build_placeholder_when(feature)
        return out

    best = max(candidates, key=lambda c: c["gate_score"])

    out.pop("disabled", None)
    out.pop("disabled_reason", None)
    out["when"] = _build_when_from_calibration(feature, best)
    out["comment"] = (
        f"locked_calibration: direction={best['direction']}, "
        f"gate_score={best['gate_score']:.3f}, "
        f"tail_capture={best['tail_capture']:.3f}, "
        f"good_deny={best['good_deny_rate']:.3f}"
    )
    out["last_calibration_score"] = best["gate_score"]
    return out


def _build_when_from_calibration(
    feature: str, result: Dict[str, Any]
) -> Dict[str, Any]:
    """根据校准结果构建 when 子句。"""
    direction = result["direction"]
    if direction == "gt":
        return {feature: {"value_gt": result["threshold"]}}
    elif direction == "lt":
        return {feature: {"value_lt": result["threshold"]}}
    else:
        return {
            "all_of": [
                {feature: {"value_gt": result["threshold_low"]}},
                {feature: {"value_lt": result["threshold_high"]}},
            ]
        }


def _build_placeholder_when(feature: str) -> Dict[str, Any]:
    """构建占位 when（disabled 时保留结构）。"""
    return {feature: {"value_gt": 0.0}}


# ---------------------------------------------------------------------------
# Batch calibration
# ---------------------------------------------------------------------------


def calibrate_all_locked_gates(
    gate_path: Path,
    df: pd.DataFrame,
    bad_col: str = "is_bad",
    min_gate_score: float = 0.0,
    write_back: bool = True,
) -> List[Dict[str, Any]]:
    """对 gate.yaml 中所有 locked 规则进行阈值校准。

    非 locked 规则保持不变。
    """
    if not gate_path.exists():
        return []

    raw = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    calibrated_rules: List[Dict[str, Any]] = []

    for section in ("hard_gates", "system_safety", "guardrails"):
        rules = raw.get(section) or []
        new_rules: List[Dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                new_rules.append(rule)
                continue
            if rule.get("locked"):
                calibrated = calibrate_locked_gate_rule(
                    rule, df, bad_col=bad_col, min_gate_score=min_gate_score
                )
                new_rules.append(calibrated)
                calibrated_rules.append(calibrated)
            else:
                new_rules.append(rule)
        raw[section] = new_rules

    if write_back:
        gate_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    return calibrated_rules
