"""Read-only regime calibration / drift status for CMS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from mlbot_console.services.strategy_registry import (
    account_layer_label,
    get_live_console_strategies,
    strategy_account_layer,
)


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
    candidates: List[Path] = []
    for pattern in ("**/drift_report.json", "**/report.json"):
        candidates.extend(base.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_drift_document(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = doc.get("strategies")
    if rows is None:
        rows = doc.get("report")
    out: Dict[str, Dict[str, Any]] = {}
    for item in rows or []:
        if isinstance(item, dict) and item.get("strategy"):
            out[str(item["strategy"])] = item
    return out


def _summarize_drift(item: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Human-readable drift status + detail for the Regime ops table."""
    if not item:
        return "—", "未找到 drift 报告（运行 scripts/regime_drift_monitor.py）"
    if item.get("skipped"):
        return "跳过", str(item.get("skipped") or "")
    items = item.get("items") or []
    if not items:
        if item.get("any_alert"):
            return "告警", "无监测项"
        return "未监测", "regime.yaml 无 last_calibration.plateaus"
    statuses = [str(i.get("status") or "") for i in items]
    if any(s == "DRIFT" for s in statuses):
        drift_feats = [
            str(i.get("feature") or "?")
            for i in items
            if i.get("status") == "DRIFT"
        ]
        return "漂移", ", ".join(drift_feats[:4])
    if any(s in {"MISSING_FEATURE", "INSUFFICIENT_DATA"} for s in statuses):
        bad = [
            f"{i.get('feature') or '?'}:{i.get('status')}"
            for i in items
            if i.get("status") in {"MISSING_FEATURE", "INSUFFICIENT_DATA"}
        ]
        return "数据不足", ", ".join(bad[:4])
    return "正常", f"{len(items)} 项 plateau 在带内"


def _discover_strategy_slugs(strategies_root: Path) -> List[str]:
    """Constitution-enabled strategies that exist under strategies_root."""
    slugs: List[str] = []
    seen: set[str] = set()
    for meta in get_live_console_strategies():
        sid = str(meta.get("id") or "").strip().lower()
        if not sid or sid in seen:
            continue
        arch = strategies_root / sid / "archetypes"
        if not arch.is_dir():
            continue
        seen.add(sid)
        slugs.append(sid)
    if slugs:
        return slugs
    if not strategies_root.is_dir():
        return ["tpc"]
    for child in sorted(strategies_root.iterdir()):
        if not child.is_dir():
            continue
        arch = child / "archetypes"
        if arch.is_dir() and (
            (arch / "regime.yaml").is_file() or (arch / "prefilter.yaml").is_file()
        ):
            slugs.append(child.name.lower())
    return slugs or ["tpc"]


def _resolve_regime_source(
    strategies_root: Path, slug: str
) -> Tuple[Path, Dict[str, Any], str, bool]:
    """
    Return (path, config, source_label, present).

    B·Trend uses archetypes/regime.yaml; A·Spot / C·Multi-leg often embed regime in prefilter.yaml.
    """
    arch = strategies_root / slug / "archetypes"
    regime_path = arch / "regime.yaml"
    if regime_path.is_file():
        return regime_path, _load_regime_yaml(regime_path), "regime.yaml", True
    pre_path = arch / "prefilter.yaml"
    if pre_path.is_file():
        pre = _load_regime_yaml(pre_path)
        if isinstance(pre.get("regime"), dict):
            return pre_path, pre, "prefilter.yaml · regime", True
        if pre.get("rules"):
            return pre_path, pre, "prefilter.yaml · rules", True
        return pre_path, pre, "prefilter.yaml", False
    return regime_path, {}, "缺失", False


def _count_regime_rules(data: Dict[str, Any], source_label: str) -> int:
    rules = data.get("rules") or []
    n = len(rules) if isinstance(rules, list) else 0
    if "regime" in source_label and isinstance(data.get("regime"), dict):
        n += 1
    return n


def _allowed_sides_for(slug: str, data: Dict[str, Any]) -> List[str]:
    sides = data.get("allowed_sides")
    if isinstance(sides, list) and sides:
        return [str(s) for s in sides]
    layer = strategy_account_layer(slug)
    if layer == "spot":
        return ["long"]
    return ["long", "short"]


def _last_calibration_for_display(
    data: Dict[str, Any], source_label: str
) -> Dict[str, Any]:
    lc = data.get("last_calibration")
    if isinstance(lc, dict) and lc:
        return lc
    notes: List[str] = []
    regime = data.get("regime")
    if isinstance(regime, dict):
        entry = regime.get("entry_chop_min")
        exit_b = regime.get("exit_chop_below")
        if entry is not None and exit_b is not None:
            notes.append(f"chop≥{entry} 退出<{exit_b}")
        elif regime.get("entry_feature"):
            notes.append(str(regime.get("entry_feature")))
    ev = data.get("last_multileg_evaluation")
    if isinstance(ev, dict) and ev.get("run_id"):
        notes.append(f"eval {ev.get('run_id')}")
    for rule in data.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        rat = rule.get("rationale") or rule.get("lock_reason")
        if rat:
            notes.append(str(rat)[:60])
            break
    if notes:
        return {"notes": " · ".join(notes)}
    return {}


def fetch_regime_ops_snapshot(
    strategies_root: Path,
    *,
    project_root: Path,
    strategies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Per-strategy regime summary + optional latest drift monitor row."""
    slugs = strategies or _discover_strategy_slugs(strategies_root)
    drift_path = _latest_drift_report(project_root)
    drift_by_strategy: Dict[str, Dict[str, Any]] = {}
    if drift_path and drift_path.is_file():
        try:
            drift_doc = json.loads(drift_path.read_text(encoding="utf-8"))
            drift_by_strategy = _parse_drift_document(drift_doc)
        except json.JSONDecodeError:
            pass

    meta_by_id = {s["id"]: s for s in get_live_console_strategies()}

    rows: List[Dict[str, Any]] = []
    for slug in slugs:
        path, data, source_label, present = _resolve_regime_source(strategies_root, slug)
        meta = meta_by_id.get(slug) or {}
        layer = meta.get("account_layer") or strategy_account_layer(slug)
        drift_item = drift_by_strategy.get(slug)
        drift_status, drift_detail = _summarize_drift(drift_item)
        rows.append(
            {
                "strategy": slug,
                "account_layer": layer,
                "account_layer_title": meta.get("title")
                or account_layer_label(layer),
                "regime_path": str(path),
                "regime_source": source_label,
                "present": present,
                "n_rules": _count_regime_rules(data, source_label),
                "allowed_sides": _allowed_sides_for(slug, data),
                "allowed_regimes": list(data.get("allowed_regimes") or []),
                "last_calibration": _last_calibration_for_display(data, source_label),
                "drift": drift_item,
                "drift_status": drift_status,
                "drift_detail": drift_detail,
                "drift_report_path": str(drift_path) if drift_path else None,
            }
        )

    layer_order = {"trend": 0, "spot": 1, "multi_leg": 2}
    rows.sort(
        key=lambda r: (layer_order.get(str(r.get("account_layer")), 9), r.get("strategy"))
    )
    return rows


def regime_drift_meta(project_root: Path) -> Dict[str, Any]:
    """Latest drift report path/time for API meta (not duplicated on every row)."""
    drift_path = _latest_drift_report(project_root)
    if not drift_path or not drift_path.is_file():
        return {"drift_report_path": None, "drift_generated_at": None}
    generated_at: Optional[str] = None
    try:
        doc = json.loads(drift_path.read_text(encoding="utf-8"))
        generated_at = doc.get("generated_at")
    except json.JSONDecodeError:
        pass
    return {
        "drift_report_path": str(drift_path),
        "drift_generated_at": generated_at,
    }
