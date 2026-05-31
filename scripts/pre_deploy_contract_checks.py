#!/usr/bin/env python3
"""Pre-deploy contract checks from research ``pre_deploy_replay.yaml`` ``contract_checks``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LAYER_YAML = {
    "regime": "regime.yaml",
    "prefilter": "prefilter.yaml",
    "gate": "gate.yaml",
    "entry": "entry_filters.yaml",
}
_ARCHETYPE_LAYERS = tuple(_LAYER_YAML.values())

_ENTRY_OP_TO_DENY = {
    "<=": "gt",
    "<": "gt",
    "le": "gt",
    ">=": "lt",
    ">": "lt",
    "ge": "lt",
}


def _load_contract_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("contract_checks")
    return block if isinstance(block, dict) else {}


def _check_regime_yaml(
    strategy: str,
    strategies_root: Path,
) -> Tuple[bool, str]:
    path = strategies_root / strategy / "archetypes" / "regime.yaml"
    if not path.is_file():
        return False, f"missing regime.yaml: {path}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return False, f"regime.yaml parse error: {exc}"
    rules = data.get("rules") or []
    if not rules:
        return False, "regime.yaml has no rules (empty regime)"
    return True, "ok"


def _run_strict_locked_features(
    strategy: str,
    *,
    predictions: Path,
    config_root: Path,
    results_root: Path,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "posthoc_layer_effectiveness.py"),
        "--strategies",
        strategy,
        "--predictions",
        str(predictions),
        "--config-root",
        str(config_root),
        "--results-root",
        str(results_root),
        "--strict-locked-features",
    ]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    src_scripts = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts'}"
    env["PYTHONPATH"] = (
        f"{src_scripts}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_scripts
    )
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "") + (proc.stderr or "")
        return False, tail[-2000:] if tail else f"exit {proc.returncode}"
    return True, "ok"


def _collect_locked_threshold_rules(
    node: Any, *, inherited_locked: bool = False
) -> List[Dict[str, Any]]:
    """Collect locked rules with numeric thresholds from archetype yaml."""
    rules: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        locked = bool(node.get("locked")) or inherited_locked
        if locked and "feature" in node and "operator" in node and "value" in node:
            try:
                float(node["value"])
                rules.append(
                    {
                        "feature": str(node["feature"]),
                        "operator": str(node["operator"]),
                        "value": float(node["value"]),
                    }
                )
            except (TypeError, ValueError):
                pass
        for key in ("rules", "conditions", "any_of"):
            for child in node.get(key) or []:
                rules.extend(
                    _collect_locked_threshold_rules(child, inherited_locked=locked)
                )
    elif isinstance(node, list):
        for child in node:
            rules.extend(
                _collect_locked_threshold_rules(
                    child, inherited_locked=inherited_locked
                )
            )
    return rules


def _plateau_entry_for_rule(
    plateaus: List[Any], feature: str, operator: str
) -> Optional[Dict[str, Any]]:
    for entry in plateaus:
        if not isinstance(entry, dict):
            continue
        if (
            str(entry.get("feature", "")) == feature
            and str(entry.get("operator", "")) == operator
        ):
            return entry
    return None


def _check_yaml_plateau_ranges(
    strategy: str,
    strategies_root: Path,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Verify locked thresholds still fall within last_calibration.plateau ranges."""
    items: List[Dict[str, Any]] = []
    any_drift = False
    arch = strategies_root / strategy / "archetypes"
    for fname in _ARCHETYPE_LAYERS:
        path = arch / fname
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            items.append({"file": fname, "status": "ERROR", "detail": str(exc)})
            any_drift = True
            continue
        plateaus = ((data.get("last_calibration") or {}).get("plateaus")) or []
        if not plateaus:
            continue
        from scripts.plateau_stability import plateau_range_from_dict

        locked = _collect_locked_threshold_rules(data)
        for rule in locked:
            entry = _plateau_entry_for_rule(plateaus, rule["feature"], rule["operator"])
            if entry is None:
                continue
            plateau = plateau_range_from_dict(entry.get("plateau"))
            if plateau is None:
                continue
            val = rule["value"]
            in_range = plateau.start <= val <= plateau.end
            status = "OK" if in_range else "DRIFT"
            if not in_range:
                any_drift = True
            items.append(
                {
                    "file": fname,
                    "feature": rule["feature"],
                    "operator": rule["operator"],
                    "value": val,
                    "plateau_start": plateau.start,
                    "plateau_end": plateau.end,
                    "status": status,
                }
            )
    return not any_drift, items


def _check_gate_robustness_on_locked_rules(
    strategy: str,
    strategies_root: Path,
    features_parquet: Path,
    *,
    label_col: str = "success_no_rr_extreme",
    min_overall_score: float = 0.0,
    layers: Optional[List[str]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run compute_robustness_score on locked rules (gate deny semantics)."""
    from src.research.stat_kernels.robustness import (
        UnifiedOptimizationConfig,
        compute_robustness_score,
    )

    df = pd.read_parquet(features_parquet)
    items: List[Dict[str, Any]] = []
    any_weak = False
    arch = strategies_root / strategy / "archetypes"
    cfg = UnifiedOptimizationConfig()
    layer_names = layers or ["gate"]

    for layer in layer_names:
        fname = _LAYER_YAML.get(layer, f"{layer}.yaml")
        path = arch / fname
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for rule in _collect_locked_threshold_rules(data):
            feat = rule["feature"]
            if feat not in df.columns:
                items.append(
                    {
                        "file": fname,
                        "feature": feat,
                        "status": "MISSING_FEATURE",
                    }
                )
                any_weak = True
                continue
            deny_op = _ENTRY_OP_TO_DENY.get(rule["operator"])
            if deny_op is None:
                continue
            use_label = label_col
            work = df
            if use_label in df.columns and df[use_label].dtype == bool:
                work = df.assign(_is_good=df[use_label].astype(int))
                use_label = "_is_good"
            elif use_label not in df.columns:
                items.append(
                    {
                        "file": fname,
                        "feature": feat,
                        "status": "MISSING_LABEL",
                        "label_col": label_col,
                    }
                )
                continue
            score = compute_robustness_score(
                work,
                feat,
                deny_op,
                rule["value"],
                label_col=use_label,
                config=cfg,
            )
            ok = score.overall_score >= min_overall_score
            if not ok:
                any_weak = True
            items.append(
                {
                    "file": fname,
                    "feature": feat,
                    "operator": rule["operator"],
                    "threshold": rule["value"],
                    "deny_operator": deny_op,
                    "overall_score": score.overall_score,
                    "status": "OK" if ok else "LOW_ROBUSTNESS",
                }
            )
    return not any_weak, items


def _check_plateau_stability(
    strategy: str,
    strategies_root: Path,
    plateau_cfg: Dict[str, Any],
    *,
    features_parquet: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Plateau stability: yaml range check + optional gate robustness on parquet."""
    detail_parts: List[str] = []
    meta: Dict[str, Any] = {}
    alerts: List[str] = []

    yaml_ok, yaml_items = _check_yaml_plateau_ranges(strategy, strategies_root)
    meta["yaml_items"] = yaml_items
    if yaml_items:
        drift = [i for i in yaml_items if i.get("status") == "DRIFT"]
        if drift:
            alerts.extend(
                f"{strategy}:{i['file']}:{i['feature']} value outside plateau"
                for i in drift
            )
            detail_parts.append(
                f"{len(drift)} threshold(s) outside last_calibration plateau"
            )
        else:
            detail_parts.append(
                f"{len(yaml_items)} locked rule(s) within plateau baseline"
            )
    else:
        detail_parts.append(
            "no last_calibration.plateaus baseline (yaml check skipped)"
        )

    robust_ok = True
    if features_parquet is not None and features_parquet.is_file():
        min_score = float(plateau_cfg.get("min_robustness_score", 0.0))
        label_col = str(plateau_cfg.get("label_col", "success_no_rr_extreme"))
        robust_ok, robust_items = _check_gate_robustness_on_locked_rules(
            strategy,
            strategies_root,
            features_parquet,
            label_col=label_col,
            min_overall_score=min_score,
            layers=plateau_cfg.get("robustness_layers"),
        )
        meta["robustness_items"] = robust_items
        weak = [i for i in robust_items if i.get("status") == "LOW_ROBUSTNESS"]
        if weak:
            alerts.extend(
                f"{strategy}:{i['file']}:{i['feature']} robustness={i['overall_score']:.3f}"
                for i in weak
            )
            detail_parts.append(f"{len(weak)} rule(s) below min robustness score")
        elif robust_items:
            detail_parts.append(
                f"gate robustness OK on {len(robust_items)} locked rule(s)"
            )

    ok = yaml_ok and robust_ok and not alerts
    return ok, "; ".join(detail_parts), meta


def _check_cross_regime_evidence(
    strategy: str,
    cross_cfg: Dict[str, Any],
    *,
    project_root: Path,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Require variant-grid evidence for recent + bull (or explicit windows)."""
    index_path = cross_cfg.get("experiment_index")
    if index_path:
        idx = Path(str(index_path))
        if not idx.is_absolute():
            idx = (project_root / idx).resolve()
    else:
        idx = (
            project_root
            / "results"
            / strategy
            / "experiments"
            / "EXPERIMENT_INDEX.json"
        )
    required_windows = cross_cfg.get("required_windows") or ["recent", "bull"]
    meta: Dict[str, Any] = {"index": str(idx), "required_windows": required_windows}

    if not idx.is_file():
        return False, f"missing experiment index: {idx}", meta

    import json

    raw = json.loads(idx.read_text(encoding="utf-8"))
    experiments = (
        raw
        if isinstance(raw, list)
        else raw.get("experiments") or raw.get("runs") or []
    )
    if not isinstance(experiments, list):
        return False, "experiment index has no experiments list", meta

    found: Dict[str, bool] = {w: False for w in required_windows}
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        tags = exp.get("tags") or exp.get("windows") or []
        name = str(exp.get("name") or exp.get("run_id") or "")
        for w in required_windows:
            w_l = str(w).lower()
            if w_l in name.lower() or w_l in [str(t).lower() for t in tags]:
                found[w] = True
            win = exp.get("window") or exp.get("calendar_window") or ""
            if w_l in str(win).lower():
                found[w] = True

    missing = [w for w, ok in found.items() if not ok]
    meta["found"] = found
    if missing:
        return (
            False,
            f"missing cross-regime variant-grid evidence for: {', '.join(missing)}",
            meta,
        )
    return True, f"cross-regime evidence OK ({', '.join(required_windows)})", meta


def run_pre_deploy_contract_checks(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    strategies_root: Path,
    project_root: Optional[Path] = None,
    predictions_by_strategy: Optional[Dict[str, Path]] = None,
    features_parquet_by_strategy: Optional[Dict[str, Path]] = None,
    results_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run contract_checks block; return summary with per-strategy status."""
    root = project_root or PROJECT_ROOT
    contract = _load_contract_cfg(cfg)
    if not contract:
        return {"enabled": False, "status": "SKIP", "strategies": {}}

    strategies_root = strategies_root.resolve()
    results_root = (results_root or (root / "results")).resolve()
    predictions_by_strategy = predictions_by_strategy or {}
    features_parquet_by_strategy = features_parquet_by_strategy or {}

    locked_cfg = contract.get("locked_features") or {}
    locked_enabled = bool(
        isinstance(locked_cfg, dict) and locked_cfg.get("enabled", False)
    )
    on_missing = str(locked_cfg.get("on_missing", "BLOCKED")).upper()
    regime_required = bool((contract.get("regime_yaml") or {}).get("required", False))

    per: Dict[str, Any] = {}
    blocked: List[str] = []
    alerts: List[str] = []

    for strat in strategies:
        st: Dict[str, Any] = {"status": "PASS", "checks": {}}
        if regime_required:
            ok, msg = _check_regime_yaml(strat, strategies_root)
            st["checks"]["regime_yaml"] = {"ok": ok, "detail": msg}
            if not ok:
                st["status"] = "BLOCKED"
                blocked.append(f"{strat}: regime_yaml — {msg}")

        if locked_enabled:
            pred = predictions_by_strategy.get(strat)
            if pred is None or not Path(pred).is_file():
                detail = "predictions path missing for strict locked-features check"
                st["checks"]["locked_features"] = {"ok": False, "detail": detail}
                if on_missing == "BLOCKED":
                    st["status"] = "BLOCKED"
                    blocked.append(f"{strat}: locked_features — {detail}")
                else:
                    st["status"] = "ALERT"
                    alerts.append(f"{strat}: locked_features — {detail}")
            else:
                ok, msg = _run_strict_locked_features(
                    strat,
                    predictions=pred,
                    config_root=strategies_root,
                    results_root=results_root,
                )
                st["checks"]["locked_features"] = {"ok": ok, "detail": msg[:500]}
                if not ok:
                    if on_missing == "BLOCKED":
                        st["status"] = "BLOCKED"
                        blocked.append(f"{strat}: locked_features")
                    else:
                        st["status"] = "ALERT"
                        alerts.append(f"{strat}: locked_features")

        plateau_cfg = contract.get("plateau_stability") or {}
        if isinstance(plateau_cfg, dict) and plateau_cfg.get("enabled"):
            feat_pq = features_parquet_by_strategy.get(strat)
            if feat_pq is None:
                pred = predictions_by_strategy.get(strat)
                if pred is not None and Path(pred).is_file():
                    feat_pq = pred
            ok, detail, meta = _check_plateau_stability(
                strat,
                strategies_root,
                plateau_cfg,
                features_parquet=feat_pq,
            )
            st["checks"]["plateau_stability"] = {
                "ok": ok,
                "detail": detail,
                **meta,
            }
            on_drift = str(plateau_cfg.get("on_drift_outside_plateau", "ALERT")).upper()
            if not ok:
                msg = f"{strat}: plateau_stability — {detail}"
                if on_drift == "BLOCKED":
                    st["status"] = "BLOCKED"
                    blocked.append(msg)
                else:
                    if st["status"] == "PASS":
                        st["status"] = "ALERT"
                    alerts.append(msg)

        cross_cfg = contract.get("cross_regime_evidence") or {}
        if isinstance(cross_cfg, dict) and cross_cfg.get("enabled"):
            ok, detail, meta = _check_cross_regime_evidence(
                strat, cross_cfg, project_root=root
            )
            st["checks"]["cross_regime_evidence"] = {
                "ok": ok,
                "detail": detail,
                **meta,
            }
            on_missing = str(cross_cfg.get("on_missing", "BLOCKED")).upper()
            if not ok:
                msg = f"{strat}: cross_regime_evidence — {detail}"
                if on_missing == "BLOCKED":
                    st["status"] = "BLOCKED"
                    blocked.append(msg)
                else:
                    if st["status"] == "PASS":
                        st["status"] = "ALERT"
                    alerts.append(msg)

        per[strat] = st

    overall = "BLOCKED" if blocked else ("ALERT" if alerts else "PASS")
    return {
        "enabled": True,
        "status": overall,
        "blocked": blocked,
        "alerts": alerts,
        "strategies": per,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run pre_deploy contract_checks")
    p.add_argument("--config", required=True, help="Pipeline YAML with contract_checks")
    p.add_argument("--strategy", "-s", action="append", required=True)
    p.add_argument(
        "--strategies-root",
        default="config/strategies",
    )
    p.add_argument("--predictions", action="append", default=[], help="strat=path")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    pred_map: Dict[str, Path] = {}
    for item in args.predictions:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        pred_map[k.strip()] = Path(v)

    summary = run_pre_deploy_contract_checks(
        cfg=cfg,
        strategies=args.strategy,
        strategies_root=Path(args.strategies_root),
        predictions_by_strategy=pred_map,
    )
    import json

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary.get("status") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
