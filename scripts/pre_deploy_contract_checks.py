#!/usr/bin/env python3
"""Pre-deploy contract checks from research ``pre_deploy_replay.yaml`` ``contract_checks``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def run_pre_deploy_contract_checks(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    strategies_root: Path,
    project_root: Optional[Path] = None,
    predictions_by_strategy: Optional[Dict[str, Path]] = None,
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
            st["checks"]["plateau_stability"] = {
                "ok": True,
                "detail": "deferred — run plateau_stability.py / regime_drift_monitor.py manually",
            }

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
