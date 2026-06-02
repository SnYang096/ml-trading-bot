#!/usr/bin/env python3
"""Run monitor steps from a YAML manifest (config/monitoring/*.yaml)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _subst_run_ts(text: str, run_ts: str) -> str:
    return text.replace("{run_ts}", run_ts)


def _resolve_path(raw: str, *, run_ts: str) -> Path:
    p = Path(_subst_run_ts(raw, run_ts))
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return data


def _window_cfg(manifest: Dict[str, Any], key: str) -> Dict[str, Any]:
    windows = manifest.get("windows") or {}
    if key not in windows:
        raise KeyError(f"window {key!r} not in manifest windows")
    row = windows[key]
    if not isinstance(row, dict):
        raise ValueError(f"window {key!r} must be a mapping")
    return row


def _run_py(script: str, argv: List[str]) -> int:
    """Legacy subprocess executor (kept for emergency / third-party steps)."""
    cmd = [sys.executable, str(PROJECT_ROOT / script), *argv]
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"}
    return int(
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False).returncode
    )


def _run_monitor_script(script: str, argv: List[str]) -> int:
    return _run_py(f"scripts/{script}", argv)


def _use_subprocess_fallback() -> bool:
    """Allow forcing the old subprocess path via env var (emergency escape hatch)."""
    return os.environ.get("MLBOT_MONITOR_FORCE_SUBPROCESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _watchdog_namespace(
    *,
    pq: Path,
    baseline: Path,
    wd_out: Path,
    strategies_csv: str,
    step_cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> "argparse.Namespace":
    """Build watchdog args from manifest step + optional manifest.watchdog_defaults."""
    import argparse

    defaults = manifest.get("watchdog_defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    cfg = {**defaults, **step_cfg}

    ic_json = str(
        cfg.get("ic_baseline_json")
        or "config/monitoring/factor_ic_baseline_tpc_20260526.json"
    )
    psi_raw = cfg.get("psi_features")
    if isinstance(psi_raw, list):
        psi_features = ",".join(str(x) for x in psi_raw)
    else:
        psi_features = str(
            psi_raw or "ema_1200_position,vol_persistence,vol_leverage_asymmetry"
        )

    return argparse.Namespace(
        window_parquet=str(pq),
        baseline_json=str(baseline),
        out_dir=str(wd_out),
        strategies=strategies_csv,
        strategies_root=str(cfg.get("strategies_root") or "config/strategies"),
        bull_share_tol=float(cfg.get("bull_share_tol", 0.10)),
        trigger_drift_tol_rel=float(cfg.get("trigger_drift_tol_rel", 0.50)),
        ic_baseline_json=ic_json,
        psi_features=psi_features,
        psi_tol=float(cfg.get("psi_tol", 0.25)),
        ic_flip_min_abs=float(cfg.get("ic_flip_min_abs", 0.02)),
    )


def execute_manifest(
    manifest: Dict[str, Any],
    *,
    config_path: Path,
    run_ts: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[int, str, Path]:
    run_ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    steps = manifest.get("steps") or []
    if not steps:
        raise ValueError("manifest has no steps")

    strategies = manifest.get("strategies") or ["bpc", "tpc", "me", "srb"]
    if isinstance(strategies, list):
        strategies_csv = ",".join(str(s) for s in strategies)
    else:
        strategies_csv = str(strategies)

    out_dir_raw = (
        manifest.get("output_dir")
        or f"results/monitoring/{manifest.get('monitor_id', 'run')}/{{run_ts}}"
    )
    out_dir = _resolve_path(str(out_dir_raw), run_ts=run_ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_paths: Dict[str, Path] = {}
    exit_code = 0

    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"each step must be a single-key dict, got: {step!r}")
        name, cfg = next(iter(step.items()))
        cfg = cfg if isinstance(cfg, dict) else {}

        if name == "export-window":
            win_key = str(cfg.get("window", "short"))
            win = _window_cfg(manifest, win_key)
            parquet = _resolve_path(
                str(
                    win.get("parquet")
                    or f"results/monitoring/window/{run_ts}/features_current_7d.parquet"
                ),
                run_ts=run_ts,
            )
            window_paths[win_key] = parquet
            argv = [
                "--bus-root",
                str(
                    cfg.get("bus_root")
                    or os.environ.get(
                        "MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus"
                    )
                ),
                "--timeframe",
                str(win.get("timeframe", "120T")),
                "--lookback-days",
                str(win.get("lookback_days", 7)),
                "--output",
                str(parquet),
            ]
            if win.get("symbols"):
                argv.extend(["--symbols", str(win["symbols"])])
            if dry_run:
                print(f"[dry-run] export-window → {parquet}")
                continue
            if _use_subprocess_fallback():
                rc = _run_py("scripts/monitoring/export_feature_bus_window.py", argv)
            else:
                from scripts.monitoring.export_feature_bus_window import (
                    export_feature_bus_window,
                )

                try:
                    export_feature_bus_window(
                        bus_root=Path(
                            cfg.get("bus_root")
                            or os.environ.get(
                                "MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus"
                            )
                        ),
                        timeframe=str(win.get("timeframe", "120T")),
                        lookback_days=int(win.get("lookback_days", 7)),
                        output=parquet,
                        symbols=str(win.get("symbols") or "") or None,
                    )
                    rc = 0
                except Exception as exc:
                    print(f"ERROR export-window: {exc}", file=sys.stderr)
                    rc = 3
            if rc != 0:
                return rc, run_ts, out_dir

        elif name == "archive-batch":
            win_key = str(cfg.get("window", "long"))
            win = _window_cfg(manifest, win_key)
            parquet = _resolve_path(
                str(
                    win.get("parquet")
                    or f"results/monitoring/window/{run_ts}/features_current_6m.parquet"
                ),
                run_ts=run_ts,
            )
            window_paths[win_key] = parquet
            argv = [
                "--strategy",
                str(cfg.get("strategy") or win.get("strategy") or "tpc"),
                "--segment",
                str(cfg.get("segment") or win.get("segment") or "recent_6m_oos"),
                "--output",
                str(parquet),
            ]
            if dry_run:
                print(f"[dry-run] archive-batch → {parquet}")
                continue
            if _use_subprocess_fallback():
                rc = _run_py("scripts/monitoring/archive_batch_window.py", argv)
            else:
                from scripts.monitoring.archive_batch_window import archive_batch_window

                try:
                    archive_batch_window(
                        strategy=str(
                            cfg.get("strategy") or win.get("strategy") or "tpc"
                        ),
                        segment=str(
                            cfg.get("segment") or win.get("segment") or "recent_6m_oos"
                        ),
                        output=parquet,
                        market_segment_path=_resolve_path(
                            "config/market_segment.yaml", run_ts=run_ts
                        ),
                    )
                    rc = 0
                except Exception as exc:
                    print(f"ERROR archive-batch: {exc}", file=sys.stderr)
                    rc = 3
            if rc != 0:
                return rc, run_ts, out_dir

        elif name == "watchdog":
            win_key = str(cfg.get("window", "short"))
            pq = window_paths.get(win_key) or _resolve_path(
                str((_window_cfg(manifest, win_key).get("parquet"))),
                run_ts=run_ts,
            )
            baseline = _resolve_path(
                str(
                    cfg.get("baseline")
                    or "config/monitoring/regime_watchdog_baseline.json"
                ),
                run_ts=run_ts,
            )
            wd_out = out_dir / "watchdog"
            argv = [
                "--strategies",
                strategies_csv,
                "--window-parquet",
                str(pq),
                "--baseline-json",
                str(baseline),
                "--out-dir",
                str(wd_out),
            ]
            if dry_run:
                print(f"[dry-run] watchdog {pq}")
                continue
            if _use_subprocess_fallback():
                wd_cfg = {**(manifest.get("watchdog_defaults") or {}), **cfg}
                extra = [
                    "--ic-baseline-json",
                    str(
                        wd_cfg.get("ic_baseline_json")
                        or "config/monitoring/factor_ic_baseline_tpc_20260526.json"
                    ),
                    "--psi-features",
                    str(
                        wd_cfg.get("psi_features")
                        or "ema_1200_position,vol_persistence,vol_leverage_asymmetry"
                    ),
                    "--psi-tol",
                    str(wd_cfg.get("psi_tol", 0.25)),
                    "--ic-flip-min-abs",
                    str(wd_cfg.get("ic_flip_min_abs", 0.02)),
                    "--bull-share-tol",
                    str(wd_cfg.get("bull_share_tol", 0.10)),
                    "--trigger-drift-tol-rel",
                    str(wd_cfg.get("trigger_drift_tol_rel", 0.50)),
                ]
                rc = _run_monitor_script("regime_watchdog.py", argv + extra)
            else:
                from scripts.regime_watchdog import run_watchdog

                ns = _watchdog_namespace(
                    pq=pq,
                    baseline=baseline,
                    wd_out=wd_out,
                    strategies_csv=strategies_csv,
                    step_cfg=cfg,
                    manifest=manifest,
                )
                rc = run_watchdog(ns)
            if rc != 0:
                exit_code = 1

        elif name == "drift":
            win_key = str(cfg.get("window", "long"))
            pq = window_paths.get(win_key) or _resolve_path(
                str((_window_cfg(manifest, win_key).get("parquet"))),
                run_ts=run_ts,
            )
            drift_out = out_dir / "drift"
            argv = [
                "--strategies",
                strategies_csv,
                "--window-parquet",
                str(pq),
                "--out-dir",
                str(drift_out),
            ]
            if cfg.get("emit_rd_loop_suggestions"):
                argv.append("--emit-rd-loop-suggestions")
            if dry_run:
                print(f"[dry-run] drift {pq}")
                continue
            if _use_subprocess_fallback():
                rc = _run_monitor_script("regime_drift_monitor.py", argv)
            else:
                import argparse
                from scripts.regime_drift_monitor import run_drift_monitor

                ns = argparse.Namespace(
                    window_parquet=str(pq),
                    out_dir=str(drift_out),
                    strategies=strategies_csv,
                    strategies_root="config/strategies",
                    drift_quantile=0.5,
                    emit_rd_loop_suggestions=bool(cfg.get("emit_rd_loop_suggestions")),
                )
                rc = run_drift_monitor(ns)
            if rc != 0:
                exit_code = 1

        else:
            raise ValueError(f"unknown manifest step: {name!r}")

    if dry_run:
        return 0, run_ts, out_dir

    short_key = "short"
    long_key = "long"
    wd_pq = window_paths.get(short_key)
    dr_pq = window_paths.get(long_key)
    if wd_pq:
        os.environ["WATCHDOG_PARQUET"] = str(wd_pq)
    if dr_pq:
        os.environ["DRIFT_PARQUET"] = str(dr_pq)

    import json

    heartbeat = {
        "task": manifest.get("monitor_id", "monitor_manifest"),
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "ALERT" if exit_code else "OK",
        "manifest": str(config_path),
        "run_ts": run_ts,
        "watchdog_parquet": str(wd_pq) if wd_pq else None,
        "drift_parquet": str(dr_pq) if dr_pq else None,
        "output_dir": str(out_dir),
    }
    (out_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, indent=2), encoding="utf-8"
    )
    print(f"monitoring manifest: {out_dir} (status={heartbeat['status']})")
    return exit_code, run_ts, out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Run monitor manifest YAML")
    p.add_argument(
        "--config",
        required=True,
        help="Manifest path (e.g. config/monitoring/weekly_rule_stack.yaml)",
    )
    p.add_argument("--run-ts", default="", help="Override {run_ts} substitution")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"ERROR: manifest not found: {cfg_path}", file=sys.stderr)
        return 3

    try:
        manifest = _load_manifest(cfg_path)
        rc, _, _ = execute_manifest(
            manifest,
            config_path=cfg_path,
            run_ts=str(args.run_ts).strip() or None,
            dry_run=bool(args.dry_run),
        )
        return rc
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
