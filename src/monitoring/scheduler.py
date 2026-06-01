"""Schedule monitor cadences: manifest → index → optional Telegram."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.monitoring.staleness_check import run_staleness_check
from src.monitoring.store import (
    PROJECT_ROOT,
    index_monitor_run,
    load_monitoring_index,
    load_schedules,
)
from src.monitoring.telegram import notify_cadence_result

# In-process manifest executor (the preferred path after manifest 内聚 refactor).
# Import is done here so callers can stay inside src/monitoring.
from scripts.monitoring.run_monitor_manifest import (
    _load_manifest,
    execute_manifest as _execute_manifest_raw,
)

ExecuteManifestFn = Callable[..., Tuple[int, str, Path]]


def default_execute_manifest(
    manifest: Dict[str, Any],
    *,
    config_path: Path,
    run_ts: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[int, str, Path]:
    """Default in-process manifest runner (no subprocess for known steps)."""
    return _execute_manifest_raw(
        manifest, config_path=config_path, run_ts=run_ts, dry_run=dry_run
    )


def default_load_manifest(path: Path) -> Dict[str, Any]:
    return _load_manifest(path)


def resolve_manifest_path(rel: str, *, repo_root: Path = PROJECT_ROOT) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def list_cadences(schedules_path: Optional[Path] = None) -> List[str]:
    cfg = load_schedules(schedules_path)
    raw = cfg.get("schedules") or {}
    return sorted(raw.keys()) if isinstance(raw, dict) else []


def _registry_db_path(cfg: Dict[str, Any], repo_root: Path) -> Optional[Path]:
    registry_db = os.environ.get("MLBOT_RD_REGISTRY_DB") or cfg.get("registry_db")
    if not registry_db:
        return None
    db_path = Path(str(registry_db))
    if not db_path.is_absolute():
        db_path = (repo_root / db_path).resolve()
    return db_path


def post_run_hooks(
    *,
    cadence: str,
    exit_code: int,
    schedules_path: Optional[Path],
    repo_root: Path = PROJECT_ROOT,
) -> None:
    """Telegram on business ALERT; daily cadence also runs staleness check."""
    if not os.environ.get("MLBOT_MONITOR_SKIP_TG", "").strip():
        cfg = load_schedules(schedules_path)
        db_path = _registry_db_path(cfg, repo_root)
        index = load_monitoring_index(repo_root)
        row = (index.get("cadences") or {}).get(cadence) or {}
        if isinstance(row, dict):
            notify_cadence_result(
                cadence=cadence,
                exit_code=int(exit_code),
                index_row=row,
                registry_db=db_path or (repo_root / "results/rd_registry.sqlite"),
            )

    if cadence == "daily" and not os.environ.get("MLBOT_MONITOR_SKIP_STALENESS", "").strip():
        run_staleness_check(repo_root=repo_root, schedules_path=schedules_path)


def run_cadence(
    cadence: str,
    *,
    execute_manifest: ExecuteManifestFn,
    load_manifest: Callable[[Path], Dict[str, Any]],
    schedules_path: Optional[Path] = None,
    repo_root: Path = PROJECT_ROOT,
    run_ts: Optional[str] = None,
    dry_run: bool = False,
    skip_index: bool = False,
) -> int:
    cfg = load_schedules(schedules_path)
    schedules = cfg.get("schedules") or {}
    if cadence not in schedules:
        known = ", ".join(sorted(schedules))
        raise KeyError(f"unknown cadence {cadence!r}; known: {known}")

    entry = schedules[cadence]
    if not isinstance(entry, dict):
        raise ValueError(f"schedule {cadence!r} must be a mapping")
    manifest_rel = str(entry.get("manifest") or "")
    if not manifest_rel:
        raise ValueError(f"schedule {cadence!r} missing manifest")

    manifest_path = resolve_manifest_path(manifest_rel, repo_root=repo_root)
    manifest = load_manifest(manifest_path)
    exit_code, used_ts, out_dir = execute_manifest(
        manifest,
        config_path=manifest_path,
        run_ts=run_ts,
        dry_run=dry_run,
    )

    if dry_run:
        return int(exit_code)

    if not skip_index:
        db_path = _registry_db_path(cfg, repo_root)
        index_monitor_run(
            cadence=cadence,
            run_ts=used_ts,
            exit_code=int(exit_code),
            output_dir=out_dir,
            manifest_path=str(manifest_path),
            registry_db=db_path,
        )
        print(f"indexed: cadence={cadence} run_ts={used_ts} exit={exit_code}")

    post_run_hooks(
        cadence=cadence,
        exit_code=int(exit_code),
        schedules_path=schedules_path,
        repo_root=repo_root,
    )
    return int(exit_code)


def run_all_due(
    *,
    execute_manifest: ExecuteManifestFn,
    load_manifest: Callable[[Path], Dict[str, Any]],
    cadences: Optional[List[str]] = None,
    schedules_path: Optional[Path] = None,
    repo_root: Path = PROJECT_ROOT,
    dry_run: bool = False,
) -> int:
    names = cadences or list_cadences(schedules_path)
    worst = 0
    for name in names:
        print(f"=== monitor schedule: {name} ===")
        rc = run_cadence(
            name,
            execute_manifest=execute_manifest,
            load_manifest=load_manifest,
            schedules_path=schedules_path,
            repo_root=repo_root,
            dry_run=dry_run,
        )
        worst = max(worst, rc)
    return worst
