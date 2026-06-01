"""File + SQLite index for monitor runs (CMS / local compare)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULES = PROJECT_ROOT / "config" / "monitoring" / "schedules.yaml"
DEFAULT_INDEX = PROJECT_ROOT / "results" / "monitoring" / "index.json"
DEFAULT_REGISTRY_DB = PROJECT_ROOT / "results" / "rd_registry.sqlite"

MONITOR_EVENT_DDL = """
CREATE TABLE IF NOT EXISTS monitor_event (
  id            TEXT PRIMARY KEY,
  cadence       TEXT,
  source        TEXT,
  strategy      TEXT,
  status        TEXT,
  detail_json   TEXT,
  report_path   TEXT,
  run_ts        TEXT,
  output_dir    TEXT,
  ts            TEXT
);
"""


def repo_root() -> Path:
    return PROJECT_ROOT


def load_schedules(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_SCHEDULES
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_monitoring_index(root: Optional[Path] = None) -> Dict[str, Any]:
    """Load results/monitoring/index.json (CMS cadence summary)."""
    base = root or PROJECT_ROOT
    path = base / "results" / "monitoring" / "index.json"
    if not path.is_file():
        return {"cadences": {}, "updated_at": None}
    data = _read_json(path)
    return data if isinstance(data, dict) else {"cadences": {}, "updated_at": None}


def init_registry_db(db_path: Optional[Path] = None) -> Path:
    p = db_path or DEFAULT_REGISTRY_DB
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        conn.executescript(MONITOR_EVENT_DDL)
        conn.commit()
    finally:
        conn.close()
    return p


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_glob(parent: Path, pattern: str) -> Optional[Path]:
    if not parent.is_dir():
        return None
    hits = sorted(parent.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def collect_run_artifacts(output_dir: Path) -> Dict[str, Any]:
    """Gather heartbeat + latest watchdog/drift JSON under a manifest output_dir."""
    out: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "heartbeat": None,
        "watchdog_report": None,
        "drift_report": None,
    }
    hb = output_dir / "heartbeat.json"
    if hb.is_file():
        out["heartbeat"] = _read_json(hb)

    wd = _latest_glob(output_dir / "watchdog", "**/report.json")
    if wd:
        out["watchdog_report"] = str(wd)
        out["watchdog"] = _read_json(wd)

    dr = _latest_glob(output_dir / "drift", "**/drift_report.json")
    if dr:
        out["drift_report"] = str(dr)
        out["drift"] = _read_json(dr)

    return out


def update_monitoring_index(
    *,
    cadence: str,
    run_ts: str,
    exit_code: int,
    output_dir: Path,
    manifest_path: str,
    index_path: Optional[Path] = None,
) -> Path:
    """Merge cadence run into results/monitoring/index.json (file index for CMS)."""
    idx_path = index_path or DEFAULT_INDEX
    if not idx_path.is_absolute():
        idx_path = (PROJECT_ROOT / idx_path).resolve()
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Any] = {}
    if idx_path.is_file():
        index = _read_json(idx_path) or {}

    artifacts = collect_run_artifacts(output_dir)
    hb = artifacts.get("heartbeat") or {}
    status = str(hb.get("status") or ("ALERT" if exit_code else "OK"))

    cadence_row = {
        "cadence": cadence,
        "run_ts": run_ts,
        "status": status,
        "exit_code": int(exit_code),
        "manifest": manifest_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in artifacts.items() if k not in ("watchdog", "drift")},
    }
    if artifacts.get("watchdog"):
        cadence_row["watchdog_any_alert"] = bool(
            (artifacts["watchdog"] or {}).get("any_alert")
        )
    if artifacts.get("drift"):
        cadence_row["drift_any_alert"] = bool(
            (artifacts["drift"] or {}).get("any_alert")
        )

    cadences = index.setdefault("cadences", {})
    if not isinstance(cadences, dict):
        cadences = {}
        index["cadences"] = cadences
    cadences[cadence] = cadence_row
    index["updated_at"] = datetime.now(timezone.utc).isoformat()

    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_link = idx_path.parent / f"latest_{cadence}.json"
    latest_link.write_text(json.dumps(cadence_row, indent=2), encoding="utf-8")
    return idx_path


def upsert_monitor_events_from_run(
    *,
    cadence: str,
    run_ts: str,
    output_dir: Path,
    db_path: Optional[Path] = None,
) -> int:
    """Insert per-strategy rows into monitor_event from watchdog/drift reports."""
    db = init_registry_db(db_path)
    artifacts = collect_run_artifacts(output_dir)
    now = datetime.now(timezone.utc).isoformat()
    rows: List[tuple] = []

    wd = artifacts.get("watchdog") or {}
    for r in wd.get("reports") or []:
        if not isinstance(r, dict):
            continue
        strat = str(r.get("strategy", ""))
        if not strat:
            continue
        st = "ALERT" if r.get("any_alert") else "OK"
        eid = f"{cadence}:{run_ts}:watchdog:{strat}"
        rows.append(
            (
                eid,
                cadence,
                "watchdog",
                strat,
                st,
                json.dumps(r, ensure_ascii=False),
                artifacts.get("watchdog_report"),
                run_ts,
                str(output_dir),
                now,
            )
        )
    fh = wd.get("factor_health") or {}
    if fh.get("any_alert"):
        eid = f"{cadence}:{run_ts}:watchdog:factor_health"
        rows.append(
            (
                eid,
                cadence,
                "watchdog",
                "_factor_health",
                "ALERT",
                json.dumps(fh, ensure_ascii=False),
                artifacts.get("watchdog_report"),
                run_ts,
                str(output_dir),
                now,
            )
        )

    dr = artifacts.get("drift") or {}
    for r in dr.get("report") or []:
        if not isinstance(r, dict):
            continue
        strat = str(r.get("strategy", ""))
        if not strat:
            continue
        st = "ALERT" if r.get("any_alert") else "OK"
        eid = f"{cadence}:{run_ts}:drift:{strat}"
        rows.append(
            (
                eid,
                cadence,
                "drift",
                strat,
                st,
                json.dumps(r, ensure_ascii=False),
                artifacts.get("drift_report"),
                run_ts,
                str(output_dir),
                now,
            )
        )

    if not rows:
        return 0

    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO monitor_event
            (id, cadence, source, strategy, status, detail_json, report_path, run_ts, output_dir, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def index_monitor_run(
    *,
    cadence: str,
    run_ts: str,
    exit_code: int,
    output_dir: Path,
    manifest_path: str,
    registry_db: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Update file index + SQLite after a scheduled manifest run."""
    idx = update_monitoring_index(
        cadence=cadence,
        run_ts=run_ts,
        exit_code=exit_code,
        output_dir=output_dir,
        manifest_path=manifest_path,
        index_path=index_path,
    )
    n = upsert_monitor_events_from_run(
        cadence=cadence,
        run_ts=run_ts,
        output_dir=output_dir,
        db_path=registry_db,
    )
    return {"index_path": str(idx), "sqlite_rows": n}
