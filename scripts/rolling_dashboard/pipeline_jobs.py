"""Background ``auto_research_pipeline.py`` runs from the dashboard (localhost ops only)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import PACKAGE_DIR
from src.config.strategy_layout import is_research_turbo_or_slow_yaml

_VALID_STAGES = frozenset(
    {
        "full",
        "prefilter",
        "gate",
        "entry_filter",
        "slow_snapshot",
        "execution_opt",
        "event_backtest",
        "fast_month",
        "rolling_sim",
        "pcm_joint",
        "pcm_slot_grid",
    }
)

_STRATEGIES_ROOT = Path("config/strategies")
_EXCLUDED_STRATEGY_SUBDIRS = frozenset({"bad-candidates", "tree_strategies"})

# PCM 多策略编排（与 scripts/auto_research_pipeline.DEFAULT_PCM_ORCHESTRATE 对齐）
_DEFAULT_PCM_ORCHESTRATE_REL = "config/pipelines/pcm_orchestrate_2h.yaml"

# Ordered phases (substring match on recent log); pct upper bound for that phase while running.
_PROGRESS_MARKERS: List[Tuple[str, int]] = [
    ("自动研究流水线", 3),
    ("数据下载已禁用", 6),
    ("数据范围", 8),
    ("下载步骤", 10),
    ("转换步骤", 12),
    ("ConfigCheck", 14),
    ("locked tuning", 18),
    ("实验配置隔离", 20),
    ("Prefilter", 30),
    ("Gate", 44),
    ("SHAP", 52),
    ("Fast Month Replay", 62),
    ("Replay:", 65),
    ("rolling_sim", 70),
    ("_rolling_sim", 72),
    ("Stitch", 88),
    ("stitched_summary", 91),
    ("report.json", 94),
    ("对比决策", 97),
    ("ADOPT", 98),
    ("KEEP", 98),
]


def resolve_project_root(results_root: Path) -> Path:
    """Infer repo root from ``…/results`` or fall back to package parents."""
    rr = results_root.resolve()
    cand = rr.parent
    if (cand / "scripts" / "auto_research_pipeline.py").is_file():
        return cand
    return PACKAGE_DIR.parents[1]


def pipeline_run_enabled() -> bool:
    return os.environ.get(
        "ROLLING_DASHBOARD_PIPELINE_RUN", "1"
    ).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strategy_slug_from_strategies_rel(rel_posix: str) -> Optional[str]:
    parts = rel_posix.split("/")
    if len(parts) < 4:
        return None
    if parts[0] != "config" or parts[1] != "strategies":
        return None
    return parts[2]


def _is_research_pipeline_yaml_rel(rel_posix: str) -> bool:
    """Only ``config/strategies/<slug>/research/**/*.yaml`` (per-strategy research pipelines)."""
    parts = rel_posix.split("/")
    if len(parts) < 5:
        return False
    if parts[0] != "config" or parts[1] != "strategies":
        return False
    slug = parts[2]
    if slug in _EXCLUDED_STRATEGY_SUBDIRS:
        return False
    return parts[3] == "research"


def list_bpc_research_configs(project_root: Path) -> List[Dict[str, str]]:
    """Research pipeline YAMLs: ``config/strategies/<slug>/research/**/*.yaml`` (and ``.yml``).

    Omits ``archetypes/``, ``features/``, etc. — only the ``research/`` subtree.
    Excludes strategy roots ``bad-candidates`` and ``tree_strategies``.
    """
    root = project_root.resolve()
    d = (root / _STRATEGIES_ROOT).resolve()
    if not d.is_dir():
        return []
    paths: List[Path] = []
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        rel = p.relative_to(root).as_posix()
        if not _is_research_pipeline_yaml_rel(rel):
            continue
        paths.append(p)
    return [
        {"name": p.name, "rel_path": p.relative_to(root).as_posix()}
        for p in sorted(paths, key=lambda x: x.relative_to(root).as_posix())
    ]


def estimate_progress_from_log(tail: str, *, job_status: str) -> Dict[str, Any]:
    """Map recent log text to a 0–100 pct and a short label (no raw log dump in UI)."""
    running = job_status == "running"
    lines = [ln.rstrip() for ln in (tail or "").splitlines() if ln.strip()]
    if not lines:
        return {
            "pct": 2 if running else 0,
            "label": "启动中…",
            "indeterminate": running,
        }

    blob = "\n".join(lines[-80:])
    best_pct = 0
    best_label = ""
    for needle, pct in _PROGRESS_MARKERS:
        if needle not in blob:
            continue
        for line in reversed(lines[-60:]):
            if needle in line:
                if pct >= best_pct:
                    best_pct = pct
                    best_label = line.strip()
                break

    if not best_label:
        best_label = lines[-1].strip()
    if len(best_label) > 140:
        best_label = best_label[:137] + "…"

    if job_status in ("failed", "interrupted"):
        pct = min(max(best_pct, 8), 95)
        lab = "已中断" if job_status == "interrupted" else (best_label or "失败")
        return {"pct": pct, "label": lab, "indeterminate": False}

    if running:
        pct = max(2, min(best_pct, 99))
        indeterminate = best_pct < 8
        return {"pct": pct, "label": best_label or "…", "indeterminate": indeterminate}

    # done
    return {"pct": 100, "label": best_label or "完成", "indeterminate": False}


def result_navigation_hints(
    strategy_dir: str, *, run_all: bool, ok: bool
) -> List[Dict[str, str]]:
    """Post-run links back into the dashboard / browse UI."""
    if not ok:
        return []
    links: List[Dict[str, str]] = []
    if run_all:
        links.append(
            {
                "label": "研究管线看板（PCM 多策略）",
                "href": "/dashboard/research",
            }
        )
        return links

    strat_q = strategy_dir.strip()
    if strat_q and strat_q != "all":
        links.append(
            {
                "label": f"研究管线看板（{strat_q}）",
                "href": f"/dashboard/research?strategy={strat_q}",
            }
        )
        links.append(
            {
                "label": f"浏览 results/{strat_q}",
                "href": f"/browse/{strat_q}",
            }
        )
    return links


@dataclass
class PipelineJob:
    job_id: str
    strategy: str
    stage: str  # display: "full" when no --stage
    run_all: bool
    config_path: Optional[str]
    skip_shap: bool
    status: str  # running | done | failed | interrupted
    started_at: str
    log_path: str  # relative to results root, posix
    returncode: Optional[int] = None
    error: Optional[str] = None
    ended_at: Optional[str] = None


_jobs_lock = threading.Lock()
_jobs: Dict[str, PipelineJob] = {}
_db_lock = threading.Lock()
_reconciled_results_roots: set[str] = set()


def _jobs_sqlite_path(results_root: Path) -> Path:
    return results_root.resolve() / ".pipeline_run_dashboard.sqlite"


def _db_connect(results_root: Path) -> sqlite3.Connection:
    p = _jobs_sqlite_path(results_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_jobs (
            job_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            stage TEXT NOT NULL,
            run_all INTEGER NOT NULL,
            config_path TEXT,
            skip_shap INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            returncode INTEGER,
            error TEXT,
            log_path TEXT NOT NULL,
            cmd_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.commit()


def _maybe_reconcile_stale_running(results_root: Path) -> None:
    """Mark DB ``running`` rows as interrupted once per results_root (after process restart)."""
    key = str(results_root.resolve())
    with _db_lock:
        if key in _reconciled_results_roots:
            return
        conn = _db_connect(results_root)
        try:
            _init_schema(conn)
            now = _utc_iso()
            msg = "dashboard 进程重启：此前 running 状态未再跟踪"
            conn.execute(
                """
                UPDATE pipeline_jobs
                SET status = 'interrupted', ended_at = ?, error = ?
                WHERE status = 'running'
                """,
                (now, msg),
            )
            conn.commit()
        finally:
            conn.close()
        _reconciled_results_roots.add(key)


def _row_to_job(row: sqlite3.Row) -> PipelineJob:
    return PipelineJob(
        job_id=str(row["job_id"]),
        strategy=str(row["strategy"]),
        stage=str(row["stage"]),
        run_all=bool(row["run_all"]),
        config_path=str(row["config_path"]) if row["config_path"] else None,
        skip_shap=bool(row["skip_shap"]),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        log_path=str(row["log_path"]),
        returncode=(int(row["returncode"]) if row["returncode"] is not None else None),
        error=str(row["error"]) if row["error"] else None,
        ended_at=str(row["ended_at"]) if row["ended_at"] else None,
    )


def _persist_job_insert(results_root: Path, job: PipelineJob, cmd: List[str]) -> None:
    _maybe_reconcile_stale_running(results_root)
    with _db_lock:
        conn = _db_connect(results_root)
        try:
            _init_schema(conn)
            conn.execute(
                """
                INSERT INTO pipeline_jobs (
                    job_id, strategy, stage, run_all, config_path, skip_shap,
                    status, started_at, ended_at, returncode, error, log_path, cmd_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.job_id,
                    job.strategy,
                    job.stage,
                    int(job.run_all),
                    job.config_path,
                    int(job.skip_shap),
                    job.status,
                    job.started_at,
                    None,
                    None,
                    None,
                    job.log_path,
                    json.dumps(cmd, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _persist_job_finalize(
    results_root: Path,
    job_id: str,
    *,
    status: str,
    ended_at: str,
    returncode: Optional[int],
    error: Optional[str],
) -> None:
    with _db_lock:
        conn = _db_connect(results_root)
        try:
            _init_schema(conn)
            conn.execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, ended_at = ?, returncode = ?, error = ?
                WHERE job_id = ?
                """,
                (status, ended_at, returncode, error, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def _fetch_job_row(results_root: Path, job_id: str) -> Optional[sqlite3.Row]:
    _maybe_reconcile_stale_running(results_root)
    with _db_lock:
        conn = _db_connect(results_root)
        try:
            _init_schema(conn)
            cur = conn.execute(
                "SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,)
            )
            return cur.fetchone()
        finally:
            conn.close()


def list_trackable_jobs(
    results_root: Path, *, running_only: bool = False, limit: int = 100
) -> List[PipelineJob]:
    _maybe_reconcile_stale_running(results_root)
    with _db_lock:
        conn = _db_connect(results_root)
        try:
            _init_schema(conn)
            if running_only:
                cur = conn.execute(
                    """
                    SELECT * FROM pipeline_jobs
                    WHERE status = 'running'
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM pipeline_jobs
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [_row_to_job(row) for row in cur.fetchall()]
        finally:
            conn.close()


def _job_to_json(j: PipelineJob) -> Dict[str, Any]:
    return {
        "id": j.job_id,
        "strategy": j.strategy,
        "stage": j.stage,
        "run_all": j.run_all,
        "config_path": j.config_path,
        "skip_shap": j.skip_shap,
        "status": j.status,
        "started_at": j.started_at,
        "ended_at": j.ended_at,
        "returncode": j.returncode,
        "error": j.error,
        "log_path": j.log_path,
        "log_url": "/" + j.log_path.replace("\\", "/"),
    }


def get_job(job_id: str, results_root: Path) -> Optional[PipelineJob]:
    with _jobs_lock:
        cached = _jobs.get(job_id)
        if cached is not None:
            return cached
    row = _fetch_job_row(results_root, job_id)
    if row is None:
        return None
    job = _row_to_job(row)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def read_log_tail(results_root: Path, rel_log: str, *, max_bytes: int = 96_000) -> str:
    path = (results_root / rel_log).resolve()
    rr = results_root.resolve()
    try:
        path.relative_to(rr)
    except ValueError:
        return ""
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


_RE_SAFE_YAML_NAME = re.compile(r"[a-zA-Z0-9_.-]+\.(yaml|yml)\Z")


def _slug_ok(s: str) -> bool:
    return all(c.isalnum() or c in "_-" for c in s) and bool(s)


def validate_payload(
    raw: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (normalized_payload, error_message)."""
    run_all = bool(raw.get("run_all"))
    strategy = str(raw.get("strategy") or "").strip().lower()
    skip_shap = bool(raw.get("skip_shap"))

    # stage: absent / empty / "full" → full pipeline (no --stage)
    stage_raw = raw.get("stage")
    stage_opt: Optional[str]
    if stage_raw is None or str(stage_raw).strip() == "":
        stage_opt = None
    else:
        s0 = str(stage_raw).strip()
        if s0.lower() == "full":
            stage_opt = None
        elif s0 not in _VALID_STAGES:
            return None, f"invalid_stage:{s0}"
        else:
            stage_opt = s0

    config_path_s: Optional[str] = None
    cp_raw = raw.get("config_path")
    if cp_raw is not None and str(cp_raw).strip():
        cp = Path(str(cp_raw).strip())
        if cp.is_absolute():
            return None, "config_path_must_be_relative"
        if ".." in cp.parts:
            return None, "config_path_invalid"
        config_path_s = str(cp).strip()

    bpc_rc = raw.get("bpc_research_config")
    if bpc_rc is not None and str(bpc_rc).strip():
        base = str(bpc_rc).strip()
        if run_all:
            return None, "bpc_research_config_conflicts_run_all"
        if not _RE_SAFE_YAML_NAME.match(base):
            return None, "invalid_bpc_research_config"
        derived = (_STRATEGIES_ROOT / "bpc" / "research" / base).as_posix()
        if config_path_s and config_path_s != derived:
            return None, "config_path_conflicts_bpc_research_config"
        config_path_s = derived

    pcm_rel = os.environ.get("ROLLING_DASHBOARD_PCM_CONFIG", "").strip()
    if not pcm_rel:
        pcm_rel = _DEFAULT_PCM_ORCHESTRATE_REL

    if run_all:
        if strategy:
            return None, "run_all_conflicts_with_strategy"
        config_path_s = pcm_rel
    else:
        if config_path_s:
            if not _is_research_pipeline_yaml_rel(config_path_s):
                return None, "config_path_not_under_strategies_research_or_excluded"
            derived_strategy = _strategy_slug_from_strategies_rel(config_path_s)
            if derived_strategy is None:
                return None, "config_path_not_under_strategies_or_excluded"
            if strategy and strategy != derived_strategy:
                return None, "strategy_mismatches_config_path"
            strategy = derived_strategy
        else:
            if not strategy or not _slug_ok(strategy):
                return None, "invalid_strategy"
        if len(strategy) > 64:
            return None, "strategy_too_long"

    if (
        config_path_s
        and stage_opt is None
        and is_research_turbo_or_slow_yaml(Path(config_path_s))
    ):
        stage_opt = "rolling_sim"

    return (
        {
            "run_all": run_all,
            "strategy": strategy,
            "stage": stage_opt,
            "config_path": config_path_s,
            "skip_shap": skip_shap,
        },
        None,
    )


def _verify_config_exists(project_root: Path, rel: str) -> bool:
    p = (project_root / rel).resolve()
    try:
        p.relative_to(project_root.resolve())
    except ValueError:
        return False
    return p.is_file()


def start_pipeline_job(
    results_root: Path, payload: Dict[str, Any]
) -> tuple[Optional[PipelineJob], Optional[str]]:
    """Start background pipeline; returns (job, error_code)."""
    if not pipeline_run_enabled():
        return None, "disabled"

    norm, err = validate_payload(payload)
    if err or not norm:
        return None, err or "invalid_payload"

    project_root = resolve_project_root(results_root)
    script = project_root / "scripts" / "auto_research_pipeline.py"
    if not script.is_file():
        return None, "missing_auto_research_pipeline"

    cfg_rel = (norm.get("config_path") or "").strip()
    stage_run: Optional[str] = norm.get("stage")
    if not norm["run_all"] and not cfg_rel and norm.get("strategy"):
        from src.config.strategy_layout import resolve_default_pipeline_config

        p, _ = resolve_default_pipeline_config(project_root, norm["strategy"], None)
        pr = project_root.resolve()
        try:
            cfg_rel = str(p.resolve().relative_to(pr))
        except ValueError:
            cfg_rel = str(p.resolve())

    if cfg_rel and stage_run is None and is_research_turbo_or_slow_yaml(Path(cfg_rel)):
        stage_run = "rolling_sim"

    if cfg_rel and not _verify_config_exists(project_root, cfg_rel):
        return None, "config_not_found"

    logs_base = results_root.resolve() / "logs"
    strat_dir = "all" if norm["run_all"] else norm["strategy"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:12]
    log_rel = f"logs/{strat_dir}/pipeline_{ts}_{job_id}.log"
    log_path = logs_base / strat_dir / f"pipeline_{ts}_{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [sys.executable, str(script)]
    if norm["run_all"]:
        cmd.append("--all")
    else:
        cmd.extend(["--strategy", norm["strategy"]])
    if cfg_rel:
        cmd.extend(["--config", cfg_rel])
    if stage_run:
        cmd.extend(["--stage", stage_run])
    if norm["skip_shap"]:
        cmd.append("--skip-shap")

    stage_disp = stage_run or "full"

    job = PipelineJob(
        job_id=job_id,
        strategy=strat_dir,
        stage=stage_disp,
        run_all=norm["run_all"],
        config_path=cfg_rel or None,
        skip_shap=norm["skip_shap"],
        status="running",
        started_at=_utc_iso(),
        log_path=log_rel.replace("\\", "/"),
    )

    env = os.environ.copy()
    pythonpath_parts = [str(project_root), str(project_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.insert(0, env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env["PYTHONUNBUFFERED"] = "1"

    meta_path = log_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "cwd": str(project_root),
                "started_at": job.started_at,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _persist_job_insert(results_root, job, cmd)

    def _worker() -> None:
        rc: Optional[int] = None
        err_msg: Optional[str] = None
        final_status = "failed"
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(f"# {' '.join(cmd)}\n# cwd={project_root}\n\n")
                lf.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                rc = proc.wait()
        except Exception as exc:  # noqa: BLE001 — surface to UI
            err_msg = str(exc)
            rc = -1
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n\n[DASHBOARD ERROR]\n{err_msg}\n")
            except OSError:
                pass
        finally:
            ended = _utc_iso()
            if rc == 0:
                final_status = "done"
                err_final: Optional[str] = None
            else:
                final_status = "failed"
                err_final = err_msg or (f"exit {rc}" if rc is not None else "error")
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j.ended_at = ended
                    j.returncode = rc
                    j.status = final_status
                    j.error = err_final
            _persist_job_finalize(
                results_root,
                job_id,
                status=final_status,
                ended_at=ended,
                returncode=rc,
                error=err_final,
            )

    with _jobs_lock:
        _jobs[job_id] = job

    th = threading.Thread(target=_worker, name=f"pipeline-{job_id}", daemon=True)
    th.start()

    return job, None


def job_status_json(job_id: str, results_root: Path) -> Optional[Dict[str, Any]]:
    j = get_job(job_id, results_root)
    if not j:
        return None
    out = dict(_job_to_json(j))
    tail = read_log_tail(results_root, j.log_path, max_bytes=96_000)
    out["progress"] = estimate_progress_from_log(tail, job_status=j.status)
    ok = j.status == "done" and (j.returncode == 0)
    out["result_links"] = result_navigation_hints(
        j.strategy,
        run_all=j.run_all,
        ok=ok,
    )
    return out


def list_jobs_status_json(
    results_root: Path, *, running_only: bool = False, limit: int = 100
) -> List[Dict[str, Any]]:
    jobs = list_trackable_jobs(results_root, running_only=running_only, limit=limit)
    out: List[Dict[str, Any]] = []
    for job in jobs:
        payload = job_status_json(job.job_id, results_root)
        if payload:
            out.append(payload)
    return out
