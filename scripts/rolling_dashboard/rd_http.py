"""HTTP helpers for local R&D experiment API (``/api/rd/*``)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

from .constants import PROJECT_ROOT, experiments_root_path
from . import rd_experiments


def _ok(data: Any, *, meta: Optional[Dict[str, Any]] = None) -> bytes:
    return json.dumps(
        {"ok": True, "data": data, "meta": meta or {}},
        ensure_ascii=False,
    ).encode("utf-8")


def _err(message: str, *, code: int = 400) -> Tuple[int, bytes]:
    body = json.dumps(
        {"ok": False, "error": {"code": "error", "message": message}},
        ensure_ascii=False,
    ).encode("utf-8")
    return code, body


def handle_rd_refresh() -> bytes:
    rd_experiments.clear_experiments_cache()
    rows = rd_experiments.list_experiments()
    return _ok({"refreshed": True, "count": len(rows)})


def handle_rd_experiments_list(qs: Dict[str, List[str]]) -> bytes:
    strategy = (qs.get("strategy") or [None])[0]
    q = (qs.get("q") or [None])[0]
    since = (qs.get("since") or [None])[0]
    category = (qs.get("category") or [None])[0]
    rows = rd_experiments.list_experiments(
        strategy=strategy, q=q, since=since, category=category
    )
    return _ok(
        rows,
        meta={
            "count": len(rows),
            "experiments_root": str(experiments_root_path()),
            "repo_root": str(PROJECT_ROOT),
            "strategies": rd_experiments.list_strategies(),
        },
    )


def handle_rd_experiment_detail(experiment_id: str) -> Tuple[int, bytes]:
    row = rd_experiments.get_experiment(experiment_id)
    if not row:
        return _err(f"experiment not found: {experiment_id}", code=404)
    return 200, _ok(
        row,
        meta={
            "experiments_root": str(experiments_root_path()),
            "repo_root": str(PROJECT_ROOT),
        },
    )


def handle_rd_experiment_raw(experiment_id: str, filename: str) -> Tuple[int, bytes]:
    payload = rd_experiments.get_experiment_raw_file(experiment_id, filename)
    if not payload:
        return _err("file not found or not allowed", code=404)
    return 200, _ok(payload)


def dispatch_rd_get(path: str, query: str) -> Optional[Tuple[int, bytes, str]]:
    """Return (status, body, content_type) or None if not an RD route."""
    qs = parse_qs(query or "")

    if path == "/api/rd/experiments":
        return 200, handle_rd_experiments_list(qs), "application/json; charset=utf-8"

    if path == "/api/rd/refresh":
        return 200, handle_rd_refresh(), "application/json; charset=utf-8"

    prefix = "/api/rd/experiment/"
    if path.startswith(prefix):
        rest = path[len(prefix) :]
        if "/raw/" in rest:
            exp_id, _, filename = rest.partition("/raw/")
            if exp_id and filename:
                status, body = handle_rd_experiment_raw(exp_id, filename)
                return status, body, "application/json; charset=utf-8"
            status, body = _err("bad raw path", code=400)
            return status, body, "application/json; charset=utf-8"
        exp_id = rest.strip("/")
        if exp_id:
            status, body = handle_rd_experiment_detail(exp_id)
            return status, body, "application/json; charset=utf-8"

    return None
