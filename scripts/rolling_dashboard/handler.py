"""Threading HTTP handler: dashboard routes, JSON APIs, packaged static assets, ``results/`` files."""

from __future__ import annotations

import json
import os
import shutil
import sys
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .browse import render_browse_page
from .constants import (
    DASHBOARD_ASSET_PREFIX,
    PACKAGE_DIR,
    dashboard_api_cache_ttl_s,
    dashboard_visibility,
)
from .response_cache import DashboardAPICaches
from .dashboard_cards_slice import build_dashboard_cards_slice_html
from .dashboard_render import (
    render_dashboard,
    render_dashboard_hub,
    render_pipeline_run_page,
)
from .paths import (
    _safe_browse_target,
    bulk_delete_paths,
    build_ledger_detail_json,
    resolve_dashboard_run_dir,
)
from .scan import list_flat_run_paths, list_incomplete_rolling_paths
from .stats import build_layer_stats_for_dashboard
from . import pipeline_jobs
from . import rd_http
from .rd_render import render_rd_page

_ALLOWED_ASSETS = frozenset(
    {"dashboard.css", "dashboard.js", "pipeline_run.js", "rd.css", "rd-page.js"}
)


def _pipeline_post_allowed(client_host: str) -> bool:
    """默认仅本机可触发管线；设 ``ROLLING_DASHBOARD_PIPELINE_REMOTE=1`` 放开。"""
    if os.environ.get("ROLLING_DASHBOARD_PIPELINE_REMOTE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    ch = (client_host or "").strip()
    if ch in ("127.0.0.1", "::1"):
        return True
    if ch.startswith("::ffff:") and ch.rsplit(":", 1)[-1] == "127.0.0.1":
        return True
    return False


def _parse_nonneg_int(raw: str | None, default: int, *, upper: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(0, min(v, upper))


def _parse_positive_int(raw: str | None, default: int, *, upper: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(1, min(v, upper))


def _dashboard_asset_file(url_path: str) -> Path | None:
    """Serve only whitelisted files from ``static/``."""
    pfx = DASHBOARD_ASSET_PREFIX.rstrip("/")
    if not url_path.startswith(pfx):
        return None
    rel = url_path[len(pfx) :].lstrip("/")
    if rel not in _ALLOWED_ASSETS:
        return None
    cand = (PACKAGE_DIR / "static" / rel).resolve()
    try:
        cand.relative_to((PACKAGE_DIR / "static").resolve())
    except ValueError:
        return None
    return cand if cand.is_file() else None


def _redirect_location(location_path: str, query: str) -> str:
    if query:
        return f"{location_path}?{query}"
    return location_path


def _api_cache_control(ttl_s: float) -> str:
    if ttl_s <= 0:
        return "no-store"
    return f"private, max-age={max(1, int(ttl_s))}"


def build_request_handler(results_root: Path):
    root = results_root.resolve()
    _ttl = dashboard_api_cache_ttl_s()
    caches = DashboardAPICaches(_ttl)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(root), **kwargs)

        def _invalidate_api_cache(self) -> None:
            caches.invalidate_all()

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
            )
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query or "")
            strategy_f = (qs.get("strategy") or [None])[0]
            q = (qs.get("q") or [None])[0]

            asset = _dashboard_asset_file(path)
            if asset is not None:
                data = asset.read_bytes()
                ct = (
                    "text/css; charset=utf-8"
                    if asset.suffix.lower() == ".css"
                    else "application/javascript; charset=utf-8"
                )
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                # pipeline_run.js 与页面 DOM 强耦合；长缓存会导致旧脚本 + 新 HTML → 功能缺失。
                if asset.name == "pipeline_run.js":
                    cc = "no-store, max-age=0, must-revalidate"
                else:
                    cc = "public, max-age=3600"
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/browse" or path.startswith("/browse/"):
                rel_u = ""
                if path.startswith("/browse/"):
                    rel_u = path[len("/browse/") :].strip("/")
                tgt = _safe_browse_target(root, rel_u)
                if tgt is None:
                    self.send_error(403)
                    return
                if not tgt.is_dir():
                    self.send_error(404)
                    return
                body = render_browse_page(root, tgt, rel_u).encode("utf-8")
                self._send_html(body)
                return

            if path == "/rd":
                body = render_rd_page().encode("utf-8")
                self._send_html(body)
                return

            vis = dashboard_visibility()
            qstr = parsed.query or ""

            if path == "/dashboard/research":
                if not vis["research"]:
                    self.send_error(404)
                    return
                body = render_dashboard(
                    root, strategy_filter=strategy_f, q=q, page="research"
                ).encode("utf-8")
                self._send_html(body)
                return

            if path == "/dashboard/research/pipeline":
                if not vis["research"]:
                    self.send_error(404)
                    return
                body = render_pipeline_run_page(root).encode("utf-8")
                self._send_html(body)
                return

            if path == "/dashboard/prod":
                if not vis["prod"]:
                    self.send_error(404)
                    return
                body = render_dashboard(
                    root, strategy_filter=strategy_f, q=q, page="prod"
                ).encode("utf-8")
                self._send_html(body)
                return

            if path == "/dashboard/all":
                loc = _redirect_location("/dashboard", qstr)
                self.send_response(302)
                self.send_header("Location", loc)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if path in ("/", "/dashboard", "/index.html"):
                if not vis["research"] and not vis["prod"]:
                    body = render_dashboard_hub(root).encode("utf-8")
                    self._send_html(body)
                    return
                if vis["research"] and not vis["prod"]:
                    loc = _redirect_location("/dashboard/research", qstr)
                    self.send_response(302)
                    self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if vis["prod"] and not vis["research"]:
                    loc = _redirect_location("/dashboard/prod", qstr)
                    self.send_response(302)
                    self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = render_dashboard_hub(root).encode("utf-8")
                self._send_html(body)
                return

            rd_resp = rd_http.dispatch_rd_get(path, qstr)
            if rd_resp is not None:
                status, payload, ctype = rd_resp
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/dashboard-cards.html":
                kind = (qs.get("kind") or ["rolling"])[0]
                if kind not in ("rolling", "flat"):
                    self.send_error(400)
                    return
                offset = _parse_nonneg_int(
                    (qs.get("offset") or ["0"])[0], 0, upper=200000
                )
                limit = _parse_positive_int(
                    (qs.get("limit") or ["80"])[0], 80, upper=500
                )
                strat_raw = (qs.get("strategy") or [None])[0]
                strategy_tab = None if strat_raw in (None, "", "__all__") else strat_raw
                qv = (qs.get("q") or [None])[0]
                adopt_raw = (qs.get("adopt_buttons") or ["0"])[0]
                adopt_btns = str(adopt_raw or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                ck = (
                    f"c|{root}|{kind}|{offset}|{limit}|"
                    f"{strategy_tab or ''}|{qv or ''}|{int(adopt_btns)}"
                )
                hit_cards = caches.cards.get(ck)
                if hit_cards is not None:
                    raw, total, next_off = hit_cards
                else:
                    frag, total, next_off = build_dashboard_cards_slice_html(
                        root,
                        kind=kind,
                        offset=offset,
                        limit=limit,
                        strategy_tab=strategy_tab,
                        q=qv,
                        show_adopt_buttons=adopt_btns,
                    )
                    raw = frag.encode("utf-8")
                    caches.cards.set(ck, (raw, total, next_off))
                cc = _api_cache_control(_ttl)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("X-Total-Count", str(total))
                self.send_header("X-Next-Offset", str(next_off))
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/dashboard-stats.json":
                sk = f"s|{root}|{strategy_f or ''}|{q or ''}"
                payload = caches.json_bytes.get(sk)
                if payload is None:
                    stats_obj = build_layer_stats_for_dashboard(
                        root, strategy_filter=strategy_f, q=q
                    )
                    payload = json.dumps(stats_obj, ensure_ascii=False).encode("utf-8")
                    caches.json_bytes.set(sk, payload)
                cc = _api_cache_control(_ttl)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/incomplete-rolling.json":
                ik = f"i|{root}|{strategy_f or ''}|{q or ''}"
                payload = caches.json_bytes.get(ik)
                if payload is None:
                    paths = list_incomplete_rolling_paths(
                        root, strategy_filter=strategy_f, q=q
                    )
                    payload = json.dumps(
                        {"count": len(paths), "paths": paths},
                        indent=2,
                        ensure_ascii=False,
                    ).encode("utf-8")
                    caches.json_bytes.set(ik, payload)
                cc = _api_cache_control(_ttl)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/flat-run-paths.json":
                fk = f"f|{root}|{strategy_f or ''}|{q or ''}"
                payload = caches.json_bytes.get(fk)
                if payload is None:
                    paths = list_flat_run_paths(root, strategy_filter=strategy_f, q=q)
                    payload = json.dumps(
                        {"count": len(paths), "paths": paths},
                        indent=2,
                        ensure_ascii=False,
                    ).encode("utf-8")
                    caches.json_bytes.set(fk, payload)
                cc = _api_cache_control(_ttl)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/ledger-detail.json":
                ledger_param = (qs.get("ledger") or [""])[0]
                ledger_rel = unquote(ledger_param)
                lk = f"l|{root}|{ledger_rel}"
                payload = caches.json_bytes.get(lk)
                if payload is None:
                    detail = build_ledger_detail_json(root, ledger_rel)
                    payload = json.dumps(detail, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    )
                    caches.json_bytes.set(lk, payload)
                cc = _api_cache_control(_ttl)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cc)
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/pipeline-run/jobs":
                run_only_raw = (
                    (qs.get("running_only") or qs.get("running") or ["1"])[0]
                    .strip()
                    .lower()
                )
                running_only = run_only_raw not in ("0", "false", "no", "all")
                items = pipeline_jobs.list_jobs_status_json(
                    root, running_only=running_only, limit=100
                )
                out = json.dumps(
                    {"ok": True, "jobs": items},
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(out)
                return

            if path == "/api/pipeline-run/status":
                jid = (qs.get("id") or [""])[0].strip()
                if not jid:
                    out = json.dumps(
                        {"ok": False, "error": "missing_id"}, ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(out)
                    return
                st = pipeline_jobs.job_status_json(jid, root)
                if not st:
                    out = json.dumps(
                        {"ok": False, "error": "not_found"}, ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(out)
                    return
                out = json.dumps(
                    {"ok": True, "job": st}, indent=2, ensure_ascii=False
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(out)
                return

            if path == "/api/pipeline-run/log":
                jid = (qs.get("id") or [""])[0].strip()
                if not jid:
                    out = json.dumps(
                        {"ok": False, "error": "missing_id"}, ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(out)
                    return
                j = pipeline_jobs.get_job(jid, root)
                if not j:
                    out = json.dumps(
                        {"ok": False, "error": "not_found"}, ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(out)
                    return
                mb_raw = (qs.get("max_bytes") or ["96000"])[0]
                max_b = _parse_nonneg_int(mb_raw, 96000, upper=500_000)
                text = pipeline_jobs.read_log_tail(root, j.log_path, max_bytes=max_b)
                out = json.dumps(
                    {"ok": True, "tail": text, "log_path": j.log_path},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(out)
                return

            if path == "/api/bpc-research-configs.json":
                proj = pipeline_jobs.resolve_project_root(root)
                items = pipeline_jobs.list_bpc_research_configs(proj)
                out = json.dumps(
                    {"ok": True, "configs": items},
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(out)
                return

            return SimpleHTTPRequestHandler.do_GET(self)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path not in (
                "/api/delete-run",
                "/api/bulk-delete",
                "/api/pipeline-run",
                "/api/rd/refresh",
            ):
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n > 2_000_000:
                self.send_error(413)
                return
            try:
                raw = self.rfile.read(n) if n else b"{}"
                payload = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                out = json.dumps(
                    {"ok": False, "error": "bad_json"}, ensure_ascii=False
                ).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            if path == "/api/pipeline-run":
                if not pipeline_jobs.pipeline_run_enabled():
                    out = json.dumps(
                        {"ok": False, "error": "pipeline_run_disabled"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if not _pipeline_post_allowed(self.client_address[0]):
                    out = json.dumps(
                        {"ok": False, "error": "forbidden_non_loopback"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(403)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                job, err = pipeline_jobs.start_pipeline_job(root, payload)
                if err == "disabled":
                    out = json.dumps(
                        {"ok": False, "error": "pipeline_run_disabled"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if err == "missing_auto_research_pipeline":
                    out = json.dumps(
                        {"ok": False, "error": err}, ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if err or job is None:
                    out = json.dumps(
                        {"ok": False, "error": err or "start_failed"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                out = json.dumps(
                    {
                        "ok": True,
                        "job": pipeline_jobs.job_status_json(job.job_id, root),
                    },
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            if path == "/api/rd/refresh":
                body = rd_http.handle_rd_refresh()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/bulk-delete":
                raw_paths = payload.get("paths")
                if not isinstance(raw_paths, list):
                    out = json.dumps(
                        {"ok": False, "error": "paths_must_be_list"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                paths_in = [str(x) for x in raw_paths[:500]]
                result = bulk_delete_paths(root, paths_in)
                if result.get("deleted"):
                    self._invalidate_api_cache()
                out = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            ledger = str(payload.get("ledger") or "")
            target = resolve_dashboard_run_dir(root, ledger)
            if target is None:
                out = json.dumps(
                    {"ok": False, "error": "invalid_ledger"},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            try:
                shutil.rmtree(target)
                self._invalidate_api_cache()
            except OSError as exc:
                out = json.dumps(
                    {"ok": False, "error": str(exc)},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            out = json.dumps(
                {"ok": True, "deleted": ledger}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler
