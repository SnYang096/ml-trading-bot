"""R&D experiment API on local rolling_dashboard (/rd)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.rolling_dashboard_server import build_request_handler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _serve(handler_cls, port: int = 0):
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server


def test_rd_experiments_list_api(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/rd/experiments")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert isinstance(body["data"], list)
        assert body["meta"]["count"] == len(body["data"])
        assert body["meta"]["count"] >= 10
    finally:
        server.shutdown()


def test_rd_experiment_detail_known(tmp_path: Path) -> None:
    exp_id = "20260531_tpc_gate_validate"
    if not (PROJECT_ROOT / "config" / "experiments" / exp_id).is_dir():
        pytest.skip("missing fixture experiment")

    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rd/experiment/{exp_id}"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        data = body["data"]
        assert data["id"] == exp_id
        assert data["has_decision"] is True
        assert "decision_text" in data
    finally:
        server.shutdown()


def test_rd_page_served(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/rd")
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "R&D 实验" in html
        assert "/api/rd/experiments" not in html
        assert "rd-page.js" in html
    finally:
        server.shutdown()


def test_rd_refresh_post(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rd/refresh",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["data"]["refreshed"] is True
    finally:
        server.shutdown()
