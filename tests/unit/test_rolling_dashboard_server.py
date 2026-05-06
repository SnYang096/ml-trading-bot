"""rolling_dashboard_server：HTML 内嵌 JSON、策略 Tab、API 路径与 ledger 明细逻辑回归测试。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

from scripts.rolling_dashboard_server import (
    build_layer_stats_for_dashboard,
    bulk_delete_paths,
    build_ledger_detail_json,
    build_request_handler,
    list_flat_run_paths,
    list_incomplete_rolling_paths,
)
from scripts.rolling_dashboard.constants import PACKAGE_DIR
from scripts.rolling_dashboard.dashboard_cards_slice import (
    build_dashboard_cards_slice_html,
)
from scripts.rolling_dashboard.dashboard_render import (
    render_dashboard_hub,
    render_pipeline_run_page,
)
from scripts.rolling_dashboard_server import _render_dashboard


def test_dashboard_stats_api_payload_roundtrip(tmp_path: Path) -> None:
    """统计走独立 JSON 接口，键名可含任意字符，不再嵌入 HTML。"""
    (tmp_path / "bpc" / "x" / "_rolling_sim" / "20260101_120000").mkdir(parents=True)
    stats = build_layer_stats_for_dashboard(tmp_path, strategy_filter=None, q=None)
    raw = json.dumps(stats, ensure_ascii=False)
    assert json.loads(raw) == stats


def test_dashboard_html_has_tab_controller(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    html = _render_dashboard(root, strategy_filter=None, q=None)
    js = (PACKAGE_DIR / "static" / "dashboard.js").read_text(encoding="utf-8")
    assert 'id="strategy-tabs"' in html
    assert "/__dashboard__/dashboard.css" in html
    assert "/__dashboard__/dashboard.js" in html
    assert 'id="ledger-list-rolling"' in html
    assert "layer-stats" not in html
    assert "/api/dashboard-stats.json" in js
    assert "/api/dashboard-cards.html" in js
    assert "applyPartitionView" in js
    assert "activeStrategyTab" in js
    assert "/api/flat-run-paths.json" in js
    assert 'href="/browse"' in html
    assert "浏览 results 根目录" in html


def test_dashboard_hub_includes_scope_links(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    html = render_dashboard_hub(tmp_path)
    assert "/dashboard/research" in html
    assert "/dashboard/prod" in html
    assert "/dashboard/all" not in html


def test_flat_cards_include_adopt_buttons_when_flag(tmp_path: Path) -> None:
    d = tmp_path / "bpc" / "turbo-rolling-sim" / "me" / "20260101_120000"
    d.mkdir(parents=True)
    (d / "report.json").write_text("{}", encoding="utf-8")
    frag, _, _ = build_dashboard_cards_slice_html(
        tmp_path,
        kind="flat",
        offset=0,
        limit=5,
        strategy_tab=None,
        q=None,
        show_adopt_buttons=True,
    )
    assert "btn-adopt-cmd" in frag


def test_dashboard_card_quick_links_include_trading_maps(tmp_path: Path) -> None:
    """快捷链接行含 stitched / continuous 等，无需重复大绿钮。"""
    d = tmp_path / "bpc" / "turbo-rolling-sim" / "_rolling_sim" / "20260101_120000"
    d.mkdir(parents=True)
    (d / "trading_map_stitched.html").write_text("<html/>", encoding="utf-8")
    (d / "stitched_summary.json").write_text("{}", encoding="utf-8")

    frag, _, _ = build_dashboard_cards_slice_html(
        tmp_path,
        kind="rolling",
        offset=0,
        limit=20,
        strategy_tab=None,
        q=None,
    )
    assert "btn-map" not in frag
    assert "打开交易地图" not in frag
    assert (
        'href="/bpc/turbo-rolling-sim/_rolling_sim/20260101_120000/trading_map_stitched.html"'
        in frag
        and ">stitched</a>" in frag
    )


def test_root_path_is_dashboard_not_browse(tmp_path: Path) -> None:
    """内部合并渲染（page=all）仍用于回归；浏览器入口仅为研究/上线两页。"""
    root = tmp_path / "results"
    root.mkdir()
    html = _render_dashboard(root, strategy_filter=None, q=None, page="all")
    assert "实验看板" in html or "合并视图" in html


def test_research_dashboard_has_no_embedded_pipeline_runner(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    html = _render_dashboard(root, strategy_filter=None, q=None, page="research")
    assert 'id="pipeline-runner-panel"' not in html


def test_pipeline_run_page_includes_config_ui(tmp_path: Path) -> None:
    html = render_pipeline_run_page(tmp_path)
    assert "bpc-config-select" in html
    assert "/dashboard/research/pipeline" in html
    assert "/api/bpc-research-configs.json" in (
        PACKAGE_DIR / "static" / "pipeline_run.js"
    ).read_text(encoding="utf-8")


def test_estimate_progress_failed_not_full_bar() -> None:
    from scripts.rolling_dashboard.pipeline_jobs import estimate_progress_from_log

    r = estimate_progress_from_log("Prefilter done line\n", job_status="failed")
    assert r["pct"] < 100
    r2 = estimate_progress_from_log("ok\n", job_status="done")
    assert r2["pct"] == 100


def test_ledger_detail_quick_links(tmp_path: Path) -> None:
    rel = "me/turbo-rolling-sim/_rolling_sim/20260202_101010"
    d = tmp_path / rel
    d.mkdir(parents=True)
    (d / "trading_map_stitched.html").write_text("<html/>", encoding="utf-8")
    (d / "stitched_summary.json").write_text("{}", encoding="utf-8")

    data = build_ledger_detail_json(tmp_path, rel.replace("\\", "/"))
    assert "error" not in data
    hrefs = [x["href"] for x in data.get("quick_links", [])]
    assert any("trading_map_stitched.html" in h for h in hrefs)


def test_incomplete_rolling_includes_corrupt_stitched_summary(tmp_path: Path) -> None:
    """损坏或空的 stitched_summary 也应进入一键删除列表。"""
    roll_parent = tmp_path / "bpc" / "turbo-rolling-sim" / "_rolling_sim"
    roll_parent.mkdir(parents=True)
    ld = roll_parent / "20260707_101010"
    ld.mkdir()
    (ld / "stitched_summary.json").write_text("{bad", encoding="utf-8")
    inc = list_incomplete_rolling_paths(tmp_path, strategy_filter=None, q=None)
    assert any("20260707_101010" in p for p in inc)


def test_bulk_delete_accepts_trimmed_path(tmp_path: Path) -> None:
    """POST 路径带首尾空白时仍能解析并删除。"""
    roll = tmp_path / "x" / "pipe" / "_rolling_sim" / "20260808_111111"
    roll.mkdir(parents=True)
    rel = roll.relative_to(tmp_path).as_posix()
    out = bulk_delete_paths(tmp_path, [f"  {rel}  "])
    assert out["ok"]
    assert len(out["deleted"]) == 1
    assert not roll.exists()


def test_list_flat_and_incomplete(tmp_path: Path) -> None:
    flat = tmp_path / "research_history" / "me" / "20260303_121212"
    flat.mkdir(parents=True)
    (flat / "report.json").write_text("{}", encoding="utf-8")

    roll_parent = tmp_path / "bpc" / "turbo-rolling-sim" / "_rolling_sim"
    roll_parent.mkdir(parents=True)
    bad_roll = roll_parent / "20260404_131313"
    bad_roll.mkdir()
    # 无 stitched_summary → incomplete rolling

    flats = list_flat_run_paths(tmp_path, strategy_filter=None, q=None)
    assert any("20260303_121212" in p for p in flats)

    inc = list_incomplete_rolling_paths(tmp_path, strategy_filter=None, q=None)
    assert any("20260404_131313" in p for p in inc)


def _serve(handler_cls, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server


def test_http_flat_run_paths_and_ledger_detail(tmp_path: Path) -> None:
    """线程内嵌 HTTP 服务器：flat-run-paths / ledger-detail 可访问。"""
    flat = tmp_path / "research_history" / "x" / "20260505_141414"
    flat.mkdir(parents=True)
    (flat / "x.txt").write_text("a", encoding="utf-8")

    roll = tmp_path / "bpc" / "pipe" / "_rolling_sim" / "20260506_151515"
    roll.mkdir(parents=True)
    (roll / "trading_map_continuous.html").write_text("x", encoding="utf-8")

    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/flat-run-paths.json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["count"] >= 1
        assert any("20260505_141414" in p for p in body["paths"])

        rel = "bpc/pipe/_rolling_sim/20260506_151515"
        q = quote(rel, safe="")
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ledger-detail.json?ledger={q}"
        )
        with urllib.request.urlopen(req2, timeout=5) as resp:
            detail = json.loads(resp.read().decode("utf-8"))
        assert detail.get("quick_links")

        req_stats = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/dashboard-stats.json"
        )
        with urllib.request.urlopen(req_stats, timeout=5) as rs:
            st = json.loads(rs.read().decode("utf-8"))
        assert "__all__" in st
    finally:
        server.shutdown()


def test_http_browse_lists_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "me").mkdir()
    (tmp_path / "me" / "readme.txt").write_text("x", encoding="utf-8")

    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/browse")
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "results 目录浏览" in html
        assert "me" in html
        assert 'href="/browse/me"' in html or "/browse/me" in html

        inner = urllib.request.Request(f"http://127.0.0.1:{port}/browse/me")
        with urllib.request.urlopen(inner, timeout=5) as resp2:
            html2 = resp2.read().decode("utf-8")
        assert "readme.txt" in html2
    finally:
        server.shutdown()


def test_http_browse_rejects_path_traversal(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/browse/foo/../../../etc/passwd"
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=5)
        assert ei.value.code == 403
    finally:
        server.shutdown()
