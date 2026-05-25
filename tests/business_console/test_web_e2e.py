"""Playwright e2e against business console with fake data."""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright

pytest.importorskip("uvicorn")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(console_settings, monkeypatch):
    for mod in (
        "mlbot_console.config",
        "mlbot_console.routers.trade_map",
        "mlbot_console.routers.bus",
        "mlbot_console.routers.health",
        "mlbot_console.routers.constitution",
        "mlbot_console.routers.spot",
        "mlbot_console.routers.links",
    ):
        monkeypatch.setattr(f"{mod}.SETTINGS", console_settings)

    import uvicorn
    from mlbot_console.main import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.mark.integration
def test_trade_map_page_loads(live_server):
    url = (
        f"{live_server}/trade-map" "?from=2024-01-01T00:00:00Z&to=2024-01-02T00:00:00Z"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector("#symbolSelect option", state="attached", timeout=20000)
        page.wait_for_function(
            """() => {
              const t = document.getElementById('statusLine')?.textContent || '';
              return t.includes('bars') || t.includes('markers') || t.includes('加载中');
            }""",
            timeout=20000,
        )
        page.wait_for_function(
            "() => !document.getElementById('statusLine').textContent.includes('启动失败')",
            timeout=20000,
        )
        status = page.locator("#statusLine").text_content() or ""
        assert "bars" in status or "markers" in status
        assert page.locator("#symbolSelect").input_value() in {"ETHUSDT", "SOLUSDT"}
        assert page.locator("#appNav").text_content()
        assert "策略信号" in (page.locator("#appNav").text_content() or "")
        browser.close()


@pytest.mark.integration
def test_regime_page_main_uses_column_layout(live_server):
    """Regression: global main{display:flex} row broke Regime table (empty left gutter)."""
    url = f"{live_server}/regime"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector("#regimeBody tr", state="attached", timeout=20000)
        main = page.locator("main.regime-main")
        assert main.evaluate("el => getComputedStyle(el).flexDirection") == "column"
        widths = page.locator(".signals-table-wrap").evaluate(
            """el => {
              const main = el.closest('main');
              const r = el.getBoundingClientRect();
              const m = main.getBoundingClientRect();
              return { wrap: r.width, main: m.width, left: r.left - m.left };
            }"""
        )
        assert widths["wrap"] >= widths["main"] * 0.85
        assert widths["left"] < 40
        browser.close()
