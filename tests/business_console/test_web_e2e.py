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
        "app.config",
        "app.routers.trade_map",
        "app.routers.bus",
        "app.routers.health",
        "app.routers.constitution",
        "app.routers.spot",
        "app.routers.links",
    ):
        monkeypatch.setattr(f"{mod}.SETTINGS", console_settings)

    import uvicorn
    from app.main import app

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
        elig = page.locator("#eligibilityBody").text_content()
        assert "can_buy" in elig
        browser.close()
