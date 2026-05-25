"""Regime CMS page layout — regression for horizontal main flex bug."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "mlbot_console" / "static"


def test_regime_html_uses_vertical_page_shell() -> None:
    html = (STATIC / "regime.html").read_text(encoding="utf-8")
    assert 'class="page-regime"' in html
    assert 'class="regime-main"' in html
    assert 'class="signals-table-wrap"' in html
    assert html.index("regime-main") < html.index("signals-table-wrap")


def test_trade_map_css_does_not_apply_global_main_flex_row() -> None:
    css = (STATIC / "css" / "trade-map.css").read_text(encoding="utf-8")
    assert not re.search(
        r"(?m)^main\s*\{", css
    ), "bare main{} breaks non–trade-map pages"
    assert ".page-trade-map main" in css
    assert "display: flex" in css


def test_regime_main_uses_column_flex_not_row() -> None:
    css = (STATIC / "css" / "signals.css").read_text(encoding="utf-8")
    block = css.split(".page-regime .regime-main {", 1)[1].split("}", 1)[0]
    assert "flex-direction: column" in block
    assert "flex-direction: row" not in block


def test_regime_table_wrap_is_full_width() -> None:
    css = (STATIC / "css" / "signals.css").read_text(encoding="utf-8")
    assert ".page-regime .regime-main .signals-table-wrap" in css
    wrap = css.split(".page-regime .regime-main .signals-table-wrap {", 1)[1].split(
        "}", 1
    )[0]
    assert "width: 100%" in wrap


def test_regime_page_js_marks_config_column() -> None:
    js = (STATIC / "regime-page.js").read_text(encoding="utf-8")
    assert "regime-config-cell" in js


def test_regime_route_serves_html(client) -> None:
    r = client.get("/regime")
    assert r.status_code == 200
    assert "page-regime" in r.text
    assert "regimeTable" in r.text
