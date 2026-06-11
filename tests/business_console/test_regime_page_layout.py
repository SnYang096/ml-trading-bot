"""Regime CMS page layout — React SPA regression."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (FRONTEND_SRC / rel).read_text(encoding="utf-8")


def test_regime_page_uses_vertical_table_layout() -> None:
    tsx = _read("pages/Regime/RegimePage.tsx")
    assert "Regime Ops" in tsx
    assert "data-table" in tsx
    assert "regime_path" in tsx
    assert tsx.index("data-table") < tsx.index("regime_path")


def test_global_css_does_not_apply_bare_main_flex_row() -> None:
    css = _read("styles/global.css")
    assert not re.search(r"(?m)^main\s*\{", css)


def test_trade_map_page_uses_column_flex_not_global_main() -> None:
    css = _read("pages/TradeMap/TradeMapPage.module.css")
    assert "flex-direction: column" in css
    assert "display: flex" in css


def test_data_table_is_full_width_in_global_css() -> None:
    css = _read("styles/global.css")
    assert ".data-table" in css
    block = css.split(".data-table {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in block


def test_regime_page_marks_config_column() -> None:
    tsx = _read("pages/Regime/RegimePage.tsx")
    assert "regime_source" in tsx
    assert "regime_path" in tsx


def test_regime_route_serves_spa(client) -> None:
    r = client.get("/regime")
    assert r.status_code == 200
    assert 'id="root"' in r.text
