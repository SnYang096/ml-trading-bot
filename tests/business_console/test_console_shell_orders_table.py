"""Orders table — marker deep-link attributes in React Orders page."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PAGE = REPO_ROOT / "frontend" / "src" / "pages" / "Orders" / "OrdersPage.tsx"


def test_orders_page_links_markers_to_trade_map():
    tsx = ORDERS_PAGE.read_text(encoding="utf-8")
    assert "marker_id" in tsx
    assert "/trade-map" in tsx
    assert "在地图查看" in tsx
