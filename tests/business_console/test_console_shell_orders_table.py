"""Orders table — marker deep-link attributes in React Orders page."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PAGE = REPO_ROOT / "frontend" / "src" / "pages" / "Orders" / "OrdersPage.tsx"


def test_orders_page_displays_qty_with_zero_filled_fallback():
    tsx = ORDERS_PAGE.read_text(encoding="utf-8")
    shell = (REPO_ROOT / "frontend" / "src" / "lib" / "shell.ts").read_text(
        encoding="utf-8"
    )
    assert "displayOrderQty" in shell
    assert "displayOrderQty(r)" in tsx
    assert "strategyFilter" in tsx
    assert "listStrategiesForLayers" in tsx
    assert "exclude_status" in tsx
    assert "marker_id" in tsx
    assert "/trade-map" in tsx
    assert "在地图查看" in tsx
