"""Packaged ``results/`` experiment dashboard (Rolling + History)."""

from __future__ import annotations

from .browse import render_browse_page
from .dashboard_cards_slice import build_dashboard_cards_slice_html
from .dashboard_render import render_dashboard
from .handler import build_request_handler
from .paths import bulk_delete_paths, build_ledger_detail_json
from .scan import list_flat_run_paths, list_incomplete_rolling_paths
from .server import main, run_from_project, run_server
from .stats import build_layer_stats_for_dashboard

# Backward-compatible alias for tests / older imports
_render_dashboard = render_dashboard

__all__ = [
    "build_dashboard_cards_slice_html",
    "build_layer_stats_for_dashboard",
    "build_request_handler",
    "bulk_delete_paths",
    "build_ledger_detail_json",
    "list_flat_run_paths",
    "list_incomplete_rolling_paths",
    "main",
    "render_browse_page",
    "render_dashboard",
    "run_from_project",
    "run_server",
    "_render_dashboard",
]
