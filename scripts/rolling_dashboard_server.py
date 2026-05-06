#!/usr/bin/env python3
"""
本地 ``results/`` 实验看板 — **实现已迁至** ``scripts/rolling_dashboard/`` 包。

保留本文件以便原有命令与导入路径不变::

    PYTHONPATH=. python scripts/rolling_dashboard_server.py --port 8008 --root results

见 ``scripts/rolling_dashboard/server.py`` 与 ``handler.py``。
"""

from __future__ import annotations

from scripts.rolling_dashboard import (
    build_layer_stats_for_dashboard,
    bulk_delete_paths,
    build_ledger_detail_json,
    build_request_handler,
    list_flat_run_paths,
    list_incomplete_rolling_paths,
    main,
    render_dashboard,
    run_from_project,
    run_server,
)

_render_dashboard = render_dashboard

__all__ = [
    "build_layer_stats_for_dashboard",
    "bulk_delete_paths",
    "build_ledger_detail_json",
    "build_request_handler",
    "list_flat_run_paths",
    "list_incomplete_rolling_paths",
    "main",
    "render_dashboard",
    "run_from_project",
    "run_server",
    "_render_dashboard",
]

if __name__ == "__main__":
    main()
