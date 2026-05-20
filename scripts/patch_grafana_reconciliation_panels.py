#!/usr/bin/env python3
"""One-off helper: inject reconciliation panels into Grafana dashboards."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "deploy/monitoring/grafana-provisioning/dashboards"


def load(name: str) -> dict:
    return json.loads((DASH / name).read_text(encoding="utf-8"))


def save(name: str, data: dict) -> None:
    (DASH / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _stat_panel(
    *,
    pid: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 6,
    h: int = 4,
) -> dict:
    return {
        "type": "stat",
        "title": title,
        "id": pid,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
        "targets": [{"refId": "A", "expr": expr, "instant": True}],
        "fieldConfig": {
            "defaults": {
                "min": 0,
                "max": 1,
                "mappings": [
                    {
                        "type": "value",
                        "options": {
                            "0": {"text": "需人工检查", "color": "red", "index": 0},
                            "1": {"text": "正常", "color": "green", "index": 1},
                        },
                    }
                ],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": 1},
                    ],
                },
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
            "colorMode": "background",
            "graphMode": "none",
        },
    }


def patch_home() -> None:
    home = load("quant_home.json")
    recon = json.loads((DASH / "_recon_panels_home.json").read_text(encoding="utf-8"))
    shift = 12
    for panel in home["panels"]:
        g = panel.get("gridPos") or {}
        if g.get("y", 0) >= 22:
            g["y"] = g["y"] + shift
    idx = next(i for i, p in enumerate(home["panels"]) if p.get("id") == 17)
    for j, panel in enumerate(recon):
        home["panels"].insert(idx + j, panel)
    save("quant_home.json", home)


def patch_trend() -> None:
    trend = load("quant_strategy_map_trend.json")
    y0 = 66
    trend["panels"].extend(
        [
            {
                "type": "row",
                "title": "Trend · 订单对账（DB vs 交易所）",
                "id": 901,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y0},
                "collapsed": False,
            },
            {
                "type": "text",
                "title": "",
                "id": 902,
                "gridPos": {"h": 2, "w": 24, "x": 0, "y": y0 + 1},
                "options": {
                    "mode": "markdown",
                    "content": "**频率**：`MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS` 默认 **60s**（含 open + **openAlgoOrders**）。`reconciliation_ok=0` → 检查 `order_management.db` pending 行 vs 币安。",
                },
                "transparent": True,
            },
            _stat_panel(
                pid=903,
                title="Trend · 对账 OK",
                expr='min(mlbot_reconciliation_ok{job=~"$job",scope="trend"})',
                x=0,
                y=y0 + 3,
            ),
            {
                "type": "stat",
                "title": "Trend · 问题总数",
                "id": 904,
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": y0 + 3},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'sum(mlbot_reconciliation_issue_count{job=~"$job",scope="trend"})',
                        "instant": True,
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 1},
                            ],
                        }
                    },
                    "overrides": [],
                },
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                    "colorMode": "background",
                    "graphMode": "none",
                },
            },
            {
                "type": "table",
                "title": "Trend · 对账问题 by issue",
                "id": 907,
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": y0 + 7},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'mlbot_reconciliation_issue_count{job=~"$job",scope="trend"} > 0',
                        "format": "table",
                        "instant": True,
                    }
                ],
                "fieldConfig": {"defaults": {}, "overrides": []},
                "options": {"showHeader": True},
            },
            {
                "type": "timeseries",
                "title": "Trend · 对账问题趋势",
                "id": 908,
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": y0 + 7},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'sum by (issue) (mlbot_reconciliation_issue_count{job=~"$job",scope="trend"})',
                        "legendFormat": "{{issue}}",
                    }
                ],
                "fieldConfig": {"defaults": {"min": 0}, "overrides": []},
                "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
            },
        ]
    )
    save("quant_strategy_map_trend.json", trend)


def patch_hedge() -> None:
    hedge = load("quant_strategy_map_hedge.json")
    y0 = 92
    hedge["panels"].extend(
        [
            {
                "type": "row",
                "title": "Hedge · 订单对账（engine/DB vs 交易所）",
                "id": 901,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y0},
                "collapsed": False,
            },
            {
                "type": "text",
                "title": "",
                "id": 902,
                "gridPos": {"h": 2, "w": 24, "x": 0, "y": y0 + 1},
                "options": {
                    "mode": "markdown",
                    "content": "**频率**：daemon reconcile 每 **60s**（有新 bar 时）；REST backfill **60s** 修正 DB stale open。任一 issue>0 → **需人工检查**。",
                },
                "transparent": True,
            },
            _stat_panel(
                pid=903,
                title="Hedge · 对账 OK (min)",
                expr='min(mlbot_reconciliation_ok{job=~"$job",scope="hedge"})',
                x=0,
                y=y0 + 3,
            ),
            {
                "type": "stat",
                "title": "Hedge · 问题总数",
                "id": 904,
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": y0 + 3},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'sum(mlbot_reconciliation_issue_count{job=~"$job",scope="hedge"})',
                        "instant": True,
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 1},
                            ],
                        }
                    },
                    "overrides": [],
                },
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                    "colorMode": "background",
                    "graphMode": "none",
                },
            },
            {
                "type": "table",
                "title": "Hedge · 对账问题 by strategy/symbol/issue",
                "id": 906,
                "gridPos": {"h": 7, "w": 12, "x": 0, "y": y0 + 7},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'mlbot_reconciliation_issue_count{job=~"$job",scope="hedge"} > 0',
                        "format": "table",
                        "instant": True,
                    }
                ],
                "fieldConfig": {"defaults": {}, "overrides": []},
                "options": {"showHeader": True},
            },
            {
                "type": "timeseries",
                "title": "Hedge · 对账问题趋势",
                "id": 907,
                "gridPos": {"h": 7, "w": 12, "x": 12, "y": y0 + 7},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'sum by (strategy, symbol, issue) (mlbot_reconciliation_issue_count{job=~"$job",scope="hedge"})',
                        "legendFormat": "{{strategy}}/{{symbol}} {{issue}}",
                    }
                ],
                "fieldConfig": {"defaults": {"min": 0}, "overrides": []},
                "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
            },
        ]
    )
    save("quant_strategy_map_hedge.json", hedge)


def patch_system() -> None:
    system = load("quant_system.json")
    y0 = 27
    system["panels"].extend(
        [
            {
                "type": "row",
                "title": "Spot · 订单对账",
                "id": 201,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y0},
                "collapsed": False,
            },
            _stat_panel(
                pid=202,
                title="Spot · 对账 OK",
                expr='min(mlbot_reconciliation_ok{job="quant-spot-accum",scope="spot"})',
                x=0,
                y=y0 + 1,
                w=8,
            ),
            {
                "type": "table",
                "title": "Spot · 对账问题",
                "id": 203,
                "gridPos": {"h": 5, "w": 16, "x": 8, "y": y0 + 1},
                "datasource": {"type": "prometheus", "uid": "prometheus-monitoring"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": 'mlbot_reconciliation_issue_count{job="quant-spot-accum",scope="spot"} > 0',
                        "format": "table",
                        "instant": True,
                    }
                ],
                "fieldConfig": {"defaults": {}, "overrides": []},
                "options": {"showHeader": True},
            },
        ]
    )
    save("quant_system.json", system)


def main() -> None:
    patch_home()
    patch_trend()
    patch_hedge()
    patch_system()
    print("Grafana reconciliation panels patched.")


if __name__ == "__main__":
    main()
