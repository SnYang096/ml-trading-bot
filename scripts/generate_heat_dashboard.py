#!/usr/bin/env python3
"""Generate Grafana Market Heat dashboard JSON.

Follows the same pattern as generate_strategy_dashboards.py.
Output: deploy/monitoring/grafana-provisioning/dashboards/market_heat.json
"""

import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.market_heat.sector_registry import load_sector_registry

UID = "quant-market-heat"
TITLE = "🌡️ Market Heat Dashboard"

HEAT_COLORS = {
    "hot": "#73BF69",
    "warm": "#FADE2A",
    "cold": "#F2495C",
}


def _nav_links() -> list:
    return [
        {
            "title": "📊 Overview",
            "url": "/d/quant-live-overview",
            "type": "link",
            "icon": "apps",
            "targetBlank": False,
        },
        {
            "title": "📡 Signal Pipeline",
            "url": "/d/quant-signal-pipeline",
            "type": "link",
            "icon": "bolt",
            "targetBlank": False,
        },
        {
            "title": "💰 Account & Market",
            "url": "/d/quant-account-market",
            "type": "link",
            "icon": "cloud",
            "targetBlank": False,
        },
    ]


def generate_dashboard() -> dict:
    registry = load_sector_registry()
    sector_names = sorted(registry.sectors.keys())

    pid = 0

    def nid():
        nonlocal pid
        pid += 1
        return pid

    panels = []

    # ── Row 1: Market Overview ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
            "id": nid(),
            "title": "🌍 Market Overview",
            "type": "row",
        }
    )

    # Market Heat Score (big stat)
    panels.append(
        {
            "description": "Overall crypto market heat score (BTC 60% + ETH 40% weighted)",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "thresholds": {
                        "steps": [
                            {"color": HEAT_COLORS["cold"], "value": None},
                            {"color": HEAT_COLORS["warm"], "value": 0.2},
                            {"color": HEAT_COLORS["hot"], "value": 0.5},
                        ]
                    },
                    "min": 0,
                    "max": 1,
                }
            },
            "gridPos": {"h": 5, "w": 6, "x": 0, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
                "textMode": "value_and_name",
            },
            "targets": [
                {
                    "expr": 'mlbot_heat_market_score{market="crypto"}',
                    "legendFormat": "Market Heat",
                }
            ],
            "title": "🌡️ Market Heat",
            "type": "stat",
        }
    )

    # HOT sectors count
    panels.append(
        {
            "description": "Number of sectors with state = HOT (score >= 0.5)",
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": HEAT_COLORS["cold"], "value": None},
                            {"color": HEAT_COLORS["warm"], "value": 1},
                            {"color": HEAT_COLORS["hot"], "value": 3},
                        ]
                    },
                }
            },
            "gridPos": {"h": 5, "w": 4, "x": 6, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {
                    "expr": "count(mlbot_heat_sector_score >= 0.5)",
                    "legendFormat": "",
                }
            ],
            "title": "HOT Sectors",
            "type": "stat",
        }
    )

    # COLD sectors count
    panels.append(
        {
            "description": "Number of sectors with state = COLD (score < 0.2)",
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": HEAT_COLORS["hot"], "value": None},
                            {"color": HEAT_COLORS["warm"], "value": 1},
                            {"color": HEAT_COLORS["cold"], "value": 3},
                        ]
                    },
                }
            },
            "gridPos": {"h": 5, "w": 4, "x": 10, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {
                    "expr": "count(mlbot_heat_sector_score < 0.2)",
                    "legendFormat": "",
                }
            ],
            "title": "COLD Sectors",
            "type": "stat",
        }
    )

    # Market heat over time
    panels.append(
        {
            "description": "Market heat score trend",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": HEAT_COLORS["cold"], "value": None},
                            {"color": HEAT_COLORS["warm"], "value": 0.2},
                            {"color": HEAT_COLORS["hot"], "value": 0.5},
                        ],
                    },
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "scheme",
                        "thresholdsStyle": {"mode": "area"},
                    },
                }
            },
            "gridPos": {"h": 5, "w": 10, "x": 14, "y": 1},
            "id": nid(),
            "targets": [
                {
                    "expr": 'mlbot_heat_market_score{market="crypto"}',
                    "legendFormat": "Market",
                }
            ],
            "title": "📈 Market Heat Trend",
            "type": "timeseries",
        }
    )

    # ── Row 2: Sector Heat Map ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 6},
            "id": nid(),
            "title": "📊 Sector Heat Map",
            "type": "row",
        }
    )

    # Sector bar gauge (ranked horizontal)
    panels.append(
        {
            "description": "Sector heat scores ranked high to low",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": HEAT_COLORS["cold"], "value": None},
                            {"color": HEAT_COLORS["warm"], "value": 0.2},
                            {"color": HEAT_COLORS["hot"], "value": 0.5},
                        ]
                    },
                }
            },
            "gridPos": {"h": 8, "w": 10, "x": 0, "y": 7},
            "id": nid(),
            "options": {
                "displayMode": "gradient",
                "orientation": "horizontal",
                "reduceOptions": {"calcs": ["lastNotNull"]},
                "showUnfilled": True,
            },
            "targets": [
                {
                    "expr": "mlbot_heat_sector_score",
                    "legendFormat": "{{sector}}",
                }
            ],
            "title": "🏷️ Sector Scores",
            "type": "bargauge",
        }
    )

    # Sector table
    panels.append(
        {
            "description": "Sector heat detail table",
            "fieldConfig": {
                "defaults": {
                    "custom": {"align": "center"},
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "sector"},
                        "properties": [{"id": "custom.width", "value": 80}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Value"},
                        "properties": [
                            {"id": "displayName", "value": "Score"},
                            {"id": "decimals", "value": 3},
                            {
                                "id": "thresholds",
                                "value": {
                                    "steps": [
                                        {"color": HEAT_COLORS["cold"], "value": None},
                                        {"color": HEAT_COLORS["warm"], "value": 0.2},
                                        {"color": HEAT_COLORS["hot"], "value": 0.5},
                                    ]
                                },
                            },
                            {"id": "custom.displayMode", "value": "color-background"},
                        ],
                    },
                ],
            },
            "gridPos": {"h": 8, "w": 14, "x": 10, "y": 7},
            "id": nid(),
            "options": {
                "showHeader": True,
                "sortBy": [{"displayName": "Score", "desc": True}],
            },
            "targets": [
                {
                    "expr": "mlbot_heat_sector_score",
                    "legendFormat": "",
                    "format": "table",
                    "instant": True,
                }
            ],
            "title": "📋 Sector Detail",
            "type": "table",
            "transformations": [
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {
                            "Time": True,
                            "__name__": True,
                            "instance": True,
                            "job": True,
                        },
                    },
                },
            ],
        }
    )

    # Sector heat over time (all sectors)
    panels.append(
        {
            "description": "All sector scores over time for trend comparison",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "min": 0,
                    "max": 1,
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 0,
                        "showPoints": "never",
                    },
                }
            },
            "gridPos": {"h": 7, "w": 24, "x": 0, "y": 15},
            "id": nid(),
            "targets": [
                {
                    "expr": "mlbot_heat_sector_score",
                    "legendFormat": "{{sector}}",
                }
            ],
            "title": "📈 Sector Heat Trends",
            "type": "timeseries",
        }
    )

    # ── Row 3: Symbol Detail (per sector, collapsed) ──
    y_offset = 22

    for sector_name in sector_names:
        sector_info = registry.sectors[sector_name]
        symbols = sector_info.symbols

        row_panels = []

        # Symbol table for this sector
        row_panels.append(
            {
                "description": f"{sector_name} sector symbol heat detail",
                "fieldConfig": {
                    "defaults": {
                        "custom": {"align": "center"},
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "symbol"},
                            "properties": [{"id": "custom.width", "value": 80}],
                        },
                        {
                            "matcher": {"id": "byName", "options": "Value"},
                            "properties": [
                                {"id": "displayName", "value": "Score"},
                                {"id": "decimals", "value": 3},
                                {
                                    "id": "thresholds",
                                    "value": {
                                        "steps": [
                                            {
                                                "color": HEAT_COLORS["cold"],
                                                "value": None,
                                            },
                                            {
                                                "color": HEAT_COLORS["warm"],
                                                "value": 0.2,
                                            },
                                            {"color": HEAT_COLORS["hot"], "value": 0.5},
                                        ]
                                    },
                                },
                                {
                                    "id": "custom.displayMode",
                                    "value": "color-background",
                                },
                            ],
                        },
                    ],
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_offset + 1},
                "id": nid(),
                "options": {
                    "showHeader": True,
                    "sortBy": [{"displayName": "Score", "desc": True}],
                },
                "targets": [
                    {
                        "expr": f'mlbot_heat_score{{sector="{sector_name}"}}',
                        "legendFormat": "",
                        "format": "table",
                        "instant": True,
                    }
                ],
                "title": f"📋 {sector_name} Symbols",
                "type": "table",
                "transformations": [
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": {
                                "Time": True,
                                "__name__": True,
                                "instance": True,
                                "job": True,
                                "sector": True,
                            },
                        },
                    },
                ],
            }
        )

        # Symbol heat score time series for this sector
        row_panels.append(
            {
                "description": f"{sector_name} symbol heat scores over time",
                "fieldConfig": {
                    "defaults": {
                        "decimals": 3,
                        "min": 0,
                        "max": 1,
                        "custom": {
                            "drawStyle": "line",
                            "lineWidth": 1,
                            "fillOpacity": 0,
                            "showPoints": "never",
                        },
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_offset + 1},
                "id": nid(),
                "targets": [
                    {
                        "expr": f'mlbot_heat_score{{sector="{sector_name}"}}',
                        "legendFormat": "{{symbol}}",
                    }
                ],
                "title": f"📈 {sector_name} Symbol Trends",
                "type": "timeseries",
            }
        )

        # Collapsed row wrapping this sector
        panels.append(
            {
                "collapsed": True,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_offset},
                "id": nid(),
                "title": f"🏷️ {sector_name} — {sector_info.description} ({len(symbols)} symbols)",
                "type": "row",
                "panels": row_panels,
            }
        )

        y_offset += 1

    # ── Templating: $sector variable ──
    templating = {
        "list": [
            {
                "allValue": "",
                "current": {"text": "All", "value": "$__all"},
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "definition": "label_values(mlbot_heat_score, sector)",
                "hide": 0,
                "includeAll": True,
                "label": "Sector",
                "multi": True,
                "name": "sector",
                "query": "label_values(mlbot_heat_score, sector)",
                "refresh": 1,
                "regex": "",
                "type": "query",
            }
        ]
    }

    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "id": None,
        "links": _nav_links(),
        "panels": panels,
        "refresh": "5m",
        "schemaVersion": 39,
        "tags": ["trading", "quant", "market-heat"],
        "templating": templating,
        "time": {"from": "now-7d", "to": "now"},
        "timepicker": {},
        "timezone": "utc",
        "title": TITLE,
        "uid": UID,
        "version": 1,
    }


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "deploy",
        "monitoring",
        "grafana-provisioning",
        "dashboards",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "market_heat.json")
    dashboard = generate_dashboard()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=4, ensure_ascii=False)
    print(f"Generated: {path}")


if __name__ == "__main__":
    main()
