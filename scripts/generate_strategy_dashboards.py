#!/usr/bin/env python3
"""生成 BPC / FER / ME 三个独立 Grafana 策略监控面板 JSON"""

import json
import os

STRATEGIES = [
    {
        "name": "bpc",
        "label": "BPC",
        "emoji": "🔵",
        "color": "#3274D9",
        "color_light": "#8AB8FF",
        "uid": "quant-strategy-bpc",
    },
    {
        "name": "fer",
        "label": "FER",
        "emoji": "🟣",
        "color": "#B877D9",
        "color_light": "#D9B2FF",
        "uid": "quant-strategy-fer",
    },
    {
        "name": "me-long",
        "label": "ME",
        "emoji": "🟠",
        "color": "#FF9830",
        "color_light": "#FFCB7D",
        "uid": "quant-strategy-me",
    },
]


def _nav_links(current_uid: str) -> list:
    """导航链接 (排除当前面板)"""
    links = [
        {
            "title": "📊 Overview",
            "url": "/d/quant-engine-main",
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
    for s in STRATEGIES:
        if s["uid"] != current_uid:
            links.append(
                {
                    "title": f"{s['emoji']} {s['label']} Strategy",
                    "url": f"/d/{s['uid']}",
                    "type": "link",
                    "icon": "dashboard",
                    "targetBlank": False,
                }
            )
    return links


def _funnel_overrides(color_main: str) -> list:
    """信号漏斗颜色"""
    stages = [
        ("direction", "#8AB8FF"),
        ("gate", "#3274D9"),
        ("entry_filter", "#73BF69"),
        ("evidence", "#FADE2A"),
        ("pcm", "#FF9830"),
        ("order", "#F2495C"),
    ]
    return [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [
                {"id": "color", "value": {"mode": "fixed", "fixedColor": c}}
            ],
        }
        for name, c in stages
    ]


def generate_dashboard(strat: dict) -> dict:
    s = strat["name"]
    S = strat["label"]
    emoji = strat["emoji"]
    color = strat["color"]
    color_light = strat["color_light"]
    uid = strat["uid"]

    pid = 400  # panel id counter

    def nid():
        nonlocal pid
        pid += 1
        return pid

    panels = []

    # ── Row: Status Overview ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
            "id": nid(),
            "title": f"{emoji} {S} 状态总览",
            "type": "row",
        }
    )

    # Slot Active
    panels.append(
        {
            "description": f"{S} 当前活跃 Slot 数",
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                        ]
                    },
                    "max": 2,
                    "min": 0,
                }
            },
            "gridPos": {"h": 4, "w": 4, "x": 0, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {
                    "expr": f'mlbot_strategy_slots_active{{strategy="{s}"}}',
                    "legendFormat": "",
                }
            ],
            "title": "活跃 Slot",
            "type": "stat",
        }
    )

    # Slot Max
    panels.append(
        {
            "description": f"{S} 最大 Slot 上限",
            "fieldConfig": {
                "defaults": {
                    "thresholds": {"steps": [{"color": "blue", "value": None}]}
                }
            },
            "gridPos": {"h": 4, "w": 4, "x": 4, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "none",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {
                    "expr": f'mlbot_strategy_slots_max{{strategy="{s}"}}',
                    "legendFormat": "",
                }
            ],
            "title": "Slot 上限",
            "type": "stat",
        }
    )

    # Total Orders
    panels.append(
        {
            "description": f"{S} 累计下单数",
            "fieldConfig": {
                "defaults": {"thresholds": {"steps": [{"color": color, "value": None}]}}
            },
            "gridPos": {"h": 4, "w": 4, "x": 8, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {"expr": f'mlbot_orders_total{{strategy="{s}"}}', "legendFormat": ""}
            ],
            "title": "累计下单",
            "type": "stat",
        }
    )

    # Total Signals
    panels.append(
        {
            "description": f"{S} 累计信号数 (Evidence通过)",
            "fieldConfig": {
                "defaults": {
                    "thresholds": {"steps": [{"color": color_light, "value": None}]}
                }
            },
            "gridPos": {"h": 4, "w": 4, "x": 12, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {"expr": f'mlbot_signals_total{{strategy="{s}"}}', "legendFormat": ""}
            ],
            "title": "累计信号",
            "type": "stat",
        }
    )

    # Retrain Status
    panels.append(
        {
            "description": f"{S} 重训触发状态: 0=正常, 1=需要重训",
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "options": {
                                "0": {"text": "✅ OK", "color": "green"},
                                "1": {"text": "⚠️ RETRAIN", "color": "red"},
                            },
                            "type": "value",
                        }
                    ],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 1},
                        ]
                    },
                }
            },
            "gridPos": {"h": 4, "w": 4, "x": 16, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {
                    "expr": f'mlbot_retrain_triggered{{strategy="{s}"}}',
                    "legendFormat": "",
                }
            ],
            "title": "重训状态",
            "type": "stat",
        }
    )

    # Sharpe 30d
    panels.append(
        {
            "description": f"{S} 滚动30天 Sharpe Ratio",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0},
                            {"color": "green", "value": 0.5},
                        ]
                    },
                }
            },
            "gridPos": {"h": 4, "w": 4, "x": 20, "y": 1},
            "id": nid(),
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            },
            "targets": [
                {"expr": f'mlbot_sharpe_live_30d{{strategy="{s}"}}', "legendFormat": ""}
            ],
            "title": "Sharpe 30d",
            "type": "stat",
        }
    )

    # ── Row: Signal Funnel ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5},
            "id": nid(),
            "title": "📡 信号漏斗",
            "type": "row",
        }
    )

    # Funnel chart
    stages = ["direction", "gate", "entry_filter", "evidence", "pcm", "order"]
    panels.append(
        {
            "description": f"{S} 信号漏斗各阶段通过数 (1h 窗口)",
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 70,
                        "stacking": {"mode": "normal"},
                    }
                },
                "overrides": _funnel_overrides(color),
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
            "id": nid(),
            "targets": [
                {
                    "expr": f'increase(mlbot_funnel_total{{stage="{st}",strategy="{s}"}}[1h])',
                    "legendFormat": st,
                }
                for st in stages
            ],
            "title": f"{emoji} {S} 信号漏斗 (1h)",
            "type": "timeseries",
        }
    )

    # Direction distribution (long/short)
    panels.append(
        {
            "description": f"{S} Long/Short 方向分布",
            "fieldConfig": {
                "defaults": {"custom": {"drawStyle": "bars", "fillOpacity": 60}},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Long"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": "green"},
                            }
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Short"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": "red"},
                            }
                        ],
                    },
                ],
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
            "id": nid(),
            "targets": [
                {
                    "expr": f'increase(mlbot_direction_total{{strategy="{s}",side="long"}}[1h])',
                    "legendFormat": "Long",
                },
                {
                    "expr": f'increase(mlbot_direction_total{{strategy="{s}",side="short"}}[1h])',
                    "legendFormat": "Short",
                },
            ],
            "title": f"{emoji} {S} 方向分布 (1h)",
            "type": "timeseries",
        }
    )

    # ── Row: Gate & Filtering ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14},
            "id": nid(),
            "title": "🚧 过滤详情 (Gate / Entry Filter)",
            "type": "row",
        }
    )

    # Gate rejection count
    panels.append(
        {
            "description": f"{S} Gate 阶段拦截次数趋势",
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "fixed", "fixedColor": color},
                    "custom": {"drawStyle": "bars", "fillOpacity": 70},
                }
            },
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 15},
            "id": nid(),
            "targets": [
                {
                    "expr": f'increase(mlbot_gate_rejected_total{{strategy="{s}"}}[1h])',
                    "legendFormat": "Gate 拦截",
                }
            ],
            "title": f"Gate 拦截次数 (1h)",
            "type": "timeseries",
        }
    )

    # Gate rejection reasons
    panels.append(
        {
            "description": f"{S} Gate 拦截原因分布 (如 HARD_ROC_5, VOLATILITY_LOW 等)",
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 80,
                        "stacking": {"mode": "normal"},
                    }
                }
            },
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 15},
            "id": nid(),
            "targets": [
                {
                    "expr": f'increase(mlbot_gate_reject_reasons_total{{strategy="{s}"}}[1h])',
                    "legendFormat": "{{reason}}",
                }
            ],
            "title": "Gate 拦截原因明细 (1h)",
            "type": "timeseries",
        }
    )

    # Orders per hour
    panels.append(
        {
            "description": f"{S} 每小时下单速率",
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "fixed", "fixedColor": color},
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "scheme",
                    },
                }
            },
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 15},
            "id": nid(),
            "targets": [
                {
                    "expr": f'rate(mlbot_orders_total{{strategy="{s}"}}[1h])*3600',
                    "legendFormat": "下单/h",
                }
            ],
            "title": "每小时下单数",
            "type": "timeseries",
        }
    )

    # ── Row: Performance & Retrain Monitor ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 23},
            "id": nid(),
            "title": "📈 表现 & 重训监控",
            "type": "row",
        }
    )

    # Sharpe 30d trend
    panels.append(
        {
            "description": f"{S} 滚动30天 Sharpe Ratio 走势",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "color": {"mode": "fixed", "fixedColor": color},
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "gradientMode": "scheme",
                    },
                }
            },
            "gridPos": {"h": 7, "w": 8, "x": 0, "y": 24},
            "id": nid(),
            "targets": [
                {
                    "expr": f'mlbot_sharpe_live_30d{{strategy="{s}"}}',
                    "legendFormat": "Sharpe 30d",
                }
            ],
            "title": "Sharpe 30d 走势",
            "type": "timeseries",
        }
    )

    # Sharpe Decay & Alpha Decay
    panels.append(
        {
            "description": f"{S} Sharpe 衰减比 (Live/Baseline) & Alpha 衰减",
            "fieldConfig": {
                "defaults": {
                    "decimals": 3,
                    "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 10},
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Sharpe Decay"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": color},
                            }
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Alpha Decay"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": "#F2495C"},
                            }
                        ],
                    },
                ],
            },
            "gridPos": {"h": 7, "w": 8, "x": 8, "y": 24},
            "id": nid(),
            "targets": [
                {
                    "expr": f'mlbot_sharpe_decay_ratio{{strategy="{s}"}}',
                    "legendFormat": "Sharpe Decay",
                },
                {
                    "expr": f'mlbot_alpha_decay_max{{strategy="{s}"}}',
                    "legendFormat": "Alpha Decay",
                },
            ],
            "title": "Sharpe / Alpha 衰减",
            "type": "timeseries",
        }
    )

    # Consecutive Losses & Days Since Train
    panels.append(
        {
            "description": f"{S} 连续亏损次数 & 距上次训练天数",
            "fieldConfig": {
                "defaults": {
                    "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 10}
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "连续亏损"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": "#F2495C"},
                            }
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "训练间隔(天)"},
                        "properties": [
                            {
                                "id": "color",
                                "value": {"mode": "fixed", "fixedColor": "#FADE2A"},
                            }
                        ],
                    },
                ],
            },
            "gridPos": {"h": 7, "w": 8, "x": 16, "y": 24},
            "id": nid(),
            "targets": [
                {
                    "expr": f'mlbot_consecutive_losses{{strategy="{s}"}}',
                    "legendFormat": "连续亏损",
                },
                {
                    "expr": f'mlbot_days_since_last_train{{strategy="{s}"}}',
                    "legendFormat": "训练间隔(天)",
                },
            ],
            "title": "连续亏损 & 训练间隔",
            "type": "timeseries",
        }
    )

    # ── Row: Equity & PnL ──
    panels.append(
        {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 31},
            "id": nid(),
            "title": "💰 资金曲线 & 盈亏",
            "type": "row",
        }
    )

    # Account balance (equity curve)
    panels.append(
        {
            "description": "账户总余额趋势 (资金曲线)",
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "decimals": 2,
                    "color": {"mode": "fixed", "fixedColor": "green"},
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "scheme",
                        "showPoints": "never",
                    },
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32},
            "id": nid(),
            "options": {
                "tooltip": {"mode": "single"},
                "legend": {"displayMode": "hidden"},
            },
            "targets": [
                {
                    "expr": 'mlbot_account_balance{type="total"}',
                    "legendFormat": "总余额",
                }
            ],
            "title": "💰 资金曲线 (账户余额)",
            "type": "timeseries",
        }
    )

    # Cumulative PnL
    panels.append(
        {
            "description": "累计已实现盈亏趋势 (占本金百分比)",
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "color": {"mode": "fixed", "fixedColor": color},
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "scheme",
                        "showPoints": "never",
                    },
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32},
            "id": nid(),
            "options": {
                "tooltip": {"mode": "single"},
                "legend": {"displayMode": "hidden"},
            },
            "targets": [{"expr": "mlbot_pnl_realized_total", "legendFormat": "PnL"}],
            "title": "📈 累计 PnL 曲线",
            "type": "timeseries",
        }
    )

    # ── Row: System context ──
    panels.append(
        {
            "collapsed": True,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 40},
            "id": nid(),
            "title": "🖥️ 系统状态 (展开)",
            "type": "row",
            "panels": [
                {
                    "description": f"{S} 重训触发条件数 (0~5)",
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 2},
                                    {"color": "red", "value": 4},
                                ]
                            },
                            "max": 5,
                            "min": 0,
                        }
                    },
                    "gridPos": {"h": 5, "w": 6, "x": 0, "y": 41},
                    "id": nid(),
                    "options": {
                        "colorMode": "background",
                        "graphMode": "area",
                        "reduceOptions": {"calcs": ["lastNotNull"]},
                    },
                    "targets": [
                        {
                            "expr": f'mlbot_retrain_trigger_count{{strategy="{s}"}}',
                            "legendFormat": "",
                        }
                    ],
                    "title": "重训触发条件数",
                    "type": "stat",
                },
                {
                    "description": "宪法熔断状态",
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [
                                {
                                    "options": {
                                        "0": {"text": "🟢 RUNNING", "color": "green"},
                                        "1": {"text": "🔴 HALTED", "color": "red"},
                                    },
                                    "type": "value",
                                }
                            ],
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 1},
                                ]
                            },
                        }
                    },
                    "gridPos": {"h": 5, "w": 6, "x": 6, "y": 41},
                    "id": nid(),
                    "options": {
                        "colorMode": "background",
                        "graphMode": "none",
                        "reduceOptions": {"calcs": ["lastNotNull"]},
                    },
                    "targets": [
                        {"expr": "mlbot_kill_switch_halted", "legendFormat": ""}
                    ],
                    "title": "宪法状态",
                    "type": "stat",
                },
                {
                    "description": "当前最大回撤",
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percentunit",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 0.12},
                                    {"color": "red", "value": 0.16},
                                ]
                            },
                            "max": 0.2,
                        }
                    },
                    "gridPos": {"h": 5, "w": 6, "x": 12, "y": 41},
                    "id": nid(),
                    "options": {
                        "colorMode": "background",
                        "graphMode": "area",
                        "reduceOptions": {"calcs": ["lastNotNull"]},
                    },
                    "targets": [{"expr": "mlbot_drawdown", "legendFormat": ""}],
                    "title": "Drawdown",
                    "type": "stat",
                },
                {
                    "description": "机器人运行时长",
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "thresholds": {
                                "steps": [{"color": "green", "value": None}]
                            },
                        }
                    },
                    "gridPos": {"h": 5, "w": 6, "x": 18, "y": 41},
                    "id": nid(),
                    "options": {
                        "colorMode": "none",
                        "graphMode": "area",
                        "reduceOptions": {"calcs": ["lastNotNull"]},
                    },
                    "targets": [{"expr": "mlbot_uptime_seconds", "legendFormat": ""}],
                    "title": "Uptime",
                    "type": "stat",
                },
            ],
        }
    )

    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "id": None,
        "links": _nav_links(uid),
        "panels": panels,
        "refresh": "15s",
        "schemaVersion": 39,
        "tags": ["trading", "quant", "strategy", s],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "utc",
        "title": f"{emoji} {S} Strategy",
        "uid": uid,
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
    for strat in STRATEGIES:
        path = os.path.join(out_dir, f"strategy_{strat['name']}.json")
        dashboard = generate_dashboard(strat)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=4, ensure_ascii=False)
        print(f"✅ Generated: {path}")


if __name__ == "__main__":
    main()
