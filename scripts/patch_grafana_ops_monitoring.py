#!/usr/bin/env python3
"""Patch Grafana dashboards for ops monitoring refactor (see plan grafana ops)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "deploy/monitoring/grafana-provisioning/dashboards"

PROM_DS = {"type": "prometheus", "uid": "prometheus-monitoring"}


def _stat_panel(
    pid: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 3,
    h: int = 4,
    legend: str = "",
    max_val: float = 1.0,
) -> dict[str, Any]:
    return {
        "type": "stat",
        "title": title,
        "id": pid,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": PROM_DS,
        "targets": [
            {
                "refId": "A",
                "expr": expr,
                "instant": True,
                "legendFormat": legend or title,
            }
        ],
        "fieldConfig": {
            "defaults": {
                "min": 0,
                "max": max_val,
                "decimals": 0,
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


def _shift_panels_y(
    panels: list[dict], delta: int, skip_ids: set[int] | None = None
) -> None:
    skip = skip_ids or set()
    for p in panels:
        if p.get("id") in skip:
            continue
        gp = p.get("gridPos")
        if gp and "y" in gp:
            gp["y"] = int(gp["y"]) + delta


def patch_quant_system(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data["panels"]
    remove_ids = {306, 405}
    panels = [p for p in panels if p.get("id") not in remove_ids]

    # IDs 1100+ avoid collision with legacy panel ids (101–119 disk section).
    overview_row = {
        "type": "row",
        "title": "一眼总览 · 进程 / 管线 / 对账",
        "id": 1100,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
        "collapsed": False,
    }
    overview_stats = [
        _stat_panel(1101, "FB UP", 'up{job="quant-feature-bus"}', 0, 1, w=2),
        _stat_panel(1102, "Tr UP", 'up{job="quant-trend-fattail"}', 2, 1, w=2),
        _stat_panel(1103, "Hg UP", 'up{job="quant-hedge-multileg"}', 4, 1, w=2),
        _stat_panel(1104, "Sp UP", 'up{job="quant-spot-accum"}', 6, 1, w=2),
        _stat_panel(
            1105, "WS 连", 'min(mlbot_ws_connected{job="quant-feature-bus"})', 8, 1, w=2
        ),
        _stat_panel(
            1106,
            "新鲜",
            'min(mlbot_pipeline_data_fresh{job="quant-feature-bus"})',
            10,
            1,
            w=2,
        ),
        _stat_panel(
            1107,
            "Tr 消费",
            'sum(rate(mlbot_bars_processed_total{job="quant-trend-fattail"}[5m])) > bool 0',
            12,
            1,
            w=2,
        ),
        _stat_panel(
            1108,
            "Hg 消费",
            '(sum(rate(mlbot_multi_leg_bars_processed_total{job="quant-hedge-multileg"}[5m])) + sum(rate(mlbot_multi_leg_daemon_polls_total{job="quant-hedge-multileg"}[5m]))) > bool 0',
            14,
            1,
            w=2,
        ),
        _stat_panel(
            1109,
            "Tr 账",
            'min(mlbot_reconciliation_ok{job="quant-trend-fattail",scope="trend"})',
            16,
            1,
            w=2,
        ),
        _stat_panel(
            1110,
            "Hg 账",
            'min(mlbot_reconciliation_ok{job="quant-hedge-multileg",scope="hedge"})',
            18,
            1,
            w=2,
        ),
        _stat_panel(
            1111,
            "Sp 账",
            'min(mlbot_reconciliation_ok{job="quant-spot-accum",scope="spot"})',
            20,
            1,
            w=2,
        ),
    ]
    skip = {1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111}
    # Drop prior overview row/stats if patch re-run (legacy ids 100–119).
    legacy_overview = set(range(100, 120))
    panels = [
        p
        for p in panels
        if p.get("id") not in skip
        and p.get("id") not in legacy_overview
        and not (p.get("type") == "row" and "一眼总览" in p.get("title", ""))
    ]
    _shift_panels_y(panels, 5, skip_ids=set())
    panels = [overview_row] + overview_stats + panels

    data["panels"] = panels
    data["title"] = "System Health · 运维总览"
    data["version"] = int(data.get("version", 1)) + 1
    links = data.get("links", [])
    extra = [
        {
            "title": "Ops Hub",
            "url": "/d/quant-home",
            "type": "link",
            "targetBlank": False,
        },
        {"title": "Logs", "url": "/d/quant-logs", "type": "link", "targetBlank": False},
        {
            "title": "CMS",
            "url": "http://127.0.0.1:8800",
            "type": "link",
            "targetBlank": True,
        },
    ]
    seen = {ln.get("title") for ln in links}
    for ln in extra:
        if ln["title"] not in seen:
            links.append(ln)
    data["links"] = links
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_quant_home(path: Path) -> None:
    data = {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 0,
        "id": None,
        "links": [
            {
                "title": "System Health",
                "url": "/d/quant-system",
                "type": "link",
                "targetBlank": False,
            },
            {
                "title": "Logs",
                "url": "/d/quant-logs",
                "type": "link",
                "targetBlank": False,
            },
            {
                "title": "Strategy Map",
                "url": "/d/quant-strategy-map",
                "type": "link",
                "targetBlank": False,
            },
            {
                "title": "CMS · 业务台",
                "url": "http://127.0.0.1:8800",
                "type": "link",
                "targetBlank": True,
            },
        ],
        "panels": [
            {
                "type": "text",
                "title": "Ops Hub",
                "id": 1,
                "gridPos": {"h": 14, "w": 24, "x": 0, "y": 0},
                "options": {
                    "mode": "markdown",
                    "content": (
                        "## Ops Hub · 导航\n\n"
                        "| 用途 | 入口 |\n"
                        "|------|------|\n"
                        "| **系统健康**（进程 UP、管线、bus 消费、对账灯） | "
                        "[System Health](/d/quant-system) |\n"
                        "| **集中日志**（journald + 审计 JSONL） | [Logs](/d/quant-logs) |\n"
                        "| **策略管道**（信号 / 拒因 / 对账） | "
                        "[Strategy Map Hub](/d/quant-strategy-map) |\n"
                        "| **账户 / 订单 / K 线 / PnL** | "
                        "[业务 CMS](http://127.0.0.1:8800)（:8800） |\n\n"
                        "Grafana 默认首页已改为 **System Health**；本页仅作快捷导航。\n"
                        "告警：Grafana → Telegram `telegram-quant-ops`（token 见 `monitoring/.env`）。"
                    ),
                },
                "transparent": False,
            }
        ],
        "schemaVersion": 39,
        "style": "dark",
        "tags": ["quant", "ops", "hub"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "utc",
        "title": "Ops Hub · 导航",
        "uid": "quant-home",
        "version": 2,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _remove_panels_by_id(path: Path, remove_ids: set[int]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    before = len(data["panels"])
    data["panels"] = [p for p in data["panels"] if p.get("id") not in remove_ids]
    if len(data["panels"]) < before:
        data["version"] = int(data.get("version", 1)) + 1
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _remove_panels_by_title_substr(path: Path, needles: tuple[str, ...]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    before = len(data["panels"])

    def _keep(panel: dict) -> bool:
        title = panel.get("title", "")
        return not any(n in title for n in needles)

    data["panels"] = [p for p in data["panels"] if _keep(p)]
    if len(data["panels"]) < before:
        data["version"] = int(data.get("version", 1)) + 1
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def patch_strategy_hub(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data.get("panels", []):
        if p.get("type") == "text":
            p["options"]["content"] = (
                "## Strategy Map · 策略管道健康\n\n"
                "业务账户/订单/K 线请用 **业务 CMS**（:8800）。本组只看 **信号、拒因、对账、风控**。\n\n"
                "- **[Trend](/d/quant-strategy-map-trend)** — `quant-trend-fattail` :9190\n"
                "- **[Multi-leg Hedge](/d/quant-strategy-map-hedge)** — `quant-hedge-multileg` :9191\n"
                "- **[Spot Accum](/d/quant-strategy-map-spot)** — `quant-spot-accum` :9193\n\n"
                "[System Health](/d/quant-system) · [Logs](/d/quant-logs) · [Ops Hub](/d/quant-home)"
            )
    data["version"] = int(data.get("version", 1)) + 1
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    patch_quant_system(DASH / "quant_system.json")
    write_quant_home(DASH / "quant_home.json")
    _remove_panels_by_title_substr(
        DASH / "quant_strategy_map_trend.json",
        ("交易所持仓",),
    )
    patch_strategy_hub(DASH / "quant_strategy_map.json")
    # hedge: remove classic funnel panel if id known
    hedge = json.loads(
        (DASH / "quant_strategy_map_hedge.json").read_text(encoding="utf-8")
    )
    hedge["panels"] = [
        p
        for p in hedge["panels"]
        if "经典漏斗" not in p.get("title", "")
        and "交易所持仓" not in p.get("title", "")
    ]
    hedge["version"] = int(hedge.get("version", 1)) + 1
    (DASH / "quant_strategy_map_hedge.json").write_text(
        json.dumps(hedge, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    spot = json.loads(
        (DASH / "quant_strategy_map_spot.json").read_text(encoding="utf-8")
    )
    spot["panels"] = [
        p
        for p in spot["panels"]
        if "account_balance" not in json.dumps(p.get("targets", []))
    ]
    spot["version"] = int(spot.get("version", 1)) + 1
    (DASH / "quant_strategy_map_spot.json").write_text(
        json.dumps(spot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Grafana dashboard patches applied.")


if __name__ == "__main__":
    main()
