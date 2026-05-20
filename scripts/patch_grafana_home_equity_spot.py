#!/usr/bin/env python3
"""Patch quant_home.json: portfolio equity summary, per-account equity, Spot account row."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "deploy/monitoring/grafana-provisioning/dashboards/quant_home.json"

DS = {"type": "prometheus", "uid": "prometheus-monitoring"}
STAT_OPTS = {
    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
    "colorMode": "value",
    "graphMode": "area",
}
USD = {"defaults": {"unit": "currencyUSD"}, "overrides": []}


def stat_panel(
    *,
    pid: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 4,
    h: int = 4,
    unit: str = "currencyUSD",
    color_mode: str = "value",
    graph_mode: str = "area",
    instant: bool = True,
) -> dict:
    fc: dict = {"defaults": {"unit": unit}, "overrides": []}
    if unit == "none" and "min" not in str(fc):
        fc = {"defaults": {}, "overrides": []}
    return {
        "type": "stat",
        "title": title,
        "id": pid,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [
            {"refId": "A", "expr": expr, **({"instant": True} if instant else {})}
        ],
        "fieldConfig": fc,
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
            "colorMode": color_mode,
            "graphMode": graph_mode,
        },
    }


def shift_panels(panels: list[dict], *, min_y: int, delta: int) -> None:
    for p in panels:
        g = p.get("gridPos") or {}
        if g.get("y", 0) >= min_y:
            g["y"] = g["y"] + delta


def main() -> None:
    data = json.loads(HOME.read_text(encoding="utf-8"))
    panels: list[dict] = data["panels"]

    # --- Portfolio summary (insert at top) ---
    shift_panels(panels, min_y=0, delta=5)
    summary = [
        {
            "type": "row",
            "title": "Portfolio Summary",
            "id": 100,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
            "collapsed": False,
        },
        stat_panel(
            pid=101,
            title="Total Equity (all)",
            expr=(
                'sum(mlbot_account_balance{job=~"quant-trend-fattail|quant-hedge-multileg|quant-spot-accum",type="margin"})'
            ),
            x=0,
            y=1,
            w=6,
        ),
        stat_panel(
            pid=102,
            title="Total Unrealized PnL",
            expr='sum(mlbot_unrealized_pnl_total{job=~"quant-trend-fattail|quant-hedge-multileg"})',
            x=6,
            y=1,
            w=6,
        ),
        stat_panel(
            pid=103,
            title="Trend Equity",
            expr='mlbot_account_balance{job="quant-trend-fattail",type="margin"}',
            x=12,
            y=1,
            w=4,
        ),
        stat_panel(
            pid=104,
            title="Hedge Equity",
            expr='mlbot_account_balance{job="quant-hedge-multileg",type="margin"}',
            x=16,
            y=1,
            w=4,
        ),
        stat_panel(
            pid=105,
            title="Spot Equity",
            expr='mlbot_account_balance{job="quant-spot-accum",type="margin"}',
            x=20,
            y=1,
            w=4,
        ),
    ]
    panels[0:0] = summary

    # --- Per-account equity stat + resize trend/hedge stats to w=3 ---
    for p in panels:
        t = p.get("title", "")
        g = p.get("gridPos") or {}
        if p.get("type") != "stat":
            continue
        if t in {
            "Trend Wallet",
            "Trend Available",
            "Trend Margin Ratio",
            "Trend Unrealized PnL",
            "Trend Account Update OK",
            "Trend Account Age (s)",
            "Hedge Wallet",
            "Hedge Available",
            "Hedge Margin Ratio",
            "Hedge Unrealized PnL",
            "Hedge Account Update OK",
            "Hedge Account Age (s)",
        }:
            g["w"] = 3
        if t == "Trend Wallet":
            g["x"] = 0
        elif t == "Trend Available":
            g["x"] = 3
        elif t == "Trend Margin Ratio":
            g["x"] = 9
        elif t == "Trend Unrealized PnL":
            g["x"] = 12
        elif t == "Trend Account Update OK":
            g["x"] = 15
        elif t == "Trend Account Age (s)":
            g["x"] = 18
        elif t == "Hedge Wallet":
            g["x"] = 0
        elif t == "Hedge Available":
            g["x"] = 3
        elif t == "Hedge Margin Ratio":
            g["x"] = 9
        elif t == "Hedge Unrealized PnL":
            g["x"] = 12
        elif t == "Hedge Account Update OK":
            g["x"] = 15
        elif t == "Hedge Account Age (s)":
            g["x"] = 18

    # Insert Trend/Hedge equity panels if missing
    def has_title(title: str) -> bool:
        return any(p.get("title") == title for p in panels)

    if not has_title("Trend Equity"):
        trend_y = next(
            p["gridPos"]["y"] for p in panels if p.get("title") == "Trend Wallet"
        )
        panels.append(
            stat_panel(
                pid=203,
                title="Trend Equity",
                expr='mlbot_account_balance{job="quant-trend-fattail",type="margin"}',
                x=6,
                y=trend_y,
                w=3,
            )
        )
    if not has_title("Hedge Equity"):
        hedge_y = next(
            p["gridPos"]["y"] for p in panels if p.get("title") == "Hedge Wallet"
        )
        panels.append(
            stat_panel(
                pid=204,
                title="Hedge Equity",
                expr='mlbot_account_balance{job="quant-hedge-multileg",type="margin"}',
                x=6,
                y=hedge_y,
                w=3,
            )
        )

    # Add equity to balance timelines
    for p in panels:
        if p.get("title") == "Trend Balance Timeline":
            p.setdefault("targets", []).append(
                {
                    "refId": "D",
                    "expr": 'mlbot_account_balance{job="quant-trend-fattail",type="margin"}',
                    "legendFormat": "equity",
                }
            )
        if p.get("title") == "Hedge Balance Timeline":
            p.setdefault("targets", []).append(
                {
                    "refId": "D",
                    "expr": 'mlbot_account_balance{job="quant-hedge-multileg",type="margin"}',
                    "legendFormat": "equity",
                }
            )

    # --- Spot account section (before reconciliation row) ---
    recon_y = next(p["gridPos"]["y"] for p in panels if p.get("id") == 21)
    shift_panels(panels, min_y=recon_y, delta=11)

    spot_block = [
        {
            "type": "row",
            "title": "Spot Account",
            "id": 30,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": recon_y},
            "collapsed": False,
        },
        stat_panel(
            pid=31,
            title="Spot USDT",
            expr='mlbot_account_balance{job="quant-spot-accum",type="total"}',
            x=0,
            y=recon_y + 1,
            w=3,
        ),
        stat_panel(
            pid=32,
            title="Spot Available",
            expr='mlbot_account_balance{job="quant-spot-accum",type="available"}',
            x=3,
            y=recon_y + 1,
            w=3,
        ),
        stat_panel(
            pid=33,
            title="Spot Equity",
            expr='mlbot_account_balance{job="quant-spot-accum",type="margin"}',
            x=6,
            y=recon_y + 1,
            w=3,
        ),
        stat_panel(
            pid=34,
            title="Spot Holdings (USDT)",
            expr='sum(mlbot_position_notional_usdt{job="quant-spot-accum",scope="spot"})',
            x=9,
            y=recon_y + 1,
            w=3,
        ),
        stat_panel(
            pid=35,
            title="Spot · 对账状态",
            expr='min(mlbot_reconciliation_ok{job="quant-spot-accum",scope="spot"})',
            x=12,
            y=recon_y + 1,
            w=3,
            color_mode="background",
            graph_mode="none",
            unit="none",
        ),
        {
            "type": "stat",
            "title": "Spot Account Update OK",
            "id": 36,
            "gridPos": {"h": 4, "w": 3, "x": 15, "y": recon_y + 1},
            "datasource": DS,
            "targets": [
                {
                    "refId": "A",
                    "expr": 'mlbot_account_update_success{job="quant-spot-accum",scope="spot"}',
                    "instant": True,
                }
            ],
            "fieldConfig": {"defaults": {"min": 0, "max": 1}, "overrides": []},
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                "colorMode": "background",
                "graphMode": "none",
            },
        },
        {
            "type": "stat",
            "title": "Spot Account Age (s)",
            "id": 37,
            "gridPos": {"h": 4, "w": 3, "x": 18, "y": recon_y + 1},
            "datasource": DS,
            "targets": [
                {
                    "refId": "A",
                    "expr": 'mlbot_account_update_age_seconds{job="quant-spot-accum",scope="spot"}',
                    "instant": True,
                }
            ],
            "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                "colorMode": "value",
                "graphMode": "none",
            },
        },
        {
            "type": "timeseries",
            "title": "Spot Balance Timeline",
            "id": 38,
            "gridPos": {"h": 6, "w": 24, "x": 0, "y": recon_y + 5},
            "datasource": DS,
            "targets": [
                {
                    "refId": "A",
                    "expr": 'mlbot_account_balance{job="quant-spot-accum",type="total"}',
                    "legendFormat": "usdt",
                },
                {
                    "refId": "B",
                    "expr": 'sum(mlbot_position_notional_usdt{job="quant-spot-accum",scope="spot"})',
                    "legendFormat": "holdings_usdt",
                },
                {
                    "refId": "C",
                    "expr": 'mlbot_account_balance{job="quant-spot-accum",type="margin"}',
                    "legendFormat": "equity",
                },
            ],
            "fieldConfig": USD,
            "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
        },
    ]
    # fix spot recon stat fieldConfig for 0/1 mapping
    spot_block[5]["fieldConfig"] = {
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
    }

    insert_at = next(i for i, p in enumerate(panels) if p.get("id") == 21)
    for j, panel in enumerate(spot_block):
        panels.insert(insert_at + j, panel)

    # Remove duplicate standalone Spot recon stat in reconciliation row (id 25) — keep table
    panels[:] = [
        p
        for p in panels
        if not (p.get("title") == "Spot · 对账状态" and p.get("id") == 25)
    ]

    HOME.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("quant_home.json patched: equity + spot account")


if __name__ == "__main__":
    main()
