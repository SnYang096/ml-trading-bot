#!/usr/bin/env python3
"""Generate Grafana unified alerting rules YAML for quant ops."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy/monitoring/grafana-provisioning/alerting/quant_ops.yaml"

DS = "prometheus-monitoring"


def _rule(
    uid: str,
    title: str,
    expr: str,
    for_dur: str,
    severity: str,
    summary: str,
    description: str,
) -> dict:
    return {
        "uid": uid,
        "title": title,
        "condition": "C",
        "data": [
            {
                "refId": "A",
                "relativeTimeRange": {"from": 300, "to": 0},
                "datasourceUid": DS,
                "model": {
                    "expr": expr,
                    "instant": True,
                    "intervalMs": 1000,
                    "maxDataPoints": 43200,
                    "refId": "A",
                },
            },
            {
                "refId": "B",
                "relativeTimeRange": {"from": 0, "to": 0},
                "datasourceUid": "__expr__",
                "model": {
                    "type": "reduce",
                    "expression": "A",
                    "reducer": "last",
                    "refId": "B",
                },
            },
            {
                "refId": "C",
                "relativeTimeRange": {"from": 0, "to": 0},
                "datasourceUid": "__expr__",
                "model": {
                    "type": "threshold",
                    "expression": "B",
                    "conditions": [{"evaluator": {"type": "gt", "params": [0]}}],
                    "refId": "C",
                },
            },
        ],
        "noDataState": "OK",
        "execErrState": "Alerting",
        "for": for_dur,
        "annotations": {"summary": summary, "description": description},
        "labels": {"severity": severity},
        "isPaused": False,
    }


def main() -> None:
    rules = [
        _rule(
            "quant_target_down",
            "QuantTargetDown",
            'up{job=~"quant-feature-bus|quant-trend-swing|quant-hedge-multileg|quant-spot-accum"} == 0',
            "1m",
            "critical",
            "量化进程 Prometheus target 不可用",
            "检查 systemctl status / journalctl -u quant-* ；Grafana System Health",
        ),
        _rule(
            "quant_trend_reconciliation",
            "QuantTrendReconciliationManualCheck",
            '(min(mlbot_reconciliation_ok{scope="trend"}) == 0) or (sum(mlbot_reconciliation_issue_count{scope="trend"}) > 0)',
            "3m",
            "warning",
            "Trend 订单对账异常",
            "DB vs 币安 open/algo；见 order_management.db / quant-strategy-map-trend",
        ),
        _rule(
            "quant_hedge_reconciliation",
            "QuantHedgeReconciliationManualCheck",
            '(min(mlbot_reconciliation_ok{scope="hedge"}) == 0) or (sum(mlbot_reconciliation_issue_count{scope="hedge"}) > 0)',
            "3m",
            "warning",
            "Multi-leg 对账异常",
            "engine/DB vs 交易所；见 HEDGE_RECONCILIATION_CN.md",
        ),
        _rule(
            "quant_spot_reconciliation",
            "QuantSpotReconciliationManualCheck",
            '(min(mlbot_reconciliation_ok{scope="spot"}) == 0) or (sum(mlbot_reconciliation_issue_count{scope="spot"}) > 0)',
            "3m",
            "warning",
            "Spot 对账异常",
            "pending/ledger vs 交易所",
        ),
        _rule(
            "quant_pipeline_stale",
            "QuantPipelineStale",
            'min(mlbot_pipeline_data_fresh{job="quant-feature-bus"}) == 0',
            "5m",
            "warning",
            "Feature-bus 数据管线不新鲜",
            "ticks/bars/bus 停滞；检查 quant-feature-bus WS 与磁盘写入",
        ),
        _rule(
            "quant_disk_critical",
            "QuantDiskCritical",
            'mlbot_disk_used_percent{job="quant-feature-bus",volume="root"} > 90',
            "5m",
            "critical",
            "根分区磁盘使用率过高",
            "清理旧日志/warmup；见 quant_system Disk 面板",
        ),
    ]
    doc = {
        "apiVersion": 1,
        "groups": [
            {
                "orgId": 1,
                "name": "quant_ops",
                "folder": "Quant Ops",
                "interval": "1m",
                "rules": rules,
            }
        ],
    }
    OUT.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=4096),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
