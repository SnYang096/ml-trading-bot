"""Unit tests for CMS dashboard aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.dashboard import (
    build_monitoring_dashboard,
    strategy_alerts_by_cadence,
)
from src.monitoring.store import init_registry_db, upsert_monitor_events_from_run


def test_build_monitoring_dashboard_cards(tmp_path):
    sched = tmp_path / "config/monitoring/schedules.yaml"
    sched.parent.mkdir(parents=True)
    sched.write_text(
        """
staleness_hours:
  weekly: 10000
schedules:
  weekly:
    manifest: w.yaml
""",
        encoding="utf-8",
    )
    idx_dir = tmp_path / "results/monitoring"
    idx_dir.mkdir(parents=True)
    (idx_dir / "index.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-01T00:00:00+00:00",
                "cadences": {
                    "weekly": {
                        "run_ts": "20260601_1200",
                        "status": "OK",
                        "exit_code": 0,
                        "watchdog_any_alert": False,
                        "drift_any_alert": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    db = init_registry_db(tmp_path / "registry.sqlite")
    dash = build_monitoring_dashboard(tmp_path, db, schedules_path=sched)
    assert dash["summary"]["n_cards"] == 1
    assert dash["cards"][0]["display_status"] == "OK"


def test_strategy_alerts_filters_by_latest_run_ts():
    cards = [{"cadence": "weekly", "run_ts": "20260102_0000"}]
    events = [
        {
            "cadence": "weekly",
            "run_ts": "20260102_0000",
            "source": "drift",
            "strategy": "tpc",
            "status": "ALERT",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260101_0000",
            "source": "drift",
            "strategy": "bpc",
            "status": "ALERT",
        },
    ]
    out = strategy_alerts_by_cadence(events, cards)
    assert len(out["weekly"]) == 1
    assert out["weekly"][0]["strategy"] == "tpc"


def test_load_monitoring_index_from_store(tmp_path):
    from src.monitoring.store import load_monitoring_index

    p = tmp_path / "results/monitoring"
    p.mkdir(parents=True)
    (p / "index.json").write_text('{"cadences":{}}', encoding="utf-8")
    assert load_monitoring_index(tmp_path)["cadences"] == {}
