"""FastAPI integration tests for the new /api/monitoring endpoints (M2)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlbot_console.config import ConsoleSettings


@pytest.fixture
def monitoring_client(
    console_settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Client with a controlled repo_root containing fake monitoring data."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Minimal schedules.yaml so dashboard can load it
    sched_dir = repo_root / "config" / "monitoring"
    sched_dir.mkdir(parents=True)
    (sched_dir / "schedules.yaml").write_text(
        """
staleness_hours:
  weekly: 192
schedules:
  weekly:
    manifest: config/monitoring/weekly_rule_stack.yaml
""",
        encoding="utf-8",
    )

    # Fake index.json
    mon_dir = repo_root / "results" / "monitoring"
    mon_dir.mkdir(parents=True)
    index = {
        "updated_at": "2026-06-01T12:00:00+00:00",
        "cadences": {
            "weekly": {
                "cadence": "weekly",
                "run_ts": "20260601_0800",
                "status": "ALERT",
                "exit_code": 1,
                "watchdog_any_alert": True,
                "drift_any_alert": False,
                "manifest": "config/monitoring/weekly_rule_stack.yaml",
            },
            "daily": {
                "cadence": "daily",
                "run_ts": "20260520_0600",
                "status": "OK",
                "exit_code": 0,
            },
        },
    }
    (mon_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    # Create a real sqlite with the expected schema + data
    db_path = repo_root / "results" / "rd_registry.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS monitor_event (
            id TEXT PRIMARY KEY,
            cadence TEXT,
            source TEXT,
            strategy TEXT,
            status TEXT,
            detail_json TEXT,
            report_path TEXT,
            run_ts TEXT,
            output_dir TEXT,
            ts TEXT
        );
        """
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO monitor_event
        (id, cadence, source, strategy, status, detail_json, report_path, run_ts, output_dir, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "weekly:20260601_0800:watchdog:tpc",
                "weekly",
                "watchdog",
                "tpc",
                "ALERT",
                json.dumps({"alerts": ["PSI_DRIFT: ema_1200_position"]}),
                str(mon_dir / "weekly" / "report.json"),
                "20260601_0800",
                str(mon_dir / "weekly_rule_stack" / "20260601_0800"),
                "2026-06-01T08:05:00+00:00",
            ),
            (
                "weekly:20260601_0800:drift:bpc",
                "weekly",
                "drift",
                "bpc",
                "OK",
                "{}",
                str(mon_dir / "weekly" / "drift_report.json"),
                "20260601_0800",
                str(mon_dir / "weekly_rule_stack" / "20260601_0800"),
                "2026-06-01T08:10:00+00:00",
            ),
        ],
    )
    conn.commit()
    conn.close()

    # Patch console settings to point at our controlled repo
    patched_settings = replace(console_settings, repo_root=repo_root)
    for mod in (
        "mlbot_console.config",
        "mlbot_console.routers.monitoring",
    ):
        monkeypatch.setattr(f"{mod}.SETTINGS", patched_settings)

    from mlbot_console.main import app

    return TestClient(app), patched_settings


def test_monitoring_index_endpoint(monitoring_client):
    client, _ = monitoring_client
    r = client.get("/api/monitoring/index")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "weekly" in data["data"]["cadences"]
    assert data["data"]["cadences"]["weekly"]["status"] == "ALERT"


def test_monitoring_dashboard_cards_and_summary(monitoring_client):
    client, _ = monitoring_client
    r = client.get("/api/monitoring/dashboard")
    assert r.status_code == 200
    data = r.json()["data"]
    assert "cards" in data
    assert "summary" in data
    assert data["summary"]["any_alert"] is True
    # We expect at least the weekly card to be present
    cadences = [c["cadence"] for c in data["cards"]]
    assert "weekly" in cadences


def test_monitoring_events_endpoint(monitoring_client):
    client, _ = monitoring_client
    r = client.get("/api/monitoring/events", params={"cadence": "weekly", "limit": 10})
    assert r.status_code == 200
    events = r.json()["data"]
    assert len(events) >= 1
    assert any(e["strategy"] == "tpc" and e["status"] == "ALERT" for e in events)


def test_monitoring_overview_includes_dashboard(monitoring_client):
    client, _ = monitoring_client
    r = client.get("/api/monitoring/overview")
    assert r.status_code == 200
    payload = r.json()["data"]
    assert "dashboard" in payload
    assert "index" in payload
    assert payload["dashboard"]["summary"]["n_cards"] >= 1
