"""Unit tests for monitor manifest runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.monitoring.run_monitor_manifest import execute_manifest, _load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEKLY_MANIFEST = PROJECT_ROOT / "config" / "monitoring" / "weekly_rule_stack.yaml"


def test_weekly_manifest_loads_and_has_four_steps():
    manifest = _load_manifest(WEEKLY_MANIFEST)
    assert manifest["monitor_id"] == "weekly_rule_stack"
    steps = manifest["steps"]
    assert [next(iter(s)) for s in steps] == [
        "export-window",
        "export-window",
        "watchdog",
        "drift",
    ]
    assert manifest["windows"]["long"]["source"] == "feature_bus_export"
    assert "short" in manifest["windows"]
    assert "long" in manifest["windows"]


def test_execute_manifest_dry_run_substitutes_run_ts(capsys):
    manifest = yaml.safe_load(WEEKLY_MANIFEST.read_text(encoding="utf-8"))
    rc, run_ts, _out = execute_manifest(
        manifest,
        config_path=WEEKLY_MANIFEST,
        run_ts="20260101_1200",
        dry_run=True,
    )
    assert rc == 0
    assert run_ts == "20260101_1200"
    out = capsys.readouterr().out
    assert "20260101_1200" in out
    assert "features_current_7d.parquet" in out
    assert "features_current_long.parquet" in out
    assert out.count("[dry-run] export-window") == 2
    assert "[dry-run] watchdog" in out
    assert "[dry-run] drift" in out


def test_execute_manifest_rejects_unknown_step(tmp_path):
    manifest = {
        "monitor_id": "bad",
        "windows": {"short": {"parquet": "x.parquet"}},
        "steps": [{"noop": {}}],
    }
    with pytest.raises(ValueError, match="unknown manifest step"):
        execute_manifest(
            manifest,
            config_path=tmp_path / "bad.yaml",
            run_ts="20260101_1200",
            dry_run=True,
        )


def test_execute_manifest_writes_heartbeat_on_success(tmp_path, monkeypatch):
    """Watchdog/drift subprocesses are stubbed; heartbeat still written."""
    manifest = {
        "monitor_id": "test_stack",
        "output_dir": str(tmp_path / "out/{run_ts}"),
        "windows": {
            "short": {"parquet": str(tmp_path / "short.parquet")},
            "long": {"parquet": str(tmp_path / "long.parquet")},
        },
        "strategies": ["tpc"],
        "steps": [
            {"watchdog": {"window": "short"}},
            {"drift": {"window": "long"}},
        ],
    }
    (tmp_path / "short.parquet").write_bytes(b"")  # not read when mocked
    (tmp_path / "long.parquet").write_bytes(b"")

    def fake_run(script: str, argv):  # noqa: ANN001
        return 0

    import scripts.monitoring.run_monitor_manifest as mod

    monkeypatch.setattr(mod, "_run_monitor_script", fake_run)

    rc, _, out_dir = execute_manifest(
        manifest,
        config_path=tmp_path / "m.yaml",
        run_ts="20260102_0000",
        dry_run=False,
    )
    assert rc == 0
    assert out_dir == tmp_path / "out" / "20260102_0000"
    hb_path = tmp_path / "out" / "20260102_0000" / "heartbeat.json"
    assert hb_path.is_file()
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    assert hb["status"] == "OK"
    assert hb["task"] == "test_stack"
