"""Regime ops snapshot for /regime CMS page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlbot_console.services.regime_ops import (
    _config_reference_at,
    _parse_drift_document,
    _summarize_drift,
    fetch_regime_ops_snapshot,
    regime_drift_meta,
)


def test_parse_drift_document_report_key():
    doc = {
        "generated_at": "2026-05-20T12:00:00",
        "report": [
            {"strategy": "tpc", "any_alert": False, "items": []},
            {
                "strategy": "chop_grid",
                "any_alert": True,
                "items": [{"feature": "x", "status": "DRIFT"}],
            },
        ],
    }
    by = _parse_drift_document(doc)
    assert "tpc" in by
    assert by["chop_grid"]["any_alert"] is True


def test_summarize_drift_no_report():
    st, detail = _summarize_drift(None)
    assert st == "—"
    assert "regime_drift_monitor" in detail


def test_summarize_drift_no_plateaus():
    st, detail = _summarize_drift({"strategy": "tpc", "any_alert": False, "items": []})
    assert st == "未监测"


def test_config_reference_at_from_eval_run_id():
    assert _config_reference_at(
        {"last_multileg_evaluation": {"run_id": "20260510_232349"}}
    ).startswith("2026-05-10 23:23:49")


def test_config_reference_at_prefers_last_calibration_timestamp():
    assert (
        _config_reference_at(
            {
                "last_calibration": {"timestamp": "2026-05-20T08:00:00+00:00"},
                "last_multileg_evaluation": {"run_id": "20260510_232349"},
            }
        )
        == "2026-05-20T08:00:00+00:00"
    )


def test_summarize_drift_drift_alert():
    st, _ = _summarize_drift(
        {
            "strategy": "bpc",
            "any_alert": True,
            "items": [{"feature": "tpc_semantic_chop", "status": "DRIFT"}],
        }
    )
    assert st == "漂移"


def test_fetch_regime_ops_includes_live_strategies(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "strategies"
    for slug, layer_block in (
        ("tpc", "regime.yaml"),
        ("spot_accum_simple", "prefilter.yaml"),
        ("chop_grid", "prefilter.yaml"),
    ):
        arch = root / slug / "archetypes"
        arch.mkdir(parents=True)
        if slug == "tpc":
            (arch / "regime.yaml").write_text(
                "allowed_sides: [long, short]\nrules: []\n", encoding="utf-8"
            )
        elif slug == "spot_accum_simple":
            (arch / "prefilter.yaml").write_text(
                "rules:\n  - feature: weekly_ema_200_position\n    operator: <\n    value: 0\n",
                encoding="utf-8",
            )
        else:
            (arch / "prefilter.yaml").write_text(
                "regime:\n  entry_chop_min: 0.5\nrules:\n  - feature: box_pos_60\n"
                "last_multileg_evaluation:\n  run_id: '20260510_232349'\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "mlbot_console.services.regime_ops.get_live_console_strategies",
        lambda: [
            {"id": "tpc", "account_layer": "trend", "title": "TPC"},
            {"id": "spot_accum_simple", "account_layer": "spot", "title": "Spot"},
            {"id": "chop_grid", "account_layer": "multi_leg", "title": "Chop"},
        ],
    )

    rows = fetch_regime_ops_snapshot(root, project_root=tmp_path)
    ids = [r["strategy"] for r in rows]
    assert ids == ["tpc", "spot_accum_simple", "chop_grid"]
    spot = next(r for r in rows if r["strategy"] == "spot_accum_simple")
    assert spot["account_layer"] == "spot"
    assert "prefilter" in spot["regime_source"]
    chop = next(r for r in rows if r["strategy"] == "chop_grid")
    assert chop["account_layer"] == "multi_leg"
    assert "regime" in chop["regime_source"]
    assert chop["config_reference_at"].startswith("2026-05-10 23:23:49")


def test_regime_ops_reads_drift_report_json(tmp_path: Path) -> None:
    drift_dir = tmp_path / "results" / "regime_drift_monitor" / "20260520_120000"
    drift_dir.mkdir(parents=True)
    (drift_dir / "drift_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-20T12:00:00",
                "report": [{"strategy": "tpc", "any_alert": False, "items": []}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "strategies"
    arch = root / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "regime.yaml").write_text("rules: []\n", encoding="utf-8")

    rows = fetch_regime_ops_snapshot(root, project_root=tmp_path, strategies=["tpc"])
    assert rows[0]["drift_status"] == "未监测"
    assert rows[0]["drift_checked_at"] == "2026-05-20T12:00:00"
    meta = regime_drift_meta(tmp_path)
    assert meta["drift_report_path"] is not None
    assert "2026-05-20" in (meta["drift_generated_at"] or "")
