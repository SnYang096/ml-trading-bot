"""Smoke test: rd_loop step dispatch (mocked subprocess)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.rd_loop import run_loop


def test_rd_loop_runs_three_steps(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "test_loop",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "condition-set",
                        "features_parquet": "dummy.parquet",
                        "condition": ["H: x>0"],
                    }
                ],
                "variant_grid": "config/experiments/tpc_variant_grid_smoke.yaml",
                "decision_doc": {
                    "topic": "test_loop",
                    "topic_template": "default",
                    "experiment_index": "idx.json",
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert len(calls) == 3
    assert "quick_layer_scan.py" in " ".join(calls[0])
    assert "variant-grid" in " ".join(calls[1])
    assert "_new_decision_doc.py" in " ".join(calls[2])
    state = (tmp_path / "out" / "rd_loop_state.json").read_text(encoding="utf-8")
    assert "quick_layer_scan" in state
    assert "decision_doc" in state


def test_rd_loop_resume_skips_completed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "rd_loop_state.json").write_text(
        '{"completed_steps": ["quick_layer_scan", "variant_grid"], "steps": {}}',
        encoding="utf-8",
    )
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "resume_test",
                "output_dir": str(out),
                "decision_doc": {"topic": "resume_test"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        rc = run_loop(hyp, output_dir=out, resume=True)

    assert rc == 0
    assert len(calls) == 1
    assert "_new_decision_doc.py" in " ".join(calls[0])
