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
    joined0 = " ".join(calls[0])
    assert "research" in joined0 and "scan" in joined0 and "condition-set" in joined0
    assert "variant-grid" in " ".join(calls[1])
    assert "_new_decision_doc.py" in " ".join(calls[2])
    state = (tmp_path / "out" / "rd_loop_state.json").read_text(encoding="utf-8")
    assert "research_scan" in state
    assert "decision_doc" in state


def test_rd_loop_resume_skips_completed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "rd_loop_state.json").write_text(
        '{"completed_steps": ["research_scan", "variant_grid"], "steps": {}}',
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


def test_rd_loop_pair_scan_cmd(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "pair",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "pair-scan",
                        "features_parquet": "dummy.parquet",
                        "pair_a": "a:<=:0,1",
                        "pair_b": "b:>=:0,1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        run_loop(hyp, output_dir=tmp_path / "out")

    joined = " ".join(calls[0])
    assert "pair-scan" in joined
    assert "--pair-a" in joined


def test_rd_loop_snotio_plateau_passes_subject(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "snotio",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "snotio-plateau",
                        "features_parquet": "dummy.parquet",
                        "feature": "pulse_z",
                        "grid": "-1,0,1",
                        "subject": "feature:pulse_z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        run_loop(hyp, output_dir=tmp_path / "out")

    joined = " ".join(calls[0])
    assert "plateau" in joined
    assert "--subject" in joined
    assert "feature:pulse_z" in joined


def test_rd_loop_entry_plateau_runs_batch(tmp_path: Path) -> None:
    import pandas as pd

    pq = tmp_path / "logs.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(pq)
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "entry_plateau",
                "strategy": "srb",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "entry-plateau",
                        "features_parquet": str(pq),
                        "snotio_mode": "entry_rr",
                        "steps": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict] = []

    def fake_batch(*args, **kwargs):  # noqa: ANN002, ANN003
        batch_calls.append({"args": args, "kwargs": kwargs})
        return {"summary_path": str(tmp_path / "out" / "summary.json")}

    with patch(
        "scripts.research.entry_plateau_scan.run_entry_plateau_batch",
        side_effect=fake_batch,
    ):
        rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert len(batch_calls) == 1
    assert batch_calls[0]["kwargs"].get("snotio_mode") == "entry_rr"
