import json
from pathlib import Path

import yaml
from click.testing import CliRunner


def _write_minimal_task_spec(tmp_path: Path, feature_plan_ref: str) -> Path:
    p = tmp_path / "task_spec.yaml"
    p.write_text(
        "\n".join(
            [
                "version: 1",
                "task_id: TASK_TEST_COMPARE_FEATURE_SETS",
                "windows:",
                "  oos:",
                "    start: '2025-05-01'",
                "    end: '2025-10-31'",
                f"feature_plan_ref: '{feature_plan_ref}'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def test_compare_feature_sets_generates_reports(tmp_path: Path) -> None:
    # Import CLI entry
    from src.cli.main import cli

    # Use repo's real base config/feature plan (keeps the test small and stable)
    base_config = Path("config/nnmultihead/path_primitives_4h_80h_min")
    feature_plan_ref = "config/nnmultihead/path_primitives_4h_80h_min/feature_plan.yaml"

    task_spec = _write_minimal_task_spec(tmp_path, feature_plan_ref=feature_plan_ref)

    # PoolB YAML as list of feature nodes (intentionally tiny)
    poolb_yaml = tmp_path / "poolb.yaml"
    poolb_yaml.write_text(yaml.safe_dump(["atr_f", "roc_5_f"]), encoding="utf-8")

    out_dir = tmp_path / "out"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "nnmultihead",
            "compare-feature-sets",
            "--no-docker",
            "--task-spec",
            str(task_spec),
            "--base-config",
            str(base_config),
            "--poolb-yaml",
            str(poolb_yaml),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    js = out_dir / "features_compare_summary.json"
    md = out_dir / "features_compare_summary.md"
    assert js.exists()
    assert md.exists()

    obj = json.loads(js.read_text(encoding="utf-8"))
    assert obj["tier_required_n"] >= 1
    assert obj["poolb_n"] == 2
    assert "overlap" in obj
