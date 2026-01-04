import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_small_panel_csv(path: Path) -> None:
    ts = [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T04:00:00Z")]
    syms = ["A", "B", "C", "D"]

    rows = []
    # Make factor predictive of future_return_1
    for t in ts:
        for i, s in enumerate(syms):
            rows.append(
                {
                    "timestamp": t.isoformat(),
                    "symbol": s,
                    "factor": float(i),
                    "future_return_1": float(i) * 0.01,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _env_with_src_pythonpath() -> dict:
    env = os.environ.copy()
    proj = Path(__file__).resolve().parents[2]
    src = proj / "src"
    # Match `mlbot` behavior: include project root + src/
    env["PYTHONPATH"] = f"{proj}:{src}:{env.get('PYTHONPATH','')}".rstrip(":")
    return env


def test_factor_eval_script_runs(tmp_path: Path):
    panel_csv = tmp_path / "panel.csv"
    _write_small_panel_csv(panel_csv)

    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "src/cross_sectional/scripts/factor_eval.py",
        "--input",
        str(panel_csv),
        "--factors",
        "factor",
        "--target",
        "future_return_1",
        "--min-assets",
        "4",
        "--quantiles",
        "2",
        "--fee-bps",
        "0",
        "--output-dir",
        str(outdir),
    ]
    subprocess.check_call(cmd, env=_env_with_src_pythonpath())

    assert (outdir / "summary.csv").exists()
    summary = pd.read_csv(outdir / "summary.csv")
    assert "factor" in summary.columns or "factor" in summary.to_string()


def test_pipeline_runs_on_parquet_source(tmp_path: Path):
    panel_csv = tmp_path / "panel.csv"
    _write_small_panel_csv(panel_csv)

    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"output_root: {tmp_path.as_posix()}/pipeline_out",
                "panel:",
                "  source: parquet",
                f"  path: {panel_csv.as_posix()}",
                "factor_eval:",
                "  factors: factor",
                "  target: future_return_1",
                "  min_assets: 4",
                "  quantiles: 2",
                "  fee_bps: 0.0",
                "select:",
                "  enabled: true",
                "  min_assets: 4",
                "  per_category_top: 1",
                "  global_top: 1",
            ]
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "src/cross_sectional/scripts/pipeline.py",
        "--config",
        str(cfg),
    ]
    subprocess.check_call(cmd, env=_env_with_src_pythonpath())

    out_root = tmp_path / "pipeline_out"
    assert (out_root / "factor_eval" / "summary.csv").exists()
    assert (out_root / "pipeline_manifest.json").exists()
    # selection should exist (because source=parquet)
    assert (out_root / "selected_factors.txt").exists()
