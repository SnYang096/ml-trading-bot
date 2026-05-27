#!/usr/bin/env python3
"""R&D loop driver: mlbot research scan → variant_grid → decision doc skeleton.

Reads a hypothesis YAML (see ``config/experiments/rd_loop_example.yaml``) and runs
steps in order, persisting progress to ``<output_dir>/rd_loop_state.json`` so a
failed run can resume with ``--resume``.

Does not modify live yaml.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_hypothesis(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("hypothesis yaml must be a mapping")
    return data


def _state_path(output_dir: Path) -> Path:
    return output_dir / "rd_loop_state.json"


def _load_state(output_dir: Path) -> Dict[str, Any]:
    p = _state_path(output_dir)
    if not p.exists():
        return {"completed_steps": [], "steps": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(output_dir: Path, state: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _state_path(output_dir).write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_cmd(cmd: List[str], *, cwd: Path) -> int:
    print("\n>>>", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _mlbot_cmd() -> List[str]:
    exe = shutil.which("mlbot")
    if exe:
        return [exe]
    return [sys.executable, "-m", "cli.main"]


def _append_common_scan_args(
    cmd: List[str], scan: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    cmd += ["--features-parquet", str(scan["features_parquet"])]
    strategy = scan.get("strategy") or cfg.get("strategy")
    if strategy:
        cmd += ["--strategy", str(strategy)]
    if scan.get("layer"):
        cmd += ["--layer", str(scan["layer"])]
    if scan.get("label"):
        cmd += ["--label", str(scan["label"])]
    for f in scan.get("filter") or []:
        cmd += ["--filter", str(f)]
    if scan.get("subset"):
        cmd += ["--subset", str(scan["subset"])]
    if scan.get("calendar_window"):
        cmd += ["--calendar-window", str(scan["calendar_window"])]


def _build_research_scan_cmd(
    scan: Dict[str, Any], output_dir: Path, cfg: Dict[str, Any]
) -> List[str]:
    mode = str(scan.get("mode", "condition-set"))
    out_rel = scan.get("out") or f"quick_scan/scan_{mode.replace('-', '_')}.md"
    out = Path(out_rel)
    if not out.is_absolute():
        out = (output_dir / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bucket_by = scan.get("bucket_by")
    if bucket_by:
        cmd = _mlbot_cmd() + [
            "research",
            "segment",
            "--bucket-by",
            str(bucket_by),
            "--mode",
            mode if mode in ("condition-set", "feature-plateau") else "condition-set",
            "--output",
            str(out),
        ]
        _append_common_scan_args(cmd, scan, cfg)
        if mode == "feature-plateau":
            cmd += [
                "--feature",
                str(scan["feature"]),
                "--operator",
                str(scan.get("operator", "<=")),
                "--grid",
                str(scan["grid"]),
            ]
        else:
            for c in scan.get("condition") or []:
                cmd += ["--condition", str(c)]
        return cmd

    if mode == "ic-decay":
        cmd = _mlbot_cmd() + ["research", "ic", "--output", str(out)]
        _append_common_scan_args(cmd, scan, cfg)
        cmd += [
            "--features",
            str(scan["features"]),
            "--horizons",
            str(scan.get("horizons", "1,3,5,10,20")),
        ]
        if scan.get("target"):
            cmd += ["--target", str(scan["target"])]
        if scan.get("baseline_json"):
            cmd += ["--baseline-json", str(scan["baseline_json"])]
        return cmd

    if mode == "snotio-plateau":
        snotio_mode = str(scan.get("snotio_mode", "proxy"))
        cmd = _mlbot_cmd() + [
            "research",
            "plateau",
            "--kpi",
            "snotio",
            "--snotio-mode",
            snotio_mode,
            "--output",
            str(out.with_suffix(".json") if out.suffix == ".md" else out),
        ]
        _append_common_scan_args(cmd, scan, cfg)
        cmd += [
            "--feature",
            str(scan["feature"]),
            "--operator",
            str(scan.get("operator", "<=")),
            "--grid",
            str(scan["grid"]),
        ]
        if scan.get("r_col"):
            cmd += ["--r-col", str(scan["r_col"])]
        return cmd

    verb = "research"
    cmd = _mlbot_cmd() + [verb, "scan", mode, "--output", str(out)]
    _append_common_scan_args(cmd, scan, cfg)

    if mode == "feature-plateau":
        cmd += [
            "--feature",
            str(scan["feature"]),
            "--operator",
            str(scan.get("operator", "<=")),
            "--grid",
            str(scan["grid"]),
        ]
    elif mode == "condition-set":
        for c in scan.get("condition") or []:
            cmd += ["--condition", str(c)]
    elif mode == "pair-scan":
        cmd += ["--pair-a", str(scan["pair_a"]), "--pair-b", str(scan["pair_b"])]
    return cmd


def _step_research_scan(cfg: Dict[str, Any], output_dir: Path) -> int:
    scans = cfg.get("quick_layer_scans") or cfg.get("research_scans") or []
    if not isinstance(scans, list):
        raise ValueError("quick_layer_scans / research_scans must be a list")
    worst = 0
    for i, scan in enumerate(scans):
        if not isinstance(scan, dict):
            continue
        cmd = _build_research_scan_cmd(scan, output_dir, cfg)
        rc = _run_cmd(cmd, cwd=PROJECT_ROOT)
        worst = max(worst, rc)
    return worst


def _step_variant_grid(cfg: Dict[str, Any], output_dir: Path) -> int:
    grid = cfg.get("variant_grid")
    if not grid:
        return 0
    grid_path = Path(str(grid))
    if not grid_path.is_absolute():
        grid_path = (PROJECT_ROOT / grid_path).resolve()
    cmd = [
        sys.executable,
        "-m",
        "scripts.event_backtest",
        "--variant-grid",
        str(grid_path),
    ]
    extra = cfg.get("variant_grid_extra_argv") or []
    if isinstance(extra, list):
        cmd += [str(x) for x in extra]
    return _run_cmd(cmd, cwd=PROJECT_ROOT)


def _step_decision_doc(cfg: Dict[str, Any], output_dir: Path) -> int:
    dec = cfg.get("decision_doc") or {}
    if not dec:
        return 0
    index = dec.get("experiment_index")
    if not index:
        strategy = str(cfg.get("strategy") or "tpc")
        index = (
            PROJECT_ROOT
            / "results"
            / strategy
            / "experiments"
            / "EXPERIMENT_INDEX.json"
        )
    index_path = Path(str(index))
    if not index_path.is_absolute():
        index_path = (PROJECT_ROOT / index_path).resolve()
    topic = str(dec.get("topic") or cfg.get("topic") or "rd_loop")
    out = dec.get("out")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "_new_decision_doc.py"),
        "--experiment-index",
        str(index_path),
        "--topic",
        topic,
    ]
    if dec.get("topic_template"):
        cmd += ["--topic-template", str(dec["topic_template"])]
    if out:
        out_path = Path(str(out))
        if not out_path.is_absolute():
            out_path = (output_dir / out_path).resolve()
        cmd += ["--out", str(out_path)]
    return _run_cmd(cmd, cwd=PROJECT_ROOT)


def run_loop(
    hypothesis_path: Path,
    *,
    output_dir: Optional[Path] = None,
    resume: bool = False,
) -> int:
    cfg = _load_hypothesis(hypothesis_path)
    topic = str(cfg.get("topic") or hypothesis_path.stem)
    out = output_dir or Path(str(cfg.get("output_dir") or f"results/rd_loop/{topic}"))
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    state = _load_state(out) if resume else {"completed_steps": [], "steps": {}}
    completed = set(state.get("completed_steps") or [])

    steps = [
        ("research_scan", lambda: _step_research_scan(cfg, out)),
        ("variant_grid", lambda: _step_variant_grid(cfg, out)),
        ("decision_doc", lambda: _step_decision_doc(cfg, out)),
    ]

    worst = 0
    for name, fn in steps:
        if name in completed:
            print(f"skip (already done): {name}")
            continue
        print(f"\n=== rd_loop step: {name} ===")
        rc = fn()
        state["steps"][name] = {
            "exit_code": rc,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if rc == 0:
            completed.add(name)
            state["completed_steps"] = sorted(completed)
        _save_state(out, state)
        worst = max(worst, rc)
        if rc != 0:
            print(f"rd_loop stopped at step {name} (exit {rc})", file=sys.stderr)
            return rc
    print(f"\nrd_loop complete → {out}")
    return worst


def main() -> int:
    p = argparse.ArgumentParser(
        description="R&D loop driver (mlbot research scan → grid → decision doc)"
    )
    p.add_argument(
        "--hypothesis-yaml",
        required=True,
        help="Hypothesis config (quick_layer_scans / research_scans + variant_grid).",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps already marked complete in rd_loop_state.json.",
    )
    args = p.parse_args()

    hyp = Path(args.hypothesis_yaml)
    if not hyp.is_absolute():
        hyp = (PROJECT_ROOT / hyp).resolve()
    if not hyp.exists():
        print(f"ERROR: not found: {hyp}", file=sys.stderr)
        return 3
    out = Path(args.output_dir) if args.output_dir else None
    if out and not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    return run_loop(hyp, output_dir=out, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
