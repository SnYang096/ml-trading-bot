#!/usr/bin/env python3
"""Phase 5: promote monitor_bundle draft into git-tracked baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitoring.export_monitor_bundle import (
    export_and_promote_direct,
    load_bundle,
    promote_monitor_bundle,
)


def _load_promote_overrides(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "promote_baseline.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def promote_baseline_main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Promote monitor_bundle draft to git monitoring baselines (Phase 5)"
    )
    p.add_argument(
        "--experiment-dir",
        default=None,
        help="Experiment dir with monitor_bundle/bundle.json from Phase 1 draft",
    )
    p.add_argument("--strategy", default=None)
    p.add_argument("--layer", default="regime")
    p.add_argument(
        "--parquet",
        default=None,
        help="Direct path: export+promote without prior draft (one-shot migration)",
    )
    p.add_argument("--enable-drift-ready", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.parquet and args.strategy:
        result = export_and_promote_direct(
            strategy=args.strategy,
            layer=args.layer,
            parquet=Path(args.parquet),
            dry_run=args.dry_run,
            enable_drift_ready=args.enable_drift_ready,
        )
        for line in result.get("actions") or []:
            print(line)
        return 0

    if not args.experiment_dir:
        print(
            "ERROR: pass --experiment-dir or (--strategy + --parquet)",
            file=sys.stderr,
        )
        return 2

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_absolute():
        exp_dir = (PROJECT_ROOT / exp_dir).resolve()
    bundle_path = exp_dir / "monitor_bundle" / "bundle.json"
    if not bundle_path.is_file():
        print(
            f"ERROR: missing {bundle_path}\n"
            "  Run Phase 1 rd_loop with monitor_bundle.mode: draft first.",
            file=sys.stderr,
        )
        return 3

    overrides = _load_promote_overrides(exp_dir)
    enable = args.enable_drift_ready or bool(overrides.get("enable_drift_ready"))
    dry_run = args.dry_run or bool(overrides.get("dry_run"))

    bundle = load_bundle(bundle_path)
    result = promote_monitor_bundle(
        bundle,
        dry_run=dry_run,
        enable_drift_ready=enable,
        bundle_dir=exp_dir / "monitor_bundle",
    )
    for line in result.get("actions") or []:
        print(line)
    if dry_run:
        print("dry-run: no files written")
    else:
        print(f"promoted baseline for {bundle.get('strategy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(promote_baseline_main())
