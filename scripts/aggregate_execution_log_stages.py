#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate split-stage execution logs into canonical JSONL."
    )
    ap.add_argument("--stage-dir", required=True, help="base dir with stage subdirs")
    ap.add_argument("--out", required=True, help="output canonical jsonl")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = aggregate_stage_logs(Path(args.stage_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
