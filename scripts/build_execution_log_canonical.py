#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.time_series_model.diagnostics.execution_log_canonical import (
    load_pipeline_inputs,
    build_canonical_from_pipeline,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build canonical execution log from pipeline outputs."
    )
    ap.add_argument("--preds", required=True, help="preds file/dir")
    ap.add_argument("--mode", default=None, help="mode_3action file/dir (optional)")
    ap.add_argument("--logs", default=None, help="logs_3action file/dir (optional)")
    ap.add_argument("--out", required=True, help="output jsonl file")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeframe", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    preds_df, mode_df, logs_df = load_pipeline_inputs(
        Path(args.preds),
        Path(args.mode) if args.mode else None,
        Path(args.logs) if args.logs else None,
    )
    records = build_canonical_from_pipeline(
        preds_df=preds_df,
        mode_df=mode_df,
        logs_df=logs_df,
        run_id=args.run_id,
        timeframe=args.timeframe,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
