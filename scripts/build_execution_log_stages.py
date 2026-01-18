#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.time_series_model.diagnostics.execution_log import ExecutionStageLogWriter
from src.time_series_model.diagnostics.execution_log_canonical import (
    load_pipeline_inputs,
    build_stage_logs_from_pipeline,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build split-stage execution logs from pipeline outputs."
    )
    ap.add_argument("--preds", required=True, help="preds file/dir")
    ap.add_argument("--mode", default=None, help="mode_3action file/dir (optional)")
    ap.add_argument("--logs", default=None, help="logs_3action file/dir (optional)")
    ap.add_argument("--out-dir", required=True, help="output base dir")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--strategy-name", default="pipeline")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    preds_df, mode_df, logs_df = load_pipeline_inputs(
        Path(args.preds),
        Path(args.mode) if args.mode else None,
        Path(args.logs) if args.logs else None,
    )
    records = build_stage_logs_from_pipeline(
        preds_df=preds_df,
        mode_df=mode_df,
        logs_df=logs_df,
        run_id=args.run_id,
        timeframe=args.timeframe,
        strategy_name=args.strategy_name,
    )
    writers = {}
    for rec in records:
        stage = rec.get("stage")
        if not stage:
            continue
        writer = writers.get(stage)
        if writer is None:
            writer = ExecutionStageLogWriter(
                base_dir=Path(args.out_dir), stage=str(stage)
            )
            writers[stage] = writer
        writer.write(rec, decision_ts_ns=int(rec.get("decision_ts_ns", 0)))


if __name__ == "__main__":
    main()
