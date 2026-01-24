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
    ap.add_argument(
        "--gated-logs",
        default=None,
        help="gated logs file (from apply-tree-gate, optional)",
    )
    ap.add_argument("--out-dir", required=True, help="output base dir")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--strategy-name", default="pipeline")
    return ap.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    preds_df, mode_df, logs_df = load_pipeline_inputs(
        Path(args.preds),
        Path(args.mode) if args.mode else None,
        Path(args.logs) if args.logs else None,
    )

    # Load gated logs if provided
    gated_df = None
    if args.gated_logs:
        gated_path = Path(args.gated_logs)
        if gated_path.exists():
            if gated_path.suffix.lower() == ".parquet":
                gated_df = pd.read_parquet(gated_path)
            else:
                gated_df = pd.read_csv(gated_path)
            print(f"✅ Loaded gated logs: {len(gated_df)} rows")
        else:
            print(f"⚠️ Warning: Gated logs file not found: {gated_path}")

    records = build_stage_logs_from_pipeline(
        preds_df=preds_df,
        mode_df=mode_df,
        logs_df=logs_df,
        gated_df=gated_df,
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
