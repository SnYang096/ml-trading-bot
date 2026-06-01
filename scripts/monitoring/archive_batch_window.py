#!/usr/bin/env python3
"""Build a long-window features parquet via train final --prepare-only (monitor drift)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.event_backtest.market_segment import (
    load_market_segments,
    resolve_segment_run,
)


def _strategy_config_dir(slug: str) -> Path:
    return PROJECT_ROOT / "config" / "strategies" / slug


def archive_batch_window(
    *,
    strategy: str,
    segment: str,
    output: Path,
    market_segment_path: Path,
    prepare_output_root: Optional[Path] = None,
    symbol: str = "BTCUSDT",
    timeframe: str = "120T",
) -> Path:
    segments = load_market_segments(market_segment_path)
    run = resolve_segment_run({"segment": segment}, segments=segments)
    start_date = str(run["start_date"])
    end_date = str(run["end_date"])

    cfg = _strategy_config_dir(strategy)
    if not cfg.is_dir():
        raise FileNotFoundError(f"strategy config not found: {cfg}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = prepare_output_root or (
        PROJECT_ROOT / "results" / "monitoring" / "archive_batch" / strategy / ts
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_strategy_pipeline.py"),
        "--prepare-only",
        "--config",
        str(cfg),
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output-root",
        str(out_root),
        "--train-all",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts'}:{PROJECT_ROOT}",
    }
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"train final --prepare-only failed (exit {proc.returncode}) for {strategy}"
        )

    labeled = out_root / "features_labeled.parquet"
    if not labeled.is_file():
        candidates = sorted(out_root.rglob("features_labeled.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"features_labeled.parquet not found under {out_root}"
            )
        labeled = candidates[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    if labeled.resolve() != output.resolve():
        df = pd.read_parquet(labeled)
        df.to_parquet(output, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "segment": segment,
        "start_date": start_date,
        "end_date": end_date,
        "prepare_output_root": str(out_root),
        "source_labeled": str(labeled),
    }
    import json

    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output


def main() -> int:
    p = argparse.ArgumentParser(description="Archive-batch window for monitor drift")
    p.add_argument(
        "--strategy", default="tpc", help="Strategy slug (config/strategies/<slug>)"
    )
    p.add_argument(
        "--segment",
        default="recent_6m_oos",
        help="market_segment.yaml id (default: recent_6m_oos)",
    )
    p.add_argument(
        "--market-segment",
        default="config/market_segment.yaml",
        help="Path to market_segment.yaml",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output parquet (copy of features_labeled.parquet)",
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="120T")
    p.add_argument(
        "--prepare-output-root",
        default="",
        help="Optional train_final output root (default: results/monitoring/archive_batch/...)",
    )
    args = p.parse_args()

    seg_path = Path(args.market_segment)
    if not seg_path.is_absolute():
        seg_path = (PROJECT_ROOT / seg_path).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    prep_root = (
        Path(args.prepare_output_root)
        if str(args.prepare_output_root).strip()
        else None
    )
    if prep_root and not prep_root.is_absolute():
        prep_root = (PROJECT_ROOT / prep_root).resolve()

    try:
        out_path = archive_batch_window(
            strategy=str(args.strategy),
            segment=str(args.segment),
            output=output,
            market_segment_path=seg_path,
            prepare_output_root=prep_root,
            symbol=str(args.symbol),
            timeframe=str(args.timeframe),
        )
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    rows = len(pd.read_parquet(out_path))
    print(f"saved: {out_path} ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
