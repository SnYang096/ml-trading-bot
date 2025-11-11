"""Combined daily workflow: forward selection -> gated experts -> monitoring -> cross-sectional portfolio -> merged weights.

Usage:
  PYTHONPATH=src python -m time_series_model.pipeline.workflows.combined_daily \
      --data-dir data/parquet_data \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT \
      --ts-symbol BTCUSDT \
      --ts-timeframes 15T,60T,240T \
      --ts-horizons 2,6,12 \
      --cs-timeframe 15T \
      --cs-horizon 12 \
      --output-dir results/combined_daily
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from time_series_model.pipeline.training import forward_selection as fsel
from time_series_model.pipeline.training import train_regime_gated as gated_train
from time_series_model.pipeline.training import walkforward_gated as gated_walkforward
from time_series_model.pipeline.workflows import gated_to_position as gated_positions
from time_series_model.monitoring import online_monitors as monitors
from time_series_model.monitoring.model_registry import ModelRegistry
from cross_sectional import workflow as cs_workflow


@contextmanager
def _argv_context(args: Sequence[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = [sys.argv[0], *args]
    try:
        yield
    finally:
        sys.argv = original


def run_forward_selection(
    data_dir: str,
    symbol: str,
    timeframes: str,
    out_dir: Path,
    registry: Optional[ModelRegistry],
) -> Path:
    save_dir = out_dir / "forward_selection"
    with _argv_context(
        [
            "--data-dir",
            data_dir,
            "--symbol",
            symbol,
            "--timeframes",
            timeframes,
            "--max-forward",
            "48",
            "--save-dir",
            str(save_dir),
        ]
    ):
        fsel.main()
    if registry:
        registry.log(
            pipeline="forward_selection",
            symbol=symbol,
            artifact_path=str(save_dir),
            metrics={},
            params={"timeframes": timeframes},
        )
    return save_dir


def run_gated_training(
    data_dir: str,
    symbol: str,
    feature_type: str,
    timeframes: str,
    horizons: str,
    save_dir: Path,
    registry: Optional[ModelRegistry],
) -> Tuple[Path, Path]:
    train_args = [
        "--data-dir",
        data_dir,
        "--symbol",
        symbol,
        "--feature-type",
        feature_type,
        "--timeframes",
        timeframes,
    ]
    with _argv_context(train_args):
        gated_train.main()
    positions_args = [
        "--data-dir",
        data_dir,
        "--symbol",
        symbol,
        "--timeframes",
        timeframes,
        "--save-dir",
        str(save_dir),
    ]
    if horizons:
        positions_args.extend(["--multi-horizons", horizons])
    with _argv_context(positions_args):
        gated_positions.main()
    # Walk-forward evaluation
    walkforward_args = [
        "--data-dir",
        data_dir,
        "--symbol",
        symbol,
        "--timeframes",
        timeframes,
        "--start",
        "2024-01",
        "--end",
        "2024-12",
        "--train-months",
        "3",
        "--test-months",
        "1",
        "--save-dir",
        str(save_dir / "walkforward"),
    ]
    if horizons:
        walkforward_args.extend(["--multi-horizons", horizons])
    with _argv_context(walkforward_args):
        gated_walkforward.main()
    positions_path = Path(save_dir) / symbol / "positions.parquet"
    walkforward_summary = Path(save_dir) / "walkforward" / symbol / "summary.json"
    if registry and positions_path.exists():
        registry.log(
            pipeline="ts_gated_positions",
            symbol=symbol,
            artifact_path=str(positions_path),
            metrics={},
            params={"timeframes": timeframes, "horizons": horizons or "6"},
        )
    if registry and walkforward_summary.exists():
        registry.log(
            pipeline="ts_walkforward",
            symbol=symbol,
            artifact_path=str(walkforward_summary),
            metrics={},
            params={"timeframes": timeframes, "horizons": horizons or "6"},
        )
    return positions_path, walkforward_summary


def run_monitoring(
    data_dir: str,
    symbol: str,
    positions_path: Path,
    out_dir: Path,
    registry: Optional[ModelRegistry],
) -> Path:
    monitor_args = [
        "--data-dir",
        data_dir,
        "--symbol",
        symbol,
        "--positions",
        str(positions_path),
        "--price-tf",
        "60T",
        "--forward-bars",
        "6",
        "--save-dir",
        str(out_dir / "monitoring"),
    ]
    with _argv_context(monitor_args):
        monitors.main()
    summary_path = out_dir / "monitoring" / symbol / "online_summary.json"
    if registry and summary_path.exists():
        registry.log(
            pipeline="ts_monitoring",
            symbol=symbol,
            artifact_path=str(summary_path),
            metrics={},
            params={},
        )
    return summary_path


def run_cross_sectional(
    data_dir: str,
    symbols: str,
    timeframe: str,
    horizon: int,
    feature_type: str,
    out_dir: Path,
    registry: Optional[ModelRegistry],
) -> Path:
    cs_args = [
        "--data-dir",
        data_dir,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--horizon",
        str(horizon),
        "--feature-type",
        feature_type,
        "--save-dir",
        str(out_dir / "cross_sectional"),
        "--winsorize-sigma",
        "3.0",
        "--zscore-clip",
        "3.0",
        "--liq-quantile",
        "0.2",
        "--de-corr-threshold",
        "0.9",
        "--regime-overlay",
    ]
    with _argv_context(cs_args):
        cs_workflow.main()
    weights_path = out_dir / "cross_sectional" / "weights.parquet"
    if registry and weights_path.exists():
        registry.log(
            pipeline="cs_weights",
            symbol=symbols,
            artifact_path=str(weights_path),
            metrics={},
            params={"timeframe": timeframe, "horizon": horizon},
        )
    return weights_path


def merge_weights(
    ts_positions: Path,
    cs_weights: Path,
    out_path: Path,
    registry: Optional[ModelRegistry],
    symbol: str,
) -> None:
    if not ts_positions.exists() or not cs_weights.exists():
        raise FileNotFoundError("Missing TS positions or CS weights for merge.")
    ts_df = pd.read_parquet(ts_positions)
    cs_df = pd.read_parquet(cs_weights)
    latest_ts = ts_df.index.max()
    ts_last = ts_df.loc[latest_ts]
    cs_last = cs_df
    merged = pd.DataFrame({
        "ts_position": ts_last["position"] if "position" in ts_last else ts_last,
        "cs_weight": cs_last["weight"],
    })
    merged["ts_abs"] = merged["ts_position"].abs()
    merged["ts_norm"] = merged["ts_abs"] / merged["ts_abs"].sum() if merged["ts_abs"].sum() > 0 else 0.0
    merged["cs_abs"] = merged["cs_weight"].abs()
    merged["cs_norm"] = merged["cs_abs"] / merged["cs_abs"].sum() if merged["cs_abs"].sum() > 0 else 0.0
    merged["merged_weight"] = 0.5 * merged["ts_norm"] + 0.5 * merged["cs_norm"]
    merged.to_parquet(out_path)
    if registry:
        registry.log(
            pipeline="merged_weights",
            symbol=symbol,
            artifact_path=str(out_path),
            metrics={},
            params={},
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined daily workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbols", required=True, help="Comma-separated cross-sectional symbols")
    p.add_argument("--ts-symbol", required=True, help="Single symbol for TS pipeline")
    p.add_argument("--ts-timeframes", default="15T,60T,240T")
    p.add_argument("--ts-horizons", default="2,6,12")
    p.add_argument("--cs-timeframe", default="15T")
    p.add_argument("--cs-horizon", type=int, default=12)
    p.add_argument("--feature-type", default="baseline")
    p.add_argument("--output-dir", default="results/combined_daily")
    p.add_argument("--registry-path", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(args.registry_path) if args.registry_path else None

    # 1. Forward selection
    run_forward_selection(args.data_dir, args.ts_symbol, args.ts_timeframes, out_dir, registry=registry)
    # 2. Gated training + positions
    gated_dir = out_dir / "gated_positions"
    positions_path, walkforward_summary = run_gated_training(
        args.data_dir,
        args.ts_symbol,
        args.feature_type,
        args.ts_timeframes,
        args.ts_horizons,
        gated_dir,
        registry=registry,
    )
    # 3. Online monitoring
    run_monitoring(args.data_dir, args.ts_symbol, positions_path, out_dir, registry=registry)
    # 4. Cross-sectional allocation
    cs_weights = run_cross_sectional(
        args.data_dir,
        args.symbols,
        args.cs_timeframe,
        args.cs_horizon,
        args.feature_type,
        out_dir,
        registry=registry,
    )
    # 5. Merge (equal blend of normalized TS positions & CS weights)
    merged_path = out_dir / "merged_weights.parquet"
    merge_weights(positions_path, cs_weights, merged_path, registry=registry, symbol="combined")
    print(f"✅ Combined daily workflow completed. Merged weights at {merged_path}")


if __name__ == "__main__":
    main()


