#!/usr/bin/env python3
"""Sweep regime.box_prefilter thresholds (live-aligned block_stable_box)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chop_grid_backtest import (  # noqa: E402
    ChopGridEngine,
    collect_chop_grid_trades_for_symbol,
)
from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    build_features,
    merge_chop_grid_yaml,
    recompute_box_prefilter_column,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.pipeline.multileg_portfolio_metrics import (  # noqa: E402
    portfolio_metrics_from_trades,
)
from scripts.sweep_chop_oos_layers import _make_engine  # noqa: E402
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

DEFAULT_YAML = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)

BASELINE = {
    "stability_min": 0.85,
    "width_min": 0.04,
    "width_max": 0.30,
    "touches_min": 5,
}


def _feat_base(defaults: dict) -> GridConfig:
    return GridConfig(
        box_window=int(defaults.get("box_window", 120)),
        chop_min=float(defaults.get("chop_min", 0.52)),
        exit_chop_min=float(defaults.get("exit_chop_min", 0.33)),
        min_segment_bars=int(defaults.get("min_segment_bars", 6)),
        max_segment_bars=int(defaults.get("max_segment_bars", 120)),
        grid_atr_mult=float(defaults.get("grid_atr_mult", 1.18)),
        grid_pct=float(defaults.get("grid_pct", 0.011)),
        max_levels=int(defaults.get("max_levels", 2)),
        fee_bps=float(defaults.get("fee_bps", 20.0)),
        chop_signal=str(defaults.get("chop_signal", "raw")),
        chop_ts_window=int(defaults.get("chop_ts_window", 1200)),
        chop_ts_min_periods=int(defaults.get("chop_ts_min_periods", 150)),
        compute_semantic_chop_ts_q=True,
        feature_store_dir=defaults.get("feature_store_dir"),
        feature_store_layer=defaults.get("feature_store_layer"),
        feature_store_timeframe=defaults.get("feature_store_timeframe"),
        stability_min=float(BASELINE["stability_min"]),
        width_min=float(BASELINE["width_min"]),
        width_max=float(BASELINE["width_max"]),
        touches_min=int(BASELINE["touches_min"]),
        prefilter_rules=tuple(
            x
            for x in (defaults.get("prefilter_rules", []) or [])
            if isinstance(x, dict)
        ),
    )


def _load_frames(
    symbols: Sequence[str],
    data_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feat_base: GridConfig,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]]:
    warmup = start - pd.Timedelta(days=120)
    sig_delta = timeframe_to_timedelta("120T")
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]] = {}
    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup, end)
        if raw.empty:
            continue
        bars_signal = _resample_ohlcv(raw, "120T")
        df = build_features(symbol, bars_signal, feat_base, bars_timeframe="120T")
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue
        bars_1m = _resample_ohlcv(raw, "1min")
        df_exec = merge_signal_features_onto_execution_bars(
            bars_1m, df, signal_bar_delta=sig_delta
        )
        out[symbol] = (df, df_exec, sig_delta)
    return out


def _run_cell(
    *,
    symbols: Sequence[str],
    symbol_frames: Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]
    ],
    cfg: GridConfig,
    engine: ChopGridEngine,
    box_params: dict,
    block_stable_box: bool,
) -> dict:
    all_trades: List[dict] = []
    all_segments: List[dict] = []
    box_rates: List[float] = []
    for symbol in symbols:
        df, df_exec, sig_delta = symbol_frames[symbol]
        df = recompute_box_prefilter_column(df, **box_params)
        box_rates.append(float(df["box_prefilter"].mean()))
        tlist, slist, _, entry_rate = collect_chop_grid_trades_for_symbol(
            symbol,
            df,
            df_exec,
            sig_delta,
            cfg,
            engine,
            block_stable_box=block_stable_box,
            exec_timeframe="1min",
            parquet_data_dir=Path("data/parquet_data"),
        )
        all_trades.extend(tlist)
        all_segments.extend(slist)
    trades = pd.DataFrame(all_trades)
    metrics = portfolio_metrics_from_trades(trades)
    metrics["n_trades"] = len(trades)
    metrics["n_segments"] = len(all_segments)
    metrics["mean_box_prefilter_rate"] = (
        float(sum(box_rates) / len(box_rates)) if box_rates else 0.0
    )
    if not trades.empty and "exit_reason" in trades.columns:
        metrics["grid_tp_rate"] = float((trades["exit_reason"] == "grid_tp").mean())
    return metrics


def _sweep_grid() -> List[dict]:
    """One-at-a-time + small 2D grid around baseline."""
    cells: List[dict] = [dict(BASELINE)]
    for k, vals in [
        ("stability_min", [0.75, 0.80, 0.85, 0.90, 0.95]),
        ("width_min", [0.03, 0.04, 0.05, 0.06]),
        ("width_max", [0.22, 0.26, 0.30, 0.34, 0.38]),
        ("touches_min", [2, 3, 5, 7, 9]),
    ]:
        for v in vals:
            if v == BASELINE[k]:
                continue
            c = dict(BASELINE)
            c[k] = v
            cells.append(c)
    for stab, wmax in itertools.product([0.80, 0.85, 0.90], [0.26, 0.30, 0.34]):
        c = dict(BASELINE)
        c["stability_min"] = stab
        c["width_max"] = wmax
        if c not in cells:
            cells.append(c)
    # dedupe
    seen = set()
    out: List[dict] = []
    for c in cells:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--config-yaml", default=str(DEFAULT_YAML))
    ap.add_argument(
        "--out-dir",
        default="results/chop_grid/experiments/box_prefilter_sweep_20260603",
    )
    ap.add_argument(
        "--compare-no-block",
        action="store_true",
        help="Also run baseline with block_stable_box=false (prior backtest bug path).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults = merge_chop_grid_yaml(Path(args.config_yaml))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    feat_base = _feat_base(defaults)
    symbol_frames = _load_frames(symbols, Path(args.data_dir), start, end, feat_base)
    max_replenish = defaults.get("max_replenish_per_level")
    if max_replenish is not None:
        max_replenish = int(max_replenish)

    block = bool(defaults.get("block_stable_box", True))
    engine = _make_engine(
        defaults,
        chop_min=feat_base.chop_min,
        exit_chop_min=feat_base.exit_chop_min,
        grid_atr_mult=feat_base.grid_atr_mult,
        grid_pct=feat_base.grid_pct,
        max_levels=feat_base.max_levels,
        max_replenish=max_replenish,
    )

    rows: List[dict] = []
    for params in _sweep_grid():
        label = (
            f"stab{params['stability_min']:.2f}_wmin{params['width_min']:.2f}_"
            f"wmax{params['width_max']:.2f}_touch{params['touches_min']}"
        )
        cfg = replace(
            feat_base,
            stability_min=float(params["stability_min"]),
            width_min=float(params["width_min"]),
            width_max=float(params["width_max"]),
            touches_min=int(params["touches_min"]),
        )
        m = _run_cell(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            box_params=params,
            block_stable_box=block,
        )
        row = {"label": label, "block_stable_box": block, **params, **m}
        rows.append(row)
        print(
            f"{label}: timeline={m.get('return_pct_timeline', 0):+.3f}% "
            f"trades={m.get('n_trades', 0)} box_rate={m.get('mean_box_prefilter_rate', 0):.1%}"
        )

    if args.compare_no_block:
        m0 = _run_cell(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=feat_base,
            engine=engine,
            box_params=BASELINE,
            block_stable_box=False,
        )
        rows.append(
            {
                "label": "baseline_no_block",
                "block_stable_box": False,
                **BASELINE,
                **m0,
            }
        )
        print(
            f"baseline_no_block: timeline={m0.get('return_pct_timeline', 0):+.3f}% "
            f"trades={m0.get('n_trades', 0)}"
        )

    csv_path = out_dir / "box_prefilter_oos.csv"
    keys: List[str] = []
    seen_k = set()
    for row in rows:
        for k in row:
            if k not in seen_k:
                seen_k.add(k)
                keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ranked = sorted(
        rows, key=lambda r: float(r.get("return_pct_timeline", -999)), reverse=True
    )
    baseline_row = next(
        (r for r in rows if r["label"].startswith("stab0.85_wmin0.04_wmax0.30_touch5")),
        ranked[0],
    )
    summary = {
        "window": f"{args.start} → {args.end}",
        "block_stable_box": block,
        "baseline_timeline": baseline_row.get("return_pct_timeline"),
        "spread_pp": float(ranked[0]["return_pct_timeline"])
        - float(ranked[-1]["return_pct_timeline"]),
        "top5": ranked[:5],
        "note": "block_stable_box aligns live chop (exclude_box_prefilter=false).",
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {csv_path}")
    print(f"Baseline (live-aligned): {baseline_row.get('return_pct_timeline'):+.3f}%")
    print(f"Best: {ranked[0]['label']} → {ranked[0].get('return_pct_timeline'):+.3f}%")


if __name__ == "__main__":
    main()
