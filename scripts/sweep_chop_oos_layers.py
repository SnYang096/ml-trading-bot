#!/usr/bin/env python3
"""OOS layer sweep: exec timeframe, spacing, regime, prefilter (live-aligned 1min default)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chop_grid_backtest import (  # noqa: E402
    ChopGridEngine,
    GridEngineConfig,
    collect_chop_grid_trades_for_symbol,
)
from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    build_features,
    merge_chop_grid_yaml,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.pipeline.multileg_portfolio_metrics import (  # noqa: E402
    portfolio_metrics_from_trades,
)
from src.time_series_model.grid.agg100ms_replay import (  # noqa: E402
    load_segment_tick_bars_from_parquet,
)
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

DEFAULT_YAML = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)


def _prefilter_box_range(lo: float, hi: float) -> Tuple[Dict[str, Any], ...]:
    from scripts.diagnose_chop_grid import prefilter_box_range

    return prefilter_box_range(lo, hi)


def resolve_prefilter_rules(
    defaults: dict,
    *,
    box_pos_min: float | None = None,
    box_pos_max: float | None = None,
) -> Tuple[Dict[str, Any], ...]:
    from scripts.diagnose_chop_grid import resolve_prefilter_rules as _resolve

    return _resolve(defaults, box_pos_min=box_pos_min, box_pos_max=box_pos_max)


def _run_portfolio(
    *,
    symbols: Sequence[str],
    symbol_frames: Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]
    ],
    cfg: GridConfig,
    engine: ChopGridEngine,
    block_stable_box: bool,
    exec_timeframe: str,
    agg_data_dir: Path | None,
    parquet_data_dir: Path,
) -> Dict[str, Any]:
    all_trades: List[dict] = []
    all_segments: List[dict] = []
    for symbol in symbols:
        df, df_exec, sig_delta = symbol_frames[symbol]
        tlist, slist, n_seg, entry_rate = collect_chop_grid_trades_for_symbol(
            symbol,
            df,
            df_exec,
            sig_delta,
            cfg,
            engine,
            block_stable_box=block_stable_box,
            exec_timeframe=exec_timeframe,
            agg_data_dir=agg_data_dir,
            parquet_data_dir=parquet_data_dir,
        )
        all_trades.extend(tlist)
        all_segments.extend(slist)
    trades = pd.DataFrame(all_trades)
    segments = pd.DataFrame(all_segments)
    metrics = portfolio_metrics_from_trades(trades)
    metrics["n_trades"] = len(trades)
    metrics["n_segments"] = len(segments)
    if not trades.empty and "exit_reason" in trades.columns:
        metrics["grid_tp_rate"] = float((trades["exit_reason"] == "grid_tp").mean())
        metrics["regime_exit_rate"] = float(
            (trades["exit_reason"] == "regime_exit").mean()
        )
    else:
        metrics["grid_tp_rate"] = 0.0
        metrics["regime_exit_rate"] = 0.0
    return metrics


def _make_engine(
    defaults: dict,
    *,
    chop_min: float,
    exit_chop_min: float,
    grid_atr_mult: float,
    grid_pct: float,
    max_levels: int,
    max_replenish: int | None,
) -> ChopGridEngine:
    fee = float(defaults.get("fee_bps", 20.0))
    slip = float(defaults.get("slippage_bps", 0.0))
    return ChopGridEngine(
        GridEngineConfig(
            box_window=int(defaults.get("box_window", 120)),
            entry_chop_min=chop_min,
            exit_chop_below=exit_chop_min,
            min_segment_bars=int(defaults.get("min_segment_bars", 6)),
            max_segment_bars=int(defaults.get("max_segment_bars", 120)),
            grid_atr_mult=grid_atr_mult,
            grid_min_pct=grid_pct,
            max_levels_per_side=max_levels,
            fee_bps=fee + slip,
            maker_fee_bps=float(defaults.get("maker_fee_bps", fee)),
            taker_fee_bps=float(defaults.get("taker_fee_bps", fee)),
            forced_exit_slippage_bps=float(
                defaults.get("forced_exit_slippage_bps", fee)
            )
            + slip,
            funding_cost_bps_per_8h=float(defaults.get("funding_cost_bps_per_8h", fee)),
            max_loss_per_grid=float(defaults.get("max_loss_per_grid", 0.03)),
            max_open_levels_total=int(defaults.get("max_open_levels_total", 4)),
            max_replenish_per_level_per_segment=max_replenish,
        )
    )


def analyze_parquet_tick_resolution(data_dir: Path, symbol: str, month: str) -> dict:
    path = data_dir / f"{symbol}_{month}.parquet"
    if not path.exists():
        return {"symbol": symbol, "month": month, "error": "missing"}
    df = pd.read_parquet(path, columns=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    uniq = ts.nunique()
    span_min = (ts.max() - ts.min()).total_seconds() / 60.0
    ms100 = ts.dt.floor("100ms").nunique()
    ms1m = ts.dt.floor("1min").nunique()
    delta = ts.sort_values().diff().dropna()
    return {
        "symbol": symbol,
        "month": month,
        "rows": len(df),
        "unique_timestamps": int(uniq),
        "floor_100ms_buckets": int(ms100),
        "floor_1min_buckets": int(ms1m),
        "median_delta_sec": float(delta.dt.total_seconds().median()),
        "p95_delta_sec": float(delta.dt.total_seconds().quantile(0.95)),
        "span_minutes": round(span_min, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
    )
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--config-yaml", default=str(DEFAULT_YAML))
    ap.add_argument(
        "--out-dir",
        default="results/chop_grid/experiments/oos_layer_sweep_20260603",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    defaults = merge_chop_grid_yaml(Path(args.config_yaml))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=120)
    sig_delta = timeframe_to_timedelta("120T")
    agg_dir = Path(defaults.get("agg_data_dir") or "data/agg_data")

    tick_diag = [
        analyze_parquet_tick_resolution(data_dir, sym, "2025-10") for sym in symbols[:2]
    ]
    (out_dir / "parquet_tick_resolution.json").write_text(
        json.dumps(tick_diag, indent=2), encoding="utf-8"
    )
    print("Tick resolution:", json.dumps(tick_diag, indent=2))

    feat_base = GridConfig(
        box_window=int(defaults.get("box_window", 120)),
        chop_min=float(defaults.get("chop_min", 0.50)),
        exit_chop_min=float(defaults.get("exit_chop_min", 0.32)),
        min_segment_bars=int(defaults.get("min_segment_bars", 6)),
        max_segment_bars=int(defaults.get("max_segment_bars", 120)),
        grid_atr_mult=float(defaults.get("grid_atr_mult", 1.0)),
        grid_pct=float(defaults.get("grid_pct", 0.010)),
        max_levels=int(defaults.get("max_levels", 2)),
        fee_bps=float(defaults.get("fee_bps", 20.0)),
        chop_signal=str(defaults.get("chop_signal", "raw")),
        chop_ts_window=int(defaults.get("chop_ts_window", 1200)),
        chop_ts_min_periods=int(defaults.get("chop_ts_min_periods", 150)),
        compute_semantic_chop_ts_q=True,
        feature_store_dir=defaults.get("feature_store_dir"),
        feature_store_layer=defaults.get("feature_store_layer"),
        feature_store_timeframe=defaults.get("feature_store_timeframe"),
        stability_min=float(defaults.get("stability_min", 0.85)),
        width_min=float(defaults.get("width_min", 0.04)),
        width_max=float(defaults.get("width_max", 0.30)),
        touches_min=int(defaults.get("touches_min", 5)),
        prefilter_rules=tuple(
            x
            for x in (defaults.get("prefilter_rules", []) or [])
            if isinstance(x, dict)
        ),
    )

    symbol_frames: Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]
    ] = {}
    raw_by_symbol: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        raw_by_symbol[symbol] = raw
        bars_signal = _resample_ohlcv(raw, "120T")
        df = build_features(symbol, bars_signal, feat_base)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue
        bars_1m = _resample_ohlcv(raw, "1min")
        df_exec_1m = merge_signal_features_onto_execution_bars(
            bars_1m, df, signal_bar_delta=sig_delta
        )
        symbol_frames[symbol] = (df, df_exec_1m, sig_delta)

    max_replenish = defaults.get("max_replenish_per_level")
    if max_replenish is not None:
        max_replenish = int(max_replenish)

    rows: List[dict] = []

    def add_row(label: str, sweep: str, metrics: dict, **extra: Any) -> None:
        row = {"label": label, "sweep": sweep, **extra, **metrics}
        rows.append(row)
        print(
            f"{label:40s} timeline={metrics.get('return_pct_timeline', 0):+.3f}% "
            f"trades={metrics.get('n_trades', 0)} tp={metrics.get('grid_tp_rate', 0):.1%}"
        )

    prod_chop = (
        float(defaults.get("chop_min", 0.50)),
        float(defaults.get("exit_chop_min", 0.32)),
    )
    prod_spacing = (
        float(defaults.get("grid_atr_mult", 1.0)),
        float(defaults.get("grid_pct", 0.010)),
    )
    prod_block_stable_box = bool(defaults.get("block_stable_box", True))

    # --- exec timeframe: 1min vs 100ms (parquet ticks) ---
    for exec_tf in ("1min", "100ms"):
        cfg = deepcopy(feat_base)
        engine = _make_engine(
            defaults,
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            grid_atr_mult=prod_spacing[0],
            grid_pct=prod_spacing[1],
            max_levels=int(defaults.get("max_levels", 2)),
            max_replenish=max_replenish,
        )
        frames = symbol_frames
        if exec_tf == "100ms":
            frames = {
                sym: (symbol_frames[sym][0], None, sig_delta) for sym in symbol_frames
            }
        m = _run_portfolio(
            symbols=list(symbol_frames),
            symbol_frames=frames,
            cfg=cfg,
            engine=engine,
            block_stable_box=prod_block_stable_box,
            exec_timeframe=exec_tf,
            agg_data_dir=agg_dir if agg_dir.exists() else None,
            parquet_data_dir=data_dir,
        )
        add_row(
            f"prod_{exec_tf}",
            "exec_timeframe",
            m,
            exec_timeframe=exec_tf,
            atr_mult=prod_spacing[0],
            min_pct=prod_spacing[1],
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            block_stable_box=prod_block_stable_box,
        )

    # --- spacing grid around 1.0/1.0% and 1.25/1.2% ---
    spacing_grid = [
        (0.80, 0.008),
        (0.90, 0.009),
        (1.00, 0.010),
        (1.10, 0.011),
        (1.20, 0.011),
        (1.25, 0.012),
        (1.30, 0.012),
        (1.35, 0.013),
        (1.40, 0.013),
        (1.50, 0.015),
    ]
    for atr_m, min_p in spacing_grid:
        cfg = deepcopy(feat_base)
        engine = _make_engine(
            defaults,
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            grid_atr_mult=atr_m,
            grid_pct=min_p,
            max_levels=int(defaults.get("max_levels", 2)),
            max_replenish=max_replenish,
        )
        m = _run_portfolio(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            block_stable_box=prod_block_stable_box,
            exec_timeframe="1min",
            agg_data_dir=None,
            parquet_data_dir=data_dir,
        )
        add_row(
            f"spacing_{atr_m:.2f}_{min_p:.3f}",
            "spacing",
            m,
            exec_timeframe="1min",
            atr_mult=atr_m,
            min_pct=min_p,
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            block_stable_box=prod_block_stable_box,
        )

    # --- regime thresholds (raw + ts_quantile) ---
    regime_pairs = [
        ("raw", 0.50, 0.32),
        ("raw", 0.45, 0.30),
        ("raw", 0.55, 0.35),
        ("raw", 0.35, 0.22),
        ("ts_quantile", 0.50, 0.32),
        ("ts_quantile", 0.45, 0.28),
        ("ts_quantile", 0.55, 0.35),
    ]
    best_spacing = max(
        (r for r in rows if r["sweep"] == "spacing"),
        key=lambda r: r.get("return_pct_timeline", -999),
    )
    best_atr = float(best_spacing["atr_mult"])
    best_min = float(best_spacing["min_pct"])

    for sig, cm, em in regime_pairs:
        cfg = GridConfig(
            **{
                **feat_base.__dict__,
                "chop_min": cm,
                "exit_chop_min": em,
                "chop_signal": sig,
                "grid_atr_mult": best_atr,
                "grid_pct": best_min,
            }
        )
        engine = _make_engine(
            defaults,
            chop_min=cm,
            exit_chop_min=em,
            grid_atr_mult=best_atr,
            grid_pct=best_min,
            max_levels=int(defaults.get("max_levels", 2)),
            max_replenish=max_replenish,
        )
        m = _run_portfolio(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            block_stable_box=prod_block_stable_box,
            exec_timeframe="1min",
            agg_data_dir=None,
            parquet_data_dir=data_dir,
        )
        add_row(
            f"regime_{sig}_{cm}_{em}",
            "regime",
            m,
            exec_timeframe="1min",
            chop_signal=sig,
            chop_min=cm,
            exit_chop_min=em,
            atr_mult=best_atr,
            min_pct=best_min,
            block_stable_box=prod_block_stable_box,
        )

    # --- prefilter variants (best spacing + prod regime) ---
    prefilter_variants: List[Tuple[str, Tuple[Dict[str, Any], ...] | None, bool]] = [
        ("prod_box_35_65", _prefilter_box_range(0.35, 0.65), False),
        ("no_prefilter_rules", (), False),
        ("box_30_70", _prefilter_box_range(0.30, 0.70), False),
        ("box_40_60", _prefilter_box_range(0.40, 0.60), False),
        ("no_block_stable_box", _prefilter_box_range(0.35, 0.65), False),
    ]
    for name, rules, block_sb in prefilter_variants:
        cfg = GridConfig(
            **{
                **feat_base.__dict__,
                "grid_atr_mult": best_atr,
                "grid_pct": best_min,
                "prefilter_rules": rules or (),
            }
        )
        engine = _make_engine(
            defaults,
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            grid_atr_mult=best_atr,
            grid_pct=best_min,
            max_levels=int(defaults.get("max_levels", 2)),
            max_replenish=max_replenish,
        )
        m = _run_portfolio(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            block_stable_box=block_sb,
            exec_timeframe="1min",
            agg_data_dir=None,
            parquet_data_dir=data_dir,
        )
        add_row(
            name,
            "prefilter",
            m,
            exec_timeframe="1min",
            atr_mult=best_atr,
            min_pct=best_min,
            chop_min=prod_chop[0],
            exit_chop_min=prod_chop[1],
            block_stable_box=block_sb,
            prefilter=name,
        )

    csv_path = out_dir / "sweep_results.csv"
    if rows:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "window": f"{args.start} → {args.end}",
        "symbols": symbols,
        "tick_resolution": tick_diag,
        "best_spacing": {
            "atr_mult": best_atr,
            "min_pct": best_min,
            "return_pct_timeline": best_spacing.get("return_pct_timeline"),
        },
        "top5": sorted(
            rows,
            key=lambda r: r.get("return_pct_timeline", -999),
            reverse=True,
        )[:5],
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Summary -> {out_dir / 'SUMMARY.json'}")


if __name__ == "__main__":
    main()
