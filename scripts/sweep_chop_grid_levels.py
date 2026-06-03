#!/usr/bin/env python3
"""Sweep: max_levels_per_side × atr_mult on recent_6m_oos.

Research question: does placing more grid levels within the chop range improve
profitability?

Two controlled experiments:

  fixed_spacing  – keep atr_mult constant (1.18); adding levels widens the total
                   grid range.  Tests whether outer levels ever get filled.

  fixed_span     – reduce atr_mult proportionally so that the total grid span
                   (atr_mult × max_levels_per_side) stays ≈ baseline (2 × 1.18).
                   Tests whether denser layers within the same price window improve
                   round-trip capture rate.

Outputs (results/chop_grid/experiments/levels_sweep_YYYYMMDD/):
  summary.csv         — one row per configuration, portfolio-level metrics
  segment_stats.csv   — per-configuration segment-level utilisation stats
  README.md           — interpretation summary
"""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

DEFAULT_YAML = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
DEFAULT_START = "2025-10-01"
DEFAULT_END = "2026-03-31"


# ---------------------------------------------------------------------------
# Grid variants
# ---------------------------------------------------------------------------

BASELINE_LEVELS = 2
BASELINE_ATR_MULT = 1.18
BASELINE_SPAN = BASELINE_LEVELS * BASELINE_ATR_MULT  # ≈ 2.36 ATR per side


# Median chop segment range ≈ 2.0% peak-to-trough (±1.0% from center).
# To place dense levels INSIDE the box, per-side span must be ≤ ~1.0%, so spacing
# (and the binding min_pct floor) must drop well below the production 1.1%.
MEDIAN_RANGE_PCT = 0.020


def _build_variants() -> List[Dict[str, Any]]:
    """Configurations sweeping grid DENSITY inside the box (min_pct is the lever).

    Each variant pins spacing via min_pct (atr_mult set high so min_pct binds),
    so we directly control how many levels fit inside the ~±1% box.
    """
    variants: List[Dict[str, Any]] = []

    # Baseline (production): min_pct=1.1%, 2 levels — most levels land outside box.
    variants.append(
        {
            "id": "baseline_prod",
            "strategy": "baseline",
            "max_levels": BASELINE_LEVELS,
            "atr_mult": BASELINE_ATR_MULT,
            "min_pct": 0.011,
            "label": "baseline 2L × min_pct 1.10%",
        }
    )

    # Dense-inside-box: target per-side span ≈ 1.0% (= half the median 2% range),
    # so all levels sit inside the box. spacing = 1.0% / n_levels.
    for n in (2, 3, 4, 5):
        target_span = MEDIAN_RANGE_PCT / 2.0  # ~1.0% per side
        spacing = round(target_span / n, 4)
        variants.append(
            {
                "id": f"dense_{n}L",
                "strategy": "dense_inside_box",
                "max_levels": n,
                "atr_mult": 0.01,  # tiny → min_pct (floor) always binds
                "min_pct": spacing,
                "label": f"dense {n}L × min_pct {spacing*100:.2f}% (span≈1.0%)",
            }
        )

    # Half-density: per-side span ≈ 0.6% (tighter, well inside box).
    for n in (3, 4, 5):
        spacing = round(0.006 / n, 4)
        variants.append(
            {
                "id": f"tight_{n}L",
                "strategy": "tight_inside_box",
                "max_levels": n,
                "atr_mult": 0.01,
                "min_pct": spacing,
                "label": f"tight {n}L × min_pct {spacing*100:.2f}% (span≈0.6%)",
            }
        )

    return variants


# ---------------------------------------------------------------------------
# Engine / backtest helpers
# ---------------------------------------------------------------------------


def _make_engine(
    defaults: dict,
    *,
    max_levels: int,
    atr_mult: float,
    min_pct: float,
) -> ChopGridEngine:
    return ChopGridEngine(
        GridEngineConfig(
            box_window=int(defaults.get("box_window", 120)),
            entry_chop_min=float(defaults.get("chop_min", 0.52)),
            exit_chop_below=float(defaults.get("exit_chop_min", 0.33)),
            min_segment_bars=int(defaults.get("min_segment_bars", 6)),
            max_segment_bars=int(defaults.get("max_segment_bars", 120)),
            grid_atr_mult=atr_mult,
            grid_min_pct=min_pct,
            max_levels_per_side=max_levels,
            fee_bps=float(defaults.get("fee_bps", 20.0)),
            maker_fee_bps=float(defaults.get("maker_fee_bps", 20.0)),
            taker_fee_bps=float(defaults.get("taker_fee_bps", 20.0)),
            forced_exit_slippage_bps=float(
                defaults.get("forced_exit_slippage_bps", 20.0)
            ),
            funding_cost_bps_per_8h=float(defaults.get("funding_cost_bps_per_8h", 0.0)),
            max_loss_per_grid=float(defaults.get("max_loss_per_grid", 0.03)),
            max_open_levels_total=max_levels * 2,  # symmetric cap scales with depth
            max_replenish_per_level_per_segment=1,
        )
    )


def _run_variant(
    *,
    symbols: Sequence[str],
    symbol_frames: Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]
    ],
    cfg: GridConfig,
    engine: ChopGridEngine,
    defaults: dict,
    exec_timeframe: str = "1min",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run backtest for all symbols; return (portfolio_metrics, segment_df)."""
    all_trades: List[dict] = []
    all_segments: List[dict] = []
    agg_dir = Path(defaults.get("agg_data_dir") or "data/agg_data")
    parquet_data_dir = Path(defaults.get("data_dir") or "data/parquet_data")

    for symbol in symbols:
        if symbol not in symbol_frames:
            continue
        df, df_exec, sig_delta = symbol_frames[symbol]
        tlist, slist, _n_seg, _entry_rate = collect_chop_grid_trades_for_symbol(
            symbol,
            df,
            df_exec,
            sig_delta,
            cfg,
            engine,
            block_stable_box=not bool(defaults.get("exclude_box_prefilter", False)),
            exec_timeframe=exec_timeframe,
            agg_data_dir=agg_dir if agg_dir.exists() else None,
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
    return metrics, segments


def _segment_utilisation(segments: pd.DataFrame, max_levels: int) -> Dict[str, Any]:
    """Aggregate segment-level grid-utilisation stats."""
    if segments.empty:
        return {
            "seg_count": 0,
            "avg_trades_per_seg": 0.0,
            "avg_max_open_levels": 0.0,
            "pct_segs_fully_packed": 0.0,  # max_open == 2×levels
            "avg_grid_full_span_pct": 0.0,
            "avg_segment_range_pct": 0.0,
            "avg_span_to_range": 0.0,  # <1 = grid fits inside price range
        }
    n = len(segments)
    avg_trades = (
        float(segments["trades"].mean()) if "trades" in segments.columns else 0.0
    )
    avg_open = (
        float(segments["max_open_levels"].mean())
        if "max_open_levels" in segments.columns
        else 0.0
    )
    pct_full = (
        float((segments["max_open_levels"] >= max_levels * 2).mean())
        if "max_open_levels" in segments.columns
        else 0.0
    )
    avg_span = (
        float(segments["grid_full_span_pct"].mean())
        if "grid_full_span_pct" in segments.columns
        else 0.0
    )
    avg_range = (
        float(segments["segment_range_pct"].mean())
        if "segment_range_pct" in segments.columns
        else 0.0
    )
    avg_s2r = (
        float(segments["grid_full_span_to_range"].mean())
        if "grid_full_span_to_range" in segments.columns
        else 0.0
    )
    return {
        "seg_count": n,
        "avg_trades_per_seg": round(avg_trades, 2),
        "avg_max_open_levels": round(avg_open, 2),
        "pct_segs_fully_packed": round(pct_full, 3),
        "avg_grid_full_span_pct": round(avg_span, 4),
        "avg_segment_range_pct": round(avg_range, 4),
        "avg_span_to_range": round(avg_s2r, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Sweep max_levels_per_side × atr_mult on chop_grid OOS"
    )
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--config-yaml", default=str(DEFAULT_YAML))
    ap.add_argument(
        "--out-dir",
        default=f"results/chop_grid/experiments/levels_sweep_{datetime.now():%Y%m%d}",
    )
    ap.add_argument(
        "--min-pct",
        type=float,
        default=None,
        help="Override grid_min_pct (default: from config, typically 0.011)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    defaults = merge_chop_grid_yaml(Path(args.config_yaml))
    defaults["data_dir"] = str(data_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=120)
    sig_delta = timeframe_to_timedelta("120T")
    min_pct = args.min_pct or float(defaults.get("grid_pct", 0.011))

    print(f"Data:    {data_dir}  ({args.start} – {args.end})")
    print(f"Symbols: {symbols}")
    print(f"min_pct floor: {min_pct:.4f}  ({min_pct*100:.2f}%)")
    print(f"Output:  {out_dir}")
    print()

    # Build features once (shared across all variants)
    feat_base = GridConfig(
        box_window=int(defaults.get("box_window", 120)),
        chop_min=float(defaults.get("chop_min", 0.52)),
        exit_chop_min=float(defaults.get("exit_chop_min", 0.33)),
        min_segment_bars=int(defaults.get("min_segment_bars", 6)),
        max_segment_bars=int(defaults.get("max_segment_bars", 120)),
        grid_atr_mult=BASELINE_ATR_MULT,
        grid_pct=min_pct,
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

    print("Loading price data and computing features …")
    symbol_frames: Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]
    ] = {}
    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"  skip {symbol}: no data")
            continue
        bars_signal = _resample_ohlcv(raw, "120T")
        df = build_features(symbol, bars_signal, feat_base)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            print(f"  skip {symbol}: empty after feature build")
            continue
        bars_1m = _resample_ohlcv(raw, "1min")
        df_exec = merge_signal_features_onto_execution_bars(
            bars_1m, df, signal_bar_delta=sig_delta
        )
        symbol_frames[symbol] = (df, df_exec, sig_delta)
        print(f"  {symbol}: {len(df)} signal bars, {len(df_exec)} exec bars")

    if not symbol_frames:
        print("No data loaded — exiting.")
        return

    variants = _build_variants()
    summary_rows: List[dict] = []
    seg_rows: List[dict] = []

    print(
        f"\n{'Variant':<45} {'total_r':>8} {'n_trades':>8} {'tp_rate':>7} "
        f"{'avg_open':>8} {'span/range':>10} {'%full':>6}"
    )
    print("-" * 100)

    for v in variants:
        v_min_pct = float(v.get("min_pct", min_pct))
        cfg = deepcopy(
            feat_base
        )  # features don't depend on spacing; engine drives grid
        engine = _make_engine(
            defaults,
            max_levels=v["max_levels"],
            atr_mult=v["atr_mult"],
            min_pct=v_min_pct,
        )
        metrics, segments = _run_variant(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            defaults=defaults,
        )
        util = _segment_utilisation(segments, v["max_levels"])

        total_r = metrics.get("return_pct_timeline", 0.0)
        n_trades = metrics.get("n_trades", 0)
        tp_rate = metrics.get("grid_tp_rate", 0.0)
        avg_open = util["avg_max_open_levels"]
        span_to_range = util["avg_span_to_range"]
        pct_full = util["pct_segs_fully_packed"]

        row = {
            "id": v["id"],
            "strategy": v["strategy"],
            "label": v["label"],
            "max_levels": v["max_levels"],
            "atr_mult": v["atr_mult"],
            "min_pct": v_min_pct,
            **{
                k: round(float(val), 4) if isinstance(val, float) else val
                for k, val in metrics.items()
            },
            **{
                k: round(float(val), 4) if isinstance(val, float) else val
                for k, val in util.items()
            },
        }
        summary_rows.append(row)

        for _, seg in (segments.iterrows() if not segments.empty else []):
            seg_rows.append(
                {
                    "variant_id": v["id"],
                    "strategy": v["strategy"],
                    "max_levels": v["max_levels"],
                    "atr_mult": v["atr_mult"],
                    **{
                        c: seg[c]
                        for c in segments.columns
                        if c
                        in {
                            "symbol",
                            "trades",
                            "max_open_levels",
                            "grid_full_span_pct",
                            "segment_range_pct",
                            "grid_full_span_to_range",
                            "bars",
                            "entry_chop",
                            "median_chop",
                        }
                        and c in segments.index
                    },
                }
            )

        print(
            f"{v['label']:<45} {total_r:>+8.3f}% {n_trades:>8} {tp_rate:>7.1%} "
            f"{avg_open:>8.2f} {span_to_range:>10.2f} {pct_full:>6.1%}"
        )

    # Save outputs
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    if seg_rows:
        seg_df = pd.DataFrame(seg_rows)
        seg_df.to_csv(out_dir / "segment_stats.csv", index=False)

    # Print interpretation table
    _print_interpretation(summary_df, out_dir)
    print(f"\nResults saved → {out_dir}/")


def _print_interpretation(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    baseline = (
        df[df["id"] == "baseline_2L"].iloc[0]
        if "baseline_2L" in df["id"].values
        else None
    )

    lines = [
        "# Chop Grid Levels Sweep — Results",
        "",
        "## Setup",
        f"- Baseline: {BASELINE_LEVELS} levels/side × {BASELINE_ATR_MULT:.2f} ATR spacing",
        f"- Fixed-spacing: same ATR mult, more levels → wider total grid",
        f"- Fixed-span: reduce ATR mult proportionally, same total span → denser levels",
        "",
        "## Key columns",
        "- `total_r`: portfolio return % over the OOS window",
        "- `avg_max_open_levels`: average max simultaneously open levels per segment",
        "- `avg_span_to_range`: grid total width ÷ actual price range in segment",
        "  - <1 = grid fits inside move; >1 = outer levels never reached",
        "- `pct_segs_fully_packed`: fraction of segments where all levels filled",
        "",
        "## Results",
        "",
        df[
            [
                "label",
                "return_pct_timeline",
                "n_trades",
                "grid_tp_rate",
                "avg_max_open_levels",
                "avg_span_to_range",
                "pct_segs_fully_packed",
                "n_segments",
            ]
        ].to_string(index=False),
        "",
        "## Key insight",
    ]

    if baseline is not None:
        span_to_range_base = baseline.get("avg_span_to_range", 1.0)
        if span_to_range_base > 1.2:
            lines.append(
                f"- Baseline grid span/range = {span_to_range_base:.2f} → "
                "outer levels already rarely reached. Adding more layers at same spacing "
                "will fill even less — **fixed_span (denser) is the right direction**."
            )
        elif span_to_range_base < 0.8:
            lines.append(
                f"- Baseline grid span/range = {span_to_range_base:.2f} → "
                "price moves beyond the grid regularly. **fixed_spacing (wider grid) "
                "might capture more**."
            )
        else:
            lines.append(
                f"- Baseline grid span/range ≈ {span_to_range_base:.2f} → "
                "grid roughly matches price range. Both directions worth comparing."
            )

    readme = "\n".join(lines)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
