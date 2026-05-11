#!/usr/bin/env python3
"""Grid-search chop_min / exit_chop_min (and optional ts window) for chop_grid backtest.

Loads OHLC once per symbol per (ts_window, ts_min_periods), builds features once,
then reuses the feature frame for all (chop_signal, chop_min, exit_chop_min) combos
so threshold sweeps stay cheap.

Raw vs rolling quantile (semantic_chop_ts_q):
  * **raw** semantic_chop is an absolute 0–1 score; thresholds are comparable across
    long samples if the feature definition is stable.
  * **ts_quantile** is relative to each symbol's recent distribution — good when
    volatility regime shifts and you want "high chop vs this coin's last N bars".
    Numeric cutoffs are **not** interchangeable with raw; sweep both on the same
    calendar range and compare out-of-sample metrics.

Example::

    python scripts/sweep_chop_regime_thresholds.py \\
      --symbols BTCUSDT --start 2024-01-01 --end 2024-12-31 --no-exec-merge \\
      --sweep-signals raw,ts_quantile \\
      --sweep-chop-min 0.35,0.4,0.45,0.5,0.55,0.6 \\
      --sweep-exit-chop-min 0.15,0.2,0.25,0.3 \\
      --sweep-ts-windows 1200x150,480x60 \\
      --out-csv results/chop_grid_sweep.csv

``dual_add`` / trend regime: use ``scripts/diagnose_dual_add_trend.py`` with the same
``--chop-min`` / ``--exit-chop-min`` ideas, or wrap this CSV workflow for ``chop`` regime
in a shell loop (that script does not share this sweeper yet).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chop_grid_backtest import (  # noqa: E402
    ChopGridEngine,
    GridEngineConfig,
    _load_grid_defaults,
    collect_chop_grid_trades_for_symbol,
    summarize_dual_add_aligned,
)
from scripts.diagnose_chop_grid import GridConfig, build_features  # noqa: E402
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

DEFAULT_GRID_CONFIG = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_signals(s: str) -> List[str]:
    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    for x in out:
        if x not in {"raw", "ts_quantile"}:
            raise ValueError(f"unknown signal {x!r} (use raw or ts_quantile)")
    if not out:
        raise ValueError("empty --sweep-signals")
    return out


def _parse_ts_windows(spec: str) -> List[Tuple[int, int]]:
    """``1200x150`` or ``1200:150`` comma-separated pairs -> [(window, min_periods), ...]."""
    pairs: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        sep = "x" if "x" in part.lower() else ":"
        a, b = part.lower().split(sep, 1)
        pairs.append((int(a.strip()), int(b.strip())))
    if not pairs:
        raise ValueError("empty --sweep-ts-windows")
    return pairs


def _threshold_pairs(
    cm_list: Sequence[float], em_list: Sequence[float]
) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for cm in cm_list:
        for em in em_list:
            if em < cm:
                pairs.append((cm, em))
    return pairs


def _one_row_metrics(trades: pd.DataFrame, segments: pd.DataFrame) -> dict:
    if trades.empty or segments.empty:
        return {
            "segments": 0,
            "trades": 0,
            "return_pct": 0.0,
            "segment_win_rate": 0.0,
            "sum_pnl_per_capital": 0.0,
            "risk_stop_rate": 0.0,
            "tp_rate": 0.0,
            "forced_rate": 0.0,
        }
    s = summarize_dual_add_aligned(trades, segments)
    return s.iloc[0].to_dict()


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_GRID_CONFIG))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    defaults = _load_grid_defaults(config_path)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(config_path))
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--warmup-days", type=int, default=120)
    ap.add_argument("--timeframe", default="2h")
    ap.add_argument("--execution-timeframe", default=None)
    ap.add_argument(
        "--no-exec-merge",
        action="store_true",
        help="Ignore execution timeframe (faster sweep; signal bars only).",
    )
    ap.add_argument(
        "--box-window",
        type=int,
        default=defaults.get("box_window", 120),
        choices=[60, 120, 240],
    )
    ap.add_argument(
        "--min-segment-bars", type=int, default=defaults.get("min_segment_bars", 6)
    )
    ap.add_argument(
        "--max-segment-bars", type=int, default=defaults.get("max_segment_bars", 120)
    )
    ap.add_argument(
        "--grid-atr-mult", type=float, default=defaults.get("grid_atr_mult", 0.50)
    )
    ap.add_argument("--grid-pct", type=float, default=defaults.get("grid_pct", 0.004))
    ap.add_argument("--max-levels", type=int, default=defaults.get("max_levels", 3))
    ap.add_argument("--fee-bps", type=float, default=defaults.get("fee_bps", 4.0))
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument(
        "--maker-fee-bps", type=float, default=defaults.get("maker_fee_bps", 4.0)
    )
    ap.add_argument(
        "--taker-fee-bps", type=float, default=defaults.get("taker_fee_bps", 4.0)
    )
    ap.add_argument(
        "--forced-exit-slippage-bps",
        type=float,
        default=defaults.get("forced_exit_slippage_bps", 0.0),
    )
    ap.add_argument(
        "--funding-cost-bps-per-8h",
        type=float,
        default=defaults.get("funding_cost_bps_per_8h", 0.0),
    )
    ap.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    ap.add_argument(
        "--max-loss-per-grid",
        type=float,
        default=defaults.get("max_loss_per_grid", 0.03),
    )
    ap.add_argument(
        "--max-open-levels-total",
        type=int,
        default=defaults.get("max_open_levels_total", 6),
    )
    ap.add_argument(
        "--sweep-signals",
        default="raw,ts_quantile",
        help="Comma list: raw, ts_quantile (or one of them).",
    )
    ap.add_argument(
        "--sweep-chop-min",
        required=True,
        help="Comma-separated entry thresholds (e.g. 0.35,0.4,0.45,0.5).",
    )
    ap.add_argument(
        "--sweep-exit-chop-min",
        required=True,
        help="Comma-separated hold/exit thresholds; only pairs with exit < chop are run.",
    )
    ap.add_argument(
        "--sweep-ts-windows",
        default="1200x150",
        help="Comma-separated window×min_periods for ts_quantile feature build, e.g. 1200x150,480x60.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/chop_grid_threshold_sweep.csv"),
    )
    args = ap.parse_args()
    signals = _parse_signals(args.sweep_signals)
    need_ts_col = "ts_quantile" in signals
    cm_grid = _parse_csv_floats(args.sweep_chop_min)
    em_grid = _parse_csv_floats(args.sweep_exit_chop_min)
    pairs = _threshold_pairs(cm_grid, em_grid)
    if not pairs:
        raise SystemExit("No valid (chop_min, exit_chop_min) pairs with exit < chop.")

    ts_specs = _parse_ts_windows(args.sweep_ts_windows)
    if not need_ts_col and len(ts_specs) > 1:
        ts_specs = [ts_specs[0]]

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)

    exec_tf = None if args.no_exec_merge else args.execution_timeframe
    rows_out: List[dict] = []

    for win, mp in ts_specs:
        feat_base = GridConfig(
            box_window=args.box_window,
            chop_min=0.4,
            exit_chop_min=0.25,
            min_segment_bars=args.min_segment_bars,
            max_segment_bars=args.max_segment_bars,
            grid_atr_mult=args.grid_atr_mult,
            grid_pct=args.grid_pct,
            max_levels=args.max_levels,
            fee_bps=args.fee_bps,
            chop_signal="raw",
            chop_ts_window=win,
            chop_ts_min_periods=mp,
            compute_semantic_chop_ts_q=need_ts_col,
        )
        for symbol in symbols:
            raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
            if raw.empty:
                print(f"skip {symbol}: no data")
                continue
            bars_signal = _resample_ohlcv(raw, args.timeframe)
            df = build_features(symbol, bars_signal, feat_base)
            df = df[(df.index >= start) & (df.index <= end)].copy()
            if df.empty:
                continue

            if exec_tf and exec_tf != args.timeframe:
                bars_exec = _resample_ohlcv(raw, exec_tf)
                sig_delta = timeframe_to_timedelta(args.timeframe)
                df_exec = merge_signal_features_onto_execution_bars(
                    bars_exec, df, signal_bar_delta=sig_delta
                )
            else:
                df_exec = None
                sig_delta = None

            for sig in signals:
                if sig == "ts_quantile" and not need_ts_col:
                    continue
                for cm, em in pairs:
                    cfg = GridConfig(
                        box_window=args.box_window,
                        chop_min=cm,
                        exit_chop_min=em,
                        min_segment_bars=args.min_segment_bars,
                        max_segment_bars=args.max_segment_bars,
                        grid_atr_mult=args.grid_atr_mult,
                        grid_pct=args.grid_pct,
                        max_levels=args.max_levels,
                        fee_bps=args.fee_bps,
                        chop_signal=sig,
                        chop_ts_window=win,
                        chop_ts_min_periods=mp,
                        compute_semantic_chop_ts_q=need_ts_col,
                    )
                    engine = ChopGridEngine(
                        GridEngineConfig(
                            box_window=args.box_window,
                            entry_chop_min=cm,
                            exit_chop_below=em,
                            min_segment_bars=args.min_segment_bars,
                            max_segment_bars=args.max_segment_bars,
                            grid_atr_mult=args.grid_atr_mult,
                            grid_min_pct=args.grid_pct,
                            max_levels_per_side=args.max_levels,
                            fee_bps=args.fee_bps + args.slippage_bps,
                            maker_fee_bps=args.maker_fee_bps,
                            taker_fee_bps=args.taker_fee_bps,
                            forced_exit_slippage_bps=args.forced_exit_slippage_bps
                            + args.slippage_bps,
                            funding_cost_bps_per_8h=args.funding_cost_bps_per_8h,
                            max_loss_per_grid=args.max_loss_per_grid,
                            max_open_levels_total=args.max_open_levels_total,
                        )
                    )
                    tlist, slist, n_seg, entry_rate = (
                        collect_chop_grid_trades_for_symbol(
                            symbol,
                            df,
                            df_exec,
                            sig_delta,
                            cfg,
                            engine,
                            exclude_box=bool(args.exclude_box),
                        )
                    )
                    trades = pd.DataFrame(tlist)
                    segments = pd.DataFrame(slist)
                    m = _one_row_metrics(trades, segments)
                    row = {
                        "symbol": symbol,
                        "chop_signal": sig,
                        "chop_ts_window": win,
                        "chop_ts_min_periods": mp,
                        "chop_min": cm,
                        "exit_chop_min": em,
                        "entry_mask_rate": round(entry_rate, 6),
                    }
                    row.update(m)
                    row["segments_built"] = n_seg
                    rows_out.append(row)
                    print(
                        f"{symbol} {sig} w={win} mp={mp} cm={cm} em={em} "
                        f"segs={n_seg} return_pct={m.get('return_pct', 0):.4f}"
                    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        print("No rows written (no data or no valid symbols).")
        return
    keys = list(rows_out[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
