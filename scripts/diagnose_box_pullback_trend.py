"""Diagnostic for box/chop pullbacks aligned with macro trend.

This is a research-only probe for a possible strategy distinct from CRF:

  - box/chop defines the pullback window
  - BTC EMA1200 state defines macro direction
  - only take box-edge entries aligned with macro direction

The goal is to test whether these windows are additive to TPC-style trend
pullbacks before creating a full strategy family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_crf_edge import (  # noqa: E402
    StudyConfig,
    _load_symbol_1m,
    _resample_ohlcv,
    build_symbol_dataset,
    collect_execution_samples,
)


def _macro_state_from_btc(
    data_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_days: int,
    timeframe: str,
    ema_span: int,
    slope_bars: int,
) -> pd.DataFrame:
    raw = _load_symbol_1m(
        data_dir, "BTCUSDT", start - pd.Timedelta(days=warmup_days), end
    )
    if raw.empty:
        raise SystemExit("BTCUSDT data is required for macro filter")
    bars = _resample_ohlcv(raw, timeframe)
    btc = bars[(bars.index >= start) & (bars.index <= end)].copy()
    btc["btc_ema"] = (
        btc["close"]
        .ewm(span=ema_span, adjust=False, min_periods=max(30, ema_span // 10))
        .mean()
    )
    btc["btc_ema_slope"] = btc["btc_ema"].pct_change(slope_bars)
    btc["macro_state"] = "flat"
    btc.loc[
        (btc["close"] > btc["btc_ema"]) & (btc["btc_ema_slope"] > 0), "macro_state"
    ] = "up"
    btc.loc[
        (btc["close"] < btc["btc_ema"]) & (btc["btc_ema_slope"] < 0), "macro_state"
    ] = "down"
    return btc.reset_index(names="timestamp")[
        ["timestamp", "close", "btc_ema", "btc_ema_slope", "macro_state"]
    ].rename(columns={"close": "btc_close"})


def _summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": len(g),
                "sum_r": g["r"].sum(),
                "mean_r": g["r"].mean(),
                "median_r": g["r"].median(),
                "win_rate": (g["r"] > 0).mean(),
                "tp_rate": (g["exit_reason"] == "tp").mean(),
                "sl_rate": (g["exit_reason"] == "sl").mean(),
                "timeout_rate": (g["exit_reason"] == "timeout").mean(),
                "long_n": (g["side"] == "LONG").sum(),
                "short_n": (g["side"] == "SHORT").sum(),
                "median_bars_held": g["bars_held"].median(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sum_r", ascending=False)


def run(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    cfg = StudyConfig(
        box_window=args.box_window,
        horizon_bars=args.horizon_bars,
        edge_frac=args.edge_frac,
        stop_buffer_frac=args.stop_buffer_frac,
        atr_stop_r=args.atr_stop_r,
        atr_tp_r=args.atr_tp_r,
        rsi_long_max=args.rsi_long_max,
        rsi_short_min=args.rsi_short_min,
        chop_min=args.chop_min,
        stability_min=args.stability_min,
        width_min=args.width_min,
        width_max=args.width_max,
        touches_min=args.touches_min,
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    data_dir = Path(args.data_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    macro = _macro_state_from_btc(
        data_dir,
        start=start,
        end=end,
        warmup_days=args.warmup_days,
        timeframe=args.timeframe,
        ema_span=args.macro_ema_span,
        slope_bars=args.macro_slope_bars,
    )

    all_features = []
    all_samples = []
    for symbol in symbols:
        raw = _load_symbol_1m(
            data_dir, symbol, start - pd.Timedelta(days=args.warmup_days), end
        )
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        bars = _resample_ohlcv(raw, args.timeframe)
        features = build_symbol_dataset(symbol, bars, cfg)
        features = features[(features.index >= start) & (features.index <= end)]
        samples = collect_execution_samples(features, cfg)
        all_features.append(features.reset_index(names="timestamp"))
        all_samples.append(samples)
        print(
            f"{symbol}: bars={len(features)} "
            f"box_prefilter={features['box_prefilter'].mean():.1%} "
            f"edge={(features['edge_side'] != '').mean():.1%} "
            f"samples={len(samples)}"
        )

    if not all_samples:
        raise SystemExit("No samples collected")
    samples_df = pd.concat(all_samples, ignore_index=True)
    samples_df["timestamp"] = pd.to_datetime(samples_df["timestamp"], utc=True)
    samples_df = samples_df.merge(macro, on="timestamp", how="left")
    samples_df["macro_alignment"] = "flat"
    samples_df.loc[
        ((samples_df["side"] == "LONG") & (samples_df["macro_state"] == "up"))
        | ((samples_df["side"] == "SHORT") & (samples_df["macro_state"] == "down")),
        "macro_alignment",
    ] = "aligned"
    samples_df.loc[
        ((samples_df["side"] == "LONG") & (samples_df["macro_state"] == "down"))
        | ((samples_df["side"] == "SHORT") & (samples_df["macro_state"] == "up")),
        "macro_alignment",
    ] = "counter"

    aligned = samples_df[samples_df["macro_alignment"] == "aligned"].copy()
    edge_chop = samples_df[samples_df["signal"] == "edge_chop"].copy()
    aligned_edge_chop = aligned[aligned["signal"] == "edge_chop"].copy()

    monthly = aligned_edge_chop.copy()
    monthly["month"] = monthly["timestamp"].dt.strftime("%Y-%m")
    yearly = aligned_edge_chop.copy()
    yearly["year"] = yearly["timestamp"].dt.year.astype(str)

    features_df = pd.concat(all_features, ignore_index=True)
    return {
        "samples": samples_df,
        "features": features_df,
        "macro_state": macro,
        "summary_by_signal_execution_alignment": _summarize(
            samples_df, ["signal", "execution", "macro_alignment"]
        ),
        "aligned_edge_chop_by_execution": _summarize(aligned_edge_chop, ["execution"]),
        "aligned_edge_chop_by_year": _summarize(yearly, ["year", "execution"]),
        "aligned_edge_chop_by_month": _summarize(monthly, ["month", "execution"]),
        "edge_chop_by_execution_alignment": _summarize(
            edge_chop, ["execution", "macro_alignment"]
        ),
    }


def _print_table(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context("display.max_columns", 40, "display.width", 220):
        print(df.head(max_rows).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/parquet_data")
    p.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--warmup-days", type=int, default=180)
    p.add_argument("--timeframe", default="2h")
    p.add_argument("--box-window", type=int, default=240, choices=[60, 120, 240])
    p.add_argument("--horizon-bars", type=int, default=60)
    p.add_argument("--edge-frac", type=float, default=0.12)
    p.add_argument("--stop-buffer-frac", type=float, default=0.25)
    p.add_argument("--atr-stop-r", type=float, default=1.5)
    p.add_argument("--atr-tp-r", type=float, default=1.2)
    p.add_argument("--rsi-long-max", type=float, default=40.0)
    p.add_argument("--rsi-short-min", type=float, default=60.0)
    p.add_argument("--chop-min", type=float, default=0.40)
    p.add_argument("--stability-min", type=float, default=0.90)
    p.add_argument("--width-min", type=float, default=0.04)
    p.add_argument("--width-max", type=float, default=0.25)
    p.add_argument("--touches-min", type=float, default=8.0)
    p.add_argument("--macro-ema-span", type=int, default=1200)
    p.add_argument("--macro-slope-bars", type=int, default=10)
    p.add_argument(
        "--out-dir",
        default="results/bad-candidates/box_pullback_trend/diagnostic",
    )
    args = p.parse_args()

    out = run(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "outputs": {k: str(out_dir / f"{k}.csv") for k in out},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_table(
        "Aligned Edge Chop by Execution", out["aligned_edge_chop_by_execution"]
    )
    _print_table(
        "Edge Chop by Execution + Macro Alignment",
        out["edge_chop_by_execution_alignment"],
    )
    _print_table(
        "Aligned Edge Chop by Year", out["aligned_edge_chop_by_year"], max_rows=50
    )
    print(f"\nSaved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
