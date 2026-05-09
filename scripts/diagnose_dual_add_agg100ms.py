"""Replay dual_add_trend on Binance aggTrades aggregated to 100ms.

This is a stricter execution sanity check than 1min OHLC replay. Signals and
segments are still computed on the signal timeframe (default 2h); execution
starts only after the signal bar's right edge, then uses 100ms OHLC built from
aggTrades inside active segments.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    _hysteresis_segments,
    build_features,
    regime_chop_series,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.diagnose_dual_add_trend import (  # noqa: E402
    DEFAULT_DUAL_ADD_CONFIG,
    DualAddConfig,
    _add_trend_features,
    _load_dual_add_defaults,
    simulate_dual_add_segment,
    summarize,
)
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start.year, start.month, 1, tz=start.tz)
    last = pd.Timestamp(end.year, end.month, 1, tz=end.tz)
    while cur <= last:
        yield cur
        cur = cur + pd.DateOffset(months=1)


def _build_cfg(args: argparse.Namespace) -> DualAddConfig:
    hparts = [
        x.strip() for x in str(args.trend_return_horizons).split(",") if x.strip()
    ]
    trend_horizons = tuple(int(x) for x in hparts) if hparts else (3, 5, 10)
    return DualAddConfig(
        regime="trend",
        add_mode=args.add_mode,
        flip_action=args.flip_action,
        chop_signal=args.chop_signal,
        chop_ts_window=args.chop_ts_window,
        chop_ts_min_periods=args.chop_ts_min_periods,
        compute_semantic_chop_ts_q=args.compute_chop_ts_q,
        trend_return_horizons=trend_horizons,
        stability_min=args.stability_min,
        width_min=args.width_min,
        width_max=args.width_max,
        touches_min=args.touches_min,
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        trend_min=args.trend_min,
        trend_exit_min=args.trend_exit_min,
        box_window=args.box_window,
        step_atr_mult=args.step_atr_mult,
        take_profit_mode=args.take_profit_mode,
        tp_atr_mult=args.tp_atr_mult,
        tp_abs=args.tp_abs,
        tp_pct=args.tp_pct,
        max_adds_per_side=args.max_adds_per_side,
        max_net_exposure=args.max_net_exposure,
        max_gross_exposure=args.max_gross_exposure,
        max_loser_hold_bars=args.max_loser_hold_bars,
        max_segment_bars=args.max_segment_bars,
        min_segment_bars=args.min_segment_bars,
        fee_bps=args.fee_bps_list[0],
        max_loss_per_segment=args.max_loss_per_segment,
        risk_stop_mode=args.risk_stop_mode,
        initial_hedge=args.initial_hedge,
    )


def _build_signal_segments(
    *,
    symbol: str,
    args: argparse.Namespace,
    cfg: DualAddConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    grid_cfg = GridConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        exit_chop_min=cfg.exit_chop_min,
        chop_signal=cfg.chop_signal,
        chop_ts_window=cfg.chop_ts_window,
        chop_ts_min_periods=cfg.chop_ts_min_periods,
        compute_semantic_chop_ts_q=cfg.compute_semantic_chop_ts_q,
        stability_min=cfg.stability_min,
        width_min=cfg.width_min,
        width_max=cfg.width_max,
        touches_min=cfg.touches_min,
    )
    raw = _load_symbol_1m(
        Path(args.data_dir), symbol, start - pd.Timedelta(days=args.warmup_days), end
    )
    if raw.empty:
        return pd.DataFrame(), []
    signal_bars = _resample_ohlcv(raw, args.timeframe)
    df = build_features(symbol, signal_bars, grid_cfg, bars_timeframe=args.timeframe)
    df = _add_trend_features(df, cfg.trend_return_horizons)
    df = df[(df.index >= start) & (df.index <= end)].copy()
    if df.empty:
        return df, []

    chop_s = regime_chop_series(df, grid_cfg)
    entry = (df["trend_confidence"] >= cfg.trend_min) & (chop_s <= cfg.exit_chop_min)
    hold = (df["trend_confidence"] >= cfg.trend_exit_min) & (chop_s <= cfg.chop_min)
    if args.exclude_box:
        entry = entry & (~df["box_prefilter"])
        hold = hold & (~df["box_prefilter"])
    raw_segments = _hysteresis_segments(
        entry, hold, min_len=cfg.min_segment_bars, max_len=cfg.max_segment_bars
    )
    sig_delta = timeframe_to_timedelta(args.timeframe)
    windows: list[dict] = []
    for seq, (s, e) in enumerate(raw_segments, start=1):
        windows.append(
            {
                "symbol": symbol,
                "seq": seq,
                "s": s,
                "e": e,
                "start": pd.Timestamp(df.index[s]) + sig_delta,
                "end": pd.Timestamp(df.index[e]) + sig_delta,
                "segment_id": f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H%M')}",
                "direction": str(df["trend_direction"].iloc[s]),
                "center": float(df["close"].iloc[s]),
                "atr": float(df["atr14"].iloc[s]),
                "parts": [],
            }
        )
    return df, windows


def _load_segment_100ms(
    *,
    symbol: str,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    windows: list[dict],
) -> None:
    if not windows:
        return
    for month in _month_starts(start, end):
        zip_path = (
            Path(args.agg_data_dir)
            / f"{symbol}-aggTrades-{month.strftime('%Y-%m')}.zip"
        )
        if not zip_path.exists():
            print(f"[warn] missing {zip_path}", flush=True)
            continue
        print(f"[read] {zip_path}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                for i, chunk in enumerate(
                    pd.read_csv(
                        fh,
                        usecols=["price", "quantity", "transact_time"],
                        chunksize=args.chunksize,
                    ),
                    start=1,
                ):
                    ts = pd.to_datetime(chunk["transact_time"], unit="ms", utc=True)
                    for w in windows:
                        mask = (ts >= w["start"]) & (ts < w["end"])
                        if not mask.any():
                            continue
                        c = chunk.loc[mask, ["price", "quantity"]].copy()
                        c.index = ts[mask].dt.floor("100ms")
                        bars = c.groupby(level=0).agg(
                            open=("price", "first"),
                            high=("price", "max"),
                            low=("price", "min"),
                            close=("price", "last"),
                            volume=("quantity", "sum"),
                        )
                        w["parts"].append(bars)
                    if i % 20 == 0:
                        print(
                            f"[progress] {symbol} {month.strftime('%Y-%m')} chunks={i}",
                            flush=True,
                        )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    cfg_base = _build_cfg(args)
    sig_delta = timeframe_to_timedelta(args.timeframe)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    segment_payloads: list[tuple[pd.DataFrame, dict, pd.DataFrame]] = []
    inventory: list[dict] = []
    for symbol in symbols:
        df_signal, windows = _build_signal_segments(
            symbol=symbol, args=args, cfg=cfg_base, start=start, end=end
        )
        print(f"[segments] {symbol} count={len(windows)}", flush=True)
        _load_segment_100ms(
            symbol=symbol, args=args, start=start, end=end, windows=windows
        )
        for w in windows:
            if not w["parts"]:
                inventory.append(
                    {**{k: v for k, v in w.items() if k != "parts"}, "bars": 0}
                )
                continue
            bars = (
                pd.concat(w["parts"])
                .sort_index()
                .groupby(level=0)
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
            )
            bars = merge_signal_features_onto_execution_bars(
                bars, df_signal, signal_bar_delta=sig_delta
            )
            segment_payloads.append((bars, w, df_signal))
            inventory.append(
                {**{k: v for k, v in w.items() if k != "parts"}, "bars": len(bars)}
            )

    summaries = []
    for fee_bps in args.fee_bps_list:
        cfg = replace(cfg_base, fee_bps=float(fee_bps))
        all_trades: list[dict] = []
        all_segments: list[dict] = []
        for bars, w, _df_signal in segment_payloads:
            trades, seg_summary = simulate_dual_add_segment(
                bars,
                cfg=cfg,
                symbol=w["symbol"],
                segment_id=w["segment_id"],
                direction=w["direction"],
                frozen_center=w["center"],
                frozen_atr=w["atr"],
            )
            all_trades.extend(trades)
            if seg_summary:
                seg_summary["agg100ms_bars"] = len(bars)
                all_segments.append(seg_summary)
        trades_df = pd.DataFrame(all_trades)
        segments_df = pd.DataFrame(all_segments)
        summary_df = summarize(trades_df, segments_df)
        run_name = f"fee{float(fee_bps):g}"
        trades_df.to_csv(out_dir / f"dual_add_trades_{run_name}.csv", index=False)
        segments_df.to_csv(out_dir / f"dual_add_segments_{run_name}.csv", index=False)
        summary_df.to_csv(out_dir / f"summary_{run_name}.csv", index=False)
        if not summary_df.empty:
            row = summary_df.iloc[0].to_dict()
            row["fee_bps"] = float(fee_bps)
            summaries.append(row)
            print(
                f"[summary] fee_bps={fee_bps} return_pct={row.get('return_pct')}",
                flush=True,
            )

    pd.DataFrame(summaries).to_csv(out_dir / "summary_by_fee.csv", index=False)
    pd.DataFrame(inventory).to_csv(out_dir / "segment_inventory.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    config_path = DEFAULT_DUAL_ADD_CONFIG
    defaults = _load_dual_add_defaults(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--agg-data-dir", default="data/agg_data")
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-03-31")
    ap.add_argument("--warmup-days", type=int, default=120)
    ap.add_argument("--timeframe", default="2h")
    ap.add_argument("--fee-bps-list", type=float, nargs="+", default=[8.0, 12.0, 20.0])
    ap.add_argument(
        "--add-mode",
        choices=["both", "trend"],
        default=defaults.get("add_mode", "trend"),
    )
    ap.add_argument(
        "--flip-action",
        choices=["keep", "close_offside_adds", "close_offside_all"],
        default=defaults.get("flip_action", "close_offside_all"),
    )
    ap.add_argument(
        "--take-profit-mode",
        choices=["basket", "per_leg"],
        default=defaults.get("take_profit_mode", "basket"),
    )
    ap.add_argument(
        "--initial-hedge",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("initial_hedge", True),
    )
    ap.add_argument(
        "--step-atr-mult", type=float, default=defaults.get("step_atr_mult", 0.50)
    )
    ap.add_argument(
        "--tp-atr-mult", type=float, default=defaults.get("tp_atr_mult", 0.25)
    )
    ap.add_argument("--tp-pct", type=float, default=defaults.get("tp_pct", 0.0005))
    ap.add_argument("--tp-abs", type=float, default=defaults.get("tp_abs", 0.0))
    ap.add_argument("--trend-min", type=float, default=defaults.get("trend_min", 0.80))
    ap.add_argument(
        "--trend-exit-min", type=float, default=defaults.get("trend_exit_min", 0.50)
    )
    ap.add_argument("--chop-min", type=float, default=defaults.get("chop_min", 0.40))
    ap.add_argument(
        "--exit-chop-min", type=float, default=defaults.get("exit_chop_min", 0.25)
    )
    ap.add_argument(
        "--chop-signal",
        choices=["raw", "ts_quantile"],
        default=str(defaults.get("chop_signal", "raw")),
    )
    ap.add_argument(
        "--chop-ts-window", type=int, default=int(defaults.get("chop_ts_window", 1200))
    )
    ap.add_argument(
        "--chop-ts-min-periods",
        type=int,
        default=int(defaults.get("chop_ts_min_periods", 150)),
    )
    ap.add_argument(
        "--compute-chop-ts-q",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("compute_chop_ts_q"),
    )
    ap.add_argument(
        "--trend-return-horizons",
        default=",".join(
            str(x) for x in defaults.get("trend_return_horizons", (3, 5, 10))
        ),
    )
    ap.add_argument("--box-window", type=int, default=defaults.get("box_window", 120))
    ap.add_argument(
        "--stability-min", type=float, default=defaults.get("stability_min", 0.85)
    )
    ap.add_argument("--width-min", type=float, default=defaults.get("width_min", 0.04))
    ap.add_argument("--width-max", type=float, default=defaults.get("width_max", 0.30))
    ap.add_argument("--touches-min", type=int, default=defaults.get("touches_min", 5))
    ap.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    ap.add_argument(
        "--max-adds-per-side", type=int, default=defaults.get("max_adds_per_side", 3)
    )
    ap.add_argument(
        "--max-net-exposure", type=int, default=defaults.get("max_net_exposure", 2)
    )
    ap.add_argument(
        "--max-gross-exposure", type=int, default=defaults.get("max_gross_exposure", 4)
    )
    ap.add_argument("--max-loser-hold-bars", type=int, default=1_728_000)
    ap.add_argument(
        "--max-loss-per-segment",
        type=float,
        default=defaults.get("max_loss_per_segment", 0.01),
    )
    ap.add_argument(
        "--risk-stop-mode",
        choices=["mtm", "regime_only"],
        default=defaults.get("risk_stop_mode", "mtm"),
    )
    ap.add_argument(
        "--min-segment-bars", type=int, default=defaults.get("min_segment_bars", 6)
    )
    ap.add_argument(
        "--max-segment-bars", type=int, default=defaults.get("max_segment_bars", 120)
    )
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--out-dir", default="results/dual_add_trend/agg100ms_check")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
