"""Offline diagnostic for chop-regime grid trading.

Hypothesis:
  Stable boxes are rare and hard to identify causally in crypto. A broader
  "no clear trend / semantic chop" regime may be better traded with a small
  neutral grid, exiting all inventory once chop ends.

This script compares where chop regimes occur and simulates a conservative
fixed-level grid inside contiguous chop segments.

Example:
    python scripts/diagnose_chop_grid.py \\
        --start 2024-01-01 --end 2024-12-31 \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_crf_edge import (  # noqa: E402
    StudyConfig,
    _atr,
    _bb_width_pctile,
    _load_symbol_1m,
    _resample_ohlcv,
    _semantic_chop,
    build_symbol_dataset,
)


@dataclass(frozen=True)
class GridConfig:
    box_window: int = 120
    chop_min: float = 0.40
    exit_chop_min: float = 0.25
    min_segment_bars: int = 6
    max_segment_bars: int = 120
    grid_atr_mult: float = 0.75
    grid_pct: float = 0.004
    max_levels: int = 3
    fee_bps: float = 4.0


def _segments(mask: pd.Series, *, min_len: int, max_len: int) -> List[Tuple[int, int]]:
    return _hysteresis_segments(mask, mask, min_len=min_len, max_len=max_len)


def _hysteresis_segments(
    entry_mask: pd.Series,
    hold_mask: pd.Series,
    *,
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int]]:
    """Build segments that enter on entry_mask and exit once hold_mask fails."""
    entry = entry_mask.fillna(False).to_numpy(dtype=bool)
    hold = hold_mask.fillna(False).to_numpy(dtype=bool)
    segs: List[Tuple[int, int]] = []
    i = 0
    n = len(entry)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and hold[i + 1] and (i + 1 - start) < max_len:
            i += 1
        end = i
        if end - start + 1 >= min_len:
            segs.append((start, end))
        i += 1
    return segs


def _pnl_long(entry: float, exit_px: float, fee: float) -> float:
    return (exit_px - entry) / entry - 2.0 * fee


def _pnl_short(entry: float, exit_px: float, fee: float) -> float:
    return (entry - exit_px) / entry - 2.0 * fee


def simulate_fixed_grid(
    seg: pd.DataFrame,
    *,
    cfg: GridConfig,
) -> Dict[str, float | int | str | pd.Timestamp]:
    """Conservative fixed-level neutral grid over one chop segment.

    Levels are fixed from segment start:
      - buy levels below center; each long takes profit one spacing higher
      - sell levels above center; each short takes profit one spacing lower

    If a level fills, its target is only eligible from the next bar onward. This
    avoids over-crediting same-bar entry+exit when OHLC ordering is unknown.
    At segment exit, all open inventory is marked to the final close.
    """
    if seg.empty:
        return {"status": "empty"}
    center = float(seg["close"].iloc[0])
    atr = float(seg["atr14"].iloc[0])
    if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
        return {"status": "invalid"}

    spacing = max(cfg.grid_atr_mult * atr, cfg.grid_pct * center)
    if spacing <= 0:
        return {"status": "invalid"}
    fee = cfg.fee_bps / 10000.0

    long_levels = [center - spacing * k for k in range(1, cfg.max_levels + 1)]
    short_levels = [center + spacing * k for k in range(1, cfg.max_levels + 1)]
    open_longs: Dict[int, Tuple[float, int]] = {}
    open_shorts: Dict[int, Tuple[float, int]] = {}
    realized = 0.0
    fills = 0
    cycles = 0
    max_open = 0
    pnl_path = []

    for bar_i, (_, row) in enumerate(seg.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Close positions opened on prior bars only.
        for level_i, (entry, fill_bar) in list(open_longs.items()):
            target = entry + spacing
            if bar_i > fill_bar and high >= target:
                realized += _pnl_long(entry, target, fee)
                cycles += 1
                del open_longs[level_i]
        for level_i, (entry, fill_bar) in list(open_shorts.items()):
            target = entry - spacing
            if bar_i > fill_bar and low <= target:
                realized += _pnl_short(entry, target, fee)
                cycles += 1
                del open_shorts[level_i]

        # Fill inactive levels.
        for level_i, px in enumerate(long_levels):
            if level_i not in open_longs and low <= px:
                open_longs[level_i] = (px, bar_i)
                fills += 1
        for level_i, px in enumerate(short_levels):
            if level_i not in open_shorts and high >= px:
                open_shorts[level_i] = (px, bar_i)
                fills += 1

        mtm = realized
        for entry, _ in open_longs.values():
            mtm += (close - entry) / entry - fee
        for entry, _ in open_shorts.values():
            mtm += (entry - close) / entry - fee
        pnl_path.append(mtm)
        max_open = max(max_open, len(open_longs) + len(open_shorts))

    exit_close = float(seg["close"].iloc[-1])
    forced = len(open_longs) + len(open_shorts)
    for entry, _ in open_longs.values():
        realized += _pnl_long(entry, exit_close, fee)
    for entry, _ in open_shorts.values():
        realized += _pnl_short(entry, exit_close, fee)

    capital_units = max(1, 2 * cfg.max_levels)
    pnl_per_capital = realized / capital_units
    max_drawdown = 0.0
    if pnl_path:
        arr = np.asarray(pnl_path, dtype=float) / capital_units
        running_max = np.maximum.accumulate(arr)
        max_drawdown = float((arr - running_max).min())

    return {
        "status": "ok",
        "start": seg.index[0],
        "end": seg.index[-1],
        "bars": len(seg),
        "center": center,
        "spacing_pct": spacing / center,
        "spacing_atr": spacing / atr,
        "fills": fills,
        "cycles": cycles,
        "forced_exits": forced,
        "max_open_levels": max_open,
        "gross_unit_pnl": realized,
        "pnl_per_capital": pnl_per_capital,
        "max_drawdown": max_drawdown,
    }


def build_features(symbol: str, bars: pd.DataFrame, cfg: GridConfig) -> pd.DataFrame:
    study_cfg = StudyConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        stability_min=0.85,
        width_min=0.04,
        width_max=0.30,
        touches_min=5,
    )
    # Reuse CRF diagnostic features so box/chop definitions stay comparable.
    df = build_symbol_dataset(symbol, bars, study_cfg)
    df["atr14"] = _atr(df, 14)
    if "semantic_chop" not in df:
        df["bb_width_pctile"] = _bb_width_pctile(df["close"])
        df["semantic_chop"] = _semantic_chop(df["close"], df["bb_width_pctile"])
    return df


def run_one_period(
    *,
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: Path,
    timeframe: str,
    warmup_days: int,
    cfg: GridConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows = []
    regime_rows = []
    warmup_start = start - pd.Timedelta(days=warmup_days)

    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        bars = _resample_ohlcv(raw, timeframe)
        if bars.empty:
            print(f"skip {symbol}: no bars")
            continue
        df = build_features(symbol, bars, cfg)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue

        chop = df["semantic_chop"] >= cfg.chop_min
        chop_hold = df["semantic_chop"] >= cfg.exit_chop_min
        box = df["box_prefilter"]
        chop_not_box = chop & ~box
        regimes = {
            "semantic_chop": (chop, chop_hold),
            "box_prefilter": (box, box),
            "chop_not_box": (chop_not_box, chop_hold & ~box),
        }

        for regime_name, (entry_mask, hold_mask) in regimes.items():
            segs = _hysteresis_segments(
                entry_mask,
                hold_mask,
                min_len=cfg.min_segment_bars,
                max_len=cfg.max_segment_bars,
            )
            regime_rows.append(
                {
                    "symbol": symbol,
                    "regime": regime_name,
                    "bars": int(entry_mask.sum()),
                    "bar_rate": float(entry_mask.mean()),
                    "hold_bars": int(hold_mask.sum()),
                    "hold_bar_rate": float(hold_mask.mean()),
                    "segments": len(segs),
                    "median_segment_bars": (
                        float(np.median([e - s + 1 for s, e in segs])) if segs else 0.0
                    ),
                }
            )
            for s, e in segs:
                seg = df.iloc[s : e + 1]
                sim = simulate_fixed_grid(seg, cfg=cfg)
                if sim.get("status") != "ok":
                    continue
                segment_rows.append(
                    {
                        "symbol": symbol,
                        "regime": regime_name,
                        "entry_chop": float(seg["semantic_chop"].iloc[0]),
                        "median_chop": float(seg["semantic_chop"].median()),
                        "entry_box_stability": float(seg["box_stability"].iloc[0]),
                        "entry_box_width_pct": float(seg["box_width_pct"].iloc[0]),
                        **sim,
                    }
                )
        print(
            f"{symbol}: chop={chop.mean():.1%}, box={box.mean():.1%}, "
            f"chop_not_box={chop_not_box.mean():.1%}"
        )

    return pd.DataFrame(segment_rows), pd.DataFrame(regime_rows)


def summarize_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    rows = []
    for (regime,), g in segments.groupby(["regime"], sort=True):
        rows.append(
            {
                "regime": regime,
                "segments": len(g),
                "win_rate": (g["pnl_per_capital"] > 0).mean(),
                "sum_pnl_per_capital": g["pnl_per_capital"].sum(),
                "mean_pnl_per_capital": g["pnl_per_capital"].mean(),
                "median_pnl_per_capital": g["pnl_per_capital"].median(),
                "p25_pnl": g["pnl_per_capital"].quantile(0.25),
                "p75_pnl": g["pnl_per_capital"].quantile(0.75),
                "median_bars": g["bars"].median(),
                "median_spacing_pct": g["spacing_pct"].median(),
                "median_cycles": g["cycles"].median(),
                "median_forced_exits": g["forced_exits"].median(),
                "worst_segment": g["pnl_per_capital"].min(),
                "median_max_drawdown": g["max_drawdown"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_pnl_per_capital", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/parquet_data")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--timeframe", default="2h")
    parser.add_argument("--box-window", type=int, default=120, choices=[60, 120, 240])
    parser.add_argument("--chop-min", type=float, default=0.40)
    parser.add_argument("--exit-chop-min", type=float, default=0.25)
    parser.add_argument("--min-segment-bars", type=int, default=6)
    parser.add_argument("--max-segment-bars", type=int, default=120)
    parser.add_argument("--grid-atr-mult", type=float, default=0.75)
    parser.add_argument("--grid-pct", type=float, default=0.004)
    parser.add_argument("--max-levels", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--out-dir", default="results/chop_grid_diagnostic")
    args = parser.parse_args()

    cfg = GridConfig(
        box_window=args.box_window,
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        min_segment_bars=args.min_segment_bars,
        max_segment_bars=args.max_segment_bars,
        grid_atr_mult=args.grid_atr_mult,
        grid_pct=args.grid_pct,
        max_levels=args.max_levels,
        fee_bps=args.fee_bps,
    )
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    segments, regimes = run_one_period(
        symbols=symbols,
        start=start,
        end=end,
        data_dir=Path(args.data_dir),
        timeframe=args.timeframe,
        warmup_days=args.warmup_days,
        cfg=cfg,
    )
    summary = summarize_segments(segments)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(out_dir / "grid_segments.csv", index=False)
    regimes.to_csv(out_dir / "regime_coverage.csv", index=False)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"args": vars(args)}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Regime Coverage ===")
    if regimes.empty:
        print("(empty)")
    else:
        print(
            regimes.groupby("regime")
            .agg(
                bars=("bars", "sum"),
                mean_bar_rate=("bar_rate", "mean"),
                hold_bars=("hold_bars", "sum"),
                mean_hold_bar_rate=("hold_bar_rate", "mean"),
                segments=("segments", "sum"),
                median_segment_bars=("median_segment_bars", "median"),
            )
            .reset_index()
            .to_string(index=False)
        )
    print("\n=== Grid Summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(empty)")
    print(f"\nSaved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
