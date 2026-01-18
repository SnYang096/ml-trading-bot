#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ENTRY_COL_CANDIDATES = [
    "Entry Timestamp",
    "entry_timestamp",
    "entry_time",
    "entry_ts",
    "entry_dt",
    "entry",
]
EXIT_COL_CANDIDATES = [
    "Exit Timestamp",
    "exit_timestamp",
    "exit_time",
    "exit_ts",
    "exit_dt",
    "exit",
]
SYMBOL_COL_CANDIDATES = ["Symbol", "symbol", "ticker", "asset"]


def _load_trades(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "trades" in raw:
            raw = raw["trades"]
        return pd.json_normalize(raw)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported trades file type: {path}")


def _pick_col(
    df: pd.DataFrame, explicit: Optional[str], candidates: Iterable[str]
) -> Optional[str]:
    if explicit:
        return explicit if explicit in df.columns else None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_ts(val) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    ts = pd.to_datetime(val, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.tz_convert(None)


def _merge_intervals(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], gap_sec: int = 0
):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + pd.Timedelta(seconds=gap_sec):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_windows(
    trades: pd.DataFrame,
    *,
    entry_col: str,
    exit_col: Optional[str],
    symbol_col: Optional[str],
    default_symbol: Optional[str],
    pre_minutes: int,
    post_minutes: int,
) -> List[Dict[str, str]]:
    pre = pd.Timedelta(minutes=int(pre_minutes))
    post = pd.Timedelta(minutes=int(post_minutes))
    windows: List[Dict[str, str]] = []
    for _, row in trades.iterrows():
        entry_ts = _to_ts(row.get(entry_col))
        if entry_ts is None:
            continue
        exit_ts = _to_ts(row.get(exit_col)) if exit_col else None
        start = entry_ts - pre
        end = (exit_ts + post) if exit_ts is not None else (entry_ts + post)
        sym = None
        if symbol_col and symbol_col in row and pd.notna(row[symbol_col]):
            sym = str(row[symbol_col])
        elif default_symbol:
            sym = str(default_symbol)
        windows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbol": sym,
                "source": "vectorbt_trade",
                "entry_ts": entry_ts.isoformat(),
                "exit_ts": exit_ts.isoformat() if exit_ts is not None else None,
            }
        )
    return windows


def _expand_timeline_windows(
    timeline: pd.DataFrame,
    *,
    ts_col: str,
    symbol_col: Optional[str],
    n_windows: int,
    pre_minutes: int,
    post_minutes: int,
    rng: np.random.Generator,
    deny_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
) -> List[Dict[str, str]]:
    if ts_col not in timeline.columns:
        raise ValueError(f"timeline missing ts_col: {ts_col}")
    tdf = timeline.copy()
    tdf[ts_col] = pd.to_datetime(tdf[ts_col], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    tdf = tdf.dropna(subset=[ts_col])
    if symbol_col and symbol_col in tdf.columns:
        groups = list(tdf.groupby(symbol_col))
    else:
        groups = [(None, tdf)]
    pre = pd.Timedelta(minutes=int(pre_minutes))
    post = pd.Timedelta(minutes=int(post_minutes))
    windows: List[Dict[str, str]] = []
    max_tries = max(n_windows * 5, 100)
    tries = 0
    while len(windows) < n_windows and tries < max_tries:
        tries += 1
        sym, g = groups[rng.integers(0, len(groups))]
        if g.empty:
            continue
        row = g.iloc[int(rng.integers(0, len(g)))]
        ts = _to_ts(row.get(ts_col))
        if ts is None:
            continue
        start = ts - pre
        end = ts + post
        deny_list = deny_intervals.get(
            str(sym) if sym is not None else "__GLOBAL__", []
        )
        overlap = False
        for d_start, d_end in deny_list:
            if start <= d_end and end >= d_start:
                overlap = True
                break
        if overlap:
            continue
        windows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbol": str(sym) if sym is not None else None,
                "source": "negative_sample",
            }
        )
    return windows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build time-window JSON for Nautilus backtest from vectorbt trades."
    )
    ap.add_argument("--trades", required=True, help="Trades file (json/csv/parquet)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--entry-col", default=None)
    ap.add_argument("--exit-col", default=None)
    ap.add_argument("--symbol-col", default=None)
    ap.add_argument("--default-symbol", default=None)
    ap.add_argument("--pre-minutes", type=int, default=480)
    ap.add_argument("--post-minutes", type=int, default=480)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--merge-overlap", action="store_true")
    ap.add_argument("--merge-gap-minutes", type=int, default=0)
    ap.add_argument("--negative-ratio", type=float, default=0.0)
    ap.add_argument("--timeline-parquet", default=None)
    ap.add_argument("--timeline-ts-col", default="timestamp")
    ap.add_argument("--timeline-symbol-col", default="symbol")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    trades_path = Path(args.trades)
    trades = _load_trades(trades_path)

    entry_col = _pick_col(trades, args.entry_col, ENTRY_COL_CANDIDATES)
    if not entry_col:
        raise ValueError(
            f"Entry timestamp column not found in trades: {trades.columns}"
        )
    exit_col = _pick_col(trades, args.exit_col, EXIT_COL_CANDIDATES)
    symbol_col = _pick_col(trades, args.symbol_col, SYMBOL_COL_CANDIDATES)

    windows = _build_windows(
        trades,
        entry_col=entry_col,
        exit_col=exit_col,
        symbol_col=symbol_col,
        default_symbol=args.default_symbol,
        pre_minutes=args.pre_minutes,
        post_minutes=args.post_minutes,
    )

    if args.max_windows is not None and args.max_windows > 0:
        windows = windows[: int(args.max_windows)]

    if args.merge_overlap:
        merged_windows: List[Dict[str, str]] = []
        by_symbol: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for w in windows:
            sym = str(w.get("symbol") or "__GLOBAL__")
            start = _to_ts(w.get("start"))
            end = _to_ts(w.get("end"))
            if start is None or end is None:
                continue
            by_symbol.setdefault(sym, []).append((start, end))
        for sym, intervals in by_symbol.items():
            merged = _merge_intervals(
                intervals, gap_sec=int(args.merge_gap_minutes) * 60
            )
            for start, end in merged:
                merged_windows.append(
                    {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "symbol": None if sym == "__GLOBAL__" else sym,
                        "source": "merged",
                    }
                )
        windows = merged_windows

    if args.negative_ratio and args.negative_ratio > 0:
        if not args.timeline_parquet:
            raise ValueError("--timeline-parquet is required for negative sampling")
        timeline = pd.read_parquet(args.timeline_parquet)
        deny_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for w in windows:
            sym = str(w.get("symbol") or "__GLOBAL__")
            start = _to_ts(w.get("start"))
            end = _to_ts(w.get("end"))
            if start is None or end is None:
                continue
            deny_intervals.setdefault(sym, []).append((start, end))
        rng = np.random.default_rng(int(args.seed))
        n_neg = int(max(0, round(len(windows) * float(args.negative_ratio))))
        neg_windows = _expand_timeline_windows(
            timeline,
            ts_col=str(args.timeline_ts_col),
            symbol_col=(
                str(args.timeline_symbol_col) if args.timeline_symbol_col else None
            ),
            n_windows=n_neg,
            pre_minutes=args.pre_minutes,
            post_minutes=args.post_minutes,
            rng=rng,
            deny_intervals=deny_intervals,
        )
        windows.extend(neg_windows)

    out = {
        "windows": windows,
        "meta": {
            "trades_path": str(trades_path),
            "entry_col": entry_col,
            "exit_col": exit_col,
            "symbol_col": symbol_col,
            "default_symbol": args.default_symbol,
            "pre_minutes": int(args.pre_minutes),
            "post_minutes": int(args.post_minutes),
            "merge_overlap": bool(args.merge_overlap),
            "negative_ratio": float(args.negative_ratio),
            "total_windows": int(len(windows)),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(windows)} windows to {out_path}")


if __name__ == "__main__":
    main()
