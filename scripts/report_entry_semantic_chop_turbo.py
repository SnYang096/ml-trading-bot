#!/usr/bin/env python3
"""
Join turbo rolling ``event_trades_*.csv`` parent entries to semantic_chop / *_ts_q
at the entry bar (120T), then summarize distributions and PnL-by-bin.

Uses the same causal chop + rolling quantile definitions as research CLIs:
  - TPC/BPC: ``build_symbol_dataset`` semantic_chop + ``semantic_chop_ts_quantile``
  - ME: ``compute_momentum_expansion_soft_phase_from_series`` with
        ``bb_width_normalized`` from ``_bb_width_pctile(close)`` (matches narrow-band intent).

Example (auto-pick two newest runs per strategy under results/<strat>/calibrate_roll.default/_rolling_sim):

  python scripts/report_entry_semantic_chop_turbo.py \\
    --data-path data/parquet_data \\
    --out-csv results/_tmp_entry_chop_stats.csv

Or pin runs explicitly:

  python scripts/report_entry_semantic_chop_turbo.py \\
    --tpc-runs .../20260429_053105 .../20260429_193223 \\
    --me-runs .../20260429_090245 .../20260430_011929 \\
    --bpc-runs .../20260428_212439 .../20260429_144352
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
)
from src.features.time_series.momentum_expansion_features import (  # noqa: E402
    compute_momentum_expansion_soft_phase_from_series,
)
from src.features.time_series.semantic_chop_ts_quantile import (  # noqa: E402
    DEFAULT_CHOP_TS_MIN_PERIODS,
    DEFAULT_CHOP_TS_WINDOW,
    semantic_chop_ts_quantile,
)


def _discover_latest_runs(strat: str, root: Path, k: int = 2) -> List[Path]:
    base = root / "results" / strat / "calibrate_roll.default" / "_rolling_sim"
    if not base.is_dir():
        return []
    cands = [
        p
        for p in base.iterdir()
        if p.is_dir() and (p / "stitched_summary.json").exists()
    ]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[:k]


def _month_range_from_token(token: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # fast_month_2024-04
    y, m = token.replace("fast_month_", "").split("-")
    start = pd.Timestamp(year=int(y), month=int(m), day=1, tz="UTC")
    end = start + pd.offsets.MonthEnd(0)
    end = end + pd.Timedelta(days=1)
    return start, end


def _bpc_tpc_feature_frame(
    symbol: str,
    month_token: str,
    data_dir: Path,
    study_cfg: StudyConfig,
    timeframe: str,
    warmup_days: int,
) -> Optional[pd.DataFrame]:
    start, end = _month_range_from_token(month_token)
    warm = start - pd.Timedelta(days=warmup_days)
    raw = _load_symbol_1m(data_dir, symbol, warm, end)
    if raw.empty:
        return None
    bars = _resample_ohlcv(raw, timeframe)
    if bars.empty:
        return None
    feat = build_symbol_dataset(symbol, bars, study_cfg)
    chop = feat["semantic_chop"].to_numpy(dtype=float, copy=False)
    tsq = semantic_chop_ts_quantile(
        chop,
        feat.index,
        window=DEFAULT_CHOP_TS_WINDOW,
        min_periods=DEFAULT_CHOP_TS_MIN_PERIODS,
    )
    tsq_s = pd.Series(tsq, index=feat.index)
    out = feat.loc[(feat.index >= start) & (feat.index < end), ["semantic_chop"]].copy()
    out["semantic_chop_ts_q"] = tsq_s.reindex(out.index).to_numpy(dtype=float)
    return out


def _me_feature_frame(
    symbol: str,
    month_token: str,
    data_dir: Path,
    timeframe: str,
    warmup_days: int,
) -> Optional[pd.DataFrame]:
    start, end = _month_range_from_token(month_token)
    warm = start - pd.Timedelta(days=warmup_days)
    raw = _load_symbol_1m(data_dir, symbol, warm, end)
    if raw.empty:
        return None
    bars = _resample_ohlcv(raw, timeframe)
    if bars.empty:
        return None
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    vol = bars["volume"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    from scripts.diagnose_crf_edge import _bb_width_pctile  # noqa: E402

    bb_norm = _bb_width_pctile(close)
    me_df = compute_momentum_expansion_soft_phase_from_series(
        close=close,
        high=high,
        low=low,
        volume=vol,
        atr=atr,
        cvd_change_5=None,
        delta=None,
        bb_width_normalized=bb_norm,
    )
    chop = me_df["me_semantic_chop"].to_numpy(dtype=float, copy=False)
    tsq = semantic_chop_ts_quantile(
        chop,
        me_df.index,
        window=DEFAULT_CHOP_TS_WINDOW,
        min_periods=DEFAULT_CHOP_TS_MIN_PERIODS,
    )
    tsq_s = pd.Series(tsq, index=me_df.index)
    sub = me_df.loc[(me_df.index >= start) & (me_df.index < end)].copy()
    sub["semantic_chop_ts_q"] = tsq_s.reindex(sub.index).to_numpy(dtype=float)
    sub.rename(columns={"me_semantic_chop": "semantic_chop"}, inplace=True)
    return sub[["semantic_chop", "semantic_chop_ts_q"]]


def _lookup_bar(
    feat: pd.DataFrame, entry_ts: pd.Timestamp
) -> Optional[Tuple[float, float]]:
    if feat is None or feat.empty:
        return None
    idx = feat.index.searchsorted(entry_ts, side="right") - 1
    if idx < 0:
        return None
    row = feat.iloc[idx]
    return float(row["semantic_chop"]), float(row["semantic_chop_ts_q"])


def _bins_ts_q() -> List[Tuple[float, float, str]]:
    return [
        (0.0, 0.25, "[0,0.25)"),
        (0.25, 0.40, "[0.25,0.40)"),
        (0.40, 0.55, "[0.40,0.55)"),
        (0.55, 0.70, "[0.55,0.70)"),
        (0.70, 0.85, "[0.70,0.85)"),
        (0.85, 1.0001, "[0.85,1]"),
    ]


def _bins_raw() -> List[Tuple[float, float, str]]:
    return [
        (0.0, 0.2, "[0,0.2)"),
        (0.2, 0.4, "[0.2,0.4)"),
        (0.4, 0.55, "[0.4,0.55)"),
        (0.55, 0.7, "[0.55,0.7)"),
        (0.7, 0.85, "[0.7,0.85)"),
        (0.85, 1.0001, "[0.85,1]"),
    ]


def _summarize(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"\n=== {label}: no rows ===")
        return
    print(f"\n=== {label}: n={len(df)} ===")
    for col in ("semantic_chop_ts_q", "semantic_chop"):
        s = pd.to_numeric(df[col], errors="coerce")
        print(
            f"  {col}: mean={s.mean():.4f} std={s.std():.4f} "
            f"p10={s.quantile(0.1):.4f} p50={s.quantile(0.5):.4f} p90={s.quantile(0.9):.4f}"
        )
    bins_ts = _bins_ts_q()
    rows = []
    for lo, hi, lab in bins_ts:
        m = (df["semantic_chop_ts_q"] >= lo) & (df["semantic_chop_ts_q"] < hi)
        bx = df.loc[m, "pnl_r"]
        if bx.empty:
            continue
        rows.append(
            {
                "bin_ts_q": lab,
                "n": len(bx),
                "win_rate": float((bx > 0).mean()),
                "mean_r": float(bx.mean()),
                "sum_r": float(bx.sum()),
            }
        )
    ts_df = pd.DataFrame(rows)
    print("  PnL by semantic_chop_ts_q bin (parent entries):")
    if ts_df.empty:
        print("    (empty)")
    else:
        print(ts_df.to_string(index=False))

    rows2 = []
    for lo, hi, lab in _bins_raw():
        m = (df["semantic_chop"] >= lo) & (df["semantic_chop"] < hi)
        bx = df.loc[m, "pnl_r"]
        if bx.empty:
            continue
        rows2.append(
            {
                "bin_raw": lab,
                "n": len(bx),
                "win_rate": float((bx > 0).mean()),
                "mean_r": float(bx.mean()),
                "sum_r": float(bx.sum()),
            }
        )
    print("  PnL by semantic_chop (raw) bin:")
    raw_df = pd.DataFrame(rows2)
    if raw_df.empty:
        print("    (empty)")
    else:
        print(raw_df.to_string(index=False))

    if not ts_df.empty:
        best = ts_df.sort_values("mean_r", ascending=False).head(3)
        worst = ts_df.sort_values("mean_r", ascending=True).head(3)
        print("  Best mean_r bins (ts_q):", best["bin_ts_q"].tolist())
        print("  Worst mean_r bins (ts_q):", worst["bin_ts_q"].tolist())


def _process_strategy(
    strat: str,
    run_roots: List[Path],
    data_dir: Path,
    timeframe: str,
    warmup_days: int,
    study_cfg: StudyConfig,
) -> pd.DataFrame:
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    out_rows: List[dict] = []
    for run_root in run_roots:
        run_id = run_root.name
        for month_dir in sorted(run_root.glob("fast_month_*")):
            token = month_dir.name
            fp = month_dir / strat / f"event_trades_{strat}.csv"
            if not fp.exists() or fp.stat().st_size < 10:
                continue
            try:
                tdf = pd.read_csv(fp)
            except Exception:
                continue
            if tdf.empty:
                continue
            if "is_add_position" in tdf.columns:
                add = (
                    tdf["is_add_position"]
                    .astype(str)
                    .str.lower()
                    .isin(("1", "true", "yes"))
                )
                tdf = tdf.loc[~add]
            if tdf.empty:
                continue
            syms = sorted(
                {str(r.get("symbol", "")).strip().upper() for _, r in tdf.iterrows()}
            )
            syms = [s for s in syms if s]
            for sym in syms:
                key = (strat, sym, token)
                if key not in cache:
                    if strat == "me":
                        fdf = _me_feature_frame(
                            sym, token, data_dir, timeframe, warmup_days
                        )
                    else:
                        fdf = _bpc_tpc_feature_frame(
                            sym, token, data_dir, study_cfg, timeframe, warmup_days
                        )
                    cache[key] = fdf
                fdf = cache[key]
                sub = tdf[tdf["symbol"].astype(str).str.upper() == sym]
                for _, row in sub.iterrows():
                    try:
                        ts = pd.Timestamp(row["entry_time"])
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        else:
                            ts = ts.tz_convert("UTC")
                    except Exception:
                        continue
                    pair = _lookup_bar(fdf, ts) if fdf is not None else None
                    if pair is None:
                        continue
                    raw_chop, tsq = pair
                    try:
                        pr = float(row.get("pnl_r", 0.0) or 0.0)
                    except Exception:
                        pr = 0.0
                    out_rows.append(
                        {
                            "strategy": strat,
                            "run_id": run_id,
                            "month": token.replace("fast_month_", ""),
                            "symbol": sym,
                            "entry_time": ts.isoformat(),
                            "semantic_chop": raw_chop,
                            "semantic_chop_ts_q": tsq,
                            "pnl_r": pr,
                        }
                    )
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-path", type=Path, default=PROJECT_ROOT / "data" / "parquet_data"
    )
    ap.add_argument(
        "--timeframe", default="120min", help="Resample rule (default 120min = 2H)"
    )
    ap.add_argument("--warmup-days", type=int, default=200)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--tpc-runs", nargs="*", type=Path, default=None)
    ap.add_argument("--me-runs", nargs="*", type=Path, default=None)
    ap.add_argument("--bpc-runs", nargs="*", type=Path, default=None)
    args = ap.parse_args()

    study_cfg = StudyConfig()

    def resolve(strat: str, explicit: Optional[List[Path]]) -> List[Path]:
        if explicit:
            return [Path(p).resolve() for p in explicit]
        return _discover_latest_runs(strat, PROJECT_ROOT, 2)

    tpc_runs = resolve("tpc", args.tpc_runs)
    me_runs = resolve("me", args.me_runs)
    bpc_runs = resolve("bpc", args.bpc_runs)

    for name, runs in [("tpc", tpc_runs), ("me", me_runs), ("bpc", bpc_runs)]:
        print(f"\n{name.upper()} runs ({len(runs)}):")
        for r in runs:
            print(f"  {r}")

    all_parts: List[pd.DataFrame] = []
    for strat, runs in [("tpc", tpc_runs), ("me", me_runs), ("bpc", bpc_runs)]:
        if not runs:
            print(f"skip {strat}: no run directories", file=sys.stderr)
            continue
        part = _process_strategy(
            strat, runs, args.data_path, args.timeframe, args.warmup_days, study_cfg
        )
        all_parts.append(part)
        for run in runs:
            sub = part[part["run_id"] == run.name]
            _summarize(sub, f"{strat} {run.name}")

    full = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if not full.empty:
        _summarize(full, "ALL RUNS POOLED")
    if args.out_csv and not full.empty:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(args.out_csv, index=False)
        print(f"\nWrote detail rows → {args.out_csv}")


if __name__ == "__main__":
    main()
