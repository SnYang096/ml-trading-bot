"""Diagnose FBF-strict entries: did they really land at OLS(96) boundary?

For each trade in the rolling-sim output:
  1. Load raw 1-min parquet for the symbol spanning (entry_time - 30d) .. entry_time
  2. Resample to 2H OHLCV
  3. Compute fer_ols_pos (OLS 96-bar channel), bb_position (20,2σ),
     fer_range_pos_20 (20-bar lo/hi), wide_sr_side (240-bar hi/lo shift=12)
  4. Pin to the 2H bar whose open-time == entry_time; report values.

Usage:
    python scripts/diag_fbf_strict_entries.py \\
        --run-dir results/fbf/calibrate_roll.default-strict/_rolling_sim/20260423_154304 \\
        --strategy fbf \\
        --out reports/fbf_strict_entry_diag.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_series.fer_features import _rolling_ols_channel  # noqa: E402


def _load_month_1m(symbol: str, year: int, month: int) -> pd.DataFrame:
    p = (
        PROJECT_ROOT
        / "data"
        / "parquet_data"
        / f"{symbol}_{year:04d}-{month:02d}.parquet"
    )
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    # collapse buy/sell rows into 1-min bars
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    grp = df.groupby("timestamp")
    bars = grp.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    )
    return bars


def _load_1m_spanning(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    months = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_m = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= end_m:
        months.append((cur.year, cur.month))
        cur = cur + pd.offsets.MonthBegin(1)
    frames = []
    for y, m in months:
        f = _load_month_1m(symbol, y, m)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    allbars = pd.concat(frames).sort_index()
    return allbars[~allbars.index.duplicated(keep="first")]


def _resample_2h(m1: pd.DataFrame) -> pd.DataFrame:
    if m1.empty:
        return m1
    o = m1["open"].resample("2h", origin="epoch", label="left", closed="left").first()
    h = m1["high"].resample("2h", origin="epoch", label="left", closed="left").max()
    l = m1["low"].resample("2h", origin="epoch", label="left", closed="left").min()
    c = m1["close"].resample("2h", origin="epoch", label="left", closed="left").last()
    v = m1["volume"].resample("2h", origin="epoch", label="left", closed="left").sum()
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(
        subset=["close"]
    )
    return df


def _compute_features(bars2h: pd.DataFrame) -> pd.DataFrame:
    close = bars2h["close"]
    high = bars2h["high"]
    low = bars2h["low"]

    ols_mid, ols_width = _rolling_ols_channel(close, window=96)
    ols_half = (ols_width / 2.0).replace(0.0, np.nan)
    fer_ols_pos = ((close - (ols_mid - ols_half)) / (2.0 * ols_half)).clip(0.0, 1.0)
    fer_ols_pos = fer_ols_pos.fillna(0.5)

    rng_hi = high.rolling(20, min_periods=1).max()
    rng_lo = low.rolling(20, min_periods=1).min()
    den = (rng_hi - rng_lo).replace(0.0, np.nan)
    range_pos_20 = ((close - rng_lo) / den).clip(0.0, 1.0).fillna(0.5)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_lo = bb_mid - 2 * bb_std
    bb_den = (bb_up - bb_lo).replace(0.0, np.nan)
    bb_position = ((close - bb_lo) / bb_den).clip(-1.0, 2.0).fillna(0.5)

    w_hi = high.shift(12).rolling(240, min_periods=30).max()
    w_lo = low.shift(12).rolling(240, min_periods=30).min()
    wide_side = pd.Series(0, index=close.index, dtype="int8")
    wide_side[close > w_hi] = 1
    wide_side[close < w_lo] = -1

    return pd.DataFrame(
        {
            "close": close,
            "fer_ols_pos": fer_ols_pos,
            "ols_mid": ols_mid,
            "ols_width": ols_width,
            "range_pos_20": range_pos_20,
            "bb_position": bb_position,
            "wide_sr_side": wide_side,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="results/fbf/calibrate_roll.default-strict/_rolling_sim/20260423_154304",
    )
    ap.add_argument("--strategy", default="fbf")
    ap.add_argument("--out", default="reports/fbf_strict_entry_diag.csv")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    csvs = sorted(
        run_dir.glob(f"fast_month_*/{args.strategy}/event_trades_{args.strategy}.csv")
    )
    frames = []
    for p in csvs:
        d = pd.read_csv(p)
        if d.empty:
            continue
        d["_month"] = p.parent.parent.name.replace("fast_month_", "")
        frames.append(d)
    trades = pd.concat(frames, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"[info] loaded {len(trades)} trades from {len(csvs)} month dirs")

    cache: dict[str, pd.DataFrame] = {}

    rows = []
    for _, row in trades.iterrows():
        sym = row["symbol"]
        et = row["entry_time"]
        cache_key = f"{sym}|{et.year}-{et.month:02d}"
        if cache_key not in cache:
            # need OLS-96 warm-up: 96 * 2h = 8 days ≈ include prior month
            warmup_start = (
                (et - pd.Timedelta(days=30))
                .to_pydatetime()
                .replace(tzinfo=pd.Timestamp("2020-01-01", tz="UTC").tz)
            )
            m1 = _load_1m_spanning(sym, pd.Timestamp(warmup_start), et)
            bars2h = _resample_2h(m1)
            feats = _compute_features(bars2h)
            cache[cache_key] = feats

        feats = cache[cache_key]
        # entry_time is expected to be exactly a 2H bar open
        if et in feats.index:
            rec = feats.loc[et].to_dict()
        else:
            # try aligning to the 2H bar start
            bar_open = et.floor("2h")
            rec = feats.loc[bar_open].to_dict() if bar_open in feats.index else {}
        out = {
            "month": row["_month"],
            "symbol": sym,
            "side": row["side"],
            "entry_time": et.isoformat(),
            "entry_price": row["entry_price"],
            "pnl_r": row["pnl_r"],
            "exit_reason": row["exit_reason"],
            "bars_held": row.get("bars_held", np.nan),
        }
        out.update(
            {
                k: rec.get(k, np.nan)
                for k in (
                    "fer_ols_pos",
                    "range_pos_20",
                    "bb_position",
                    "wide_sr_side",
                    "ols_mid",
                    "ols_width",
                )
            }
        )
        rows.append(out)

    res = pd.DataFrame(rows)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path}")

    def _dir_ok(r):
        # LONG must have fer_ols_pos <= 0.10; SHORT must have fer_ols_pos >= 0.90
        if pd.isna(r["fer_ols_pos"]):
            return np.nan
        if r["side"] == "LONG":
            return r["fer_ols_pos"] <= 0.10
        return r["fer_ols_pos"] >= 0.90

    def _dir_wide(r):
        if pd.isna(r["fer_ols_pos"]):
            return np.nan
        if r["side"] == "LONG":
            return r["fer_ols_pos"] <= 0.20
        return r["fer_ols_pos"] >= 0.80

    def _bb_side(r):
        if pd.isna(r["bb_position"]):
            return np.nan
        if r["side"] == "LONG":
            return r["bb_position"] <= 0.05
        return r["bb_position"] >= 0.95

    res["ols_boundary_strict"] = res.apply(_dir_ok, axis=1)
    res["ols_boundary_soft"] = res.apply(_dir_wide, axis=1)
    res["bb_extreme"] = res.apply(_bb_side, axis=1)

    print("\n=== 语义命中率（36 笔）===")
    print(
        f"OLS 严格贴边 (long<=0.10 / short>=0.90) :"
        f" {res['ols_boundary_strict'].sum():.0f} / {res['ols_boundary_strict'].notna().sum()}"
        f"  = {100*res['ols_boundary_strict'].mean():.1f}%"
    )
    print(
        f"OLS 宽松贴边 (long<=0.20 / short>=0.80) :"
        f" {res['ols_boundary_soft'].sum():.0f} / {res['ols_boundary_soft'].notna().sum()}"
        f"  = {100*res['ols_boundary_soft'].mean():.1f}%"
    )
    print(
        f"BB 极值  (long<=0.05 / short>=0.95)    :"
        f" {res['bb_extreme'].sum():.0f} / {res['bb_extreme'].notna().sum()}"
        f"  = {100*res['bb_extreme'].mean():.1f}%"
    )

    print("\n=== fer_ols_pos 分布 (按 side) ===")
    for side, g in res.groupby("side"):
        print(
            f"  {side:<5} n={len(g):2d} "
            f"mean={g['fer_ols_pos'].mean():.3f} "
            f"median={g['fer_ols_pos'].median():.3f} "
            f"p10={g['fer_ols_pos'].quantile(0.1):.3f} "
            f"p90={g['fer_ols_pos'].quantile(0.9):.3f}"
        )

    print("\n=== 严格命中 vs 未命中 pnl 对比 ===")
    for tag, sub in res.groupby("ols_boundary_strict"):
        if sub.empty:
            continue
        print(
            f"  strict={tag}: n={len(sub):2d} total_r={sub['pnl_r'].sum():.2f} "
            f"mean_r={sub['pnl_r'].mean():.3f} win={100*(sub['pnl_r']>0).mean():.1f}%"
        )

    print("\n=== 每笔入场详情 ===")
    show_cols = [
        "month",
        "symbol",
        "side",
        "entry_time",
        "fer_ols_pos",
        "range_pos_20",
        "bb_position",
        "wide_sr_side",
        "pnl_r",
        "exit_reason",
    ]
    print(res[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
