"""
Quantify whether FBF entries sit on / beyond each boundary family.

For every trade in event_trades_*.csv we output, at entry time:
  - close
  - distance (in ATR) to Boll upper/lower
  - fer_range_pos_20 (0..1)  — inside 20-bar swing H/L envelope
  - fer_ols_pos (0..1)       — inside OLS(96) channel
  - bb_position              — (close - bb_lower) / (bb_upper - bb_lower)
  - wide_sr_dist_atr         — distance to L3 swing edge in ATR
  - vp_poc_deviation         — % dist to POC
  - summary flags: is_outside_boll, is_at_boll_rail, is_outside_swing20

使用:
  python scripts/diag_fbf_entry_boundaries.py \
    --trades 'results/fbf/research_roll.features_on-exp-trail/_rolling_sim/20260422_202736/fast_month_*/fbf/event_trades_fbf.csv' \
    --feature-store feature_store/features_fbf_120T_06702ab6f8 \
    --out reports/fbf_entry_boundary_diag.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_bars(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period, min_periods=max(2, period // 2)).mean()
    sd = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    return mid + std_dev * sd, mid, mid - std_dev * sd


def rolling_swing(high: pd.Series, low: pd.Series, window: int):
    return (
        high.rolling(window, min_periods=max(2, window // 4)).max(),
        low.rolling(window, min_periods=max(2, window // 4)).min(),
    )


def diag(bars: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(bars["close"], errors="coerce").astype(float)
    high = pd.to_numeric(bars["high"], errors="coerce").astype(float)
    low = pd.to_numeric(bars["low"], errors="coerce").astype(float)
    atr = pd.to_numeric(
        bars.get("atr", pd.Series(np.nan, index=bars.index)), errors="coerce"
    ).astype(float)

    bb_up, bb_mid, bb_lo = bollinger(close, 20, 2.0)
    bb_pos = ((close - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)).clip(-0.5, 1.5)

    sw_hi20, sw_lo20 = rolling_swing(high, low, 20)

    out = pd.DataFrame(
        {
            "close": close,
            "atr": atr,
            "bb_upper": bb_up,
            "bb_lower": bb_lo,
            "bb_position": bb_pos,
            "dist_to_bb_upper_atr": (bb_up - close) / atr,
            "dist_to_bb_lower_atr": (close - bb_lo) / atr,
            "swing20_high": sw_hi20,
            "swing20_low": sw_lo20,
            "dist_to_swing20_high_atr": (sw_hi20 - close) / atr,
            "dist_to_swing20_low_atr": (close - sw_lo20) / atr,
        },
        index=bars.index,
    )

    for c in (
        "fer_range_pos_20",
        "fer_ols_pos",
        "fer_ols_width_norm",
        "fer_sr_failed_breakout_score",
        "fer_sr_failed_breakout_direction_signed",
        "sr_strength_max",
        "trend_r2_20",
        "bb_width_normalized_pct",
        "wide_sr_dist_atr",
        "wide_sr_range_width_atr",
        "wide_sr_side",
        "vp_poc_deviation",
    ):
        if c in bars.columns:
            out[c] = pd.to_numeric(bars[c], errors="coerce")
    return out


def run(trades_glob: str, store: str, out_path: str) -> None:
    files = sorted(glob.glob(trades_glob))
    frames = []
    for f in files:
        if os.path.getsize(f) < 20:
            continue
        d = pd.read_csv(f)
        if len(d):
            frames.append(d)
    if not frames:
        print("no trades")
        return
    trades = pd.concat(frames, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)

    bars_cache: Dict[str, pd.DataFrame] = {}
    diag_cache: Dict[str, pd.DataFrame] = {}
    rows = []
    for _, t in trades.iterrows():
        sym = t["symbol"]
        if sym not in bars_cache:
            b = load_bars(store, sym)
            if b.empty:
                continue
            bars_cache[sym] = b
            diag_cache[sym] = diag(b)
        d = diag_cache[sym]
        ts = t["entry_time"]
        if ts is pd.NaT:
            continue
        loc = d.index.searchsorted(ts)
        if loc >= len(d):
            loc = len(d) - 1
        row_d = d.iloc[loc]
        rec = {
            "symbol": sym,
            "side": t.get("side"),
            "entry_time": ts,
            "entry_price": float(t.get("entry_price", float("nan"))),
            "pnl_r": float(t.get("pnl_r", 0) or 0),
            "exit_reason": t.get("exit_reason", ""),
        }
        for c in d.columns:
            rec[c] = float(row_d[c]) if pd.notna(row_d[c]) else float("nan")
        rows.append(rec)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}  ({len(df)} rows)")

    if len(df) == 0:
        return

    # summary
    print("\n=== ENTRY BOUNDARY SUMMARY ===")

    def _pct(mask):
        return 100.0 * float(np.nansum(mask)) / max(len(df), 1)

    print(f"n_entries                 : {len(df)}")
    for name, mask in [
        ("bb_position >= 0.95 (near/over upper)", df["bb_position"] >= 0.95),
        ("bb_position <= 0.05 (near/over lower)", df["bb_position"] <= 0.05),
        (
            "bb_position outside [0,1]            ",
            (df["bb_position"] < 0) | (df["bb_position"] > 1),
        ),
        (
            "|dist_to_bb_rail| <= 0.1 ATR         ",
            (df[["dist_to_bb_upper_atr", "dist_to_bb_lower_atr"]].min(axis=1) <= 0.1),
        ),
        (
            "fer_range_pos_20 >= 0.9 or <= 0.1    ",
            (df["fer_range_pos_20"] >= 0.9) | (df["fer_range_pos_20"] <= 0.1),
        ),
        (
            "fer_ols_pos >= 0.9 or <= 0.1         ",
            (df["fer_ols_pos"] >= 0.9) | (df["fer_ols_pos"] <= 0.1),
        ),
        (
            "wide_sr_dist_atr <= 2                ",
            df.get("wide_sr_dist_atr", pd.Series(np.nan, index=df.index)) <= 2.0,
        ),
    ]:
        m = mask.reindex(df.index, fill_value=False).fillna(False).astype(bool)
        print(f"  {name}  {_pct(m):6.1f}%  ({int(m.sum())}/{len(df)})")

    print("\n=== DESCRIBE (at entry) ===")
    cols = [
        "bb_position",
        "dist_to_bb_upper_atr",
        "dist_to_bb_lower_atr",
        "fer_range_pos_20",
        "fer_ols_pos",
        "wide_sr_dist_atr",
        "fer_sr_failed_breakout_score",
        "sr_strength_max",
        "trend_r2_20",
    ]
    print(df[[c for c in cols if c in df.columns]].describe().T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--feature-store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.trades, args.feature_store, args.out)
