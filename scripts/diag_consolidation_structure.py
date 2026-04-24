"""Offline diagnostic: scan for "trend -> exhaustion -> consolidation" structures.

Goal:
  Prove whether two proposed strategies are viable on BTC 2024 bull-phase data:
    - CRF (Consolidation Range Fade): within the consolidation box, fade both
      sides (long near box-low, short near box-high).
    - CBC (Consolidation Breakout Continuation): buy the breakout in the
      direction of the prior trend after the box ends (chan-lun 2-buy).

Algorithm (per bar, 2H frame):
  1. Find a local "trend-peak bar" P (last 60-bar rolling-high of close).
  2. Require a prior uptrend at P:
        close[P] > ema_1200[P]   AND   trend_r2_20[P] >= 0.35
  3. Decay leg:   min close in [P, P+decay_max] drops >= 8% from close[P]
                  within decay_max = 30 bars (~2.5 days on 2H).
     Mark the bottom bar as B (end of exhaustion leg).
  4. Consolidation box search starting at B:
        box_hi = max(high) over [B, B+k]
        box_lo = min(low)  over [B, B+k]
     Extend k forward as long as:
        - k < max_len (720)
        - every bar's close stays inside [box_lo - tol, box_hi + tol]
          where tol = max(1 * atr_14, 0.02 * mid_box)
     Stop once a bar violates, or k == max_len. Require k >= min_len (30).
  5. Outcome classification (look at next look_fwd=120 bars after box end E):
        - "break_up"    : high exceeds box_hi + tol, before low breaks box_lo-tol
        - "break_down"  : low  breaks box_lo - tol first
        - "timeout"     : neither in look_fwd window
     Record Max-Favorable-Excursion (MFE) and Max-Adverse-Excursion (MAE) for
     each direction as pct of mid_box.

  Additionally, within each box record the number of touches of box_hi and
  box_lo (tol-ed) -- proxies for CRF tradability.

Outputs CSV at `reports/consolidation_btc_2024.csv` with one row per event.

Usage:
    python scripts/diag_consolidation_structure.py \\
        --symbol BTCUSDT --start 2024-05-01 --end 2024-10-31 \\
        --out reports/consolidation_btc_2024.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------- data loading -----------------------------


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


def _load_1m_range(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_m = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    frames = []
    while cur <= end_m:
        f = _load_month_1m(symbol, cur.year, cur.month)
        if not f.empty:
            frames.append(f)
        cur = cur + pd.offsets.MonthBegin(1)
    if not frames:
        return pd.DataFrame()
    allbars = pd.concat(frames).sort_index()
    return allbars[~allbars.index.duplicated(keep="first")]


def _resample_2h(m1: pd.DataFrame) -> pd.DataFrame:
    if m1.empty:
        return m1
    agg = m1.resample("2h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return agg.dropna(subset=["close"])


# ------------------------- indicators -------------------------------


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _trend_r2(close: pd.Series, n: int = 20) -> pd.Series:
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _one(y: np.ndarray) -> float:
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        y_var = ((y - y_mean) ** 2).sum()
        if y_var <= 0 or x_var <= 0:
            return 0.0
        return (cov * cov) / (x_var * y_var)

    return close.rolling(n, min_periods=n).apply(_one, raw=True)


# ------------------------- structure scanner ------------------------


def scan_events(
    df: pd.DataFrame,
    *,
    peak_window: int = 60,
    decay_max: int = 30,
    decay_pct: float = 0.08,
    min_len: int = 30,
    max_len: int = 720,
    tol_atr: float = 1.0,
    tol_pct: float = 0.02,
    look_fwd: int = 120,
    trend_r2_min: float = 0.35,
) -> pd.DataFrame:
    df = df.copy()
    df["ema1200"] = _ema(df["close"], 1200)
    df["atr14"] = _atr(df, 14)
    df["r2_20"] = _trend_r2(df["close"], 20)
    df["roll_hi_60"] = df["close"].rolling(peak_window, min_periods=peak_window).max()

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    ema = df["ema1200"].values
    atr = df["atr14"].values
    r2 = df["r2_20"].values
    roll_hi = df["roll_hi_60"].values
    ts = df.index

    events = []
    i = peak_window + 50  # minimal warm-up; ema1200 may still be NaN early
    while i < n - look_fwd:
        # peak: close[i] is within 0.3% of the 60-bar rolling high
        if np.isnan(roll_hi[i]):
            i += 1
            continue
        at_peak = close[i] >= roll_hi[i] * 0.997
        ema_ok = (not np.isnan(ema[i])) and close[i] > ema[i]
        # if ema1200 not warmed yet, fall back to 'close > close 200 bars ago' proxy
        if not ema_ok and i >= 200:
            ema_ok = close[i] > close[i - 200]
        if not (at_peak and ema_ok and r2[i] >= trend_r2_min):
            i += 1
            continue

        peak_idx = i
        peak_close = close[peak_idx]

        # decay search
        decay_end = min(peak_idx + decay_max, n - 1)
        decay_seg_close = close[peak_idx : decay_end + 1]
        if len(decay_seg_close) == 0:
            i += 1
            continue
        b_off = int(np.argmin(decay_seg_close))
        B = peak_idx + b_off
        drop = (peak_close - close[B]) / peak_close
        if drop < decay_pct:
            i += 1
            continue

        # consolidation box from B
        box_hi = high[B]
        box_lo = low[B]
        tol_ref = max(
            tol_atr * (atr[B] if not np.isnan(atr[B]) else 0.0),
            tol_pct * (box_hi + box_lo) * 0.5,
        )
        k = 0
        broken = False
        E = B
        touches_hi = 0
        touches_lo = 0
        for j in range(B + 1, min(B + max_len, n)):
            nh = max(box_hi, high[j])
            nl = min(box_lo, low[j])
            mid = (nh + nl) * 0.5
            new_tol = max(
                tol_atr * (atr[j] if not np.isnan(atr[j]) else 0.0),
                tol_pct * mid,
            )
            # require close within tol-ed box
            if close[j] > nh + new_tol or close[j] < nl - new_tol:
                broken = True
                E = j - 1
                break
            # acceptable extension
            box_hi = nh
            box_lo = nl
            tol_ref = new_tol
            if high[j] >= box_hi - new_tol * 0.25:
                touches_hi += 1
            if low[j] <= box_lo + new_tol * 0.25:
                touches_lo += 1
            E = j
            k = j - B
            if k >= max_len:
                break
        box_len = E - B
        if box_len < min_len:
            i = B + 1
            continue

        # look-forward outcome
        mid = (box_hi + box_lo) * 0.5
        fwd_start = E + 1
        fwd_end = min(E + look_fwd, n - 1)
        outcome = "timeout"
        break_bar = None
        for j in range(fwd_start, fwd_end + 1):
            if high[j] > box_hi + tol_ref:
                outcome = "break_up"
                break_bar = j
                break
            if low[j] < box_lo - tol_ref:
                outcome = "break_down"
                break_bar = j
                break

        # post-break MFE/MAE in pct of mid
        def _mfe_mae(direction: str):
            if break_bar is None:
                return np.nan, np.nan
            hi_j = min(break_bar + look_fwd, n - 1)
            seg_hi = high[break_bar : hi_j + 1]
            seg_lo = low[break_bar : hi_j + 1]
            ref = close[break_bar]
            if direction == "long":
                mfe = (seg_hi.max() - ref) / ref
                mae = (ref - seg_lo.min()) / ref
            else:
                mfe = (ref - seg_lo.min()) / ref
                mae = (seg_hi.max() - ref) / ref
            return mfe, mae

        if outcome == "break_up":
            mfe, mae = _mfe_mae("long")
        elif outcome == "break_down":
            mfe, mae = _mfe_mae("short")
        else:
            mfe, mae = (np.nan, np.nan)

        # --- CRF simulated in-box swings ---
        # Entry rule: when low <= box_lo + edge*width -> open long at close
        #             when high >= box_hi - edge*width -> open short at close
        # Exit:  opposite touch, or box end E.
        # Stop:  beyond box by 1.0 * box_width * stop_mult -> force close.
        edge_frac = 0.15
        stop_mult = 0.25  # stop = 25% of box width beyond the entry-side edge
        width = box_hi - box_lo
        lo_edge = box_lo + edge_frac * width
        hi_edge = box_hi - edge_frac * width
        stop_long = box_lo - stop_mult * width
        stop_short = box_hi + stop_mult * width
        tgt_long = box_hi - edge_frac * width  # exit long at opposite edge zone
        tgt_short = box_lo + edge_frac * width

        pos = 0  # 0 flat, +1 long, -1 short
        entry_price = 0.0
        entry_idx = 0
        crf_trades = []  # list of (side, bars_held, ret_pct, exit_reason)
        for j in range(B, E + 1):
            if pos == 0:
                if low[j] <= lo_edge:
                    pos = 1
                    entry_price = min(close[j], lo_edge)
                    entry_idx = j
                elif high[j] >= hi_edge:
                    pos = -1
                    entry_price = max(close[j], hi_edge)
                    entry_idx = j
            elif pos == 1:
                if low[j] <= stop_long:
                    ret = (stop_long - entry_price) / entry_price
                    crf_trades.append(("L", j - entry_idx, ret, "stop"))
                    pos = 0
                elif high[j] >= tgt_long:
                    ret = (tgt_long - entry_price) / entry_price
                    crf_trades.append(("L", j - entry_idx, ret, "tgt"))
                    pos = 0
            elif pos == -1:
                if high[j] >= stop_short:
                    ret = (entry_price - stop_short) / entry_price
                    crf_trades.append(("S", j - entry_idx, ret, "stop"))
                    pos = 0
                elif low[j] <= tgt_short:
                    ret = (entry_price - tgt_short) / entry_price
                    crf_trades.append(("S", j - entry_idx, ret, "tgt"))
                    pos = 0
        # force-close at E
        if pos == 1:
            ret = (close[E] - entry_price) / entry_price
            crf_trades.append(("L", E - entry_idx, ret, "boxend"))
        elif pos == -1:
            ret = (entry_price - close[E]) / entry_price
            crf_trades.append(("S", E - entry_idx, ret, "boxend"))

        crf_n = len(crf_trades)
        crf_n_long = sum(1 for t in crf_trades if t[0] == "L")
        crf_n_short = sum(1 for t in crf_trades if t[0] == "S")
        crf_ret_sum = sum(t[2] for t in crf_trades)
        crf_wins = sum(1 for t in crf_trades if t[2] > 0)
        crf_long_ret = sum(t[2] for t in crf_trades if t[0] == "L")
        crf_short_ret = sum(t[2] for t in crf_trades if t[0] == "S")
        crf_avg_hold = np.mean([t[1] for t in crf_trades]) if crf_trades else 0.0

        events.append(
            {
                "peak_time": ts[peak_idx],
                "peak_close": peak_close,
                "B_time": ts[B],
                "B_close": close[B],
                "decay_drop_pct": drop,
                "E_time": ts[E],
                "box_len_bars": box_len,
                "box_hi": box_hi,
                "box_lo": box_lo,
                "box_width_pct": (box_hi - box_lo) / mid,
                "touches_hi": touches_hi,
                "touches_lo": touches_lo,
                "prior_r2_20": r2[peak_idx],
                "prior_up": 1,  # we only scanned up-trends so far
                "outcome": outcome,
                "cbc_mfe_pct": mfe,
                "cbc_mae_pct": mae,
                "crf_n": crf_n,
                "crf_n_long": crf_n_long,
                "crf_n_short": crf_n_short,
                "crf_wins": crf_wins,
                "crf_ret_sum": crf_ret_sum,
                "crf_long_ret": crf_long_ret,
                "crf_short_ret": crf_short_ret,
                "crf_avg_hold_bars": crf_avg_hold,
            }
        )

        # skip to end of this box to avoid overlap
        i = E + 1

    return pd.DataFrame(events)


# ------------------------- causal scanner ---------------------------


def scan_events_causal(
    df: pd.DataFrame,
    *,
    min_len: int = 30,
    max_len: int = 720,
    look_fwd: int = 120,
) -> pd.DataFrame:
    """Causal box scanner using the registered box_structure features.

    Box = contiguous run where ``box_regime_label in {small, mid, big}``
    (derived from rolling-window stability + width — strictly causal).

    For each completed box we:
      * Record box_hi / box_lo as the running max(high) / min(low) inside B..E
        (these are the *effective* box boundaries from the perspective of a
        trader standing at E — no look-ahead because they only consume past
        values during the walk).
      * Simulate CRF swings inside the box using the same rules as the oracle
        version so PnL is directly comparable.
      * Look forward ``look_fwd`` bars for break_up / break_down and compute
        MFE/MAE identically.
    """
    from src.features.time_series.box_structure_features import (
        compute_box_structure_from_series,
    )

    feats = compute_box_structure_from_series(
        close=df["close"], high=df["high"], low=df["low"]
    )
    regime = feats["box_regime_label"].values
    ts = df.index
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    events = []
    i = 0
    while i < n - look_fwd:
        if regime[i] == "none":
            i += 1
            continue

        # start of a box run
        B = i
        entry_label = regime[B]
        # track effective box boundaries from running max/min
        box_hi = high[B]
        box_lo = low[B]
        E = B
        for j in range(B + 1, min(B + max_len, n)):
            if regime[j] == "none":
                E = j - 1
                break
            box_hi = max(box_hi, high[j])
            box_lo = min(box_lo, low[j])
            E = j
        box_len = E - B
        if box_len < min_len:
            i = E + 1
            continue

        mid = (box_hi + box_lo) * 0.5
        width = box_hi - box_lo
        tol_ref = 0.015 * mid

        # touches within box
        touches_hi = 0
        touches_lo = 0
        for j in range(B, E + 1):
            if high[j] >= box_hi - 0.25 * tol_ref:
                touches_hi += 1
            if low[j] <= box_lo + 0.25 * tol_ref:
                touches_lo += 1

        # CRF simulated in-box swings (same rules as oracle) ------------------
        edge_frac = 0.15
        stop_mult = 0.25
        lo_edge = box_lo + edge_frac * width
        hi_edge = box_hi - edge_frac * width
        stop_long = box_lo - stop_mult * width
        stop_short = box_hi + stop_mult * width
        tgt_long = box_hi - edge_frac * width
        tgt_short = box_lo + edge_frac * width

        pos = 0
        entry_price = 0.0
        entry_idx = 0
        crf_trades = []
        for j in range(B, E + 1):
            if pos == 0:
                if low[j] <= lo_edge:
                    pos = 1
                    entry_price = min(close[j], lo_edge)
                    entry_idx = j
                elif high[j] >= hi_edge:
                    pos = -1
                    entry_price = max(close[j], hi_edge)
                    entry_idx = j
            elif pos == 1:
                if low[j] <= stop_long:
                    ret = (stop_long - entry_price) / entry_price
                    crf_trades.append(("L", j - entry_idx, ret, "stop"))
                    pos = 0
                elif high[j] >= tgt_long:
                    ret = (tgt_long - entry_price) / entry_price
                    crf_trades.append(("L", j - entry_idx, ret, "tgt"))
                    pos = 0
            elif pos == -1:
                if high[j] >= stop_short:
                    ret = (entry_price - stop_short) / entry_price
                    crf_trades.append(("S", j - entry_idx, ret, "stop"))
                    pos = 0
                elif low[j] <= tgt_short:
                    ret = (entry_price - tgt_short) / entry_price
                    crf_trades.append(("S", j - entry_idx, ret, "tgt"))
                    pos = 0
        if pos == 1:
            ret = (close[E] - entry_price) / entry_price
            crf_trades.append(("L", E - entry_idx, ret, "boxend"))
        elif pos == -1:
            ret = (entry_price - close[E]) / entry_price
            crf_trades.append(("S", E - entry_idx, ret, "boxend"))

        crf_n = len(crf_trades)
        crf_n_long = sum(1 for t in crf_trades if t[0] == "L")
        crf_n_short = sum(1 for t in crf_trades if t[0] == "S")
        crf_ret_sum = sum(t[2] for t in crf_trades)
        crf_wins = sum(1 for t in crf_trades if t[2] > 0)
        crf_long_ret = sum(t[2] for t in crf_trades if t[0] == "L")
        crf_short_ret = sum(t[2] for t in crf_trades if t[0] == "S")
        crf_avg_hold = float(np.mean([t[1] for t in crf_trades])) if crf_trades else 0.0

        # look-forward outcome
        fwd_start = E + 1
        fwd_end = min(E + look_fwd, n - 1)
        outcome = "timeout"
        break_bar = None
        for j in range(fwd_start, fwd_end + 1):
            if high[j] > box_hi + tol_ref:
                outcome = "break_up"
                break_bar = j
                break
            if low[j] < box_lo - tol_ref:
                outcome = "break_down"
                break_bar = j
                break

        def _mfe_mae(direction: str):
            if break_bar is None:
                return np.nan, np.nan
            hi_j = min(break_bar + look_fwd, n - 1)
            seg_hi = high[break_bar : hi_j + 1]
            seg_lo = low[break_bar : hi_j + 1]
            ref = close[break_bar]
            if direction == "long":
                return (seg_hi.max() - ref) / ref, (ref - seg_lo.min()) / ref
            return (ref - seg_lo.min()) / ref, (seg_hi.max() - ref) / ref

        if outcome == "break_up":
            mfe, mae = _mfe_mae("long")
        elif outcome == "break_down":
            mfe, mae = _mfe_mae("short")
        else:
            mfe, mae = (np.nan, np.nan)

        events.append(
            {
                "B_time": ts[B],
                "B_close": close[B],
                "E_time": ts[E],
                "box_len_bars": box_len,
                "box_hi": box_hi,
                "box_lo": box_lo,
                "box_width_pct": width / mid if mid else 0.0,
                "entry_label": entry_label,
                "touches_hi": touches_hi,
                "touches_lo": touches_lo,
                "outcome": outcome,
                "cbc_mfe_pct": mfe,
                "cbc_mae_pct": mae,
                "crf_n": crf_n,
                "crf_n_long": crf_n_long,
                "crf_n_short": crf_n_short,
                "crf_wins": crf_wins,
                "crf_ret_sum": crf_ret_sum,
                "crf_long_ret": crf_long_ret,
                "crf_short_ret": crf_short_ret,
                "crf_avg_hold_bars": crf_avg_hold,
                # oracle compatibility columns
                "peak_time": ts[B],
                "peak_close": close[B],
                "decay_drop_pct": np.nan,
                "prior_r2_20": np.nan,
                "prior_up": 0,
            }
        )

        i = E + 1

    return pd.DataFrame(events)


# ------------------------- summary ----------------------------------


def summarize(ev: pd.DataFrame) -> str:
    if ev.empty:
        return "(no events)"
    lines = []
    lines.append(f"events: {len(ev)}")
    out = ev["outcome"].value_counts().to_dict()
    lines.append(f"outcome mix: {out}")
    for side in ("break_up", "break_down"):
        sub = ev[ev["outcome"] == side]
        if sub.empty:
            continue
        lines.append(
            f"  {side}: n={len(sub)}  "
            f"mfe med={sub['cbc_mfe_pct'].median():.3f}  "
            f"mae med={sub['cbc_mae_pct'].median():.3f}  "
            f"mfe/mae med={(sub['cbc_mfe_pct']/sub['cbc_mae_pct'].replace(0,np.nan)).median():.2f}"
        )
    lines.append(
        f"box_len bars: median={ev['box_len_bars'].median():.0f}  "
        f"p25={ev['box_len_bars'].quantile(.25):.0f}  "
        f"p75={ev['box_len_bars'].quantile(.75):.0f}"
    )
    lines.append(
        f"box_width_pct: median={ev['box_width_pct'].median():.3f}  "
        f"decay_drop_pct: median={ev['decay_drop_pct'].median():.3f}"
    )
    lines.append(
        f"touches_hi median={ev['touches_hi'].median():.0f}  "
        f"touches_lo median={ev['touches_lo'].median():.0f}  "
        "(proxy for CRF tradability)"
    )
    # CRF aggregate
    if "crf_n" in ev.columns:
        total = int(ev["crf_n"].sum())
        long_n = int(ev["crf_n_long"].sum())
        short_n = int(ev["crf_n_short"].sum())
        wins = int(ev["crf_wins"].sum())
        ret_sum = ev["crf_ret_sum"].sum()
        long_ret = ev["crf_long_ret"].sum()
        short_ret = ev["crf_short_ret"].sum()
        wr = wins / total if total else 0.0
        lines.append(
            f"\n[CRF sim] trades={total}  long={long_n}  short={short_n}  "
            f"winrate={wr:.1%}  total_ret={ret_sum:.3f}  "
            f"long_ret={long_ret:.3f}  short_ret={short_ret:.3f}  "
            f"avg_hold_bars={ev['crf_avg_hold_bars'].mean():.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--end", default="2024-10-31")
    ap.add_argument("--out", default="reports/consolidation_btc_2024.csv")
    ap.add_argument("--decay-pct", type=float, default=0.08)
    ap.add_argument("--decay-max", type=int, default=30)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=720)
    ap.add_argument("--tol-atr", type=float, default=1.0)
    ap.add_argument("--tol-pct", type=float, default=0.02)
    ap.add_argument("--look-fwd", type=int, default=120)
    ap.add_argument(
        "--mode",
        choices=("oracle", "causal"),
        default="oracle",
        help="oracle=lookahead scanner (alpha ceiling); causal=box feature based",
    )
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    print(f"[load] {args.symbol} 1m  {start.date()} -> {end.date()}")
    m1 = _load_1m_range(args.symbol, start, end)
    if m1.empty:
        print("No data.")
        return
    df2h = _resample_2h(m1)
    df2h = df2h[(df2h.index >= start) & (df2h.index <= end)]
    print(f"[load] 2h bars: {len(df2h)}")

    print(f"[mode] {args.mode}")
    if args.mode == "oracle":
        ev = scan_events(
            df2h,
            decay_max=args.decay_max,
            decay_pct=args.decay_pct,
            min_len=args.min_len,
            max_len=args.max_len,
            tol_atr=args.tol_atr,
            tol_pct=args.tol_pct,
            look_fwd=args.look_fwd,
        )
    else:
        ev = scan_events_causal(
            df2h,
            min_len=args.min_len,
            max_len=args.max_len,
            look_fwd=args.look_fwd,
        )
    print("\n" + summarize(ev))
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out_path, index=False)
    print(f"\n[write] {out_path} ({len(ev)} rows)")


if __name__ == "__main__":
    main()
