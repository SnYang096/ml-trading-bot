"""
Offline variant experiment: on each existing baseline FBF trade, evaluate a
proposed **stricter prefilter** ("near-rail" + "exhaustion evidence") at the
entry bar. Drop trades that fail, then re-simulate execution on the subset
using the same 2H replay pipeline as ``simulate_exec_trail.py``.

This is a **subset filter** — it cannot invent new entries the old prefilter
missed, but it answers: "if we had required near-rail + exhaustion, how would
the resulting trade set perform vs the current production FBF?".

Scenarios (kept deliberately simple):
  A  near_ols_only     : fer_ols_pos >= 0.9 OR <= 0.1
  B  near_any_rail     : A  OR  fer_range_pos_20 extreme  OR  bb_position extreme
  C  B + exhaustion    : B AND any one of {efficiency_flip_strength,
                           aggressor_absorption, momentum_efficiency_decay}
                           above given percentile threshold

Usage:
  python scripts/filter_then_resim_fbf.py \
    --trades 'results/fbf/slow-rolling-sim-exp-trail/_rolling_sim/20260422_202736/fast_month_*/fbf/event_trades_fbf.csv' \
    --feature-store feature_store/features_fbf_120T_06702ab6f8 \
    --exec-config config/strategies/fbf/archetypes/execution.yaml \
    --out-dir reports/fbf_prefilter_variant
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.simulate_exec_trail import (  # noqa: E402
    load_exec,
    simulate_trade,
)


def load_bars_full(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    """Load all feature columns (simulate_exec_trail's loader strips to OHLC+ATR)."""
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    for col in ("open", "high", "low", "close", "atr"):
        if col not in df.columns:
            df[col] = np.nan
    return df


EXTREME_HI = 0.90
EXTREME_LO = 0.10
BBPOS_HI = 0.95
BBPOS_LO = 0.05


def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period, min_periods=max(2, period // 2)).mean()
    sd = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    return mid + std_dev * sd, mid, mid - std_dev * sd


def enrich_bars(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    bb_up, _, bb_lo = bollinger(close, 20, 2.0)
    denom = (bb_up - bb_lo).replace(0.0, np.nan)
    out["bb_position"] = ((close - bb_lo) / denom).clip(-1.0, 2.0)
    return out


def near_rail_mask(
    row: pd.Series, use_bb: bool, use_swing: bool, use_ols: bool
) -> bool:
    ok = False
    if use_ols and pd.notna(row.get("fer_ols_pos")):
        v = float(row["fer_ols_pos"])
        ok = ok or (v >= EXTREME_HI) or (v <= EXTREME_LO)
    if use_swing and pd.notna(row.get("fer_range_pos_20")):
        v = float(row["fer_range_pos_20"])
        ok = ok or (v >= EXTREME_HI) or (v <= EXTREME_LO)
    if use_bb and pd.notna(row.get("bb_position")):
        v = float(row["bb_position"])
        ok = ok or (v >= BBPOS_HI) or (v <= BBPOS_LO)
    return ok


def exhaustion_mask(row: pd.Series, q: Dict[str, float]) -> bool:
    ok = False
    for c, thr in q.items():
        v = row.get(c)
        if pd.notna(v) and float(v) >= thr:
            ok = True
            break
    return ok


def percentile_thresholds(
    bars_cache: Dict[str, pd.DataFrame], cols, q: float = 0.60
) -> Dict[str, float]:
    """Global quantile across all cached symbols (flat pool)."""
    pool: Dict[str, list] = {c: [] for c in cols}
    for df in bars_cache.values():
        for c in cols:
            if c in df.columns:
                pool[c].append(pd.to_numeric(df[c], errors="coerce").dropna().values)
    out = {}
    for c, arrs in pool.items():
        if arrs:
            cat = np.concatenate(arrs)
            if cat.size:
                out[c] = float(np.quantile(cat, q))
    return out


def summarize(out: pd.DataFrame, label: str) -> dict:
    n = int(len(out))
    total = float(out["pnl_r"].sum()) if n else 0.0
    mean = float(out["pnl_r"].mean()) if n else 0.0
    win = float((out["pnl_r"] > 0).mean() * 100) if n else 0.0
    max_r = float(out["pnl_r"].max()) if n else 0.0
    if n and "entry_time" in out.columns:
        tmp = out.copy()
        tmp["m"] = pd.to_datetime(tmp["entry_time"]).dt.to_period("M").astype(str)
        mon = tmp.groupby("m")["pnl_r"].sum()
        sharpe = float(mon.mean() / max(mon.std(), 1e-9) * np.sqrt(12))
        cum = tmp.sort_values("entry_time")["pnl_r"].cumsum()
        dd = float((cum - cum.cummax()).min())
    else:
        sharpe = 0.0
        dd = 0.0
    return dict(
        scenario=label,
        n=n,
        totalR=round(total, 2),
        meanR=round(mean, 3),
        winPct=round(win, 1),
        sharpe=round(sharpe, 2),
        maxDD=round(dd, 2),
        maxWin=round(max_r, 2),
    )


def run(
    trades_glob: str,
    store: str,
    exec_path: str,
    out_dir: str,
    *,
    q_exhaust: float = 0.60,
) -> None:
    exec_cfg = load_exec(exec_path)

    files = sorted(glob.glob(trades_glob))
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 20]
    frames = [d for d in frames if len(d)]
    trades = pd.concat(frames, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    print(f"baseline trades: {len(trades)} from {len(files)} files")

    bars_cache: Dict[str, pd.DataFrame] = {}
    for sym in trades["symbol"].unique():
        b = load_bars_full(store, sym)
        if b.empty:
            continue
        bars_cache[sym] = enrich_bars(b)
    print(f"loaded bars for {len(bars_cache)} symbols")

    exhaust_cols = [
        "fer_efficiency_flip_strength",
        "fer_aggressor_absorption",
        "fer_momentum_efficiency_decay",
    ]
    exhaust_thr = percentile_thresholds(bars_cache, exhaust_cols, q=q_exhaust)
    print(f"exhaustion q{int(q_exhaust*100)} thresholds: {exhaust_thr}")

    rows_at_entry = []
    for _, t in trades.iterrows():
        sym = t["symbol"]
        if sym not in bars_cache:
            continue
        b = bars_cache[sym]
        idx = b.index.searchsorted(t["entry_time"])
        if idx >= len(b):
            idx = len(b) - 1
        row = b.iloc[idx]
        rec = {
            "symbol": sym,
            "side": t["side"],
            "entry_time": t["entry_time"],
            "entry_price": float(t.get("entry_price", np.nan)),
            "atr": float(t.get("atr", 0.0) or 0.0),
            "baseline_pnl_r": float(t.get("pnl_r", 0) or 0),
        }
        for c in (
            "bb_position",
            "fer_ols_pos",
            "fer_range_pos_20",
            "fer_efficiency_flip_strength",
            "fer_aggressor_absorption",
            "fer_momentum_efficiency_decay",
        ):
            rec[c] = float(row[c]) if c in row.index and pd.notna(row[c]) else np.nan
        rec["near_ols"] = (
            (rec["fer_ols_pos"] >= EXTREME_HI) or (rec["fer_ols_pos"] <= EXTREME_LO)
            if pd.notna(rec["fer_ols_pos"])
            else False
        )
        rec["near_swing20"] = (
            (rec["fer_range_pos_20"] >= EXTREME_HI)
            or (rec["fer_range_pos_20"] <= EXTREME_LO)
            if pd.notna(rec["fer_range_pos_20"])
            else False
        )
        rec["near_bb"] = (
            (rec["bb_position"] >= BBPOS_HI) or (rec["bb_position"] <= BBPOS_LO)
            if pd.notna(rec["bb_position"])
            else False
        )
        rec["exhaust"] = exhaustion_mask(pd.Series(rec), exhaust_thr)
        rows_at_entry.append(rec)

    at = pd.DataFrame(rows_at_entry)

    scenarios = {
        "BASELINE": pd.Series(True, index=at.index),
        "A_near_ols_only": at["near_ols"],
        "B_near_any_rail": (at["near_ols"] | at["near_swing20"] | at["near_bb"]),
        "C_near_any_and_exhaust": (
            (at["near_ols"] | at["near_swing20"] | at["near_bb"]) & at["exhaust"]
        ),
        "D_near_ols_and_exhaust": at["near_ols"] & at["exhaust"],
    }

    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    for name, mask in scenarios.items():
        sub = at.loc[mask].reset_index(drop=True)
        print(f"\n===== {name}  (kept {len(sub)}/{len(at)}) =====")
        if name == "BASELINE":
            # baseline: use reported pnl_r (no resim) so it matches prod figures
            out = sub.rename(columns={"baseline_pnl_r": "pnl_r"})
        else:
            # resim with the same execution config on the filtered subset
            replay_rows = []
            for _, r in sub.iterrows():
                sym = r["symbol"]
                if sym not in bars_cache:
                    continue
                atr = float(r["atr"])
                if atr <= 0:
                    continue
                res = simulate_trade(
                    bars_cache[sym],
                    entry_time=r["entry_time"],
                    side=r["side"],
                    entry_price=float(r["entry_price"]),
                    atr_at_entry=atr,
                    exec_cfg=exec_cfg,
                )
                res["symbol"] = sym
                res["side"] = r["side"]
                res["entry_time"] = r["entry_time"]
                res["entry_price"] = float(r["entry_price"])
                res["baseline_pnl_r"] = float(r["baseline_pnl_r"])
                replay_rows.append(res)
            out = pd.DataFrame(replay_rows)

        stats = summarize(out, name)
        summary_rows.append(stats)
        print(
            f"  n={stats['n']:4d}  totalR={stats['totalR']:+7.2f}  meanR={stats['meanR']:+6.3f}"
            f"  win={stats['winPct']:5.1f}%  sharpe={stats['sharpe']:+5.2f}"
            f"  maxDD={stats['maxDD']:+6.2f}  maxWin={stats['maxWin']:+5.2f}"
        )
        out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nsaved to: {out_dir}/summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--feature-store", required=True)
    ap.add_argument("--exec-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--q-exhaust", type=float, default=0.60)
    args = ap.parse_args()
    run(
        args.trades,
        args.feature_store,
        args.exec_config,
        args.out_dir,
        q_exhaust=args.q_exhaust,
    )
