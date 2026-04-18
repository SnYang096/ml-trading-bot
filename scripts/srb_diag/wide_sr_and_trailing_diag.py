"""
SRB diagnostic: wide SR vs narrow SR + trailing-stop ATR expansion analysis.

Reads closed-trade CSVs from a rolling run, rebuilds 2H OHLC from tick parquet,
then for each SRB trade computes:
  - narrow SR (lookback=20) and wide SR (lookback=96) at entry_time
  - distance from entry -> nearest narrow SR vs nearest wide SR (same side)
  - for trailing_sl trades: atr_at_entry vs primary-tf ATR at exit,
    and the post-exit MFE in original direction (was it a washout?)

Outputs a markdown summary + csv.

Usage:
  python scripts/srb_diag/wide_sr_and_trailing_diag.py \
    --rolling-dir results/srb/slow-rolling-sim/_rolling_sim/20260417_163432 \
    --out-dir results/srb/diag/wide_sr_trailing_20260418
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PARQ_ROOT = "/home/yin/trading/ml_trading_bot/data/parquet_data"


def _month_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def _load_2h_ohlc(symbol: str, months: List[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        path = os.path.join(PARQ_ROOT, f"{symbol}_{m}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        frames.append(df[["price", "volume"]])
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames).sort_index()
    oh = raw["price"].resample("2H").ohlc()
    vol = raw["volume"].resample("2H").sum()
    bar = oh.join(vol).dropna()
    return bar


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _swing_sr(
    bar: pd.DataFrame, ts: pd.Timestamp, lookback: int
) -> Tuple[Optional[float], Optional[float]]:
    sub = bar.loc[:ts].tail(lookback)
    if sub.empty or len(sub) < max(3, min(lookback, len(sub))):
        return None, None
    return float(sub["low"].min()), float(sub["high"].max())


def _collect_trades(rolling_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    for f in sorted(
        glob.glob(os.path.join(rolling_dir, "fast_month_*/srb/event_trades_srb.csv"))
    ):
        with open(f) as fp:
            for r in csv.DictReader(fp):
                rows.append(r)
    return rows


def _needed_months(entry_ts: pd.Timestamp) -> List[str]:
    # need 96 bars of 2H lookback = 8 days -> fetch current + prev month
    months: List[str] = []
    cur = pd.Timestamp(year=entry_ts.year, month=entry_ts.month, day=1, tz="UTC")
    prev = cur - pd.Timedelta(days=2)
    nxt = (cur + pd.Timedelta(days=40)).replace(day=1)
    for ts in (prev, cur, nxt):
        months.append(_month_str(ts))
    return months


def _distance_to_sr(
    side: str, entry: float, sup: Optional[float], res: Optional[float]
) -> Optional[float]:
    """Distance (in %) to 'the SR on the opposing side' for direction judgement.
    Long → support below; Short → resistance above."""
    if side.upper() == "LONG" and sup is not None and sup > 0:
        return (entry - sup) / entry
    if side.upper() == "SHORT" and res is not None and res > 0:
        return (res - entry) / entry
    return None


def run(rolling_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    trades = _collect_trades(rolling_dir)
    print(f"[diag] loaded {len(trades)} SRB trades from {rolling_dir}")

    # cache ohlc by (symbol, month-span) to avoid reloading
    ohlc_cache: Dict[Tuple[str, Tuple[str, ...]], pd.DataFrame] = {}

    enriched: List[Dict] = []
    for r in trades:
        symbol = r["symbol"]
        side = r["side"]
        entry_ts = pd.to_datetime(r["entry_time"], utc=True)
        exit_ts = pd.to_datetime(r["exit_time"], utc=True)
        entry = float(r["entry_price"])
        exit_px = float(r["exit_price"])
        atr_entry = float(r.get("atr") or 0.0)
        exit_reason = r["exit_reason"]
        pnl_r = float(r["pnl_r"])

        months = sorted(set(_needed_months(entry_ts) + _needed_months(exit_ts)))
        # extend for post-exit MFE lookahead by 10 bars (adds 1 more month if near boundary)
        key = (symbol, tuple(months))
        bar = ohlc_cache.get(key)
        if bar is None:
            bar = _load_2h_ohlc(symbol, months)
            ohlc_cache[key] = bar
        if bar.empty:
            continue

        atr_series = _atr(bar, 14)

        # snap to closest bar <= entry_ts (backtest opens at bar close)
        idx_entry = bar.index.searchsorted(entry_ts, side="right") - 1
        idx_exit = bar.index.searchsorted(exit_ts, side="right") - 1
        if idx_entry < 0 or idx_exit < 0 or idx_entry >= len(bar):
            continue
        entry_bar_ts = bar.index[idx_entry]
        exit_bar_ts = bar.index[idx_exit]

        sup_n, res_n = _swing_sr(bar, entry_bar_ts, 20)
        sup_w, res_w = _swing_sr(bar, entry_bar_ts, 96)

        dist_narrow = _distance_to_sr(side, entry, sup_n, res_n)
        dist_wide = _distance_to_sr(side, entry, sup_w, res_w)

        atr_exit_primary = (
            float(atr_series.iloc[idx_exit])
            if idx_exit < len(atr_series) and not np.isnan(atr_series.iloc[idx_exit])
            else np.nan
        )
        atr_ratio = (
            (atr_exit_primary / atr_entry)
            if (atr_entry > 0 and not np.isnan(atr_exit_primary))
            else np.nan
        )

        # post-exit MFE in original direction (next 10 bars)
        fwd = bar.iloc[idx_exit + 1 : idx_exit + 11]
        if side.upper() == "LONG" and not fwd.empty:
            mfe = (fwd["high"].max() - exit_px) / (
                atr_entry if atr_entry > 0 else 1.0
            )  # in R-ish units (atr)
            mfe_pct = (fwd["high"].max() - exit_px) / exit_px
        elif side.upper() == "SHORT" and not fwd.empty:
            mfe = (exit_px - fwd["low"].min()) / (atr_entry if atr_entry > 0 else 1.0)
            mfe_pct = (exit_px - fwd["low"].min()) / exit_px
        else:
            mfe = np.nan
            mfe_pct = np.nan

        enriched.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": r["entry_time"],
                "exit_time": r["exit_time"],
                "exit_reason": exit_reason,
                "pnl_r": pnl_r,
                "is_add_position": r["is_add_position"],
                "is_reverse": r["is_reverse"],
                "entry_price": entry,
                "exit_price": exit_px,
                "atr_entry": atr_entry,
                "atr_exit_primary": atr_exit_primary,
                "atr_ratio_exit_over_entry": atr_ratio,
                "sup_narrow": sup_n,
                "res_narrow": res_n,
                "sup_wide": sup_w,
                "res_wide": res_w,
                "dist_narrow_pct": dist_narrow,
                "dist_wide_pct": dist_wide,
                "post_exit_mfe_atr": mfe,
                "post_exit_mfe_pct": mfe_pct,
            }
        )

    out_csv = os.path.join(out_dir, "trades_with_wide_sr_and_atr.csv")
    df = pd.DataFrame(enriched)
    df.to_csv(out_csv, index=False)
    print(f"[diag] wrote {out_csv} ({len(df)} rows)")

    # ---- Summaries ----
    lines: List[str] = []
    lines.append("# SRB Wide-SR & Trailing-ATR Diagnostic\n")
    lines.append(f"Rolling run: `{rolling_dir}`\n")
    lines.append(f"Total trades analysed: {len(df)}\n")

    # -- 1. wide vs narrow SR distance --
    lines.append("\n## 1. Wide SR (96 bars ≈ 8 天) vs Narrow SR (20 bars ≈ 40h)\n")
    lines.append(
        '窄窗 SR = 近 20 根 2H low/high；宽窗 SR = 近 96 根 low/high（约 8 天），更贴近"二期" swing 结构。\n'
    )

    # filter rows where both exist
    valid = df.dropna(subset=["dist_narrow_pct", "dist_wide_pct"]).copy()
    if not valid.empty:
        lines.append(f"- 样本（narrow+wide 均可用）：{len(valid)}\n")
        lines.append(
            f"- 入场→narrow SR 距离（%）：median={valid['dist_narrow_pct'].median()*100:.2f}%, p90={valid['dist_narrow_pct'].quantile(0.9)*100:.2f}%\n"
        )
        lines.append(
            f"- 入场→wide   SR 距离（%）：median={valid['dist_wide_pct'].median()*100:.2f}%, p90={valid['dist_wide_pct'].quantile(0.9)*100:.2f}%\n"
        )
        widened = (valid["dist_wide_pct"] > valid["dist_narrow_pct"] * 1.05).sum()
        equal = (
            abs(valid["dist_wide_pct"] - valid["dist_narrow_pct"])
            / valid["dist_narrow_pct"].abs()
            < 0.05
        ).sum()
        lines.append(
            f"- wide 比 narrow 更远（>5%）：{widened}/{len(valid)} ({widened/len(valid)*100:.1f}%) → 宽窗能看到更靠后的结构位\n"
        )
        lines.append(
            f"- wide ≈ narrow（差<5%）：{equal}/{len(valid)} ({equal/len(valid)*100:.1f}%) → 结构高度压缩，宽窗不额外给信号\n"
        )

    # -- 2. trailing_sl behavior --
    lines.append("\n## 2. trailing_sl 退出的 ATR 扩张 & 洗出检验\n")
    tr = df[df["exit_reason"] == "trailing_sl"].copy()
    lines.append(
        f"- trailing_sl 样本：{len(tr)}（占总量 {len(tr)/len(df)*100:.1f}% ）\n"
    )
    if not tr.empty:
        ratios = tr["atr_ratio_exit_over_entry"].dropna()
        if not ratios.empty:
            lines.append(
                f"- ATR(exit_bar) / ATR(at_entry) 分布：median={ratios.median():.2f}, p75={ratios.quantile(0.75):.2f}, p90={ratios.quantile(0.9):.2f}\n"
            )
            expand = (ratios >= 1.2).sum()
            lines.append(
                f"  - 其中 ratio ≥ 1.2 的：{expand}/{len(ratios)} ({expand/len(ratios)*100:.1f}%) → 这批 trailing_sl 时波动已显著放大，用入场 ATR 跟踪偏紧\n"
            )
        mfe = tr["post_exit_mfe_atr"].dropna()
        if not mfe.empty:
            lines.append(
                f"- 退出后 10 根 bar 内的 MFE（以入场 ATR 为单位，原方向）：median={mfe.median():.2f} ATR, p75={mfe.quantile(0.75):.2f}, p90={mfe.quantile(0.9):.2f}\n"
            )
            wash = (mfe >= 2.0).sum()
            lines.append(
                f"  - MFE ≥ 2 ATR：{wash}/{len(mfe)} ({wash/len(mfe)*100:.1f}%) → 被洗出、随后出现 ≥2 ATR 同向延续\n"
            )

        # cross-tab: add_position vs not
        addp = tr[tr["is_add_position"] == "True"]
        main = tr[tr["is_add_position"] != "True"]
        if not main.empty:
            lines.append(
                f"- 主仓 trailing_sl：{len(main)}；平均 R = {main['pnl_r'].astype(float).mean():+.2f}\n"
            )
        if not addp.empty:
            lines.append(
                f"- 加仓 trailing_sl：{len(addp)}；平均 R = {addp['pnl_r'].astype(float).mean():+.2f}\n"
            )

    # -- 3. breakdown per symbol --
    lines.append("\n## 3. 按 symbol 拆分 trailing_sl\n")
    if not tr.empty:
        by_sym = (
            tr.groupby("symbol")
            .agg(
                n=("symbol", "count"),
                mean_r=("pnl_r", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                median_mfe=("post_exit_mfe_atr", "median"),
                median_ratio=("atr_ratio_exit_over_entry", "median"),
            )
            .round(3)
        )
        lines.append(
            "| symbol | n | mean_R | median post_exit MFE (ATR) | median ATR(exit)/ATR(entry) |\n|---|---|---|---|---|\n"
        )
        for sym, row in by_sym.iterrows():
            lines.append(
                f"| {sym} | {int(row['n'])} | {row['mean_r']:+.2f} | {row['median_mfe']:.2f} | {row['median_ratio']:.2f} |\n"
            )

    md_path = os.path.join(out_dir, "wide_sr_trailing_diag.md")
    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"[diag] wrote {md_path}")

    # also dump a small JSON for quick grep
    summary = {
        "n_trades": int(len(df)),
        "n_trailing_sl": int(len(tr)),
        "median_dist_narrow_pct": (
            float(valid["dist_narrow_pct"].median()) if not valid.empty else None
        ),
        "median_dist_wide_pct": (
            float(valid["dist_wide_pct"].median()) if not valid.empty else None
        ),
        "median_atr_ratio_at_trailing_exit": (
            float(tr["atr_ratio_exit_over_entry"].median()) if not tr.empty else None
        ),
        "median_post_exit_mfe_atr": (
            float(tr["post_exit_mfe_atr"].median()) if not tr.empty else None
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolling-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run(args.rolling_dir, args.out_dir)
