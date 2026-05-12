"""
Plot BTC (or any symbol) 2H candles with the FBF boundary family + trade dots.

目的：核验「入场是否真的在边界/边界外」。画出:
  - close 线 (+ 可选蜡烛)
  - Boll(period, std_dev) 上/下/中
  - OLS(window) 上/下/中  (和 `fer_features._rolling_ols_channel` 一致)
  - 20 根 rolling swing high/low (和 `fer_range_pos_20` 同源)
  - 240 根 L3 wide swing (shift=12)
  - FBF event_trades 入场点 (颜色区分 LONG/SHORT, 悬停显示 pnl_r / exit_reason)
  - 旁路面板: fer_range_pos_20, fer_ols_pos 子图 (若 feature store 提供)

用法:
  python scripts/plot_fbf_boundaries.py \
      --symbol BTCUSDT \
      --trades 'results/fbf/research_roll.features_on-exp-trail/_rolling_sim/20260422_202736/fast_month_*/fbf/event_trades_fbf.csv' \
      --feature-store feature_store/features_fbf_120T_06702ab6f8 \
      --start 2024-01-01 --end 2024-12-31 \
      --out reports/fbf_boundaries_BTC.html
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.features.time_series.fer_features import _rolling_ols_channel  # noqa: E402


def load_bars(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period, min_periods=max(2, period // 2)).mean()
    sd = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    return mid + std_dev * sd, mid, mid - std_dev * sd


def rolling_swing(high: pd.Series, low: pd.Series, window: int, shift: int = 0):
    hi = high.rolling(window, min_periods=max(2, window // 4)).max()
    lo = low.rolling(window, min_periods=max(2, window // 4)).min()
    if shift:
        hi = hi.shift(shift)
        lo = lo.shift(shift)
    return hi, lo


def load_trades(glob_pat: str, symbol: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pat))
    dfs = []
    for f in files:
        if os.path.getsize(f) < 20:
            continue
        d = pd.read_csv(f)
        if len(d):
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    trades = pd.concat(dfs, ignore_index=True)
    trades = trades[trades["symbol"] == symbol].copy()
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    if "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
    return trades


def run(
    symbol: str,
    trades_glob: str,
    store: str,
    start: Optional[str],
    end: Optional[str],
    out_path: str,
    *,
    boll_period: int = 20,
    boll_std: float = 2.0,
    ols_window: int = 96,
    swing_window: int = 20,
    wide_window: int = 240,
    wide_shift: int = 12,
    feature_store_sub: bool = True,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bars = load_bars(store, symbol)
    if bars.empty:
        print(f"no bars for {symbol} under {store}")
        return
    if start:
        bars = bars.loc[bars.index >= pd.Timestamp(start)]
    if end:
        bars = bars.loc[bars.index <= pd.Timestamp(end)]
    if bars.empty:
        print("empty bar range after date filter")
        return

    close = pd.to_numeric(bars["close"], errors="coerce").astype(float)
    high = pd.to_numeric(bars["high"], errors="coerce").astype(float)
    low = pd.to_numeric(bars["low"], errors="coerce").astype(float)

    bb_up, bb_mid, bb_lo = bollinger(close, boll_period, boll_std)

    ols_mid, ols_width = _rolling_ols_channel(close, ols_window)
    ols_half = ols_width / 2.0
    ols_up = ols_mid + ols_half
    ols_lo = ols_mid - ols_half

    sw_hi, sw_lo = rolling_swing(high, low, swing_window, shift=0)
    wide_hi, wide_lo = rolling_swing(high, low, wide_window, shift=wide_shift)

    # Optional feature panel
    have_fr = "fer_range_pos_20" in bars.columns
    have_op = "fer_ols_pos" in bars.columns
    have_sf = "fer_sr_failed_breakout_score" in bars.columns
    n_extra = int(have_fr or have_op) + int(have_sf)

    rows = 1 + n_extra
    row_heights = (
        [0.68] + [(1 - 0.68) / max(n_extra, 1)] * n_extra if n_extra else [1.0]
    )
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="price",
            increasing_line_color="#d97575",
            decreasing_line_color="#7aa5d2",
            increasing_fillcolor="rgba(217,117,117,0.4)",
            decreasing_fillcolor="rgba(122,165,210,0.4)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    def line(y, name, color, dash=None, width=1):
        fig.add_trace(
            go.Scatter(
                x=bars.index,
                y=y,
                name=name,
                mode="lines",
                line=dict(color=color, dash=dash, width=width),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    line(bb_up, f"Boll({boll_period},{boll_std}) upper", "#e67e22", dash="dot")
    line(bb_lo, f"Boll({boll_period},{boll_std}) lower", "#e67e22", dash="dot")
    line(
        bb_mid,
        f"Boll({boll_period},{boll_std}) mid",
        "#e67e22",
        dash="dashdot",
        width=1,
    )

    line(ols_up, f"OLS({ols_window}) upper", "#9b59b6", dash="dash")
    line(ols_lo, f"OLS({ols_window}) lower", "#9b59b6", dash="dash")
    line(ols_mid, f"OLS({ols_window}) mid", "#9b59b6", width=1)

    line(sw_hi, f"swing H({swing_window})", "#27ae60", dash="dot")
    line(sw_lo, f"swing L({swing_window})", "#27ae60", dash="dot")

    line(wide_hi, f"wide SR H({wide_window}, shift={wide_shift})", "#2c3e50")
    line(wide_lo, f"wide SR L({wide_window}, shift={wide_shift})", "#2c3e50")

    trades = load_trades(trades_glob, symbol)
    if start:
        trades = trades[trades["entry_time"] >= pd.Timestamp(start)]
    if end:
        trades = trades[trades["entry_time"] <= pd.Timestamp(end)]
    print(f"{symbol}: bars={len(bars)}, trades={len(trades)}")

    def side_color(side: str, pnl: float) -> str:
        if pnl > 0:
            return "#0b8457" if str(side).upper() in ("LONG", "BUY") else "#b44a4a"
        return "#95a5a6"

    if len(trades):
        longs = trades[trades["side"].str.upper().isin(["LONG", "BUY"])]
        shorts = trades[trades["side"].str.upper().isin(["SHORT", "SELL"])]

        def _hover(row):
            return (
                f"{row['side']}  entry={row['entry_price']:.2f}<br>"
                f"{row['entry_time']}<br>"
                f"pnl_r={row.get('pnl_r', float('nan')):+.2f}"
                + (
                    f"  exit={row.get('exit_reason','')}"
                    if "exit_reason" in row
                    else ""
                )
            )

        if len(longs):
            fig.add_trace(
                go.Scatter(
                    x=longs["entry_time"],
                    y=longs["entry_price"],
                    mode="markers",
                    name=f"LONG ({len(longs)})",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=[side_color("LONG", p) for p in longs["pnl_r"].fillna(0)],
                        line=dict(width=1, color="black"),
                    ),
                    text=[_hover(r) for _, r in longs.iterrows()],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        if len(shorts):
            fig.add_trace(
                go.Scatter(
                    x=shorts["entry_time"],
                    y=shorts["entry_price"],
                    mode="markers",
                    name=f"SHORT ({len(shorts)})",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=[
                            side_color("SHORT", p) for p in shorts["pnl_r"].fillna(0)
                        ],
                        line=dict(width=1, color="black"),
                    ),
                    text=[_hover(r) for _, r in shorts.iterrows()],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

    next_row = 2
    if have_fr or have_op:
        if have_fr:
            fig.add_trace(
                go.Scatter(
                    x=bars.index,
                    y=bars["fer_range_pos_20"],
                    name="fer_range_pos_20",
                    mode="lines",
                    line=dict(color="#27ae60", width=1),
                ),
                row=next_row,
                col=1,
            )
        if have_op:
            fig.add_trace(
                go.Scatter(
                    x=bars.index,
                    y=bars["fer_ols_pos"],
                    name="fer_ols_pos",
                    mode="lines",
                    line=dict(color="#9b59b6", width=1),
                ),
                row=next_row,
                col=1,
            )
        fig.add_hline(y=0.85, line=dict(color="#555", dash="dot"), row=next_row, col=1)
        fig.add_hline(y=0.15, line=dict(color="#555", dash="dot"), row=next_row, col=1)
        fig.update_yaxes(
            title_text="pos [0,1]", row=next_row, col=1, range=[-0.05, 1.05]
        )
        next_row += 1
    if have_sf:
        fig.add_trace(
            go.Scatter(
                x=bars.index,
                y=bars["fer_sr_failed_breakout_score"],
                name="fer_sr_failed_breakout_score",
                mode="lines",
                line=dict(color="#2c3e50", width=1),
            ),
            row=next_row,
            col=1,
        )
        fig.add_hline(
            y=0.38, line=dict(color="#c0392b", dash="dot"), row=next_row, col=1
        )
        fig.update_yaxes(title_text="fer score", row=next_row, col=1)
        next_row += 1

    fig.update_layout(
        title=f"FBF boundaries · {symbol} · {start or 'all'} → {end or 'all'}",
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument(
        "--trades",
        required=True,
        help="glob to event_trades_*.csv (wildcard allowed)",
    )
    ap.add_argument(
        "--feature-store",
        required=True,
        help="feature_store/features_<strat>_120T_<hash>/",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boll-period", type=int, default=20)
    ap.add_argument("--boll-std", type=float, default=2.0)
    ap.add_argument("--ols-window", type=int, default=96)
    ap.add_argument("--swing-window", type=int, default=20)
    ap.add_argument("--wide-window", type=int, default=240)
    ap.add_argument("--wide-shift", type=int, default=12)
    args = ap.parse_args()
    run(
        args.symbol,
        args.trades,
        args.feature_store,
        args.start,
        args.end,
        args.out,
        boll_period=args.boll_period,
        boll_std=args.boll_std,
        ols_window=args.ols_window,
        swing_window=args.swing_window,
        wide_window=args.wide_window,
        wide_shift=args.wide_shift,
    )
