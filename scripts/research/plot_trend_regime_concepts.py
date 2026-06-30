#!/usr/bin/env python3
"""Annotated K-line teaching chart for trend_scalp regime concepts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ohlc_from_close(
    close: np.ndarray,
    *,
    vol: float = 0.004,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0] * 0.998
    open_[1:] = close[:-1]
    wiggle = rng.normal(0, vol, n) * close
    high = np.maximum.reduce([open_, close, close + np.abs(wiggle)])
    low = np.minimum.reduce([open_, close, close - np.abs(wiggle)])
    idx = pd.date_range("2024-01-01", periods=n, freq="2h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def _scenario_trend_confidence() -> pd.DataFrame:
    """Multi-horizon aligned uptrend → high trend_confidence."""
    n = 24
    t = np.arange(n)
    close = 100 + t * 1.2 + np.sin(t / 3) * 0.3
    return _ohlc_from_close(close, vol=0.003, seed=1)


def _scenario_semantic_chop() -> pd.DataFrame:
    """Narrow range + direction flip → high semantic_chop."""
    n = 24
    t = np.arange(n)
    close = 100 + np.sin(t * 0.9) * 1.5
    return _ohlc_from_close(close, vol=0.006, seed=2)


def _scenario_box_prefilter() -> pd.DataFrame:
    """Stable box with repeated hi/lo touches → box_prefilter true."""
    n = 30
    t = np.arange(n)
    # Sawtooth inside band: touch top and bottom repeatedly
    phase = t % 6
    close = np.where(
        phase < 3,
        100 + phase * 0.8,
        102.4 - (phase - 3) * 0.8,
    )
    close = close + np.sin(t * 0.5) * 0.15
    return _ohlc_from_close(close, vol=0.0025, seed=3)


def _add_candles(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            name="K线",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _hline(
    fig: go.Figure,
    y: float,
    *,
    row: int,
    color: str,
    dash: str = "dash",
    width: int = 1,
) -> None:
    fig.add_hline(
        y=y,
        line=dict(color=color, dash=dash, width=width),
        row=row,
        col=1,
    )


def build_figure() -> go.Figure:
    df_trend = _scenario_trend_confidence()
    df_chop = _scenario_semantic_chop()
    df_box = _scenario_box_prefilter()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            "① trend_confidence 高（≥0.70）— 多周期同向上涨",
            "② semantic_chop 高（>0.25）— 窄幅来回扫，方向打架",
            "③ box_prefilter = true — 稳定箱体（trend 禁止新开段）",
        ),
    )

    _add_candles(fig, df_trend, row=1)
    _add_candles(fig, df_chop, row=2)
    _add_candles(fig, df_box, row=3)

    # Panel 1 annotations: aligned horizons
    x1 = df_trend.index
    fig.add_annotation(
        x=x1[4], y=df_trend["close"].iloc[4], text="3根↑", showarrow=True,
        arrowhead=2, ax=40, ay=-30, font=dict(size=11, color="#1565c0"),
        bgcolor="rgba(255,255,255,0.85)", row=1, col=1,
    )
    fig.add_annotation(
        x=x1[10], y=df_trend["close"].iloc[10], text="5根↑", showarrow=True,
        arrowhead=2, ax=50, ay=-35, font=dict(size=11, color="#1565c0"),
        bgcolor="rgba(255,255,255,0.85)", row=1, col=1,
    )
    fig.add_annotation(
        x=x1[18], y=df_trend["close"].iloc[18], text="10根↑<br>方向一致",
        showarrow=True, arrowhead=2, ax=-60, ay=-40,
        font=dict(size=11, color="#1565c0"), bgcolor="rgba(255,255,255,0.85)",
        row=1, col=1,
    )
    fig.add_annotation(
        x=x1[-2], y=df_trend["close"].iloc[-2] + 2,
        text="✓ 可开 trend 段（若 chop 低 & 非箱体）",
        showarrow=False, font=dict(size=12, color="#2e7d32"),
        bgcolor="rgba(200,230,201,0.9)", row=1, col=1,
    )

    # Panel 2: chop zone
    mid = df_chop["close"].mean()
    _hline(fig, mid + 1.2, row=2, color="#9e9e9e", dash="dot")
    _hline(fig, mid - 1.2, row=2, color="#9e9e9e", dash="dot")
    fig.add_annotation(
        x=df_chop.index[6], y=mid + 2.5,
        text="BB 收窄 + 涨跌交替<br>semantic_chop 高",
        showarrow=False, font=dict(size=11, color="#c62828"),
        bgcolor="rgba(255,205,210,0.9)", row=2, col=1,
    )
    fig.add_annotation(
        x=df_chop.index[15], y=mid - 2.8,
        text="✗ trend 不开新段<br>→ 留给 chop_grid 区间",
        showarrow=False, font=dict(size=11, color="#c62828"),
        bgcolor="rgba(255,205,210,0.9)", row=2, col=1,
    )

    # Panel 3: box
    hi = float(df_box["high"].max())
    lo = float(df_box["low"].min())
    _hline(fig, hi, row=3, color="#7b1fa2", dash="solid", width=2)
    _hline(fig, lo, row=3, color="#7b1fa2", dash="solid", width=2)
    fig.add_annotation(
        x=df_box.index[3], y=hi + 0.3, text="上沿 touches≥5",
        showarrow=False, font=dict(size=10, color="#7b1fa2"), row=3, col=1,
    )
    fig.add_annotation(
        x=df_box.index[20], y=lo - 0.5, text="下沿 touches≥5",
        showarrow=False, font=dict(size=10, color="#7b1fa2"), row=3, col=1,
    )
    fig.add_annotation(
        x=df_box.index[14], y=(hi + lo) / 2,
        text="stability≥0.85<br>width 4%~30%<br>box_prefilter=true",
        showarrow=False, font=dict(size=11, color="#6a1b9a"),
        bgcolor="rgba(225,190,231,0.9)", row=3, col=1,
    )
    fig.add_annotation(
        x=df_box.index[-2], y=hi + 0.8,
        text="✗ trend 禁止新开（排除稳定箱体）",
        showarrow=False, font=dict(size=12, color="#6a1b9a"),
        bgcolor="rgba(225,190,231,0.9)", row=3, col=1,
    )

    fig.update_layout(
        title=dict(
            text="trend_scalp 开仓三条件 — K 线示意图（合成数据，仅供理解）",
            x=0.5,
            font=dict(size=16),
        ),
        height=1100,
        width=1000,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        margin=dict(t=100, b=40),
    )
    for i in range(1, 4):
        fig.update_yaxes(title_text="价格", row=i, col=1)
        fig.update_xaxes(title_text="时间（示意 2h K）", row=i, col=1)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/trend_scalp/regime_concepts_annotated.html"),
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
