#!/usr/bin/env python3
"""BTC K-line + trend_scalp regime annotations (real data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    _hysteresis_segments,
    build_features,
    regime_chop_series,
    resolve_optional_repo_path,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.diagnose_dual_add_trend import (  # noqa: E402
    _add_trend_features,
    _load_dual_add_defaults,
)
from scripts.pipeline.multileg_prefilter_rules import apply_prefilter_rules  # noqa: E402

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml"
)


def _load_regime_thresholds(config_path: Path) -> dict:
    defaults = _load_dual_add_defaults(config_path)
    return {
        "trend_min": float(defaults.get("trend_min", 0.70)),
        "trend_exit_min": float(defaults.get("trend_exit_min", 0.40)),
        "exit_chop_min": float(defaults.get("exit_chop_min", 0.25)),
        "chop_min": float(defaults.get("chop_min", 0.40)),
        "min_segment_bars": int(defaults.get("min_segment_bars", 6)),
        "max_segment_bars": int(defaults.get("max_segment_bars", 120)),
        "exclude_box": bool(defaults.get("exclude_box", True)),
        "prefilter_rules": tuple(
            x for x in (defaults.get("prefilter_rules", []) or []) if isinstance(x, dict)
        ),
        "box_window": int(defaults.get("box_window", 120)),
        "stability_min": float(defaults.get("stability_min", 0.85)),
        "width_min": float(defaults.get("width_min", 0.04)),
        "width_max": float(defaults.get("width_max", 0.30)),
        "touches_min": int(defaults.get("touches_min", 5)),
        "chop_signal": str(defaults.get("chop_signal", "raw")),
        "chop_ts_window": int(defaults.get("chop_ts_window", 1200)),
        "chop_ts_min_periods": int(defaults.get("chop_ts_min_periods", 150)),
        "compute_chop_ts_q": defaults.get("compute_chop_ts_q"),
        "feature_store_dir": resolve_optional_repo_path(defaults.get("feature_store_dir")),
        "feature_store_layer": defaults.get("feature_store_layer"),
        "feature_store_timeframe": defaults.get("feature_store_timeframe"),
    }


def _build_signal_frame(
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: Path,
    cfg: dict,
    timeframe: str,
    warmup_days: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[Tuple[int, int]]]:
    warmup_start = start - pd.Timedelta(days=warmup_days)
    raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
    if raw.empty:
        raise RuntimeError(f"No 1m data for {symbol} in [{warmup_start}, {end}]")
    bars = _resample_ohlcv(raw, timeframe)
    grid_cfg = GridConfig(
        box_window=cfg["box_window"],
        chop_min=cfg["chop_min"],
        exit_chop_min=cfg["exit_chop_min"],
        chop_signal=cfg["chop_signal"],
        chop_ts_window=cfg["chop_ts_window"],
        chop_ts_min_periods=cfg["chop_ts_min_periods"],
        compute_semantic_chop_ts_q=cfg["compute_chop_ts_q"],
        stability_min=cfg["stability_min"],
        width_min=cfg["width_min"],
        width_max=cfg["width_max"],
        touches_min=cfg["touches_min"],
        feature_store_dir=cfg["feature_store_dir"],
        feature_store_layer=cfg["feature_store_layer"],
        feature_store_timeframe=cfg["feature_store_timeframe"],
    )
    df = build_features(symbol, bars, grid_cfg, bars_timeframe=timeframe)
    df = _add_trend_features(df)
    df = df[(df.index >= start) & (df.index <= end)].copy()
    chop_s = regime_chop_series(df, grid_cfg)
    entry = (df["trend_confidence"] >= cfg["trend_min"]) & (chop_s <= cfg["exit_chop_min"])
    hold = (df["trend_confidence"] >= cfg["trend_exit_min"]) & (chop_s <= cfg["chop_min"])
    rule_mask = apply_prefilter_rules(
        df,
        list(cfg["prefilter_rules"]),
        feature_aliases={
            "atr": "atr14",
            "bpc_semantic_chop": "semantic_chop",
            "bpc_semantic_chop_ts_q": "semantic_chop_ts_q",
        },
    )
    entry &= rule_mask
    hold &= rule_mask
    if cfg["exclude_box"] and "box_prefilter" in df.columns:
        entry &= ~df["box_prefilter"].astype(bool)
        hold &= ~df["box_prefilter"].astype(bool)
    segs = _hysteresis_segments(
        entry,
        hold,
        min_len=cfg["min_segment_bars"],
        max_len=cfg["max_segment_bars"],
    )
    return df, chop_s, entry, hold, segs


def _contiguous_true_spans(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if mask.empty:
        return spans
    vals = mask.fillna(False).to_numpy(dtype=bool)
    idx = mask.index
    i = 0
    n = len(vals)
    while i < n:
        if not vals[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and vals[j + 1]:
            j += 1
        spans.append((idx[i], idx[j]))
        i = j + 1
    return spans


def _pick_callouts(
    df: pd.DataFrame,
    chop_s: pd.Series,
    entry: pd.Series,
    segs: List[Tuple[int, int]],
    cfg: dict,
) -> List[dict]:
    """Pick a few real bars to label for teaching."""
    out: List[dict] = []
    if segs:
        s, _e = segs[0]
        row = df.iloc[s]
        out.append(
            {
                "x": df.index[s],
                "y": float(row["high"]) * 1.02,
                "text": (
                    f"段首 #{1}<br>"
                    f"conf={row['trend_confidence']:.2f}≥{cfg['trend_min']:.2f}<br>"
                    f"chop={chop_s.iloc[s]:.2f}≤{cfg['exit_chop_min']:.2f}<br>"
                    f"box={bool(row.get('box_prefilter', False))}<br>"
                    f"→ {row.get('trend_direction', '?')}"
                ),
                "color": "#2e7d32",
            }
        )
    # High chop blocks entry despite decent trend
    cand = df[(df["trend_confidence"] >= cfg["trend_min"]) & (chop_s > cfg["exit_chop_min"])]
    if not cand.empty:
        ts = cand.index[len(cand) // 3]
        i = df.index.get_loc(ts)
        row = df.loc[ts]
        out.append(
            {
                "x": ts,
                "y": float(row["high"]) * 1.03,
                "text": (
                    "chop 过高<br>"
                    f"conf={row['trend_confidence']:.2f} OK<br>"
                    f"chop={chop_s.loc[ts]:.2f}>{cfg['exit_chop_min']:.2f}<br>"
                    "✗ 不开 trend"
                ),
                "color": "#c62828",
            }
        )
    # Stable box
    if "box_prefilter" in df.columns:
        box = df[df["box_prefilter"].astype(bool)]
        if not box.empty:
            ts = box.index[len(box) // 2]
            row = df.loc[ts]
            out.append(
                {
                    "x": ts,
                    "y": float(row["low"]) * 0.97,
                    "text": (
                        "稳定箱体<br>"
                        "box_prefilter=true<br>"
                        "✗ trend 禁止新开"
                    ),
                    "color": "#6a1b9a",
                }
            )
    # Entry-ready but between segments (all three pass, not in segment)
    ready = (
        (df["trend_confidence"] >= cfg["trend_min"])
        & (chop_s <= cfg["exit_chop_min"])
    )
    if "box_prefilter" in df.columns:
        ready &= ~df["box_prefilter"].astype(bool)
    in_seg = pd.Series(False, index=df.index)
    for s, e in segs:
        in_seg.iloc[s : e + 1] = True
    gap = df[ready & ~in_seg]
    if not gap.empty:
        ts = gap.index[0]
        row = df.loc[ts]
        out.append(
            {
                "x": ts,
                "y": float(row["close"]),
                "text": (
                    "三条件过关<br>"
                    "但不在段内<br>"
                    "(迟滞/段长约束)"
                ),
                "color": "#1565c0",
            }
        )
    return out[:4]


def build_figure(
    df: pd.DataFrame,
    chop_s: pd.Series,
    entry: pd.Series,
    segs: List[Tuple[int, int]],
    cfg: dict,
    *,
    symbol: str,
    year: int,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(
            f"{symbol} {year} 2h K线 — 浅色竖条=trend段(绿↑/红↓)，紫色=稳定箱体",
            f"trend_confidence（入场≥{cfg['trend_min']:.2f}，持仓≥{cfg['trend_exit_min']:.2f}）",
            f"semantic_chop（入场≤{cfg['exit_chop_min']:.2f}，持仓≤{cfg['chop_min']:.2f}）",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="K线",
        ),
        row=1,
        col=1,
    )

    # Box prefilter shading (purple, behind segments)
    if "box_prefilter" in df.columns:
        for x0, x1 in _contiguous_true_spans(df["box_prefilter"].astype(bool)):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(156,39,176,0.12)",
                line_width=0,
                row=1,
                col=1,
            )

    # Trend segments
    for s, e in segs:
        direction = str(df["trend_direction"].iloc[s])
        color = (
            "rgba(46,125,50,0.18)" if direction == "UP" else "rgba(198,40,40,0.18)"
        )
        fig.add_vrect(
            x0=df.index[s],
            x1=df.index[e],
            fillcolor=color,
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index[s]],
                y=[float(df["close"].iloc[s])],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up" if direction == "UP" else "triangle-down",
                    size=10,
                    color="#1565c0" if direction == "UP" else "#7b1fa2",
                ),
                text=[direction],
                textposition="top center",
                showlegend=False,
                hovertext=f"段首 {df.index[s]} {direction}",
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    conf = pd.to_numeric(df["trend_confidence"], errors="coerce")
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=conf,
            mode="lines",
            line=dict(color="#1565c0", width=1),
            name="trend_confidence",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=cfg["trend_min"],
        line=dict(color="#2e7d32", dash="dash"),
        annotation_text="入场 0.70",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=cfg["trend_exit_min"],
        line=dict(color="#9e9e9e", dash="dot"),
        annotation_text="持仓 0.40",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=chop_s,
            mode="lines",
            line=dict(color="#ef6c00", width=1),
            name="semantic_chop",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=cfg["exit_chop_min"],
        line=dict(color="#2e7d32", dash="dash"),
        annotation_text="入场 0.25",
        row=3,
        col=1,
    )
    fig.add_hline(
        y=cfg["chop_min"],
        line=dict(color="#9e9e9e", dash="dot"),
        annotation_text="持仓 0.40",
        row=3,
        col=1,
    )

    for call in _pick_callouts(df, chop_s, entry, segs, cfg):
        fig.add_annotation(
            x=call["x"],
            y=call["y"],
            text=call["text"],
            showarrow=True,
            arrowhead=2,
            arrowcolor=call["color"],
            font=dict(size=10, color=call["color"]),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=call["color"],
            borderwidth=1,
            row=1,
            col=1,
        )

    n_seg = len(segs)
    fig.update_layout(
        title=dict(
            text=(
                f"{symbol} {year} trend_scalp 三条件对照 "
                f"（{n_seg} 个 trend 段，真实 2h 数据）"
            ),
            x=0.5,
        ),
        height=1000,
        width=1400,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="价格 USDT", row=1, col=1)
    fig.update_yaxes(title_text="confidence", row=2, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="chop", row=3, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="时间 (UTC)", row=3, col=1)

    # Legend notes as annotation
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=-0.02,
        showarrow=False,
        align="left",
        font=dict(size=11),
        text=(
            "图例：浅绿/浅红竖条=trend 段(UP/DOWN) | 紫色=box_prefilter 稳定箱体 | "
            "△段首方向 | 下方曲线同步显示 confidence/chop 与阈值线"
        ),
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--timeframe", default="2h")
    ap.add_argument("--warmup-days", type=int, default=90)
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: config/strategies/trend_scalp/{symbol}_{year}_regime_annotated.html",
    )
    args = ap.parse_args()

    start = pd.Timestamp(f"{args.year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{args.year}-12-31 23:59:59", tz="UTC")
    cfg = _load_regime_thresholds(Path(args.config))
    if not Path(args.config).is_absolute():
        cfg_path = PROJECT_ROOT / args.config
    else:
        cfg_path = Path(args.config)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    print(f"Loading {args.symbol} {args.year} …")
    df, chop_s, entry, _hold, segs = _build_signal_frame(
        symbol=args.symbol.upper(),
        start=start,
        end=end,
        data_dir=data_dir,
        cfg=cfg,
        timeframe=args.timeframe,
        warmup_days=args.warmup_days,
    )
    print(f"bars={len(df)} segments={len(segs)}")

    fig = build_figure(
        df, chop_s, entry, segs, cfg, symbol=args.symbol.upper(), year=args.year
    )

    out = args.out
    if out is None:
        out = (
            PROJECT_ROOT
            / f"config/strategies/trend_scalp/{args.symbol.upper()}_{args.year}_regime_annotated.html"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Wrote {out.resolve()}")

    results_out = (
        PROJECT_ROOT / f"results/trend_scalp/{args.symbol.upper()}_{args.year}_regime_annotated.html"
    )
    try:
        results_out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(results_out), include_plotlyjs="cdn")
        print(f"Wrote {results_out.resolve()}")
    except OSError as exc:
        print(f"(skip results copy: {exc})")


if __name__ == "__main__":
    main()
