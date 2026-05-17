"""Shared Bokeh trading maps for standalone multi-leg backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv


def _resolve_symbols(value: str, fallback: str) -> list[str]:
    raw = value if value.strip() else fallback
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def write_continuous_trading_map(
    *,
    out_path: Path,
    data_dir: Path,
    symbols: str,
    map_symbols: str,
    timeframe: str,
    start: str,
    end: str,
    warmup_days: int,
    trades: pd.DataFrame,
    segments: pd.DataFrame,
    title: str,
) -> None:
    """Write a single multi-symbol continuous map with candle and trade overlays."""
    if trades.empty and segments.empty:
        return
    try:
        from bokeh.io import output_file, save
        from bokeh.layouts import column
        from bokeh.models import BoxAnnotation, ColumnDataSource, Div, HoverTool
        from bokeh.plotting import figure
    except Exception as exc:  # pragma: no cover - optional report dependency
        print(f"skip continuous trading map: bokeh unavailable ({exc})")
        return

    selected_symbols = _resolve_symbols(map_symbols, symbols)
    if not selected_symbols:
        return

    x_start = pd.Timestamp(start, tz="UTC")
    x_end = pd.Timestamp(end, tz="UTC")
    warmup_start = x_start - pd.Timedelta(days=warmup_days)
    bar_width_ms = pd.Timedelta(timeframe).total_seconds() * 1000 * 0.72
    summary_html = _summary_html(
        trades=trades,
        start=x_start,
        end=x_end,
        width=1300,
        title=title,
    )
    figs = []

    for symbol in selected_symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, x_end)
        if raw.empty:
            continue
        bars = _resample_ohlcv(raw, timeframe)
        df = bars[(bars.index >= x_start) & (bars.index <= x_end)].copy()
        if df.empty:
            continue
        _ema_n = 1200
        df["tp_vwap_1200"] = _rolling_tp_vwap(df, 1200)
        df["ema_1200"] = df["close"].ewm(span=_ema_n, adjust=False).mean()
        df = df.reset_index(names="time")
        inc = df["close"] >= df["open"]
        dec = ~inc
        p = figure(
            x_axis_type="datetime",
            width=1300,
            height=420,
            title=f"{title} - {symbol}",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.segment(
            df["time"], df["high"], df["time"], df["low"], color="#6b7280", alpha=0.6
        )
        p.vbar(
            df.loc[inc, "time"],
            bar_width_ms,
            df.loc[inc, "open"],
            df.loc[inc, "close"],
            fill_color="#16a34a",
            line_color="#16a34a",
            alpha=0.62,
        )
        p.vbar(
            df.loc[dec, "time"],
            bar_width_ms,
            df.loc[dec, "open"],
            df.loc[dec, "close"],
            fill_color="#dc2626",
            line_color="#dc2626",
            alpha=0.62,
        )
        p.line(
            df["time"],
            df["tp_vwap_1200"],
            line_color="#c026d3",
            line_width=1.35,
            line_alpha=0.78,
            legend_label="Rolling TP-VWAP (1200x2H, local symbol price)",
        )
        p.line(
            df["time"],
            df["ema_1200"],
            line_color="#ea580c",
            line_width=1.25,
            line_alpha=0.82,
            legend_label=f"EMA({_ema_n}) on close (local symbol price)",
        )

        _add_segment_boxes(p, segments, symbol, x_start, x_end)
        _add_trade_overlays(p, trades, symbol, x_start, x_end)
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price"
        figs.append(p)

    if not figs:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_file(out_path, title=title)
    save(column(Div(text=summary_html, width=1300), *figs, sizing_mode="stretch_width"))
    print(f"Saved continuous trading map -> {out_path}")


def _rolling_tp_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df.get("volume", pd.Series(1.0, index=df.index)).fillna(0.0)
    denom = volume.rolling(window, min_periods=1).sum().replace(0.0, pd.NA)
    return (typical * volume).rolling(window, min_periods=1).sum() / denom


def _summary_html(
    *,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    width: int,
    title: str,
) -> str:
    total_trades = int(len(trades)) if not trades.empty else 0
    pnl_usd = 0.0
    if total_trades > 0 and "pnl_usd_realized" in trades.columns:
        pnl_usd = float(
            pd.to_numeric(trades["pnl_usd_realized"], errors="coerce").fillna(0.0).sum()
        )
    pnl_col = "pnl_per_capital" if "pnl_per_capital" in trades.columns else "pnl_r"
    total_r = (
        float(pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0).sum())
        if total_trades > 0 and pnl_col in trades.columns
        else 0.0
    )
    win_rate = (
        float((pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0) > 0).mean())
        if total_trades > 0 and pnl_col in trades.columns
        else 0.0
    )
    if "strategy" in trades.columns and not trades.empty:
        source = ", ".join(
            f"multi_leg:{k}:{int(v)}"
            for k, v in trades["strategy"].value_counts().items()
        )
    else:
        source = f"multi_leg:{total_trades}"
    months = f"{start.strftime('%Y-%m')}~{end.strftime('%Y-%m')}"
    return (
        f"<h2>{title}</h2>"
        f"<p>months={months} | trades={total_trades} | total_r={total_r:.4f} "
        f"| realized_pnl_usd={pnl_usd:,.2f} | win_rate={win_rate:.2%} | source=({source})</p>"
        f"<p style='font-size:13px;line-height:1.45;max-width:{width}px'>"
        "<b>图例（价格图）</b> 叠在图内左上角。品红实线 = 各 symbol <b>自身</b> 2H K 线上"
        "滚动典型价 VWAP（1200 根 bar，仅价格尺度展示）；橙线 = 同周期 <b>EMA(1200)</b> on close。"
        "Multi-leg 策略不走旧 <code>TradeIntent/event_backtest</code>，因此本连续图未拼接"
        "PCM/prefilter/gate 阶梯漏斗附图；这些策略的阈值/结构选择来自 rolling calibration window "
        "写出的 <code>strategies_calibrated</code>。"
        "绿/红 K 线为涨跌。入场→出场：<b>实线</b>=首仓腿，<b>虚线</b>=加仓腿；"
        "△ 多 · ▽ 空 · ◇ 加仓 · □ 平仓。"
        "</p>"
    )


def _add_segment_boxes(
    plot,
    segments: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    from bokeh.models import BoxAnnotation

    if segments.empty or "symbol" not in segments.columns:
        return
    sseg = segments[segments["symbol"] == symbol].copy()
    if sseg.empty or "start" not in sseg.columns or "end" not in sseg.columns:
        return
    sseg["start"] = pd.to_datetime(sseg["start"], utc=True)
    sseg["end"] = pd.to_datetime(sseg["end"], utc=True)
    sseg = sseg[(sseg["end"] >= start) & (sseg["start"] <= end)]
    for _, row in sseg.iterrows():
        fill_color = _segment_color(row)
        plot.add_layout(
            BoxAnnotation(
                left=row["start"],
                right=row["end"],
                fill_color=fill_color,
                fill_alpha=0.07,
                line_alpha=0.0,
            )
        )


def _segment_color(row: pd.Series) -> str:
    direction = str(row.get("direction", "") or "").upper()
    if direction == "UP":
        return "#22c55e"
    if direction == "DOWN":
        return "#ef4444"
    return "#22c55e"


def _add_trade_overlays(
    plot,
    trades: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    from bokeh.models import ColumnDataSource, HoverTool

    required = {
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
    }
    if trades.empty or not required.issubset(set(trades.columns)):
        return
    strades = trades[trades["symbol"] == symbol].copy()
    if strades.empty:
        return
    strades["entry_time"] = pd.to_datetime(strades["entry_time"], utc=True)
    strades["exit_time"] = pd.to_datetime(strades["exit_time"], utc=True)
    strades = strades[(strades["exit_time"] >= start) & (strades["entry_time"] <= end)]
    if strades.empty:
        return

    for side, color, marker in [
        ("LONG", "#2563eb", "triangle"),
        ("SHORT", "#9333ea", "inverted_triangle"),
    ]:
        src_df = strades[strades["side"] == side].copy()
        if src_df.empty:
            continue
        src_df = _ensure_hover_columns(src_df, ["seq", "level"])
        src_df["_is_add"] = _is_add_leg(src_df)
        renderers = []
        for is_add, dash, entry_marker, label_suffix in [
            (False, "solid", marker, "primary"),
            (True, "dashed", "diamond", "add"),
        ]:
            leg_df = src_df[src_df["_is_add"] == is_add]
            if leg_df.empty:
                continue
            src = ColumnDataSource(leg_df)
            plot.segment(
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                source=src,
                color=color,
                alpha=0.40,
                line_width=1.5,
                line_dash=dash,
                legend_label=f"{side} {label_suffix}",
            )
            glyph = getattr(plot, entry_marker)
            r = glyph(
                "entry_time",
                "entry_price",
                source=src,
                size=8 if is_add else 7,
                color=color,
                alpha=0.9,
                legend_label=f"{side} entry {label_suffix}",
            )
            renderers.append(r)
        plot.add_tools(
            HoverTool(
                renderers=renderers,
                tooltips=[
                    ("side", "@side"),
                    ("seq", "@seq"),
                    ("level", "@level"),
                    ("entry", "@entry_price{0.0000}"),
                    ("exit", "@exit_price{0.0000}"),
                    ("pnl", "@pnl_pct{0.0000}"),
                    ("reason", "@exit_reason"),
                ],
            )
        )

    exit_src = ColumnDataSource(_ensure_hover_columns(strades, ["seq", "level"]))
    plot.circle(
        "exit_time",
        "exit_price",
        source=exit_src,
        size=4,
        color="#111827",
        alpha=0.50,
        legend_label="exit",
    )
    plot.square(
        "exit_time",
        "exit_price",
        source=exit_src,
        size=6,
        color="#111827",
        alpha=0.65,
        legend_label="exit square",
    )


def _is_add_leg(df: pd.DataFrame) -> pd.Series:
    seq = pd.to_numeric(df.get("seq", 0), errors="coerce").fillna(0.0)
    level = pd.to_numeric(df.get("level", 0), errors="coerce").fillna(0.0)
    if "seq" in df.columns:
        return seq > 0
    if "level" in df.columns:
        return level > 1
    return pd.Series(False, index=df.index)


def _ensure_hover_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out
