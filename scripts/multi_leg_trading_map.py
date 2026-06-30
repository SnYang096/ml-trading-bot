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
    initial_capital: float | None = None,
) -> None:
    """Write a single multi-symbol continuous map with candle and trade overlays.

    Layout (SRB trading-map style, top → bottom):
      1. Summary metrics + per-symbol comparison table
      2. Portfolio equity (USDT) + drawdown
      3. One price panel per symbol
    """
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
    n_sym = max(
        1,
        int(trades["symbol"].nunique()) if not trades.empty and "symbol" in trades.columns else 1,
    )
    cap = float(initial_capital if initial_capital is not None else 10_000.0 * n_sym)
    summary_html = _summary_html(
        trades=trades,
        start=x_start,
        end=x_end,
        width=1300,
        title=title,
        initial_capital=cap,
    )
    figs = []
    eq_fig = _build_portfolio_equity_figure(trades, initial_capital=cap)
    if eq_fig is not None:
        figs.append(eq_fig)

    price_figs = []

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
        price_figs.append(p)

    figs.extend(price_figs)
    if not price_figs:
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
    initial_capital: float = 10_000.0,
) -> str:
    total_trades = int(len(trades)) if not trades.empty else 0
    pnl_col = "pnl_per_capital" if "pnl_per_capital" in trades.columns else "pnl_r"
    win_rate = (
        float((pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0) > 0).mean())
        if total_trades > 0 and pnl_col in trades.columns
        else 0.0
    )
    months = f"{start.strftime('%Y-%m')}~{end.strftime('%Y-%m')}"
    per_sym_html = _per_symbol_table_html(trades, initial_capital=initial_capital)
    port = _portfolio_headline(trades, initial_capital=initial_capital)
    return (
        f"<h2>{title}</h2>"
        f"<p><b>区间</b> {months} | <b>成交</b> {total_trades} | "
        f"<b>组合收益</b> {port['return_pct']:.2f}% | "
        f"<b>期末权益</b> ${port['final_capital']:,.0f} | "
        f"<b>最大回撤</b> {port['max_dd_pct']:.2f}% | "
        f"<b>胜率</b> {win_rate:.2%}</p>"
        f"{per_sym_html}"
        f"<p style='font-size:13px;line-height:1.45;max-width:{width}px'>"
        "<b>上方资金图</b> = 多币等权组合权益（每笔平仓后更新，口径同 capital_report）。"
        "<b>图例（价格图）</b> 叠在图内左上角。品红实线 = 各 symbol <b>自身</b> 2H K 线上"
        "滚动典型价 VWAP（1200 根 bar，仅价格尺度展示）；橙线 = 同周期 <b>EMA(1200)</b> on close。"
        "Multi-leg 策略不走旧 <code>TradeIntent/event_backtest</code>，因此无 PCM 漏斗附图。"
        "绿/红 K 线为涨跌。入场→出场：<b>实线</b>=首仓腿，<b>虚线</b>=加仓腿；"
        "△ 多 · ▽ 空 · ◇ 加仓 · □ 平仓。全窗口标记密集时请框选 zoom。"
        "</p>"
    )


def _portfolio_headline(
    trades: pd.DataFrame, *, initial_capital: float
) -> dict[str, float]:
    from scripts.pipeline.multileg_portfolio_metrics import build_portfolio_equity_curve

    if trades.empty or "pnl_per_capital" not in trades.columns:
        return {
            "final_capital": float(initial_capital),
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
        }
    curve = build_portfolio_equity_curve(trades)
    if curve.empty:
        return {
            "final_capital": float(initial_capital),
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
        }
    final_pc = float(curve["cum_pnl_per_capital"].iloc[-1])
    final_cap = float(initial_capital) * (1.0 + final_pc)
    dd = float(curve["drawdown"].min()) if "drawdown" in curve.columns else 0.0
    return {
        "final_capital": final_cap,
        "return_pct": final_pc * 100.0,
        "max_dd_pct": abs(dd) * 100.0,
    }


def _per_symbol_table_html(
    trades: pd.DataFrame, *, initial_capital: float
) -> str:
    if trades.empty or "symbol" not in trades.columns:
        return ""
    pc = pd.to_numeric(trades.get("pnl_per_capital"), errors="coerce").fillna(0.0)
    n = max(1, int(trades["symbol"].nunique()))
    bucket = float(initial_capital) / n
    rows = []
    for sym, grp in trades.assign(_pc=pc).groupby("symbol", sort=True):
        ret = float(grp["_pc"].sum()) * 100.0
        profit = bucket * float(grp["_pc"].sum())
        wr = float((grp["_pc"] > 0).mean()) * 100.0
        rows.append(
            f"<tr><td>{sym}</td><td>{len(grp)}</td>"
            f"<td>{ret:.2f}%</td><td>${profit:,.0f}</td><td>{wr:.1f}%</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h3>分币种对比（等权资金桶）</h3>"
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px'>"
        "<tr><th>Symbol</th><th>Trades</th><th>Return%</th>"
        "<th>Profit $</th><th>Win%</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _build_portfolio_equity_figure(
    trades: pd.DataFrame, *, initial_capital: float
):
    """Bokeh equity + drawdown panel (aligned with event_backtest trading map)."""
    if trades.empty or "pnl_per_capital" not in trades.columns:
        return None
    try:
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.plotting import figure
    except Exception:
        return None

    from scripts.pipeline.multileg_portfolio_metrics import build_portfolio_equity_curve

    curve = build_portfolio_equity_curve(trades)
    if curve.empty or len(curve) < 2:
        return None

    t = pd.to_datetime(curve["exit_time"], utc=True)
    eq = float(initial_capital) * (1.0 + curve["cum_pnl_per_capital"].astype(float))
    peak = eq.cummax()
    dd_pct = (eq - peak) / peak.replace(0, pd.NA) * 100.0

    # Downsample for browser performance on 10k+ trades
    if len(eq) > 800:
        step = max(1, len(eq) // 800)
        idx = list(range(0, len(eq), step))
        if idx[-1] != len(eq) - 1:
            idx.append(len(eq) - 1)
        t = t.iloc[idx]
        eq = eq.iloc[idx]
        dd_pct = dd_pct.iloc[idx]

    src = ColumnDataSource({"t": t, "equity": eq, "dd_pct": dd_pct})
    p = figure(
        title="组合权益曲线（USDT，多币等权；每笔平仓后更新）",
        x_axis_type="datetime",
        width=1300,
        height=260,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.line("t", "equity", source=src, line_width=2, color="#2563eb")
    p.add_tools(
        HoverTool(
            tooltips=[
                ("Time", "@t{%F %H:%M}"),
                ("Equity ($)", "@equity{0,0.0}"),
                ("Drawdown%", "@dd_pct{0.2f}"),
            ],
            formatters={"@t": "datetime"},
        )
    )
    p.yaxis.axis_label = "Equity (USDT)"
    p.grid.grid_line_alpha = 0.25
    return p


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
