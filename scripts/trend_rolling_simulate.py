#!/usr/bin/env python3
"""趋势滚仓模拟器 v3 — 冠军入场 + 可重入 + 交易地图

Entry presets (--entry):
  champion (默认): 周线EMA200下方5% + EMA1200金叉VWAP + 短中期动量向上
  base:           周线EMA200下方 + EMA1200金叉VWAP
  deep:           周线EMA200下方10% + EMA1200金叉VWAP
  compression:    ATR低位 + 价格突破EMA1200
  custom:         自定义weekly_pos_lt + ema_cross + momentum

Leverage: 2x→3x (价格从峰值跌20%且浮盈>0)
Risk: 权益DD≥50%减半仓 + 价格DD≥60%减半仓 (28天冷却)
Exit: 权益≥5x阶梯越长越卖, ≥15x可清仓
Re-entry: 清仓后可重新入场 (60天冷却)

用法:
  python scripts/trend_rolling_simulate.py --entry champion --plot
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

ENTRY_PRESETS = {
    "champion": {
        "desc": "冠军: W-EMA200下方5%+EMA1200金叉VWAP+动量向上",
        "weekly_pos_lt": -0.05,
        "ema_cross": True,
        "momentum": True,
    },
    "r2": {
        "desc": "R2 VWAP滚仓: 冠军入场+VWAP1200附近滚仓(回撤-45%)",
        "weekly_pos_lt": -0.05,
        "ema_cross": True,
        "momentum": True,
        "roll_near_vwap": True,
    },
    "r5": {
        "desc": "R5 高杠杆: 2x→5x步进+仓位上限+紧风控",
        "weekly_pos_lt": -0.05,
        "ema_cross": True,
        "momentum": True,
        "roll_near_vwap": True,
        "max_leverage": 5.0,
        "max_position_notional": 500000,
        "eq_dd_stop": 0.35,
        "ladder_base_frac": 0.05,
    },
    "r6": {
        "desc": "R6 保守: 冠军入场+3x卖出(回撤-28%)",
        "weekly_pos_lt": -0.05,
        "ema_cross": True,
        "momentum": True,
        "ladder_trigger": 3.0,
    },
    "base": {
        "desc": "基准: W-EMA200下方+EMA1200金叉VWAP",
        "weekly_pos_lt": 0.0,
        "ema_cross": True,
        "momentum": False,
    },
    "deep": {
        "desc": "深熊: W-EMA200下方10%+EMA1200金叉VWAP",
        "weekly_pos_lt": -0.10,
        "ema_cross": True,
        "momentum": False,
    },
    "compression": {
        "desc": "压缩: ATR低位+价格突破EMA1200",
        "weekly_pos_lt": 0.0,
        "ema_cross": False,
        "momentum": False,
        "compression": True,
    },
}


# ═══ Data loading ═══
def load_ohlc(sym):
    df = pd.read_parquet(_REPO_ROOT / "cache" / "timeframes" / f"{sym}_120T.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def compute_weekly_ema200(ohlc):
    w = ohlc["close"].resample("W").last().dropna()
    return w.ewm(span=200, adjust=False).mean().reindex(ohlc.index, method="ffill")


def compute_ema1200(c):
    return c.ewm(span=1200, adjust=False).mean()


def compute_vwap1200(ohlc):
    h, l, c, vol = ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]
    tp = (h + l + c) / 3.0
    return (tp * vol).rolling(1200, min_periods=120).sum() / vol.rolling(
        1200, min_periods=120
    ).sum().replace(0, np.nan)


def _cross_above(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


# ═══ Core simulation ═══
def simulate_symbol(
    ohlc,
    *,
    initial_capital=10000.0,
    initial_leverage=2.0,
    max_leverage=3.0,
    ladder_trigger=5.0,
    ladder_base_frac=0.08,
    eq_dd_stop=0.50,
    px_dd_stop=0.60,
    risk_cooldown=336,
    allow_reentry=True,
    reentry_cooldown=720,
    weekly_pos_lt=-0.05,
    ema_cross=True,
    momentum=False,
    compression=False,
    roll_near_vwap=False,
    max_position_notional=float("inf"),
):
    ohlc = ohlc.copy()
    close = ohlc["close"]
    wema = compute_weekly_ema200(ohlc)
    ohlc["wema200_pos"] = (close - wema) / close.replace(0, np.nan)
    ohlc["ema1200"] = compute_ema1200(close)
    ohlc["vwap1200"] = compute_vwap1200(ohlc)
    ema_cross_up = _cross_above(ohlc["ema1200"], ohlc["vwap1200"])
    roc5 = close.pct_change(5)
    roc20 = close.pct_change(20)

    # Compression signal
    comp_sig = pd.Series(False, index=ohlc.index)
    if compression:
        h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        def _rk(arr):
            if len(arr) < 2 or np.isnan(arr[-1]):
                return np.nan
            cur = arr[-1]
            hist = arr[:-1]
            hist = hist[np.isfinite(hist)]
            return (hist <= cur).sum() / len(hist) if len(hist) > 0 else np.nan

        atr_pct = atr.rolling(540, min_periods=540).apply(_rk, raw=True)
        comp_sig = (atr_pct < 0.20) & _cross_above(close, ohlc["ema1200"])

    # Entry signal
    entry_sig = pd.Series(False, index=ohlc.index)
    for i in range(len(ohlc)):
        wp = ohlc["wema200_pos"].iloc[i]
        if pd.isna(wp):
            continue
        ok = bool(wp < weekly_pos_lt)
        if ema_cross:
            ok = ok and bool(ema_cross_up.iloc[i])
        if momentum:
            ok = ok and (roc5.iloc[i] > 0 and roc20.iloc[i] > 0)
        if compression:
            ok = ok and bool(comp_sig.iloc[i])
        entry_sig.iloc[i] = ok

    # State
    eq = initial_capital
    peak_eq = eq
    max_dd = 0.0
    pos_qty = 0.0
    entry_px = 0.0
    peak_px = 0.0
    lev = 0.0
    in_pos = False
    busted = False
    rolls = 0
    sells = 0
    sell_notional = 0.0
    risk_cuts = 0
    num_entries = 0
    last_risk_bar = -risk_cooldown
    last_exit_bar = -reentry_cooldown
    last_exit_profit = False  # 上次出场是否盈利
    trades = []
    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        ts = ohlc.index[i]
        sig = bool(entry_sig.iloc[i])
        bars_since_exit = i - last_exit_bar

        if not in_pos:
            # 空仓时继续更新peak_px，确保再入场回撤计算准确
            peak_px = max(peak_px, c)
            ema1200_col = ohlc["ema1200"]
            can_reentry = False
            if sig and bars_since_exit >= reentry_cooldown:
                can_reentry = True  # 完整入场信号 + 冷却期
            elif (
                last_exit_profit
                and bars_since_exit >= 48  # 利润出场后最短4天(48*2h)
                and (c / peak_px < 0.88)  # 从上次峰值回撤>12%才允许重新上车
                and roc5.iloc[i] > 0  # 短期动量确认
            ):
                can_reentry = True  # 第二波：利润出场后趋势延续再入场
            if can_reentry:
                entry_px = c
                peak_px = c
                lev = initial_leverage
                pos_qty = (eq * lev) / c
                in_pos = True
                last_risk_bar = -risk_cooldown
                last_exit_profit = False
                num_entries += 1
                trades.append(
                    dict(
                        entry_time=str(ts),
                        entry_price=entry_px,
                        leverage=lev,
                        pos_qty=float(pos_qty),
                        type="entry",
                        entry_num=num_entries,
                        is_reentry=last_exit_profit,
                    )
                )

        if in_pos:
            upnl = pos_qty * (c - entry_px)
            cur_eq = eq + upnl
            peak_px = max(peak_px, c)
            if cur_eq <= 100:
                eq = 100
                pos_qty = 0
                lev = 0
                in_pos = False
                busted = True
                last_exit_bar = i
                trades.append(
                    dict(exit_time=str(ts), exit_price=c, exit_equity=eq, type="bust")
                )
                break
            peak_eq = max(peak_eq, cur_eq)
            eq_dd = (cur_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
            max_dd = min(max_dd, eq_dd)
            bars_since_risk = i - last_risk_bar

            # Risk stops
            if (
                eq_dd <= -eq_dd_stop
                and pos_qty > 0
                and bars_since_risk >= risk_cooldown
            ):
                cut = pos_qty * 0.5
                eq += cut * (c - entry_px)
                pos_qty -= cut
                risk_cuts += 1
                last_risk_bar = i
                trades.append(
                    dict(
                        time=str(ts),
                        price=c,
                        type="risk_eq_dd",
                        reduce_qty=float(cut),
                        eq_dd=float(eq_dd),
                        remaining_qty=float(pos_qty),
                    )
                )
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit_bar = i
                    last_exit_profit = eq > initial_capital
                    continue
            px_dd = (c - entry_px) / entry_px if entry_px > 0 else 0.0
            if (
                px_dd <= -px_dd_stop
                and pos_qty > 0
                and bars_since_risk >= risk_cooldown
            ):
                cut = pos_qty * 0.5
                eq += cut * (c - entry_px)
                pos_qty -= cut
                risk_cuts += 1
                last_risk_bar = i
                trades.append(
                    dict(
                        time=str(ts),
                        price=c,
                        type="risk_px_dd",
                        reduce_qty=float(cut),
                        px_dd=float(px_dd),
                        remaining_qty=float(pos_qty),
                    )
                )
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit_bar = i
                    last_exit_profit = eq > initial_capital
                    continue

            # Roll (step-wise +1x, with position notional cap)
            px_dd_peak = (c - peak_px) / peak_px
            # 先计算受notional cap约束的最大可达杠杆
            max_achievable_lev = min(
                max_leverage,
                max_position_notional / cur_eq if cur_eq > 0 else max_leverage,
            )
            # 只在杠杆有实际提升空间时才触发roll
            roll_ok = (
                px_dd_peak <= -0.20
                and upnl > 0
                and lev < max_achievable_lev - 0.01
                and c > 0
            )
            if roll_near_vwap and roll_ok and "vwap1200" in ohlc.columns:
                vwap_val = float(ohlc["vwap1200"].iloc[i])
                roll_ok = roll_ok and abs(c - vwap_val) / c < 0.05
            if roll_ok:
                old_lev = lev
                lev = min(lev + 1.0, max_leverage)
                new_notional = cur_eq * lev
                if new_notional > max_position_notional:
                    lev = max_position_notional / cur_eq if cur_eq > 0 else lev
                # 双重确保杠杆真正增加
                if lev <= old_lev + 0.01:
                    lev = old_lev
                    continue
                pos_qty = (cur_eq * lev) / c
                rolls += 1
                trades.append(
                    dict(
                        time=str(ts),
                        price=c,
                        type="roll",
                        old_leverage=old_lev,
                        new_leverage=lev,
                        pnl=float(upnl),
                        px_dd_peak=float(px_dd_peak),
                        peak_px=float(peak_px),
                    )
                )

            # Ladder — 只在上涨时卖出
            eq_mult = cur_eq / initial_capital
            peak_pullback = (peak_px - c) / peak_px if peak_px > 0 else 0
            ema1200_val = (
                float(ohlc["ema1200"].iloc[i])
                if not pd.isna(ohlc["ema1200"].iloc[i])
                else c
            )
            is_uptrend = c > ema1200_val
            near_peak = peak_pullback < 0.15  # 离峰值回撤<15%才算"近高点"
            if eq_mult >= ladder_trigger and pos_qty > 0 and is_uptrend and near_peak:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                max_frac = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sell_frac = min(max_frac, ladder_base_frac * spd)
                sq = pos_qty * sell_frac
                # 底仓保护: 保留至少20%仓位，永不卖完
                min_residual = (eq * initial_leverage / c) * 0.20 if c > 0 else 0
                if pos_qty - sq < min_residual:
                    sq = max(0, pos_qty - min_residual)
                if sq > 0:
                    eq += sq * (c - entry_px)
                    pos_qty -= sq
                    sells += 1
                    sell_notional += sq * c
                    trades.append(
                        dict(
                            time=str(ts),
                            price=c,
                            type="ladder_sell",
                            sell_qty=float(sq),
                            sell_notional=float(sq * c),
                            sell_frac=float(sell_frac),
                            speed=float(spd),
                            eq_mult=float(eq_mult),
                            remaining_qty=float(pos_qty),
                            equity=float(eq),
                            peak_pullback=float(peak_pullback),
                        )
                    )
                    if pos_qty <= min_residual * 1.01:  # 接近底仓时停止卖出
                        pass  # 保持持仓，等待下一波上涨
        else:
            peak_eq = max(peak_eq, eq)

    if in_pos and not busted:
        eq += pos_qty * (float(close.iloc[-1]) - entry_px)
        if eq < 0:
            eq = 100
            busted = True

    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_bars]).days / 365.25)
    tot_ret = eq / initial_capital
    cagr_val = tot_ret ** (1.0 / years) - 1.0 if eq > 0 else -1.0
    calmar = tot_ret / abs(max_dd) if max_dd < 0 else tot_ret

    return dict(
        final_equity=float(eq),
        initial_capital=initial_capital,
        total_return=float(tot_ret),
        cagr=float(cagr_val),
        max_dd=float(max_dd),
        calmar=float(calmar),
        busted=busted,
        roll_count=rolls,
        risk_reduces=risk_cuts,
        total_sells=sells,
        total_sell_notional=float(sell_notional),
        num_entries=num_entries,
        years=float(years),
        trades=trades,
    )


# ═══ Plotting ═══
def plot_trading_map(symbol, ohlc, result, out_dir):
    """Bokeh HTML 交易地图 — K线 + VWAP/EMA + 入场/滚仓/卖出标记 + 权益曲线"""
    try:
        from bokeh.plotting import figure as bk_figure, output_file, save
        from bokeh.models import (
            HoverTool,
            ColumnDataSource,
            Span,
            Div,
            NumeralTickFormatter,
        )
        from bokeh.layouts import column as bk_column
        from bokeh.resources import INLINE
    except ImportError:
        print("  [SKIP] bokeh not installed, pip install bokeh")
        return

    trades = result.get("trades", [])
    if not trades:
        return

    close = ohlc["close"]
    init_cap = result.get("initial_capital", 10000.0)

    # Resample to 4h for cleaner K-line
    ohlc_4h = (
        ohlc.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    # Recompute indicators on 4h
    ema1200_4h = ohlc_4h["close"].ewm(span=1200, adjust=False).mean()
    h, l, c = ohlc_4h["high"], ohlc_4h["low"], ohlc_4h["close"]
    tp = (h + l + c) / 3.0
    vol = ohlc_4h.get("volume", pd.Series(1, index=ohlc_4h.index)).clip(lower=0.0)
    num = (tp * vol).rolling(1200, min_periods=120).sum()
    den = vol.rolling(1200, min_periods=120).sum()
    vwap1200_4h = num / den.replace(0, np.nan)

    # Weekly EMA200 (resample to weekly, compute, then reindex back to 4h)
    weekly_close = ohlc_4h["close"].resample("W").last().dropna()
    weekly_ema200 = weekly_close.ewm(span=200, adjust=False).mean()
    wema200_4h = weekly_ema200.reindex(ohlc_4h.index, method="ffill")

    bar_w = pd.Timedelta(hours=4).total_seconds() * 1000 * 0.7

    # ═══ Panel 1: K-line + indicators + markers ═══
    p = bk_figure(
        title=f"{symbol} — Trend Rolling ({result.get('num_entries',0)} entries, {result.get('roll_count',0)} rolls)",
        x_axis_type="datetime",
        width=1400,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    inc = ohlc_4h.close >= ohlc_4h.open
    dec = ~inc
    p.segment(
        ohlc_4h.index[inc],
        ohlc_4h.high[inc],
        ohlc_4h.index[inc],
        ohlc_4h.low[inc],
        color="#26a69a",
        line_width=1,
    )
    p.segment(
        ohlc_4h.index[dec],
        ohlc_4h.high[dec],
        ohlc_4h.index[dec],
        ohlc_4h.low[dec],
        color="#ef5350",
        line_width=1,
    )
    p.vbar(
        ohlc_4h.index[inc],
        bar_w,
        ohlc_4h.open[inc],
        ohlc_4h.close[inc],
        fill_color="#26a69a",
        line_color="#26a69a",
        fill_alpha=0.8,
    )
    p.vbar(
        ohlc_4h.index[dec],
        bar_w,
        ohlc_4h.open[dec],
        ohlc_4h.close[dec],
        fill_color="#ef5350",
        line_color="#ef5350",
        fill_alpha=0.8,
    )

    p.line(
        ohlc_4h.index,
        wema200_4h,
        line_color="#38bdf8",
        line_width=2.0,
        line_alpha=0.8,
        legend_label="W-EMA(200)",
    )
    p.line(
        ohlc_4h.index,
        vwap1200_4h,
        line_color="#c026d3",
        line_width=1.5,
        line_alpha=0.8,
        legend_label="VWAP(1200)",
    )
    p.line(
        ohlc_4h.index,
        ema1200_4h,
        line_color="#f59e0b",
        line_width=1.5,
        line_alpha=0.8,
        legend_label="EMA(1200)",
    )

    # Collect markers
    entry_x, entry_y, entry_lbl = [], [], []
    roll_x, roll_y, roll_lbl = [], [], []
    sell_x, sell_y, sell_lbl = [], [], []
    risk_x, risk_y, risk_lbl = [], [], []

    for t in trades:
        ts_raw = t.get("entry_time") or t.get("time") or t.get("exit_time")
        if ts_raw is None:
            continue
        ts = pd.Timestamp(ts_raw)
        tp = t.get("type", "")
        px = t.get("entry_price") or t.get("price", 0)
        if px <= 0:
            continue
        if tp == "entry":
            entry_x.append(ts)
            entry_y.append(px)
            entry_lbl.append(f"Entry #{t.get('entry_num','?')} {t.get('leverage','')}x")
        elif tp == "roll":
            roll_x.append(ts)
            roll_y.append(px)
            roll_lbl.append(
                f"Roll {t.get('old_leverage','')}->{t.get('new_leverage','')}x"
            )
        elif tp == "ladder_sell":
            sell_x.append(ts)
            sell_y.append(px)
            sell_lbl.append(
                f"Sell {t.get('sell_frac',0)*100:.0f}% @{t.get('eq_mult',0):.1f}x"
            )
        elif "risk" in tp:
            risk_x.append(ts)
            risk_y.append(px)
            risk_lbl.append(f"Risk: {tp}")

    renderers = []
    if entry_x:
        src = ColumnDataSource({"x": entry_x, "y": entry_y, "label": entry_lbl})
        r = p.scatter(
            "x",
            "y",
            source=src,
            marker="triangle",
            size=14,
            color="#22c55e",
            line_color="#166534",
            legend_label="Entry",
            fill_alpha=0.9,
        )
        renderers.append(r)
        p.add_tools(HoverTool(tooltips=[("Entry", "@label")], renderers=[r]))
    if roll_x:
        src = ColumnDataSource({"x": roll_x, "y": roll_y, "label": roll_lbl})
        r = p.scatter(
            "x",
            "y",
            source=src,
            marker="diamond",
            size=12,
            color="#3b82f6",
            line_color="#1e40af",
            legend_label="Roll",
            fill_alpha=0.9,
        )
        renderers.append(r)
        p.add_tools(HoverTool(tooltips=[("Roll", "@label")], renderers=[r]))
    if sell_x:
        src = ColumnDataSource({"x": sell_x, "y": sell_y, "label": sell_lbl})
        r = p.scatter(
            "x",
            "y",
            source=src,
            marker="inverted_triangle",
            size=9,
            color="#ef4444",
            line_color="#991b1b",
            legend_label="Ladder Sell",
            fill_alpha=0.7,
        )
        renderers.append(r)
        p.add_tools(HoverTool(tooltips=[("Sell", "@label")], renderers=[r]))
    if risk_x:
        src = ColumnDataSource({"x": risk_x, "y": risk_y, "label": risk_lbl})
        r = p.scatter(
            "x",
            "y",
            source=src,
            marker="x",
            size=12,
            color="#7f1d1d",
            legend_label="Risk Reduce",
        )
        renderers.append(r)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.grid.grid_line_alpha = 0.25
    p.yaxis.axis_label = "Price (USDT)"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")

    # ═══ Panel 2: Equity curve ═══
    eq_curve = _build_eq_curve(ohlc, trades, init_cap)
    p_eq = bk_figure(
        title="Equity Curve (USDT)",
        x_axis_type="datetime",
        width=1400,
        height=200,
        x_range=p.x_range,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p_eq.line(eq_curve.index, eq_curve.values, line_width=2, color="#2563eb")
    p_eq.add_layout(
        Span(
            location=init_cap,
            dimension="width",
            line_color="#9ca3af",
            line_dash="dashed",
            line_width=1,
        )
    )
    p_eq.add_layout(
        Span(
            location=init_cap * 5,
            dimension="width",
            line_color="#f59e0b",
            line_dash="dashed",
            line_width=1,
        )
    )
    p_eq.grid.grid_line_alpha = 0.25
    p_eq.yaxis.axis_label = "Equity ($)"
    p_eq.yaxis.formatter = NumeralTickFormatter(format="$0,0")

    # ═══ Panel 3: Leverage ═══
    lev_curve = _build_lev_curve(ohlc, trades)
    p_lev = bk_figure(
        title="Leverage",
        x_axis_type="datetime",
        width=1400,
        height=120,
        x_range=p.x_range,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p_lev.line(lev_curve.index, lev_curve.values, line_width=2, color="#8b5cf6")
    for lv in [1, 2, 3, 5]:
        p_lev.add_layout(
            Span(
                location=lv,
                dimension="width",
                line_color="#9ca3af",
                line_dash="dotted",
                line_width=0.5,
            )
        )
    p_lev.grid.grid_line_alpha = 0.25
    p_lev.yaxis.axis_label = "Lev"
    p_lev.yaxis.formatter = NumeralTickFormatter(format="0.0")

    # Summary
    busted = result.get("busted", False)
    summary = (
        f"<p style='font-size:13px;max-width:1400px'>"
        f"<b>{result.get('num_entries',0)} entries</b> | "
        f"Final: <b>${result.get('final_equity',0):,.0f}</b> ({result.get('total_return',0):.2f}x) | "
        f"CAGR: {result.get('cagr',0)*100:.1f}% | MaxDD: {result.get('max_dd',0)*100:.1f}% | "
        f"Rolls: {result.get('roll_count',0)} | Sells: {result.get('total_sells',0)} | "
        f"RiskCuts: {result.get('risk_reduces',0)} | "
        f"<span style='color:{"#ef4444" if busted else "#22c55e"}'>{'BUSTED' if busted else 'OK'}</span>"
        f"</p>"
    )

    layout = bk_column(Div(text=summary, width=1400), p, p_eq, p_lev)
    out_path = out_dir / f"trading_map_{symbol}.html"
    output_file(str(out_path), title=f"Trend Rolling — {symbol}", mode="inline")
    save(layout, resources=INLINE)
    print(f"  Trading map: {out_path}")


def _build_eq_curve(ohlc, trades, init_cap):
    eq_s = pd.Series(init_cap, index=ohlc.index, dtype=float)
    in_pos = False
    qty = 0.0
    ep = 0.0
    realized = init_cap
    ti = 0
    for i in range(len(ohlc)):
        c = float(ohlc.iloc[i]["close"])
        tsi = ohlc.index[i]
        while ti < len(trades):
            t = trades[ti]
            tts = pd.Timestamp(
                t.get("entry_time") or t.get("time") or t.get("exit_time")
            )
            if tts is None or tts > tsi:
                break
            tp = t.get("type", "")
            if tp == "entry":
                in_pos = True
                qty = float(t.get("pos_qty", 0))
                ep = float(t.get("entry_price", 0))
            elif tp in ("ladder_sell", "risk_eq_dd", "risk_px_dd"):
                sq = float(t.get("sell_qty") or t.get("reduce_qty", 0))
                realized += sq * (float(t.get("price", 0)) - ep)
                qty -= sq
                if qty <= 0:
                    in_pos = False
            elif tp == "bust":
                in_pos = False
                qty = 0
                realized = float(t.get("exit_equity", 100))
            ti += 1
        eq_s.iloc[i] = realized + qty * (c - ep) if in_pos and qty > 0 else realized
    return eq_s


def _build_lev_curve(ohlc, trades):
    lev_s = pd.Series(0.0, index=ohlc.index, dtype=float)
    cur = 0.0
    ti = 0
    for i in range(len(ohlc)):
        tsi = ohlc.index[i]
        while ti < len(trades):
            t = trades[ti]
            tts = pd.Timestamp(
                t.get("entry_time") or t.get("time") or t.get("exit_time")
            )
            if tts is None or tts > tsi:
                break
            tp = t.get("type", "")
            if tp == "entry":
                cur = float(t.get("leverage", 0))
            elif tp == "roll":
                cur = float(t.get("new_leverage", cur))
            elif tp == "bust":
                cur = 0.0
            ti += 1
        lev_s.iloc[i] = cur
    return lev_s


# ═══ Runner ═══
def run(
    symbols,
    start,
    end,
    entry_preset="champion",
    initial_capital=10000.0,
    initial_leverage=2.0,
    max_leverage=3.0,
    ladder_trigger=5.0,
    eq_dd_stop=0.50,
    px_dd_stop=0.60,
    no_reentry=False,
    plot=False,
    out_dir=None,
):
    preset = ENTRY_PRESETS.get(entry_preset, ENTRY_PRESETS["champion"])
    results = {}
    for sym in symbols:
        print(f"  Simulating {sym}...")
        try:
            ohlc = load_ohlc(sym).loc[start:end]
            if len(ohlc) < 2000:
                results[sym] = {"error": "insufficient_data"}
                continue
            r = simulate_symbol(
                ohlc,
                initial_capital=initial_capital,
                initial_leverage=initial_leverage,
                max_leverage=preset.get("max_leverage", max_leverage),
                ladder_trigger=preset.get("ladder_trigger", ladder_trigger),
                ladder_base_frac=preset.get("ladder_base_frac", 0.08),
                eq_dd_stop=preset.get("eq_dd_stop", eq_dd_stop),
                px_dd_stop=px_dd_stop,
                allow_reentry=not no_reentry,
                roll_near_vwap=preset.get("roll_near_vwap", False),
                max_position_notional=preset.get("max_position_notional", float("inf")),
                **{
                    k: v
                    for k, v in preset.items()
                    if k
                    not in (
                        "desc",
                        "ladder_trigger",
                        "roll_near_vwap",
                        "max_leverage",
                        "max_position_notional",
                        "eq_dd_stop",
                        "ladder_base_frac",
                    )
                },
            )
            results[sym] = r
            print(
                f"    Equity: ${r['final_equity']:,.0f} | {'+'if r['cagr']>0 else ''}{r['cagr']*100:.1f}% | "
                f"MaxDD: {r['max_dd']*100:.1f}% | Entries: {r['num_entries']} | "
                f"Rolls: {r['roll_count']} | Sells: {r['total_sells']} | RiskCut: {r['risk_reduces']}"
            )
            if plot and out_dir and "error" not in r:
                plot_trading_map(sym, ohlc, r, out_dir)
        except FileNotFoundError:
            results[sym] = {"error": "data_not_found"}
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[sym] = {"error": str(e)}
    valid = {k: v for k, v in results.items() if "error" not in v}
    n = len(valid) or 1
    return dict(
        per_symbol=results,
        total_equity=float(sum(v["final_equity"] for v in valid.values())),
        total_return=float(
            sum(v["final_equity"] for v in valid.values()) / (n * initial_capital)
        ),
        avg_cagr=float(np.mean([v["cagr"] for v in valid.values()])),
        worst_max_dd=float(min((v["max_dd"] for v in valid.values()), default=0)),
        worst_calmar=float(
            min((v.get("calmar", 0) for v in valid.values()), default=0)
        ),
        busts=sum(1 for v in valid.values() if v["busted"]),
        symbol_count=n,
    )


# ═══ CLI ═══
def main():
    p = argparse.ArgumentParser(description="趋势滚仓模拟器 v3")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-06-01")
    p.add_argument(
        "--entry",
        default="champion",
        choices=list(ENTRY_PRESETS.keys()),
        help="入场预设",
    )
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--initial-leverage", type=float, default=2.0)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--ladder-trigger", type=float, default=5.0)
    p.add_argument("--eq-dd-stop", type=float, default=0.50)
    p.add_argument("--px-dd-stop", type=float, default=0.60)
    p.add_argument("--no-reentry", action="store_true", help="禁止重新入场")
    p.add_argument("--output", default="results/trend_rolling")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = ENTRY_PRESETS[args.entry]

    print("=" * 70)
    print("  趋势滚仓模拟器 v3")
    print("=" * 70)
    print(f"  Entry:   {args.entry} — {preset['desc']}")
    print(f"  Symbols: {symbols}  |  Period: {args.start}→{args.end}")
    print(
        f"  Capital: ${args.initial_capital:,.0f}/sym  |  Lev: {args.initial_leverage}x→{preset.get('max_leverage', args.max_leverage)}x"
    )
    print(
        f"  Ladder:  {preset.get('ladder_trigger', args.ladder_trigger)}x equity  |  EqDD stop: {preset.get('eq_dd_stop', args.eq_dd_stop)*100:.0f}%  |  PxDD stop: {args.px_dd_stop*100:.0f}%"
    )
    print(f"  Re-entry: {'OFF' if args.no_reentry else 'ON (60d cooldown)'}")
    print("=" * 70)

    results = run(
        symbols,
        args.start,
        args.end,
        entry_preset=args.entry,
        initial_capital=args.initial_capital,
        initial_leverage=args.initial_leverage,
        max_leverage=args.max_leverage,
        ladder_trigger=args.ladder_trigger,
        eq_dd_stop=args.eq_dd_stop,
        px_dd_stop=args.px_dd_stop,
        no_reentry=args.no_reentry,
        plot=args.plot,
        out_dir=out_dir,
    )

    print(f"\n{'='*70}\n  SUMMARY (entry: {args.entry})\n{'='*70}")
    print(
        f"  Total Equity: ${results['total_equity']:,.0f}  |  Return: {results['total_return']:.2f}x"
    )
    print(
        f"  Avg CAGR: {results['avg_cagr']*100:.1f}%  |  Worst MaxDD: {results['worst_max_dd']*100:.1f}%"
    )
    print(f"  Busts: {results['busts']}/{results['symbol_count']}")
    hdr = f"  {'Symbol':<12s} {'Final $':>10s} {'Return':>7s} {'CAGR':>8s} {'MaxDD':>7s} {'Entries':>7s} {'Rolls':>6s} {'Sells':>6s} {'Risk':>5s} {'Bust':>5s}"
    print(f"\n{hdr}\n  " + "-" * len(hdr))
    for sym in symbols:
        r = results["per_symbol"].get(sym, {})
        if "error" in r:
            print(f"  {sym:<12s} ERROR: {r['error']}")
        else:
            print(
                f"  {sym:<12s} ${r['final_equity']:>8,.0f} {r['total_return']:6.2f}x {r['cagr']*100:7.1f}% {r['max_dd']*100:6.1f}% {r['num_entries']:6d} {r['roll_count']:5d} {r['total_sells']:5d} {r['risk_reduces']:4d} {'YES' if r['busted'] else 'NO':>5s}"
            )

    summary = {k: v for k, v in results.items() if k != "per_symbol"}
    summary["per_symbol"] = {}
    for sym, r in results["per_symbol"].items():
        summary["per_symbol"][sym] = (
            {k: v for k, v in r.items() if k != "trades"} if "error" not in r else r
        )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / "full_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
