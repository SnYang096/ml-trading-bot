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
    trades = []
    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        ts = ohlc.index[i]
        sig = bool(entry_sig.iloc[i])
        bars_since_exit = i - last_exit_bar

        if not in_pos and sig and bars_since_exit >= reentry_cooldown:
            entry_px = c
            peak_px = c
            lev = initial_leverage
            pos_qty = (eq * lev) / c
            in_pos = True
            last_risk_bar = -risk_cooldown
            num_entries += 1
            trades.append(
                dict(
                    entry_time=str(ts),
                    entry_price=entry_px,
                    leverage=lev,
                    pos_qty=float(pos_qty),
                    type="entry",
                    entry_num=num_entries,
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
                    continue

            # Roll (step-wise +1x, with position notional cap)
            px_dd_peak = (c - peak_px) / peak_px
            roll_ok = px_dd_peak <= -0.20 and upnl > 0 and lev < max_leverage and c > 0
            if roll_near_vwap and roll_ok and "vwap1200" in ohlc.columns:
                vwap_val = float(ohlc["vwap1200"].iloc[i])
                roll_ok = roll_ok and abs(c - vwap_val) / c < 0.05
            if roll_ok:
                old_lev = lev
                lev = min(lev + 1.0, max_leverage)
                new_notional = cur_eq * lev
                if new_notional > max_position_notional:
                    lev = max_position_notional / cur_eq if cur_eq > 0 else lev
                    if lev <= old_lev:
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

            # Ladder
            eq_mult = cur_eq / initial_capital
            if eq_mult >= ladder_trigger and pos_qty > 0:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                max_frac = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sell_frac = min(max_frac, ladder_base_frac * spd)
                sq = pos_qty * sell_frac
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
                        )
                    )
                    if pos_qty <= 0:
                        in_pos = False
                        last_exit_bar = i
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return
    trades = result.get("trades", [])
    if not trades:
        return
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    ax_p, ax_e, ax_l = axes[0], axes[1], axes[2]
    ax_p.plot(ohlc.index, ohlc["close"], color="gray", alpha=0.4, lw=0.5, label="Close")
    if "ema1200" in ohlc.columns:
        ax_p.plot(
            ohlc.index,
            ohlc["ema1200"],
            color="blue",
            alpha=0.5,
            lw=0.8,
            label="EMA1200",
        )
    if "vwap1200" in ohlc.columns:
        ax_p.plot(
            ohlc.index,
            ohlc["vwap1200"],
            color="orange",
            alpha=0.5,
            lw=0.8,
            label="VWAP1200",
        )
    for t in trades:
        ts = pd.Timestamp(t.get("entry_time") or t.get("time") or t.get("exit_time"))
        if ts is None:
            continue
        tp = t.get("type", "")
        px = t.get("entry_price") or t.get("price", 0)
        if tp == "entry":
            ax_p.axvline(ts, color="green", alpha=0.5, lw=1.5)
            ax_p.annotate(
                f"E#{t.get('entry_num','')} {t.get('leverage','')}x",
                (ts, px),
                fontsize=6,
                color="green",
                xytext=(5, 15),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="green", alpha=0.4),
            )
        elif tp == "roll":
            ax_p.axvline(ts, color="purple", alpha=0.4, lw=1, ls=":")
        elif tp == "ladder_sell":
            ax_p.scatter(ts, px, color="red", s=8, alpha=0.3, marker="v")
        elif "risk" in tp:
            ax_p.axvline(ts, color="darkred", alpha=0.4, lw=1, ls="--")
    ax_p.set_ylabel("Price (USDT)")
    ax_p.legend(loc="upper left", fontsize=7)
    ax_p.set_title(
        f"{symbol} — Trend Rolling v3 ({result.get('num_entries',0)} entries)",
        fontsize=12,
        fontweight="bold",
    )
    ax_p.grid(True, alpha=0.3)
    eq_curve = _build_eq_curve(ohlc, trades, result["initial_capital"])
    ax_e.plot(eq_curve.index, eq_curve.values, color="darkgreen", lw=1)
    ax_e.axhline(result["initial_capital"], color="gray", ls="--", alpha=0.5)
    ax_e.axhline(
        result["initial_capital"] * 5,
        color="orange",
        ls="--",
        alpha=0.5,
        label="5x trigger",
    )
    ax_e.set_ylabel("Equity ($)")
    ax_e.legend(loc="upper left", fontsize=7)
    ax_e.grid(True, alpha=0.3)
    lev_curve = _build_lev_curve(ohlc, trades)
    ax_l.fill_between(lev_curve.index, 0, lev_curve.values, color="blue", alpha=0.3)
    ax_l.plot(lev_curve.index, lev_curve.values, color="blue", lw=1)
    for lv in [1, 2, 3]:
        ax_l.axhline(lv, color="gray", ls="--", alpha=0.3)
    ax_l.set_ylabel("Leverage")
    ax_l.set_xlabel("Date")
    ax_l.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"trading_map_{symbol}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trading map: {p}")


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
    p.add_argument("--start", default="2022-01-01")
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
        f"  Capital: ${args.initial_capital:,.0f}/sym  |  Lev: {args.initial_leverage}x→{args.max_leverage}x"
    )
    print(
        f"  Ladder:  {args.ladder_trigger}x equity  |  EqDD stop: {args.eq_dd_stop*100:.0f}%  |  PxDD stop: {args.px_dd_stop*100:.0f}%"
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
