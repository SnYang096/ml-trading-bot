#!/usr/bin/env python3
"""趋势滚仓 vs Spot 对比

在同一数据上对比:
  - Spot:     冠军入场信号 + 1x无杠杆 + 同样阶梯卖出
  - Rolling:  冠军入场信号 + 2x→3x杠杆 + 同样阶梯卖出
  - Spot DCA: 周线EMA200下方任意位置定投 (原始spot_accum逻辑)

用法:
  python scripts/trend_rolling_vs_spot.py
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


# ═══ Data ═══
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


# ═══ Common entry signal (champion) ═══
def champion_entry_signal(ohlc: pd.DataFrame) -> pd.Series:
    """冠军入场: 深熊5% + EMA1200金叉VWAP + 动量向上"""
    close = ohlc["close"]
    wema = compute_weekly_ema200(ohlc)
    wema_pos = (close - wema) / close.replace(0, np.nan)
    ema1200 = compute_ema1200(close)
    vwap1200 = compute_vwap1200(ohlc)
    cross = _cross_above(ema1200, vwap1200)
    roc5 = close.pct_change(5)
    roc20 = close.pct_change(20)
    sig = (wema_pos < -0.05) & cross & (roc5 > 0) & (roc20 > 0)
    return sig.fillna(False)


# ═══ DCA entry signal (spot原始逻辑) ═══
def dca_entry_signal(ohlc: pd.DataFrame) -> pd.Series:
    """Spot DCA: 周线EMA200下方任意位置"""
    close = ohlc["close"]
    wema = compute_weekly_ema200(ohlc)
    wema_pos = (close - wema) / close.replace(0, np.nan)
    return wema_pos < 0


# ═══ Simulators ═══


def sim_spot_champion(
    ohlc,
    entry_sig,
    *,
    initial_capital=10000.0,
    ladder_trigger=5.0,
    base_frac=0.08,
    cooldown_bars=720,
):
    """Spot (1x, 无杠杆) with champion entry + profit ladder"""
    close = ohlc["close"]
    eq = initial_capital
    peak_eq = eq
    max_dd = 0.0
    pos_qty = 0.0
    entry_px = 0.0
    in_pos = False
    sells = 0
    entries = 0
    last_exit = -cooldown_bars
    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        sig = bool(entry_sig.iloc[i]) if i < len(entry_sig) else False

        if not in_pos and sig and (i - last_exit) >= cooldown_bars:
            entry_px = c
            pos_qty = eq / c
            in_pos = True
            entries += 1

        if in_pos:
            upnl = pos_qty * (c - entry_px)
            cur_eq = eq + upnl
            if cur_eq <= 100:
                eq = 100
                pos_qty = 0
                in_pos = False
                break
            peak_eq = max(peak_eq, cur_eq)
            eq_dd = (cur_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
            max_dd = min(max_dd, eq_dd)

            # Ladder
            eq_mult = cur_eq / initial_capital
            if eq_mult >= ladder_trigger and pos_qty > 0:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                max_frac = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sf = min(max_frac, base_frac * spd)
                sq = pos_qty * sf
                if sq > 0:
                    eq += sq * (c - entry_px)
                    pos_qty -= sq
                    sells += 1
                    if pos_qty <= 0:
                        in_pos = False
                        last_exit = i
        else:
            peak_eq = max(peak_eq, eq)

    if in_pos:
        eq += pos_qty * (float(close.iloc[-1]) - entry_px)
    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_bars]).days / 365.25)
    tot = eq / initial_capital
    return dict(
        final_equity=float(eq),
        total_return=float(tot),
        cagr=float(tot ** (1.0 / years) - 1 if tot > 0 else -1),
        max_dd=float(max_dd),
        entries=entries,
        sells=sells,
    )


def sim_rolling_champion(
    ohlc,
    entry_sig,
    *,
    initial_capital=10000.0,
    initial_lev=2.0,
    max_lev=3.0,
    ladder_trigger=5.0,
    base_frac=0.08,
    eq_dd_stop=0.50,
    px_dd_stop=0.60,
    risk_cooldown=336,
    reentry_cooldown=720,
):
    """Rolling (2x→3x levered) with champion entry (same as simulator v3)"""
    close = ohlc["close"]
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
    risk_cuts = 0
    entries = 0
    last_risk = -risk_cooldown
    last_exit = -reentry_cooldown
    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        sig = bool(entry_sig.iloc[i]) if i < len(entry_sig) else False

        if not in_pos and sig and (i - last_exit) >= reentry_cooldown:
            entry_px = c
            peak_px = c
            lev = initial_lev
            pos_qty = (eq * lev) / c
            in_pos = True
            last_risk = -risk_cooldown
            entries += 1

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
                break
            peak_eq = max(peak_eq, cur_eq)
            eq_dd = (cur_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
            max_dd = min(max_dd, eq_dd)
            bars_since_risk = i - last_risk

            if (
                eq_dd <= -eq_dd_stop
                and pos_qty > 0
                and bars_since_risk >= risk_cooldown
            ):
                cut = pos_qty * 0.5
                eq += cut * (c - entry_px)
                pos_qty -= cut
                risk_cuts += 1
                last_risk = i
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit = i
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
                last_risk = i
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit = i
                    continue

            px_dd_peak = (c - peak_px) / peak_px
            if px_dd_peak <= -0.20 and upnl > 0 and lev < max_lev and c > 0:
                lev = min(lev + 1.0, max_lev)
                pos_qty = (cur_eq * lev) / c
                rolls += 1

            eq_mult = cur_eq / initial_capital
            if eq_mult >= ladder_trigger and pos_qty > 0:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                max_frac = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sf = min(max_frac, base_frac * spd)
                sq = pos_qty * sf
                if sq > 0:
                    eq += sq * (c - entry_px)
                    pos_qty -= sq
                    sells += 1
                    if pos_qty <= 0:
                        in_pos = False
                        last_exit = i
        else:
            peak_eq = max(peak_eq, eq)

    if in_pos and not busted:
        eq += pos_qty * (float(close.iloc[-1]) - entry_px)
    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_bars]).days / 365.25)
    tot = eq / initial_capital
    return dict(
        final_equity=float(eq),
        total_return=float(tot),
        cagr=float(tot ** (1.0 / years) - 1 if tot > 0 else -1),
        max_dd=float(max_dd),
        entries=entries,
        rolls=rolls,
        sells=sells,
        risk_cuts=risk_cuts,
        busted=busted,
    )


def sim_spot_accum_real(
    ohlc,
    *,
    symbol_budget=2500.0,
    deploy_per_bar_base=50.0,
    ladder_trigger=5.0,
    base_frac=0.05,
    ladder_exp=0.75,
    max_speed=4.0,
    deploy_bars=12,
):  # 每天最多1笔 (12 bar @2h)
    """真实 spot_accum_simple: 深熊DCA + deploy_decay + 5x成本阶梯"""
    close = ohlc["close"]
    cash = symbol_budget
    peak_eq = cash
    max_dd = 0.0
    pos_qty = 0.0
    cost_total = 0.0
    deploys = 0
    sells = 0
    last_deploy = -deploy_bars
    last_sell = -deploy_bars
    wema = compute_weekly_ema200(ohlc)
    wema_pos = (close - wema) / close.replace(0, np.nan)
    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        in_bear = float(wema_pos.iloc[i]) < 0

        # Deploy: max 1 per day, with decay
        deployed_pct = 100.0 * (1.0 - cash / symbol_budget) if symbol_budget > 0 else 0
        if deployed_pct < 30:
            decay = 1.0
        elif deployed_pct < 60:
            decay = 0.7
        elif deployed_pct < 80:
            decay = 0.4
        else:
            decay = 0.2

        deploy_qty = (deploy_per_bar_base * decay) / c if c > 0 else 0
        if (
            in_bear
            and cash >= deploy_per_bar_base * decay
            and (i - last_deploy) >= deploy_bars
        ):
            pos_qty += deploy_qty
            cash -= deploy_per_bar_base * decay
            cost_total += deploy_per_bar_base * decay
            deploys += 1
            last_deploy = i

        # Ladder sell (cost-based, like real spot)
        if pos_qty > 0 and (i - last_sell) >= deploy_bars:
            cur_val = pos_qty * c
            mtm = cur_val / cost_total if cost_total > 0 else 0.0
            if mtm >= ladder_trigger:
                spd = min(max_speed, (mtm / ladder_trigger) ** ladder_exp)
                sf = min(0.99, base_frac * spd)
                sq = pos_qty * sf
                if sq > 0:
                    cash += sq * c
                    pos_qty -= sq
                    sells += 1
                    last_sell = i

        # Track DD
        eq_now = cash + pos_qty * c if pos_qty > 0 else cash
        peak_eq = max(peak_eq, eq_now)
        if peak_eq > 0:
            dd = (eq_now - peak_eq) / peak_eq
            max_dd = min(max_dd, dd)

    final_eq = cash + pos_qty * float(close.iloc[-1])
    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_bars]).days / 365.25)
    tot = final_eq / symbol_budget if symbol_budget > 0 else 1.0
    return dict(
        final_equity=float(final_eq),
        total_return=float(tot),
        cagr=float(tot ** (1.0 / years) - 1 if tot > 0 else -1),
        max_dd=float(max_dd),
        deploys=deploys,
        sells=sells,
        budget=float(symbol_budget),
    )


# ═══ Main ═══
def main():
    parser = argparse.ArgumentParser(description="Rolling vs Spot 对比")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 100)
    print("  趋势滚仓 (Rolling) vs 现货 (Spot) — 同入场信号对比")
    print("=" * 100)
    print(f"  Symbols: {symbols}  |  Period: {args.start} → {args.end}")
    print(f"  Capital: ${args.capital:,.0f} per symbol")
    print(f"  Entry:   冠军入场 (深熊5% + EMA1200金叉VWAP + 动量向上)")
    print(f"  Rolling: 2x→3x杠杆 + 权益5x阶梯卖出 + 风控(DD50%/PD60%)")
    print(f"  Spot 真实: spot_accum_simple (深熊DCA + deploy_decay + 成本5x阶梯)")
    print(f"  Spot 冠军: 同冠军入场 + 1x无杠杆 (对照)")
    print("=" * 100)

    # Budgets from constitution (spot_accum_simple)
    SPOT_BUDGETS = {"BTCUSDT": 5000, "ETHUSDT": 2500, "SOLUSDT": 2500, "BNBUSDT": 2500}
    DEPLOY_BASE = {"BTCUSDT": 100, "ETHUSDT": 50, "SOLUSDT": 50, "BNBUSDT": 50}

    all_results = []
    for sym in symbols:
        print(f"\n  ── {sym} ──")
        try:
            ohlc = load_ohlc(sym).loc[args.start : args.end]
            if len(ohlc) < 2000:
                print(f"    SKIP: insufficient data")
                continue
            sig_champ = champion_entry_signal(ohlc)

            budget = SPOT_BUDGETS.get(sym, 2500)
            deploy_base = DEPLOY_BASE.get(sym, 50)

            r_rolling = sim_rolling_champion(
                ohlc, sig_champ, initial_capital=args.capital
            )
            r_spot_real = sim_spot_accum_real(
                ohlc, symbol_budget=budget, deploy_per_bar_base=deploy_base
            )
            r_spot_champ = sim_spot_champion(
                ohlc, sig_champ, initial_capital=args.capital
            )

            def _fmt(r, name):
                eq = r.get("final_equity", 0)
                ret = r.get("total_return", 0)
                cagr = r.get("cagr", 0)
                dd = r.get("max_dd", 0)
                entries = r.get("entries", r.get("deploys", 0))
                sells = r.get("sells", 0)
                print(
                    f"    {name:<12s} ${eq:>9,.0f}  {ret:5.2f}x  "
                    f"CAGR {cagr*100:+6.1f}%  MaxDD {dd*100:6.1f}%  "
                    f"op={entries:>4}  sells={sells:>5}  "
                    f"{'BUSTED!' if r.get('busted') else ''}"
                )
                return r

            rr = _fmt(r_rolling, "Rolling")
            rs = _fmt(r_spot_real, "Spot真实")
            rc = _fmt(r_spot_champ, "Spot冠军")

            all_results.append(
                dict(symbol=sym, rolling=rr, spot_real=rs, spot_champ=rc)
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    ERROR: {e}")

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  汇总对比")
    print(f"{'='*100}")
    n = len(all_results) or 1
    total_rolling = sum(r["rolling"]["final_equity"] for r in all_results)
    total_spot_real = sum(r["spot_real"]["final_equity"] for r in all_results)
    total_spot_champ = sum(r["spot_champ"]["final_equity"] for r in all_results)
    # Rolling uses $10k/sym, spot uses actual budget (BTC $5k, others $2.5k)
    init_rolling = n * args.capital
    init_spot = sum(SPOT_BUDGETS.get(r["symbol"], 2500) for r in all_results)

    def _agg_row(label, total, init):
        ret = total / init if init > 0 else 1.0
        dds = [r[label]["max_dd"] for r in all_results if r[label]["max_dd"] < 0]
        dd = min(dds) if dds else 0.0
        cagr = np.mean([r[label]["cagr"] for r in all_results])
        entries = sum(
            r[label].get("entries", r[label].get("deploys", 0)) for r in all_results
        )
        busts = sum(1 for r in all_results if r[label].get("busted"))
        calmar = ret / abs(dd) if dd < 0 else ret
        return ret, dd, cagr, entries, busts, calmar

    print(
        f"  {'Strategy':<14s} {'本金':>8s} {'终值':>10s} {'Return':>7s} {'CAGR':>8s} {'MaxDD':>7s} {'Calmar':>7s} {'操作':>6s} {'Busts':>6s}"
    )
    print(f"  {'-'*80}")
    for label, name, total, init in [
        ("rolling", "Rolling 2-3x", total_rolling, init_rolling),
        ("spot_real", "⭐ Spot真实", total_spot_real, init_spot),
        ("spot_champ", "Spot冠军入场", total_spot_champ, init_rolling),
    ]:
        ret, dd, cagr, entries, busts, calmar = _agg_row(label, total, init)
        print(
            f"  {name:<14s} ${init:>7,.0f} ${total:>8,.0f} {ret:6.2f}x {cagr*100:7.1f}% {dd*100:6.1f}% {calmar:6.2f} {entries:5d} {busts:5d}"
        )

    print(
        f"\n  ⭐ Spot真实 = 你的 spot_accum_simple 策略 (深熊DCA + deploy_decay + 每币不同预算)"
    )
    print(
        f"  📌 注意: Spot真实本金=${init_spot:,.0f} (按宪法预算), Rolling本金=${init_rolling:,.0f} (统一$10k)"
    )


if __name__ == "__main__":
    main()
