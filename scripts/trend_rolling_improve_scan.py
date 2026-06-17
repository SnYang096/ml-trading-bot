#!/usr/bin/env python3
"""Rolling & Spot 改进实验 — 大均线滚仓 + EMA加速DCA

测试改进方案:
  Rolling:
    roll_near_ema:  滚仓条件加"价格靠近EMA1200"（大均线支撑处滚仓）
    roll_near_vwap: 滚仓条件加"价格靠近VWAP1200"
    entry_relaxed:  入场放松（冠军 OR 深熊+金叉），让BTC也能入场
    ladder_3x:      卖出触发从5x降到3x
  Spot:
    dca_ema_boost: 靠近EMA1200时加速DCA（1.5x deploy）

用法:
  python scripts/trend_rolling_improve_scan.py
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ═══ Data ═══
def load_ohlc(sym):
    df = pd.read_parquet(_REPO_ROOT / "cache" / "timeframes" / f"{sym}_120T.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def wema(ohlc):
    w = ohlc["close"].resample("W").last().dropna()
    return w.ewm(span=200, adjust=False).mean().reindex(ohlc.index, method="ffill")


def ema1200(c):
    return c.ewm(span=1200, adjust=False).mean()


def vwap1200(ohlc):
    h, l, c, vol = ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]
    tp = (h + l + c) / 3.0
    return (tp * vol).rolling(1200, min_periods=120).sum() / vol.rolling(
        1200, min_periods=120
    ).sum().replace(0, np.nan)


def _cross(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


# ═══ Entry signals ═══
def champion_signal(ohlc):
    c = ohlc["close"]
    wp = (c - wema(ohlc)) / c.replace(0, np.nan)
    return (
        (wp < -0.05)
        & _cross(ema1200(c), vwap1200(ohlc))
        & (c.pct_change(5) > 0)
        & (c.pct_change(20) > 0)
    )


def relaxed_signal(ohlc):
    """冠军 OR (深熊+金叉, 不要动量)"""
    c = ohlc["close"]
    wp = (c - wema(ohlc)) / c.replace(0, np.nan)
    champ = (
        (wp < -0.05)
        & _cross(ema1200(c), vwap1200(ohlc))
        & (c.pct_change(5) > 0)
        & (c.pct_change(20) > 0)
    )
    relaxed = (wp < 0.0) & _cross(ema1200(c), vwap1200(ohlc))  # 深熊+金叉, no momentum
    return champ | relaxed


# ═══ Rolling sim (parameterized) ═══
def sim_rolling(
    ohlc,
    entry_sig,
    *,
    initial_capital=10000.0,
    initial_lev=2.0,
    max_lev=3.0,
    ladder_trigger=5.0,
    base_frac=0.08,
    eq_dd=0.50,
    px_dd=0.60,
    risk_cd=336,
    reentry_cd=720,
    roll_near_ema=False,
    roll_near_vwap=False,
    roll_near_ema200=False,
):
    """Rolling with optional MA-proximity roll condition"""
    close = ohlc["close"]
    e1200 = ema1200(close)
    v1200 = vwap1200(ohlc)
    e200 = close.ewm(span=200, adjust=False).mean()
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
    last_risk = -risk_cd
    last_exit = -reentry_cd
    min_b = max(1200, 200 * 7)

    for i in range(min_b, len(ohlc)):
        c = float(close.iloc[i])
        sig = bool(entry_sig.iloc[i]) if i < len(entry_sig) else False

        if not in_pos and sig and (i - last_exit) >= reentry_cd:
            entry_px = c
            peak_px = c
            lev = initial_lev
            pos_qty = (eq * lev) / c
            in_pos = True
            last_risk = -risk_cd
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
            eq_dd_v = (cur_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
            max_dd = min(max_dd, eq_dd_v)
            bsr = i - last_risk

            if eq_dd_v <= -eq_dd and pos_qty > 0 and bsr >= risk_cd:
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
            pdd = (c - entry_px) / entry_px if entry_px > 0 else 0.0
            if pdd <= -px_dd and pos_qty > 0 and bsr >= risk_cd:
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

            # Roll trigger
            px_dd_peak = (c - peak_px) / peak_px
            roll_ok = px_dd_peak <= -0.20 and upnl > 0 and lev < max_lev and c > 0
            if roll_near_ema:
                roll_ok = roll_ok and abs(c - e1200.iloc[i]) / c < 0.05
            if roll_near_vwap:
                roll_ok = roll_ok and abs(c - v1200.iloc[i]) / c < 0.05
            if roll_near_ema200:
                roll_ok = roll_ok and abs(c - e200.iloc[i]) / c < 0.05
            if roll_ok:
                lev = min(lev + 1.0, max_lev)
                pos_qty = (cur_eq * lev) / c
                rolls += 1

            # Ladder
            eq_mult = cur_eq / initial_capital
            if eq_mult >= ladder_trigger and pos_qty > 0:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                mf = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sf = min(mf, base_frac * spd)
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
    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_b]).days / 365.25)
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


# ═══ Spot sim (parameterized) ═══
def sim_spot(
    ohlc,
    *,
    budget=2500.0,
    deploy_base=50.0,
    ladder_trig=5.0,
    base_frac=0.05,
    ladder_exp=0.75,
    max_spd=4.0,
    deploy_bars=12,
    ema_boost=False,
):
    """Spot DCA with optional EMA-proximity boost"""
    close = ohlc["close"]
    cash = budget
    peak_eq = cash
    max_dd = 0.0
    pos_qty = 0.0
    cost_total = 0.0
    deploys = 0
    sells = 0
    last_deploy = -deploy_bars
    last_sell = -deploy_bars
    wp = (close - wema(ohlc)) / close.replace(0, np.nan)
    e1200 = ema1200(close)
    min_b = max(1200, 200 * 7)

    for i in range(min_b, len(ohlc)):
        c = float(close.iloc[i])
        in_bear = float(wp.iloc[i]) < 0
        deployed_pct = 100.0 * (1.0 - cash / budget) if budget > 0 else 0
        if deployed_pct < 30:
            decay = 1.0
        elif deployed_pct < 60:
            decay = 0.7
        elif deployed_pct < 80:
            decay = 0.4
        else:
            decay = 0.2

        # EMA boost: accelerate DCA when price near EMA1200
        boost = 1.5 if (ema_boost and abs(c - e1200.iloc[i]) / c < 0.05) else 1.0
        deploy_amt = deploy_base * decay * boost
        deploy_qty = deploy_amt / c if c > 0 else 0

        if in_bear and cash >= deploy_amt and (i - last_deploy) >= deploy_bars:
            pos_qty += deploy_qty
            cash -= deploy_amt
            cost_total += deploy_amt
            deploys += 1
            last_deploy = i

        if pos_qty > 0 and (i - last_sell) >= deploy_bars:
            cur_val = pos_qty * c
            mtm = cur_val / cost_total if cost_total > 0 else 0.0
            if mtm >= ladder_trig:
                spd = min(max_spd, (mtm / ladder_trig) ** ladder_exp)
                sf = min(0.99, base_frac * spd)
                sq = pos_qty * sf
                if sq > 0:
                    cash += sq * c
                    pos_qty -= sq
                    sells += 1
                    last_sell = i

        eq_now = cash + pos_qty * c if pos_qty > 0 else cash
        peak_eq = max(peak_eq, eq_now)
        if peak_eq > 0:
            max_dd = min(max_dd, (eq_now - peak_eq) / peak_eq)

    final_eq = cash + pos_qty * float(close.iloc[-1])
    years = max(0.01, (ohlc.index[-1] - ohlc.index[min_b]).days / 365.25)
    tot = final_eq / budget if budget > 0 else 1.0
    return dict(
        final_equity=float(final_eq),
        total_return=float(tot),
        cagr=float(tot ** (1.0 / years) - 1 if tot > 0 else -1),
        max_dd=float(max_dd),
        deploys=deploys,
        sells=sells,
    )


# ═══ Main ═══
def main():
    parser = argparse.ArgumentParser(description="Rolling & Spot 改进扫描")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-01")
    args = parser.parse_args()
    syms = [s.strip() for s in args.symbols.split(",")]

    # Experiment matrix
    experiments = {
        # ── Rolling baseline ──
        "R0_champion": dict(entry="champion"),
        # ── Rolling roll improvements ──
        "R1_roll_near_ema": dict(entry="champion", roll_near_ema=True),
        "R2_roll_near_vwap": dict(entry="champion", roll_near_vwap=True),
        "R3_roll_near_ema200": dict(entry="champion", roll_near_ema200=True),
        # ── Rolling entry improvements ──
        "R4_entry_relaxed": dict(entry="relaxed"),
        "R5_relaxed+roll_ema": dict(entry="relaxed", roll_near_ema=True),
        # ── Rolling ladder ──
        "R6_ladder_3x": dict(entry="champion", ladder_trigger=3.0),
        "R7_ladder_3x+roll_ema": dict(
            entry="champion", ladder_trigger=3.0, roll_near_ema=True
        ),
        # ── Spot baseline ──
        "S0_spot_baseline": dict(spot=True),
        # ── Spot improvements ──
        "S1_spot_ema_boost": dict(spot=True, ema_boost=True),
    }

    print("=" * 110)
    print("  Rolling & Spot 改进扫描")
    print("=" * 110)
    print(f"  Symbols: {syms}  |  Period: {args.start} → {args.end}")
    print(f"  {'ID':<22s} {'Type':<6s} {'描述'}")
    for eid, cfg in experiments.items():
        if cfg.get("spot"):
            desc = (
                "Spot基线"
                if "ema_boost" not in cfg or not cfg["ema_boost"]
                else "Spot+EMA加速DCA"
            )
        else:
            parts = []
            if cfg.get("entry") == "relaxed":
                parts.append("入场放松")
            else:
                parts.append("冠军入场")
            if cfg.get("roll_near_ema"):
                parts.append("滚仓@EMA1200")
            if cfg.get("roll_near_vwap"):
                parts.append("滚仓@VWAP")
            if cfg.get("roll_near_ema200"):
                parts.append("滚仓@EMA200")
            if cfg.get("ladder_trigger", 5.0) != 5.0:
                parts.append(f"卖出{cfg['ladder_trigger']}x")
            if (
                not any(
                    k
                    for k in ["roll_near_ema", "roll_near_vwap", "roll_near_ema200"]
                    if cfg.get(k)
                )
                and cfg.get("ladder_trigger", 5.0) == 5.0
                and cfg.get("entry") != "relaxed"
            ):
                parts = ["基线"]
            desc = " + ".join(parts)
        print(f"  {eid:<22s} {'Spot' if cfg.get('spot') else 'Rolling':<6s} {desc}")
    print("=" * 110)

    SPOT_BUDGETS = {"BTCUSDT": 5000, "ETHUSDT": 2500, "SOLUSDT": 2500, "BNBUSDT": 2500}
    DEPLOY_BASE = {"BTCUSDT": 100, "ETHUSDT": 50, "SOLUSDT": 50, "BNBUSDT": 50}

    all_rows = []
    for sym in syms:
        print(f"\n  ── {sym} ──")
        try:
            ohlc = load_ohlc(sym).loc[args.start : args.end]
            if len(ohlc) < 2000:
                continue
            sig_champ = champion_signal(ohlc)
            sig_relaxed = relaxed_signal(ohlc)

            for eid, cfg in experiments.items():
                if cfg.get("spot"):
                    r = sim_spot(
                        ohlc,
                        budget=SPOT_BUDGETS.get(sym, 2500),
                        deploy_base=DEPLOY_BASE.get(sym, 50),
                        ema_boost=cfg.get("ema_boost", False),
                    )
                else:
                    sig = sig_relaxed if cfg.get("entry") == "relaxed" else sig_champ
                    r = sim_rolling(
                        ohlc,
                        sig,
                        initial_capital=10000.0,
                        ladder_trigger=cfg.get("ladder_trigger", 5.0),
                        roll_near_ema=cfg.get("roll_near_ema", False),
                        roll_near_vwap=cfg.get("roll_near_vwap", False),
                        roll_near_ema200=cfg.get("roll_near_ema200", False),
                    )
                r["experiment"] = eid
                r["symbol"] = sym
                all_rows.append(r)
        except Exception as e:
            import traceback

            traceback.print_exc()

    df = pd.DataFrame(all_rows)

    # ── Aggregate ──
    print(f"\n{'='*110}")
    print(f"  汇总排名 (按总收益)")
    print(f"{'='*110}")

    agg = (
        df.groupby("experiment")
        .agg(
            total_eq=("final_equity", "sum"),
            avg_ret=("total_return", "mean"),
            avg_cagr=("cagr", "mean"),
            worst_dd=("max_dd", "min"),
            total_ops=("entries", lambda x: x.sum() if x.notna().any() else 0),
            total_sells=("sells", "sum"),
            busts=("busted", "sum"),
        )
        .reset_index()
    )

    # Compute total return relative to invested capital
    spot_invested = sum(SPOT_BUDGETS.get(s, 2500) for s in syms)
    roll_invested = len(syms) * 10000.0
    agg["invested"] = agg["experiment"].apply(
        lambda eid: spot_invested if "S" in str(eid) else roll_invested
    )
    agg["total_ret"] = (agg["total_eq"] / agg["invested"]).round(3)
    agg["calmar"] = (agg["total_ret"] / abs(agg["worst_dd"].clip(upper=-0.001))).round(
        2
    )
    agg["score"] = (
        agg["total_ret"]
        * (1 + agg["calmar"].clip(lower=0))
        * (1 - agg["worst_dd"].abs() / 100)
    ).round(2)

    ranked = agg.sort_values("score", ascending=False)
    hdr = f"  {'Rank':<4s} {'ID':<24s} {'总收益':>10s} {'Return':>7s} {'CAGR':>8s} {'MaxDD':>7s} {'Calmar':>7s} {'操作':>6s} {'卖出':>6s} {'Score':>7s}"
    print(hdr)
    print("  " + "-" * 100)
    for i, (_, row) in enumerate(ranked.iterrows()):
        print(
            f"  {i+1:<4d} {row['experiment']:<24s} ${row['total_eq']:>8,.0f} {row['total_ret']:6.2f}x "
            f"{row['avg_cagr']*100:7.1f}% {row['worst_dd']*100:6.1f}% {row['calmar']:6.2f} "
            f"{int(row['total_ops']):5d} {int(row['total_sells']):5d} {row['score']:7.2f}"
        )

    # Best per category
    print(f"\n  🏆 各赛道最佳:")
    categories = {
        "基线": "R0",
        "大均线滚仓": "R1|R2|R3",
        "入场放松": "R4|R5",
        "卖出优化": "R6|R7",
        "Spot": "S",
    }
    for cat, pat in categories.items():
        sub = ranked[ranked["experiment"].str.contains(pat)]
        if len(sub) > 0:
            best = sub.iloc[0]
            print(
                f"  [{cat}] {best['experiment']}: ${best['total_eq']:,.0f} | {best['total_ret']:.2f}x | DD {best['worst_dd']*100:.1f}%"
            )

    print(f"\n  💡 R0=基线, R1-R3=大均线滚仓, R4-R5=入场放松, R6-R7=卖出优化, S=Spot")


if __name__ == "__main__":
    main()
