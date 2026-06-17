#!/usr/bin/env python3
"""趋势滚仓入场扫描器 — Phase A Entry Scan

系统测试 20+ 入场条件变体, 用 trend_rolling_simulate 回测,
按爆炸利润潜力排序。

入场条件分类:
  A) 深度: 周线EMA200下方多深才入场
  B) 交叉: EMA/VWAP/MACD 金叉信号
  C) 压缩: 低波动后的突破（压缩→爆发）
  D) 量能: 成交量/CVD 确认
  E) 动量: 短期趋势已启动
  F) 组合: 多条件 AND

用法:
  python scripts/trend_rolling_entry_scan.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
    --start 2022-01-01 --end 2026-06-01 \
    --top 10
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Reuse helpers from trend_rolling_simulate ──
def load_ohlc(symbol: str) -> pd.DataFrame:
    p = _REPO_ROOT / "cache" / "timeframes" / f"{symbol}_120T.parquet"
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def compute_weekly_ema200(ohlc: pd.DataFrame) -> pd.Series:
    weekly = ohlc["close"].resample("W").last().dropna()
    weekly_ema200 = weekly.ewm(span=200, adjust=False).mean()
    return weekly_ema200.reindex(ohlc.index, method="ffill")


def compute_ema1200(close: pd.Series) -> pd.Series:
    return close.ewm(span=1200, adjust=False).mean()


def compute_vwap1200(ohlc: pd.DataFrame) -> pd.Series:
    h, l, c, vol = ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]
    tp = (h + l + c) / 3.0
    num = (tp * vol).rolling(window=1200, min_periods=120).sum()
    den = vol.rolling(window=1200, min_periods=120).sum()
    return num / den.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════
# Entry condition library
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EntryVariant:
    """An entry condition variant to test."""

    id: str
    name: str
    category: str
    description: str
    # Function that returns a boolean Series (True = entry signal at that bar)
    signal_fn: Callable = field(repr=False)

    def compute_signal(self, ohlc: pd.DataFrame) -> pd.Series:
        """Compute boolean entry signal series."""
        return self.signal_fn(ohlc)


def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """a crosses ABOVE b: a > b AND prev_a <= prev_b"""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """a crosses BELOW b"""
    return (a < b) & (a.shift(1) >= b.shift(1))


def build_entry_variants() -> List[EntryVariant]:
    """Build the library of entry condition variants."""
    variants = []

    def _reg(id_, name, cat, desc, fn):
        variants.append(
            EntryVariant(
                id=id_, name=name, category=cat, description=desc, signal_fn=fn
            )
        )

    # ── A: Trend Depth ──
    _reg(
        "A0_base",
        "周线EMA200下方(基准)",
        "深度",
        "weekly_ema_200_position < 0",
        lambda ohlc: ohlc["weekly_ema_200_position"] < 0,
    )

    _reg(
        "A1_deep",
        "周线EMA200深跌-5%",
        "深度",
        "weekly_ema_200_position < -0.05",
        lambda ohlc: ohlc["weekly_ema_200_position"] < -0.05,
    )

    _reg(
        "A2_deeper",
        "周线EMA200深跌-10%",
        "深度",
        "weekly_ema_200_position < -0.10",
        lambda ohlc: ohlc["weekly_ema_200_position"] < -0.10,
    )

    _reg(
        "A3_extreme",
        "周线EMA200极端深跌-15%",
        "深度",
        "weekly_ema_200_position < -0.15",
        lambda ohlc: ohlc["weekly_ema_200_position"] < -0.15,
    )

    # ── B: Cross Signals ──
    _reg(
        "B0_ema1200_vwap",
        "EMA1200金叉VWAP1200(基准)",
        "交叉",
        "ema1200 crosses above vwap1200",
        lambda ohlc: _cross_above(ohlc["ema1200"], ohlc["vwap1200"]),
    )

    _reg(
        "B1_price_ema1200",
        "价格上穿EMA1200",
        "交叉",
        "close crosses above ema1200",
        lambda ohlc: _cross_above(ohlc["close"], ohlc["ema1200"]),
    )

    _reg(
        "B2_ema50_ema200",
        "EMA50金叉EMA200",
        "交叉",
        "ema50 crosses above ema200",
        lambda ohlc: _cross_above(
            ohlc["close"].ewm(span=50, adjust=False).mean(),
            ohlc["close"].ewm(span=200, adjust=False).mean(),
        ),
    )

    _reg(
        "B3_ema1200_near_vwap",
        "EMA1200在VWAP附近金叉",
        "交叉",
        "ema1200 above vwap AND ema1200_position > -0.10",
        lambda ohlc: (ohlc["ema1200"] > ohlc["vwap1200"])
        & (ohlc["ema_1200_position"] > -0.10)
        & _cross_above(ohlc["ema1200"], ohlc["vwap1200"]),
    )

    _reg(
        "B4_macd_zero",
        "MACD柱上穿零轴",
        "交叉",
        "macd_histogram crosses above 0",
        lambda ohlc: _cross_above(_macd_hist(ohlc), pd.Series(0, index=ohlc.index)),
    )

    # ── C: Compression Breakout ──
    _reg(
        "C0_atr_low",
        "ATR低位+价格突破20bar高点",
        "压缩",
        "atr_percentile < 0.20 AND close > 20-bar high",
        lambda ohlc: (_atr_pct(ohlc) < 0.20)
        & (ohlc["close"] > ohlc["high"].rolling(20).max().shift(1)),
    )

    _reg(
        "C1_atr_low_ema",
        "ATR低位+价格突破EMA1200",
        "压缩",
        "atr_percentile < 0.20 AND price crosses above ema1200",
        lambda ohlc: (_atr_pct(ohlc) < 0.20)
        & _cross_above(ohlc["close"], ohlc["ema1200"]),
    )

    _reg(
        "C2_bb_squeeze",
        "布林带挤压+突破上轨",
        "压缩",
        "bb_width in bottom 20% AND close > bb_upper",
        lambda ohlc: (_bb_pct(ohlc) < 0.20) & (ohlc["close"] > _bb_upper(ohlc)),
    )

    # ── D: Volume Confirmation ──
    _reg(
        "D0_vol_spike",
        "放量(vol>1.5x均值)",
        "量能",
        "volume_ratio > 1.5",
        lambda ohlc: _vol_ratio(ohlc) > 1.5,
    )

    _reg(
        "D1_cvd_positive",
        "CVD正向",
        "量能",
        "cvd_change_5 > 0",
        lambda ohlc: ohlc.get("cvd_change_5", pd.Series(0, index=ohlc.index)) > 0,
    )

    _reg(
        "D2_vol_price_up",
        "放量+价格上涨",
        "量能",
        "volume_ratio > 1.2 AND close > open",
        lambda ohlc: (_vol_ratio(ohlc) > 1.2) & (ohlc["close"] > ohlc["open"]),
    )

    # ── E: Momentum ──
    _reg(
        "E0_roc_both_up",
        "短中期动量向上",
        "动量",
        "roc_5 > 0 AND roc_20 > 0",
        lambda ohlc: (_roc(ohlc, 5) > 0) & (_roc(ohlc, 20) > 0),
    )

    _reg(
        "E1_trend_aligned",
        "均线多头排列",
        "动量",
        "sma20 > sma50 > sma200",
        lambda ohlc: (_sma(ohlc, 20) > _sma(ohlc, 50))
        & (_sma(ohlc, 50) > _sma(ohlc, 200)),
    )

    _reg(
        "E2_strong_trend",
        "强趋势(r2>0.5)",
        "动量",
        "trend_r2_20 > 0.5",
        lambda ohlc: _trend_r2(ohlc) > 0.5,
    )

    # ── F: Winning Combos (deep bear + timing + confirmation) ──
    _reg(
        "F0_winner1",
        "深熊10%+EMA1200金叉VWAP+放量",
        "组合",
        "A2 + B0 + D0 (AND)",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.10)
        & _cross_above(ohlc["ema1200"], ohlc["vwap1200"])
        & (_vol_ratio(ohlc) > 1.5),
    )

    _reg(
        "F1_winner2",
        "深熊5%+EMA1200金叉+动量向上",
        "组合",
        "A1 + B0 + E0 (AND)",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.05)
        & _cross_above(ohlc["ema1200"], ohlc["vwap1200"])
        & (_roc(ohlc, 5) > 0)
        & (_roc(ohlc, 20) > 0),
    )

    _reg(
        "F2_winner3",
        "深熊10%+压缩爆发+放量",
        "组合",
        "A2 + C0 + D0 (AND)",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.10)
        & (_atr_pct(ohlc) < 0.20)
        & (ohlc["close"] > ohlc["high"].rolling(20).max().shift(1))
        & (_vol_ratio(ohlc) > 1.5),
    )

    _reg(
        "F3_winner4",
        "深熊5%+价格上穿EMA1200+CVD正",
        "组合",
        "A1 + B1 + D1 (AND)",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.05)
        & _cross_above(ohlc["close"], ohlc["ema1200"])
        & (ohlc.get("cvd_change_5", pd.Series(0, index=ohlc.index)) > 0),
    )

    _reg(
        "F4_winner5",
        "极端深熊15%+EMA金叉VWAP",
        "组合",
        "A3 + B0 (AND) — 极端深熊中金叉",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.15)
        & _cross_above(ohlc["ema1200"], ohlc["vwap1200"]),
    )

    _reg(
        "F5_winner6",
        "深熊+均线多头+放量突破",
        "组合",
        "A2 + E1 + C0 (AND)",
        lambda ohlc: (ohlc["weekly_ema_200_position"] < -0.10)
        & (_sma(ohlc, 20) > _sma(ohlc, 50))
        & (ohlc["close"] > ohlc["high"].rolling(20).max().shift(1)),
    )

    return variants


# ── Helper feature computers ──


def _atr_pct(ohlc: pd.DataFrame, window: int = 540) -> pd.Series:
    """ATR percentile (rolling rank)"""
    h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
        axis=1
    )
    atr = tr.rolling(14).mean()

    def _rank(arr):
        if len(arr) < 2 or np.isnan(arr[-1]):
            return np.nan
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        return (hist <= cur).sum() / len(hist) if len(hist) > 0 else np.nan

    return atr.rolling(window, min_periods=window).apply(_rank, raw=True)


def _bb_upper(ohlc: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
    mid = ohlc["close"].rolling(period).mean()
    std_s = ohlc["close"].rolling(period).std()
    return mid + std * std_s


def _bb_pct(ohlc: pd.DataFrame, period: int = 20) -> pd.Series:
    """BB width percentile"""
    mid = ohlc["close"].rolling(period).mean()
    std_s = ohlc["close"].rolling(period).std()
    width = 4 * std_s / mid.replace(0, np.nan)

    def _rank(arr):
        if len(arr) < 2 or np.isnan(arr[-1]):
            return np.nan
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        return (hist <= cur).sum() / len(hist) if len(hist) > 0 else np.nan

    return width.rolling(540, min_periods=540).apply(_rank, raw=True)


def _vol_ratio(ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    return ohlc["volume"] / ohlc["volume"].rolling(window).mean().replace(0, np.nan)


def _macd_hist(ohlc: pd.DataFrame, fast=12, slow=26, sig=9) -> pd.Series:
    ema_f = ohlc["close"].ewm(span=fast, adjust=False).mean()
    ema_s = ohlc["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal


def _roc(ohlc: pd.DataFrame, period: int) -> pd.Series:
    return ohlc["close"].pct_change(period)


def _sma(ohlc: pd.DataFrame, period: int) -> pd.Series:
    return ohlc["close"].rolling(period).mean()


def _trend_r2(ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    log_p = np.log(ohlc["close"].replace(0, np.nan)).ffill()

    def _r2(arr):
        if len(arr) < 3:
            return 0.0
        try:
            x = np.arange(len(arr))
            y = arr
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return max(0.0, min(1.0, 1 - ss_res / ss_tot)) if ss_tot != 0 else 0.0
        except:
            return 0.0

    return log_p.rolling(window, min_periods=3).apply(_r2, raw=False)


# ═══════════════════════════════════════════════════════════════════
# Rolling simulator (compact version for scanning)
# ═══════════════════════════════════════════════════════════════════


def simulate_with_entry(
    ohlc: pd.DataFrame,
    entry_signal: pd.Series,
    *,
    initial_capital: float = 10000.0,
    initial_leverage: float = 2.0,
    max_leverage: float = 3.0,
    ladder_trigger: float = 5.0,
    ladder_base_frac: float = 0.08,
    eq_dd_stop: float = 0.50,
    px_dd_stop: float = 0.60,
    risk_cooldown: int = 336,
    allow_reentry: bool = True,  # NEW: allow re-entering after exiting
    reentry_cooldown: int = 720,  # 60 days between re-entries
) -> Dict:
    """Run rolling simulation with a given entry signal series.

    allow_reentry: if True, after exiting a position, can enter again on new signals.
    """
    close = ohlc["close"]

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

    min_bars = max(1200, 200 * 7)

    for i in range(min_bars, len(ohlc)):
        c = float(close.iloc[i])
        sig = bool(entry_signal.iloc[i]) if i < len(entry_signal) else False
        bars_since_exit = i - last_exit_bar

        # ── Entry (with re-entry support) ──
        if not in_pos and sig and bars_since_exit >= reentry_cooldown:
            entry_px = c
            peak_px = c
            lev = initial_leverage
            pos_qty = (eq * lev) / c
            in_pos = True
            last_risk_bar = -risk_cooldown
            num_entries += 1

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
                break

            peak_eq = max(peak_eq, cur_eq)
            eq_dd = (cur_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
            max_dd = min(max_dd, eq_dd)
            bars_since_risk = i - last_risk_bar

            # Risk: Equity DD stop
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
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit_bar = i
                    continue

            # Risk: Price DD stop
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
                upnl = pos_qty * (c - entry_px)
                cur_eq = eq + upnl
                if pos_qty <= 0:
                    in_pos = False
                    last_exit_bar = i
                    continue

            # Roll
            px_dd_peak = (c - peak_px) / peak_px
            if px_dd_peak <= -0.20 and upnl > 0 and lev < max_leverage and c > 0:
                lev = min(lev + 1.0, max_leverage)
                pos_qty = (cur_eq * lev) / c
                rolls += 1

            # Ladder sell (equity-based)
            eq_mult = cur_eq / initial_capital
            if eq_mult >= ladder_trigger and pos_qty > 0:
                spd = min(5.0, (eq_mult / ladder_trigger) ** 1.0)
                max_frac = 1.0 if eq_mult >= ladder_trigger * 3 else 0.99
                sell_frac = min(max_frac, ladder_base_frac * spd)
                sell_qty = pos_qty * sell_frac
                if sell_qty > 0:
                    eq += sell_qty * (c - entry_px)
                    pos_qty -= sell_qty
                    sells += 1
                    sell_notional += sell_qty * c
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
    cagr = tot_ret ** (1.0 / years) - 1.0 if eq > 0 else -1.0

    # Score: Calmar ratio (total_return / |max_dd|)
    calmar = tot_ret / abs(max_dd) if max_dd < 0 else tot_ret

    return dict(
        final_equity=float(eq),
        total_return=float(tot_ret),
        cagr=float(cagr),
        max_dd=float(max_dd),
        calmar=float(calmar),
        busted=busted,
        roll_count=rolls,
        total_sells=sells,
        risk_reduces=risk_cuts,
        num_entries=num_entries,
        years=float(years),
    )


# ═══════════════════════════════════════════════════════════════════
# Entry scan main logic
# ═══════════════════════════════════════════════════════════════════


def prepare_ohlc(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLC and compute all needed indicators."""
    ohlc = load_ohlc(symbol).loc[start:end].copy()
    close = ohlc["close"]

    # Core indicators
    weekly_ema200 = compute_weekly_ema200(ohlc)
    ohlc["weekly_ema_200_position"] = (close - weekly_ema200) / close.replace(0, np.nan)
    ohlc["ema1200"] = compute_ema1200(close)
    ohlc["vwap1200"] = compute_vwap1200(ohlc)
    ohlc["ema_1200_position"] = (close - ohlc["ema1200"]) / close.replace(0, np.nan)

    return ohlc


def scan_entries(
    symbols: List[str],
    start: str,
    end: str,
    variants: List[EntryVariant],
    **sim_kwargs,
) -> pd.DataFrame:
    """Run entry scan across symbols and variants."""
    rows = []

    for sym in symbols:
        print(f"\n  Loading {sym}...")
        try:
            ohlc = prepare_ohlc(sym, start, end)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        for v in variants:
            try:
                signal = v.compute_signal(ohlc)
                n_sigs = int(signal.sum())
                if n_sigs == 0:
                    rows.append(
                        dict(
                            symbol=sym,
                            variant=v.id,
                            name=v.name,
                            category=v.category,
                            final_equity=sim_kwargs.get("initial_capital", 10000),
                            total_return=1.0,
                            cagr=0.0,
                            max_dd=0.0,
                            calmar=0,
                            num_entries=0,
                            n_signals=0,
                            busted=False,
                        )
                    )
                    continue

                r = simulate_with_entry(ohlc, signal, **sim_kwargs)
                rows.append(
                    dict(
                        symbol=sym,
                        variant=v.id,
                        name=v.name,
                        category=v.category,
                        description=v.description,
                        final_equity=r["final_equity"],
                        total_return=round(r["total_return"], 3),
                        cagr=round(r["cagr"] * 100, 1),
                        max_dd=round(r["max_dd"] * 100, 1),
                        calmar=round(r["calmar"], 2),
                        num_entries=r["num_entries"],
                        n_signals=n_sigs,
                        total_sells=r["total_sells"],
                        risk_reduces=r["risk_reduces"],
                        busted=r["busted"],
                    )
                )
            except Exception as e:
                rows.append(
                    dict(
                        symbol=sym,
                        variant=v.id,
                        name=v.name,
                        category=v.category,
                        error=str(e),
                    )
                )

    return pd.DataFrame(rows)


def rank_and_report(df: pd.DataFrame, top_n: int = 15):
    """Aggregate across symbols and rank variants."""
    # Aggregate: sum equity across symbols
    agg = (
        df.groupby(["variant", "name", "category", "description"])
        .agg(
            total_equity=("final_equity", "sum"),
            avg_return=("total_return", "mean"),
            avg_cagr=("cagr", "mean"),
            worst_dd=("max_dd", "min"),
            avg_calmar=("calmar", "mean"),
            total_entries=("num_entries", "sum"),
            total_signals=("n_signals", "sum"),
            busts=("busted", "sum"),
        )
        .reset_index()
    )

    n_syms = df["symbol"].nunique()
    init_total = n_syms * 10000
    agg["total_return"] = agg["total_equity"] / init_total
    agg["total_return"] = agg["total_return"].round(3)

    # Composite score: total_return * (1 + avg_calmar) * (1 - worst_dd/100) * log(1+total_entries)
    agg["explosive_score"] = (
        agg["total_return"]
        * (1 + agg["avg_calmar"].clip(lower=0))
        * (1 - agg["worst_dd"].abs() / 100)
        * np.log1p(agg["total_entries"])
    ).round(3)

    ranked = agg.sort_values("explosive_score", ascending=False)

    print(f"\n{'='*90}")
    print(f"  ENTRY VARIANT LEADERBOARD (ranked by explosive profit score)")
    print(f"  Symbols: {n_syms} | Capital: ${init_total:,.0f} total")
    print(f"{'='*90}")
    print(
        f"  {'Rank':<5s} {'ID':<18s} {'Category':<6s} {'Total$':>10s} {'Return':>7s} "
        f"{'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'Entries':>7s} {'Score':>7s}"
    )
    print(f"  {'-'*85}")

    for idx, (_, row) in enumerate(ranked.head(top_n).iterrows()):
        print(
            f"  {idx+1:<5d} {row['variant']:<18s} {row['category']:<6s} "
            f"${row['total_equity']:>8,.0f} {row['total_return']:6.2f}x "
            f"{row['avg_cagr']:6.1f}% {row['worst_dd']:6.1f}% "
            f"{row['avg_calmar']:6.2f} {row['total_entries']:6d} "
            f"{row['explosive_score']:7.3f}"
        )

    return ranked


def main():
    parser = argparse.ArgumentParser(description="趋势滚仓入场扫描器")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--initial-leverage", type=float, default=2.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--ladder-trigger", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=15, help="Show top N variants")
    parser.add_argument("--output", default="results/trend_rolling/entry_scan.csv")
    parser.add_argument("--no-reentry", action="store_true", help="Disable re-entry")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    variants = build_entry_variants()

    print(f"{'='*90}")
    print(f"  趋势滚仓入场扫描器 — Phase A Entry Scan")
    print(f"{'='*90}")
    print(f"  Symbols: {symbols}")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Leverage: {args.initial_leverage}x→{args.max_leverage}x")
    print(f"  Ladder trigger: {args.ladder_trigger}x equity")
    print(f"  Entry variants to test: {len(variants)}")
    print(f"  Re-entry: {'OFF' if args.no_reentry else 'ON'} (cooldown 60 days)")
    print(f"{'='*90}")

    sim_kwargs = dict(
        initial_capital=args.initial_capital,
        initial_leverage=args.initial_leverage,
        max_leverage=args.max_leverage,
        ladder_trigger=args.ladder_trigger,
        allow_reentry=not args.no_reentry,
    )

    df = scan_entries(symbols, args.start, args.end, variants, **sim_kwargs)

    # Handle errors
    errs = df[df["error"].notna()] if "error" in df.columns else pd.DataFrame()
    df_ok = df[df.get("error", pd.Series()).isna()] if "error" in df.columns else df

    if len(errs) > 0:
        print(f"\n  ⚠️  {len(errs)} errors (skipped)")

    ranked = rank_and_report(df_ok, top_n=args.top)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    print(f"\n  Full leaderboard saved to {out_path}")

    # Best per category
    print(f"\n{'='*90}")
    print(f"  BEST PER CATEGORY")
    print(f"{'='*90}")
    for cat in ["深度", "交叉", "压缩", "量能", "动量", "组合"]:
        cat_df = ranked[ranked["category"] == cat]
        if len(cat_df) == 0:
            continue
        best = cat_df.iloc[0]
        print(
            f"  [{cat}] {best['variant']} ({best['name']}): "
            f"${best['total_equity']:,.0f} | {best['total_return']:.2f}x | "
            f"MaxDD {best['worst_dd']:.1f}% | {best['total_entries']} entries"
        )


if __name__ == "__main__":
    main()
