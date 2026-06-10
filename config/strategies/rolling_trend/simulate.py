#!/usr/bin/env python3
"""滚仓策略模拟器 v2 — 多段多币种

用法:
  python config/strategies/rolling_trend/simulate.py \
    --trades results/tpc/experiments/exit_regime_20260610/E13_structural/ \
    --segments bear_2022,bull_2023_2024,recent_range_to_bear \
    --leverage 2,3 \
    --output results/rolling_trend/
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

SEGMENTS = {
    "bear_2022": ("2022-01-01", "2023-11-01"),
    "bull_2023_2024": ("2023-06-01", "2025-01-01"),
    "recent_range_to_bear": ("2025-01-01", "2026-04-01"),
    "recent_6m_oos": ("2025-12-01", "2026-06-01"),
}


def load_trades_for_segment(trades_dir: str, segment: str) -> pd.DataFrame:
    """加载指定段的 TPC 交易数据"""
    csv_path = Path(trades_dir) / segment / "event_trades_tpc.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, trying parent dir...")
        # Try the parent directory (single CSV for all segments)
        parent_csv = Path(trades_dir) / "event_trades_tpc.csv"
        if parent_csv.exists():
            csv_path = parent_csv
        else:
            raise FileNotFoundError(f"No trades CSV found in {trades_dir}")

    df = pd.read_csv(csv_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    # Filter to segment dates if needed
    if segment in SEGMENTS:
        seg_start, seg_end = SEGMENTS[segment]
        df = df[
            (df["entry_time"] >= seg_start) & (df["entry_time"] < seg_end)
        ]

    return df


def load_ohlc(symbol: str) -> pd.DataFrame:
    """加载 OHLC 数据"""
    p = _REPO_ROOT / "cache" / "timeframes" / f"{symbol}_120T.parquet"
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def simulate_symbol(
    trades: pd.DataFrame,
    ohlc: pd.DataFrame,
    seg_end: str,
    *,
    initial_leverage: float = 2.0,
    max_leverage: float = 3.0,
    initial_capital: float = 10000.0,
) -> Dict:
    """单币种滚仓模拟"""
    ohlc = ohlc.copy()
    ohlc["ema1200"] = ohlc["close"].ewm(span=1200, adjust=False).mean()

    sym_trades = trades.sort_values("entry_time").reset_index(drop=True)

    start = sym_trades["entry_time"].min()
    end = pd.Timestamp(seg_end, tz="UTC")
    ohlc = ohlc.loc[start:end]

    eq = initial_capital
    peak = eq
    mdd = 0.0
    pos = 0.0
    ep = 0.0
    lev = 0.0
    ip = False
    bust = False
    rolls = 0
    si = 0

    for ts, bar in ohlc.iterrows():
        c = bar["close"]
        ema = bar["ema1200"]

        # Check new signals
        while si < len(sym_trades) and sym_trades.iloc[si]["entry_time"] <= ts:
            sig = sym_trades.iloc[si]
            if not ip and pd.notna(sig.get("entry_price")):
                ep = float(sig["entry_price"])
                lev = initial_leverage
                pos = (eq * lev) / ep
                ip = True
            si += 1

        if not ip:
            peak = max(peak, eq)
            continue

        # Mark-to-market
        upnl = pos * (c - ep)
        ce = eq + upnl

        if ce <= 100:
            eq = 100
            bust = True
            break

        peak = max(peak, ce)
        dd = (ce - peak) / peak if peak > 0 else 0
        mdd = min(mdd, dd)

        # Roll: DD > 20% + price > EMA1200 + leverage < max
        if dd <= -0.20 and c > ema and lev < max_leverage:
            lev = min(lev + 1.0, max_leverage)
            pos = (ce * lev) / c
            rolls += 1

        # TP1: 5x entry → reduce 50%
        mult = c / ep
        if mult >= 5.0 and pos > 0 and lev >= 1.5:
            red = pos * 0.5
            eq += red * (c - ep)
            pos -= red
            lev = max(0.5, lev * 0.5)

        # TP2: 10x entry → close all
        if mult >= 10.0 and pos > 0:
            eq += pos * (c - ep)
            pos = 0
            lev = 0
            ip = False

    # Close at end
    if ip and not bust:
        eq += pos * (ohlc.iloc[-1]["close"] - ep)
    if bust:
        eq = 100

    years = (end - start).days / 365.25
    cagr = (eq / initial_capital) ** (1 / max(years, 0.01)) - 1 if eq > 0 else -1

    return {
        "final_equity": eq,
        "cagr": cagr,
        "max_dd": mdd,
        "bust": bust,
        "rolls": rolls,
        "signals": len(sym_trades),
    }


def run_segment(
    trades_dir: str,
    segment: str,
    initial_leverage: float,
    max_leverage: float,
) -> Dict:
    """运行单段模拟"""
    all_trades = load_trades_for_segment(trades_dir, segment)

    results = {}
    for sym in SYMBOLS:
        sym_trades = all_trades[
            (all_trades["symbol"] == sym)
            & (all_trades["is_add_position"] != "True")
        ]
        if len(sym_trades) == 0:
            results[sym] = {
                "final_equity": 10000,
                "cagr": 0,
                "max_dd": 0,
                "bust": False,
                "rolls": 0,
                "signals": 0,
            }
        else:
            ohlc = load_ohlc(sym)
            seg_end = SEGMENTS.get(segment, ("2026-01-01",))[1]
            results[sym] = simulate_symbol(
                sym_trades,
                ohlc,
                seg_end,
                initial_leverage=initial_leverage,
                max_leverage=max_leverage,
            )

    total_eq = sum(r["final_equity"] for r in results.values())
    busts = sum(1 for r in results.values() if r["bust"])
    total_sigs = sum(r["signals"] for r in results.values())
    avg_cagrs = [r["cagr"] for r in results.values() if not r["bust"]]
    avg_cagr = np.mean(avg_cagrs) if avg_cagrs else 0

    return {
        "segment": segment,
        "total_equity": total_eq,
        "multiplier": total_eq / (len(SYMBOLS) * 10000),
        "avg_cagr": avg_cagr,
        "max_dd": min(r["max_dd"] for r in results.values()),
        "busts": busts,
        "total_signals": total_sigs,
        "per_symbol": results,
    }


def main():
    parser = argparse.ArgumentParser(description="滚仓策略模拟器 v2")
    parser.add_argument(
        "--trades",
        default="results/tpc/experiments/exit_regime_20260610/E13_structural",
    )
    parser.add_argument(
        "--segments",
        default="bear_2022,bull_2023_2024",
    )
    parser.add_argument("--initial-leverage", type=float, default=2.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--output", default="results/rolling_trend")
    args = parser.parse_args()

    segments = [s.strip() for s in args.segments.split(",")]

    print(f"=== Rolling Trend Simulation ===")
    print(f"Trades: {args.trades}")
    print(f"Leverage: {args.initial_leverage}x → {args.max_leverage}x")
    print(f"Segments: {segments}")
    print()

    print(
        f"{'Segment':25s} {'Total $':>10s} {'Mult':>6s} {'avgCAGR':>8s} {'maxDD':>7s} {'Busts':>5s} {'Sigs':>5s}"
    )
    print("-" * 72)

    all_results = {}
    for seg in segments:
        try:
            r = run_segment(args.trades, seg, args.initial_leverage, args.max_leverage)
            all_results[seg] = r
            print(
                f"{seg:25s} ${r['total_equity']:>8,.0f} {r['multiplier']:5.1f}x {r['avg_cagr']*100:7.1f}% {r['max_dd']*100:6.1f}% {r['busts']:4d}/6 {r['total_signals']:5d}"
            )
        except FileNotFoundError as e:
            print(f"{seg:25s} SKIPPED: {e}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'simulation_results.json'}")


if __name__ == "__main__":
    main()
