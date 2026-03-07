#!/usr/bin/env python3
"""
entry_offset 对比实验 — 验证延迟入场对 snotio (mean R-multiples) 的影响

原理:
  entry_offset=0: 信号 bar 收盘价入场
  entry_offset=1: 下一根 bar open 入场 (当前生产配置)
  entry_offset=2: 信号后第 2 根 bar open 入场
  entry_offset=3: 信号后第 3 根 bar open 入场

验证假设 (来自 token microstructure 文章):
  "延迟 1-3 根 K 线入场可以躲过 liquidity sweep, 提升 Sharpe"

用法:
  python scripts/compare_entry_offset.py --logs results/walk_forward/bpc/oos_fold_2/logs_gated_oos.parquet
  python scripts/compare_entry_offset.py --logs <任意 logs_gated.parquet> --strategy bpc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_rr_for_offset(
    df: pd.DataFrame,
    entry_offset: int,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    max_holding_bars: int = 50,
    trailing_atr_mult: float = 1.5,
    use_trailing: bool = True,
) -> pd.DataFrame:
    """
    对 gate-allow 的交易，模拟不同 entry_offset 下的 R/R 结果。
    返回每笔交易的 realized_r, exit_type, holding_bars。
    """
    results = []
    ohlc = df[["open", "high", "low", "close"]].values
    atr_arr = df["atr"].values
    direction_arr = df["entry_direction"].values
    n = len(df)

    for i in range(n):
        sig = direction_arr[i]
        if sig == 0 or np.isnan(sig):
            continue

        # 入场价
        entry_bar = i + entry_offset
        if entry_bar >= n:
            continue
        entry_price = ohlc[entry_bar, 0]  # open of entry_bar
        atr = atr_arr[i]  # ATR at signal time

        if np.isnan(entry_price) or np.isnan(atr) or atr <= 0:
            continue

        # 止损/止盈
        if sig > 0:  # long
            sl = entry_price - stop_loss_r * atr
            tp = entry_price + take_profit_r * atr
        else:  # short
            sl = entry_price + stop_loss_r * atr
            tp = entry_price - take_profit_r * atr

        # 扫描出场
        trail_sl = sl
        exit_type = "timeout"
        exit_price = None
        bars_held = 0

        scan_start = entry_bar + 1
        scan_end = min(entry_bar + max_holding_bars + 1, n)

        for j in range(scan_start, scan_end):
            h, l, c = ohlc[j, 1], ohlc[j, 2], ohlc[j, 3]
            bars_held = j - entry_bar

            if sig > 0:  # long
                # 止损
                if l <= trail_sl:
                    exit_price = trail_sl
                    exit_type = "sl"
                    break
                # 止盈
                if tp > 0 and h >= tp:
                    exit_price = tp
                    exit_type = "tp"
                    break
                # Trailing
                if use_trailing and trailing_atr_mult > 0:
                    new_trail = h - trailing_atr_mult * atr
                    if new_trail > trail_sl:
                        trail_sl = new_trail
            else:  # short
                if h >= trail_sl:
                    exit_price = trail_sl
                    exit_type = "sl"
                    break
                if tp > 0 and l <= tp:
                    exit_price = tp
                    exit_type = "tp"
                    break
                if use_trailing and trailing_atr_mult > 0:
                    new_trail = l + trailing_atr_mult * atr
                    if new_trail < trail_sl:
                        trail_sl = new_trail

        # 超时平仓
        if exit_price is None:
            if scan_end - 1 < n:
                exit_price = ohlc[min(scan_end - 1, n - 1), 3]  # close of last bar
            else:
                exit_price = ohlc[n - 1, 3]
            exit_type = "timeout"

        # 计算 R-multiple
        if atr > 0:
            if sig > 0:
                realized_r = (exit_price - entry_price) / (stop_loss_r * atr)
            else:
                realized_r = (entry_price - exit_price) / (stop_loss_r * atr)
        else:
            realized_r = 0.0

        results.append(
            {
                "bar_idx": i,
                "entry_offset": entry_offset,
                "direction": sig,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr": atr,
                "realized_r": realized_r,
                "exit_type": exit_type,
                "bars_held": bars_held,
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="entry_offset 对比实验")
    parser.add_argument(
        "--logs",
        required=True,
        help="logs_gated.parquet 路径",
    )
    parser.add_argument(
        "--strategy",
        default="bpc",
        help="策略名 (用于读取 direction 列名, 默认 bpc)",
    )
    parser.add_argument(
        "--offsets",
        default="0,1,2,3",
        help="逗号分隔的 entry_offset 值 (默认 0,1,2,3)",
    )
    parser.add_argument(
        "--stop-loss-r",
        type=float,
        default=1.0,
        help="止损 R 倍数 (默认 1.0)",
    )
    parser.add_argument(
        "--take-profit-r",
        type=float,
        default=2.0,
        help="止盈 R 倍数 (默认 2.0)",
    )
    parser.add_argument(
        "--max-holding-bars",
        type=int,
        default=50,
        help="最大持仓 bars (默认 50)",
    )
    parser.add_argument(
        "--trailing-atr-mult",
        type=float,
        default=1.5,
        help="trailing stop ATR 倍数 (默认 1.5)",
    )
    parser.add_argument(
        "--no-trailing",
        action="store_true",
        help="禁用 trailing stop",
    )
    args = parser.parse_args()

    # ── 加载数据 ──
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ 文件不存在: {logs_path}")
        sys.exit(1)

    df = pd.read_parquet(logs_path)
    print(f"📊 加载 {logs_path.name}: {len(df)} 行")

    # Gate 过滤
    if "gate_decision" in df.columns:
        n_before = len(df)
        df = df[df["gate_decision"] == "allow"].copy()
        print(f"🚪 Gate filter: {len(df)} allow / {n_before} total")

    if len(df) < 10:
        print("❌ Gate 后数据不足 10 行, 无法分析")
        sys.exit(1)

    # 方向列 — 按优先级搜索
    dir_candidates = [
        f"{args.strategy}_breakout_direction",
        f"{args.strategy}_impulse_failure_direction",
        "bpc_breakout_direction",
        "fer_impulse_failure_direction",
        "entry_direction",
    ]
    dir_col = None
    for cand in dir_candidates:
        if cand in df.columns:
            dir_col = cand
            break
    if dir_col is None:
        print(f"❌ 找不到方向列, 尝试过: {dir_candidates}")
        sys.exit(1)
    print(f"📌 使用方向列: {dir_col}")

    df["entry_direction"] = df[dir_col].astype(float)
    n_long = (df["entry_direction"] > 0).sum()
    n_short = (df["entry_direction"] < 0).sum()
    print(f"📈 方向: {n_long} long, {n_short} short")

    # 按 symbol 分组处理
    symbol_col = "_symbol" if "_symbol" in df.columns else "symbol"
    symbols = df[symbol_col].unique() if symbol_col in df.columns else ["ALL"]
    print(f"🪙  Symbols: {', '.join(str(s) for s in symbols)}")

    offsets = [int(x) for x in args.offsets.split(",")]

    # ── 运行对比 ──
    print()
    print("=" * 80)
    print(
        f"{'offset':>6} | {'trades':>6} | {'snotio':>8} | {'winrate':>7} | {'mean_R':>7} | {'median_R':>8} | {'SL%':>5} | {'TP%':>5} | {'TO%':>5} | {'bars':>5}"
    )
    print("-" * 80)

    all_results = {}
    for offset in offsets:
        # 按 symbol 分组 (保持 OHLC 连续性)
        trades_list = []
        for sym in symbols:
            if symbol_col in df.columns:
                sym_df = df[df[symbol_col] == sym].copy()
            else:
                sym_df = df.copy()

            sym_df = sym_df.sort_index().reset_index(drop=True)

            res = simulate_rr_for_offset(
                sym_df,
                entry_offset=offset,
                stop_loss_r=args.stop_loss_r,
                take_profit_r=args.take_profit_r,
                max_holding_bars=args.max_holding_bars,
                trailing_atr_mult=args.trailing_atr_mult,
                use_trailing=not args.no_trailing,
            )
            if len(res) > 0:
                res["symbol"] = sym
            trades_list.append(res)

        trades = pd.concat(trades_list, ignore_index=True)
        all_results[offset] = trades

        if len(trades) == 0:
            print(
                f"{offset:>6} | {'N/A':>6} | {'N/A':>8} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8} | {'N/A':>5} | {'N/A':>5} | {'N/A':>5} | {'N/A':>5}"
            )
            continue

        r = trades["realized_r"]
        n_trades = len(trades)
        snotio = r.mean()
        win_rate = (r > 0).mean() * 100
        mean_r = r.mean()
        median_r = r.median()
        sl_pct = (trades["exit_type"] == "sl").mean() * 100
        tp_pct = (trades["exit_type"] == "tp").mean() * 100
        to_pct = (trades["exit_type"] == "timeout").mean() * 100
        mean_bars = trades["bars_held"].mean()

        print(
            f"{offset:>6} | {n_trades:>6} | {snotio:>+8.4f} | {win_rate:>6.1f}% | {mean_r:>+7.4f} | {median_r:>+8.4f} | {sl_pct:>4.1f}% | {tp_pct:>4.1f}% | {to_pct:>4.1f}% | {mean_bars:>5.1f}"
        )

    print("=" * 80)

    # ── 统计显著性 (z-test vs baseline offset=1) ──
    baseline_offset = 1
    if baseline_offset in all_results and len(all_results[baseline_offset]) > 0:
        bl = all_results[baseline_offset]["realized_r"]
        bl_mean = bl.mean()
        bl_std = bl.std()
        bl_n = len(bl)

        print()
        print(f"📐 统计显著性 (vs offset={baseline_offset}, snotio={bl_mean:+.4f}):")
        print(
            f"{'offset':>6} | {'Δ snotio':>10} | {'z-stat':>8} | {'p-value':>8} | {'显著?':>6}"
        )
        print("-" * 55)

        for offset in offsets:
            if offset == baseline_offset:
                continue
            if offset not in all_results or len(all_results[offset]) == 0:
                continue

            exp = all_results[offset]["realized_r"]
            exp_mean = exp.mean()
            exp_n = len(exp)
            delta = exp_mean - bl_mean

            # Welch's t-test approximation
            se = np.sqrt(bl_std**2 / bl_n + exp.std() ** 2 / exp_n)
            if se > 0:
                z = delta / se
                from scipy.stats import norm

                p = 2 * (1 - norm.cdf(abs(z)))  # two-tailed
            else:
                z = 0
                p = 1.0

            sig = "✅" if p < 0.05 else "❌"
            print(f"{offset:>6} | {delta:>+10.4f} | {z:>+8.3f} | {p:>8.4f} | {sig:>6}")

    print()
    print("💡 结论指引:")
    print("  - offset > 1 的 snotio 高于 offset=1 → 支持延迟入场假设")
    print("  - offset > 1 的 snotio 低于 offset=1 → 不支持, 当前配置已是最优")
    print("  - p < 0.05 → 差异有统计显著性")


if __name__ == "__main__":
    main()
