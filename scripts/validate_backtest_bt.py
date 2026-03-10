#!/usr/bin/env python3
"""
Backtrader 独立验证: 随机入场 + trailing stop → WR 是否真的 ~88%?

验证逻辑:
  1. 从 tick 数据构建真实 1min OHLCV K 线
  2. 用 backtrader 框架跑 trailing stop 策略
  3. 入场: 每隔 N 根 1H bar 随机入场 (随机方向)
  4. 出场: backtrader 原生逻辑 — 初始 SL + trailing stop
  5. 对比自实现结果

同时加 Monte Carlo 随机游走理论验证 (数学证明)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import backtrader as bt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ================================================================
# Part 1: Backtrader 验证
# ================================================================


class RandomTrailingStopStrategy(bt.Strategy):
    """随机入场 + ME 执行参数的 trailing stop.

    注意: ATR 必须从 1H K 线计算 (与事件回测一致),
    然后通过 hourly_atr_map 传入 1min 策略.
    """

    params = dict(
        initial_r=3.0,
        activation_r=0.5,
        trail_r=0.5,
        entry_interval=60,  # 每 60 根 1min bar (~1H) 尝试入场
        seed=42,
        hourly_atr_map=None,  # {1H_timestamp → ATR}, 从外部传入
    )

    def __init__(self):
        self.rng = np.random.RandomState(self.p.seed)
        self.bar_count = 0
        self.trades_log = []
        self._entry_price = None
        self._entry_atr = None
        self._direction = 0
        self._stop_loss = None
        self._trailing_active = False
        self._trailing_stop = None
        self._best_price = None
        self._entry_bar = 0

    def _get_hourly_atr(self):
        """从预计算的 1H ATR map 查找当前时间对应的 ATR."""
        if self.p.hourly_atr_map is None:
            return None
        dt = self.data.datetime.datetime(0)
        # 向下取整到小时
        hour_ts = dt.replace(minute=0, second=0, microsecond=0)
        val = self.p.hourly_atr_map.get(hour_ts)
        if val is None:
            # fallback: 找最近的前一个小时
            for delta_h in range(1, 4):
                prev_ts = hour_ts - timedelta(hours=delta_h)
                val = self.p.hourly_atr_map.get(prev_ts)
                if val is not None:
                    break
        return val

    def next(self):
        self.bar_count += 1

        # 如果有持仓 → 手动管理 trailing stop
        if self.position:
            self._manage_position()
            return

        # 无持仓 → 尝试入场
        if self.bar_count % self.p.entry_interval != 0:
            return

        atr = self._get_hourly_atr()
        if atr is None or atr <= 0:
            return

        # 随机方向
        direction = self.rng.choice([-1, 1])
        price = self.data.close[0]

        self._entry_price = price
        self._entry_atr = atr
        self._direction = direction
        self._stop_loss = (
            price - self.p.initial_r * atr
            if direction == 1
            else price + self.p.initial_r * atr
        )
        self._trailing_active = False
        self._trailing_stop = None
        self._best_price = price
        self._entry_bar = self.bar_count

        if direction == 1:
            self.buy()
        else:
            self.sell()

    def _manage_position(self):
        h = self.data.high[0]
        l = self.data.low[0]
        c = self.data.close[0]
        d = self._direction

        if d == 1:  # LONG
            # SL check
            if l <= self._stop_loss:
                self.close()
                pnl_r = (self._stop_loss - self._entry_price) / self._entry_atr
                self.trades_log.append(
                    {
                        "pnl_r": pnl_r,
                        "exit_reason": "stop_loss",
                        "bars": self.bar_count - self._entry_bar,
                    }
                )
                return

            # Update best
            if h > self._best_price:
                self._best_price = h

            # Activation
            activation_dist = self.p.activation_r * self._entry_atr
            if (
                not self._trailing_active
                and (self._best_price - self._entry_price) >= activation_dist
            ):
                self._trailing_active = True
                self._trailing_stop = (
                    self._best_price - self.p.trail_r * self._entry_atr
                )

            # Trailing
            if self._trailing_active:
                new_ts = self._best_price - self.p.trail_r * self._entry_atr
                if self._trailing_stop is None or new_ts > self._trailing_stop:
                    self._trailing_stop = new_ts
                if l <= self._trailing_stop:
                    self.close()
                    pnl_r = (self._trailing_stop - self._entry_price) / self._entry_atr
                    self.trades_log.append(
                        {
                            "pnl_r": pnl_r,
                            "exit_reason": "trailing_stop",
                            "bars": self.bar_count - self._entry_bar,
                        }
                    )

        else:  # SHORT
            if h >= self._stop_loss:
                self.close()
                pnl_r = (self._entry_price - self._stop_loss) / self._entry_atr
                self.trades_log.append(
                    {
                        "pnl_r": pnl_r,
                        "exit_reason": "stop_loss",
                        "bars": self.bar_count - self._entry_bar,
                    }
                )
                return

            if l < self._best_price:
                self._best_price = l

            activation_dist = self.p.activation_r * self._entry_atr
            if (
                not self._trailing_active
                and (self._entry_price - self._best_price) >= activation_dist
            ):
                self._trailing_active = True
                self._trailing_stop = (
                    self._best_price + self.p.trail_r * self._entry_atr
                )

            if self._trailing_active:
                new_ts = self._best_price + self.p.trail_r * self._entry_atr
                if self._trailing_stop is None or new_ts < self._trailing_stop:
                    self._trailing_stop = new_ts
                if h >= self._trailing_stop:
                    self.close()
                    pnl_r = (self._entry_price - self._trailing_stop) / self._entry_atr
                    self.trades_log.append(
                        {
                            "pnl_r": pnl_r,
                            "exit_reason": "trailing_stop",
                            "bars": self.bar_count - self._entry_bar,
                        }
                    )

    def stop(self):
        # 关闭残留持仓
        if self.position:
            c = self.data.close[0]
            pnl_r = (c - self._entry_price) / self._entry_atr * self._direction
            self.trades_log.append(
                {
                    "pnl_r": pnl_r,
                    "exit_reason": "end_of_data",
                    "bars": self.bar_count - self._entry_bar,
                }
            )


def load_1min_bars(symbol: str, months: list[str]) -> tuple[pd.DataFrame, dict]:
    """从 tick parquet 构建 1min OHLCV + 1H ATR map.

    返回:
        min_bars: 1min OHLCV (index = datetime)
        hourly_atr_map: {datetime → ATR} (14期 1H ATR)
    """
    dfs = []
    for m in months:
        path = Path("data/parquet_data") / f"{symbol}_{m}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dfs.append(df)

    ticks = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    min_bars = (
        ticks.groupby("timestamp")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
    )
    # Fill missing minutes
    full_idx = pd.date_range(min_bars.index[0], min_bars.index[-1], freq="1min")
    min_bars = min_bars.reindex(full_idx)
    min_bars = min_bars.ffill()
    min_bars.index.name = "datetime"

    # 1H OHLCV → 14H ATR (与 validate_backtest_vbt.py 一致)
    hourly = (
        min_bars.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h_high, h_low, h_close_prev = (
        hourly["high"],
        hourly["low"],
        hourly["close"].shift(1),
    )
    tr = pd.concat(
        [h_high - h_low, (h_high - h_close_prev).abs(), (h_low - h_close_prev).abs()],
        axis=1,
    ).max(axis=1)
    hourly_atr = tr.rolling(14).mean()

    # 转为 dict, key = naive datetime (与 backtrader datetime 对齐)
    hourly_atr_map = {}
    for ts, val in hourly_atr.dropna().items():
        hourly_atr_map[ts.to_pydatetime().replace(tzinfo=None)] = val

    print(f"  1H ATR sample: {list(hourly_atr.dropna().tail(3).items())}")
    return min_bars, hourly_atr_map


def run_backtrader_validation(symbol: str, months: list[str]):
    """用 backtrader 跑随机入场 + trailing stop (1H ATR)."""
    print(f"\n{'='*70}")
    print(f"🔬 Backtrader 验证: {symbol} 1min bars + 1H ATR + trailing stop")
    print(f"{'='*70}")

    min_bars, hourly_atr_map = load_1min_bars(symbol, months)
    print(f"  1min bars: {len(min_bars)}, hourly ATR entries: {len(hourly_atr_map)}")

    # backtrader 不能处理 tz-aware index, 去掉时区
    min_bars_naive = min_bars.copy()
    min_bars_naive.index = min_bars_naive.index.tz_localize(None)

    # backtrader data feed
    data = bt.feeds.PandasData(
        dataname=min_bars_naive,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        RandomTrailingStopStrategy,
        initial_r=3.0,
        activation_r=0.5,
        trail_r=0.5,
        entry_interval=60,
        seed=42,
        hourly_atr_map=hourly_atr_map,
    )
    cerebro.broker.setcash(1_000_000)
    cerebro.broker.setcommission(commission=0.0004)

    print("  Running backtrader...")
    results = cerebro.run()
    strat = results[0]

    trades = strat.trades_log
    if not trades:
        print("  ❌ No trades generated")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_r"] > 0).sum()
    wr = wins / n
    mean_r = df["pnl_r"].mean()

    print(f"\n  📊 Backtrader 结果:")
    print(f"     Trades:   {n}")
    print(f"     Win Rate: {wr:.1%}")
    print(f"     Mean R:   {mean_r:+.4f}")
    print(f"     Total R:  {df['pnl_r'].sum():.2f}")

    for reason, grp in df.groupby("exit_reason"):
        grp_wr = (grp["pnl_r"] > 0).mean()
        print(
            f"     {reason:20s}: {len(grp):4d}, "
            f"WR={grp_wr:.1%}, mean_r={grp['pnl_r'].mean():+.4f}"
        )

    return df


# ================================================================
# Part 2: Monte Carlo 随机游走理论验证
# ================================================================


def monte_carlo_random_walk(
    n_sims: int = 50_000,
    activation_r: float = 0.5,
    trail_r: float = 0.5,
    initial_r: float = 3.0,
    max_steps: int = 200_000,
    step_std: float = 0.05,
    seed: int = 42,
):
    """纯随机游走 Monte Carlo: 验证 trailing stop 的理论 WR.

    模拟: 价格每步 N(0, step_std), 无趋势, 纯随机
    出场: initial SL + trailing activation + trailing stop

    参数设计:
      - step_std=0.05, max_steps=200k → 足够覆盖 ±3R 的扩散范围
      - √max_steps * step_std ≈ √200000 * 0.05 ≈ 22.4 >> initial_r(3.0)
    """
    print(f"\n{'='*70}")
    print(f"📐 Monte Carlo 随机游走 ({n_sims:,} 模拟)")
    print(f"    initial_r={initial_r}, activation_r={activation_r}, trail_r={trail_r}")
    print(f"    step_std={step_std}, max_steps={max_steps:,}")
    print(f"{'='*70}")

    rng = np.random.RandomState(seed)
    results = []

    for _ in range(n_sims):
        price = 0.0  # 标准化: 入场价=0, ATR=1
        best_price = 0.0
        trailing_active = False
        trailing_stop = None
        exit_reason = "max_steps"
        pnl = 0.0

        for step in range(max_steps):
            price += rng.normal(0, step_std)

            # SL check (LONG only, symmetric for SHORT)
            if price <= -initial_r:
                pnl = -initial_r
                exit_reason = "stop_loss"
                break

            # Update best
            if price > best_price:
                best_price = price

            # Activation
            if not trailing_active and best_price >= activation_r:
                trailing_active = True
                trailing_stop = best_price - trail_r

            # Trailing
            if trailing_active:
                new_ts = best_price - trail_r
                if trailing_stop is None or new_ts > trailing_stop:
                    trailing_stop = new_ts
                if price <= trailing_stop:
                    pnl = trailing_stop
                    exit_reason = "trailing_stop"
                    break
        else:
            pnl = price
            exit_reason = "max_steps"

        results.append({"pnl": pnl, "exit_reason": exit_reason})

    df = pd.DataFrame(results)
    resolved = df[df["exit_reason"] != "max_steps"]
    unresolved = df[df["exit_reason"] == "max_steps"]
    wins = (df["pnl"] > 0).sum()
    wr = wins / len(df)
    mean_pnl = df["pnl"].mean()

    print(f"\n  📊 Monte Carlo 结果:")
    print(f"     Simulations: {n_sims:,}")
    print(f"     Resolved:    {len(resolved):,} ({len(resolved)/len(df):.1%})")
    print(f"     Unresolved:  {len(unresolved):,}")
    print(f"     Win Rate:    {wr:.1%}")
    print(f"     Mean PnL:    {mean_pnl:+.4f} R")
    if len(resolved) > 0:
        res_wins = (resolved["pnl"] > 0).sum()
        res_wr = res_wins / len(resolved)
        print(f"     WR (resolved only): {res_wr:.1%}")

    for reason, grp in df.groupby("exit_reason"):
        grp_wr = (grp["pnl"] > 0).mean()
        print(
            f"     {reason:20s}: {len(grp):5d}, "
            f"WR={grp_wr:.1%}, mean={grp['pnl'].mean():+.4f}"
        )

    # 理论值
    # 随机游走中, P(hit +a before -b) = b / (a + b)
    p_activate = initial_r / (activation_r + initial_r)
    print(f"\n  📐 理论参考:")
    print(f"     P(hit +{activation_r} before -{initial_r}) = {p_activate:.1%}")
    print(f"     (一旦 trailing 激活, 最低收益 ≈ 0R, 所以 WR ≈ P(activate))")
    print(
        f"     理论 WR ≈ {p_activate:.1%}, 实测 resolved WR = " f"{res_wr:.1%}"
        if len(resolved) > 0
        else "N/A"
    )

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--months", default="2025-11,2025-12,2026-01")
    p.add_argument("--monte-carlo", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    months = [m.strip() for m in args.months.split(",")]

    # Part 1: Backtrader
    bt_df = run_backtrader_validation(args.symbol, months)

    # Part 2: Monte Carlo
    mc_df = monte_carlo_random_walk(
        n_sims=args.monte_carlo,
        seed=args.seed,
    )

    # Summary
    print(f"\n{'='*70}")
    print("📋 三方对比")
    print(f"{'='*70}")
    bt_wr = (bt_df["pnl_r"] > 0).mean() if bt_df is not None else 0
    mc_wr = (mc_df["pnl"] > 0).mean()
    theory = 3.0 / (0.5 + 3.0)
    print(f"  Backtrader (真实 1min):  WR = {bt_wr:.1%}")
    print(f"  Monte Carlo (随机游走):  WR = {mc_wr:.1%}")
    print(f"  理论值 P(+0.5 before -3): WR = {theory:.1%}")
    print(f"  之前自实现 validate_backtest_vbt: WR = 88.3%")
    print(f"  事件回测 ME (旧配置):    WR = 71.9%")


if __name__ == "__main__":
    sys.exit(main() or 0)
