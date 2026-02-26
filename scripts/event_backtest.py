#!/usr/bin/env python3
"""
事件驱动回测 — 用 1min bar 精确模拟实盘持仓管理

与向量化回测 (backtest_execution_layer) 的区别:
  向量回测: entry_direction + 4h bar 级 trailing → 快速迭代
  事件回测: GenericLiveStrategy.decide() + 1min bar 7步持仓管理 → 实盘验证

数据流:
  1min bars + ticks → IFC.compute_features_dataframe → 信号时钟特征
  → GenericLiveStrategy.decide() → TradeIntent
  → PositionSimulator: 1min bar 逐 bar 持仓管理 (time/breakeven/trailing/SL/TP)
  → 报告

用法:
    # 单策略回测
    python scripts/event_backtest.py --strategy bpc --days 180

    # 指定 symbol
    python scripts/event_backtest.py --strategy me --symbols BTCUSDT,ETHUSDT --days 90

    # 与向量回测对比
    python scripts/event_backtest.py --strategy bpc --days 180 \
        --compare results/train_final_.../bpc/predictions.parquet

    # 导出交易明细
    python scripts/event_backtest.py --strategy bpc --days 180 --export trades.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.live.execution_profile_apply import pick_atr
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.portfolio.live_pcm import LivePCM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("event_backtest")


# ═════════════════════════════════════════════════════════════════════════════
# 1. ClosedTrade — 已关闭交易记录
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ClosedTrade:
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    atr_at_entry: float
    pnl_r: float  # PnL in R-multiples
    pnl_usd: float  # notional PnL (per-unit)
    exit_reason: str
    tier_name: str = ""
    evidence_score: float = 0.0
    bars_held: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. PositionSimulator — 复制实盘 _enforce_open_positions 逻辑
# ═════════════════════════════════════════════════════════════════════════════


class PositionSimulator:
    """
    持仓模拟器 — 调用 position_logic 共享模块，与实盘完全同一份代码。

    回测用 1min bar OHLC (保守假设 SL 优先):
      LONG: if low <= SL → 止损; elif high >= TP → 止盈
      SHORT: if high >= SL → 止损; elif low <= TP → 止盈
    """

    def __init__(self, default_bar_minutes: int = 240, max_positions: int = 1):
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.default_bar_minutes = default_bar_minutes
        self.max_positions = max_positions
        self.closed_trades: List[ClosedTrade] = []

    @property
    def has_positions(self) -> bool:
        return len(self._positions) > 0

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def open_position(
        self,
        intent: Any,
        entry_bar: Dict[str, Any],
        features: Dict[str, Any],
    ) -> Optional[str]:
        """从 TradeIntent + 当前 bar 创建虚拟持仓 (调用共享 build_position_dict)"""
        if len(self._positions) >= self.max_positions:
            return None

        pid = str(uuid.uuid4())[:12]

        entry_price = float(entry_bar.get("close", 0))
        atr = pick_atr(features) or float(entry_bar.get("atr", 0)) or 0.0

        # 解析 entry_time
        bar_ts = entry_bar.get("timestamp")
        if isinstance(bar_ts, str):
            entry_time = pd.Timestamp(bar_ts).to_pydatetime()
        elif isinstance(bar_ts, pd.Timestamp):
            entry_time = bar_ts.to_pydatetime()
        elif isinstance(bar_ts, datetime):
            entry_time = bar_ts
        else:
            entry_time = datetime.now(timezone.utc)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        # 调用共享模块构建持仓 dict
        pos = build_position_dict(
            intent=intent,
            entry_price=entry_price,
            atr=atr,
            bar_minutes=self.default_bar_minutes,
            entry_time=entry_time,
        )
        self._positions[pid] = pos
        return pid

    def update(self, bar_1min: Dict[str, Any]) -> List[ClosedTrade]:
        """用 1min bar 更新所有持仓 — 调用共享 enforce_position"""
        if not self._positions:
            return []

        bar_high = float(bar_1min.get("high", 0))
        bar_low = float(bar_1min.get("low", 0))
        bar_close = float(bar_1min.get("close", 0))

        bar_ts = bar_1min.get("timestamp")
        if isinstance(bar_ts, str):
            now = pd.Timestamp(bar_ts).to_pydatetime()
        elif isinstance(bar_ts, pd.Timestamp):
            now = bar_ts.to_pydatetime()
        elif isinstance(bar_ts, datetime):
            now = bar_ts
        else:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        closed = []
        to_remove = []

        for pid, pos in self._positions.items():
            # 调用共享 7 步持仓管理
            close_reason, exit_price = enforce_position(
                pos,
                price_high=bar_high,
                price_low=bar_low,
                price_close=bar_close,
                now=now,
                default_bar_minutes=self.default_bar_minutes,
            )

            if close_reason:
                entry_price = pos["entry_price"]
                pos_atr = pos.get("atr_at_entry", 0) or 0
                is_long = pos["side"] in {"LONG", "BUY"}
                pnl_usd = (
                    (exit_price - entry_price)
                    if is_long
                    else (entry_price - exit_price)
                )
                pnl_r = pnl_usd / pos_atr if pos_atr > 0 else 0.0

                trade = ClosedTrade(
                    symbol=pos.get("symbol", ""),
                    side=pos["side"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=pos["entry_time"],
                    exit_time=now,
                    atr_at_entry=pos_atr,
                    pnl_r=pnl_r,
                    pnl_usd=pnl_usd,
                    exit_reason=close_reason,
                    tier_name=pos.get("tier_name", ""),
                    evidence_score=pos.get("evidence_score", 0),
                    bars_held=pos.get("bars_counted", 0),
                )
                closed.append(trade)
                self.closed_trades.append(trade)
                to_remove.append(pid)

        for pid in to_remove:
            self._positions.pop(pid, None)

        # 计数存活持仓的 bar 数
        for pos in self._positions.values():
            pos["bars_counted"] = pos.get("bars_counted", 0) + 1

        return closed

    def force_close_all(self, price: float, now: datetime) -> List[ClosedTrade]:
        """回测结束时关闭所有持仓"""
        closed = []
        for pid, pos in list(self._positions.items()):
            is_long = pos["side"] in {"LONG", "BUY"}
            entry_price = pos["entry_price"]
            pos_atr = pos.get("atr_at_entry", 0) or 1
            pnl_usd = (price - entry_price) if is_long else (entry_price - price)
            pnl_r = pnl_usd / pos_atr if pos_atr > 0 else 0.0
            trade = ClosedTrade(
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                exit_price=price,
                entry_time=pos["entry_time"],
                exit_time=now,
                atr_at_entry=pos_atr,
                pnl_r=pnl_r,
                pnl_usd=pnl_usd,
                exit_reason="end_of_backtest",
                tier_name=pos.get("tier_name", ""),
                evidence_score=pos.get("evidence_score", 0),
                bars_held=pos.get("bars_counted", 0),
            )
            closed.append(trade)
            self.closed_trades.append(trade)
        self._positions.clear()
        return closed


# ═════════════════════════════════════════════════════════════════════════════
# 3. BacktestResult — 结果汇总
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    strategy: str
    trades: List[ClosedTrade] = field(default_factory=list)
    funnel: Dict[str, int] = field(default_factory=dict)
    per_symbol: Dict[str, List[ClosedTrade]] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def pnl_r_array(self) -> np.ndarray:
        return np.array([t.pnl_r for t in self.trades]) if self.trades else np.array([])

    @property
    def win_rate(self) -> float:
        arr = self.pnl_r_array
        return float(np.mean(arr > 0)) if len(arr) > 0 else 0.0

    @property
    def sharpe(self) -> float:
        arr = self.pnl_r_array
        if len(arr) < 2:
            return 0.0
        return (
            float(np.mean(arr) / np.std(arr, ddof=1))
            if np.std(arr, ddof=1) > 0
            else 0.0
        )

    @property
    def mean_r(self) -> float:
        arr = self.pnl_r_array
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    @property
    def total_r(self) -> float:
        return float(np.sum(self.pnl_r_array)) if self.trades else 0.0

    @property
    def max_drawdown_r(self) -> float:
        arr = self.pnl_r_array
        if len(arr) == 0:
            return 0.0
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd))

    def print_report(self):
        """输出汇总报告"""
        arr = self.pnl_r_array
        print()
        print("=" * 72)
        print(f"  📊 事件驱动回测报告: {self.strategy.upper()}")
        print("=" * 72)
        print(f"  交易数:       {self.n_trades}")
        print(f"  胜率:         {self.win_rate:.1%}")
        print(f"  Sharpe (R):   {self.sharpe:.4f}")
        print(f"  Mean R:       {self.mean_r:.4f}")
        print(f"  Total R:      {self.total_r:.2f}")
        print(f"  Max DD (R):   {self.max_drawdown_r:.2f}")
        if len(arr) > 0:
            print(f"  Best trade:   {arr.max():.2f}R")
            print(f"  Worst trade:  {arr.min():.2f}R")

        # 出场原因分布
        reasons = defaultdict(int)
        for t in self.trades:
            reasons[t.exit_reason] += 1
        if reasons:
            print(f"\n  出场原因:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:20s}: {count:4d} ({count/self.n_trades:.1%})")

        # 每 symbol 明细
        if self.per_symbol:
            print(f"\n  Per-symbol:")
            print(
                f"    {'Symbol':12s} {'Trades':>7s} {'WinRate':>8s} {'MeanR':>8s} {'TotalR':>8s}"
            )
            for sym in sorted(self.per_symbol.keys()):
                strades = self.per_symbol[sym]
                sarr = np.array([t.pnl_r for t in strades])
                swr = float(np.mean(sarr > 0)) if len(sarr) > 0 else 0.0
                print(
                    f"    {sym:12s} {len(strades):>7d} {swr:>7.1%} "
                    f"{np.mean(sarr):>8.3f} {np.sum(sarr):>8.2f}"
                )

        # 漏斗
        if self.funnel:
            print(f"\n  信号漏斗:")
            for k, v in self.funnel.items():
                print(f"    {k:30s}: {v}")

        print("=" * 72)

    def export_trades_csv(self, path: str):
        """导出交易明细 CSV"""
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "atr": t.atr_at_entry,
                    "pnl_r": round(t.pnl_r, 4),
                    "pnl_usd": round(t.pnl_usd, 4),
                    "exit_reason": t.exit_reason,
                    "tier": t.tier_name,
                    "evidence": round(t.evidence_score, 4),
                    "bars_held": t.bars_held,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"\n  📤 Trades exported: {len(df)} rows → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. 辅助函数
# ═════════════════════════════════════════════════════════════════════════════


def row_to_features(row: pd.Series) -> Dict[str, float]:
    """DataFrame 行 → 特征 dict"""
    features = {}
    for k, v in row.items():
        try:
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[str(k)] = float(v)
        except (ValueError, TypeError):
            continue
    return features


def _get_bar_minutes(strategy: str) -> int:
    """策略 → 信号时钟分钟数"""
    return {"me": 60, "fer": 240, "bpc": 240}.get(strategy.lower(), 240)


def _get_timeframe(strategy: str) -> str:
    """策略 → timeframe string"""
    return {"me": "60T", "fer": "240T", "bpc": "240T"}.get(strategy.lower(), "240T")


# ═════════════════════════════════════════════════════════════════════════════
# 5. EventBacktester — 主回测类
# ═════════════════════════════════════════════════════════════════════════════


class EventBacktester:
    """
    事件驱动回测主类 — 完全模拟实盘环境

    与实盘一致的架构:
      1. LivePCM 仲裁 (全局 slot 控制, 优先级排序)
      2. GenericLiveStrategy.decide() 信号生成
      3. PositionSimulator: 1min bar 持仓管理
      4. 跨 symbol 时间线交叉处理 (同实盘顺序)

    用法:
        bt = EventBacktester(strategy="bpc", live_root="live/highcap")
        result = bt.run(symbols=["BTCUSDT", ...], days=180)
        result.print_report()
    """

    def __init__(
        self,
        strategy: str,
        live_root: str = "live/highcap",
        strategies_root: Optional[str] = None,
    ):
        self.strategy_name = strategy
        self.live_root = live_root
        self.strategies_root = strategies_root or f"{live_root}/config/strategies"
        self.timeframe = _get_timeframe(strategy)
        self.bar_minutes = _get_bar_minutes(strategy)

        # 初始化 GenericLiveStrategy
        self.strat = GenericLiveStrategy(
            strategy_name=strategy,
            strategies_root=self.strategies_root,
            primary_timeframe=self.timeframe,
            bar_minutes=self.bar_minutes,
        )

        # LivePCM 仲裁器 (同实盘: 读取 constitution slot 配置)
        constitution_yaml = str(
            Path(self.live_root) / "config" / "constitution" / "constitution.yaml"
        )
        pcm_regime_yaml = str(Path("config") / "pcm_regime.yaml")
        self._simulators: Dict[str, PositionSimulator] = {}
        self.pcm = LivePCM(
            constitution_yaml=constitution_yaml,
            regime_config_path=(
                pcm_regime_yaml if Path(pcm_regime_yaml).exists() else None
            ),
            get_open_slot_count=self._global_open_count,
        )
        self.pcm.register(strategy, self.strat, timeframe=self.timeframe)

        # 特征计算器
        archetypes_dir = str(Path(self.strategies_root) / strategy / "archetypes")
        self.feature_computer = IncrementalFeatureComputer(
            primary_timeframe=self.timeframe,
            archetypes_dir=archetypes_dir,
        )
        # 禁用 live_feature_set 过滤 — 保留所有计算出的特征供 decide() 使用
        self.feature_computer.live_feature_set = None

    def _global_open_count(self) -> int:
        """跨所有 symbol 的当前持仓数 (供 PCM slot 检查用)"""
        return sum(sim.position_count for sim in self._simulators.values())

    def run(
        self,
        symbols: List[str],
        days: int = 180,
        warmup_days: int = 100,
    ) -> BacktestResult:
        """
        运行事件驱动回测 — 跨 symbol 时间线交叉处理
        """
        result = BacktestResult(strategy=self.strategy_name)
        funnel = defaultdict(int)

        storage = StorageManager(f"{self.live_root}/data")
        end_date = datetime.now().strftime("%Y-%m-%d")
        warmup_start = (datetime.now() - timedelta(days=days + warmup_days)).strftime(
            "%Y-%m-%d"
        )
        test_start = datetime.now() - timedelta(days=days)
        test_start_ts = pd.Timestamp(test_start, tz="UTC")

        # ── Phase 1: 加载数据 + 计算特征 ──
        sym_data: Dict[str, Dict[str, Any]] = {}  # {sym: {test_df, bars_1min_test}}
        all_quantile_dfs = []

        for sym in symbols:
            logger.info(f"{'='*60}")
            logger.info(f"Loading {sym}")
            t0 = time.time()
            bars_1min = storage.bar_1min.load_range(sym, warmup_start, end_date)
            ticks_1min = storage.ticks.load_range(sym, warmup_start, end_date)
            logger.info(
                f"  Data: {len(bars_1min)} 1min bars, {len(ticks_1min)} ticks "
                f"({time.time()-t0:.1f}s)"
            )
            if len(bars_1min) < 100:
                logger.warning(f"  {sym}: bars 不足, 跳过")
                continue

            t0 = time.time()
            features_df = self.feature_computer.compute_features_dataframe(
                bars_1min=bars_1min,
                ticks_1min=ticks_1min,
                primary_timeframe=self.timeframe,
            )
            logger.info(
                f"  Features: {len(features_df)} rows × {len(features_df.columns)} cols "
                f"({time.time()-t0:.1f}s)"
            )
            if features_df.empty:
                continue

            all_quantile_dfs.append(features_df)

            # 确保时区
            features_df.index = pd.to_datetime(features_df.index, utc=True)
            test_df = features_df[features_df.index >= test_start_ts]
            if test_df.empty:
                continue

            # 准备 1min bars 索引
            bars_1min_idx = bars_1min.copy()
            if not isinstance(bars_1min_idx.index, pd.DatetimeIndex):
                if "timestamp" in bars_1min_idx.columns:
                    bars_1min_idx.index = pd.to_datetime(
                        bars_1min_idx["timestamp"], utc=True
                    )
            if bars_1min_idx.index.tz is None:
                bars_1min_idx.index = bars_1min_idx.index.tz_localize("UTC")
            bars_1min_test = bars_1min_idx[bars_1min_idx.index >= test_start_ts]

            sym_data[sym] = {
                "test_df": test_df,
                "bars_1min_test": bars_1min_test,
            }
            logger.info(
                f"  Test: {test_df.index.min()} → {test_df.index.max()}, {len(test_df)} bars"
            )

        if not sym_data:
            logger.warning("No valid symbols")
            return result

        # 设置 Evidence 分位数 (所有 symbol 共享)
        if all_quantile_dfs:
            combined_quantiles = pd.concat(all_quantile_dfs, axis=0)
            self.pcm.set_quantiles_from_df(combined_quantiles)

        # ── Phase 2: 跨 symbol 时间线交叉处理 ──
        # 合并所有 symbol 的信号时间戳 → 统一时间线
        timeline_events: List[Tuple[pd.Timestamp, str]] = []  # (ts, symbol)
        for sym, data in sym_data.items():
            for ts in data["test_df"].index:
                timeline_events.append((ts, sym))
        timeline_events.sort(key=lambda x: x[0])

        # 初始化 per-symbol simulators
        for sym in sym_data:
            self._simulators[sym] = PositionSimulator(
                default_bar_minutes=self.bar_minutes
            )

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Timeline: {len(timeline_events)} events across {len(sym_data)} symbols"
        )
        logger.info(f"PCM max_slots={self.pcm._max_slots}")

        # 遍历统一时间线
        prev_ts: Dict[str, pd.Timestamp] = {}  # 每个 symbol 上次信号时间

        for ts, sym in timeline_events:
            data = sym_data[sym]
            simulator = self._simulators[sym]
            test_df = data["test_df"]
            bars_1min_test = data["bars_1min_test"]

            row = test_df.loc[ts]
            features = row_to_features(row)
            funnel["total_signals_checked"] += 1

            # 先用 1min bars 更新该 symbol 的持仓 (上次信号 → 当前信号)
            if sym in prev_ts and simulator.has_positions:
                mask = (bars_1min_test.index > prev_ts[sym]) & (
                    bars_1min_test.index <= ts
                )
                for bar_ts, bar_row in bars_1min_test[mask].iterrows():
                    bar_dict = {
                        "timestamp": bar_ts,
                        "open": float(bar_row.get("open", 0)),
                        "high": float(bar_row.get("high", 0)),
                        "low": float(bar_row.get("low", 0)),
                        "close": float(bar_row.get("close", 0)),
                    }
                    simulator.update(bar_dict)

            # LivePCM.decide() — 全局 slot 控制 + 仲裁
            intents = self.pcm.decide(features=features, symbol=sym)

            if intents:
                funnel["signals_generated"] += 1
                intent = intents[0]
                entry_bar = {
                    "close": features.get("close", 0),
                    "high": features.get("high", 0),
                    "low": features.get("low", 0),
                    "open": features.get("open", 0),
                    "timestamp": ts,
                    "atr": features.get("atr", 0),
                }
                opened = simulator.open_position(intent, entry_bar, features)
                if opened is None:
                    funnel["reject_max_positions"] += 1
            else:
                # 诊断拒绝原因
                last_funnel = getattr(self.strat, "_last_funnel", {})
                if not last_funnel.get("direction", True):
                    funnel["reject_no_direction"] += 1
                elif not last_funnel.get("gate", True):
                    funnel["reject_gate_deny"] += 1
                elif not last_funnel.get("entry_filter", True):
                    funnel["reject_entry_filter"] += 1
                else:
                    funnel["reject_pcm_slot_full"] += 1

            prev_ts[sym] = ts

        # ── Phase 3: 处理最后一个信号后的 1min bars + 关闭残留持仓 ──
        for sym, simulator in self._simulators.items():
            data = sym_data[sym]
            bars_1min_test = data["bars_1min_test"]
            test_df = data["test_df"]

            # 最后一个信号后的 1min bars
            if sym in prev_ts and simulator.has_positions:
                last_sig = prev_ts[sym]
                remaining = bars_1min_test[bars_1min_test.index > last_sig]
                for bar_ts, bar_row in remaining.iterrows():
                    bar_dict = {
                        "timestamp": bar_ts,
                        "open": float(bar_row.get("open", 0)),
                        "high": float(bar_row.get("high", 0)),
                        "low": float(bar_row.get("low", 0)),
                        "close": float(bar_row.get("close", 0)),
                    }
                    simulator.update(bar_dict)

            # 关闭残留持仓
            if simulator.has_positions:
                last_close = float(test_df.iloc[-1].get("close", 0))
                last_time = test_df.index[-1].to_pydatetime()
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                simulator.force_close_all(last_close, last_time)

            sym_trades = simulator.closed_trades
            result.trades.extend(sym_trades)
            result.per_symbol[sym] = sym_trades
            logger.info(f"  {sym}: {len(sym_trades)} trades")

        result.trades.sort(key=lambda t: t.entry_time)
        result.funnel = dict(funnel)
        return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="事件驱动回测 — 用 1min bar 精确模拟实盘持仓管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        "-s",
        required=True,
        help="策略名 (bpc/me/fer)",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT",
        help="逗号分隔的交易对",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="回测天数 (默认 180)",
    )
    parser.add_argument(
        "--live-root",
        default="live/highcap",
        help="实盘数据根目录",
    )
    parser.add_argument(
        "--strategies-root",
        default=None,
        help="策略配置目录 (默认 {live-root}/config/strategies)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="导出交易明细 CSV 路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存 JSON 结果路径",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 72)
    print("  🔬 事件驱动回测")
    print("=" * 72)
    print(f"  策略:    {args.strategy}")
    print(f"  Symbols: {symbols}")
    print(f"  天数:    {args.days}")
    print(f"  数据源:  {args.live_root}/data")
    print("=" * 72)

    bt = EventBacktester(
        strategy=args.strategy,
        live_root=args.live_root,
        strategies_root=args.strategies_root,
    )

    result = bt.run(symbols=symbols, days=args.days)

    result.print_report()

    if args.export:
        result.export_trades_csv(args.export)

    if args.output:
        _save_json(result, args.output)

    return 0


def _save_json(result: BacktestResult, path: str):
    """保存 JSON 结果"""
    out = {
        "strategy": result.strategy,
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 4),
        "sharpe_r": round(result.sharpe, 4),
        "mean_r": round(result.mean_r, 4),
        "total_r": round(result.total_r, 4),
        "max_drawdown_r": round(result.max_drawdown_r, 4),
        "funnel": result.funnel,
        "per_symbol": {
            sym: {
                "trades": len(trades),
                "total_r": round(sum(t.pnl_r for t in trades), 4),
            }
            for sym, trades in result.per_symbol.items()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Results saved → {path}")


if __name__ == "__main__":
    sys.exit(main())
