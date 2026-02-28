#!/usr/bin/env python3
"""
事件驱动回测 — 用 1min bar 精确模拟实盘多策略持仓管理

与向量化回测 (backtest_execution_layer) 的区别:
  向量回测: entry_direction + 4h bar 级 trailing → 快速迭代
  事件回测: GenericLiveStrategy.decide() + 1min bar 7步持仓管理 → 实盘验证

支持多策略 PCM 仲裁 (同实盘 run_live.py):
  - 多策略注册 (BPC + FER + ME)
  - 多 timeframe 特征计算 (BPC/FER=240T, ME=60T)
  - LivePCM 优先级仲裁 + Regime 感知缩放
  - 跨 symbol slot 控制

数据流:
  1min bars + ticks → IFC.compute_features_dataframe → 多 timeframe 信号时钟特征
  → LivePCM.decide(features_by_timeframe) → 多策略仲裁 → TradeIntent
  → PositionSimulator: 1min bar 逐 bar 持仓管理 (time/breakeven/trailing/SL/TP)
  → 报告

用法:
    # 多策略联合回测 (推荐 — 与 PCM 向量回测对齐)
    python scripts/event_backtest.py --strategy bpc,fer,me --days 180

    # 单策略回测
    python scripts/event_backtest.py --strategy fer --days 180

    # 指定 symbol + 导出
    python scripts/event_backtest.py --strategy bpc,fer,me --symbols BTCUSDT,ETHUSDT --days 90 --export trades.csv
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
from src.data_tools.data_handler import DataHandler
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.portfolio.live_pcm import LivePCM

# order_management integration (optional — only when --db is provided)
try:
    from src.order_management.mock_binance_api import MockBinanceAPI
    from src.order_management.storage import Storage as OMStorage
    from src.order_management.order_manager import OrderManager
    from src.order_management.position_manager import PositionManager
    from src.order_management.models import (
        PositionSide as OMPositionSide,
        OrderSide as OMOrderSide,
        OrderType as OMOrderType,
    )

    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False

try:
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import ColumnDataSource, HoverTool, Span, Range1d, Div
    from bokeh.layouts import column as bk_column
    from bokeh.resources import INLINE as BK_RESOURCES
    from bokeh.embed import file_html as bk_file_html

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

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
        # order_management 集成 (由 EventBacktester 注入)
        self._om_bridge: Optional["OMBridge"] = None

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
        bar_minutes: Optional[int] = None,
    ) -> Optional[str]:
        """从 TradeIntent + 当前 bar 创建虚拟持仓 (调用共享 build_position_dict)"""
        if len(self._positions) >= self.max_positions:
            return None

        pid = str(uuid.uuid4())[:12]

        entry_price = float(entry_bar.get("close", 0))
        # 直接取 "atr" 键 — 不用 pick_atr() 因为它会误匹配 macd_atr 等特征
        atr = float(entry_bar.get("atr", 0)) or float(features.get("atr", 0)) or 0.0

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
            bar_minutes=bar_minutes or self.default_bar_minutes,
            entry_time=entry_time,
        )
        self._positions[pid] = pos

        # 写入 order_management DB
        if self._om_bridge:
            self._om_bridge.record_open(
                pid=pid,
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                size=intent.size if hasattr(intent, "size") else 1.0,
                atr=atr,
                stop_loss=pos.get("stop_loss"),
                take_profit=pos.get("take_profit"),
                archetype=pos.get("archetype", ""),
                entry_time=entry_time,
            )
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
                # 用 initial_risk_distance (= initial_r × ATR) 归一化 R-multiple
                # 与研究回测 (backtest_execution_layer.py) 保持一致
                risk = (
                    pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
                )
                is_long = pos["side"] in {"LONG", "BUY"}
                pnl_usd = (
                    (exit_price - entry_price)
                    if is_long
                    else (entry_price - exit_price)
                )
                pnl_r = pnl_usd / risk if risk > 0 else 0.0

                trade = ClosedTrade(
                    symbol=pos.get("symbol", ""),
                    side=pos["side"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=pos["entry_time"],
                    exit_time=now,
                    atr_at_entry=pos.get("atr_at_entry", 0),
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

                # 写入 order_management DB
                if self._om_bridge:
                    self._om_bridge.record_close(
                        pid=pid,
                        exit_price=exit_price,
                        exit_time=now,
                        exit_reason=close_reason,
                        pnl_r=pnl_r,
                    )

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
            risk = pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 1
            pnl_usd = (price - entry_price) if is_long else (entry_price - price)
            pnl_r = pnl_usd / risk if risk > 0 else 0.0
            trade = ClosedTrade(
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                exit_price=price,
                entry_time=pos["entry_time"],
                exit_time=now,
                atr_at_entry=pos.get("atr_at_entry", 0),
                pnl_r=pnl_r,
                pnl_usd=pnl_usd,
                exit_reason="end_of_backtest",
                tier_name=pos.get("tier_name", ""),
                evidence_score=pos.get("evidence_score", 0),
                bars_held=pos.get("bars_counted", 0),
            )
            closed.append(trade)
            self.closed_trades.append(trade)

            # 写入 order_management DB
            if self._om_bridge:
                self._om_bridge.record_close(
                    pid=pid,
                    exit_price=price,
                    exit_time=now,
                    exit_reason="end_of_backtest",
                    pnl_r=pnl_r,
                )
        self._positions.clear()
        return closed


# ═══════════════════════════════════════════════════════════════════════════
# 2b. OMBridge — order_management 写入桥接器
# ═══════════════════════════════════════════════════════════════════════════


class OMBridge:
    """将回测交易写入 order_management SQLite DB。

    创建时初始化 MockBinanceAPI + Storage + OrderManager + PositionManager。
    PositionSimulator 在开仓/平仓时调用 record_open / record_close。
    """

    def __init__(self, db_path: str):
        if not OM_AVAILABLE:
            raise RuntimeError(
                "order_management 模块不可用, 请检查 src/order_management"
            )
        self.db_path = db_path
        self.mock_api = MockBinanceAPI()
        self.storage = OMStorage(db_path)
        self.order_manager = OrderManager(self.storage, self.mock_api)
        self.position_manager = PositionManager(self.storage, self.mock_api)
        # pid → om_position_id 映射
        self._pid_map: Dict[str, str] = {}
        logger.info(f"OMBridge initialized: {db_path}")

    def record_open(
        self,
        pid: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        atr: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        archetype: str,
        entry_time: datetime,
    ) -> None:
        """开仓时写入 DB (order + position)。"""
        try:
            self.mock_api.set_price(symbol, entry_price)
            om_side = (
                OMPositionSide.LONG if side in ("LONG", "BUY") else OMPositionSide.SHORT
            )
            order_side = (
                OMOrderSide.BUY if side in ("LONG", "BUY") else OMOrderSide.SELL
            )

            # 下单
            order = self.order_manager.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OMOrderType.MARKET,
                quantity=size,
                price=entry_price,
            )

            # 创建仓位
            position = self.position_manager.create_position(
                symbol=symbol,
                side=om_side,
                entry_price=entry_price,
                size=size,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                strategy_id=archetype,
                archetype=archetype,
                notes=f"backtest|atr={atr:.4f}",
            )
            self._pid_map[pid] = position.position_id
        except Exception as e:
            logger.warning(f"OMBridge.record_open failed: {e}")

    def record_close(
        self,
        pid: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        pnl_r: float,
    ) -> None:
        """平仓时写入 DB (close position + exit order)。"""
        om_pid = self._pid_map.get(pid)
        if not om_pid:
            return
        try:
            pos = self.position_manager.get_position(om_pid)
            if not pos or pos.status.value == "closed":
                return
            self.mock_api.set_price(pos.symbol, exit_price)
            self.position_manager.close_position(
                position_id=om_pid,
                price=exit_price,
                reason=f"{exit_reason}|pnl_r={pnl_r:.3f}",
            )
        except Exception as e:
            logger.warning(f"OMBridge.record_close failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    strategy: str
    trades: List[ClosedTrade] = field(default_factory=list)
    funnel: Dict[str, int] = field(default_factory=dict)
    per_symbol: Dict[str, List[ClosedTrade]] = field(default_factory=dict)
    # 1min bar data per symbol (for trading map)
    bars_1min: Dict[str, pd.DataFrame] = field(default_factory=dict)

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
    事件驱动回测主类 — 完全模拟实盘多策略环境

    与实盘一致的架构:
      1. LivePCM 仲裁 (全局 slot 控制, 优先级排序, Regime 感知)
      2. 多策略 GenericLiveStrategy.decide() 信号生成 (BPC + FER + ME)
      3. 多 timeframe 特征计算 (BPC/FER=240T, ME=60T)
      4. PositionSimulator: 1min bar 持仓管理
      5. 跨 symbol 时间线交叉处理 (同实盘顺序)

    用法:
        bt = EventBacktester(strategies=["bpc","fer","me"], live_root="live/highcap")
        result = bt.run(symbols=["BTCUSDT", ...], days=180)
        result.print_report()
    """

    def __init__(
        self,
        strategies: List[str],
        live_root: str = "live/highcap",
        strategies_root: Optional[str] = None,
        db_path: Optional[str] = None,
        data_path: Optional[str] = None,
    ):
        self.strategy_names = [s.lower().strip() for s in strategies]
        self.live_root = live_root
        self.data_path = data_path  # 研究数据目录 (e.g. data/parquet_data)
        self.strategies_root = strategies_root or "config/strategies"

        # Per-strategy timeframe 映射
        self._tf_map: Dict[str, str] = {}  # {strategy: "240T"}
        self._bm_map: Dict[str, int] = {}  # {strategy: 240}
        for s in self.strategy_names:
            self._tf_map[s] = _get_timeframe(s)
            self._bm_map[s] = _get_bar_minutes(s)

        # 主 bar 分钟数 (position simulator default)
        self._primary_bar_minutes = max(self._bm_map.values())

        # order_management 集成 (可选)
        self._om_bridge: Optional[OMBridge] = None
        if db_path and OM_AVAILABLE:
            self._om_bridge = OMBridge(db_path)

        # 初始化 GenericLiveStrategy — 每策略一个
        self._strats: Dict[str, GenericLiveStrategy] = {}
        for s in self.strategy_names:
            self._strats[s] = GenericLiveStrategy(
                strategy_name=s,
                strategies_root=self.strategies_root,
                primary_timeframe=self._tf_map[s],
                bar_minutes=self._bm_map[s],
            )

        # LivePCM 仲裁器 (同实盘: 读取 constitution slot 配置)
        constitution_yaml = str(Path("config") / "constitution" / "constitution.yaml")
        pcm_regime_yaml = str(Path("config") / "pcm_regime.yaml")
        self._simulators: Dict[str, PositionSimulator] = {}
        self.pcm = LivePCM(
            constitution_yaml=constitution_yaml,
            regime_config_path=(
                pcm_regime_yaml if Path(pcm_regime_yaml).exists() else None
            ),
            get_open_slot_count=self._global_open_count,
        )
        for s in self.strategy_names:
            self.pcm.register(s, self._strats[s], timeframe=self._tf_map[s])

        # 特征计算器 — 按 unique timeframe 分组 (同 run_live.py)
        # BPC+FER 共享 240T FC，ME 独立 60T FC
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        unique_tfs = sorted(set(self._tf_map.values()))
        self._feature_computers: Dict[str, IncrementalFeatureComputer] = {}

        for tf in unique_tfs:
            tf_strats = [s for s in self.strategy_names if self._tf_map[s] == tf]
            first = tf_strats[0]
            archetypes_dir = str(Path(self.strategies_root) / first / "archetypes")
            fc = IncrementalFeatureComputer(
                primary_timeframe=tf,
                archetypes_dir=archetypes_dir,
            )
            # 合并同 timeframe 其他策略的特征集 (同 run_live.py 4H FC)
            for extra in tf_strats[1:]:
                extra_dir = str(Path(self.strategies_root) / extra / "archetypes")
                try:
                    extra_feat_set, extra_feat_nodes = extract_features_from_archetypes(
                        extra_dir
                    )
                    if fc.live_feature_set:
                        fc.live_feature_set |= extra_feat_set
                    fc.live_feature_nodes = sorted(
                        set(fc.live_feature_nodes) | set(extra_feat_nodes)
                    )
                except Exception as e:
                    logger.warning(f"  Feature merge for {extra} failed: {e}")
            # 禁用 live_feature_set 过滤 — 保留所有计算出的特征
            fc.live_feature_set = None
            self._feature_computers[tf] = fc

    def _load_research_data(
        self, sym: str, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """从研究数据目录 (data/parquet_data) 加载 1min bars + ticks

        与 compare_same_data.py 的实盘路径加载逻辑一致:
          DataHandler.load_ohlcv(timeframe="1T") → 1min bars
          glob {SYMBOL}_*.parquet → ticks
        """
        data_root = Path(self.data_path)

        # 1. 加载 1min bars (resample from tick data via DataHandler)
        dh = DataHandler(str(data_root))
        bars_1min = dh.load_ohlcv(
            symbol=sym, timeframe="1T", start_date=start_date, end_date=end_date
        )
        if not bars_1min.empty:
            bars_1min.index = pd.to_datetime(bars_1min.index, utc=True)
            # 列名适配: buy_qty → buy_volume, sell_qty → sell_volume
            col_rename = {"buy_qty": "buy_volume", "sell_qty": "sell_volume"}
            bars_1min = bars_1min.rename(
                columns={k: v for k, v in col_rename.items() if k in bars_1min.columns}
            )
            if "timestamp" not in bars_1min.columns:
                bars_1min["timestamp"] = bars_1min.index

        # 2. 加载 ticks (直接读 parquet 原始数据)
        tick_frames = []
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        for fp in sorted(data_root.glob(f"{sym}_*.parquet")):
            try:
                df_tick = pd.read_parquet(fp)
                if "price" in df_tick.columns and "volume" in df_tick.columns:
                    tick_frames.append(df_tick)
            except Exception:
                pass
        if tick_frames:
            ticks_1min = pd.concat(tick_frames, ignore_index=True)
            ticks_1min["timestamp"] = pd.to_datetime(ticks_1min["timestamp"], utc=True)
            ticks_1min = ticks_1min[
                (ticks_1min["timestamp"] >= start_ts)
                & (ticks_1min["timestamp"] <= end_ts)
            ]
        else:
            ticks_1min = pd.DataFrame()

        return bars_1min, ticks_1min

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
        运行事件驱动回测 — 多策略 + 多 timeframe + 跨 symbol 时间线交叉处理
        """
        result = BacktestResult(strategy="+".join(self.strategy_names))
        funnel = defaultdict(int)

        end_date = datetime.now().strftime("%Y-%m-%d")
        warmup_start = (datetime.now() - timedelta(days=days + warmup_days)).strftime(
            "%Y-%m-%d"
        )
        test_start = datetime.now() - timedelta(days=days)
        test_start_ts = pd.Timestamp(test_start, tz="UTC")

        # 数据源: --data-path (研究数据) 或 StorageManager (实盘数据)
        use_research = self.data_path is not None
        storage = None
        if not use_research:
            storage = StorageManager(f"{self.live_root}/data")

        # ── Phase 1: 加载数据 + 按 timeframe 计算特征 ──
        sym_data: Dict[str, Dict[str, Any]] = {}
        quantile_dfs_by_tf: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        for sym in symbols:
            logger.info(f"{'='*60}")
            logger.info(f"Loading {sym}")
            t0 = time.time()

            if use_research:
                # ── 研究数据路径: DataHandler → 1min bars + ticks ──
                bars_1min, ticks_1min = self._load_research_data(
                    sym, warmup_start, end_date
                )
            else:
                # ── 实盘数据路径: StorageManager ──
                bars_1min = storage.bar_1min.load_range(sym, warmup_start, end_date)
                ticks_1min = storage.ticks.load_range(sym, warmup_start, end_date)

            logger.info(
                f"  Data: {len(bars_1min)} 1min bars, {len(ticks_1min)} ticks "
                f"({time.time()-t0:.1f}s)"
            )
            if len(bars_1min) < 100:
                logger.warning(f"  {sym}: bars 不足, 跳过")
                continue

            tf_features: Dict[str, pd.DataFrame] = {}
            for tf, fc in self._feature_computers.items():
                t0 = time.time()
                features_df = fc.compute_features_dataframe(
                    bars_1min=bars_1min,
                    ticks_1min=ticks_1min,
                    primary_timeframe=tf,
                )
                logger.info(
                    f"  Features [{tf}]: {len(features_df)} rows × "
                    f"{len(features_df.columns)} cols ({time.time()-t0:.1f}s)"
                )
                if features_df.empty:
                    continue

                features_df.index = pd.to_datetime(features_df.index, utc=True)
                quantile_dfs_by_tf[tf].append(features_df)

                test_df = features_df[features_df.index >= test_start_ts]
                if not test_df.empty:
                    tf_features[tf] = test_df

            if not tf_features:
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
                "tf_features": tf_features,
                "bars_1min_test": bars_1min_test,
            }
            for tf, tdf in tf_features.items():
                logger.info(
                    f"  Test [{tf}]: {tdf.index.min()} → {tdf.index.max()}, "
                    f"{len(tdf)} bars"
                )

        if not sym_data:
            logger.warning("No valid symbols")
            return result

        # 设置 per-strategy Evidence 分位数 (从对应 timeframe 特征)
        for s_name, s_obj in self._strats.items():
            tf = self._tf_map[s_name]
            if tf in quantile_dfs_by_tf and quantile_dfs_by_tf[tf]:
                combined = pd.concat(quantile_dfs_by_tf[tf], axis=0)
                s_obj.set_quantiles_from_df(combined)
                logger.info(f"  Quantiles: {s_name} from {tf} ({len(combined)} rows)")

        # ── Phase 2: 构建统一时间线 (多 timeframe union) ──
        timeline_events: List[Tuple[pd.Timestamp, str, Dict[str, pd.Series]]] = []
        for sym, data in sym_data.items():
            tf_features = data["tf_features"]
            ts_to_tfs: Dict[pd.Timestamp, set] = defaultdict(set)
            for tf, test_df in tf_features.items():
                for ts in test_df.index:
                    ts_to_tfs[ts].add(tf)

            for ts in sorted(ts_to_tfs.keys()):
                tf_rows = {}
                for tf in ts_to_tfs[ts]:
                    tf_rows[tf] = tf_features[tf].loc[ts]
                timeline_events.append((ts, sym, tf_rows))

        timeline_events.sort(key=lambda x: x[0])

        # 初始化 per-symbol simulators
        for sym in sym_data:
            sim = PositionSimulator(default_bar_minutes=self._primary_bar_minutes)
            if self._om_bridge:
                sim._om_bridge = self._om_bridge
            self._simulators[sym] = sim

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Timeline: {len(timeline_events)} events across {len(sym_data)} symbols"
        )
        logger.info(f"Strategies: {', '.join(self.strategy_names)}")
        logger.info(f"PCM max_slots={self.pcm._max_slots}")

        # ── Phase 3: 遍历统一时间线 ──
        prev_ts: Dict[str, pd.Timestamp] = {}

        for ts, sym, tf_rows in timeline_events:
            simulator = self._simulators[sym]
            bars_1min_test = sym_data[sym]["bars_1min_test"]
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

            # 构建 features_by_timeframe 供 PCM 路由
            features_by_tf: Dict[str, Dict[str, float]] = {}
            for tf, row in tf_rows.items():
                features_by_tf[tf] = row_to_features(row)

            # 主特征 = 第一个可用 timeframe 的特征 (PCM 回退用)
            primary_features = next(iter(features_by_tf.values()))

            # LivePCM.decide() — 多策略仲裁 + 全局 slot 控制
            intents = self.pcm.decide(
                features=primary_features,
                symbol=sym,
                features_by_timeframe=features_by_tf,
            )

            if intents:
                funnel["signals_generated"] += 1
                intent = intents[0]
                # 用获胜 archetype 对应 timeframe 的特征构建入场 bar
                winning_arch = getattr(intent, "archetype", "")
                winning_tf = self._tf_map.get(winning_arch, "")
                entry_feats = features_by_tf.get(winning_tf, primary_features)
                entry_bar = {
                    "close": entry_feats.get("close", 0),
                    "high": entry_feats.get("high", 0),
                    "low": entry_feats.get("low", 0),
                    "open": entry_feats.get("open", 0),
                    "timestamp": ts,
                    "atr": entry_feats.get("atr", 0),
                }
                winning_bm = self._bm_map.get(winning_arch, self._primary_bar_minutes)
                opened = simulator.open_position(
                    intent, entry_bar, entry_feats, bar_minutes=winning_bm
                )
                if opened is None:
                    funnel["reject_max_positions"] += 1
            else:
                # 诊断拒绝原因 (检查每个策略的漏斗)
                all_gate_deny = True
                for s_obj in self._strats.values():
                    lf = getattr(s_obj, "_last_funnel", {})
                    if lf.get("gate", True):
                        all_gate_deny = False
                        break
                if all_gate_deny:
                    funnel["reject_gate_deny"] += 1
                else:
                    funnel["reject_pcm_slot_full"] += 1

            prev_ts[sym] = ts

        # ── Phase 4: 处理最后一个信号后的 1min bars + 关闭残留持仓 ──
        for sym, simulator in self._simulators.items():
            data = sym_data[sym]
            bars_1min_test = data["bars_1min_test"]

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
                # 从任一可用 timeframe 取最后收盘价
                last_close = 0.0
                last_time = datetime.now(timezone.utc)
                tf_features = data["tf_features"]
                for tf in sorted(tf_features.keys(), reverse=True):
                    tdf = tf_features[tf]
                    if not tdf.empty:
                        last_close = float(tdf.iloc[-1].get("close", 0))
                        last_time = tdf.index[-1].to_pydatetime()
                        break
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                simulator.force_close_all(last_close, last_time)

            sym_trades = simulator.closed_trades
            result.trades.extend(sym_trades)
            result.per_symbol[sym] = sym_trades
            result.bars_1min[sym] = data["bars_1min_test"]
            logger.info(f"  {sym}: {len(sym_trades)} trades")

        result.trades.sort(key=lambda t: t.entry_time)
        result.funnel = dict(funnel)
        return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. Trading Map Generator
# ═════════════════════════════════════════════════════════════════════════════


def _resample_to_4h(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """1min bars → 4H OHLCV"""
    ohlc = (
        bars_1min.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    if "volume" in bars_1min.columns:
        ohlc["volume"] = bars_1min["volume"].resample("4h").sum()
    return ohlc


def generate_trading_map_html(
    result: BacktestResult,
    output_path: str,
) -> str:
    """生成 4H K线 + 交易标记 HTML 交易地图。

    每个 symbol 一个独立的 K线图, 上面标记入场/出场点。
    使用 Bokeh 生成可交互 HTML。
    """
    if not BOKEH_AVAILABLE:
        logger.warning("❌ Bokeh 未安装, 无法生成交易地图. pip install bokeh")
        return ""

    symbols = sorted(result.per_symbol.keys())
    if not symbols:
        logger.warning("❌ 没有交易数据, 无法生成交易地图")
        return ""

    figures = []

    # title div
    total_r = sum(t.pnl_r for t in result.trades)
    n_trades = len(result.trades)
    win_rate = (
        sum(1 for t in result.trades if t.pnl_r > 0) / n_trades if n_trades else 0
    )
    title_html = (
        f"<h2>🗺️ 交易地图: {result.strategy.upper()} | "
        f"{n_trades} trades | WR={win_rate:.1%} | Total={total_r:.2f}R</h2>"
    )
    figures.append(Div(text=title_html))

    # 颜色方案: 盈利=绿, 亏损=红
    COLOR_WIN = "#26a69a"  # 绿
    COLOR_LOSS = "#ef5350"  # 红
    COLOR_UP = "#26a69a"
    COLOR_DOWN = "#ef5350"

    for sym in symbols:
        trades = result.per_symbol.get(sym, [])
        bars_1min = result.bars_1min.get(sym)
        if bars_1min is None or bars_1min.empty:
            continue

        # resample to 4H
        df_4h = _resample_to_4h(bars_1min)
        if df_4h.empty:
            continue

        # K线数据
        inc = df_4h.close >= df_4h.open
        dec = ~inc

        # 交易统计
        sym_r = sum(t.pnl_r for t in trades)
        sym_wr = sum(1 for t in trades if t.pnl_r > 0) / len(trades) if trades else 0

        p = bk_figure(
            title=f"{sym}  |  {len(trades)} trades  |  WR={sym_wr:.1%}  |  Total={sym_r:.2f}R",
            x_axis_type="datetime",
            width=1400,
            height=350,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.grid.grid_line_alpha = 0.3

        # K线实体
        w = 4 * 60 * 60 * 1000 * 0.6  # 4h bar width in ms
        p.segment(
            df_4h.index[inc],
            df_4h.high[inc],
            df_4h.index[inc],
            df_4h.low[inc],
            color=COLOR_UP,
            line_width=1,
        )
        p.segment(
            df_4h.index[dec],
            df_4h.high[dec],
            df_4h.index[dec],
            df_4h.low[dec],
            color=COLOR_DOWN,
            line_width=1,
        )
        p.vbar(
            df_4h.index[inc],
            w,
            df_4h.open[inc],
            df_4h.close[inc],
            fill_color=COLOR_UP,
            line_color=COLOR_UP,
            fill_alpha=0.8,
        )
        p.vbar(
            df_4h.index[dec],
            w,
            df_4h.open[dec],
            df_4h.close[dec],
            fill_color=COLOR_DOWN,
            line_color=COLOR_DOWN,
            fill_alpha=0.8,
        )

        # 交易标记
        for t in trades:
            is_win = t.pnl_r > 0
            color = COLOR_WIN if is_win else COLOR_LOSS

            # 入场三角
            entry_marker = (
                "triangle" if t.side in ("LONG", "BUY") else "inverted_triangle"
            )
            p.scatter(
                x=[t.entry_time],
                y=[t.entry_price],
                marker=entry_marker,
                size=12,
                color=color,
                alpha=0.9,
                legend_label="entry",
            )

            # 出场方块
            p.scatter(
                x=[t.exit_time],
                y=[t.exit_price],
                marker="square",
                size=10,
                color=color,
                alpha=0.9,
                legend_label="exit",
            )

            # 连接线
            p.line(
                x=[t.entry_time, t.exit_time],
                y=[t.entry_price, t.exit_price],
                line_color=color,
                line_dash="dashed",
                line_alpha=0.5,
            )

        # HoverTool
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@x{%F %H:%M}"),
                    ("Price", "@y{0.2f}"),
                ],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
        )

        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        p.legend.label_text_font_size = "9pt"
        figures.append(p)

    # 生成 HTML
    layout = bk_column(*figures, sizing_mode="stretch_width")
    html = bk_file_html(
        layout, resources=BK_RESOURCES, title=f"Trading Map: {result.strategy}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"\n  🗺️  Trading map saved → {output_path}")
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="事件驱动回测 — 多策略 PCM 仲裁 + 1min bar 持仓管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        "-s",
        required=True,
        help="策略名, 逗号分隔 (例: bpc / fer / bpc,fer,me)",
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
        help="实盘数据根目录 (仅用于 --data-path 未指定时的 fallback)",
    )
    parser.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="研究数据目录 (默认 data/parquet_data, 设为 none 使用实盘数据)",
    )
    parser.add_argument(
        "--strategies-root",
        default=None,
        help="策略配置目录 (默认 config/strategies)",
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
    parser.add_argument(
        "--db",
        default=None,
        help="订单落库 SQLite 路径 (启用 order_management mock)",
    )
    parser.add_argument(
        "--trading-map",
        default=None,
        help="交易地图 HTML 输出路径 (4H K线 + 交易标记)",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategy.split(",")]
    symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 72)
    print("  🔬 事件驱动回测 (多策略 PCM 仲裁)")
    print("=" * 72)
    print(f"  策略:    {', '.join(strategies)}")
    print(f"  Symbols: {symbols}")
    print(f"  天数:    {args.days}")
    # --data-path none → 显式使用实盘数据做验证
    if args.data_path and args.data_path.lower() == "none":
        args.data_path = None

    if args.data_path:
        print(f"  数据源:  {args.data_path} (研究数据)")
    else:
        print(f"  数据源:  {args.live_root}/data (实盘数据, 验证模式)")
    if args.db:
        print(f"  订单落库: {args.db}")
    if args.trading_map:
        print(f"  交易地图: {args.trading_map}")
    print("=" * 72)

    bt = EventBacktester(
        strategies=strategies,
        live_root=args.live_root,
        strategies_root=args.strategies_root,
        db_path=args.db,
        data_path=args.data_path,
    )

    result = bt.run(symbols=symbols, days=args.days)

    result.print_report()

    if args.export:
        result.export_trades_csv(args.export)

    if args.output:
        _save_json(result, args.output)

    if args.trading_map:
        generate_trading_map_html(result, args.trading_map)

    if args.db:
        print(f"\n  💾 订单数据已保存 → {args.db}")

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
