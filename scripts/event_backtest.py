#!/usr/bin/env python3
"""
事件驱动回测 — 用 1min bar 精确模拟实盘多策略持仓管理

与向量化回测 (backtest_execution_layer) 的区别:
  向量回测: entry_direction + 4h bar 级 trailing → 快速迭代
  事件回测: GenericLiveStrategy.decide() + 1min bar 7步持仓管理 → 实盘验证

支持多策略 PCM 仲裁 (同实盘 run_live.py):
  - 多策略注册 (BPC + FER + ME)
  - 多 timeframe 特征计算（各策略 timeframe 优先读 meta.yaml）
  - LivePCM 优先级仲裁 + Regime 感知缩放
  - 跨 symbol slot 控制

数据流:
  1min bars + ticks → IFC.compute_features_dataframe → 多 timeframe 信号时钟特征
  → LivePCM.decide(features_by_timeframe) → 多策略仲裁 → TradeIntent
  → PositionSimulator: 1min bar 逐 bar 持仓管理 (time/breakeven/trailing/SL/TP)
  → 报告

用法:
    # 多策略联合回测 (推荐 — 与 PCM 向量回测对齐)
    python scripts/event_backtest.py --strategy bpc,fer,me-long --days 180

    # 单策略回测
    python scripts/event_backtest.py --strategy fer --days 180

    # 指定 symbol + 导出
    python scripts/event_backtest.py --strategy bpc,fer,me-long --symbols BTCUSDT,ETHUSDT --days 90 --export trades.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import yaml
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_data_stream.feature_storage import StorageManager
from src.feature_store import FeatureStore, FeatureStoreSpec
from src.feature_store.layer_naming import detect_layer_for_strategy
from src.data_tools.data_handler import DataHandler
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)
from src.time_series_model.live.srb_regime import (
    maybe_inject_srb_experiment_features,
    pick_srb_true_sr_level,
    should_reject_srb_add_by_shape,
    should_reject_srb_wide_entry,
    srb_add_position_allowed,
)
from src.time_series_model.portfolio.live_pcm import LivePCM
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_max_times as _shared_resolve_add_position_max_times,
    resolve_add_position_min_current_r,
    resolve_add_position_size_multiplier as _shared_resolve_add_position_size_multiplier,
    resolve_float_r_ladder_only as _shared_resolve_float_r_ladder_only,
    validate_add_position_trigger as _shared_validate_add_position_trigger,
)
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.core.constitution.runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.features.cross_symbol.macro_tp_vwap_anchor import (
    ANCHOR_COLUMN,
    parse_macro_tp_vwap_anchor_config,
)

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
    from bokeh.models import (
        HoverTool,
        Div,
        Tabs,
        TabPanel,
        FixedTicker,
    )
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


def _timeframe_to_timedelta(tf: str) -> Optional[pd.Timedelta]:
    """Parse timeframe token like 120T/4H/1D to Timedelta."""
    token = str(tf or "").strip().upper()
    if not token:
        return None
    m = re.fullmatch(r"(\d+)\s*([A-Z]+)", token)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    unit_map = {
        "T": "min",
        "MIN": "min",
        "M": "min",
        "H": "h",
        "D": "d",
    }
    if unit not in unit_map:
        return None
    try:
        return pd.to_timedelta(n, unit=unit_map[unit])
    except Exception:
        return None


def _align_feature_index_to_bar_close(
    features_df: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    """Shift feature index from bar-open label to bar-close timestamp."""
    if features_df is None or features_df.empty:
        return features_df
    tf_delta = _timeframe_to_timedelta(timeframe)
    if tf_delta is None or tf_delta <= pd.Timedelta(minutes=1):
        return features_df
    aligned = features_df.copy()
    aligned.index = pd.to_datetime(aligned.index, utc=True) + tf_delta
    return aligned


def _iter_update_bars_1min(
    bars_1min: pd.DataFrame,
    prev_ts: pd.Timestamp,
    cur_ts: pd.Timestamp,
    *,
    fast_mode: bool = False,
):
    """Yield 1min bars in (prev_ts, cur_ts] for position updates.

    `fast_mode` is preserved for CLI compatibility; update path remains
    1min-exact to keep SL/TP timing consistent with non-fast mode.
    """
    if bars_1min is None or bars_1min.empty:
        return
    mask = (bars_1min.index > prev_ts) & (bars_1min.index <= cur_ts)
    for bar_ts, bar_row in bars_1min[mask].iterrows():
        yield bar_ts, bar_row


def _feature_asof_from_sym_tf_features(
    sym_entry: Dict[str, Any],
    bar_ts: Any,
    column: str,
) -> Optional[float]:
    """Pick latest ``column`` from any timeframe row with index <= ``bar_ts``.

    Multi-symbol timeline: structural inputs must follow **the symbol being
    updated**, not only the symbol of the current PCM event. Previously only
    ``simulators[sym_event]._macro_tp_vwap_position`` was refreshed, so ADA
    could keep stale/BTC macro values during 1m ``update()`` → vwap1200 deadband
    rarely matched reality (late / missing structural exits).
    """
    tfd = sym_entry.get("tf_features") or {}
    best_ix: Optional[pd.Timestamp] = None
    best_val: Optional[float] = None
    for _tf, tdf in tfd.items():
        if tdf is None or getattr(tdf, "empty", True):
            continue
        if column not in tdf.columns:
            continue
        try:
            sub = tdf.loc[tdf.index <= bar_ts]
        except Exception:
            continue
        if sub.empty:
            continue
        ts_last = sub.index[-1]
        raw = sub[column].iloc[-1]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        if best_ix is None or ts_last > best_ix:
            best_ix = ts_last
            best_val = v
    return best_val


def _feature_row_asof_from_sym_tf_features(
    sym_entry: Dict[str, Any],
    bar_ts: Any,
    *,
    require_macro: bool = True,
) -> Optional[pd.Series]:
    """Latest feature row with index <= ``bar_ts`` (max timestamp across TFs).

    When ``require_macro`` is True, only consider frames that expose
    ``macro_tp_vwap_1200_position`` and rows with a finite value, so the
    returned row can drive both stored position and frozen VWAP level.
    """
    tfd = sym_entry.get("tf_features") or {}
    best_ix: Optional[pd.Timestamp] = None
    best_row: Optional[pd.Series] = None
    for _tf, tdf in tfd.items():
        if tdf is None or getattr(tdf, "empty", True):
            continue
        if require_macro and "macro_tp_vwap_1200_position" not in tdf.columns:
            continue
        try:
            sub = tdf.loc[tdf.index <= bar_ts]
        except Exception:
            continue
        if sub.empty:
            continue
        ts_last = sub.index[-1]
        row = sub.iloc[-1]
        if require_macro:
            raw = row.get("macro_tp_vwap_1200_position")
            try:
                pv = float(raw)
            except (TypeError, ValueError):
                continue
            if pv != pv:
                continue
        if best_ix is None or ts_last > best_ix:
            best_ix = ts_last
            best_row = row
    return best_row


def _sync_macro_tp_vwap_from_feature_row(
    sim: "PositionSimulator",
    row: Optional[pd.Series],
) -> None:
    """Set simulator macro position + frozen typical-price VWAP from one feature row.

    ``macro_tp_vwap_1200_position`` = (close - vwap) / close on the decision bar.
    Between primary-TF bar closes, VWAP level is held fixed so each 1m close can
    recompute pv = (close_1m - vwap) / close_1m — otherwise crossing the deadband
    between 2H updates would never be seen (stale pv).
    """
    if row is None:
        return
    try:
        mv = row.get("macro_tp_vwap_1200_position")
        if mv is None:
            return
        pv = float(mv)
        if pv != pv:
            return
        sim._macro_tp_vwap_position = pv
        cfeat = row.get("close")
        if cfeat is None:
            sim._macro_tp_vwap_level = None
            return
        c2 = float(cfeat)
        if c2 <= 0:
            sim._macro_tp_vwap_level = None
            return
        sim._macro_tp_vwap_level = c2 * (1.0 - pv)
    except (TypeError, ValueError):
        pass


def _sync_ema_1200_from_feature_row(
    sim: "PositionSimulator",
    row: Optional[pd.Series],
) -> None:
    """Set simulator EMA1200 position + frozen EMA1200 level from one feature row.

    Same mechanism as VWAP1200: freeze the EMA1200 price level at primary-TF
    bar close, recompute position on each 1m bar in between.
    """
    if row is None:
        return
    try:
        mv = row.get("ema_1200_position")
        if mv is None:
            return
        ev = float(mv)
        if ev != ev:
            return
        sim._ema_1200_position = ev
        cfeat = row.get("close")
        if cfeat is None:
            sim._ema_1200_level = None
            return
        c2 = float(cfeat)
        if c2 <= 0:
            sim._ema_1200_level = None
            return
        sim._ema_1200_level = c2 * (1.0 - ev)
    except (TypeError, ValueError):
        pass


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
    archetype: str = ""
    tier_name: str = ""
    evidence_score: float = 0.5  # evidence composite score (0-1)
    bars_held: int = 0
    is_add_position: bool = False  # 加仓标记
    is_reverse: bool = False  # SRB 假突破反手标记
    size_multiplier: float = 1.0  # regime position scale
    atr_stop_pct: float = 0.0
    effective_stop_pct: float = 0.0
    sizing_stop_source: str = ""
    # 平仓时刻是否已触发保本锁（止损价已按 breakeven 规则上移/下移）
    breakeven_locked_at_exit: bool = False


def _resolve_add_position_size_multiplier(
    add_rules: Dict[str, Any],
    add_number: int,
    signal: Optional[Dict[str, Any]] = None,
) -> float:
    return _shared_resolve_add_position_size_multiplier(add_rules, add_number, signal)


def _tail_contribution_rate(trades: List[ClosedTrade]) -> tuple[float, int, int]:
    """返回 top10% winners profit share 及计数。"""
    winners = sorted((float(t.pnl_r) for t in trades if t.pnl_r > 0), reverse=True)
    if not winners:
        return 0.0, 0, 0
    top_n = max(1, int(np.ceil(len(winners) * 0.1)))
    win_sum = float(np.sum(winners))
    top_sum = float(np.sum(winners[:top_n]))
    return (top_sum / win_sum) if win_sum > 1e-9 else 0.0, top_n, len(winners)


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

    def __init__(
        self,
        default_bar_minutes: int = 240,
        max_positions: int = 1,
        fee_rate: float = 0.0,
    ):
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.default_bar_minutes = default_bar_minutes
        self.max_positions = max_positions
        self.fee_rate = fee_rate  # 单边手续费率 (如 0.0004 = 0.04% taker)
        self.closed_trades: List[ClosedTrade] = []
        # structural exit: 最新 EMA200 价格 (由主循环在每根 4H bar 到达时更新)
        self._structural_price: Optional[float] = None
        self._macro_tp_vwap_position: Optional[float] = None
        # 与 macro_tp_vwap_1200_position 同根特征行的滚动典型价 VWAP 价格水平；
        # 在两次 primary-TF 收盘之间冻结，供 1m bar 用 (close_1m - level)/close_1m 检测死区穿越
        self._macro_tp_vwap_level: Optional[float] = None
        # EMA1200 结构性出场: 与 VWAP1200 同理，冻结 EMA1200 水平供 1m 重算
        # 统计依据: EMA1200 零点穿越比 VWAP1200 更可靠 (分歧时 EMA 正确率显著更高)
        self._ema_1200_position: Optional[float] = None
        self._ema_1200_level: Optional[float] = None
        # order_management 集成 (由 EventBacktester 注入)
        self._om_bridge: Optional["OMBridge"] = None
        self.max_observed_leverage: float = 0.0
        self.max_observed_notional_frac: float = 0.0
        # 记录最近一次加仓失败原因（供 funnel 细分统计）
        self.last_add_reject_reason: str = ""
        # SRB 加仓门控（由 EventBacktester 从 execution.yaml 注入）
        self._srb_add_policy: Optional[Dict[str, Any]] = None
        self._primary_bar_count: int = 0
        # 主周期（primary TF）最新收盘 ATR — trailing 可选与入场 ATR 取 max 放宽带宽
        self._primary_tf_atr: Optional[float] = None
        # L3 dynamic trailing 所需：当前 primary bar 的 wide_sr 上下沿价格
        self._wide_sr_upper_px: Optional[float] = None
        self._wide_sr_lower_px: Optional[float] = None
        # Phase D 加仓形态门 recent_momentum 所需：近 N 根 primary close 滚动缓存（净位移用）
        self._primary_close_buffer: List[float] = []
        self._primary_close_buffer_max: int = 16

    @property
    def has_positions(self) -> bool:
        return len(self._positions) > 0

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def slot_position_count(self) -> int:
        """供 PCM 全局 slot 统计：加仓腿不占全局 slot。"""
        return sum(
            1
            for pos in self._positions.values()
            if not bool((pos or {}).get("_is_add_position", False))
        )

    def snapshot_open_positions(self) -> List[Dict[str, Any]]:
        """导出当前未平仓状态 (用于跨月续跑)."""
        rows: List[Dict[str, Any]] = []
        for pid, pos in self._positions.items():
            rows.append({"pid": str(pid), "position": _json_safe(pos)})
        return rows

    def restore_open_positions(self, rows: List[Dict[str, Any]]) -> int:
        """恢复未平仓状态 (由 --resume-state 提供)."""
        loaded = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            raw_pos = row.get("position", {})
            if not isinstance(raw_pos, dict):
                continue
            pos = dict(raw_pos)
            entry_time = pos.get("entry_time")
            if isinstance(entry_time, str):
                try:
                    pos["entry_time"] = pd.Timestamp(entry_time).to_pydatetime()
                except Exception:
                    pos["entry_time"] = datetime.now(timezone.utc)
            elif not isinstance(entry_time, datetime):
                pos["entry_time"] = datetime.now(timezone.utc)
            if pos["entry_time"].tzinfo is None:
                pos["entry_time"] = pos["entry_time"].replace(tzinfo=timezone.utc)
            pid = str(row.get("pid") or str(uuid.uuid4())[:12])
            self._positions[pid] = pos
            loaded += 1
        for pid, pos in self._positions.items():
            if not bool(pos.get("_is_add_position", False)):
                continue
            parent = self._positions.get(str(pos.get("_parent_pid") or ""))
            if not isinstance(parent, dict):
                continue
            se = parent.get("structural_exit")
            if se and not pos.get("structural_exit"):
                pos["structural_exit"] = se
        return loaded

    def open_position(
        self,
        intent: Any,
        entry_bar: Dict[str, Any],
        features: Dict[str, Any],
        bar_minutes: Optional[int] = None,
    ) -> Optional[str]:
        """从 TradeIntent + 当前 bar 创建虚拟持仓 (调用共享 build_position_dict)"""
        _arch = str(getattr(intent, "archetype", "") or "").lower().strip()
        for pos in self._positions.values():
            if (
                pos.get("symbol", "") == getattr(intent, "symbol", "")
                and str(pos.get("archetype", "") or "").lower().strip() == _arch
            ):
                # 同 symbol + 同 archetype 不允许开新 slot；必须走 try_add_position。
                return None
        if len(self._positions) >= self.max_positions:
            return None

        pid = str(uuid.uuid4())[:12]

        entry_price = float(entry_bar.get("close", 0))
        # 直接取 "atr" 键 — 不用 pick_atr() 因为它会误匹配 macd_atr 等特征
        atr = float(entry_bar.get("atr", 0)) or float(features.get("atr", 0)) or 0.0

        # ATR=0 时拒绝开仓 — 无法计算止损/R-multiple
        if atr <= 0:
            return None

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
        pos["archetype"] = getattr(intent, "archetype", "") or ""
        # 存储 size_multiplier (LivePCM regime×evidence 缩放) — 与向量回测 _position_scale 对齐
        pos["_size_multiplier"] = float(getattr(intent, "size_multiplier", 1.0) or 1.0)
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
        parent_close_meta: Dict[str, Dict[str, Any]] = {}

        for pid, pos in self._positions.items():
            if bool(pos.get("_is_add_position", False)) and bool(
                pos.get("_inherit_parent_stop", False)
            ):
                parent_pid = str(pos.get("_parent_pid", "") or "")
                parent = self._positions.get(parent_pid)
                if parent is not None and parent.get("stop_loss_price") is not None:
                    # tighten-only：子仓 SL 只向入场有利方向跟随父仓
                    new_sl = float(parent.get("stop_loss_price"))
                    old_sl = pos.get("stop_loss_price")
                    is_long = str(pos.get("side", "")).upper() in {"LONG", "BUY"}
                    if old_sl is None:
                        pos["stop_loss_price"] = new_sl
                    else:
                        try:
                            old_sl_f = float(old_sl)
                            if is_long and new_sl > old_sl_f:
                                pos["stop_loss_price"] = new_sl
                            elif (not is_long) and new_sl < old_sl_f:
                                pos["stop_loss_price"] = new_sl
                        except (TypeError, ValueError):
                            pos["stop_loss_price"] = new_sl

            # vwap1200: 特征表里的 pv 只在 primary-TF 收盘更新；1m 上用冻结的 VWAP 水平对当前 close 重算 pv，
            # 否则价格在两根 2H 之间穿越死区不会触发结构性出场。
            macro_pv: Optional[float] = self._macro_tp_vwap_position
            _lvl = getattr(self, "_macro_tp_vwap_level", None)
            if _lvl is not None and bar_close > 0:
                try:
                    lv = float(_lvl)
                    bc = float(bar_close)
                    if lv == lv and bc == bc and bc > 0.0:
                        live_pv = (bc - lv) / bc
                        if live_pv == live_pv:
                            macro_pv = max(-1.0, min(1.0, float(live_pv)))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # ema1200: 同理冻结 EMA1200 水平，1m 重算 position
            ema_1200_pv: Optional[float] = self._ema_1200_position
            _ema_lvl = getattr(self, "_ema_1200_level", None)
            if _ema_lvl is not None and bar_close > 0:
                try:
                    elv = float(_ema_lvl)
                    bc = float(bar_close)
                    if elv == elv and bc == bc and bc > 0.0:
                        live_ev = (bc - elv) / bc
                        if live_ev == live_ev:
                            ema_1200_pv = max(-1.0, min(1.0, float(live_ev)))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # 调用共享 7 步持仓管理 (structural: EMA200 / vwap1200 / ema1200)
            close_reason, exit_price = enforce_position(
                pos,
                price_high=bar_high,
                price_low=bar_low,
                price_close=bar_close,
                now=now,
                default_bar_minutes=self.default_bar_minutes,
                structural_price=self._structural_price,
                macro_tp_vwap_position=macro_pv,
                ema_1200_position=ema_1200_pv,
                primary_tf_atr=self._primary_tf_atr,
                wide_sr_upper_px=getattr(self, "_wide_sr_upper_px", None),
                wide_sr_lower_px=getattr(self, "_wide_sr_lower_px", None),
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
                _raw_r = pnl_usd / risk if risk > 0 else 0.0
                # 扣除双边手续费 (开仓+平仓)
                # fee_r = (entry_price + exit_price) × fee_rate / risk
                if self.fee_rate > 0 and risk > 0:
                    fee_r = (entry_price + exit_price) * self.fee_rate / risk
                    _raw_r -= fee_r
                # 应用 position scale (regime × evidence) — 与向量回测 exec_returns *= _position_scale 对齐
                pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)

                # 归一化 exit_reason: 与向量回测对齐命名
                normalized_reason = close_reason
                if close_reason == "stop_loss":
                    normalized_reason = (
                        "trailing_sl" if pos.get("trailing_activated") else "sl"
                    )
                elif close_reason == "take_profit":
                    normalized_reason = "tp"
                elif close_reason == "time_stop":
                    normalized_reason = "timeout"

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
                    exit_reason=normalized_reason,
                    archetype=pos.get("archetype", ""),
                    tier_name=pos.get("tier_name", ""),
                    evidence_score=pos.get("evidence_score", 0),
                    bars_held=pos.get("bars_counted", 0),
                    is_add_position=pos.get("_is_add_position", False),
                    is_reverse=False,
                    size_multiplier=pos.get("_size_multiplier", 1.0),
                    atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                    effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                    sizing_stop_source=pos.get("sizing_stop_source", ""),
                    breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
                )
                closed.append(trade)
                self.closed_trades.append(trade)
                if str(pos.get("archetype", "") or "").lower() == "fer":
                    try:
                        from src.time_series_model.live.fer_diagnostics import (
                            record_fer_exit,
                        )

                        record_fer_exit(
                            pos=dict(pos),
                            close_reason_raw=str(close_reason),
                            exit_reason_normalized=str(normalized_reason),
                            exit_price=float(exit_price),
                            now=now,
                            pnl_r=float(pnl_r),
                        )
                    except Exception:
                        pass
                to_remove.append(pid)
                if not bool(pos.get("_is_add_position", False)):
                    parent_close_meta[str(pid)] = {
                        "exit_price": float(exit_price),
                        "normalized_reason": str(normalized_reason),
                        "breakeven_locked": bool(pos.get("breakeven_locked", False)),
                    }

                # 写入 order_management DB
                if self._om_bridge:
                    self._om_bridge.record_close(
                        pid=pid,
                        exit_price=exit_price,
                        exit_time=now,
                        exit_reason=close_reason,
                        pnl_r=pnl_r,
                    )

        # 母仓退出时，默认强制同 bar 带走其加仓子仓（可通过 _share_parent_exit=False 关闭）
        if parent_close_meta:
            for pid, pos in self._positions.items():
                if pid in to_remove:
                    continue
                if not bool(pos.get("_is_add_position", False)):
                    continue
                if not bool(pos.get("_share_parent_exit", True)):
                    continue
                parent_pid = str(pos.get("_parent_pid", "") or "")
                meta = parent_close_meta.get(parent_pid)
                if not meta:
                    continue
                entry_price = float(pos.get("entry_price", 0.0) or 0.0)
                risk = (
                    float(pos.get("initial_risk_distance", 0.0) or 0.0)
                    or float(pos.get("atr_at_entry", 0.0) or 0.0)
                    or 0.0
                )
                is_long = pos.get("side") in {"LONG", "BUY"}
                forced_exit = float(meta.get("exit_price", bar_close) or bar_close)
                pnl_usd = (
                    (forced_exit - entry_price)
                    if is_long
                    else (entry_price - forced_exit)
                )
                _raw_r = pnl_usd / risk if risk > 0 else 0.0
                if self.fee_rate > 0 and risk > 0:
                    fee_r = (entry_price + forced_exit) * self.fee_rate / risk
                    _raw_r -= fee_r
                pnl_r = _raw_r * float(pos.get("_size_multiplier", 1.0) or 1.0)
                trade = ClosedTrade(
                    symbol=pos.get("symbol", ""),
                    side=pos["side"],
                    entry_price=entry_price,
                    exit_price=forced_exit,
                    entry_time=pos["entry_time"],
                    exit_time=now,
                    atr_at_entry=pos.get("atr_at_entry", 0),
                    pnl_r=pnl_r,
                    pnl_usd=pnl_usd,
                    exit_reason=str(meta.get("normalized_reason", "sl")),
                    archetype=pos.get("archetype", ""),
                    tier_name=pos.get("tier_name", ""),
                    evidence_score=pos.get("evidence_score", 0),
                    bars_held=pos.get("bars_counted", 0),
                    is_add_position=True,
                    size_multiplier=pos.get("_size_multiplier", 1.0),
                    atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                    effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                    sizing_stop_source=pos.get("sizing_stop_source", ""),
                    breakeven_locked_at_exit=bool(meta.get("breakeven_locked", False)),
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
            risk = pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
            pnl_usd = (price - entry_price) if is_long else (entry_price - price)
            _raw_r = pnl_usd / risk if risk > 0 else 0.0
            if self.fee_rate > 0 and risk > 0:
                fee_r = (entry_price + price) * self.fee_rate / risk
                _raw_r -= fee_r
            pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)
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
                archetype=pos.get("archetype", ""),
                tier_name=pos.get("tier_name", ""),
                evidence_score=pos.get("evidence_score", 0),
                bars_held=pos.get("bars_counted", 0),
                is_add_position=pos.get("_is_add_position", False),
                is_reverse=False,
                size_multiplier=pos.get("_size_multiplier", 1.0),
                atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                sizing_stop_source=pos.get("sizing_stop_source", ""),
                breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
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

    def close_by_archetype(
        self, archetype: str, close_price: float, close_time: datetime
    ) -> List[ClosedTrade]:
        """关闭指定 archetype 的所有仓位 (遗留接口, 竞争驱逐已移除)"""
        closed = []
        to_remove = []
        for pid, pos in self._positions.items():
            if pos.get("archetype", "").lower() != archetype.lower():
                continue
            is_long = pos["side"] in {"LONG", "BUY"}
            entry_price = pos["entry_price"]
            risk = pos.get("initial_risk_distance") or pos.get("atr_at_entry", 0) or 0
            pnl_usd = (
                (close_price - entry_price) if is_long else (entry_price - close_price)
            )
            _raw_r = pnl_usd / risk if risk > 0 else 0.0
            if self.fee_rate > 0 and risk > 0:
                fee_r = (entry_price + close_price) * self.fee_rate / risk
                _raw_r -= fee_r
            pnl_r = _raw_r * pos.get("_size_multiplier", 1.0)
            trade = ClosedTrade(
                symbol=pos.get("symbol", ""),
                side=pos["side"],
                entry_price=entry_price,
                exit_price=close_price,
                entry_time=pos["entry_time"],
                exit_time=close_time,
                atr_at_entry=pos.get("atr_at_entry", 0),
                pnl_r=pnl_r,
                pnl_usd=pnl_usd,
                exit_reason="evicted",
                archetype=pos.get("archetype", ""),
                tier_name=pos.get("tier_name", ""),
                evidence_score=pos.get("evidence_score", 0),
                bars_held=pos.get("bars_counted", 0),
                is_add_position=pos.get("_is_add_position", False),
                is_reverse=False,
                size_multiplier=pos.get("_size_multiplier", 1.0),
                atr_stop_pct=pos.get("atr_stop_pct", 0.0),
                effective_stop_pct=pos.get("effective_stop_pct", 0.0),
                sizing_stop_source=pos.get("sizing_stop_source", ""),
                breakeven_locked_at_exit=bool(pos.get("breakeven_locked", False)),
            )
            closed.append(trade)
            self.closed_trades.append(trade)
            to_remove.append(pid)
        for pid in to_remove:
            self._positions.pop(pid, None)
        return closed

    def try_add_position(
        self,
        intent: Any,
        entry_bar: Dict[str, Any],
        features: Dict[str, Any],
        executor: ConstitutionExecutor,
        runtime_state: ConstitutionRuntimeState,
        bar_minutes: Optional[int] = None,
        *,
        skip_signal_trigger: bool = False,
    ) -> Optional[str]:
        """加仓模拟: 复用实盘 validate_add_position / record_add_position。

        与实盘 constitution_executor.py 100% 同一份代码:
          1. executor.validate_add_position() — 策略/次数/利润锁定检查
          2. executor.record_add_position() — 更新 ConstitutionRuntimeState

        skip_signal_trigger:
            True 时只检查 min_current_r_by_add（及 ATR 换算），不跑 BPC/ME 等特征 trigger。
            供 execution.add_position.trigger.type=float_r_ladder_only 浮盈阶梯加仓（事件回测）。

        Returns:
            position_id if added, None if rejected
        """
        self.last_add_reject_reason = ""
        archetype = getattr(intent, "archetype", "").lower().strip()
        new_side = (
            "LONG"
            if str(getattr(intent, "action", "")).upper() in ("LONG", "BUY")
            else "SHORT"
        )

        # 1. 查找同 symbol 同 side 同 archetype 的已有持仓
        parent_pid = None
        parent_pos = None
        for pid, pos in self._positions.items():
            _pos_arch = str(pos.get("archetype", "") or "").lower().strip()
            _pos_tier = str(pos.get("tier_name", "") or "").lower().strip()
            if (
                pos.get("symbol", "") == intent.symbol
                and pos["side"] == new_side
                and (_pos_arch == archetype or _pos_tier == archetype)
            ):
                parent_pid = pid
                parent_pos = pos
                break

        if parent_pos is None:
            self.last_add_reject_reason = "no_parent_position"
            return None  # 没有已有持仓，不是加仓场景

        _pol = getattr(self, "_srb_add_policy", None)
        if archetype == "srb" and _pol:
            _ok, _why = srb_add_position_allowed(features or {}, _pol)
            if not _ok:
                self.last_add_reject_reason = _why
                return None

        # 2. 计算 current_r (用于 validate_add_position)
        entry_price = parent_pos["entry_price"]
        risk = (
            parent_pos.get("initial_risk_distance")
            or parent_pos.get("atr_at_entry", 0)
            or 1
        )
        is_long = parent_pos["side"] in {"LONG", "BUY"}
        current_price = float(entry_bar.get("close", 0))
        current_r = (
            (
                (current_price - entry_price)
                if is_long
                else (entry_price - current_price)
            )
            / risk
            if risk > 0
            else 0.0
        )

        # 2a-Phase-D: 加仓事后形态门（srb_add_position_policy.post_hoc_shape_gate）。
        # 仅 SRB：若 post_hoc_shape_gate 下任一子项 enabled，追加形态确认。
        if archetype == "srb" and _pol:
            _gate_cfg = _pol.get("post_hoc_shape_gate") or {}
            if any(
                bool((_gate_cfg.get(_k) or {}).get("enabled", False))
                for _k in (
                    "retrace_guard",
                    "recent_momentum",
                    "trend_r2_gate",
                    "wide_sr_expansion",
                    "trend_health_gate",
                )
            ):
                # 计算母仓 MFE R（用 initial_risk_distance 归一化，与退出逻辑对齐）
                _hwm = parent_pos.get("high_water_mark")
                _lwm = parent_pos.get("low_water_mark")
                if is_long and _hwm is not None:
                    _mfe_r = (float(_hwm) - entry_price) / risk if risk > 0 else 0.0
                elif (not is_long) and _lwm is not None:
                    _mfe_r = (entry_price - float(_lwm)) / risk if risk > 0 else 0.0
                else:
                    _mfe_r = max(0.0, current_r)
                # recent_net_move_atr：近 N primary close 净变化 / ATR（与 mother 方向同向为正）
                _shape_feat = dict(features or {})
                _shape_feat["mfe_r"] = _mfe_r
                _shape_feat["current_r"] = current_r
                if "recent_net_move_atr" not in _shape_feat:
                    _rn = (_gate_cfg.get("recent_momentum") or {}).get(
                        "lookback_bars", 6
                    ) or 6
                    try:
                        _rn = int(_rn)
                    except (TypeError, ValueError):
                        _rn = 6
                    _buf = getattr(self, "_primary_close_buffer", None) or []
                    _atr_now = float(
                        features.get("atr") or parent_pos.get("atr_at_entry") or 0.0
                    )
                    if len(_buf) >= 2 and _atr_now > 0:
                        _tail = _buf[-max(2, min(_rn + 1, len(_buf))) :]
                        _net = _tail[-1] - _tail[0]
                        # 保留带符号：正 = 上涨，负 = 下跌；gate 内部按 side 判方向。
                        _shape_feat["recent_net_move_atr"] = _net / _atr_now
                # bars_since_mother_entry（E4）：用 entry_bar 的 timestamp 与 parent.entry_time
                # 的差，按 bar_minutes 转成 primary bar 数。缺失时兜底 0。
                try:
                    _parent_et = parent_pos.get("entry_time")
                    _now_ts = entry_bar.name if hasattr(entry_bar, "name") else None
                    _bm = int(
                        parent_pos.get("bar_minutes")
                        or self._primary_bar_minutes
                        or 240
                    )
                    if _parent_et is not None and _now_ts is not None and _bm > 0:
                        _pt = pd.Timestamp(_parent_et)
                        _nt = pd.Timestamp(_now_ts)
                        if _pt.tzinfo is None:
                            _pt = _pt.tz_localize("UTC")
                        if _nt.tzinfo is None:
                            _nt = _nt.tz_localize("UTC")
                        _delta_min = (_nt - _pt).total_seconds() / 60.0
                        _shape_feat["bars_since_mother_entry"] = max(
                            0.0, _delta_min / float(_bm)
                        )
                except Exception:
                    pass
                # wide_sr_dist_atr：从 features 直接读（已存在）
                _mother_ctx = {
                    "side": parent_pos.get("side"),
                    "entry_wide_sr_dist_atr": parent_pos.get(
                        "_srb_entry_wide_sr_dist_atr"
                    ),
                }
                _rej, _why = should_reject_srb_add_by_shape(
                    _shape_feat, _mother_ctx, _gate_cfg
                )
                if _rej:
                    self.last_add_reject_reason = _why
                    return None

        # 2b. 找出同 symbol + 同 direction 的活跃仓位
        same_sym_dir = [
            p
            for p in self._positions.values()
            if p.get("symbol", "") == intent.symbol and p["side"] == new_side
        ]

        rec = runtime_state.add_position.positions.get(parent_pid)
        next_add_no = int(rec.add_count) + 1 if rec is not None else 1

        # 3. 复用实盘 validate_add_position (raises ConstitutionViolation on failure)
        try:
            executor.validate_add_position(
                st=runtime_state,
                position_id=parent_pid,
                archetype=archetype,
                current_r=current_r,
                locked_profit=parent_pos.get("breakeven_locked", False),
                position_action=new_side,
            )
        except ConstitutionViolation:
            self.last_add_reject_reason = "constitution_reject"
            return None

        add_rules = dict(
            executor.resolve_add_position_for_strategy(
                archetype, position_action=new_side
            )
        )
        _intent_add = (getattr(intent, "execution_profile", {}) or {}).get(
            "add_position"
        ) or {}
        if _intent_add:
            _trig = dict(add_rules.get("trigger", {}) or {})
            _trig.update(dict(_intent_add.get("trigger", {}) or {}))
            add_rules.update(
                {k: v for k, v in dict(_intent_add).items() if k != "trigger"}
            )
            if _trig:
                add_rules["trigger"] = _trig
        if next_add_no > _shared_resolve_add_position_max_times(add_rules):
            self.last_add_reject_reason = "max_add_times"
            return None
        signal = dict(features or {})
        signal["add_position_seq"] = next_add_no
        signal.setdefault("close", current_price)
        _atr_parent = float(parent_pos.get("atr_at_entry", 0) or 0)
        _risk_parent = float(parent_pos.get("initial_risk_distance", 0) or 0)
        if _atr_parent > 0 and _risk_parent > 0:
            signal["parent_initial_r"] = _risk_parent / _atr_parent
        _risk_frac = float(
            executor.resolve_risk_for_strategy(archetype, position_action=new_side)
        )
        _current_lev = 0.0
        _current_notional_frac = 0.0
        for _pos in same_sym_dir:
            _stop_dist = float(_pos.get("initial_risk_distance", 0.0) or 0.0)
            _ep = float(_pos.get("entry_price", 0.0) or 0.0)
            _stop_pct = _stop_dist / _ep if _ep > 0 else 0.0
            if _stop_pct <= 1e-9:
                continue
            _mult = float(_pos.get("_size_multiplier", 1.0) or 1.0)
            _current_lev += _risk_frac * _mult / _stop_pct
            _current_notional_frac += _risk_frac * _mult
        _parent_ep = float(parent_pos.get("entry_price", 0.0) or 0.0)
        _parent_stop_pct = (_risk_parent / _parent_ep) if _parent_ep > 0 else 0.0
        signal["base_leverage_unit"] = (
            _risk_frac / _parent_stop_pct if _parent_stop_pct > 1e-9 else 1.0
        )
        signal["current_leverage"] = float(_current_lev)
        signal["base_notional_frac"] = float(_risk_frac)
        signal["current_notional_frac"] = float(_current_notional_frac)
        signal["equity_usd"] = float(features.get("equity", 0.0) or 0.0)
        if skip_signal_trigger:
            _thr = resolve_add_position_min_current_r(add_rules, next_add_no, signal)
            if current_r < _thr:
                self.last_add_reject_reason = "trigger_not_met"
                return None
        elif not _shared_validate_add_position_trigger(
            archetype=archetype,
            direction=1 if new_side == "LONG" else -1,
            signal=signal,
            add_position_cfg=add_rules,
            current_r=current_r,
        ):
            _thr = resolve_add_position_min_current_r(add_rules, next_add_no, signal)
            if current_r < _thr:
                self.last_add_reject_reason = "add_min_current_r"
            else:
                _trg = dict(add_rules.get("trigger") or {})
                _tt = str(_trg.get("type", "")).strip().lower()
                if _tt in {"bpc_follow_signal", "follow_signal"}:
                    self.last_add_reject_reason = "add_bpc_breakout_mismatch"
                else:
                    self.last_add_reject_reason = "add_trigger_feature_rules"
            return None
        add_mult = _resolve_add_position_size_multiplier(add_rules, next_add_no, signal)
        parent_mult = float(parent_pos.get("_size_multiplier", 1.0) or 1.0)
        projected_lev = float(_current_lev) + float(
            signal.get("base_leverage_unit", 1.0)
        ) * float(add_mult)
        projected_notional = float(_current_notional_frac) + float(_risk_frac) * float(
            parent_mult
        ) * float(add_mult)
        self.max_observed_leverage = max(self.max_observed_leverage, projected_lev)
        self.max_observed_notional_frac = max(
            self.max_observed_notional_frac, projected_notional
        )

        # 4. 记录加仓 (更新 ConstitutionRuntimeState — 同实盘 record_add_position)
        executor.record_add_position(
            st=runtime_state,
            position_id=parent_pid,
            current_r=current_r,
            locked_profit=parent_pos.get("breakeven_locked", False),
        )

        # 5. 开加仓仓位
        pid = str(uuid.uuid4())[:12]
        pos = build_position_dict(
            intent=intent,
            entry_price=float(entry_bar.get("close", 0)),
            atr=float(entry_bar.get("atr", 0)) or float(features.get("atr", 0)) or 0.0,
            bar_minutes=bar_minutes or self.default_bar_minutes,
            entry_time=(
                pd.Timestamp(entry_bar.get("timestamp")).to_pydatetime()
                if entry_bar.get("timestamp") is not None
                else datetime.now(timezone.utc)
            ),
        )
        # float_r_ladder_only 等 intent 常仅有 add_position、无 rr_constraints → 无 structural_exit。
        # 否则 vwap1200 只在首仓 enforce，加仓腿仅宽止损/共享止损离场，图上会像「结构止损失效」。
        _p_se = parent_pos.get("structural_exit")
        if _p_se and not pos.get("structural_exit"):
            pos["structural_exit"] = str(_p_se)
        pos["_is_add_position"] = True
        pos["_parent_pid"] = parent_pid
        pos["_add_position_seq"] = next_add_no
        pos["_share_parent_exit"] = bool(add_rules.get("share_parent_exit", True))
        pos["_inherit_parent_stop"] = bool(add_rules.get("inherit_parent_stop", False))
        if bool(pos["_inherit_parent_stop"]):
            parent_sl = parent_pos.get("stop_loss_price")
            if parent_sl is not None:
                pos["stop_loss_price"] = float(parent_sl)
            pos["breakeven_enabled"] = False
            pos["activation_r"] = None
            pos["trailing_activated"] = False
        # 加仓继承父仓的 regime scale，但风险预算按 add_size_multipliers 缩小
        pos["_size_multiplier"] = parent_mult * add_mult
        self._positions[pid] = pos
        self.last_add_reject_reason = ""
        return pid


def _count_open_add_legs_for_parent(sim: PositionSimulator, parent_pid: str) -> int:
    n = 0
    pp = str(parent_pid)
    for pos in sim._positions.values():
        if (
            bool(pos.get("_is_add_position", False))
            and str(pos.get("_parent_pid") or "") == pp
        ):
            n += 1
    return n


def _rehydrate_add_position_runtime_from_simulator(
    sim: PositionSimulator, st: ConstitutionRuntimeState
) -> None:
    for pid, pos in sim._positions.items():
        if bool(pos.get("_is_add_position", False)):
            continue
        n = _count_open_add_legs_for_parent(sim, str(pid))
        if n <= 0:
            continue
        st.add_position.positions[str(pid)] = AddPositionRecord(
            position_id=str(pid), add_count=n
        )


def _load_add_position_runtime_from_resume(
    resume_blob: Dict[str, Any], st: ConstitutionRuntimeState
) -> int:
    raw = (resume_blob or {}).get("add_position_state") or {}
    positions = raw.get("positions") if isinstance(raw, dict) else None
    if not isinstance(positions, dict):
        return 0
    n = 0
    for pid, row in positions.items():
        if not isinstance(row, dict):
            continue
        st.add_position.positions[str(pid)] = AddPositionRecord(
            position_id=str(row.get("position_id") or pid),
            add_count=int(row.get("add_count", 0) or 0),
            locked_profit=bool(row.get("locked_profit", False)),
            current_r=(
                float(row["current_r"])
                if row.get("current_r") is not None
                and str(row.get("current_r", "")).strip() != ""
                else None
            ),
            updated_at=(
                str(row["updated_at"])
                if isinstance(row.get("updated_at"), str)
                else None
            ),
        )
        n += 1
    return n


def _collect_open_parent_pids(simulators: Mapping[str, PositionSimulator]) -> Set[str]:
    out: Set[str] = set()
    for sim in (simulators or {}).values():
        if sim is None:
            continue
        for pid, pos in sim._positions.items():
            if not isinstance(pos, dict):
                continue
            if not bool(pos.get("_is_add_position", False)):
                out.add(str(pid))
    return out


def _prune_stale_add_position_records(
    st: ConstitutionRuntimeState, open_parent_pids: Set[str]
) -> None:
    ap = st.add_position.positions
    for k in list(ap.keys()):
        if k not in open_parent_pids:
            del ap[k]


def _merge_add_position_runtime_with_open_legs(
    sim: PositionSimulator, st: ConstitutionRuntimeState
) -> None:
    for pid, pos in sim._positions.items():
        if bool(pos.get("_is_add_position", False)):
            continue
        spid = str(pid)
        open_legs = _count_open_add_legs_for_parent(sim, spid)
        rec = st.add_position.positions.get(spid)
        if rec is None:
            if open_legs > 0:
                st.add_position.positions[spid] = AddPositionRecord(
                    position_id=spid, add_count=open_legs
                )
            continue
        rec.add_count = max(int(rec.add_count), open_legs)


def _filter_add_position_dict_for_open_parents(
    full: Dict[str, Any], rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    open_parents: Set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pos = row.get("position") or {}
        if not isinstance(pos, dict) or bool(pos.get("_is_add_position", False)):
            continue
        p = row.get("pid")
        if p is not None:
            open_parents.add(str(p))
    positions = (full or {}).get("positions") or {}
    if not isinstance(positions, dict):
        return {"positions": {}}
    out_pos = {
        k: v for k, v in positions.items() if k in open_parents and isinstance(v, dict)
    }
    return {"positions": out_pos}


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
    # Kill switch 模拟统计
    kill_switch_stats: Optional[Dict[str, Any]] = None
    # 风险 equity curve
    equity_curve: Optional[List[float]] = None
    # 加仓模拟统计
    add_position_stats: Optional[Dict[str, Any]] = None
    # 月末未平仓快照 (用于跨月续跑)
    open_positions_end: List[Dict[str, Any]] = field(default_factory=list)
    # 各注册策略 execution.add_position.trigger.type（小写 key）
    add_trigger_types: Dict[str, str] = field(default_factory=dict)
    # 时间线上每次 PCM 评估后的策略漏斗快照（用于交易地图附图）
    funnel_per_bar: List[Dict[str, Any]] = field(default_factory=list)

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
        tail_rate, tail_n, winner_n = _tail_contribution_rate(self.trades)
        if winner_n > 0:
            print(f"  Tail contrib: {tail_rate:.1%}  (top {tail_n}/{winner_n} winners)")

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

        # Kill switch 模拟统计
        if self.kill_switch_stats:
            ks = self.kill_switch_stats
            print(f"\n  🚨 Kill Switch 模拟:")
            print(f"    触发次数: {ks.get('trigger_count', 0)}")
            print(f"    跳过入场: {ks.get('trades_skipped', 0)}")
            print(f"    实际执行: {ks.get('trades_executed', 0)}")
            for trig in ks.get("triggers", [])[:5]:
                print(
                    f"    │ {trig['timestamp']}: {', '.join(trig['reasons'])} (eq=${trig['equity']:.0f})"
                )

        # 风险 Equity Curve 摘要
        if self.equity_curve and len(self.equity_curve) > 1:
            final_eq = self.equity_curve[-1]
            peak_eq = max(self.equity_curve)
            ret_pct = (final_eq - self.equity_curve[0]) / self.equity_curve[0] * 100
            max_dd_eq = 0.0
            peak = self.equity_curve[0]
            for eq in self.equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd_eq:
                    max_dd_eq = dd
            print(f"\n  💰 Risk-Based Equity ($1000):")
            print(f"    Final: ${final_eq:.0f} ({ret_pct:+.1f}%)")
            print(f"    Peak: ${peak_eq:.0f}")
            print(f"    Max DD: {max_dd_eq:.1%}")

        # 加仓统计
        if self.add_position_stats:
            ap = self.add_position_stats
            print("\n  📈 加仓模拟 (per_strategy_limits + execution.add_position):")
            print(f"    加仓成功: {ap.get('add_count', 0)} 次")
            print(f"    加仓拒绝: {ap.get('rejected_count', 0)} 次")
            print(f"    加仓交易: {ap.get('add_trades', 0)} 笔")
            if ap.get("add_trades", 0) > 0:
                print(f"    加仓 Mean R: {ap.get('add_mean_r', 0):.4f}")
                print(f"    加仓 Win%: {ap.get('add_win_rate', 0):.1%}")
                print(f"    加仓平均倍率: {ap.get('add_mean_size', 0):.2f}x")
                print(f"    观测最大杠杆: {ap.get('max_observed_leverage', 0):.2f}x")
            if isinstance(ap.get("path_efficiency_pct_at_add"), dict):
                print(
                    "\n  📐 path_efficiency_pct 分位分布 → 见脚本末尾（避免被交易地图日志顶掉）"
                )

        if self.open_positions_end:
            print(f"\n  ♻️  月末未平仓: {len(self.open_positions_end)}")

        self._print_add_position_diagnostics_footer()

        print("=" * 72)

    def _print_add_position_diagnostics_footer(self) -> None:
        """置底简短加仓诊断（pipeline run_step 只打 stdout 末段时仍能看见）。"""
        fn = self.funnel or {}
        print("\n  🔎 加仓诊断摘要 (execution.add_position / PCM slot)")
        if self.add_trigger_types:
            for sk, tv in sorted(self.add_trigger_types.items()):
                print(f"    trigger.type [{sk}]: {tv}")
        keys = [
            ("total_signals_checked", "时间线检查次数"),
            ("signals_generated", "产生意图次数"),
            ("reject_pcm_slot_full", "PCM全局/策略槽满(drop_slot)"),
            ("reject_pcm_direction_policy", "PCM宪法方向过滤(按候选intent计次)"),
            ("reject_pcm_family_conflict", "PCM同symbol家族反向冲突"),
            ("reject_pcm_daily_throttle", "PCM家族日内入场上限"),
            ("reject_pcm_struct_pass_no_intent", "结构全过但无候选intent(极少)"),
            ("reject_open_atr_nonpositive", "开仓时ATR≤0拒单"),
            ("reject_open_duplicate_archetype", "同symbol同archetype已持仓拒新开"),
            ("add_position_ok", "加仓成功次数"),
            ("add_position_rejected", "加仓拒绝次数"),
            ("float_ladder_add_ok", "浮盈阶梯成功"),
            ("reject_add_trigger", "加仓 trigger 类拒绝(合计)"),
            ("reject_add_detail_min_r", "  └ 未达 min_current_r"),
            ("reject_add_detail_bpc_breakout", "  └ bpc_breakout 与仓向不一致"),
            ("reject_add_detail_me_features", "  └ ME/其它特征 trigger"),
            ("reject_add_no_parent", "无父仓(不应常见)"),
            ("reject_add_max_times", "超 max_add_times"),
            ("reject_add_constitution", "constitution 拒"),
        ]
        for k, label in keys:
            if k in fn and int(fn[k]) > 0:
                print(f"    {label}: {fn[k]}")
        ap = self.add_position_stats
        if isinstance(ap, dict) and ap.get("enabled"):
            tries = int(ap.get("add_count", 0) or 0) + int(
                ap.get("rejected_count", 0) or 0
            )
            print(
                f"    提示: 信号加仓尝试≈{tries}；若很少，先看槽满/是否常进「无持仓加仓」分支。"
            )
            db = int(fn.get("reject_add_detail_bpc_breakout", 0) or 0)
            tr = int(fn.get("reject_add_trigger", 0) or 0)
            if tr > 0 and db >= max(1, tr // 2):
                print(
                    "    提示: bpc_breakout 拒绝占比高 → 浮盈阶梯不校验该特征，"
                    "对比 float_r_ladder_only 可区分「无信号」vs「信号方向过滤」。"
                )

    def print_path_efficiency_footer(self) -> None:
        """path_efficiency_pct 分布：放在 main() 最后打印，便于 pipeline 截尾仍可见 + 与 sidecar JSON 对齐。"""
        ap = self.add_position_stats
        if not isinstance(ap, dict):
            return
        pe = ap.get("path_efficiency_pct_at_add")
        if not isinstance(pe, dict):
            return
        print()
        print("=" * 72)
        print(
            "  📐 path_efficiency_pct @ 加仓尝试 "
            "(类 ER 历史分位 [0,1]，path_efficiency_pct_f → path_efficiency_pct)"
        )
        print("=" * 72)
        for line in _format_er_pct_summary_lines(
            pe.get("signal_add_attempts") or {},
            "signal_add（PCM 再意图 / bpc_follow_signal）",
        ):
            print(line)
        for line in _format_er_pct_summary_lines(
            pe.get("float_ladder_attempts") or {},
            "float_r_ladder_only（阶梯路径）",
        ):
            print(line)
        print("=" * 72)

    def export_trades_csv(self, path: str):
        """导出交易明细 CSV"""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
                    "archetype": t.archetype,
                    "tier": t.tier_name,
                    "evidence": round(t.evidence_score, 4),
                    "bars_held": t.bars_held,
                    "is_add_position": t.is_add_position,
                    "is_reverse": t.is_reverse,
                    "size_multiplier": round(t.size_multiplier, 4),
                    "atr_stop_pct": round(t.atr_stop_pct, 6),
                    "effective_stop_pct": round(t.effective_stop_pct, 6),
                    "sizing_stop_source": t.sizing_stop_source,
                    "breakeven_locked_at_exit": t.breakeven_locked_at_exit,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"\n  📤 Trades exported: {len(df)} rows → {out_path}")


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


def _apply_pcm_direction_ffill(
    symbol: str,
    timeframe: str,
    features: Dict[str, float],
    cache: Dict[Tuple[str, str], Dict[str, float]],
    *,
    keys: Tuple[str, ...] = ("ema_1200_position", "roc_20"),
) -> None:
    """因果填充：本 bar 缺列/NaN 时用该 (symbol, tf) 上一有效值，避免 Direction 因键缺失恒为 0。

    ``row_to_features`` 会丢弃 NaN；慢窗特征在部分 bar 上为空时，decide() 收不到
    ``ema_1200_position`` / ``roc_20``，signal_match 与 dual 均失败 —— 与 prefilter 是否通过无关。
    """
    ck = (str(symbol), str(timeframe))
    slot = cache.setdefault(ck, {})
    for k in keys:
        v = features.get(k)
        if v is not None and v == v and np.isfinite(v):
            slot[k] = float(v)
        elif k in slot:
            features[k] = slot[k]


def _extract_path_efficiency_pct(features: Mapping[str, Any]) -> Optional[float]:
    """path_efficiency 的滚动历史分位 [0,1]（path_efficiency_pct_f / 列 path_efficiency_pct），语义类似 ER。"""
    for k in ("path_efficiency_pct", "path_efficiency_pct_f"):
        v = features.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except (TypeError, ValueError):
            continue
    return None


def _er_pct_numeric_summary(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=float)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _er_pct_attempt_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: {pct: Optional[float], outcome: str}"""
    attempts = len(rows)
    with_vals = [float(r["pct"]) for r in rows if r.get("pct") is not None]
    missing = attempts - len(with_vals)
    out: Dict[str, Any] = {
        "attempts": attempts,
        "missing_path_efficiency_pct": missing,
        "with_feature": len(with_vals),
        "overall": _er_pct_numeric_summary(with_vals),
    }
    by_out: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        p = r.get("pct")
        if p is None:
            continue
        by_out[str(r.get("outcome", "unknown"))].append(float(p))
    out["by_outcome"] = {
        k: _er_pct_numeric_summary(v) for k, v in sorted(by_out.items())
    }
    return out


def _format_er_pct_summary_lines(
    stats: Dict[str, Any],
    title: str,
) -> List[str]:
    lines: List[str] = [f"    {title}:"]
    if int(stats.get("attempts", 0) or 0) == 0:
        lines.append("      (本路径无加仓尝试)")
        return lines
    lines.append(
        f"      尝试={stats['attempts']}, "
        f"有特征={stats['with_feature']}, "
        f"缺失 path_efficiency_pct={stats['missing_path_efficiency_pct']}"
    )
    ov = stats.get("overall") or {}
    if not ov:
        lines.append(
            "      无有效数值 — 请在策略 features 中包含 path_efficiency_pct_f（→ path_efficiency_pct）"
        )
        return lines
    lines.append(
        f"      分位[0,1]: n={int(ov['n'])} mean={ov['mean']:.3f} std={ov['std']:.3f} "
        f"min={ov['min']:.3f} p10={ov['p10']:.3f} p25={ov['p25']:.3f} "
        f"p50={ov['p50']:.3f} p75={ov['p75']:.3f} p90={ov['p90']:.3f} max={ov['max']:.3f}"
    )
    for ok, sub in (stats.get("by_outcome") or {}).items():
        if not sub or int(sub.get("n", 0) or 0) == 0:
            continue
        lines.append(
            f"      └ outcome={ok}: n={int(sub['n'])} "
            f"p50={sub['p50']:.3f} p10={sub['p10']:.3f} p90={sub['p90']:.3f}"
        )
    return lines


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON-safe 值."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _tf_to_minutes(tf: str) -> int:
    """'15T' → 15, '60T' → 60, '240T' → 240"""
    tf = tf.strip().upper()
    if tf.endswith("T"):
        return int(tf[:-1])
    if tf.endswith("MIN"):
        return int(tf[:-3])
    return int(tf)


def _timeframe_from_strategy_meta(strategy: str, strategies_root: str) -> Optional[str]:
    """从策略目录 meta.yaml 读取 timeframe（与 run_live / backtest_execution_layer 对齐）。"""
    import yaml

    meta_path = Path(strategies_root) / strategy / "meta.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        tf = meta.get("timeframe")
        if isinstance(tf, str) and tf.strip():
            return tf.strip()
        st = meta.get("strategy")
        if isinstance(st, dict):
            tf = st.get("timeframe")
            if isinstance(tf, str) and tf.strip():
                return tf.strip()
    except Exception:
        return None
    return None


def _get_bar_minutes(
    strategy: str, *, strategies_root: str = "config/strategies"
) -> int:
    """策略 → 信号时钟分钟数"""
    return _tf_to_minutes(_get_timeframe(strategy, strategies_root=strategies_root))


# 缺失 meta timeframe 时每个 (strategies_root, strategy) 只 warn 一次
_TIMEFRAME_FALLBACK_WARNED: Set[Tuple[str, str]] = set()


def _get_timeframe(strategy: str, *, strategies_root: str = "config/strategies") -> str:
    """策略 → timeframe：仅策略目录 meta.yaml（顶层或 strategy.timeframe）；缺失则 240T 并打一次 warning。"""
    meta_tf = _timeframe_from_strategy_meta(strategy, strategies_root)
    if meta_tf:
        return meta_tf
    key = (strategies_root, strategy)
    if key not in _TIMEFRAME_FALLBACK_WARNED:
        _TIMEFRAME_FALLBACK_WARNED.add(key)
        logger.warning(
            "strategy %r: no timeframe in %s/%s/meta.yaml — using 240T",
            strategy,
            strategies_root,
            strategy,
        )
    return "240T"


# ═════════════════════════════════════════════════════════════════════════════
# 5. EventBacktester — 主回测类
# ═════════════════════════════════════════════════════════════════════════════


class EventBacktester:
    """
    事件驱动回测主类 — 完全模拟实盘多策略环境

    与实盘一致的架构:
      1. LivePCM 仲裁 (全局 slot 控制, 优先级排序, Regime 感知)
      2. 多策略 GenericLiveStrategy.decide() 信号生成 (BPC + FER + ME)
      3. 多 timeframe 特征计算（timeframe 优先来自各策略 meta.yaml）
      4. PositionSimulator: 1min bar 持仓管理
      5. 跨 symbol 时间线交叉处理 (同实盘顺序)

    用法:
        bt = EventBacktester(strategies=["bpc", "fer", "me"], live_root="live/highcap")
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
        fee_rate: float = 0.0,
    ):
        # Keep original strategy casing (e.g. "bpc-short-120T"), because
        # config paths are case-sensitive on Linux.
        self.strategy_names = [s.strip() for s in strategies]
        self.live_root = live_root
        self.data_path = data_path  # 研究数据目录 (e.g. data/parquet_data)
        self.strategies_root = strategies_root or "config/strategies"
        self.fee_rate = fee_rate  # 单边手续费率

        # Per-strategy timeframe 映射
        self._tf_map: Dict[str, str] = {}  # {strategy: "240T"}
        self._bm_map: Dict[str, int] = {}  # {strategy: 240}
        for s in self.strategy_names:
            self._tf_map[s] = _get_timeframe(s, strategies_root=self.strategies_root)
            self._bm_map[s] = _get_bar_minutes(s, strategies_root=self.strategies_root)

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
            # 设置 DatetimeIndex — footprint 计算需要
            ticks_1min = ticks_1min.set_index("timestamp", drop=False).sort_index()
        else:
            ticks_1min = pd.DataFrame()

        return bars_1min, ticks_1min

    def _preload_anchor_macro_cache(
        self,
        *,
        anchor_sym: str,
        warmup_start: str,
        end_date_str: str,
        use_research: bool,
        storage: Optional[Any],
    ) -> Dict[str, pd.Series]:
        """Compute anchor symbol macro_tp_vwap series per timeframe (when anchor ∉ backtest universe)."""
        out: Dict[str, pd.Series] = {}
        if use_research:
            bars_1min, ticks_1min = self._load_research_data(
                anchor_sym, warmup_start, end_date_str
            )
        else:
            if storage is None:
                return out
            bars_1min = storage.bar_1min.load_range(
                anchor_sym, warmup_start, end_date_str
            )
            ticks_1min = storage.ticks.load_range(
                anchor_sym, warmup_start, end_date_str
            )
        if len(bars_1min) < 100:
            logger.warning(
                "macro_tp_vwap_anchor: %s insufficient bars for preload", anchor_sym
            )
            return out
        if "_symbol" not in bars_1min.columns:
            bars_1min = bars_1min.copy()
            bars_1min["_symbol"] = anchor_sym
        for tf, fc in self._feature_computers.items():
            fc._current_symbol = anchor_sym
            features_df = fc.compute_features_dataframe(
                bars_1min=bars_1min,
                ticks_1min=ticks_1min,
                primary_timeframe=tf,
            )
            if features_df.empty or ANCHOR_COLUMN not in features_df.columns:
                continue
            features_df.index = pd.to_datetime(features_df.index, utc=True)
            features_df = _align_feature_index_to_bar_close(features_df, tf)
            out[tf] = features_df[ANCHOR_COLUMN].copy()
        logger.info(
            "macro_tp_vwap_anchor: preloaded %s for %d timeframes",
            anchor_sym,
            len(out),
        )
        return out

    def _global_open_count(self) -> int:
        """跨所有 symbol 的全局 slot 数（仅母仓，加仓不占全局 slot）。"""
        return sum(sim.slot_position_count for sim in self._simulators.values())

    def run(
        self,
        symbols: List[str],
        days: int = 180,
        warmup_days: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fast_mode: bool = False,
        resume_state: Optional[Dict[str, Any]] = None,
        force_close_end: bool = True,
        no_kill_switch: bool = False,
    ) -> BacktestResult:
        """
        运行事件驱动回测 — 多策略 + 多 timeframe + 跨 symbol 时间线交叉处理

        时间范围:
          - 默认: end_date=now(), test_start=end_date - days
          - 指定 --start-date / --end-date: 精确控制, 用于与向量回测对齐
        """
        result = BacktestResult(strategy="+".join(self.strategy_names))
        funnel = defaultdict(int)

        # ── FeatureStore 补充: 检测可用的 FeatureStore layer 用于补充 IFC 缺失的特征 ──
        _fs_layers: Dict[str, str] = {}  # {strategy: layer_name}
        _fs = None
        for s in self.strategy_names:
            _det = detect_layer_for_strategy(
                strategy=s,
                features_store_root="feature_store",
                timeframe=self._tf_map.get(s),
            )
            if _det:
                _fs_layers[s] = _det
        if _fs_layers:
            _fs = FeatureStore("feature_store")
            logger.info(f"FeatureStore layers detected: {_fs_layers}")

        if end_date:
            _end = pd.Timestamp(end_date, tz="UTC")
        else:
            _end = pd.Timestamp(datetime.now(), tz="UTC")
        if start_date:
            _start = pd.Timestamp(start_date, tz="UTC")
        else:
            _start = _end - timedelta(days=days)

        end_date_str = _end.strftime("%Y-%m-%d")
        warmup_start = (_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        test_start_ts = _start
        logger.info(f"Time range: test={_start} → {_end}, warmup_start={warmup_start}")

        # 数据源: --data-path (研究数据) 或 StorageManager (实盘数据)
        use_research = self.data_path is not None
        storage = None
        if not use_research:
            storage = StorageManager(f"{self.live_root}/data")

        # ── Phase 1: 加载数据 + 按 timeframe 计算特征 ──
        sym_data: Dict[str, Dict[str, Any]] = {}
        quantile_dfs_by_tf: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        _meta_full_ev: Dict[str, Any] = {}
        _pri_strat = self.strategy_names[0]
        _meta_path_ev = Path(self.strategies_root) / _pri_strat / "meta.yaml"
        try:
            if _meta_path_ev.exists():
                _meta_full_ev = (
                    yaml.safe_load(_meta_path_ev.read_text(encoding="utf-8")) or {}
                )
        except Exception as _eme:
            logger.warning("macro_tp_vwap_anchor: meta read failed: %s", _eme)
        _meta_strat_ev = _meta_full_ev.get("strategy")
        if not isinstance(_meta_strat_ev, dict):
            _meta_strat_ev = _meta_full_ev if isinstance(_meta_full_ev, dict) else {}
        _anchor_en_ev, _anchor_sym_ev = parse_macro_tp_vwap_anchor_config(
            meta_strategy=_meta_strat_ev,
            meta_yaml_full=_meta_full_ev,
        )
        _anchor_macro_cache: Dict[str, pd.Series] = {}
        _anchor_u = str(_anchor_sym_ev).strip().upper()
        _syms_iter = list(symbols)
        if _anchor_en_ev:
            _syms_iter = sorted(
                _syms_iter,
                key=lambda x: 0 if str(x).strip().upper() == _anchor_u else 1,
            )
            if not any(str(s).strip().upper() == _anchor_u for s in symbols):
                _anchor_macro_cache = self._preload_anchor_macro_cache(
                    anchor_sym=_anchor_sym_ev,
                    warmup_start=warmup_start,
                    end_date_str=end_date_str,
                    use_research=use_research,
                    storage=storage,
                )

        for sym in _syms_iter:
            logger.info(f"{'='*60}")
            logger.info(f"Loading {sym}")
            t0 = time.time()

            if use_research:
                # ── 研究数据路径: DataHandler → 1min bars + ticks ──
                bars_1min, ticks_1min = self._load_research_data(
                    sym, warmup_start, end_date_str
                )
            else:
                # ── 实盘数据路径: StorageManager ──
                bars_1min = storage.bar_1min.load_range(sym, warmup_start, end_date_str)
                ticks_1min = storage.ticks.load_range(sym, warmup_start, end_date_str)

            logger.info(
                f"  Data: {len(bars_1min)} 1min bars, {len(ticks_1min)} ticks "
                f"({time.time()-t0:.1f}s)"
            )
            if len(bars_1min) < 100:
                logger.warning(f"  {sym}: bars 不足, 跳过")
                continue

            # 注入 _symbol 列 — OI join 等特征需要识别 symbol
            if "_symbol" not in bars_1min.columns:
                bars_1min["_symbol"] = sym

            tf_features: Dict[str, pd.DataFrame] = {}
            for tf, fc in self._feature_computers.items():
                t0 = time.time()
                fc._current_symbol = sym  # for health report
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

                # 特征健康报告
                fc.report_feature_health_df(features_df, symbol=sym, timeframe=tf)

                # ── FeatureStore 补充: 合并 IFC 缺失的特征列 ──
                if _fs and _fs_layers:
                    # 按 timeframe 匹配对应的 layer (e.g., 60T → features_me-long_60T_xxx)
                    _layer = None
                    for _s, _ln in _fs_layers.items():
                        _ln_parts = _ln.split("_")
                        if tf in _ln_parts:
                            _layer = _ln
                            break
                    if _layer is None:
                        _layer = next(iter(_fs_layers.values()))  # fallback
                    try:
                        _spec = FeatureStoreSpec(layer=_layer, symbol=sym, timeframe=tf)
                        _fs_start = features_df.index.min()
                        _fs_end = features_df.index.max()
                        if hasattr(_fs_start, "tz") and _fs_start.tz is not None:
                            _fs_start = _fs_start.tz_convert(None)
                            _fs_end = _fs_end.tz_convert(None)
                        _fs_df = _fs.read_range(_spec, start=_fs_start, end=_fs_end)
                        if not _fs_df.empty:
                            _fs_df.index = pd.to_datetime(_fs_df.index, utc=True)
                            _missing = [
                                c
                                for c in _fs_df.columns
                                if c not in features_df.columns
                            ]
                            if _missing:
                                features_df = features_df.join(
                                    _fs_df[_missing], how="left"
                                )
                                # ffill 填充 join 时间戳未对齐产生的 NaN
                                features_df[_missing] = features_df[_missing].ffill()
                                logger.info(
                                    f"  FeatureStore merged {len(_missing)} cols for {sym}/{tf}"
                                )
                            # 用 FeatureStore 填充已有列中的 NaN (e.g., IFC 无 funding_rate 数据)
                            _nan_fill = [
                                c
                                for c in _fs_df.columns
                                if c in features_df.columns
                                and features_df[c].isna().any()
                            ]
                            if _nan_fill:
                                _fs_aligned = _fs_df[_nan_fill].reindex(
                                    features_df.index, method="ffill"
                                )
                                features_df[_nan_fill] = features_df[_nan_fill].fillna(
                                    _fs_aligned
                                )
                    except Exception as e:
                        logger.warning(
                            f"  FeatureStore merge failed for {sym}/{tf}: {e}"
                        )

                features_df.index = pd.to_datetime(features_df.index, utc=True)
                # Keep decision timestamp at bar close to avoid look-ahead leakage.
                features_df = _align_feature_index_to_bar_close(features_df, tf)

                if _anchor_en_ev and ANCHOR_COLUMN in features_df.columns:
                    if str(sym).strip().upper() == _anchor_u:
                        _anchor_macro_cache[tf] = features_df[ANCHOR_COLUMN].copy()
                    else:
                        ser = _anchor_macro_cache.get(tf)
                        if ser is not None and len(ser) > 0:
                            fill = ser.reindex(features_df.index).ffill()
                            features_df[ANCHOR_COLUMN] = fill.to_numpy(
                                dtype=float, copy=False
                            )
                        else:
                            logger.warning(
                                "macro_tp_vwap_anchor: no anchor series for tf=%s sym=%s",
                                tf,
                                sym,
                            )

                quantile_dfs_by_tf[tf].append(features_df)

                test_df = features_df[
                    (features_df.index >= test_start_ts) & (features_df.index <= _end)
                ]
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
            bars_1min_test = bars_1min_idx[
                (bars_1min_idx.index >= test_start_ts) & (bars_1min_idx.index <= _end)
            ]

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
        # 注意: 只用 pre-test 数据校准 quantiles，避免 look-ahead bias
        for s_name, s_obj in self._strats.items():
            tf = self._tf_map[s_name]
            if tf in quantile_dfs_by_tf and quantile_dfs_by_tf[tf]:
                combined = pd.concat(quantile_dfs_by_tf[tf], axis=0)
                # 只用 test_start 之前的数据校准 (与向量回测 warmup calibration 对齐)
                calib_only = combined[combined.index < test_start_ts]
                if len(calib_only) >= 50:
                    s_obj.set_quantiles_from_df(calib_only)
                    logger.info(
                        f"  Quantiles: {s_name} from {tf} "
                        f"({len(calib_only)} pre-test rows, "
                        f"excluded {len(combined) - len(calib_only)} test rows)"
                    )
                else:
                    # 校准数据不足，用全量数据 (回退到旧行为)
                    s_obj.set_quantiles_from_df(combined)
                    logger.warning(
                        f"  Quantiles: {s_name} from {tf} "
                        f"({len(combined)} rows, pre-test only {len(calib_only)} < 50, "
                        f"using all data as fallback)"
                    )

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
            sim = PositionSimulator(
                default_bar_minutes=self._primary_bar_minutes,
                max_positions=len(self.strategy_names),  # 每策略独占 1 slot
                fee_rate=self.fee_rate,
            )
            if self._om_bridge:
                sim._om_bridge = self._om_bridge
            self._simulators[sym] = sim

        _srb_add_policy: Optional[Dict[str, Any]] = None
        if "srb" in self.strategy_names:
            try:
                _srb_add_policy = (
                    self._strats["srb"].archetype.execution.raw or {}
                ).get("srb_add_position_policy")
            except Exception:
                _srb_add_policy = None
        _srb_wide_entry_guard: Optional[Dict[str, Any]] = None
        if "srb" in self._strats:
            try:
                _srb_wide_entry_guard = (
                    self._strats["srb"].archetype.execution.raw or {}
                ).get("sr_wide_entry_guard")
            except Exception:
                _srb_wide_entry_guard = None
        for _sim in self._simulators.values():
            _sim._srb_add_policy = _srb_add_policy
            _sim._srb_wide_entry_guard = _srb_wide_entry_guard

        # 可选: 加载跨月续跑状态
        if resume_state:
            resume_symbols = resume_state.get("symbols", {}) or {}
            loaded_total = 0
            for sym, sym_obj in resume_symbols.items():
                sim = self._simulators.get(str(sym))
                if sim is None or not isinstance(sym_obj, dict):
                    continue
                rows = sym_obj.get("open_positions", []) or []
                loaded = sim.restore_open_positions(rows)
                loaded_total += loaded
                if loaded > 0 and hasattr(self.pcm, "_record_slot"):
                    for row in rows:
                        pos = (row or {}).get("position", {}) or {}
                        if bool(pos.get("_is_add_position", False)):
                            continue
                        arch = str(pos.get("archetype", "") or "").strip()
                        if not arch:
                            continue
                        ev = float(pos.get("evidence_score", 0.5) or 0.5)
                        try:
                            self.pcm._record_slot(str(sym), arch, ev)
                        except Exception:
                            pass
            if loaded_total > 0:
                logger.info("Resumed open positions: %d", loaded_total)

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Timeline: {len(timeline_events)} events across {len(sym_data)} symbols"
        )
        logger.info(f"Strategies: {', '.join(self.strategy_names)}")
        logger.info(f"PCM max_slots={self.pcm._max_slots}")

        # ── Phase 3: 遍历统一时间线 ──
        prev_ts: Dict[str, pd.Timestamp] = {}

        # ── Constitution Executor (复用实盘同一份代码) ──
        _executor: Optional[ConstitutionExecutor] = None
        _runtime_state = ConstitutionRuntimeState()
        _safety_state = SafetyRuntimeState()
        if resume_state:
            ap_root = resume_state.get("add_position_state")
            if isinstance(ap_root, dict) and ap_root.get("positions"):
                _load_add_position_runtime_from_resume(resume_state, _runtime_state)
            open_parent_ids = _collect_open_parent_pids(self._simulators)
            _prune_stale_add_position_records(_runtime_state, open_parent_ids)
            for _sim in self._simulators.values():
                _merge_add_position_runtime_with_open_legs(_sim, _runtime_state)
        constitution_path = str(Path("config") / "constitution" / "constitution.yaml")
        try:
            _executor = ConstitutionExecutor(constitution_yaml=constitution_path)
            if no_kill_switch:
                import dataclasses as _dc

                object.__setattr__(
                    _executor, "cfg", _dc.replace(_executor.cfg, kill_enabled=False)
                )
                logger.info("Kill Switch 已通过 --no-kill-switch 禁用")
            elif _executor.cfg.kill_enabled:
                logger.info(
                    f"Kill Switch (共享 evaluate_safety_state): "
                    f"max_dd={_executor.cfg.max_dd:.0%}, "
                    f"daily={_executor.cfg.daily_loss_limit:.0%}, "
                    f"cooldown={_executor.cfg.cooldown_minutes}min"
                )
        except Exception as e:
            logger.warning(f"Constitution 加载失败, kill switch/加仓禁用: {e}")

        # 加仓启用检查 (从 executor 读取, 与实盘同一份 resolve 逻辑)
        _add_pos_enabled = False
        _add_pos_count = 0
        _add_pos_rejected = 0
        if _executor:
            try:
                _psl = _executor._resolve_per_strategy_limits()
                _add_pos_enabled = any(
                    isinstance(v, dict) and v.get("allow_add_position", False)
                    for v in _psl.values()
                )
                if _add_pos_enabled:
                    _max_add_vals = [
                        int(v.get("max_add_times", 1) or 1)
                        for v in _psl.values()
                        if isinstance(v, dict) and v.get("allow_add_position", False)
                    ]
                    logger.info(
                        f"加仓模拟 (共享 validate_add_position): "
                        f"max_add={max(_max_add_vals) if _max_add_vals else 1}, "
                        "trigger_r=execution.add_position"
                    )
            except Exception:
                pass
        # execution.add_position.trigger.type=float_r_ladder_only — 浮盈阶梯加仓（事件回测，不依赖 PCM 再次发信号）
        _strats_float_ladder_meta: Dict[str, Dict[str, Any]] = {}
        _add_trigger_types: Dict[str, str] = {}
        for s in self.strategy_names:
            raw = self._strats[s].archetype.execution.raw or {}
            ap = raw.get("add_position")
            _tt = ""
            if isinstance(ap, dict):
                _trg = ap.get("trigger") or {}
                if isinstance(_trg, dict):
                    _tt = str(_trg.get("type", "") or "").strip()
            _add_trigger_types[s.lower()] = _tt or "(missing trigger.type)"
            if isinstance(ap, dict) and _shared_resolve_float_r_ladder_only(ap):
                _strats_float_ladder_meta[s.lower()] = {
                    "strategy": s,
                    "add_position": dict(ap),
                    "execution_constraints": dict(
                        raw.get("execution_constraints") or {}
                    ),
                }
        result.add_trigger_types = dict(_add_trigger_types)
        _risk_per_slot = float(
            self.pcm._constitution.get("risk_per_slot", 0.01)
            if hasattr(self.pcm, "_constitution") and self.pcm._constitution
            else 0.01
        )
        _initial_cash = 1000.0
        _equity = _initial_cash
        _equity_curve = [_equity]
        if fast_mode:
            logger.info(
                "Fast mode compatibility: using 1min-exact position updates "
                "to avoid approximation drift."
            )
        _equity_peak = _equity
        _ks_triggers: list = []
        # SRB：2a+2b staged 首仓门控（execution.srb_staged_entry_2b.enabled）
        _srb_staged_rt = None
        _srb_tf_global = (
            self._tf_map.get("srb") if "srb" in self.strategy_names else None
        )
        if "srb" in self.strategy_names:
            _st_blk = (self._strats["srb"].archetype.execution.raw or {}).get(
                "srb_staged_entry_2b"
            )
            if isinstance(_st_blk, dict) and bool(_st_blk.get("enabled")):
                from src.time_series_model.live.srb_staged_entry_2b import (
                    SrbStagedEntry2bRuntime,
                )

                _srb_staged_rt = SrbStagedEntry2bRuntime.from_execution_block(_st_blk)
                logger.info(
                    "SRB staged entry 2b: ENABLED (PCM mother entries require arm window)"
                )
        _ks_skipped = 0
        _ks_executed = 0
        _period_equity_daily = _equity
        _period_equity_weekly = _equity
        _period_equity_monthly = _equity
        _prev_day = None
        _prev_week = None
        _prev_month = None

        # ── 每日入场节流 (max_new_entries_per_day) ──
        _daily_entry_limits: Dict[str, Optional[int]] = {}
        if _executor:
            for s in self.strategy_names:
                _daily_entry_limits[s.lower()] = (
                    _executor.resolve_max_new_entries_per_day(s)
                )
        _daily_entry_counts: Dict[tuple, int] = {}  # (strategy, date) -> count
        _daily_entry_limit_log = False
        for _s, _lim in _daily_entry_limits.items():
            if _lim is not None:
                if not _daily_entry_limit_log:
                    logger.info("每日入场节流:")
                    _daily_entry_limit_log = True
                logger.info(f"  {_s}: max_new_entries_per_day={_lim}")

        # _pos_last_ts: 独立跟踪每个 symbol 持仓上次被处理到的时间点
        # 与 prev_ts (信号时间) 分离, 确保跨 symbol slot 释放不延迟
        _pos_last_ts: Dict[str, pd.Timestamp] = {}
        # path_efficiency_pct（类 ER 分位）在每次加仓尝试时的快照，供 er_gated_float_ladder 设计
        _er_rows_signal_add: List[Dict[str, Any]] = []
        _er_rows_float_ladder: List[Dict[str, Any]] = []
        _funnel_per_bar_rows: List[Dict[str, Any]] = []
        _pcm_direction_ffill: Dict[Tuple[str, str], Dict[str, float]] = {}

        for ts, sym, tf_rows in timeline_events:
            simulator = self._simulators[sym]
            bars_1min_test = sym_data[sym]["bars_1min_test"]
            funnel["total_signals_checked"] += 1

            # ── 更新所有 symbol 的持仓到当前 ts (模拟实盘实时 bar 处理) ──
            # fast_mode=True: 用当前 timeframe bar 的 OHLC 直接更新持仓 (60x faster)
            # fast_mode=False: 用 1min bars 逐分钟更新 (精确但慢)
            for upd_sym, upd_sim in self._simulators.items():
                if not upd_sim.has_positions:
                    continue
                upd_prev = _pos_last_ts.get(upd_sym)
                if upd_prev is None or upd_prev >= ts:
                    continue

                upd_bars = sym_data[upd_sym]["bars_1min_test"]
                _sym_bundle = sym_data.get(upd_sym) or {}
                for bar_ts, bar_row in _iter_update_bars_1min(
                    upd_bars,
                    upd_prev,
                    ts,
                    fast_mode=fast_mode,
                ):
                    _frow = _feature_row_asof_from_sym_tf_features(
                        _sym_bundle,
                        bar_ts,
                        require_macro=True,
                    )
                    _sync_macro_tp_vwap_from_feature_row(upd_sim, _frow)
                    _sync_ema_1200_from_feature_row(upd_sim, _frow)
                    if _frow is not None and "ema_200" in _frow.index:
                        try:
                            _e = float(_frow["ema_200"])
                            if _e == _e and _e > 0.0:
                                upd_sim._structural_price = _e
                        except (TypeError, ValueError):
                            pass
                    else:
                        _ema_upd = _feature_asof_from_sym_tf_features(
                            _sym_bundle,
                            bar_ts,
                            "ema_200",
                        )
                        if _ema_upd is not None:
                            upd_sim._structural_price = _ema_upd
                    bar_dict = {
                        "timestamp": bar_ts,
                        "open": float(bar_row.get("open", 0)),
                        "high": float(bar_row.get("high", 0)),
                        "low": float(bar_row.get("low", 0)),
                        "close": float(bar_row.get("close", 0)),
                    }
                    closed = upd_sim.update(bar_dict)
                    for ct in closed:
                        self.pcm.notify_position_closed(upd_sym, ct.archetype)
                    for ct in closed:
                        sl_r_val = 1.0
                        pnl_usd = _initial_cash * _risk_per_slot * ct.pnl_r / sl_r_val
                        _equity += pnl_usd
                        _equity = max(_equity, 0.0)
                        _equity_curve.append(_equity)
                        if _equity > _equity_peak:
                            _equity_peak = _equity
                _pos_last_ts[upd_sym] = ts

            # ── Kill switch 检查 (复用实盘 evaluate_safety_state) ──
            _ks_blocked = False
            if _executor and _executor.cfg.kill_enabled:
                # 日/周/月 边界重置
                ts_date = ts.date() if hasattr(ts, "date") else None
                ts_week = ts.isocalendar()[1] if hasattr(ts, "isocalendar") else None
                ts_month = ts.month if hasattr(ts, "month") else None
                if ts_date and ts_date != _prev_day:
                    _period_equity_daily = _equity
                    _prev_day = ts_date
                if ts_week and ts_week != _prev_week:
                    _period_equity_weekly = _equity
                    _prev_week = ts_week
                if ts_month and ts_month != _prev_month:
                    _period_equity_monthly = _equity
                    _prev_month = ts_month

                dd = (
                    (_equity_peak - _equity) / _equity_peak if _equity_peak > 0 else 0.0
                )
                d_loss = (
                    max(0.0, (_period_equity_daily - _equity) / _period_equity_daily)
                    if _period_equity_daily > 0
                    else 0.0
                )
                w_loss = (
                    max(0.0, (_period_equity_weekly - _equity) / _period_equity_weekly)
                    if _period_equity_weekly > 0
                    else 0.0
                )
                m_loss = (
                    max(
                        0.0, (_period_equity_monthly - _equity) / _period_equity_monthly
                    )
                    if _period_equity_monthly > 0
                    else 0.0
                )

                # 调用实盘同一份 evaluate_safety_state (来自 safety_runtime.py)
                now_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                if hasattr(now_dt, "tzinfo") and now_dt.tzinfo is None:
                    now_dt = now_dt.replace(tzinfo=timezone.utc)
                _safety_decision = evaluate_safety_state(
                    state=_safety_state,
                    now=now_dt,
                    cooldown_minutes=int(_executor.cfg.cooldown_minutes),
                    daily_reset_tz=_executor.cfg.daily_reset_timezone,
                    daily_loss=d_loss,
                    weekly_loss=w_loss,
                    monthly_loss=m_loss,
                    drawdown=dd,
                    hard_violation=False,
                    data_bad=False,
                    daily_cost_mean=None,
                    daily_turnover_mean=None,
                    limits={
                        "max_dd": float(_executor.cfg.max_dd),
                        "daily_loss_limit": float(_executor.cfg.daily_loss_limit),
                        "weekly_loss_limit": float(_executor.cfg.weekly_loss_limit),
                        "monthly_loss_limit": float(_executor.cfg.monthly_loss_limit),
                        "max_turnover_mean": float(_executor.cfg.max_turnover_mean),
                        "max_cost_mean": float(_executor.cfg.max_cost_mean),
                    },
                )
                if not _safety_decision.ok:
                    _ks_blocked = True
                    _ks_triggers.append(
                        {
                            "timestamp": str(ts),
                            "reasons": list(_safety_decision.reasons),
                            "equity": _equity,
                            "dd": dd,
                        }
                    )

            # 构建 features_by_timeframe 供 PCM 路由
            features_by_tf: Dict[str, Dict[str, float]] = {}
            for tf, row in tf_rows.items():
                _fd = row_to_features(row)
                _apply_pcm_direction_ffill(sym, tf, _fd, _pcm_direction_ffill)
                features_by_tf[tf] = _fd

            # 主特征 = 第一个可用 timeframe 的特征 (PCM 回退用)
            primary_features = next(iter(features_by_tf.values()))

            # SRB 实验：在策略对应 primary TF 特征上注入 srb_regime_* / srb_sr_*（供 ExecutionParamGenerator）
            if "srb" in self.strategy_names:
                _tf_srb = self._tf_map.get("srb")
                _raw_srb = self._strats["srb"].archetype.execution.raw or {}
                _df_srb = (
                    sym_data[sym]["tf_features"].get(_tf_srb)
                    if _tf_srb and sym in sym_data
                    else None
                )
                if (
                    _df_srb is not None
                    and not _df_srb.empty
                    and _tf_srb
                    and _tf_srb in features_by_tf
                ):
                    maybe_inject_srb_experiment_features(
                        df=_df_srb,
                        ts=ts,
                        exec_raw=_raw_srb,
                        out=features_by_tf[_tf_srb],
                    )
                    for _k, _v in list(features_by_tf[_tf_srb].items()):
                        if str(_k).startswith("srb_"):
                            primary_features[_k] = _v

            try:
                _pat = float(primary_features.get("atr") or 0)
                if _pat > 0:
                    simulator._primary_tf_atr = _pat
            except (TypeError, ValueError):
                pass

            # L3 dynamic trailing 需要最新的 wide_sr_upper_px / wide_sr_lower_px
            # （来自 wide_sr_swing_f，默认 240 bar 窗口）。随 primary bar 同步更新。
            for _wk in ("wide_sr_upper_px", "wide_sr_lower_px"):
                _wv = primary_features.get(_wk)
                if _wv is None:
                    continue
                try:
                    _wf = float(_wv)
                    if _wf == _wf:
                        setattr(simulator, f"_{_wk}", _wf)
                except (TypeError, ValueError):
                    pass

            # Phase D: 维护近 N primary close 的滚动缓存用于 recent_net_move_atr
            try:
                _pc_val = primary_features.get("close")
                if _pc_val is not None:
                    _pc_f = float(_pc_val)
                    if _pc_f == _pc_f:
                        simulator._primary_close_buffer.append(_pc_f)
                        if (
                            len(simulator._primary_close_buffer)
                            > simulator._primary_close_buffer_max
                        ):
                            simulator._primary_close_buffer.pop(0)
            except (TypeError, ValueError):
                pass

            simulator._primary_bar_count += 1

            # SRB staged 2b：每根 primary 推进 cross+EMA 确认，产生 arm 窗口
            if (
                _srb_staged_rt is not None
                and _srb_tf_global
                and "srb" in self.strategy_names
            ):
                _df_srb_adv = (
                    sym_data.get(sym, {}).get("tf_features", {}).get(_srb_tf_global)
                )
                if _df_srb_adv is not None and not _df_srb_adv.empty:
                    _has_srb_open = any(
                        str(p.get("archetype", "")).lower().strip() == "srb"
                        for p in simulator._positions.values()
                    )
                    _row_adv = {
                        k: primary_features.get(k)
                        for k in (
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "atr",
                            "ema_1200_position",
                        )
                    }
                    _row_adv["volume_ma"] = primary_features.get("volume_ma")
                    _srb_staged_rt.advance(
                        symbol=sym,
                        df_srb=_df_srb_adv,
                        ts=ts,
                        bar_idx=int(simulator._primary_bar_count),
                        row=_row_adv,
                        has_srb_position=_has_srb_open,
                    )

            # 更新 structural_price (EMA200) 用于 BPC trend_hold 结构性退出
            _ema_200_val = primary_features.get("ema_200")
            if _ema_200_val is not None:
                try:
                    simulator._structural_price = float(_ema_200_val)
                except (TypeError, ValueError):
                    pass
            _mv = primary_features.get("macro_tp_vwap_1200_position")
            if _mv is not None:
                try:
                    simulator._macro_tp_vwap_position = float(_mv)
                except (TypeError, ValueError):
                    pass
            else:
                simulator._macro_tp_vwap_level = None
            try:
                _pcl = primary_features.get("close")
                if _mv is not None and _pcl is not None:
                    _c = float(_pcl)
                    _m = float(_mv)
                    if _c > 0.0 and _m == _m:
                        simulator._macro_tp_vwap_level = _c * (1.0 - _m)
            except (TypeError, ValueError):
                pass

            # EMA1200 structural exit: 同步 ema_1200_position 和冻结的 EMA1200 水平
            _ev = primary_features.get("ema_1200_position")
            if _ev is not None:
                try:
                    simulator._ema_1200_position = float(_ev)
                except (TypeError, ValueError):
                    pass
            else:
                simulator._ema_1200_level = None
            try:
                if _ev is not None and _pcl is not None:
                    _c = float(_pcl)
                    _e = float(_ev)
                    if _c > 0.0 and _e == _e:
                        simulator._ema_1200_level = _c * (1.0 - _e)
            except (TypeError, ValueError):
                pass

            # LivePCM.decide() — 多策略仲裁 + 全局 slot 控制
            intents = self.pcm.decide(
                features=primary_features,
                symbol=sym,
                features_by_timeframe=features_by_tf,
                decision_time=ts,
            )
            _pcm_tr = dict(getattr(self.pcm, "_last_decide_trace", None) or {})

            for s_name, s_obj in self._strats.items():
                lf = getattr(s_obj, "_last_funnel", None) or {}
                if not lf:
                    continue
                _frow = {
                    "timestamp": ts,
                    "symbol": sym,
                    "strategy": s_name,
                    "pcm_direction_filter": lf.get("pcm_direction_filter"),
                    "prefilter": lf.get("prefilter"),
                    "gate": lf.get("gate"),
                    "entry_filter": lf.get("entry_filter"),
                    "direction": lf.get("direction"),
                    "direction_value": lf.get("direction_value"),
                    "pcm_n_candidates": int(_pcm_tr.get("all_intents", 0) or 0),
                    "pcm_n_accepted": int(len(intents)),
                    "pcm_drop_direction_policy": int(
                        _pcm_tr.get("drop_direction_policy", 0) or 0
                    ),
                    "pcm_drop_family_conflict": int(
                        _pcm_tr.get("drop_family_conflict", 0) or 0
                    ),
                    "pcm_drop_daily_limit": int(
                        _pcm_tr.get("drop_daily_limit", 0) or 0
                    ),
                    "pcm_drop_slot": int(_pcm_tr.get("drop_slot", 0) or 0),
                }
                # 供离线统计「是 gate 挡还是 prefilter 挡、触发了哪条规则」
                if lf.get("gate_reasons") is not None:
                    _frow["gate_reasons"] = lf.get("gate_reasons")
                if lf.get("prefilter_reason") is not None:
                    _frow["prefilter_reason"] = lf.get("prefilter_reason")
                _funnel_per_bar_rows.append(_frow)

            # NOTE: Evidence slot 竞争已移除 (改为入场门槛 + 仓位缩放)
            # _last_evictions 始终为空, 此块保留为 no-op 以保持兼容
            for evicted_sym, evicted_arch in getattr(self.pcm, "_last_evictions", []):
                ev_sim = self._simulators.get(evicted_sym)
                if ev_sim and ev_sim.has_positions:
                    ev_bars = sym_data[evicted_sym]["bars_1min_test"]
                    ev_close_price = 0.0
                    _ev_mask = ev_bars.index <= ts
                    if _ev_mask.any():
                        ev_close_price = float(ev_bars.loc[_ev_mask, "close"].iloc[-1])
                    ev_close_time = (
                        ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                    )
                    if (
                        hasattr(ev_close_time, "tzinfo")
                        and ev_close_time.tzinfo is None
                    ):
                        ev_close_time = ev_close_time.replace(tzinfo=timezone.utc)
                    ev_closed = ev_sim.close_by_archetype(
                        evicted_arch, ev_close_price, ev_close_time
                    )
                    for ct in ev_closed:
                        self.pcm.notify_position_closed(evicted_sym, ct.archetype)
                        pnl_usd = _initial_cash * _risk_per_slot * ct.pnl_r
                        _equity += pnl_usd
                        _equity = max(_equity, 0.0)
                        _equity_curve.append(_equity)
                        if _equity > _equity_peak:
                            _equity_peak = _equity
                    funnel.setdefault("evicted_by_evidence", 0)
                    funnel["evicted_by_evidence"] += len(ev_closed)

            if intents:
                funnel["signals_generated"] += len(intents)

                for intent in intents:
                    # Kill switch 模拟: 被暂停时拒绝新入场
                    if _ks_blocked:
                        _ks_skipped += 1
                        funnel.setdefault("reject_kill_switch", 0)
                        funnel["reject_kill_switch"] += 1
                        continue
                    _ks_executed += 1
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
                    winning_bm = self._bm_map.get(
                        winning_arch, self._primary_bar_minutes
                    )

                    # ── 每日入场节流: 仅限新开仓（非加仓），检查日内上限 ──
                    _arch_lc = str(winning_arch or "").strip().lower()
                    _entry_limit = _daily_entry_limits.get(_arch_lc)
                    _ts_date = ts.date() if hasattr(ts, "date") else None
                    _is_new_entry = not any(
                        p.get("symbol") == sym
                        and str(p.get("archetype", "")).lower().strip() == _arch_lc
                        for p in simulator._positions.values()
                    )
                    if (
                        _entry_limit is not None
                        and _ts_date is not None
                        and _is_new_entry
                    ):
                        _dk = (_arch_lc, _ts_date)
                        if _daily_entry_counts.get(_dk, 0) >= _entry_limit:
                            funnel.setdefault("reject_daily_entry_limit", 0)
                            funnel["reject_daily_entry_limit"] += 1
                            continue

                    # ── Exp 4c: 宽窗 SR 入场屏蔽 ──
                    # sr_wide_entry_guard:
                    #   enabled: true
                    #   min_distance_atr: 2.0   # 价格到"反向"宽窗 SR < min×ATR 则拒绝新开仓
                    #   apply_to_new_only: true # 仅拦截全新开仓，不拦加仓/反手
                    if _arch_lc == "srb" and _is_new_entry:
                        _wg = getattr(simulator, "_srb_wide_entry_guard", None) or {}
                        if _wg.get("enabled"):
                            _min_atr = float(_wg.get("min_distance_atr", 0) or 0)
                            _atr_wg = float(entry_feats.get("atr", 0) or 0)
                            _px_wg = float(entry_feats.get("close", 0) or 0)
                            _side_wg = str(intent.action or "").upper()
                            if should_reject_srb_wide_entry(
                                _side_wg,
                                _px_wg,
                                _atr_wg,
                                entry_feats.get("wide_sr_lower_px"),
                                entry_feats.get("wide_sr_upper_px"),
                                _min_atr,
                            ):
                                funnel.setdefault("reject_srb_wide_sr_too_close", 0)
                                funnel["reject_srb_wide_sr_too_close"] += 1
                                continue

                    # SRB staged 2b：首仓需与 cross+EMA arm 同向且在窗口内（先于 open_position）
                    if (
                        _arch_lc == "srb"
                        and _is_new_entry
                        and _srb_staged_rt is not None
                    ):
                        _pcm_side = str(intent.action or "").upper()
                        if _pcm_side in ("BUY",):
                            _pcm_side = "LONG"
                        if _pcm_side in ("SELL",):
                            _pcm_side = "SHORT"
                        if not _srb_staged_rt.match_arm(
                            sym,
                            _pcm_side,
                            int(simulator._primary_bar_count),
                        ):
                            funnel.setdefault("reject_srb_staged_2b_arm", 0)
                            funnel["reject_srb_staged_2b_arm"] += 1
                            continue

                    opened = simulator.open_position(
                        intent, entry_bar, entry_feats, bar_minutes=winning_bm
                    )
                    if (
                        opened is not None
                        and _arch_lc == "srb"
                        and _is_new_entry
                        and _srb_staged_rt is not None
                    ):
                        _srb_staged_rt.consume_arm(sym)
                    if opened is not None and _entry_limit is not None and _ts_date:
                        _dk2 = (_arch_lc, _ts_date)
                        _daily_entry_counts[_dk2] = _daily_entry_counts.get(_dk2, 0) + 1
                    if opened is not None and _arch_lc == "srb":
                        _srb_pos = simulator._positions.get(opened)
                        if _srb_pos is not None:
                            _srb_side = str(_srb_pos.get("side", "")).upper()
                            # true_sr_level.wide_fallback_atr: 窄窗紧贴入场时改用 L3 大级别锚点
                            try:
                                _tsl_cfg = (
                                    self._strats["srb"].archetype.execution.raw or {}
                                ).get("true_sr_level") or {}
                            except Exception:
                                _tsl_cfg = {}
                            _fallback_atr = float(
                                _tsl_cfg.get("wide_fallback_atr", 0) or 0
                            )
                            _entry_px = float(entry_bar.get("close", 0) or 0)
                            _atr_e = float(entry_feats.get("atr", 0) or 0)
                            _pick = pick_srb_true_sr_level(
                                _srb_side,
                                _entry_px,
                                _atr_e,
                                narrow_support=entry_feats.get("srb_sr_support"),
                                narrow_resistance=entry_feats.get("srb_sr_resistance"),
                                wide_lower_px=entry_feats.get("wide_sr_lower_px"),
                                wide_upper_px=entry_feats.get("wide_sr_upper_px"),
                                fallback_atr=_fallback_atr,
                            )
                            _srb_pos["_srb_true_sr_level"] = float(_pick)
                            # Phase D: 记录母仓入场时 wide_sr_dist_atr，供加仓 wide_sr_expansion gate 比对
                            try:
                                _ewd = entry_feats.get("wide_sr_dist_atr")
                                if _ewd is not None and _ewd == _ewd:
                                    _srb_pos["_srb_entry_wide_sr_dist_atr"] = float(
                                        _ewd
                                    )
                            except (TypeError, ValueError):
                                pass
                    if opened is None:
                        _atr_open = float(entry_feats.get("atr", 0) or 0.0)
                        _dup_open = any(
                            p.get("symbol") == sym
                            and str(p.get("archetype", "")).lower().strip() == _arch_lc
                            for p in simulator._positions.values()
                        )
                        if _atr_open <= 0.0:
                            funnel.setdefault("reject_open_atr_nonpositive", 0)
                            funnel["reject_open_atr_nonpositive"] += 1
                        elif _dup_open:
                            funnel.setdefault("reject_open_duplicate_archetype", 0)
                            funnel["reject_open_duplicate_archetype"] += 1
                        # 已有持仓，尝试加仓（trigger.type=float_r_ladder_only 时仅走下方阶梯逻辑）
                        elif _add_pos_enabled and _executor:
                            _win_lc = str(winning_arch or "").strip().lower()
                            added = None
                            _tried_signal_add = _win_lc not in _strats_float_ladder_meta
                            if _tried_signal_add:
                                added = simulator.try_add_position(
                                    intent,
                                    entry_bar,
                                    entry_feats,
                                    executor=_executor,
                                    runtime_state=_runtime_state,
                                    bar_minutes=winning_bm,
                                )
                                _er_rows_signal_add.append(
                                    {
                                        "pct": _extract_path_efficiency_pct(
                                            entry_feats
                                        ),
                                        "outcome": (
                                            "ok"
                                            if added
                                            else str(
                                                getattr(
                                                    simulator,
                                                    "last_add_reject_reason",
                                                    "",
                                                )
                                            )
                                            or "other"
                                        ),
                                    }
                                )
                            if added:
                                _add_pos_count += 1
                                funnel.setdefault("add_position_ok", 0)
                                funnel["add_position_ok"] += 1
                            elif _tried_signal_add:
                                _add_pos_rejected += 1
                                funnel.setdefault("add_position_rejected", 0)
                                funnel["add_position_rejected"] += 1
                                _why = (
                                    str(
                                        getattr(simulator, "last_add_reject_reason", "")
                                    )
                                    or "other"
                                )
                                if _why == "max_add_times":
                                    funnel.setdefault("reject_add_max_times", 0)
                                    funnel["reject_add_max_times"] += 1
                                elif _why == "constitution_reject":
                                    funnel.setdefault("reject_add_constitution", 0)
                                    funnel["reject_add_constitution"] += 1
                                elif _why in (
                                    "trigger_not_met",
                                    "add_min_current_r",
                                    "add_bpc_breakout_mismatch",
                                    "add_trigger_feature_rules",
                                ):
                                    funnel.setdefault("reject_add_trigger", 0)
                                    funnel["reject_add_trigger"] += 1
                                    if _why == "add_min_current_r":
                                        funnel.setdefault("reject_add_detail_min_r", 0)
                                        funnel["reject_add_detail_min_r"] += 1
                                    elif _why == "add_bpc_breakout_mismatch":
                                        funnel.setdefault(
                                            "reject_add_detail_bpc_breakout", 0
                                        )
                                        funnel["reject_add_detail_bpc_breakout"] += 1
                                    elif _why == "add_trigger_feature_rules":
                                        funnel.setdefault(
                                            "reject_add_detail_me_features", 0
                                        )
                                        funnel["reject_add_detail_me_features"] += 1
                                elif _why == "no_parent_position":
                                    funnel.setdefault("reject_add_no_parent", 0)
                                    funnel["reject_add_no_parent"] += 1
                                elif _why == "srb_policy_regime_bucket":
                                    funnel.setdefault("reject_add_srb_regime_bucket", 0)
                                    funnel["reject_add_srb_regime_bucket"] += 1
                                elif _why == "srb_policy_volume_compression":
                                    funnel.setdefault(
                                        "reject_add_srb_volume_compression", 0
                                    )
                                    funnel["reject_add_srb_volume_compression"] += 1
                                elif _why.startswith("shape_gate_"):
                                    _key = f"reject_add_{_why}"
                                    funnel.setdefault(_key, 0)
                                    funnel[_key] += 1
                                else:
                                    funnel.setdefault("reject_add_other", 0)
                                    funnel["reject_add_other"] += 1
                        else:
                            funnel["reject_max_positions"] += 1
            else:
                _pcm_cand = int(_pcm_tr.get("all_intents", 0) or 0)
                if _pcm_cand > 0:
                    funnel["reject_pcm_direction_policy"] = int(
                        funnel.get("reject_pcm_direction_policy", 0) or 0
                    ) + int(_pcm_tr.get("drop_direction_policy", 0) or 0)
                    funnel["reject_pcm_family_conflict"] = int(
                        funnel.get("reject_pcm_family_conflict", 0) or 0
                    ) + int(_pcm_tr.get("drop_family_conflict", 0) or 0)
                    funnel["reject_pcm_daily_throttle"] = int(
                        funnel.get("reject_pcm_daily_throttle", 0) or 0
                    ) + int(_pcm_tr.get("drop_daily_limit", 0) or 0)
                    funnel["reject_pcm_slot_full"] = int(
                        funnel.get("reject_pcm_slot_full", 0) or 0
                    ) + int(_pcm_tr.get("drop_slot", 0) or 0)
                # 诊断拒绝原因: 逐策略检查 _last_funnel 确定最深到达阶段
                _had_signal = False
                _deepest = "no_evaluable_signal"  # 最浅
                for s_name, s_obj in self._strats.items():
                    lf = getattr(s_obj, "_last_funnel", {})
                    if not lf:
                        continue  # 未评估 (timeframe 不匹配 / 空特征)
                    if lf.get("pcm_direction_filter") is False:
                        if _deepest == "no_evaluable_signal":
                            _deepest = "pcm_direction_filter"
                        continue
                    if lf.get("prefilter") is False:
                        if _deepest in ("no_evaluable_signal", "pcm_direction_filter"):
                            _deepest = "prefilter_deny"
                        continue
                    if not lf.get("direction", False):
                        continue  # direction=0, 无信号
                    # direction != 0
                    if lf.get("gate") is False:
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                            "prefilter_deny",
                        ):
                            _deepest = "gate_deny"
                        continue
                    # gate passed (or no gate)
                    if lf.get("entry_filter") is False:
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                            "prefilter_deny",
                            "gate_deny",
                        ):
                            _deepest = "entry_filter_deny"
                        continue
                    # 全部通过 → 策略层已产 intent，但 pcm.decide 返回空（见 _pcm_tr）
                    _had_signal = True
                    break
                if _pcm_cand > 0:
                    # 已在上方按 _last_decide_trace 细分，不再用 _deepest 重复归因
                    pass
                elif _had_signal:
                    funnel.setdefault("reject_pcm_struct_pass_no_intent", 0)
                    funnel["reject_pcm_struct_pass_no_intent"] += 1
                elif _deepest == "pcm_direction_filter":
                    funnel["reject_pcm_direction_filter"] += 1
                elif _deepest == "prefilter_deny":
                    funnel["reject_prefilter_deny"] += 1
                elif _deepest == "gate_deny":
                    funnel["reject_gate_deny"] += 1
                elif _deepest == "entry_filter_deny":
                    funnel["reject_entry_filter_deny"] += 1
                else:
                    funnel["reject_no_direction"] += 1

            # 浮盈阶梯加仓: 不依赖 PCM 再次发信号，按 min_current_r_by_add 逐档检查
            if (
                _strats_float_ladder_meta
                and _add_pos_enabled
                and _executor
                and not _ks_blocked
            ):
                entry_bar_primary = {
                    "close": float(primary_features.get("close", 0) or 0),
                    "high": float(primary_features.get("high", 0) or 0),
                    "low": float(primary_features.get("low", 0) or 0),
                    "open": float(primary_features.get("open", 0) or 0),
                    "timestamp": ts,
                    "atr": float(primary_features.get("atr", 0) or 0),
                }
                pf = dict(primary_features)
                pf["equity"] = float(_equity)
                for arch_lc, meta in _strats_float_ladder_meta.items():
                    ladder_done = False
                    for _pid, pos in list(simulator._positions.items()):
                        if ladder_done:
                            break
                        if pos.get("symbol") != sym:
                            continue
                        if bool(pos.get("_is_add_position", False)):
                            continue
                        if str(pos.get("archetype", "")).strip().lower() != arch_lc:
                            continue
                        min_gap_m = float(
                            (meta.get("execution_constraints") or {}).get(
                                "min_order_interval_minutes", 0
                            )
                            or 0
                        )
                        last_add = pos.get("_last_float_ladder_add_ts")
                        if min_gap_m > 0 and last_add is not None:
                            try:
                                gap_min = (
                                    pd.Timestamp(ts) - pd.Timestamp(last_add)
                                ).total_seconds() / 60.0
                            except Exception:
                                gap_min = 1.0e9
                            if gap_min < min_gap_m:
                                continue
                        arch_disp = str(pos.get("archetype", "")).strip()
                        side = pos.get("side", "LONG")
                        action = (
                            "LONG" if str(side).upper() in ("LONG", "BUY") else "SHORT"
                        )
                        ladder_intent = TradeIntent(
                            action=action,
                            symbol=sym,
                            archetype=arch_disp,
                            execution_strategy=str(meta.get("strategy", arch_disp)),
                            add_position=True,
                            size_multiplier=float(
                                pos.get("_size_multiplier", 1.0) or 1.0
                            ),
                            execution_profile={
                                "add_position": meta["add_position"],
                            },
                        )
                        ladder_bm = self._bm_map.get(
                            arch_disp, self._primary_bar_minutes
                        )
                        added_fl = simulator.try_add_position(
                            ladder_intent,
                            entry_bar_primary,
                            pf,
                            executor=_executor,
                            runtime_state=_runtime_state,
                            bar_minutes=ladder_bm,
                            skip_signal_trigger=True,
                        )
                        _er_rows_float_ladder.append(
                            {
                                "pct": _extract_path_efficiency_pct(pf),
                                "outcome": (
                                    "ok"
                                    if added_fl
                                    else str(
                                        getattr(
                                            simulator,
                                            "last_add_reject_reason",
                                            "",
                                        )
                                    )
                                    or "other"
                                ),
                            }
                        )
                        if added_fl:
                            pos["_last_float_ladder_add_ts"] = ts
                            _add_pos_count += 1
                            funnel.setdefault("add_position_ok", 0)
                            funnel["add_position_ok"] = (
                                int(funnel.get("add_position_ok", 0)) + 1
                            )
                            funnel.setdefault("float_ladder_add_ok", 0)
                            funnel["float_ladder_add_ok"] = (
                                int(funnel.get("float_ladder_add_ok", 0)) + 1
                            )
                            ladder_done = True

            # 更新 _pos_last_ts 确保当前 symbol 也被跟踪
            if sym not in _pos_last_ts or ts > _pos_last_ts[sym]:
                _pos_last_ts[sym] = ts
            prev_ts[sym] = ts

        # ── Phase 4: 处理最后一个信号后的 1min bars + 关闭残留持仓 ──
        for sym, simulator in self._simulators.items():
            data = sym_data[sym]
            bars_1min_test = data["bars_1min_test"]

            # 最后一个信号后的 bars (用 _pos_last_ts 避免重复处理)
            last_update = _pos_last_ts.get(sym)
            if last_update is not None and simulator.has_positions:
                remaining = bars_1min_test[bars_1min_test.index > last_update]
                for bar_ts, bar_row in remaining.iterrows():
                    bar_dict = {
                        "timestamp": bar_ts,
                        "open": float(bar_row.get("open", 0)),
                        "high": float(bar_row.get("high", 0)),
                        "low": float(bar_row.get("low", 0)),
                        "close": float(bar_row.get("close", 0)),
                    }
                    closed = simulator.update(bar_dict)
                    for ct in closed:
                        self.pcm.notify_position_closed(sym, ct.archetype)
                        pnl_usd = _initial_cash * _risk_per_slot * ct.pnl_r
                        _equity += pnl_usd
                        _equity = max(_equity, 0.0)
                        _equity_curve.append(_equity)
                        if _equity > _equity_peak:
                            _equity_peak = _equity

            # 关闭残留持仓
            if simulator.has_positions:
                if force_close_end:
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
                else:
                    logger.info(
                        "%s: keep %d open positions for next run",
                        sym,
                        simulator.position_count,
                    )

            if simulator.has_positions:
                for row in simulator.snapshot_open_positions():
                    result.open_positions_end.append(
                        {
                            "symbol": sym,
                            "pid": row.get("pid"),
                            "position": row.get("position", {}),
                        }
                    )

            sym_trades = simulator.closed_trades
            result.trades.extend(sym_trades)
            result.per_symbol[sym] = sym_trades
            result.bars_1min[sym] = data["bars_1min_test"]
            logger.info(f"  {sym}: {len(sym_trades)} trades")

        result.trades.sort(key=lambda t: t.entry_time)
        result.funnel = dict(funnel)
        result.funnel_per_bar = _funnel_per_bar_rows

        # 保存 equity curve 和 kill switch 统计
        result.equity_curve = _equity_curve
        if _executor and _executor.cfg.kill_enabled:
            result.kill_switch_stats = {
                "trigger_count": len(_ks_triggers),
                "trades_skipped": _ks_skipped,
                "trades_executed": _ks_executed,
                "triggers": _ks_triggers,
            }

        # 保存加仓统计
        if _add_pos_enabled:
            add_trades = [t for t in result.trades if t.is_add_position]
            add_pnl = [t.pnl_r for t in add_trades]
            add_mult = [t.size_multiplier for t in add_trades]
            max_observed_lev = max(
                (
                    float(sim.max_observed_leverage or 0.0)
                    for sim in self._simulators.values()
                ),
                default=0.0,
            )
            max_observed_notional = max(
                (
                    float(sim.max_observed_notional_frac or 0.0)
                    for sim in self._simulators.values()
                ),
                default=0.0,
            )
            result.add_position_stats = {
                "enabled": True,
                "add_count": _add_pos_count,
                "rejected_count": _add_pos_rejected,
                "add_trades": len(add_trades),
                "add_mean_r": float(np.mean(add_pnl)) if add_pnl else 0.0,
                "add_mean_size": float(np.mean(add_mult)) if add_mult else 0.0,
                "add_win_rate": (
                    float(np.mean([p > 0 for p in add_pnl])) if add_pnl else 0.0
                ),
                "max_observed_leverage": float(max_observed_lev),
                "max_observed_notional_frac": float(max_observed_notional),
                "path_efficiency_pct_at_add": {
                    "note": (
                        "path_efficiency_pct = path_efficiency_pct_f 输出的历史分位 [0,1] "
                        "(净位移/路径长度，因果 shift)；用于对照 ER-gated 加仓门槛"
                    ),
                    "signal_add_attempts": _er_pct_attempt_stats(_er_rows_signal_add),
                    "float_ladder_attempts": _er_pct_attempt_stats(
                        _er_rows_float_ladder
                    ),
                },
            }

        return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. Trading Map Generator
# ═════════════════════════════════════════════════════════════════════════════


def _resample_bars(bars_1min: pd.DataFrame, freq: str = "4h") -> pd.DataFrame:
    """1min bars → 指定 timeframe OHLCV"""
    ohlc = (
        bars_1min.resample(freq)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    if "volume" in bars_1min.columns:
        ohlc["volume"] = bars_1min["volume"].resample(freq).sum()
    return ohlc


def _rolling_tp_vwap(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Rolling typical-price VWAP: sum(tp*vol)/sum(vol) over ``window`` bars."""
    h = ohlc["high"].astype(float)
    l = ohlc["low"].astype(float)
    c = ohlc["close"].astype(float)
    tp = (h + l + c) / 3.0
    if "volume" in ohlc.columns:
        vol = ohlc["volume"].astype(float).clip(lower=0.0)
    else:
        vol = pd.Series(1.0, index=ohlc.index)
    n = len(ohlc)
    w = max(2, min(int(window), n))
    min_p = max(3, min(w // 10, max(50, w // 20)))
    num = (tp * vol).rolling(window=w, min_periods=min_p).sum()
    den = vol.rolling(window=w, min_periods=min_p).sum()
    out = num / den.replace(0, np.nan)
    return out


def _merge_1min_for_chart(
    sym: str,
    bars_1min: pd.DataFrame,
    data_path: Optional[str],
    extra_months: int,
) -> pd.DataFrame:
    """1m OHLCV from (test_start − extra_months) through test_end; merge with backtest bars."""
    if bars_1min is None or bars_1min.empty:
        return bars_1min
    b = bars_1min.copy()
    b.index = pd.to_datetime(b.index, utc=True)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in b.columns]
    if not cols or "close" not in cols:
        return bars_1min
    b = b[cols]
    if not data_path or int(extra_months) <= 0:
        return b
    dp = Path(data_path)
    if not dp.is_dir():
        return b
    idx_min = b.index.min()
    idx_max = b.index.max()
    start_ts = idx_min - pd.DateOffset(months=int(extra_months))
    start_s = start_ts.strftime("%Y-%m-%d")
    end_day = pd.Timestamp(idx_max)
    if getattr(end_day, "tz", None) is not None:
        end_day = end_day.tz_convert("UTC")
    end_s = (end_day.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        dh = DataHandler(str(dp))
        ext = dh.load_ohlcv(
            symbol=sym, timeframe="1T", start_date=start_s, end_date=end_s
        )
    except Exception as e:
        logger.warning("trading map: extended 1m load failed %s: %s", sym, e)
        return b
    if ext is None or ext.empty:
        return b
    ext = ext.sort_index()
    ext.index = pd.to_datetime(ext.index, utc=True)
    ec = [c for c in ("open", "high", "low", "close", "volume") if c in ext.columns]
    if not ec or "close" not in ec:
        return b
    ext = ext[ec]
    merged = pd.concat([ext, b]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def generate_trading_map_html(
    result: BacktestResult,
    output_path: str,
    bar_freq: str = "4h",
    compare_trades_csv: Optional[str] = None,
    *,
    data_path: Optional[str] = None,
    map_extra_months: int = 12,
    map_vwap_window_bars: int = 1200,
    map_long_ema_span: int = 1200,
) -> str:
    """生成 K线 + 交易标记 HTML 交易地图 (多策略分 Tab)。

    可选从 ``data_path`` 向前多取 ``map_extra_months`` 月 1m 数据，仅在 **计算** VWAP 时使用；
    **图上 X 轴只覆盖本次回测窗口**（``result.bars_1min`` 对应区间），不把向前扩展的历史整段画出来。
    主图价格轴叠画滚动典型价 VWAP（``map_long_ema_span`` 仅保留 CLI/API 兼容，不再绘制 EMA）。

    可视化规则:
      入场填充色   = 方向 (多=绿 / 空=红), 描边色 = 策略 (BPC=蓝 / FER=紫 / ME=橙)
      标记形状     = 方向+加仓 (△=多头, ▽=空头, ◇=加仓多, ◈=加仓空)
      连接线颜色   = 盈亏 (绿=盈利, 红=亏损)
      出场标记颜色 = 盈亏 (绿=盈利, 红=亏损)
    面板布局:
      Tabs: All | BPC | FER | ME (每个 Tab 内按 symbol 纵向堆叠)
    """
    if not BOKEH_AVAILABLE:
        logger.warning("❌ Bokeh 未安装, 无法生成交易地图. pip install bokeh")
        return ""

    symbols = sorted(set(result.per_symbol.keys()) | set(result.bars_1min.keys()))
    if not symbols:
        logger.warning("❌ 无 symbol（无成交且无 bars_1min）, 无法生成交易地图")
        return ""

    # ── 颜色方案 ──
    _STRAT_COLORS: dict = {
        "bpc": "#3274D9",  # 蓝
        "fer": "#B877D9",  # 紫
        "me": "#FF9830",  # 橙
        "me-long": "#FF9830",  # 橙（别名）
        "lv": "#73BF69",  # 绿
    }
    _STRAT_COLOR_DEFAULT = "#aaaaaa"
    _COLOR_WIN = "#26a69a"  # 盈利 绿
    _COLOR_LOSS = "#ef5350"  # 亏损 红
    _COLOR_UP = "#26a69a"  # K线 阳
    _COLOR_DOWN = "#ef5350"  # K线 阴
    # 入场填充: 多空分离 (与 K 线涨跌绿红区分略提高饱和度)
    _ENTRY_FILL_LONG = "#2e7d32"
    _ENTRY_FILL_SHORT = "#c62828"

    # ── 标记映射 ──
    _MARKER_MAP = {
        ("LONG", False): "triangle",  # △ 多头入场
        ("SHORT", False): "inverted_triangle",  # ▽ 空头入场
        ("LONG", True): "diamond",  # ◇ 加仓多
        ("SHORT", True): "diamond_cross",  # ◈ 加仓空
    }

    def _entry_fill_for_side(side: str) -> str:
        return _ENTRY_FILL_LONG if str(side).upper() == "LONG" else _ENTRY_FILL_SHORT

    _FREQ_MS = {
        "15min": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }
    bar_w = _FREQ_MS.get(bar_freq, 4 * 60 * 60 * 1000) * 0.6

    def _arch_family(name: str) -> str:
        s = str(name or "").lower().strip()
        return s.split("-")[0] if s else ""

    def _strat_color(archetype: str) -> str:
        key = str(archetype).lower()
        if key in _STRAT_COLORS:
            return _STRAT_COLORS[key]
        fam = _arch_family(key)
        return _STRAT_COLORS.get(fam, _STRAT_COLOR_DEFAULT)

    # ── 所有出现的 archetype ──
    all_archetypes = sorted(set(t.archetype for t in result.trades if t.archetype))

    # ── 构建单个 symbol K线图 ──
    # ── 加载对比交易（向量回测等）──
    _cmp_by_sym: dict = {}
    if compare_trades_csv and Path(compare_trades_csv).exists():
        try:
            _cdf = pd.read_csv(compare_trades_csv)
            for col in ["entry_time", "exit_time"]:
                if col in _cdf.columns:
                    _cdf[col] = pd.to_datetime(_cdf[col], utc=True)
            for _sym, _grp in _cdf.groupby("symbol"):
                _cmp_by_sym[_sym] = _grp.to_dict("records")
            logger.info(
                f"  🔵 对比交易已加载: {len(_cdf)} 笔 from {compare_trades_csv}"
            )
        except Exception as _e:
            logger.warning(f"  ⚠️  对比交易加载失败: {_e}")

    def _funnel_row_matches(
        row: Mapping[str, Any], sf: "str | None", fam: bool
    ) -> bool:
        if sf is None:
            return True
        s = str(row.get("strategy") or "").lower()
        if fam:
            return _arch_family(s) == str(sf).lower()
        return s == str(sf).lower()

    def _build_funnel_figures_for_sym(
        sym: str,
        *,
        strat_filter: "str | None",
        family_mode: bool,
        x_range,
        ref_index: Optional[pd.DatetimeIndex],
    ) -> list:
        rows_all = getattr(result, "funnel_per_bar", None) or []
        rows = [
            r
            for r in rows_all
            if str(r.get("symbol") or "") == sym
            and _funnel_row_matches(r, strat_filter, family_mode)
        ]
        if not rows:
            return []

        def _pcm_y(rec: Mapping[str, Any]) -> float:
            if rec.get("pcm_direction_filter") is False:
                return 0.0
            return 1.0

        def _bool_y(rec: Mapping[str, Any], key: str) -> float:
            v = rec.get(key)
            if v is None:
                return float("nan")
            return 1.0 if v else 0.0

        def _dir_y(rec: Mapping[str, Any]) -> float:
            dv = rec.get("direction_value")
            if dv is None:
                return float("nan")
            try:
                dvi = int(dv)
            except (TypeError, ValueError):
                return float("nan")
            return {-1: 0.0, 0: 0.5, 1: 1.0}.get(dvi, float("nan"))

        def _step_xy(ts: list, vals: list):
            if not ts:
                return [], []
            xs: list = []
            ys: list = []
            for i in range(len(ts)):
                if i > 0:
                    xs.append(ts[i])
                    ys.append(vals[i - 1])
                xs.append(ts[i])
                ys.append(vals[i])
            return xs, ys

        by_strat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_strat[str(r.get("strategy") or "unknown")].append(dict(r))

        figs: list = []
        _STAGES = [
            ("PCM EMA", _pcm_y, "#64748b"),
            ("Prefilter", lambda rec: _bool_y(rec, "prefilter"), "#3274D9"),
            ("Gate", lambda rec: _bool_y(rec, "gate"), "#7c3aed"),
            ("Entry filter", lambda rec: _bool_y(rec, "entry_filter"), "#ca8a04"),
            ("Direction (−1/0/+1)", _dir_y, "#059669"),
        ]

        for strat_name in sorted(by_strat.keys()):
            sub = sorted(by_strat[strat_name], key=lambda t: t["timestamp"])
            ts = [pd.Timestamp(t["timestamp"]) for t in sub]
            if ref_index is not None and getattr(ref_index, "tz", None) is not None:
                _tz = ref_index.tz
                ts = [
                    (
                        x.tz_convert(_tz)
                        if x.tzinfo
                        else x.tz_localize("UTC").tz_convert(_tz)
                    )
                    for x in ts
                ]
            pf = bk_figure(
                title=f"{sym} · {strat_name} — gate / prefilter / direction",
                x_axis_type="datetime",
                width=1400,
                height=200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_range=x_range,
                y_range=(-0.15, 4.65),
            )
            pf.yaxis.ticker = FixedTicker(ticks=[0, 1, 2, 3, 4])
            pf.yaxis.major_label_overrides = {
                0: "PCM",
                1: "Prefilter",
                2: "Gate",
                3: "EntryFlt",
                4: "Dir",
            }
            pf.grid.grid_line_alpha = 0.25

            for bi, (label, fn, color) in enumerate(_STAGES):
                vals = [float(fn(rec)) for rec in sub]
                xs, ys = _step_xy(ts, [bi + 0.35 * v for v in vals])
                if xs:
                    pf.line(
                        xs, ys, line_color=color, line_width=1.6, legend_label=label
                    )
            pf.legend.click_policy = "hide"
            pf.legend.label_text_font_size = "8pt"
            pf.legend.location = "top_left"
            pf.add_tools(
                HoverTool(
                    tooltips=[("Time", "@x{%F %H:%M}"), ("y", "@y{0.2f}")],
                    formatters={"@x": "datetime"},
                    mode="mouse",
                )
            )
            figs.append(pf)
        return figs

    def _build_symbol_figure(
        sym: str,
        trades: list,
        cmp_trades: list = None,
        *,
        strat_filter: "str | None" = None,
        family_mode: bool = False,
    ) -> object:
        bars_1min = result.bars_1min.get(sym)
        if bars_1min is None or bars_1min.empty:
            return None
        bars_full = _merge_1min_for_chart(sym, bars_1min, data_path, map_extra_months)
        df = _resample_bars(bars_full, freq=bar_freq)
        if df.empty:
            return None

        # 全量 df 上算指标（含 map_extra_months 向前扩展），图上只画回测窗
        vw_n = int(map_vwap_window_bars)
        _ = map_long_ema_span  # CLI compat; EMA overlay removed
        vwap_price = _rolling_tp_vwap(df, vw_n)

        view_start = pd.Timestamp(bars_1min.index.min())
        view_end = pd.Timestamp(bars_1min.index.max())
        idx = df.index
        if idx.tz is not None:
            if view_start.tzinfo is None:
                view_start = view_start.tz_localize("UTC").tz_convert(idx.tz)
            else:
                view_start = view_start.tz_convert(idx.tz)
            if view_end.tzinfo is None:
                view_end = view_end.tz_localize("UTC").tz_convert(idx.tz)
            else:
                view_end = view_end.tz_convert(idx.tz)
        plot_mask = (idx >= view_start) & (idx <= view_end)
        df_plot = df.loc[plot_mask]
        if df_plot.empty:
            df_plot = df
            logger.warning(
                "trading map %s: plot window empty after tz align, using full merged range",
                sym,
            )

        cmp_trades = cmp_trades or []
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

        try:
            p.x_range.start = df_plot.index.min()
            p.x_range.end = df_plot.index.max()
            p.x_range.range_padding = 0.02
        except Exception:
            pass

        inc = df_plot.close >= df_plot.open
        dec = ~inc
        p.segment(
            df_plot.index[inc],
            df_plot.high[inc],
            df_plot.index[inc],
            df_plot.low[inc],
            color=_COLOR_UP,
            line_width=1,
        )
        p.segment(
            df_plot.index[dec],
            df_plot.high[dec],
            df_plot.index[dec],
            df_plot.low[dec],
            color=_COLOR_DOWN,
            line_width=1,
        )
        p.vbar(
            df_plot.index[inc],
            bar_w,
            df_plot.open[inc],
            df_plot.close[inc],
            fill_color=_COLOR_UP,
            line_color=_COLOR_UP,
            fill_alpha=0.8,
        )
        p.vbar(
            df_plot.index[dec],
            bar_w,
            df_plot.open[dec],
            df_plot.close[dec],
            fill_color=_COLOR_DOWN,
            line_color=_COLOR_DOWN,
            fill_alpha=0.8,
        )

        vp = vwap_price.reindex(df_plot.index)
        p.line(
            df_plot.index,
            vp,
            line_color="#c026d3",
            line_width=1.35,
            line_alpha=0.78,
            legend_label=f"Rolling TP-VWAP ({vw_n} bars, price)",
        )

        # ── CRF: rolling 120 lo/hi band (same bar as prefilter; no min/max merge rects)
        _draw_box = strat_filter is None or str(strat_filter).strip().lower() == "crf"
        if _draw_box:
            try:
                import numpy as _np
                from src.features.time_series.box_structure_features import (
                    compute_box_structure_from_series as _box_feat,
                )

                _bx = _box_feat(
                    close=df_plot["close"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                )
            except Exception:
                _bx = None
            if _bx is not None and not _bx.empty:
                _bx = _bx.reindex(df_plot.index)
                _stab = pd.to_numeric(_bx.get("box_stability_120"), errors="coerce")
                _widp = pd.to_numeric(_bx.get("box_width_pct_120"), errors="coerce")
                _hi = pd.to_numeric(_bx.get("box_hi_120"), errors="coerce")
                _lo = pd.to_numeric(_bx.get("box_lo_120"), errors="coerce")
                # Keep in sync with config/strategies/crf/archetypes/prefilter.yaml:
                #   stab>=0.80, 0.04 <= width <= 0.25
                _qual = (
                    (_stab.fillna(0.0) >= 0.80)
                    & (_widp.fillna(0.0) >= 0.04)
                    & (_widp.fillna(1.0) <= 0.25)
                )
                _pass_rate = float(_qual.mean()) if len(_qual) else 0.0
                print(f"   CRF prefilter overlay: pass_rate={_pass_rate:.1%}")
                qidx = df_plot.index[_qual.values]
                if len(qidx) > 0:
                    # Per-bar vbar — true gaps where filter fails, no NaN bridging.
                    p.vbar(
                        x=qidx,
                        width=bar_w,
                        top=_hi.values[_qual.values],
                        bottom=_lo.values[_qual.values],
                        fill_color="#22c55e",
                        fill_alpha=0.18,
                        line_color=None,
                        legend_label="CRF: box (prefilter pass)",
                    )

        if trades:
            # ── 连接线: 颜色 = win/loss ──
            for wl, lc, emoji in [
                ("win", _COLOR_WIN, "📈"),
                ("loss", _COLOR_LOSS, "📉"),
            ]:
                batch = [t for t in trades if ("win" if t.pnl_r > 0 else "loss") == wl]
                if batch:
                    p.multi_line(
                        xs=[[t.entry_time, t.exit_time] for t in batch],
                        ys=[[t.entry_price, t.exit_price] for t in batch],
                        line_color=lc,
                        line_dash="dashed",
                        line_alpha=0.4,
                        line_width=1.5,
                        legend_label=f"{emoji} {wl}",
                    )

            # ── 入场标记: 颜色 = 策略, 形状 = side+is_add ──
            # group by (archetype, side, is_add)
            entry_groups: dict = {}
            for t in trades:
                key = (str(t.archetype).lower(), t.side.upper(), t.is_add_position)
                entry_groups.setdefault(key, []).append(t)

            for (arch, side, is_add), batch in sorted(entry_groups.items()):
                strat_line = _strat_color(arch)
                fill_c = _entry_fill_for_side(side)
                marker = _MARKER_MAP.get((side, is_add), "circle")
                sz = 13 if is_add else 11
                add_txt = "Add " if is_add else ""
                leg_side = "Long" if str(side).upper() == "LONG" else "Short"
                # 图例只写字，形状由 glyph 表达，避免与 Unicode 符号重复叠字
                legend_lbl = f"{arch.upper()} {add_txt}{leg_side}"
                p.scatter(
                    x=[t.entry_time for t in batch],
                    y=[t.entry_price for t in batch],
                    marker=marker,
                    size=sz,
                    fill_color=fill_c,
                    line_color=strat_line,
                    line_width=2,
                    fill_alpha=0.88,
                    legend_label=legend_lbl,
                )

            # ── 出场标记: 颜色 = win/loss ──
            for wl, ec in [("win", _COLOR_WIN), ("loss", _COLOR_LOSS)]:
                batch = [t for t in trades if ("win" if t.pnl_r > 0 else "loss") == wl]
                if batch:
                    p.scatter(
                        x=[t.exit_time for t in batch],
                        y=[t.exit_price for t in batch],
                        marker="square",
                        size=8,
                        color=ec,
                        alpha=0.6,
                        legend_label=f"□ exit_{wl}",
                    )

        # ── 对比交易标记（向量回测）── 蓝色圆圈
        if cmp_trades:
            _cmp_xs = [r["entry_time"] for r in cmp_trades]
            _cmp_ys = [r["entry_price"] for r in cmp_trades]
            p.scatter(
                x=_cmp_xs,
                y=_cmp_ys,
                marker="circle",
                size=10,
                color="#00b4d8",
                alpha=0.7,
                line_color="#0077b6",
                line_width=1.5,
                legend_label="◉ Vector BT entry",
            )

        p.add_tools(
            HoverTool(
                tooltips=[("Time", "@x{%F %H:%M}"), ("Price", "@y{0.2f}")],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
        )
        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        p.legend.label_text_font_size = "9pt"

        funnel_figs = _build_funnel_figures_for_sym(
            sym,
            strat_filter=strat_filter,
            family_mode=family_mode,
            x_range=p.x_range,
            ref_index=df_plot.index,
        )
        if funnel_figs:
            return bk_column(p, *funnel_figs, sizing_mode="stretch_width")
        return p

    # ── 构建单个 Tab ──
    def _build_tab(
        tab_label: str,
        strat_filter: "str | None",
        *,
        family_mode: bool = False,
    ) -> object:
        def _match_trade(t: ClosedTrade) -> bool:
            if strat_filter is None:
                return True
            arch = str(t.archetype).lower()
            if family_mode:
                return _arch_family(arch) == str(strat_filter).lower()
            return arch == str(strat_filter).lower()

        tab_trades = [t for t in result.trades if _match_trade(t)]
        n = len(tab_trades)
        wr = sum(1 for t in tab_trades if t.pnl_r > 0) / n if n else 0
        total = sum(t.pnl_r for t in tab_trades)
        strat_c = _strat_color(strat_filter) if strat_filter else "#888888"

        cmp_n = sum(len(v) for v in _cmp_by_sym.values())
        cmp_label = f" | 🔵 Vector={cmp_n}" if cmp_n else ""
        title_html = (
            f"<h2 style='color:{strat_c}'>🗺️ {tab_label} "
            f"| {n} trades | WR={wr:.1%} | Total={total:.2f}R{cmp_label}</h2>"
        )
        figs: list = [Div(text=title_html)]

        for sym in symbols:
            _bars = result.bars_1min.get(sym)
            if _bars is None or getattr(_bars, "empty", True):
                continue
            sym_trades = result.per_symbol.get(sym, [])
            if strat_filter is not None:
                sym_trades = [t for t in sym_trades if _match_trade(t)]
            fig = _build_symbol_figure(
                sym,
                sym_trades,
                cmp_trades=_cmp_by_sym.get(sym),
                strat_filter=strat_filter,
                family_mode=family_mode,
            )
            if fig is not None:
                figs.append(fig)

        child = bk_column(*figs, sizing_mode="stretch_width")
        return TabPanel(child=child, title=tab_label)

    # ── 组装 Tabs (All + 家族聚合 + 各策略) ──
    tabs_list = [_build_tab("All", None)]
    families = sorted({_arch_family(a) for a in all_archetypes if _arch_family(a)})
    for fam in families:
        tabs_list.append(_build_tab(fam.upper(), fam, family_mode=True))
    for arch in all_archetypes:
        tabs_list.append(_build_tab(arch.upper(), arch))

    layout = Tabs(tabs=tabs_list)
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
        help="策略名, 逗号分隔 (例: bpc / fer / bpc,fer,me-long)",
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
        help="回测天数 (默认 180, 被 --start-date/--end-date 覆盖)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="回测开始日期 (YYYY-MM-DD), 覆盖 --days",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="回测结束日期 (YYYY-MM-DD), 默认 now()",
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
    parser.add_argument(
        "--map-extra-months",
        type=int,
        default=12,
        help=(
            "交易地图: 从 --data-path 向前多加载 1m 月数，仅用于 VWAP 计算；"
            "K 线横轴仍为回测窗。0=不向前扩展"
        ),
    )
    parser.add_argument(
        "--map-vwap-window-bars",
        type=int,
        default=1200,
        help="交易地图: 滚动典型价 VWAP 窗口 (按地图 K 线根数)",
    )
    parser.add_argument(
        "--map-long-ema-span",
        type=int,
        default=1200,
        help="交易地图: 已废弃（不再绘制 EMA），保留参数以兼容旧脚本",
    )
    parser.add_argument(
        "--compare-trades",
        default=None,
        help="对比交易 CSV 路径 (向量回测导出的 trades.csv), 将以蓝圆标记叠加显示",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0004,
        help="单边手续费率 (默认 0.0004 = 0.04%%%% Binance taker, 设 0 关闭)",
    )
    parser.add_argument(
        "--universe-group",
        default=None,
        help=(
            "从 universe_groups yaml 读取 symbols，格式: universe_set/group，"
            "例: starter_a/highcap，默认文件: config/download/crypto_4h_token_universe_groups.yaml"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="兼容参数: 当前仍使用 1min 精确持仓更新（用于保证与非 fast 一致）",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="可选: 从 JSON 恢复上期未平仓状态",
    )
    parser.add_argument(
        "--dump-end-state",
        default=None,
        help="可选: 导出本期结束时未平仓状态 JSON",
    )
    parser.add_argument(
        "--keep-open-positions",
        action="store_true",
        default=False,
        help="不在回测结束时强平，保留未平仓用于下一期续跑",
    )
    parser.add_argument(
        "--no-kill-switch",
        action="store_true",
        default=False,
        help="禁用 constitution kill switch（用于诊断策略真实表现，不受亏损限额约束）",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategy.split(",")]

    # 解析 symbols：--universe-group 优先，其次 --symbols
    if args.universe_group:
        import yaml as _yaml

        _ug_file = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "download"
            / "crypto_4h_token_universe_groups.yaml"
        )
        _ug_data = _yaml.safe_load(_ug_file.read_text(encoding="utf-8"))
        _parts = args.universe_group.split("/")
        if len(_parts) != 2:
            parser.error(
                "--universe-group 格式应为 universe_set/group，例: starter_a/highcap"
            )
        _universe_set, _group = _parts
        _tokens = _ug_data["universe_sets"][_universe_set]["groups"][_group]
        _quote = _ug_data.get("quote", "USDT")
        symbols = [f"{t}{_quote}" for t in _tokens]
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 72)
    print("  🔬 事件驱动回测 (多策略 PCM 仲裁)")
    print("=" * 72)
    print(f"  策略:    {', '.join(strategies)}")
    print(f"  Symbols: {symbols}")
    print(f"  天数:    {args.days}")
    fee_pct = args.fee_rate * 100
    print(
        f"  手续费:  {fee_pct:.2f}% 单边 ({fee_pct*2:.2f}% 双边)"
        if args.fee_rate > 0
        else "  手续费:  关闭"
    )
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
    if args.resume_state:
        print(f"  恢复状态: {args.resume_state}")
    if args.dump_end_state:
        print(f"  导出状态: {args.dump_end_state}")
    print("=" * 72)

    resume_state_obj = None
    if args.resume_state:
        _rp = Path(args.resume_state)
        if _rp.exists():
            resume_state_obj = json.loads(_rp.read_text(encoding="utf-8"))
        else:
            logger.warning("resume state not found: %s", _rp)

    bt = EventBacktester(
        strategies=strategies,
        live_root=args.live_root,
        strategies_root=args.strategies_root,
        db_path=args.db,
        data_path=args.data_path,
        fee_rate=args.fee_rate,
    )

    result = bt.run(
        symbols=symbols,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        fast_mode=args.fast,
        resume_state=resume_state_obj,
        force_close_end=not bool(args.keep_open_positions),
        no_kill_switch=args.no_kill_switch,
    )

    result.print_report()

    if args.export:
        result.export_trades_csv(args.export)

    if args.output:
        _save_json(result, args.output)

    if args.dump_end_state:
        state_obj = {
            "strategy": ",".join(strategies),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "open_positions_count": len(result.open_positions_end),
            "symbols": {},
        }
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in result.open_positions_end:
            sym = str(row.get("symbol", ""))
            if not sym:
                continue
            by_symbol[sym].append(
                {"pid": row.get("pid"), "position": row.get("position", {})}
            )
        for sym, rows in by_symbol.items():
            state_obj["symbols"][sym] = {
                "open_positions_count": len(rows),
                "open_positions": rows,
            }
        _dst = Path(args.dump_end_state)
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_text(
            json.dumps(state_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  ♻️ End state saved → {_dst}")

    if args.trading_map:
        # 根据策略 timeframe 选择 K 线频率
        tf_to_freq = {"15T": "15min", "60T": "1h", "120T": "2h", "240T": "4h"}
        _sr_map = args.strategies_root or "config/strategies"
        primary_tf = _get_timeframe(strategies[0], strategies_root=_sr_map)
        map_freq = tf_to_freq.get(primary_tf, "4h")
        generate_trading_map_html(
            result,
            args.trading_map,
            bar_freq=map_freq,
            compare_trades_csv=getattr(args, "compare_trades", None),
            data_path=args.data_path,
            map_extra_months=int(getattr(args, "map_extra_months", 12)),
            map_vwap_window_bars=int(getattr(args, "map_vwap_window_bars", 1200)),
            map_long_ema_span=int(getattr(args, "map_long_ema_span", 1200)),
        )

    if args.db:
        print(f"\n  💾 订单数据已保存 → {args.db}")

    result.print_path_efficiency_footer()
    _anchor = args.output or args.export or args.trading_map
    _save_path_efficiency_sidecar(result, _anchor)

    return 0


def _trade_to_dict(t: ClosedTrade) -> dict:
    """ClosedTrade → JSON-safe dict"""
    return {
        "symbol": t.symbol,
        "side": t.side,
        "archetype": t.archetype,
        "entry_price": round(t.entry_price, 6),
        "exit_price": round(t.exit_price, 6),
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat(),
        "pnl_r": round(t.pnl_r, 4),
        "exit_reason": t.exit_reason,
        "bars_held": t.bars_held,
        "evidence_score": round(t.evidence_score, 4),
        "size_multiplier": round(t.size_multiplier, 4),
        "is_add_position": t.is_add_position,
        "is_reverse": t.is_reverse,
        "atr_stop_pct": round(t.atr_stop_pct, 6),
        "effective_stop_pct": round(t.effective_stop_pct, 6),
        "sizing_stop_source": t.sizing_stop_source,
        "breakeven_locked_at_exit": t.breakeven_locked_at_exit,
    }


def _save_json(result: BacktestResult, path: str):
    """保存 JSON 结果 (含完整交易列表)"""
    wins = [t for t in result.trades if t.pnl_r > 0]
    tail_rate, tail_n, winner_n = _tail_contribution_rate(result.trades)
    per_arch: dict = {}
    for t in result.trades:
        a = t.archetype or "unknown"
        per_arch.setdefault(a, [])
        per_arch[a].append(t)

    out = {
        "strategy": result.strategy,
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 4),
        "sharpe_r": round(result.sharpe, 4),
        "mean_r": round(result.mean_r, 4),
        "total_r": round(result.total_r, 4),
        "max_drawdown_r": round(result.max_drawdown_r, 4),
        "tail_contribution_rate": round(tail_rate, 4),
        "tail_trade_count": tail_n,
        "winner_count": winner_n,
        "funnel": result.funnel,
        "per_archetype": {
            arch: {
                "n_trades": len(ts),
                "win_rate": (
                    round(sum(1 for t in ts if t.pnl_r > 0) / len(ts), 4) if ts else 0
                ),
                "mean_r": round(sum(t.pnl_r for t in ts) / len(ts), 4) if ts else 0,
                "total_r": round(sum(t.pnl_r for t in ts), 4),
            }
            for arch, ts in per_arch.items()
        },
        "per_symbol": {
            sym: {
                "n_trades": len(trades),
                "win_rate": (
                    round(sum(1 for t in trades if t.pnl_r > 0) / len(trades), 4)
                    if trades
                    else 0
                ),
                "mean_r": (
                    round(sum(t.pnl_r for t in trades) / len(trades), 4)
                    if trades
                    else 0
                ),
                "total_r": round(sum(t.pnl_r for t in trades), 4),
                "per_archetype": {
                    arch: sum(1 for t in trades if t.archetype == arch)
                    for arch in sorted(set(t.archetype for t in trades))
                },
            }
            for sym, trades in result.per_symbol.items()
        },
        "add_position_stats": result.add_position_stats or {},
        "add_trigger_types": result.add_trigger_types,
        "open_positions_end": result.open_positions_end,
        "funnel_per_bar": _json_safe(result.funnel_per_bar or []),
        "trades": [_trade_to_dict(t) for t in result.trades],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Results saved → {path}")


def _save_path_efficiency_sidecar(
    result: BacktestResult, anchor_path: Optional[str]
) -> None:
    """与 event_backtest JSON 同目录写入 path_efficiency 分布，供后续分析。"""
    ap = result.add_position_stats
    if not isinstance(ap, dict):
        return
    pe = ap.get("path_efficiency_pct_at_add")
    if not isinstance(pe, dict):
        return
    if not anchor_path:
        return
    slug = (result.strategy or "multi").replace("+", "_").replace(",", "_")
    out_path = Path(anchor_path).with_name(f"path_efficiency_pct_at_add_{slug}.json")
    payload = {
        "strategy": result.strategy,
        "path_efficiency_pct_at_add": _json_safe(pe),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  📄 path_efficiency 分布 → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
