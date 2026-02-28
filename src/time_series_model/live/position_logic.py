"""
持仓管理共享逻辑 — 实盘 + 事件回测公用

两个核心函数:
  1. build_position_dict: 从 TradeIntent 构建持仓字典 (不含下单)
  2. enforce_position: 7步持仓管理 (不含下单)

使用方式:
  - 实盘 (order_flow_listener):
      pos = build_position_dict(intent, entry_price, atr, bar_minutes)
      reason, exit_price = enforce_position(pos, price, price, price, now)
  - 回测 (event_backtest):
      pos = build_position_dict(intent, entry_price, atr, bar_minutes)
      reason, exit_price = enforce_position(pos, bar_high, bar_low, bar_close, now)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from src.time_series_model.live.execution_profile_apply import (
    compute_rr_prices,
    holding_expired,
    pick_atr,
)


def build_position_dict(
    intent: Any,
    entry_price: float,
    atr: float,
    bar_minutes: int = 240,
    entry_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """从 TradeIntent 构建持仓字典 — 实盘/回测公用

    不执行任何下单操作, 只生成持仓数据结构.

    Args:
        intent: TradeIntent (需有 action, symbol, execution_profile, confidence)
        entry_price: 入场价
        atr: 当时的 ATR
        bar_minutes: 信号时钟分钟数 (如 240 = 4h, 60 = 1h)
        entry_time: 入场时间, None 则用 now()

    Returns:
        持仓字典, 可直接存入 _open_positions
    """
    exec_profile = intent.execution_profile or {}
    rr_constraints = exec_profile.get("rr_constraints") or {}
    strategy_specific = exec_profile.get("strategy_specific") or {}

    side = str(intent.action).upper()
    is_long = side in {"LONG", "BUY"}

    stop_loss_r = float(rr_constraints.get("stop_loss_r", 0.0) or 0.0)
    take_profit_r = float(rr_constraints.get("take_profit_r", 0.0) or 0.0)
    max_holding_bars = rr_constraints.get("max_holding_bars", 50)

    sl_price, tp_price = None, None
    # SL 和 TP 独立计算 —— 不要求两者同时 > 0
    if entry_price > 0 and atr > 0 and stop_loss_r > 0:
        # 始终计算 SL价格
        computed_sl, computed_tp = compute_rr_prices(
            side=side,
            entry_price=entry_price,
            atr=atr,
            stop_loss_r=stop_loss_r,
            take_profit_r=take_profit_r if take_profit_r > 0 else stop_loss_r,  # dummy
        )
        sl_price = computed_sl
        # TP 仅在启用时设置
        tp_price = computed_tp if take_profit_r > 0 else None

    if entry_time is None:
        entry_time = datetime.now(timezone.utc)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    pos: Dict[str, Any] = {
        "symbol": intent.symbol,
        "side": side,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "stop_loss_price": sl_price,
        "take_profit_price": tp_price,
        "allow_trailing": bool(rr_constraints.get("allow_trailing", False)),
        "trailing_atr": rr_constraints.get("trailing_atr"),
        "max_holding_bars": max_holding_bars,
        "atr_at_entry": atr,
        "initial_risk_distance": (
            stop_loss_r * atr if stop_loss_r > 0 and atr > 0 else atr
        ),
        "tier_name": strategy_specific.get("tier_name", "default"),
        "evidence_score": intent.confidence or 0.0,
        "bar_minutes": bar_minutes,
        "bars_counted": 0,
    }

    # BPC 扩展: activation trailing + breakeven
    bpc_cfg = exec_profile.get("bpc_position_config") or {}
    activation_r = (
        bpc_cfg.get("activation_r")
        or rr_constraints.get("activation_r")
        or rr_constraints.get("trailing_atr")
    )
    trail_r = bpc_cfg.get("trail_r") or rr_constraints.get("trailing_atr")

    if bpc_cfg and activation_r is not None:
        # BPC 专用 trailing
        pos["activation_r"] = float(activation_r)
        pos["trail_r"] = float(trail_r or 1.0)
        pos["trailing_activated"] = False
        pos["high_water_mark"] = entry_price if is_long else None
        pos["low_water_mark"] = entry_price if not is_long else None
        pos["breakeven_enabled"] = bpc_cfg.get("breakeven_enabled", False)
        pos["breakeven_trigger_r"] = float(bpc_cfg.get("breakeven_trigger_r", 1.0))
        pos["breakeven_locked"] = False
        if "bar_minutes" in bpc_cfg:
            pos["bar_minutes"] = bpc_cfg["bar_minutes"]
    elif activation_r is not None:
        # 通用 activation trailing (非 BPC)
        pos["activation_r"] = float(activation_r)
        pos["trail_r"] = float(trail_r or 1.0)
        pos["trailing_activated"] = False
        pos["high_water_mark"] = entry_price if is_long else None
        pos["low_water_mark"] = entry_price if not is_long else None
        pos["breakeven_enabled"] = False
        pos["breakeven_locked"] = False
    elif rr_constraints.get("allow_trailing", False):
        # 通用 trailing (allow_trailing=True 但无 activation_r)
        pos["activation_r"] = float(rr_constraints.get("trailing_atr", 1.0))
        pos["trail_r"] = float(rr_constraints.get("trailing_atr", 1.0))
        pos["trailing_activated"] = False
        pos["high_water_mark"] = entry_price if is_long else None
        pos["low_water_mark"] = entry_price if not is_long else None
        pos["breakeven_enabled"] = False
        pos["breakeven_locked"] = False

    return pos


def enforce_position(
    pos: Dict[str, Any],
    *,
    price_high: float,
    price_low: float,
    price_close: float,
    now: datetime,
    default_bar_minutes: int = 240,
) -> Tuple[Optional[str], float]:
    """7步持仓管理 — 实盘/回测公用

    检查顺序 (同 _enforce_open_positions):
      1. Time stop
      2. Breakeven lock
      3. Update HWM/LWM
      4. Activation trailing
      5. SL hit (保守: SL 优先于 TP)
      6. TP hit
      7. 返回结果

    价格参数:
      - 实盘: price_high = price_low = price_close = current_price
      - 回测: price_high = bar.high, price_low = bar.low, price_close = bar.close

    注意: 此函数会 **就地修改** pos 字典 (breakeven_locked, high_water_mark,
    trailing_activated, stop_loss_price 等), 与实盘行为一致.

    Args:
        pos: 持仓字典 (build_position_dict 产出)
        price_high: 当前最高价
        price_low: 当前最低价
        price_close: 当前收盘价/最新价
        now: 当前时间
        default_bar_minutes: 默认 bar 分钟数 (兜底)

    Returns:
        (close_reason, exit_price):
          - close_reason is None → 持仓继续
          - close_reason 非 None → 应关闭, exit_price 为成交价
    """
    entry_time = pos.get("entry_time")
    if not isinstance(entry_time, datetime):
        entry_time = datetime.now(timezone.utc)

    side_str = str(pos.get("side", "")).upper()
    is_long = side_str in {"LONG", "BUY"}
    entry_price = pos.get("entry_price", 0) or price_close
    pos_atr = pos.get("atr_at_entry", 0) or 0
    bar_minutes = pos.get("bar_minutes", default_bar_minutes)

    close_reason: Optional[str] = None
    exit_price = price_close

    # ── 1. Time stop ──
    if holding_expired(
        entry_time=entry_time,
        now=now,
        max_holding_bars=pos.get("max_holding_bars"),
        bar_minutes=bar_minutes,
    ):
        close_reason = "time_stop"
        exit_price = price_close

    # ── 2. Breakeven lock ──
    if (
        close_reason is None
        and pos.get("breakeven_enabled")
        and not pos.get("breakeven_locked")
        and pos_atr > 0
    ):
        check_price = price_high if is_long else price_low
        if is_long:
            profit_r = (check_price - entry_price) / pos_atr
        else:
            profit_r = (entry_price - check_price) / pos_atr
        if profit_r >= pos.get("breakeven_trigger_r", 1.0):
            pos["breakeven_locked"] = True
            pos["stop_loss_price"] = entry_price

    # ── 3. Update high/low water mark ──
    if is_long and pos.get("high_water_mark") is not None:
        if price_high > pos["high_water_mark"]:
            pos["high_water_mark"] = price_high
    elif not is_long and pos.get("low_water_mark") is not None:
        if price_low < pos["low_water_mark"]:
            pos["low_water_mark"] = price_low

    # ── 4. Activation trailing ──
    if close_reason is None and pos.get("activation_r") is not None and pos_atr > 0:
        check_price = price_high if is_long else price_low
        if is_long:
            profit_r = (check_price - entry_price) / pos_atr
        else:
            profit_r = (entry_price - check_price) / pos_atr

        activation_r = pos["activation_r"]
        trail_r = pos.get("trail_r", 1.0)
        if profit_r >= activation_r:
            if not pos.get("trailing_activated"):
                pos["trailing_activated"] = True
            if is_long:
                hwm = pos.get("high_water_mark", check_price)
                trail_sl = hwm - trail_r * pos_atr
            else:
                lwm = pos.get("low_water_mark", check_price)
                trail_sl = lwm + trail_r * pos_atr
            old_sl = pos.get("stop_loss_price")
            if old_sl is not None:
                if is_long and trail_sl > old_sl:
                    pos["stop_loss_price"] = trail_sl
                elif not is_long and trail_sl < old_sl:
                    pos["stop_loss_price"] = trail_sl
            else:
                pos["stop_loss_price"] = trail_sl

    # ── 5. SL hit (保守: SL 优先于 TP) ──
    if close_reason is None:
        sl = pos.get("stop_loss_price")
        if sl is not None:
            sl = float(sl)
            if is_long and price_low <= sl:
                close_reason = "stop_loss"
                exit_price = sl
            elif not is_long and price_high >= sl:
                close_reason = "stop_loss"
                exit_price = sl

    # ── 6. TP hit ──
    if close_reason is None:
        tp = pos.get("take_profit_price")
        if tp is not None:
            tp = float(tp)
            if is_long and price_high >= tp:
                close_reason = "take_profit"
                exit_price = tp
            elif not is_long and price_low <= tp:
                close_reason = "take_profit"
                exit_price = tp

    return close_reason, exit_price
