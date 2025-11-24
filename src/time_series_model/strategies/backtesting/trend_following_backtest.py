"""
趋势跟踪策略专属回测

回测逻辑：
- 方向：由动量方向决定（ROC(20) > 0 → 做多，ROC(20) < 0 → 做空）
- 入场点：趋势确认（如 ADX > 25 + Hurst > 0.6）
- 止损：动态 ATR 止损（如 2×ATR）或 trendline break
- 止盈：无固定止盈，靠 Rank 下降退出（当 Rank < 0.4，平仓）
- 仓位：position_size ∝ (Rank - 0.5)
- 加仓：✅ 每当 Rank 创新高（>0.95），加仓
- 减仓：Rank 从高位回落至 0.7 以下，减仓 50%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class TrendTrade:
    """趋势跟踪交易记录"""

    entry_idx: int
    entry_price: float
    direction: int  # 1=Long, -1=Short
    stop_loss: float
    initial_rank: float
    current_rank: float
    position_size: float
    add_position_size: float = 0.0
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "rank_exit", "sl", "timeout", "reduce"
    pnl: Optional[float] = None


def backtest_trend_following(
    df: pd.DataFrame,
    predictions: np.ndarray,  # Rank (0~1)
    momentum_direction_col: Optional[str] = None,  # 1=Up, -1=Down
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    roc_col: Optional[str] = None,  # ROC(20) for direction
    atr_window: int = 14,
    stop_loss_r: float = 2.0,  # Dynamic ATR stop loss
    max_holding_bars: int = 200,  # Longer holding period for trends
    min_rank: float = 0.4,  # Exit when rank < this
    base_position_size: float = 0.1,
    add_position_rank: float = 0.95,  # Add position when rank > this
    reduce_position_rank: float = 0.7,  # Reduce position when rank < this
    trading_fee: float = 0.001,
    enable_add_position: bool = True,
    enable_reduce_position: bool = True,
) -> Dict:
    """
    趋势跟踪策略专属回测

    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (Rank 0~1)
        momentum_direction_col: Column indicating momentum direction
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        roc_col: ROC(20) column for direction
        atr_window: ATR window if ATR column doesn't exist
        stop_loss_r: Stop loss in R units
        max_holding_bars: Maximum holding period
        min_rank: Exit when rank < this
        base_position_size: Base position size
        add_position_rank: Add position when rank > this
        reduce_position_rank: Reduce position when rank < this
        trading_fee: Trading fee per trade
        enable_add_position: Whether to enable add position logic
        enable_reduce_position: Whether to enable reduce position logic

    Returns:
        Dictionary with backtest results
    """
    # Ensure ATR exists
    if atr_col not in df.columns:
        if high_col in df.columns and low_col in df.columns and price_col in df.columns:
            high = df[high_col]
            low = df[low_col]
            close = df[price_col]
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df[atr_col] = tr.rolling(window=atr_window, min_periods=1).mean()
        else:
            raise ValueError(f"ATR column '{atr_col}' not found and cannot be computed")

    # Determine direction from momentum
    if momentum_direction_col and momentum_direction_col in df.columns:
        directions = (
            df[momentum_direction_col]
            .apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            .values
        )
    elif roc_col and roc_col in df.columns:
        directions = (
            df[roc_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).values
        )
    else:
        # Fallback: use predictions to infer direction (if rank > 0.5, assume long)
        directions = np.where(predictions > 0.5, 1, -1)

    # Extract arrays
    prices = df[price_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df[atr_col].values

    # Initialize tracking
    trades: List[TrendTrade] = []
    equity = 100000.0
    equity_curve = [equity]
    current_trade: Optional[TrendTrade] = None
    previous_rank: Optional[float] = None

    for i in range(len(df) - max_holding_bars):
        current_rank = predictions[i] if not np.isnan(predictions[i]) else None

        # Check if we should enter a new trade
        if current_trade is None:
            # Entry condition: rank >= min_rank
            if current_rank is not None and current_rank >= min_rank:
                direction = directions[i]
                if direction == 0:
                    continue

                entry_price = prices[i]
                atr = atrs[i]
                if pd.isna(atr) or atr <= 0:
                    continue

                # Calculate stop loss (dynamic ATR)
                if direction == 1:  # Long
                    stop_loss = entry_price - stop_loss_r * atr
                else:  # Short
                    stop_loss = entry_price + stop_loss_r * atr

                # Calculate position size (proportional to (Rank - 0.5))
                position_size = base_position_size * max(0, (current_rank - 0.5) * 2.0)

                current_trade = TrendTrade(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction=direction,
                    stop_loss=stop_loss,
                    initial_rank=current_rank,
                    current_rank=current_rank,
                    position_size=position_size,
                )
                previous_rank = current_rank

        # Check if we have an open trade
        if current_trade is not None:
            # Update current rank
            if current_rank is not None:
                current_trade.current_rank = current_rank

            # Check exit conditions
            exit_reason = None
            exit_price = None

            # Rank exit: when rank < min_rank
            if current_rank is not None and current_rank < min_rank:
                exit_reason = "rank_exit"
                exit_price = prices[i]

            # Stop loss
            if exit_reason is None:
                if current_trade.direction == 1:  # Long
                    if lows[i] <= current_trade.stop_loss:
                        exit_reason = "sl"
                        exit_price = current_trade.stop_loss
                else:  # Short
                    if highs[i] >= current_trade.stop_loss:
                        exit_reason = "sl"
                        exit_price = current_trade.stop_loss

            # Update stop loss (trailing stop based on ATR)
            if exit_reason is None:
                atr_current = atrs[i]
                if current_trade.direction == 1:  # Long
                    new_stop = prices[i] - stop_loss_r * atr_current
                    if new_stop > current_trade.stop_loss:
                        current_trade.stop_loss = new_stop
                else:  # Short
                    new_stop = prices[i] + stop_loss_r * atr_current
                    if new_stop < current_trade.stop_loss:
                        current_trade.stop_loss = new_stop

            # Check timeout
            if (
                exit_reason is None
                and (i - current_trade.entry_idx) >= max_holding_bars
            ):
                exit_reason = "timeout"
                exit_price = prices[i]

            # Add position logic (when rank hits new high)
            if enable_add_position and exit_reason is None and current_rank is not None:
                if current_rank > add_position_rank and previous_rank is not None:
                    if (
                        current_rank > previous_rank
                        and current_trade.add_position_size == 0.0
                    ):
                        # Add 50% position
                        current_trade.add_position_size = (
                            current_trade.position_size * 0.5
                        )

            # Reduce position logic (when rank drops from high)
            if (
                enable_reduce_position
                and exit_reason is None
                and current_rank is not None
            ):
                if previous_rank is not None and previous_rank > reduce_position_rank:
                    if current_rank < reduce_position_rank:
                        # Reduce position by 50%
                        current_trade.position_size *= 0.5
                        if current_trade.add_position_size > 0:
                            current_trade.add_position_size *= 0.5
                        exit_reason = "reduce"
                        exit_price = prices[i]

            # Exit trade if condition met
            if exit_reason is not None:
                current_trade.exit_idx = i
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason

                # Calculate PnL
                total_position = (
                    current_trade.position_size + current_trade.add_position_size
                )
                if current_trade.direction == 1:
                    pnl_pct = (
                        exit_price - current_trade.entry_price
                    ) / current_trade.entry_price
                else:
                    pnl_pct = (
                        current_trade.entry_price - exit_price
                    ) / current_trade.entry_price

                # Apply trading fees
                pnl_pct -= trading_fee * 2
                if current_trade.add_position_size > 0:
                    pnl_pct -= trading_fee

                current_trade.pnl = pnl_pct * total_position

                trades.append(current_trade)

                # Update equity
                equity *= 1.0 + current_trade.pnl
                equity_curve.append(equity)

                current_trade = None
                previous_rank = None
            else:
                previous_rank = current_rank

        # Update equity curve
        if current_trade is None:
            equity_curve.append(equity)

    # Close any remaining open trade
    if current_trade is not None:
        exit_price = prices[-1]
        current_trade.exit_idx = len(df) - 1
        current_trade.exit_price = exit_price
        current_trade.exit_reason = "timeout"

        total_position = current_trade.position_size + current_trade.add_position_size
        if current_trade.direction == 1:
            pnl_pct = (
                exit_price - current_trade.entry_price
            ) / current_trade.entry_price
        else:
            pnl_pct = (
                current_trade.entry_price - exit_price
            ) / current_trade.entry_price

        pnl_pct -= trading_fee * 2
        if current_trade.add_position_size > 0:
            pnl_pct -= trading_fee

        current_trade.pnl = pnl_pct * total_position

        trades.append(current_trade)
        equity *= 1.0 + current_trade.pnl
        equity_curve.append(equity)

    # Calculate metrics
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "rank_ic": 0.0,
            "top_bottom_spread": 0.0,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "final_equity": equity,
            "equity_curve": equity_curve,
            "trades": [],
        }

    # Trade statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]

    win_rate = len(winning_trades) / total_trades * 100.0 if total_trades > 0 else 0.0

    # Rank IC: correlation between initial rank and realized return
    if len(trades) > 1:
        initial_ranks = [t.initial_rank for t in trades]
        trade_returns = [t.pnl for t in trades if t.pnl is not None]
        if len(initial_ranks) == len(trade_returns) and len(trade_returns) > 1:
            rank_ic = np.corrcoef(initial_ranks, trade_returns)[0, 1]
            if np.isnan(rank_ic):
                rank_ic = 0.0
        else:
            rank_ic = 0.0
    else:
        rank_ic = 0.0

    # Top-Bottom Spread: average return of top decile - bottom decile
    if len(trades) > 10:
        trade_returns = [t.pnl for t in trades if t.pnl is not None]
        initial_ranks = [t.initial_rank for t in trades if t.pnl is not None]
        if len(trade_returns) == len(initial_ranks):
            # Sort by rank
            sorted_indices = np.argsort(initial_ranks)
            top_decile = int(len(sorted_indices) * 0.9)
            bottom_decile = int(len(sorted_indices) * 0.1)

            top_returns = [trade_returns[i] for i in sorted_indices[top_decile:]]
            bottom_returns = [trade_returns[i] for i in sorted_indices[:bottom_decile]]

            top_avg = np.mean(top_returns) if top_returns else 0.0
            bottom_avg = np.mean(bottom_returns) if bottom_returns else 0.0
            top_bottom_spread = top_avg - bottom_avg
        else:
            top_bottom_spread = 0.0
    else:
        top_bottom_spread = 0.0

    # PnL statistics
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    total_return = (equity / 100000.0 - 1.0) * 100.0

    if losing_trades:
        total_loss = abs(sum(t.pnl for t in losing_trades if t.pnl))
        total_win = sum(t.pnl for t in winning_trades if t.pnl)
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
    else:
        profit_factor = float("inf") if winning_trades else 0.0

    # Sharpe ratio
    if len(trades) > 1:
        trade_returns = [t.pnl for t in trades if t.pnl is not None]
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe_ratio = (
                np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252.0)
            )
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100.0

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "rank_ic": rank_ic,
        "top_bottom_spread": top_bottom_spread,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_equity": equity,
        "equity_curve": equity_curve,
        "trades": [
            {
                "entry_idx": t.entry_idx,
                "exit_idx": t.exit_idx,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "initial_rank": t.initial_rank,
                "exit_rank": t.current_rank,
                "position_size": t.position_size,
                "add_position_size": t.add_position_size,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
            }
            for t in trades
        ],
    }
