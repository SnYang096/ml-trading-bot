"""
压缩区突破策略专属回测

回测逻辑：
- 方向：直接由标签符号决定（+1 → 多，-1 → 空）
- 入场点：突破确认 K 线收盘后
- 止损：区间另一侧边界 或 突破点 ±1×ATR
- 止盈：初始 TP = 区间高度（Measured Move）
- 仓位：position_size = base_size if label ≠ 0 else 0
- 加仓：❌ 通常不加仓
- 减仓：价格达到区间高度目标，减半仓
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class CompressionTrade:
    """压缩区突破交易记录"""

    entry_idx: int
    entry_price: float
    direction: int  # 1=Long, -1=Short, 0=No trade
    stop_loss: float
    take_profit: float  # Measured Move target
    compression_range_height: float
    position_size: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "timeout", "reduce"
    pnl: Optional[float] = None
    label: int  # -1, 0, +1


def backtest_compression_breakout(
    df: pd.DataFrame,
    predictions: np.ndarray,  # -1, 0, +1
    compression_range_col: Optional[str] = None,  # Tuple of (low, high)
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    stop_loss_r: float = 1.0,
    max_holding_bars: int = 50,
    base_position_size: float = 0.1,
    trading_fee: float = 0.001,
    enable_reduce_position: bool = True,
) -> Dict:
    """
    压缩区突破策略专属回测

    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (-1, 0, +1)
        compression_range_col: Column with compression range (low, high) tuple
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        stop_loss_r: Stop loss in R units (if range not available)
        max_holding_bars: Maximum holding period
        base_position_size: Base position size
        trading_fee: Trading fee per trade
        enable_reduce_position: Whether to enable reduce position at target

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

    # Extract arrays
    prices = df[price_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df[atr_col].values

    # Get compression range if available
    compression_ranges: Optional[np.ndarray] = None
    if compression_range_col and compression_range_col in df.columns:
        compression_ranges = df[compression_range_col].values

    # Initialize tracking
    trades: List[CompressionTrade] = []
    equity = 100000.0
    equity_curve = [equity]
    current_trade: Optional[CompressionTrade] = None

    for i in range(len(df) - max_holding_bars):
        # Check if we should enter a new trade
        if current_trade is None:
            # Entry condition: prediction != 0
            if predictions[i] != 0 and not np.isnan(predictions[i]):
                direction = int(predictions[i])  # -1, +1

                entry_price = prices[i]
                atr = atrs[i]
                if pd.isna(atr) or atr <= 0:
                    continue

                # Calculate stop loss and take profit
                if compression_ranges is not None and compression_ranges[i] is not None:
                    range_low, range_high = compression_ranges[i]
                    range_height = abs(range_high - range_low)

                    if direction == 1:  # Long
                        stop_loss = max(range_low, entry_price - stop_loss_r * atr)
                        take_profit = entry_price + range_height  # Measured Move
                    else:  # Short
                        stop_loss = min(range_high, entry_price + stop_loss_r * atr)
                        take_profit = entry_price - range_height
                else:
                    # Fallback: use ATR
                    range_height = 2.0 * atr  # Approximate
                    if direction == 1:
                        stop_loss = entry_price - stop_loss_r * atr
                        take_profit = entry_price + range_height
                    else:
                        stop_loss = entry_price + stop_loss_r * atr
                        take_profit = entry_price - range_height

                current_trade = CompressionTrade(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    compression_range_height=(
                        range_height if compression_ranges is not None else 2.0 * atr
                    ),
                    position_size=base_position_size,
                    label=int(predictions[i]),
                )

        # Check if we have an open trade
        if current_trade is not None:
            # Check exit conditions
            exit_reason = None
            exit_price = None

            # Check take profit
            if current_trade.direction == 1:  # Long
                if highs[i] >= current_trade.take_profit:
                    exit_reason = "tp"
                    exit_price = current_trade.take_profit
                elif lows[i] <= current_trade.stop_loss:
                    exit_reason = "sl"
                    exit_price = current_trade.stop_loss
            else:  # Short
                if lows[i] <= current_trade.take_profit:
                    exit_reason = "tp"
                    exit_price = current_trade.take_profit
                elif highs[i] >= current_trade.stop_loss:
                    exit_reason = "sl"
                    exit_price = current_trade.stop_loss

            # Check timeout
            if (
                exit_reason is None
                and (i - current_trade.entry_idx) >= max_holding_bars
            ):
                exit_reason = "timeout"
                exit_price = prices[i]

            # Reduce position logic (when price reaches target)
            if enable_reduce_position and exit_reason is None:
                if current_trade.direction == 1:
                    if prices[i] >= current_trade.take_profit * 0.9:  # 90% of target
                        current_trade.position_size *= 0.5
                        exit_reason = "reduce"
                        exit_price = prices[i]
                else:
                    if (
                        prices[i] <= current_trade.take_profit * 1.1
                    ):  # 110% of target (for short)
                        current_trade.position_size *= 0.5
                        exit_reason = "reduce"
                        exit_price = prices[i]

            # Exit trade if condition met
            if exit_reason is not None:
                current_trade.exit_idx = i
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason

                # Calculate PnL
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

                current_trade.pnl = pnl_pct * current_trade.position_size

                trades.append(current_trade)

                # Update equity
                equity *= 1.0 + current_trade.pnl
                equity_curve.append(equity)

                current_trade = None

        # Update equity curve
        if current_trade is None:
            equity_curve.append(equity)

    # Close any remaining open trade
    if current_trade is not None:
        exit_price = prices[-1]
        current_trade.exit_idx = len(df) - 1
        current_trade.exit_price = exit_price
        current_trade.exit_reason = "timeout"

        if current_trade.direction == 1:
            pnl_pct = (
                exit_price - current_trade.entry_price
            ) / current_trade.entry_price
        else:
            pnl_pct = (
                current_trade.entry_price - exit_price
            ) / current_trade.entry_price

        pnl_pct -= trading_fee * 2
        current_trade.pnl = pnl_pct * current_trade.position_size

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
            "accuracy": 0.0,
            "f1_score": 0.0,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "final_equity": equity,
            "equity_curve": equity_curve,
            "trades": [],
            "confusion_matrix": {},
        }

    # Trade statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]

    win_rate = len(winning_trades) / total_trades * 100.0 if total_trades > 0 else 0.0

    # Accuracy: correct direction predictions
    correct_predictions = sum(
        1
        for t in trades
        if (t.direction == 1 and t.pnl > 0) or (t.direction == -1 and t.pnl > 0)
    )
    accuracy = correct_predictions / total_trades * 100.0 if total_trades > 0 else 0.0

    # F1-score: for multiclass classification
    # True positives: correct direction + profitable
    tp = correct_predictions
    # False positives: wrong direction or unprofitable
    fp = total_trades - correct_predictions
    # False negatives: missed opportunities (predictions == 0)
    # Note: We can't calculate FN without knowing true labels, so we use a simplified version
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_trades if total_trades > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Confusion matrix (simplified)
    confusion_matrix = {
        "true_positive": tp,
        "false_positive": fp,
        "precision": precision,
        "recall": recall,
    }

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
        "accuracy": accuracy,
        "f1_score": f1_score,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_equity": equity,
        "equity_curve": equity_curve,
        "confusion_matrix": confusion_matrix,
        "trades": [
            {
                "entry_idx": t.entry_idx,
                "exit_idx": t.exit_idx,
                "direction": t.direction,
                "label": t.label,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "position_size": t.position_size,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
            }
            for t in trades
        ],
    }
