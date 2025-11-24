"""
SR 反转策略专属回测

回测逻辑：
- 方向：由 SR 类型决定（支撑区 → 做多，阻力区 → 做空）
- 入场点：价格触及 SR 区边界
- 止损：入场价 ± 1×ATR（反向）
- 止盈：入场价 ± 2×ATR（同向）
- 仓位：position_size ∝ P(success)
- 加仓：❌ 通常不加仓
- 减仓：若价格未快速脱离，且 CVD 相位转负，可提前部分止盈
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    """单笔交易记录"""

    entry_idx: int
    entry_price: float
    direction: int  # 1=Long, -1=Short
    stop_loss: float
    take_profit: float
    position_size: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "timeout", "reduce"
    pnl: Optional[float] = None
    rr_achieved: Optional[float] = None


def backtest_sr_reversal(
    df: pd.DataFrame,
    predictions: np.ndarray,  # P(success) = P(label=1)
    sr_type_col: Optional[str] = None,  # "support" or "resistance"
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    cvd_col: Optional[str] = None,
    atr_window: int = 14,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    max_holding_bars: int = 50,
    min_confidence: float = 0.3,
    base_position_size: float = 0.1,
    trading_fee: float = 0.001,  # 0.1% per trade
    enable_reduce_position: bool = True,
) -> Dict:
    """
    SR 反转策略专属回测

    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (P(success))
        sr_type_col: Column indicating SR type ("support" or "resistance")
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        cvd_col: CVD column (for reduce position logic)
        atr_window: ATR window if ATR column doesn't exist
        stop_loss_r: Stop loss in R units
        take_profit_r: Take profit in R units
        max_holding_bars: Maximum holding period
        min_confidence: Minimum confidence threshold
        base_position_size: Base position size
        trading_fee: Trading fee per trade
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

    # Determine direction from SR type
    if sr_type_col and sr_type_col in df.columns:
        directions = (
            df[sr_type_col]
            .apply(lambda x: 1 if x == "support" else (-1 if x == "resistance" else 0))
            .values
        )
    else:
        # Fallback: use predictions to infer direction (if prediction > 0.5, assume long)
        # This is a simplified approach; in practice, SR type should be explicitly provided
        directions = np.ones(len(df), dtype=int)  # Default to long

    # Extract arrays for performance
    prices = df[price_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df[atr_col].values

    if cvd_col and cvd_col in df.columns:
        cvd_values = df[cvd_col].values
    else:
        cvd_values = None

    # Initialize tracking
    trades: List[Trade] = []
    equity = 100000.0  # Initial capital
    equity_curve = [equity]
    current_trade: Optional[Trade] = None

    for i in range(len(df) - max_holding_bars):
        # Check if we should enter a new trade
        if current_trade is None:
            # Entry condition: prediction > min_confidence
            if predictions[i] >= min_confidence and not np.isnan(predictions[i]):
                direction = directions[i]
                if direction == 0:  # No valid direction
                    continue

                entry_price = prices[i]
                atr = atrs[i]
                if pd.isna(atr) or atr <= 0:
                    continue

                # Calculate stop loss and take profit
                if direction == 1:  # Long
                    stop_loss = entry_price - stop_loss_r * atr
                    take_profit = entry_price + take_profit_r * atr
                else:  # Short
                    stop_loss = entry_price + stop_loss_r * atr
                    take_profit = entry_price - take_profit_r * atr

                # Calculate position size (proportional to P(success))
                position_size = base_position_size * predictions[i]

                # Create trade
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
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

            # Check reduce position (if enabled)
            if (
                enable_reduce_position
                and exit_reason is None
                and cvd_values is not None
            ):
                # Check if CVD phase turns negative (for long) or positive (for short)
                if i > 0:
                    cvd_change = cvd_values[i] - cvd_values[i - 1]
                    if (
                        current_trade.direction == 1
                        and cvd_change < 0
                        and predictions[i] < 0.7
                    ):
                        # Reduce position by 50%
                        current_trade.position_size *= 0.5
                        # Continue holding with reduced position

            # Exit trade if condition met
            if exit_reason is not None:
                current_trade.exit_idx = i
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason

                # Calculate PnL
                if current_trade.direction == 1:  # Long
                    pnl_pct = (
                        exit_price - current_trade.entry_price
                    ) / current_trade.entry_price
                else:  # Short
                    pnl_pct = (
                        current_trade.entry_price - exit_price
                    ) / current_trade.entry_price

                # Apply trading fees
                pnl_pct -= trading_fee * 2  # Entry + Exit

                current_trade.pnl = pnl_pct * current_trade.position_size

                # Calculate R/R achieved
                atr_entry = atrs[current_trade.entry_idx]
                if atr_entry > 0:
                    if current_trade.direction == 1:
                        price_move = exit_price - current_trade.entry_price
                    else:
                        price_move = current_trade.entry_price - exit_price
                    current_trade.rr_achieved = price_move / (stop_loss_r * atr_entry)

                trades.append(current_trade)

                # Update equity
                equity *= 1.0 + current_trade.pnl
                equity_curve.append(equity)

                current_trade = None

        # Update equity curve even if no trade
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

        atr_entry = atrs[current_trade.entry_idx]
        if atr_entry > 0:
            if current_trade.direction == 1:
                price_move = exit_price - current_trade.entry_price
            else:
                price_move = current_trade.entry_price - exit_price
            current_trade.rr_achieved = price_move / (stop_loss_r * atr_entry)

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
            "total_return": 0.0,
            "avg_rr": 0.0,
            "avg_win_rr": 0.0,
            "avg_loss_rr": 0.0,
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

    # R/R statistics
    rr_values = [t.rr_achieved for t in trades if t.rr_achieved is not None]
    avg_rr = np.mean(rr_values) if rr_values else 0.0

    win_rr = [t.rr_achieved for t in winning_trades if t.rr_achieved is not None]
    loss_rr = [t.rr_achieved for t in losing_trades if t.rr_achieved is not None]

    avg_win_rr = np.mean(win_rr) if win_rr else 0.0
    avg_loss_rr = np.mean(loss_rr) if loss_rr else 0.0

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
            )  # Annualized
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
        "total_return": total_return,
        "avg_rr": avg_rr,
        "avg_win_rr": avg_win_rr,
        "avg_loss_rr": avg_loss_rr,
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
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "position_size": t.position_size,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
                "rr_achieved": t.rr_achieved,
            }
            for t in trades
        ],
    }
