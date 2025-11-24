"""
SR 突破策略专属回测

回测逻辑：
- 方向：由突破方向决定（向上突破 → 做多，向下突破 → 做空）
- 入场点：突破确认（如收盘站稳 + 成交量 spike）
- 止损：突破点反向 1×ATR 或区间边界
- 止盈：动态止盈（初始 TP = 入场 + 2×ATR，若 R/R > 2.0 启用 trailing stop）
- 仓位：position_size ∝ predicted_R_R
- 加仓：✅ 若价格回踩不破且 CVD 持续流入，可在 0.5×ATR 处加仓
- 减仓：当实际 R/R 达到预测值的 80%，减半仓锁定利润
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class BreakoutTrade:
    """突破交易记录"""

    entry_idx: int
    entry_price: float
    direction: int  # 1=Long, -1=Short
    stop_loss: float
    initial_take_profit: float
    current_take_profit: float  # For trailing stop
    predicted_rr: float
    position_size: float
    add_position_size: float = 0.0  # Additional position from add-on
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = (
        None  # "tp", "sl", "timeout", "trailing_stop", "reduce"
    )
    pnl: Optional[float] = None
    realized_rr: Optional[float] = None
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    mae: Optional[float] = None  # Maximum Adverse Excursion


def backtest_sr_breakout(
    df: pd.DataFrame,
    predictions: np.ndarray,  # Predicted R/R
    breakout_direction_col: Optional[str] = None,  # "up" or "down"
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    cvd_col: Optional[str] = None,
    atr_window: int = 14,
    stop_loss_r: float = 1.0,
    initial_take_profit_r: float = 2.0,
    max_holding_bars: int = 50,
    min_predicted_rr: float = 1.0,
    base_position_size: float = 0.1,
    trailing_stop_threshold: float = 2.0,  # Enable trailing stop if predicted R/R > this
    trailing_stop_atr: float = 0.5,  # Trailing stop distance in ATR
    enable_add_position: bool = True,
    add_position_atr: float = 0.5,  # Add position when price retraces this much
    enable_reduce_position: bool = True,
    reduce_position_threshold: float = 0.8,  # Reduce when realized R/R reaches this % of predicted
    trading_fee: float = 0.001,
) -> Dict:
    """
    SR 突破策略专属回测

    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (Predicted R/R)
        breakout_direction_col: Column indicating breakout direction ("up" or "down")
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        cvd_col: CVD column (for add/reduce position logic)
        atr_window: ATR window if ATR column doesn't exist
        stop_loss_r: Stop loss in R units
        initial_take_profit_r: Initial take profit in R units
        max_holding_bars: Maximum holding period
        min_predicted_rr: Minimum predicted R/R to enter trade
        base_position_size: Base position size
        trailing_stop_threshold: Enable trailing stop if predicted R/R > this
        trailing_stop_atr: Trailing stop distance in ATR
        enable_add_position: Whether to enable add position logic
        add_position_atr: ATR distance for add position
        enable_reduce_position: Whether to enable reduce position logic
        reduce_position_threshold: Reduce position when realized R/R reaches this % of predicted
        trading_fee: Trading fee per trade

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

    # Determine direction from breakout direction
    if breakout_direction_col and breakout_direction_col in df.columns:
        directions = (
            df[breakout_direction_col]
            .apply(lambda x: 1 if x == "up" else (-1 if x == "down" else 0))
            .values
        )
    else:
        # Fallback: use predictions to infer direction (if prediction > 0, assume long)
        directions = np.ones(len(df), dtype=int)  # Default to long

    # Extract arrays
    prices = df[price_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df[atr_col].values

    if cvd_col and cvd_col in df.columns:
        cvd_values = df[cvd_col].values
    else:
        cvd_values = None

    # Initialize tracking
    trades: List[BreakoutTrade] = []
    equity = 100000.0
    equity_curve = [equity]
    current_trade: Optional[BreakoutTrade] = None

    for i in range(len(df) - max_holding_bars):
        # Check if we should enter a new trade
        if current_trade is None:
            # Entry condition: predicted R/R >= min_predicted_rr
            if predictions[i] >= min_predicted_rr and not np.isnan(predictions[i]):
                direction = directions[i]
                if direction == 0:
                    continue

                entry_price = prices[i]
                atr = atrs[i]
                if pd.isna(atr) or atr <= 0:
                    continue

                # Calculate stop loss
                if direction == 1:  # Long
                    stop_loss = entry_price - stop_loss_r * atr
                    initial_tp = entry_price + initial_take_profit_r * atr
                else:  # Short
                    stop_loss = entry_price + stop_loss_r * atr
                    initial_tp = entry_price - initial_take_profit_r * atr

                # Use predicted R/R for take profit if > threshold
                if predictions[i] > trailing_stop_threshold:
                    if direction == 1:
                        initial_tp = entry_price + predictions[i] * atr
                    else:
                        initial_tp = entry_price - predictions[i] * atr

                # Calculate position size (proportional to predicted R/R)
                position_size = base_position_size * min(predictions[i] / 2.0, 1.0)

                current_trade = BreakoutTrade(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction=direction,
                    stop_loss=stop_loss,
                    initial_take_profit=initial_tp,
                    current_take_profit=initial_tp,
                    predicted_rr=predictions[i],
                    position_size=position_size,
                )

        # Check if we have an open trade
        if current_trade is not None:
            # Track MFE and MAE
            if current_trade.direction == 1:  # Long
                current_mfe = (highs[i] - current_trade.entry_price) / (
                    stop_loss_r * atrs[current_trade.entry_idx]
                )
                current_mae = (current_trade.entry_price - lows[i]) / (
                    stop_loss_r * atrs[current_trade.entry_idx]
                )
            else:  # Short
                current_mfe = (current_trade.entry_price - lows[i]) / (
                    stop_loss_r * atrs[current_trade.entry_idx]
                )
                current_mae = (highs[i] - current_trade.entry_price) / (
                    stop_loss_r * atrs[current_trade.entry_idx]
                )

            if current_trade.mfe is None or current_mfe > current_trade.mfe:
                current_trade.mfe = current_mfe
            if current_trade.mae is None or current_mae > current_trade.mae:
                current_trade.mae = current_mae

            # Calculate realized R/R
            if current_trade.mae > 0:
                current_trade.realized_rr = current_trade.mfe / current_trade.mae

            # Check exit conditions
            exit_reason = None
            exit_price = None

            # Check take profit
            if current_trade.direction == 1:  # Long
                if highs[i] >= current_trade.current_take_profit:
                    exit_reason = "tp"
                    exit_price = current_trade.current_take_profit
                elif lows[i] <= current_trade.stop_loss:
                    exit_reason = "sl"
                    exit_price = current_trade.stop_loss
            else:  # Short
                if lows[i] <= current_trade.current_take_profit:
                    exit_reason = "tp"
                    exit_price = current_trade.current_take_profit
                elif highs[i] >= current_trade.stop_loss:
                    exit_reason = "sl"
                    exit_price = current_trade.stop_loss

            # Trailing stop logic
            if (
                exit_reason is None
                and current_trade.predicted_rr > trailing_stop_threshold
            ):
                atr_current = atrs[i]
                if current_trade.direction == 1:  # Long
                    # Update trailing stop (highest price - trailing_stop_atr * ATR)
                    new_trailing_stop = prices[i] - trailing_stop_atr * atr_current
                    if new_trailing_stop > current_trade.stop_loss:
                        current_trade.stop_loss = new_trailing_stop
                    # Check if price hits trailing stop
                    if lows[i] <= current_trade.stop_loss:
                        exit_reason = "trailing_stop"
                        exit_price = current_trade.stop_loss
                else:  # Short
                    new_trailing_stop = prices[i] + trailing_stop_atr * atr_current
                    if new_trailing_stop < current_trade.stop_loss:
                        current_trade.stop_loss = new_trailing_stop
                    if highs[i] >= current_trade.stop_loss:
                        exit_reason = "trailing_stop"
                        exit_price = current_trade.stop_loss

            # Check timeout
            if (
                exit_reason is None
                and (i - current_trade.entry_idx) >= max_holding_bars
            ):
                exit_reason = "timeout"
                exit_price = prices[i]

            # Add position logic
            if enable_add_position and exit_reason is None and cvd_values is not None:
                if current_trade.add_position_size == 0.0:  # Haven't added yet
                    # Check if price retraces to add_position_atr and CVD is positive
                    retrace_distance = (
                        abs(prices[i] - current_trade.entry_price)
                        / atrs[current_trade.entry_idx]
                    )
                    if retrace_distance <= add_position_atr:
                        if (
                            current_trade.direction == 1
                            and cvd_values[i] > cvd_values[current_trade.entry_idx]
                        ):
                            # Add 50% position
                            current_trade.add_position_size = (
                                current_trade.position_size * 0.5
                            )
                        elif (
                            current_trade.direction == -1
                            and cvd_values[i] < cvd_values[current_trade.entry_idx]
                        ):
                            current_trade.add_position_size = (
                                current_trade.position_size * 0.5
                            )

            # Reduce position logic
            if (
                enable_reduce_position
                and exit_reason is None
                and current_trade.realized_rr is not None
            ):
                if (
                    current_trade.realized_rr
                    >= current_trade.predicted_rr * reduce_position_threshold
                ):
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
                pnl_pct -= trading_fee * 2  # Entry + Exit
                if current_trade.add_position_size > 0:
                    pnl_pct -= trading_fee  # Additional entry fee

                current_trade.pnl = pnl_pct * total_position

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
            "total_return": 0.0,
            "avg_predicted_rr": 0.0,
            "avg_realized_rr": 0.0,
            "top_decile_rr": 0.0,
            "mse": 0.0,
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
    predicted_rr_values = [t.predicted_rr for t in trades]
    realized_rr_values = [t.realized_rr for t in trades if t.realized_rr is not None]

    avg_predicted_rr = np.mean(predicted_rr_values) if predicted_rr_values else 0.0
    avg_realized_rr = np.mean(realized_rr_values) if realized_rr_values else 0.0

    # Top decile R/R
    if realized_rr_values:
        top_decile_rr = np.percentile(realized_rr_values, 90)
    else:
        top_decile_rr = 0.0

    # MSE (Mean Squared Error between predicted and realized R/R)
    if len(realized_rr_values) == len(predicted_rr_values):
        mse = np.mean(
            (np.array(predicted_rr_values) - np.array(realized_rr_values)) ** 2
        )
    else:
        # Align arrays
        min_len = min(len(predicted_rr_values), len(realized_rr_values))
        mse = np.mean(
            (
                np.array(predicted_rr_values[:min_len])
                - np.array(realized_rr_values[:min_len])
            )
            ** 2
        )

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
        "total_return": total_return,
        "avg_predicted_rr": avg_predicted_rr,
        "avg_realized_rr": avg_realized_rr,
        "top_decile_rr": top_decile_rr,
        "mse": mse,
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
                "predicted_rr": t.predicted_rr,
                "realized_rr": t.realized_rr,
                "mfe": t.mfe,
                "mae": t.mae,
                "position_size": t.position_size,
                "add_position_size": t.add_position_size,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
            }
            for t in trades
        ],
    }
