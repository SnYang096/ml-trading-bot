"""Risk management module for dynamic stop loss and take profit."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ml_trading.config.settings import (
    STOP_LOSS_MULTIPLIER,
    TAKE_PROFIT_MULTIPLIER,
    MAX_CONSECUTIVE_LOSSES,
)


class RiskManager:
    """Handles risk management including dynamic stop loss and take profit."""

    def __init__(self):
        """Initialize the risk manager."""
        self.position_history = []
        self.consecutive_losses = 0
        self.rolling_returns = []
        self.rolling_window = 50  # Window for calculating rolling statistics

    def calculate_dynamic_levels(
        self, historical_prices: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels based on historical volatility.

        Args:
            historical_prices: Historical price series

        Returns:
            Tuple of (stop_loss_level, take_profit_level)
        """
        # Calculate rolling returns
        returns = historical_prices.pct_change().dropna()

        # Update rolling returns history
        self.rolling_returns.extend(returns.tolist())
        if len(self.rolling_returns) > self.rolling_window:
            self.rolling_returns = self.rolling_returns[-self.rolling_window :]

        # Calculate statistics
        if len(self.rolling_returns) > 1:
            mean_return = float(np.mean(self.rolling_returns))
            std_return = float(np.std(self.rolling_returns))
        else:
            mean_return = 0.0
            std_return = 0.01  # Default value to avoid division by zero

        # Calculate dynamic levels
        stop_loss_level = float(STOP_LOSS_MULTIPLIER * std_return)
        take_profit_level = float(TAKE_PROFIT_MULTIPLIER * std_return)

        return stop_loss_level, take_profit_level

    def check_structural_failure(self) -> bool:
        """
        Check for structural failure (consecutive losses).

        Returns:
            True if structural failure detected, False otherwise
        """
        return self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES

    def update_position_history(self, position: float, pnl: float):
        """
        Update position history and consecutive losses counter.

        Args:
            position: Current position size
            pnl: Profit and loss for the trade
        """
        self.position_history.append(
            {
                "position": float(position),
                "pnl": float(pnl),
                "timestamp": len(self.position_history),  # Simple timestamp
            }
        )

        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def adjust_position_size(
        self,
        signal: float,
        expected_return: float,
        current_price: float,
        account_value: float = 100000.0,
    ) -> float:
        """
        Adjust position size based on signal confidence and risk management rules.

        Args:
            signal: Trading signal (-1 to 1)
            expected_return: Expected return from model
            current_price: Current asset price
            account_value: Current account value

        Returns:
            Adjusted position size
        """
        # Base position size based on account value and current price
        base_position = account_value * 0.01 / current_price  # 1% of account per trade

        # Adjust based on signal confidence
        confidence = abs(signal) * abs(expected_return)
        adjusted_position = base_position * min(confidence, 1.0)

        # Apply position direction
        if signal > 0:
            final_position = adjusted_position
        elif signal < 0:
            final_position = -adjusted_position
        else:
            final_position = 0.0

        return float(final_position)

    def apply_risk_management(
        self, ensemble_df: pd.DataFrame, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply risk management rules to ensemble predictions.

        Args:
            ensemble_df: DataFrame with ensemble predictions
            price_data: Price data for dynamic level calculation

        Returns:
            DataFrame with risk-managed positions
        """
        result_df = ensemble_df.copy()

        # Initialize position column
        result_df["position"] = 0.0
        result_df["stop_loss_level"] = 0.0
        result_df["take_profit_level"] = 0.0

        # Get current price (assuming close price)
        current_price = (
            float(price_data["close"].iloc[-1]) if len(price_data) > 0 else 100.0
        )

        # Calculate dynamic levels
        stop_loss_level, take_profit_level = self.calculate_dynamic_levels(
            price_data["close"]
        )

        # Apply risk management for each prediction
        for i in range(len(result_df)):
            # Convert to native Python types using numpy
            signal = float(np.asarray(result_df["discrete_signal"].iloc[i]))
            expected_return = float(np.asarray(result_df["ensemble_return"].iloc[i]))

            # Check for structural failure
            if self.check_structural_failure():
                # Close position if structural failure detected
                result_df.loc[i, "position"] = 0.0
            else:
                # Adjust position size based on risk management
                position = self.adjust_position_size(
                    signal, expected_return, current_price
                )
                result_df.loc[i, "position"] = float(position)

            # Store dynamic levels
            result_df.loc[i, "stop_loss_level"] = float(stop_loss_level)
            result_df.loc[i, "take_profit_level"] = float(take_profit_level)

            # Simulate PnL update (in a real system, this would come from actual trading)
            # For demonstration, we'll use a simple simulation
            if i > 0:
                prev_position = float(np.asarray(result_df.loc[i - 1, "position"]))
                if prev_position != 0.0:
                    # Calculate PnL based on position and price change
                    prev_price = (
                        float(price_data["close"].iloc[-2])
                        if len(price_data) > 1
                        else current_price
                    )
                    pnl = prev_position * (current_price - prev_price) / prev_price
                    self.update_position_history(prev_position, pnl)

        return result_df
