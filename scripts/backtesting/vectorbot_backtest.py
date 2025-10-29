"""VectorBot backtest with stop loss, take profit, and position scaling."""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from ml_trading.strategies.ml_strategy import MLTradingStrategy
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.feature_engineering import FeatureEngineer


DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", "trained_model_enhanced_may_2025.pkl"
)


class VectorBotBacktest:
    """VectorBot backtest with advanced risk management."""

    def __init__(self, model_path: str, initial_capital: float = 100000):
        """
        Initialize VectorBot backtest.

        Args:
            model_path: Path to trained model
            initial_capital: Starting capital
        """
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0
        self.peak_equity = initial_capital

        # Risk management parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_position_size = 0.1  # 10% of capital per position
        self.scaling_factor = 0.5  # 50% scaling for additional positions
        self.max_positions = 3  # Maximum number of positions

        # Load trained model
        self.load_model()

    def load_model(self):
        """Load the trained model."""
        print(f"Loading trained model from {self.model_path}...")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.strategy = model_data["strategy"]
        self.data_loader = model_data["data_loader"]
        self.feature_engineer = model_data["feature_engineer"]
        self.engineered_data = model_data["engineered_data"]
        self.metrics = model_data["metrics"]

        print("✅ Model loaded successfully")
        print(f"   Training date: {model_data['training_date']}")
        print(f"   Data info: {model_data['data_info']}")

    def calculate_position_size(self, signal_strength: float,
                                current_price: float) -> float:
        """
        Calculate position size based on signal strength and risk management.

        Args:
            signal_strength: Strength of the signal (0-1)
            current_price: Current price

        Returns:
            Position size in units
        """
        # Base position size
        base_size = self.capital * self.max_position_size / current_price

        # Adjust for signal strength
        adjusted_size = base_size * signal_strength

        # Check if we can add more positions
        active_positions = len(
            [p for p in self.positions if p["status"] == "active"])
        if active_positions >= self.max_positions:
            return 0

        # Scale down for additional positions
        if active_positions > 0:
            adjusted_size *= self.scaling_factor**active_positions

        return adjusted_size

    def update_positions(self, current_price: float, timestamp: pd.Timestamp):
        """Update all active positions with current price."""
        for position in self.positions:
            if position["status"] != "active":
                continue

            # Calculate current P&L
            if position["side"] == "long":
                pnl = (current_price -
                       position["entry_price"]) * position["size"]
            else:  # short
                pnl = (position["entry_price"] -
                       current_price) * position["size"]

            position["current_pnl"] = pnl
            position["current_price"] = current_price
            position["timestamp"] = timestamp

            # Check stop loss
            if position["side"] == "long":
                stop_price = position["entry_price"] * (1 - self.stop_loss_pct)
                if current_price <= stop_price:
                    self.close_position(position, current_price, timestamp,
                                        "stop_loss")
                    continue
            else:  # short
                stop_price = position["entry_price"] * (1 + self.stop_loss_pct)
                if current_price >= stop_price:
                    self.close_position(position, current_price, timestamp,
                                        "stop_loss")
                    continue

            # Check take profit
            if position["side"] == "long":
                take_profit_price = position["entry_price"] * (
                    1 + self.take_profit_pct)
                if current_price >= take_profit_price:
                    self.close_position(position, current_price, timestamp,
                                        "take_profit")
                    continue
            else:  # short
                take_profit_price = position["entry_price"] * (
                    1 - self.take_profit_pct)
                if current_price <= take_profit_price:
                    self.close_position(position, current_price, timestamp,
                                        "take_profit")
                    continue

    def close_position(self, position: dict, exit_price: float,
                       timestamp: pd.Timestamp, reason: str):
        """Close a position and record the trade."""
        # Calculate final P&L
        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:  # short
            pnl = (position["entry_price"] - exit_price) * position["size"]

        # Update capital
        self.capital += pnl

        # Record trade
        trade = {
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": position["size"],
            "pnl": pnl,
            "return_pct":
            pnl / (position["entry_price"] * position["size"]) * 100,
            "reason": reason,
            "duration": (timestamp - position["entry_time"]).total_seconds() /
            60,  # minutes
        }

        self.trades.append(trade)
        position["status"] = "closed"
        position["exit_price"] = exit_price
        position["exit_time"] = timestamp
        position["pnl"] = pnl
        position["reason"] = reason

        print(
            f"   🔄 Closed {position['side']} position: {pnl:.2f} P&L ({reason})"
        )

    def open_position(
        self,
        side: str,
        size: float,
        price: float,
        timestamp: pd.Timestamp,
        signal_strength: float,
    ):
        """Open a new position."""
        position = {
            "id": len(self.positions),
            "side": side,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp,
            "status": "active",
            "signal_strength": signal_strength,
            "current_pnl": 0,
            "current_price": price,
        }

        self.positions.append(position)
        print(f"   📈 Opened {side} position: {size:.4f} units at {price:.2f}")

    def run_backtest(self, start_date: str = None, end_date: str = None):
        """Run the backtest."""
        print("🚀 Starting VectorBot Backtest")
        print("=" * 50)

        # Get 5T data for backtesting
        data_5t = self.engineered_data["5T"]

        # Filter by date range if specified
        if start_date:
            data_5t = data_5t[data_5t.index >= start_date]
        if end_date:
            data_5t = data_5t[data_5t.index <= end_date]

        print(f"Backtesting on {len(data_5t)} bars")
        print(f"Date range: {data_5t.index[0]} to {data_5t.index[-1]}")

        # Prepare features for prediction
        feature_columns = [
            col for col in data_5t.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X_5t = data_5t[feature_columns]
        X_5t_clean = X_5t.dropna()

        if X_5t_clean.empty:
            print("❌ No valid data for prediction")
            return

        print(f"Using {len(X_5t_clean)} clean data points for prediction")

        # Generate predictions
        stage1_model = self.strategy.pipeline.stage1_models["5T"]
        stage2_model = self.strategy.pipeline.stage2_models["5T"]

        stage1_preds = stage1_model.predict(X_5t_clean)
        stage2_preds = stage2_model.predict(X_5t_clean)

        # Create signals DataFrame
        signals = pd.DataFrame({
            "timestamp": X_5t_clean.index,
            "stage1_pred": stage1_preds,
            "stage2_pred": stage2_preds,
            "close": data_5t["close"].loc[X_5t_clean.index],
        })

        # Convert to discrete signals
        signals["discrete_signal"] = 0
        signals.loc[stage1_preds > 0.6, "discrete_signal"] = 1  # Long
        signals.loc[stage1_preds < 0.4, "discrete_signal"] = -1  # Short

        # Calculate signal strength
        signals["signal_strength"] = np.abs(signals["stage1_pred"] - 0.5)

        print(f"\n📊 Signal Statistics:")
        print(f"   Total signals: {len(signals)}")
        print(
            f"   Long signals: {len(signals[signals['discrete_signal'] == 1])}"
        )
        print(
            f"   Short signals: {len(signals[signals['discrete_signal'] == -1])}"
        )
        print(
            f"   Hold signals: {len(signals[signals['discrete_signal'] == 0])}"
        )

        # Run backtest
        print(f"\n🔄 Running backtest...")

        for i, (timestamp, row) in enumerate(signals.iterrows()):
            current_price = row["close"]
            signal = row["discrete_signal"]
            signal_strength = row["signal_strength"]

            # Update existing positions
            self.update_positions(current_price, timestamp)

            # Check for new signals
            if signal != 0 and signal_strength > 0.1:  # Lower minimum signal strength
                # Check if we can open new position
                active_positions = len(
                    [p for p in self.positions if p["status"] == "active"])

                if active_positions < self.max_positions:
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        signal_strength, current_price)

                    if position_size > 0:
                        side = "long" if signal == 1 else "short"
                        self.open_position(
                            side,
                            position_size,
                            current_price,
                            timestamp,
                            signal_strength,
                        )

            # Update equity curve
            total_pnl = sum([
                p["current_pnl"] for p in self.positions
                if p["status"] == "active"
            ])
            current_equity = self.capital + total_pnl
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": current_equity,
                "capital": self.capital,
                "open_pnl": total_pnl,
            })

            # Update drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

            # Progress update
            if i % 50 == 0:
                print(
                    f"   Processed {i+1}/{len(signals)} bars, Equity: {current_equity:.2f}"
                )

        # Close any remaining positions
        for position in self.positions:
            if position["status"] == "active":
                last_price = signals["close"].iloc[-1]
                last_timestamp = signals.index[-1]
                self.close_position(position, last_price, last_timestamp,
                                    "end_of_data")

        # Calculate final results
        self.calculate_results()

        # Save results
        self.save_results()

        print("\n🎉 Backtest completed successfully!")

    def calculate_results(self):
        """Calculate backtest results."""
        if not self.trades:
            print("❌ No trades executed")
            # Set default results
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_return": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "final_equity": self.capital,
                "initial_capital": self.initial_capital,
            }
            return

        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["pnl"] > 0])
        losing_trades = len([t for t in self.trades if t["pnl"] < 0])

        win_rate = winning_trades / total_trades * 100

        total_pnl = sum([t["pnl"] for t in self.trades])
        avg_win = (np.mean([t["pnl"] for t in self.trades
                            if t["pnl"] > 0]) if winning_trades > 0 else 0)
        avg_loss = (np.mean([t["pnl"] for t in self.trades
                             if t["pnl"] < 0]) if losing_trades > 0 else 0)

        profit_factor = (abs(avg_win * winning_trades /
                             (avg_loss * losing_trades))
                         if losing_trades > 0 else float("inf"))

        # Risk metrics
        returns = [t["return_pct"] for t in self.trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns) *
                        np.sqrt(252) if np.std(returns) > 0 else 0)

        # Final equity
        final_equity = self.capital
        total_return = ((final_equity - self.initial_capital) /
                        self.initial_capital * 100)

        self.results = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown * 100,
            "final_equity": final_equity,
            "initial_capital": self.initial_capital,
        }

        print(f"\n📈 Backtest Results:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Total P&L: {total_pnl:.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Average Win: {avg_win:.2f}")
        print(f"   Average Loss: {avg_loss:.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {self.max_drawdown * 100:.2f}%")
        print(f"   Final Equity: {final_equity:.2f}")

    def save_results(self):
        """Save backtest results."""
        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv("vectorbot_trades.csv", index=False)

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv("vectorbot_equity_curve.csv", index=False)

        # Save results
        with open("vectorbot_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"   - vectorbot_trades.csv")
        print(f"   - vectorbot_equity_curve.csv")
        print(f"   - vectorbot_results.json")


def main():
    """Main function to run VectorBot backtest."""
    # Check if model exists
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run scripts/training/train_model_enhanced.py first")
        return

    # Initialize backtest
    backtest = VectorBotBacktest(model_path, initial_capital=100000)

    # Run backtest
    backtest.run_backtest()


if __name__ == "__main__":
    main()
