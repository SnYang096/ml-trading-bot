"""VectorBot backtest with stop loss, take profit, and position scaling."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import to make class available for pickle when loading legacy bundles
from time_series_model.models.train_model import MultiTimeframeComprehensiveEngineer  # noqa: F401
from time_series_model.strategies.ml_strategy import MLTradingStrategy


class MultiTimeframeEnhancedEngineer(
        MultiTimeframeComprehensiveEngineer):  # pragma: no cover - pickle shim
    """Legacy shim for bundles serialized with older engineer name."""

    pass


class VectorBotBacktest:
    """VectorBot backtest with advanced risk management."""

    def __init__(self,
                 model_path: str,
                 symbol: Optional[str] = None,
                 initial_capital: float = 100000):
        """
        Initialize VectorBot backtest.

        Args:
            model_path: Path to trained model
            symbol: Trading symbol (for logging/output naming)
            initial_capital: Starting capital
        """
        self.model_path = model_path
        self.symbol = symbol or "UNKNOWN"
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital

        # Risk management parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_position_size = 0.1  # 10% of capital per position
        self.scaling_factor = 0.5  # 50% scaling for additional positions
        self.max_positions = 3  # Maximum number of positions

        # Load trained model
        self.load_model()

    def load_model(self) -> None:
        """Load the trained model."""
        print(f"Loading trained model from {self.model_path}...")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.strategy: MLTradingStrategy = model_data["strategy"]
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
            return 0.0

        # Scale down for additional positions
        if active_positions > 0:
            adjusted_size *= self.scaling_factor**active_positions

        return adjusted_size

    def update_positions(self, current_price: float,
                         timestamp: pd.Timestamp) -> None:
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

    def close_position(self, position: Dict, exit_price: float,
                       timestamp: pd.Timestamp, reason: str) -> None:
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
    ) -> None:
        """Open a new position."""
        position = {
            "id": len(self.positions),
            "side": side,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp,
            "status": "active",
            "signal_strength": signal_strength,
            "current_pnl": 0.0,
            "current_price": price,
        }

        self.positions.append(position)
        print(f"   📈 Opened {side} position: {size:.4f} units at {price:.2f}")

    def run_backtest(self,
                     start_date: str | None = None,
                     end_date: str | None = None,
                     output_dir: str | None = None) -> None:
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

        if data_5t.empty:
            print("❌ No data available for the specified range")
            return

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

        # Check if new 4-model architecture is available
        self.has_new_models = (
            hasattr(self.strategy.pipeline, "q50_models") and
            "5T" in self.strategy.pipeline.q50_models)

        def _ensure_1d(preds):
            arr = np.asarray(preds)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr.ravel()

        if self.has_new_models:
            # Use new 4-model architecture (q10, q50, q90, volatility)
            print("Using new 4-model architecture (q10, q50, q90, volatility)...")

            # Get predictions from all four models
            q10_pred = _ensure_1d(
                self.strategy.pipeline.q10_models["5T"].predict(X_5t_clean))
            q50_pred = _ensure_1d(
                self.strategy.pipeline.q50_models["5T"].predict(X_5t_clean))
            q90_pred = _ensure_1d(
                self.strategy.pipeline.q90_models["5T"].predict(X_5t_clean))
            vol_pred = _ensure_1d(
                self.strategy.pipeline.volatility_models["5T"].predict(
                    X_5t_clean))

            # Create signals DataFrame
            signals = pd.DataFrame({
                "timestamp": X_5t_clean.index,
                "q10": q10_pred,
                "q50": q50_pred,
                "q90": q90_pred,
                "vol": vol_pred,
                "close": data_5t["close"].loc[X_5t_clean.index],
            })

            # Calculate derived metrics
            signals["interval_width"] = signals["q90"] - signals["q10"]
            signals["confidence"] = np.abs(signals["q50"]) / (
                signals["interval_width"] + 1e-8)
            signals["signal_strength"] = signals["q50"] / (
                signals["vol"] + 1e-8)

            # Generate signals using new decision logic
            # Default thresholds (can be overridden)
            signal_strength_threshold = getattr(self.strategy,
                                                "signal_strength_threshold",
                                                1.0)
            confidence_threshold = getattr(self.strategy,
                                           "confidence_threshold", 0.3)

            signals["discrete_signal"] = 0
            # Long signal: positive q50, high signal strength, high confidence
            long_mask = (
                (signals["q50"] > 0) &
                (signals["signal_strength"] > signal_strength_threshold) &
                (signals["confidence"] > confidence_threshold))
            signals.loc[long_mask, "discrete_signal"] = 1

            # Short signal: negative q50, high signal strength (absolute), high confidence
            short_mask = (
                (signals["q50"] < 0) &
                (np.abs(signals["signal_strength"]) > signal_strength_threshold)
                & (signals["confidence"] > confidence_threshold))
            signals.loc[short_mask, "discrete_signal"] = -1

            # Use absolute signal strength for position sizing
            signals["signal_strength"] = np.abs(signals["signal_strength"])

        else:
            # Fallback to old stage1/stage2 architecture for backward compatibility
            print(
                "⚠️  New 4-model architecture not found, using old stage1/stage2 models..."
            )
            if not (hasattr(self.strategy.pipeline, "stage1_models")
                    and "5T" in self.strategy.pipeline.stage1_models):
                print("❌ No valid models found in pipeline")
                return

            stage1_model = self.strategy.pipeline.stage1_models["5T"]
            stage2_model = self.strategy.pipeline.stage2_models["5T"]

            stage1_raw = np.asarray(stage1_model.predict(X_5t_clean))
            stage2_preds = _ensure_1d(stage2_model.predict(X_5t_clean))

            if stage1_raw.ndim == 2 and stage1_raw.shape[1] >= 2:
                # Multiclass probs: 0=Hold, 1=Long, 2=Short (legacy convention)
                long_prob = stage1_raw[:, 1]
                short_prob = (
                    stage1_raw[:, 2]
                    if stage1_raw.shape[1] >= 3 else np.clip(1.0 - long_prob, 0.0, 1.0)
                )
                stage1_scalar = long_prob
            else:
                # Binary prob (compat)
                stage1_scalar = _ensure_1d(stage1_raw)
                long_prob = stage1_scalar
                short_prob = 1.0 - long_prob

            # Create signals DataFrame
            signals = pd.DataFrame({
                "timestamp": X_5t_clean.index,
                "stage1_pred": stage1_scalar,
                "stage1_long_prob": long_prob,
                "stage1_short_prob": short_prob,
                "stage2_pred": stage2_preds,
                "close": data_5t["close"].loc[X_5t_clean.index],
            })

            # Convert to discrete signals using probability dominance
            signals["discrete_signal"] = 0
            long_mask = (signals["stage1_long_prob"] > 0.55) & (
                signals["stage1_long_prob"] > signals["stage1_short_prob"])
            short_mask = (signals["stage1_short_prob"] > 0.55) & (
                signals["stage1_short_prob"] > signals["stage1_long_prob"])
            signals.loc[long_mask, "discrete_signal"] = 1
            signals.loc[short_mask, "discrete_signal"] = -1

            # Calculate signal strength from class probability spread
            signals["signal_strength"] = np.abs(
                signals["stage1_long_prob"] - signals["stage1_short_prob"]) / 2.0

        print("\n📊 Signal Statistics:")
        print(f"   Total signals: {len(signals)}")
        print(
            f"   Long signals: {len(signals[signals['discrete_signal'] == 1])}")
        print(
            f"   Short signals: {len(signals[signals['discrete_signal'] == -1])}"
        )
        print(
            f"   Hold signals: {len(signals[signals['discrete_signal'] == 0])}")

        # Show model architecture info
        if self.has_new_models:
            print("   Model architecture: 4-model (q10, q50, q90, volatility)")
            if "q50" in signals.columns:
                print(
                    f"   Avg Q50 prediction: {signals['q50'].mean():.6f}")
                print(
                    f"   Avg signal strength: {signals['signal_strength'].mean():.4f}"
                )
                print(
                    f"   Avg confidence: {signals['confidence'].mean():.4f}")
        else:
            print("   Model architecture: 2-model (stage1, stage2)")

        # Run backtest
        print("\n🔄 Running backtest...")

        for i, (timestamp, row) in enumerate(signals.iterrows()):
            current_price = row["close"]
            signal = row["discrete_signal"]
            signal_strength = row["signal_strength"]

            # Update existing positions
            self.update_positions(current_price, timestamp)

            # Check for new signals
            # Use different thresholds based on model architecture
            if self.has_new_models:
                # For new 4-model architecture, signal_strength and confidence are already filtered
                min_signal_strength = 0.01  # Lower threshold for absolute signal strength
            else:
                # For old stage1/stage2 architecture, use original threshold
                min_signal_strength = 0.1

            if signal != 0 and signal_strength > min_signal_strength:
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

            drawdown = (
                self.peak_equity - current_equity) / self.peak_equity
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
        self.save_results(output_dir, start_date=start_date, end_date=end_date)

        print("\n🎉 Backtest completed successfully!")

    def calculate_results(self) -> None:
        """Calculate backtest results."""
        if not self.trades:
            print("❌ No trades executed")
            # Set default results
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.capital,
                "initial_capital": self.initial_capital,
            }
            return

        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["pnl"] > 0])
        losing_trades = len([t for t in self.trades if t["pnl"] < 0])

        win_rate = winning_trades / total_trades * 100.0

        total_pnl = sum([t["pnl"] for t in self.trades])
        avg_win = (np.mean([t["pnl"] for t in self.trades
                            if t["pnl"] > 0]) if winning_trades > 0 else 0.0)
        avg_loss = (np.mean([t["pnl"] for t in self.trades
                             if t["pnl"] < 0]) if losing_trades > 0 else 0.0)

        profit_factor = (abs(avg_win * winning_trades /
                             (avg_loss * losing_trades))
                         if losing_trades > 0 else float("inf"))

        # Risk metrics
        returns = [t["return_pct"] for t in self.trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns) *
                        np.sqrt(252) if np.std(returns) > 0 else 0.0)

        # Final equity
        final_equity = self.capital
        total_return = ((final_equity - self.initial_capital) /
                        self.initial_capital * 100.0)

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
            "max_drawdown": self.max_drawdown * 100.0,
            "final_equity": final_equity,
            "initial_capital": self.initial_capital,
        }

        print("\n📈 Backtest Results:")
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

    def save_results(self,
                     output_dir: str | None,
                     *,
                     start_date: str | None,
                     end_date: str | None) -> None:
        """Save backtest results."""
        base_dir = Path(output_dir or "results/vectorbot_backtests")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        start_tag = (start_date or "start").replace("-", "")
        end_tag = (end_date or "end").replace("-", "")
        run_dir = base_dir / f"{self.symbol}_{start_tag}_{end_tag}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        trades_path = run_dir / "vectorbot_trades.csv"
        equity_path = run_dir / "vectorbot_equity_curve.csv"
        results_path = run_dir / "vectorbot_results.json"

        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(trades_path, index=False)

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(equity_path, index=False)

        # Save results
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n💾 Results saved:")
        print(f"   - {trades_path}")
        print(f"   - {equity_path}")
        print(f"   - {results_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="VectorBot backtest runner")
    parser.add_argument("--model",
                        type=str,
                        default=os.environ.get("MODEL_PATH"),
                        help="Path to trained model .pkl")
    parser.add_argument("--symbol",
                        type=str,
                        default=os.environ.get("SYMBOL", "BTCUSDT"),
                        help="Symbol (for logging)")
    parser.add_argument("--start",
                        type=str,
                        default=os.environ.get("START_DATE"),
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",
                        type=str,
                        default=os.environ.get("END_DATE"),
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("VECTORBOT_RESULTS_DIR",
                               "results/vectorbot_backtests"),
        help="Directory to store backtest outputs",
    )
    parser.add_argument("--initial-capital",
                        type=float,
                        default=float(
                            os.environ.get("INITIAL_CAPITAL", "100000")),
                        help="Initial capital for backtest")
    return parser


def main(argv: List[str] | None = None) -> None:
    """Main function to run VectorBot backtest."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_path = args.model
    if not model_path:
        print("❌ No model path provided. Use --model or set MODEL_PATH env.")
        return

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run `make train` first to produce the model bundle (.pkl)")
        return

    print(f"Symbol: {args.symbol}")
    if args.start or args.end:
        print(f"Backtest range: {args.start or '-∞'} → {args.end or '+∞'}")

    # Initialize backtest
    backtest = VectorBotBacktest(model_path,
                                 symbol=args.symbol,
                                 initial_capital=args.initial_capital)

    # Run backtest with optional date range
    backtest.run_backtest(start_date=args.start,
                          end_date=args.end,
                          output_dir=args.output_dir)


if __name__ == "__main__":
    main()

