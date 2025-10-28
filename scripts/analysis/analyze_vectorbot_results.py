"""Analyze VectorBot backtest results with visualizations."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime


def analyze_vectorbot_results():
    """Analyze VectorBot backtest results."""
    print("📊 VectorBot Backtest Results Analysis")
    print("=" * 50)

    # Load results
    try:
        # Try improved results first
        try:
            with open("improved_vectorbot_results.json", "r") as f:
                results = json.load(f)

            trades_df = pd.read_csv("improved_vectorbot_trades.csv")
            equity_df = pd.read_csv("improved_vectorbot_equity_curve.csv")
        except FileNotFoundError:
            # Fall back to original results
            with open("vectorbot_results.json", "r") as f:
                results = json.load(f)

            trades_df = pd.read_csv("vectorbot_trades.csv")
            equity_df = pd.read_csv("vectorbot_equity_curve.csv")

    except FileNotFoundError as e:
        print(f"❌ Results file not found: {e}")
        print("Please run vectorbot_backtest.py first")
        return

    # Convert timestamps
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])

    # Print summary
    print(f"📈 Backtest Summary:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.2f}%")
    print(f"   Total Return: {results['total_return']:.2f}%")
    print(f"   Total P&L: {results['total_pnl']:.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Final Equity: {results['final_equity']:.2f}")

    # Trade analysis
    print(f"\n📊 Trade Analysis:")
    print(f"   Winning Trades: {results['winning_trades']}")
    print(f"   Losing Trades: {results['losing_trades']}")
    print(f"   Average Win: {results['avg_win']:.2f}")
    print(f"   Average Loss: {results['avg_loss']:.2f}")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Equity Curve
    axes[0, 0].plot(
        equity_df["timestamp"], equity_df["equity"], linewidth=2, color="blue"
    )
    axes[0, 0].axhline(
        y=results["initial_capital"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Initial Capital",
    )
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].set_ylabel("Equity ($)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Drawdown
    peak = equity_df["equity"].expanding().max()
    drawdown = (equity_df["equity"] - peak) / peak * 100
    axes[0, 1].fill_between(equity_df["timestamp"], drawdown, 0, color="red", alpha=0.3)
    axes[0, 1].plot(equity_df["timestamp"], drawdown, color="red", linewidth=1)
    axes[0, 1].set_title("Drawdown")
    axes[0, 1].set_ylabel("Drawdown (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. P&L Distribution
    axes[1, 0].hist(
        trades_df["pnl"], bins=20, alpha=0.7, color="green", edgecolor="black"
    )
    axes[1, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("P&L Distribution")
    axes[1, 0].set_xlabel("P&L ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Trade Duration
    axes[1, 1].hist(
        trades_df["duration"], bins=20, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[1, 1].set_title("Trade Duration Distribution")
    axes[1, 1].set_xlabel("Duration (minutes)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vectorbot_analysis.png", dpi=300, bbox_inches="tight")

    # Additional analysis
    print(f"\n📊 Additional Analysis:")

    # Trade reasons
    reason_counts = trades_df["reason"].value_counts()
    print(f"   Trade Exit Reasons:")
    for reason, count in reason_counts.items():
        print(f"     {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    # Side analysis
    side_counts = trades_df["side"].value_counts()
    print(f"   Trade Sides:")
    for side, count in side_counts.items():
        print(f"     {side}: {count} ({count/len(trades_df)*100:.1f}%)")

    # Monthly performance
    trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
    monthly_pnl = trades_df.groupby("month")["pnl"].sum()
    print(f"\n📅 Monthly Performance:")
    for month, pnl in monthly_pnl.items():
        print(f"   {month}: {pnl:.2f}")

    # Risk metrics
    print(f"\n⚠️  Risk Metrics:")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")

    # Trade quality
    print(f"\n🎯 Trade Quality:")
    print(f"   Average Trade Duration: {trades_df['duration'].mean():.1f} minutes")
    print(f"   Best Trade: {trades_df['pnl'].max():.2f}")
    print(f"   Worst Trade: {trades_df['pnl'].min():.2f}")
    print(f"   Average Return per Trade: {trades_df['return_pct'].mean():.2f}%")

    print(f"\n📊 Plots saved to 'vectorbot_analysis.png'")
    print(f"✅ Analysis completed!")


if __name__ == "__main__":
    analyze_vectorbot_results()
