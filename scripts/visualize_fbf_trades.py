#!/usr/bin/env python3
"""
Visualize FailedBreakoutFade trades on candlestick charts.
Shows entry points, exit points, and trade outcomes.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec


def load_price_data(
    symbol: str, start_date: str, end_date: str, timeframe: str = "240T"
) -> pd.DataFrame:
    """Load OHLCV data from FeatureStore."""
    store = FeatureStore("feature_store")
    spec = FeatureStoreSpec(
        layer="nnmh_highcap6_240T_2024_202510",
        symbol=symbol,
        timeframe=timeframe,
    )
    df = store.read_range(
        spec, start=pd.Timestamp(start_date), end=pd.Timestamp(end_date)
    )

    # Ensure we have OHLCV columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[required_cols].copy()


def plot_trades_on_candles(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    output_path: Path,
    max_trades_per_chart: int = 50,
):
    """
    Plot trades on candlestick charts.

    Args:
        df: OHLCV DataFrame with timestamp index
        trades: DataFrame with trade entries (must have 'timestamp', 'ret_mean', etc.)
        symbol: Symbol name
        output_path: Output directory for charts
        max_trades_per_chart: Maximum number of trades to show per chart
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert timestamp to datetime if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError(
                "DataFrame must have timestamp index or 'timestamp' column"
            )

    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp")

    # Filter trades to date range of price data
    trades = trades[trades["timestamp"].between(df.index.min(), df.index.max())]

    if len(trades) == 0:
        print(f"⚠️  No trades found in date range for {symbol}")
        return

    print(f"📊 Plotting {len(trades)} trades for {symbol}")

    # Split into multiple charts if too many trades
    n_charts = (len(trades) + max_trades_per_chart - 1) // max_trades_per_chart

    for chart_idx in range(n_charts):
        start_idx = chart_idx * max_trades_per_chart
        end_idx = min((chart_idx + 1) * max_trades_per_chart, len(trades))
        chart_trades = trades.iloc[start_idx:end_idx]

        # Get date range for this chart
        chart_start = chart_trades["timestamp"].min() - pd.Timedelta(days=5)
        chart_end = chart_trades["timestamp"].max() + pd.Timedelta(days=5)
        chart_df = df[(df.index >= chart_start) & (df.index <= chart_end)]

        if len(chart_df) == 0:
            continue

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

        # Plot candlesticks
        for i, (ts, row) in enumerate(chart_df.iterrows()):
            open_price = row["open"]
            high_price = row["high"]
            low_price = row["low"]
            close_price = row["close"]

            color = "green" if close_price >= open_price else "red"
            alpha = 0.3

            # Body
            ax1.bar(
                ts,
                abs(close_price - open_price),
                bottom=min(open_price, close_price),
                color=color,
                alpha=alpha,
                width=0.8,
            )

            # Wicks
            ax1.plot(
                [ts, ts],
                [low_price, high_price],
                color="black",
                linewidth=0.5,
                alpha=0.5,
            )

        # Plot trades
        for _, trade in chart_trades.iterrows():
            entry_ts = trade["timestamp"]

            # Find closest bar
            closest_idx = chart_df.index.get_indexer([entry_ts], method="nearest")[0]
            if closest_idx < 0:
                continue

            entry_bar = chart_df.iloc[closest_idx]
            entry_price = entry_bar["close"]  # Use close as entry price

            # Calculate exit price (assuming ret_mean is for long, but FBF is short)
            # For FBF, ret_mean is negated, so positive ret_mean means profit
            ret_mean = trade.get("ret_mean", 0.0)
            if pd.isna(ret_mean):
                ret_mean = 0.0

            # FBF is short, so if ret_mean > 0 (after negation), price went down
            # Exit price = entry_price - abs(ret_mean) * entry_price (approximate)
            # Actually, ret_mean is already normalized, so we need to estimate
            # For simplicity, assume ret_mean is in ATR units
            atr_approx = (
                entry_bar["high"] - entry_bar["low"]
            ) * 0.5  # Rough ATR estimate
            exit_price = (
                entry_price - ret_mean * atr_approx
                if ret_mean > 0
                else entry_price + abs(ret_mean) * atr_approx
            )

            # Color: green for profit, red for loss
            # After negation, ret_mean > 0 means profit for short
            trade_color = "green" if ret_mean > 0 else "red"
            marker_size = 100 if ret_mean > 0 else 80

            # Entry marker
            ax1.scatter(
                entry_ts,
                entry_price,
                color=trade_color,
                marker="^" if ret_mean > 0 else "v",
                s=marker_size,
                alpha=0.7,
                edgecolors="black",
                linewidths=1,
                zorder=5,
            )

            # Exit marker (if we can estimate)
            if abs(ret_mean) > 1e-6:
                ax1.scatter(
                    entry_ts + pd.Timedelta(hours=4),  # Approximate exit time
                    exit_price,
                    color=trade_color,
                    marker="x",
                    s=50,
                    alpha=0.7,
                    zorder=5,
                )

                # Line connecting entry and exit
                ax1.plot(
                    [entry_ts, entry_ts + pd.Timedelta(hours=4)],
                    [entry_price, exit_price],
                    color=trade_color,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

        # Formatting
        ax1.set_title(
            f"{symbol} - FailedBreakoutFade Trades (Chart {chart_idx + 1}/{n_charts})\n"
            f"Trades: {len(chart_trades)} | "
            f"Win Rate: {(chart_trades['ret_mean'] > 0).sum()}/{len(chart_trades)} "
            f"({(chart_trades['ret_mean'] > 0).sum()/len(chart_trades)*100:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Price", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, len(chart_df) // 20))
        )
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Volume subplot
        ax2.bar(chart_df.index, chart_df["volume"], color="gray", alpha=0.3, width=0.8)
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, len(chart_df) // 20))
        )
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        # Save
        output_file = output_path / f"{symbol}_fbf_trades_chart_{chart_idx + 1}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved: {output_file}")
        plt.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Visualize FBF trades on candlestick charts"
    )
    p.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to gated execution logs parquet file",
    )
    p.add_argument(
        "--symbol",
        type=str,
        help="Symbol to visualize (if not provided, visualize all symbols)",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--timeframe",
        type=str,
        default="240T",
        help="Timeframe (default: 240T)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fbf_trade_visualization"),
        help="Output directory for charts",
    )
    p.add_argument(
        "--max-trades-per-chart",
        type=int,
        default=50,
        help="Maximum number of trades per chart (default: 50)",
    )

    args = p.parse_args()

    # Load trades
    print(f"📂 Loading trades from: {args.logs}")
    gated = pd.read_parquet(args.logs)
    trades = gated[gated["gate_ok"] == True].copy()

    if len(trades) == 0:
        print("❌ No trades found in logs")
        return 1

    print(f"✅ Loaded {len(trades)} trades")

    # Filter by symbol if provided
    symbols = [args.symbol] if args.symbol else trades["symbol"].unique()

    for symbol in symbols:
        symbol_trades = trades[trades["symbol"] == symbol]
        if len(symbol_trades) == 0:
            print(f"⚠️  No trades for {symbol}")
            continue

        try:
            # Load price data
            print(f"\n📊 Loading price data for {symbol}...")
            price_df = load_price_data(
                symbol, args.start_date, args.end_date, args.timeframe
            )

            # Plot
            plot_trades_on_candles(
                price_df,
                symbol_trades,
                symbol,
                args.output_dir,
                args.max_trades_per_chart,
            )
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n✅ Visualization complete! Charts saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
