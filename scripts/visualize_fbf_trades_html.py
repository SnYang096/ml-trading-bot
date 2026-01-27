#!/usr/bin/env python3
"""
Generate interactive HTML trading map for FBF trades.
Shows candlestick charts with clear entry/exit markers and trade list.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available, falling back to basic HTML")


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

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[required_cols].copy()


def generate_html_report(
    trades: pd.DataFrame,
    price_df: pd.DataFrame,
    symbol: str,
    output_path: Path,
):
    """Generate interactive HTML report with candlestick chart and trade list."""

    # Filter trades to date range
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp")

    if not isinstance(price_df.index, pd.DatetimeIndex):
        if "timestamp" in price_df.columns:
            price_df = price_df.set_index("timestamp")
        else:
            raise ValueError("Price DataFrame must have timestamp index")

    # Filter to date range
    trades = trades[
        trades["timestamp"].between(price_df.index.min(), price_df.index.max())
    ]

    if len(trades) == 0:
        print(f"⚠️  No trades found in date range for {symbol}")
        return

    # Prepare price data
    price_df = price_df.sort_index()
    price_df = price_df[
        (price_df.index >= trades["timestamp"].min() - pd.Timedelta(days=5))
        & (price_df.index <= trades["timestamp"].max() + pd.Timedelta(days=5))
    ]

    # Calculate returns (negated for FBF short direction)
    trades["ret_mean_negated"] = -trades["ret_mean"]
    trades["is_profit"] = trades["ret_mean_negated"] > 0
    trades["pnl_pct"] = trades["ret_mean_negated"] * 100

    if PLOTLY_AVAILABLE:
        # Create interactive Plotly chart
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"{symbol} - FailedBreakoutFade Trades", "Volume"),
            row_heights=[0.7, 0.3],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_df.index,
                open=price_df["open"],
                high=price_df["high"],
                low=price_df["low"],
                close=price_df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Entry markers (profit = green triangle up, loss = red triangle down)
        profit_trades = trades[trades["is_profit"]]
        loss_trades = trades[~trades["is_profit"]]

        if len(profit_trades) > 0:
            # Find entry prices (use close of nearest bar)
            entry_prices = []
            for ts in profit_trades["timestamp"]:
                closest_idx = price_df.index.get_indexer([ts], method="nearest")[0]
                if closest_idx >= 0:
                    entry_prices.append(price_df.iloc[closest_idx]["close"])
                else:
                    entry_prices.append(None)

            valid_mask = [p is not None for p in entry_prices]
            if any(valid_mask):
                fig.add_trace(
                    go.Scatter(
                        x=profit_trades["timestamp"][valid_mask],
                        y=[p for p, v in zip(entry_prices, valid_mask) if v],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=15,
                            color="green",
                            line=dict(width=2, color="darkgreen"),
                        ),
                        name="Entry (Profit)",
                        hovertemplate="<b>ENTRY (Profit)</b><br>"
                        + "Time: %{x}<br>"
                        + "Price: %{y:.2f}<br>"
                        + "<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        if len(loss_trades) > 0:
            entry_prices = []
            for ts in loss_trades["timestamp"]:
                closest_idx = price_df.index.get_indexer([ts], method="nearest")[0]
                if closest_idx >= 0:
                    entry_prices.append(price_df.iloc[closest_idx]["close"])
                else:
                    entry_prices.append(None)

            valid_mask = [p is not None for p in entry_prices]
            if any(valid_mask):
                fig.add_trace(
                    go.Scatter(
                        x=loss_trades["timestamp"][valid_mask],
                        y=[p for p, v in zip(entry_prices, valid_mask) if v],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=15,
                            color="red",
                            line=dict(width=2, color="darkred"),
                        ),
                        name="Entry (Loss)",
                        hovertemplate="<b>ENTRY (Loss)</b><br>"
                        + "Time: %{x}<br>"
                        + "Price: %{y:.2f}<br>"
                        + "<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Volume
        colors = [
            "red" if c < o else "green"
            for c, o in zip(price_df["close"], price_df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=price_df.index,
                y=price_df["volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"{symbol} - FailedBreakoutFade Trading Map<br>"
            + f"<sub>Total Trades: {len(trades)} | "
            + f"Win Rate: {(trades['is_profit']).sum()}/{len(trades)} "
            + f"({(trades['is_profit']).sum()/len(trades)*100:.1f}%) | "
            + f"Avg P&L: {trades['pnl_pct'].mean():.2f}%</sub>",
            height=800,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Generate HTML
        chart_html = fig.to_html(include_plotlyjs="cdn", div_id="trading-chart")
    else:
        chart_html = f"<p>Plotly not available. Install with: pip install plotly</p>"

    # Generate trade list table
    trades_display = trades.copy()
    trades_display["entry_time"] = trades_display["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    trades_display["pnl_display"] = trades_display["pnl_pct"].apply(
        lambda x: f"{x:+.2f}%"
    )
    trades_display["status"] = trades_display["is_profit"].apply(
        lambda x: "✅ Profit" if x else "❌ Loss"
    )

    trade_table_html = trades_display[
        ["entry_time", "status", "pnl_display", "ret_mean"]
    ].to_html(
        classes="table table-striped table-hover",
        table_id="trade-list",
        escape=False,
        index=False,
    )

    # Statistics
    stats = {
        "total_trades": len(trades),
        "win_count": trades["is_profit"].sum(),
        "loss_count": (~trades["is_profit"]).sum(),
        "win_rate": trades["is_profit"].mean() * 100,
        "avg_pnl": trades["pnl_pct"].mean(),
        "total_pnl": trades["pnl_pct"].sum(),
        "max_profit": trades["pnl_pct"].max(),
        "max_loss": trades["pnl_pct"].min(),
    }

    # Combine into full HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - FBF Trading Map</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 24px;
        }}
        .stat-card p {{
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .table {{
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }}
        .table th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .table tr:hover {{
            background-color: #f5f5f5;
        }}
        #trading-chart {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{symbol} - FailedBreakoutFade Trading Map</h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{stats['total_trades']}</h3>
                <p>Total Trades</p>
            </div>
            <div class="stat-card">
                <h3>{stats['win_count']}</h3>
                <p>Wins</p>
            </div>
            <div class="stat-card">
                <h3>{stats['loss_count']}</h3>
                <p>Losses</p>
            </div>
            <div class="stat-card">
                <h3>{stats['win_rate']:.1f}%</h3>
                <p>Win Rate</p>
            </div>
            <div class="stat-card">
                <h3>{stats['avg_pnl']:+.2f}%</h3>
                <p>Avg P&L</p>
            </div>
            <div class="stat-card">
                <h3>{stats['total_pnl']:+.2f}%</h3>
                <p>Total P&L</p>
            </div>
        </div>
        
        <div id="trading-chart">
            {chart_html}
        </div>
        
        <h2>Trade List</h2>
        {trade_table_html}
        
        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    print(f"✅ Saved HTML report to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML trading map for FBF trades"
    )
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to gated execution logs parquet file",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to visualize (if not provided, visualize all symbols)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="240T",
        help="Timeframe (default: 240T)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fbf_trade_visualization"),
        help="Output directory for HTML files",
    )

    args = parser.parse_args()

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

            # Generate HTML
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = args.output_dir / f"{symbol}_fbf_trading_map.html"
            generate_html_report(symbol_trades, price_df, symbol, output_file)
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n✅ Visualization complete! HTML files saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
