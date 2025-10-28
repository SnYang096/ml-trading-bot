"""Batch OOS testing - test model on multiple months at once."""

import os
import sys
import glob
import re
import zipfile
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    print("❌ LightGBM not installed")
    sys.exit(1)


def extract_zip(zip_path, suffix="test"):
    """Extract zip file."""
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_extract_{suffix}")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    csv_path = os.path.join(temp_dir, csv_files[0])
    return csv_path, temp_dir


def load_and_resample(csv_path, freq="5T"):
    """Load and resample data."""
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
        {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    )
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.dropna().ffill()

    return ohlc


def add_features(df):
    """Add same features as training."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]

    df["volatility_20"] = df["returns"].rolling(20).std()

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = calc_rsi(df["close"])

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    df["hl"] = df["high"] - df["low"]
    df["hc"] = abs(df["high"] - df["close"].shift(1))
    df["lc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
        df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df[
            "close"
        ].shift(period)

    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]

    return df


def run_backtest(
    df,
    model,
    feature_cols,
    signal_threshold=0.6,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
):
    """Run simple backtest."""
    X = df[feature_cols].values
    predictions = model.predict(X)
    df["prediction"] = predictions
    df["signal"] = (predictions > signal_threshold).astype(int)

    capital = 100000.0
    position = None
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = row["close"]

        if position is not None:
            if position["side"] == "long" and price <= position["stop"]:
                pnl = (price - position["entry"]) * position["size"]
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "side": "long",
                        "entry": position["entry"],
                        "exit": price,
                        "size": position["size"],
                        "pnl": pnl,
                        "reason": "stop_loss",
                    }
                )
                capital += pnl
                position = None

            elif position["side"] == "long" and price >= position["target"]:
                pnl = (price - position["entry"]) * position["size"]
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "side": "long",
                        "entry": position["entry"],
                        "exit": price,
                        "size": position["size"],
                        "pnl": pnl,
                        "reason": "take_profit",
                    }
                )
                capital += pnl
                position = None

        if position is None and row["signal"] == 1:
            size = (capital * 0.01) / (price * stop_loss_pct)
            position = {
                "side": "long",
                "entry": price,
                "entry_time": idx,
                "size": size,
                "stop": price * (1 - stop_loss_pct),
                "target": price * (1 + take_profit_pct),
            }

        open_pnl = 0
        if position is not None:
            open_pnl = (price - position["entry"]) * position["size"]

        equity_curve.append(
            {
                "timestamp": idx,
                "capital": capital,
                "open_pnl": open_pnl,
                "equity": capital + open_pnl,
            }
        )

    if position is not None and len(df) > 0:
        last_price = df["close"].iloc[-1]
        last_time = df.index[-1]
        pnl = (last_price - position["entry"]) * position["size"]
        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": last_time,
                "side": "long",
                "entry": position["entry"],
                "exit": last_price,
                "size": position["size"],
                "pnl": pnl,
                "reason": "end_of_data",
            }
        )
        capital += pnl

    # Always create equity_df even if no trades
    equity_df = pd.DataFrame(equity_curve)

    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        total_pnl = trades_df["pnl"].sum()
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["dd"] = (equity_df["peak"] - equity_df["equity"]) / equity_df["peak"]
        max_dd = equity_df["dd"].max() * 100

        results = {
            "total_trades": len(trades_df),
            "win_rate": float(win_rate),
            "total_pnl": float(total_pnl),
            "total_return": float((capital - 100000) / 100000 * 100),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": (
                float(abs(wins["pnl"].sum() / losses["pnl"].sum()))
                if len(losses) > 0 and losses["pnl"].sum() != 0
                else float("inf")
            ),
            "max_drawdown": float(max_dd),
            "final_equity": float(capital),
            "num_signals": int(df["signal"].sum()),
        }
    else:
        results = {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_return": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "final_equity": 100000.0,
            "num_signals": int(df["signal"].sum()),
        }

    return results, trades_df, equity_df


def find_data_files(data_dir, pattern):
    """Find data files matching pattern."""
    all_files = glob.glob(os.path.join(data_dir, "*.zip"))

    # Support regex pattern
    if pattern:
        try:
            regex = re.compile(pattern)
            matched_files = [f for f in all_files if regex.search(os.path.basename(f))]
        except re.error:
            # If regex fails, try glob-style pattern
            matched_files = glob.glob(os.path.join(data_dir, pattern))
    else:
        matched_files = all_files

    return sorted(matched_files)


def main():
    parser = argparse.ArgumentParser(description="Batch OOS testing on multiple months")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (without extension)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"D:\GitHub\trading\rlbot\data\agg_data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"BTCUSDT-aggTrades-2025-0[2-9]\.zip",
        help="Regex pattern to match data files (default: Feb-Sep 2025)",
    )
    parser.add_argument(
        "--output", type=str, default="batch_oos_results", help="Output directory name"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📊 Batch OOS Test: Multiple Months Evaluation")
    print("=" * 70)

    # Find data files
    data_files = find_data_files(args.data_dir, args.pattern)

    if not data_files:
        print(f"\n❌ No data files found matching pattern: {args.pattern}")
        print(f"   in directory: {args.data_dir}")
        return

    print(f"\n📁 Found {len(data_files)} data files:")
    for f in data_files:
        print(f"   - {os.path.basename(f)}")

    # Load model
    model_path = f"models/{args.model}.txt"
    metadata_path = f"models/{args.model}_metadata.json"

    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return

    print(f"\n📦 Loading model: {args.model}")
    model = lgb.Booster(model_file=model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    timeframe = metadata["timeframe"]

    print(
        f"✅ Model loaded ({metadata['n_features']} features, {metadata['metrics']['n_trees']} trees)"
    )

    # Create output directory
    results_dir = f"results/{args.output}"
    os.makedirs(results_dir, exist_ok=True)

    # Test each file
    all_results = {}
    all_trades = []

    print("\n" + "=" * 70)
    print("🚀 Starting Batch Testing...")
    print("=" * 70 + "\n")

    for i, data_file in enumerate(data_files, 1):
        filename = os.path.basename(data_file)
        file_key = filename.replace(".zip", "")

        print(f"[{i}/{len(data_files)}] Testing {filename}...")

        try:
            # Extract
            csv_path, temp_dir = extract_zip(data_file, suffix=f"batch_{i}")

            # Load
            df = load_and_resample(csv_path, freq=timeframe)
            print(f"   Loaded {len(df):,} bars")

            # Add features
            df = add_features(df)
            df = df.dropna()

            # Run backtest
            results, trades_df, equity_df = run_backtest(df, model, feature_cols)

            # Add metadata
            results["data_file"] = filename
            results["date_range"] = [str(df.index[0]), str(df.index[-1])]
            results["n_bars"] = len(df)

            all_results[file_key] = results

            # Save individual results
            month_dir = os.path.join(results_dir, file_key)
            os.makedirs(month_dir, exist_ok=True)

            with open(os.path.join(month_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            if len(trades_df) > 0:
                trades_df["month"] = file_key
                all_trades.append(trades_df)
                trades_df.to_csv(os.path.join(month_dir, "trades.csv"), index=False)

            equity_df.to_csv(os.path.join(month_dir, "equity_curve.csv"), index=False)

            # Print summary
            print(
                f"   ✅ Trades: {results['total_trades']}, "
                f"Return: {results['total_return']:+.2f}%, "
                f"Win Rate: {results['win_rate']:.1f}%, "
                f"Max DD: {results['max_drawdown']:.2f}%"
            )

            # Cleanup
            if os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results[file_key] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Combine all trades
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades.to_csv(os.path.join(results_dir, "all_trades.csv"), index=False)

    # Generate summary report
    print("\n" + "=" * 70)
    print("📊 Summary Report")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results).T
    # Filter out error rows
    if "error" in summary_df.columns:
        summary_df = summary_df[summary_df["error"].isna()]
        summary_df = summary_df.drop(columns=["error"])

    if len(summary_df) > 0:
        print(
            f"\n{'Month':<30} {'Trades':<8} {'Return':<10} {'Win%':<8} {'MaxDD%':<10} {'PF':<8}"
        )
        print("-" * 70)

        for idx, row in summary_df.iterrows():
            print(
                f"{idx:<30} {row['total_trades']:<8} "
                f"{row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% "
                f"{row['max_drawdown']:>8.2f}% {row['profit_factor']:>6.2f}"
            )

        print("-" * 70)
        print(
            f"{'AVERAGE':<30} {summary_df['total_trades'].mean():<8.1f} "
            f"{summary_df['total_return'].mean():>8.2f}% {summary_df['win_rate'].mean():>6.1f}% "
            f"{summary_df['max_drawdown'].mean():>8.2f}% {summary_df['profit_factor'].mean():>6.2f}"
        )

        total_return = summary_df["total_return"].sum()
        print(f"\n📈 Cumulative Return: {total_return:+.2f}%")
        print(f"📊 Total Trades: {summary_df['total_trades'].sum():.0f}")
        print(
            f"✅ Profitable Months: {(summary_df['total_return'] > 0).sum()}/{len(summary_df)}"
        )

    print(f"\n💾 Results saved to: {results_dir}")
    print("\n" + "=" * 70)
    print("✅ Batch testing completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
