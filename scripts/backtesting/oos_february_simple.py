"""Simplified OOS test using February 2025 data."""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    print("❌ LightGBM not installed")
    sys.exit(1)


def extract_zip(zip_path):
    """Extract zip file."""
    print(f"\n📦 Extracting {os.path.basename(zip_path)}...")
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_feb")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    csv_path = os.path.join(temp_dir, csv_files[0])
    print(f"✅ Extracted: {csv_files[0]}")
    return csv_path, temp_dir


def load_and_resample(csv_path, freq="5T"):
    """Load and resample data."""
    print(f"\n📊 Loading February data...")
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    print(f"   Records: {len(df):,}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
        {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    )
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.dropna().ffill()

    print(f"✅ Created {len(ohlc):,} bars")
    return ohlc


def add_features(df):
    """Add same features as training."""
    print(f"\n🔧 Engineering features...")

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

    print(f"✅ Features engineered")
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
    print(f"\n📈 Running backtest...")
    print(f"   Signal threshold: {signal_threshold}")
    print(f"   Stop loss: {stop_loss_pct*100}%")
    print(f"   Take profit: {take_profit_pct*100}%")

    # Get predictions
    X = df[feature_cols].values
    predictions = model.predict(X)
    df["prediction"] = predictions
    df["signal"] = (predictions > signal_threshold).astype(int)

    print(f"   Total signals: {df['signal'].sum()}")

    # Simple backtest
    capital = 100000.0
    position = None
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = row["close"]

        # Check existing position
        if position is not None:
            # Check stop loss
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

            # Check take profit
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

        # New signal
        if position is None and row["signal"] == 1:
            size = (capital * 0.01) / (price * stop_loss_pct)  # Risk 1% per trade
            position = {
                "side": "long",
                "entry": price,
                "entry_time": idx,
                "size": size,
                "stop": price * (1 - stop_loss_pct),
                "target": price * (1 + take_profit_pct),
            }

        # Track equity
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

    # Close final position
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

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        total_pnl = trades_df["pnl"].sum()
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        # Max drawdown
        equity_df = pd.DataFrame(equity_curve)
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
        }

    return results, trades_df, equity_df


def main():
    print("\n" + "=" * 70)
    print("📊 OOS Test: February 2025")
    print("=" * 70)

    # Paths
    zip_path = r"D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip"
    model_path = "models/model_january_2025.txt"
    metadata_path = "models/metadata_january_2025.json"
    results_dir = "results/february_2025_oos"
    os.makedirs(results_dir, exist_ok=True)

    # Check files
    if not os.path.exists(zip_path):
        print(f"\n❌ Data not found: {zip_path}")
        return

    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("\n💡 Please train first:")
        print("   python scripts/train_january_simple.py")
        return

    print(f"\n✅ Data file found ({os.path.getsize(zip_path) / (1024**3):.2f} GB)")
    print(f"✅ Model found")

    try:
        # Load model
        print(f"\n📦 Loading model...")
        model = lgb.Booster(model_file=model_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        feature_cols = metadata["feature_columns"]

        print(
            f"✅ Model loaded ({metadata['n_features']} features, {metadata['metrics']['n_trees']} trees)"
        )

        # Extract
        csv_path, temp_dir = extract_zip(zip_path)

        # Load February data
        df = load_and_resample(csv_path, freq="5T")

        # Add features
        df = add_features(df)
        df = df.dropna()

        # Run backtest
        results, trades_df, equity_df = run_backtest(df, model, feature_cols)

        # Print results
        print(f"\n" + "=" * 70)
        print(f"📊 Backtest Results")
        print(f"=" * 70)
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.2f}%")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"   Total PnL: ${results['total_pnl']:,.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Avg Win: ${results['avg_win']:,.2f}")
        print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
        print(f"   Final Equity: ${results['final_equity']:,.2f}")

        # Save results
        results_file = os.path.join(results_dir, "backtest_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        if len(trades_df) > 0:
            trades_file = os.path.join(results_dir, "trades.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"\n💾 Trades saved: {trades_file}")

        equity_file = os.path.join(results_dir, "equity_curve.csv")
        equity_df.to_csv(equity_file, index=False)

        print(f"💾 Results saved: {results_file}")
        print(f"💾 Equity curve saved: {equity_file}")

        print("\n" + "=" * 70)
        print("✅ OOS test completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n🧹 Cleaned up temp files")


if __name__ == "__main__":
    main()
