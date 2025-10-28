"""Quick test of quarterly models on next quarter data."""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def load_and_process_file(zip_path, freq="5T"):
    """Load a single file and create OHLCV."""
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        csv_path = os.path.join(temp_dir, csv_files[0])

        df = pd.read_csv(csv_path)

        if "transact_time" in df.columns or "timestamp" in df.columns:
            if "transact_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=[
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "transact_time",
                    "is_buyer_maker",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")

        df.set_index("timestamp", inplace=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df = df.dropna(subset=["price", "quantity"])

        ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
            {"price": ["first", "max", "min", "last"], "quantity": "sum"}
        )
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc.dropna().ffill()

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        return ohlc

    except Exception as e:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def add_features(df):
    """Add technical indicators."""
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


def simple_backtest(df, predictions, signal_threshold=0.6):
    """Simple backtest."""
    df = df.copy()
    df["prediction"] = predictions
    df["signal"] = (predictions > signal_threshold).astype(int)

    capital = 100000.0
    position = None
    trades = []
    equity_curve = [capital]

    for idx, row in df.iterrows():
        price = row["close"]

        if position is not None:
            if position["side"] == "long" and price <= position["stop"]:
                pnl = (price - position["entry"]) * position["size"]
                trades.append(
                    {
                        "pnl": pnl,
                        "reason": "stop_loss",
                        "entry": position["entry"],
                        "exit": price,
                    }
                )
                capital += pnl
                position = None
            elif position["side"] == "long" and price >= position["target"]:
                pnl = (price - position["entry"]) * position["size"]
                trades.append(
                    {
                        "pnl": pnl,
                        "reason": "take_profit",
                        "entry": position["entry"],
                        "exit": price,
                    }
                )
                capital += pnl
                position = None

        if position is None and row["signal"] == 1:
            stop_loss_pct = 0.02
            take_profit_pct = 0.04
            size = (capital * 0.01) / (price * stop_loss_pct)
            position = {
                "side": "long",
                "entry": price,
                "size": size,
                "stop": price * (1 - stop_loss_pct),
                "target": price * (1 + take_profit_pct),
            }

        equity_curve.append(capital)

    if position is not None:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - position["entry"]) * position["size"]
        trades.append(
            {
                "pnl": pnl,
                "reason": "end",
                "entry": position["entry"],
                "exit": last_price,
            }
        )
        capital += pnl

    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df["pnl"].sum()
        total_return = (capital - 100000) / 100000 * 100
        wins = trades_df[trades_df["pnl"] > 0]
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df["pnl"].pct_change().dropna()
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
    else:
        total_pnl = 0
        total_return = 0
        win_rate = 0
        max_drawdown = 0
        sharpe = 0

    return {
        "total_trades": len(trades),
        "total_pnl": float(total_pnl),
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "final_equity": float(capital),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
    }


def load_quarter_data(data_dir, year, quarter):
    """Load data for a specific quarter."""
    months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}

    all_data = []
    for month in months[quarter]:
        file_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-{year}-{month:02d}.zip")
        if os.path.exists(file_path):
            print(f"      Loading {year}-{month:02d}...", end=" ")
            df = load_and_process_file(file_path)
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"{len(df):,} bars")
        else:
            print(f"      ⚠️  {year}-{month:02d} not found")

    if all_data:
        return pd.concat(all_data, axis=0).sort_index()
    return None


def main():
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"
    model_dir = r"results\quarterly_rolling_btc"

    # Test cases: (model_quarter, test_quarter)
    test_cases = [
        ("2023Q1", (2023, 2)),
        ("2023Q2", (2023, 3)),
        ("2023Q3", (2023, 4)),
        ("2023Q4", (2024, 1)),
        ("2024Q1", (2024, 2)),
        ("2024Q2", (2024, 3)),
    ]

    print("\n" + "=" * 80)
    print("🚀 Quick Quarterly Model Validation")
    print("=" * 80 + "\n")

    all_results = []

    for i, (model_q, (test_year, test_q)) in enumerate(test_cases, 1):
        test_quarter_str = f"{test_year}Q{test_q}"
        model_path = os.path.join(model_dir, f"model_{model_q}.txt")

        if not os.path.exists(model_path):
            print(
                f"\n[{i}/{len(test_cases)}] ⚠️  Model {model_q} not found, skipping..."
            )
            continue

        print(
            f"\n[{i}/{len(test_cases)}] Testing: Model {model_q} → Test {test_quarter_str}"
        )
        print("-" * 80)

        # Load model
        print(f"   📦 Loading model {model_q}...")
        model = lgb.Booster(model_file=model_path)

        # Load test data
        print(f"   📊 Loading test data {test_quarter_str}...")
        test_df = load_quarter_data(data_dir, test_year, test_q)

        if test_df is None or len(test_df) == 0:
            print(f"   ❌ No test data for {test_quarter_str}")
            continue

        print(f"   ✅ Loaded {len(test_df):,} bars")

        # Add features
        print(f"   🔧 Engineering features...")
        test_df = add_features(test_df)
        test_df = test_df.dropna()

        # Prepare features
        feature_cols = [
            col
            for col in test_df.columns
            if col
            not in ["open", "high", "low", "close", "volume", "hl", "hc", "lc", "tr"]
        ]

        X_test = test_df[feature_cols].values

        # Predict
        print(f"   🔮 Predicting...")
        predictions = model.predict(X_test)

        # Backtest
        print(f"   💰 Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["model_quarter"] = model_q
        results["test_quarter"] = test_quarter_str

        all_results.append(results)

        print(f"\n   📈 Results:")
        print(f"      Trades      : {results['total_trades']}")
        print(f"      Return      : {results['total_return']:+.2f}%")
        print(f"      Win Rate    : {results['win_rate']:.1f}%")
        print(f"      Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"      Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"      Final Equity: ${results['final_equity']:,.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY REPORT")
    print("=" * 80 + "\n")

    if not all_results:
        print("❌ No results generated!")
        return

    results_df = pd.DataFrame(all_results)

    # Save results
    output_path = "results/quarterly_rolling_btc/quick_test_results.csv"
    results_df.to_csv(output_path, index=False)

    # Display table
    print(
        f"{'Model':<10} {'→':<3} {'Test':<10} {'Trades':<8} {'Return':<10} {'Win%':<8} {'MaxDD':<10} {'Sharpe':<8}"
    )
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['model_quarter']:<10} → {row['test_quarter']:<10} "
            f"{row['total_trades']:<8.0f} {row['total_return']:>8.2f}% "
            f"{row['win_rate']:>6.1f}% {row['max_drawdown']:>8.2f}% "
            f"{row['sharpe_ratio']:>6.2f}"
        )

    print("-" * 80)
    print(
        f"{'AVERAGE':<10}   {'':10} "
        f"{results_df['total_trades'].mean():<8.1f} {results_df['total_return'].mean():>8.2f}% "
        f"{results_df['win_rate'].mean():>6.1f}% {results_df['max_drawdown'].mean():>8.2f}% "
        f"{results_df['sharpe_ratio'].mean():>6.2f}"
    )

    print(f"\n💾 Results saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("💡 ANALYSIS")
    print("=" * 80 + "\n")

    positive_returns = results_df[results_df["total_return"] > 0]

    print(
        f"✅ Profitable Quarters: {len(positive_returns)}/{len(results_df)} "
        f"({len(positive_returns)/len(results_df)*100:.1f}%)"
    )
    print(f"📊 Average Return: {results_df['total_return'].mean():+.2f}%")
    print(
        f"📊 Best Quarter: {results_df.loc[results_df['total_return'].idxmax(), 'test_quarter']} "
        f"({results_df['total_return'].max():+.2f}%)"
    )
    print(
        f"📊 Worst Quarter: {results_df.loc[results_df['total_return'].idxmin(), 'test_quarter']} "
        f"({results_df['total_return'].min():+.2f}%)"
    )

    if results_df["total_return"].mean() > 0:
        print(f"\n✨ Model shows POSITIVE average performance!")
    else:
        print(f"\n⚠️  Model shows NEGATIVE average performance - needs improvement")

    print("\n" + "=" * 80)
    print("✅ Quick validation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
