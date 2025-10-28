import os, json, pandas as pd, numpy as np
import sys, zipfile, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def extract_and_prepare_data(zip_path: str):
    """Extract and prepare OHLCV data from zip file"""
    tmp = os.path.join("data", "temp_extract_monthly")
    os.makedirs(tmp, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)

    csv_files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
    if not csv_files:
        raise SystemExit(f"No CSV found in zip: {zip_path}")

    csv_path = os.path.join(tmp, csv_files[0])
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise SystemExit("timestamp column not found")

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    # Resample to 1-second bars
    per_sec = df.groupby(pd.Grouper(freq="1s")).agg(
        {
            "price": ["first", "max", "min", "last"],
            "quantity": "sum",
            "is_buyer_maker": "mean",
        }
    )
    per_sec.columns = ["open", "high", "low", "close", "volume", "is_buyer_maker"]
    per_sec = per_sec.dropna().ffill()

    # Calculate taker buy ratio and CVD
    per_sec["taker_buy"] = (~per_sec["is_buyer_maker"].round().astype(bool)).astype(int)
    per_sec["buy_qty"] = per_sec["taker_buy"] * per_sec["volume"]
    per_sec["sell_qty"] = (1 - per_sec["taker_buy"]) * per_sec["volume"]
    per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
        per_sec["buy_qty"] + per_sec["sell_qty"]
    ).replace(0, np.nan)
    per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
    per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()

    return per_sec[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buy_qty",
            "sell_qty",
            "taker_buy_ratio",
            "cvd",
        ]
    ]


def run_monthly_backtest(zip_path: str, month_name: str):
    """Run backtest for a specific month"""
    print(f"Running OOS for {zip_path} -> {month_name}")

    # Create results directory
    results_dir = os.path.join("results", f"{month_name}_2025_oos")
    os.makedirs(results_dir, exist_ok=True)

    # Extract and prepare data
    ohlc = extract_and_prepare_data(zip_path)
    print(f"Loaded {len(ohlc)} bars, range: {ohlc.index[0]} -> {ohlc.index[-1]}")

    # Create multi-timeframe data
    from ml_trading.data_tools.data_loader import MarketDataLoader

    mdl = MarketDataLoader()
    mdl.raw_data = ohlc
    mtf = mdl.get_multi_timeframe_data()
    print(f"Timeframes: {dict((k, len(v)) for k, v in mtf.items())}")

    # Load model and feature engineer
    import pickle

    MODEL_PATH = os.path.join("models", "trained_model_wavelet_may_2025.pkl")
    SCALER_PATH = os.path.join("models", "feature_scalers_wavelet_may_2025.pkl")

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    strategy = model_data["strategy"]
    feature_engineer = model_data["feature_engineer"]
    feature_engineer.load_scalers(SCALER_PATH)

    # Engineer features
    engineered = feature_engineer.engineer_features(mtf, fit=False)

    # Load parameters
    params = {
        "5T": {
            "sl": 0.04977810558513822,
            "tp": 0.05323089729377332,
            "sig": 0.1079864677101884,
            "gap": 8,
        },
        "15T": {
            "sl": 0.013290092631463418,
            "tp": 0.042092011936156096,
            "sig": 0.08039997122632116,
            "gap": 11,
        },
        "60T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
        "240T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
    }

    # Import and run backtest
    sys.path.append("scripts")
    from oos_june import run_bt

    # Run backtests for each timeframe
    results = {}
    for tf, risk in [("5T", 0.0028), ("15T", 0.0028), ("60T", 0.002), ("240T", 0.002)]:
        p = params.get(tf, {"sl": 0.03, "tp": 0.06, "sig": 0.06, "gap": 6})

        print(f"Running {tf} backtest...")
        res = run_bt(
            tf,
            p["sl"],
            p["tp"],
            p["sig"],
            p["gap"],
            risk_pct=risk,
            max_leverage=2.0,
            max_adds=1,
            add_risk_frac=0.2,
            atr_trail_mult=2.0,
            atr_stop_k=1.5,
            max_concurrent=1,
        )
        results[tf] = res

        # Save results
        jf = os.path.join(results_dir, f"wavelet_{tf}_results.json")
        with open(jf, "w") as f:
            json.dump(res, f, indent=2)

        print(f"[{tf}] Results: {res}")

    return results


def generate_drift_summary():
    """Generate a summary of drift analysis results"""
    months = ["june", "july", "august", "september"]
    timeframes = ["5T", "15T", "60T", "240T"]

    # Collect results
    all_results = {}
    for month in months:
        month_dir = os.path.join("results", f"{month}_2025_oos")
        if os.path.exists(month_dir):
            all_results[month] = {}
            for tf in timeframes:
                result_file = os.path.join(month_dir, f"wavelet_{tf}_results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        all_results[month][tf] = json.load(f)

    # Create comparison table
    comparison_data = []
    for month in months:
        if month in all_results:
            for tf in timeframes:
                if tf in all_results[month]:
                    res = all_results[month][tf]
                    comparison_data.append(
                        {
                            "Month": month.capitalize(),
                            "Timeframe": tf,
                            "Total_Trades": res.get("total_trades", 0),
                            "Win_Rate": f"{res.get('win_rate', 0):.1f}%",
                            "Total_Return": f"{res.get('total_return', 0):.2f}%",
                            "Max_Drawdown": f"{res.get('max_drawdown', 0):.2f}%",
                            "Profit_Factor": f"{res.get('profit_factor', 0):.2f}",
                            "Final_Equity": f"{res.get('final_equity', 0):.0f}",
                        }
                    )

    # Save comparison
    df_comparison = pd.DataFrame(comparison_data)
    os.makedirs("reports", exist_ok=True)
    df_comparison.to_csv("reports/drift_analysis_comparison.csv", index=False)

    print("=== Model Drift Analysis ===")
    if len(df_comparison) > 0:
        print(df_comparison.to_string(index=False))

        # Analyze drift
        print("\n=== Drift Analysis ===")
        for tf in timeframes:
            print(f"\n{tf} Timeframe:")
            tf_data = df_comparison[df_comparison["Timeframe"] == tf]
            if len(tf_data) > 1:
                returns = [float(x.replace("%", "")) for x in tf_data["Total_Return"]]
                drawdowns = [float(x.replace("%", "")) for x in tf_data["Max_Drawdown"]]

                print(f"  Return range: {min(returns):.2f}% to {max(returns):.2f}%")
                print(
                    f"  Drawdown range: {min(drawdowns):.2f}% to {max(drawdowns):.2f}%"
                )

                # Check for significant drift
                if max(returns) - min(returns) > 20:
                    print(f"  ⚠️  Significant return drift detected!")
                if max(drawdowns) - min(drawdowns) > 10:
                    print(f"  ⚠️  Significant drawdown drift detected!")
    else:
        print("No results found for drift analysis")

    return df_comparison


if __name__ == "__main__":
    # Run backtests for each month
    months_data = [
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-07.zip",
            "july",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-08.zip",
            "august",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-09.zip",
            "september",
        ),
    ]

    for zip_path, month_name in months_data:
        if os.path.exists(zip_path):
            run_monthly_backtest(zip_path, month_name)
        else:
            print(f"Missing zip: {zip_path}")

    # Generate drift analysis report
    generate_drift_summary()
