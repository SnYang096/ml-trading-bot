import os, json, pickle, zipfile
import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.feature_engineering_wavelet import WaveletFeatureEngineer
from ml_trading.strategies.ml_strategy import MLTradingStrategy

MODEL_PATH = os.path.join("models", "trained_model_wavelet_may_2025.pkl")
SCALER_PATH = os.path.join("models", "feature_scalers_wavelet_may_2025.pkl")


def prepare_ohlc(zip_path: str) -> pd.DataFrame:
    """Extract and prepare OHLCV data from zip file"""
    tmp = os.path.join("data", "temp_extract_run")
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


def load_model_and_fe():
    """Load trained model and feature engineer"""
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    strategy = model_data["strategy"]
    feature_engineer = model_data["feature_engineer"]
    feature_engineer.load_scalers(SCALER_PATH)
    return strategy, feature_engineer


def load_params(results_dir: str):
    """Load optimized parameters"""
    params = {
        "5T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
        "15T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
        "60T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
        "240T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
    }

    opt_path = os.path.join(results_dir, "optuna_best_params.json")
    grid_path = os.path.join(results_dir, "grid_tune_best.csv")

    if os.path.exists(opt_path):
        try:
            opt = json.load(open(opt_path))
            for tf, p in opt.items():
                params[str(tf)] = {
                    "sl": float(p["sl"]),
                    "tp": float(p["tp"]),
                    "sig": float(p["sig"]),
                    "gap": int(p["gap"]),
                }
        except Exception:
            pass
    elif os.path.exists(grid_path):
        try:
            best = pd.read_csv(grid_path)
            for _, row in best.iterrows():
                tf = str(row["timeframe"])
                params[tf] = {
                    "sl": float(row["sl"]),
                    "tp": float(row["tp"]),
                    "sig": float(row["sig_th"]),
                    "gap": int(row["gap"]),
                }
        except Exception:
            pass

    return params


def run_backtest(
    strategy, feature_engineer, mtf_data, tf, params, risk_pct=0.002, max_leverage=2.0
):
    """Run backtest for a specific timeframe using the existing oos_june infrastructure"""
    # Engineer features
    engineered = feature_engineer.engineer_features(mtf_data, fit=False)

    # Import the run_bt function from oos_june
    import sys

    sys.path.append("scripts")
    from oos_june import run_bt

    # Run backtest using the existing infrastructure
    results = run_bt(
        tf,
        params["sl"],
        params["tp"],
        params["sig"],
        params["gap"],
        risk_pct=risk_pct,
        max_leverage=max_leverage,
        max_adds=1,
        add_risk_frac=0.2,
        atr_trail_mult=2.0,
        atr_stop_k=1.5,
        max_concurrent=1,
    )

    return results


def run_month(zip_path: str, out_dir: str):
    """Run backtest for a specific month"""
    print(f"Running OOS for {zip_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Load model and feature engineer
    strategy, feature_engineer = load_model_and_fe()

    # Prepare data
    ohlc = prepare_ohlc(zip_path)
    print(f"Loaded {len(ohlc)} bars, range: {ohlc.index[0]} -> {ohlc.index[-1]}")

    # Create multi-timeframe data
    mdl = MarketDataLoader()
    mdl.raw_data = ohlc
    mtf = mdl.get_multi_timeframe_data()
    print(f"Timeframes: {dict((k, len(v)) for k, v in mtf.items())}")

    # Load parameters
    params = load_params(os.path.join("results", "june_2025_oos"))

    # Run backtests for each timeframe
    results = {}
    for tf, risk in [("5T", 0.0028), ("15T", 0.0028), ("60T", 0.002), ("240T", 0.002)]:
        p = params.get(tf, {"sl": 0.03, "tp": 0.06, "sig": 0.06, "gap": 6})

        print(f"Running {tf} backtest...")
        res = run_backtest(
            strategy, feature_engineer, mtf, tf, p, risk_pct=risk, max_leverage=2.0
        )
        results[tf] = res

        # Save results
        results_month_dir = os.path.join("results", out_dir)
        os.makedirs(results_month_dir, exist_ok=True)

        jf = os.path.join(results_month_dir, f"wavelet_{tf}_results.json")
        with open(jf, "w") as f:
            json.dump(res, f, indent=2)

        print(f"[{tf}] Results: {res}")

    return results


def generate_drift_report():
    """Generate drift analysis report"""
    months = ["june_2025_oos", "july_2025_oos", "august_2025_oos", "september_2025_oos"]
    timeframes = ["5T", "15T", "60T", "240T"]

    # Collect results
    all_results = {}
    for month in months:
        month_dir = os.path.join("results", month)
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
                            "Month": month.replace("_2025_oos", ""),
                            "Timeframe": tf,
                            "Total_Trades": res.get("total_trades", 0),
                            "Win_Rate": res.get("win_rate", 0),
                            "Total_Return": res.get("total_return", 0),
                            "Max_Drawdown": res.get("max_drawdown", 0),
                            "Profit_Factor": res.get("profit_factor", 0),
                            "Final_Equity": res.get("final_equity", 0),
                        }
                    )

    # Save comparison
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv("reports/drift_analysis_comparison.csv", index=False)

    print("=== Model Drift Analysis ===")
    print(df_comparison.to_string(index=False))

    return df_comparison


if __name__ == "__main__":
    # Run backtests for each month
    months_data = [
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-07.zip",
            "july_2025_oos",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-08.zip",
            "august_2025_oos",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-09.zip",
            "september_2025_oos",
        ),
    ]

    for zip_path, out_dir in months_data:
        if os.path.exists(zip_path):
            run_month(zip_path, out_dir)
        else:
            print(f"Missing zip: {zip_path}")

    # Generate drift analysis report
    generate_drift_report()
