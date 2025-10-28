import os, json, pickle, zipfile
import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_trading.data_tools.data_loader import MarketDataLoader

MODEL_PATH = os.path.join("models", "trained_model_wavelet_may_2025.pkl")
SCALER_PATH = os.path.join("models", "feature_scalers_wavelet_may_2025.pkl")


def prepare_ohlc(zip_path: str) -> pd.DataFrame:
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
    per_sec = df.groupby(pd.Grouper(freq="1s")).agg(
        {
            "price": ["first", "max", "min", "last"],
            "quantity": "sum",
            "is_buyer_maker": "mean",
        }
    )
    per_sec.columns = ["open", "high", "low", "close", "volume", "is_buyer_maker"]
    per_sec = per_sec.dropna().ffill()
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
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    strategy = model_data["strategy"]
    feature_engineer = model_data["feature_engineer"]
    feature_engineer.load_scalers(SCALER_PATH)
    return strategy, feature_engineer


def import_run_bt():
    code = open("scripts/oos_june.py", "r").read()
    ns = {
        "__file__": os.path.abspath("scripts/oos_june.py"),
        "os": os,
        "sys": sys,
        "pd": pd,
        "np": np,
        "json": json,
    }
    exec(compile(code, "scripts/oos_june.py", "exec"), ns, ns)
    return ns["run_bt"]


def load_params(results_dir: str):
    params = {
        "5T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
        "15T": {"sl": 0.02, "tp": 0.04, "sig": 0.04, "gap": 3},
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


def run_month(zip_path: str, out_dir: str):
    print("Running OOS for", zip_path, "->", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    strategy, feature_engineer = load_model_and_fe()
    ohlc = prepare_ohlc(zip_path)
    print(f"Loaded {len(ohlc)} bars, range: {ohlc.index[0]} -> {ohlc.index[-1]}")
    mdl = MarketDataLoader()
    mdl.raw_data = ohlc
    mtf = mdl.get_multi_timeframe_data()
    print(f"Timeframes: {dict((k, len(v)) for k, v in mtf.items())}")
    engineered = feature_engineer.engineer_features(mtf, fit=False)
    run_bt = import_run_bt()
    params = load_params(os.path.join("results", "june_2025_oos"))

    def run_and_save(tf: str, p: dict, risk=0.002, lev=2.0):
        res = run_bt(
            tf,
            p["sl"],
            p["tp"],
            p["sig"],
            p["gap"],
            risk_pct=risk,
            max_leverage=lev,
            max_adds=1,
            add_risk_frac=0.2,
            atr_trail_mult=2.0,
            atr_stop_k=1.5,
            max_concurrent=1,
        )
        # Ensure results directory exists
        results_month_dir = os.path.join("results", out_dir)
        os.makedirs(results_month_dir, exist_ok=True)

        # Save results directly to this month's results dir
        jf = os.path.join(results_month_dir, f"wavelet_{tf}_results.json")
        ef = os.path.join(results_month_dir, f"wavelet_{tf}_equity_curve.csv")
        tf_trades = os.path.join(results_month_dir, f"wavelet_{tf}_trades.csv")

        # Save JSON results
        with open(jf, "w") as f:
            json.dump(res, f, indent=2)

        # Copy equity curve and trades from default location
        base_dir = os.path.join("results", "june_2025_oos")
        import shutil

        for src_suf, dst in [
            (f"wavelet_{tf}_june_equity_curve.csv", ef),
            (f"wavelet_{tf}_june_trades.csv", tf_trades),
        ]:
            try:
                shutil.copyfile(os.path.join(base_dir, src_suf), dst)
            except Exception:
                pass
        print(tf, res)

    # Apply for 5T/15T/60T/240T
    for tf, risk in [("5T", 0.0028), ("15T", 0.0028), ("60T", 0.002), ("240T", 0.002)]:
        p = params.get(tf, {"sl": 0.03, "tp": 0.06, "sig": 0.06, "gap": 6})
        run_and_save(tf, p, risk=risk, lev=2.0)


if __name__ == "__main__":
    months = [
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
    for zp, out in months:
        if os.path.exists(zp):
            run_month(zp, out)
        else:
            print("Missing zip:", zp)
